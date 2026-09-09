#!/usr/bin/env python3
"""Shared lifecycle and input generation for EP backends."""
from __future__ import annotations

import abc
import os
import types
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from ep_harness import (
    time_us,
    token_ladder,
)


@dataclass
class RankInputs:
    """Inputs for one token-ladder shape at tokens_per_rank tokens on this rank.

    topk_idx/topk_weights are this rank's contiguous slice of the global routing trace
    (host tensors; moved to device at make_problem time); activations are the rank's token
    activations (already on device). The global trace is retained so Pass 1 can compute
    routing statistics and input snapshots.
    """

    tokens_per_rank: int
    topk_idx: "torch.Tensor"
    topk_weights: "torch.Tensor"
    activations: "torch.Tensor"
    global_idx: "torch.Tensor | None" = None
    global_weights: "torch.Tensor | None" = None


@dataclass
class WorkloadSpec:
    """Numeric shape + materialised inputs for one fully-specified sweep line.

    Fully default-constructible so make_inputs can early-return a tensor-free
    spec (ok=False + rc) on an empty ladder; the driver prints message and
    returns rc
    """

    ok: bool = True
    rc: int = 0
    message: str = ""
    ep_size: int = 0
    experts_per_rank: int = 0
    cap: "int | None" = None
    dropped: list = field(default_factory=list)
    max_tokens_per_rank: int = 0
    ladder: list = field(default_factory=list)
    points: dict = field(default_factory=dict)


class EPBackend(abc.ABC):
    """One expert-parallel dispatch/combine transport under a fixed benchmark contract.

    Subclasses implement the transport (create_buffer, dispatch, stage,
    combine, recv_tokens, inspect_dispatch, combine_transformed);
    everything the driver and the oracles need beyond that is provided here.
    Combine is always BF16; an adapter that supports FP8 dispatch overrides
    SUPPORTED_PRECISIONS and the semantic_payload/_validate_quantizer hooks.
    """

    name: str = ""
    # "production" = an engine can select this transport today (vLLM `--all2all-backend`,
    # SGLang `--moe-a2a-backend`); "candidate" = a real transport we benchmark that no engine
    # ships a selector for, so its numbers describe the library, not a deployable config.
    # Mirrored in configs/platform_config.json; tests/test_matrix.py holds the two in step.
    maturity: str = ""
    SUPPORTED_MODES: tuple = ("normal",)
    # Dispatch precisions the adapter realizes. BF16 is the universal control; an
    # adapter that also sends an FP8-quantized dispatch payload widens this.
    SUPPORTED_PRECISIONS: tuple = ("bf16",)
    stage_device_work = False
    # Dispatch and combine form a single-use pair: every timed combine needs a fresh
    # dispatch and every timed dispatch must be drained by a combine (double-buffered
    # low-latency result tensors; MoRI/FlashInfer phase asserts). One flag, because a
    # handle is either reusable or it is not -- no adapter has ever needed one
    # direction without the other.
    requires_fresh_pair = False
    # Shape of the receive plane dispatch delivers -- "token-rank": one row per
    # (source token, dest rank), rank-deduplicated; "token-expert": one row per
    # (source token, expert) assignment (the low-latency padded layouts). Selects the
    # correctness oracle and the artifact's `logical_copies.wire` label; independent
    # of combine_weight_semantics (MoRI's IntraNodeLL is token-rank AND unweighted in
    # low-latency mode).
    receive_layout = "token-rank"
    # WHERE the top-k gate weight enters the combine. "unweighted-rank-sum": the staged
    # combine input carries the gate folded in -- adapters that reduce activations and
    # top-k weights independently must carry the complete local weighted expert sum in
    # the activation tensor -- and the kernel only sums. "weighted-kernel-sum": the
    # kernel multiplies by the gate itself. Selects the expected-combine arithmetic.
    combine_weight_semantics = "unweighted-rank-sum"
    # Realized wire formats recorded in the artifact. Combine is always BF16;
    # dispatch_dtype is overridden per-run by an FP8 adapter (e.g. "fp8-e4m3fn").
    dispatch_dtype = "bf16"
    combine_dtype = "bf16"
    # Logical byte model for one dispatched copy: bytes per activation value and
    # per-copy scale bytes. BF16 moves 2 bytes/value with no scale payload; an FP8
    # adapter sends 1 byte/value plus (for a blockwise codec) per-block FP32 scales.
    dispatch_value_bytes = 2
    dispatch_scale_bytes_per_copy = 0
    # Handle contract, not an attribute of this class: every adapter's stage() sets
    # handle.combine_input to the tensor its combine() reads. The value need not be a torch
    # tensor -- nccl-ep stores its own nccl.ep wrapper -- because the shared paths below
    # only ever pass it through.

    # Which production FP8 consumption path the chained roundtrip models.
    #
    #   native  (default) - the expert consumes the dispatched fp8 + per-128-block scales
    #     DIRECTLY and emits BF16, so no standalone conversion pass sits between the two
    #     collectives. This is what SGLang does (its DeepEP dispatcher contains no dequant at
    #     all) and what vLLM does whenever the expert's block shape matches DeepEP's 128
    #     (`block_k == DEEPEP_QUANT_BLOCK_SIZE` -> "DeepEP kernels did the quantization for
    #     us", fp8 + scales returned untouched). deepseek-v3 block-fp8 -- the workload this
    #     suite runs -- takes that branch.
    #   dequant - vLLM's fallback for a quant-format MISMATCH: dequant_fp8() materialises the
    #     full padded [experts, max_tokens, hidden] tensor to fp32, casts to the activation
    #     dtype, then re-quantises for the expert. A real path, just not this workload's.
    #
    # `dequant` is a VERIFICATION HATCH, not a second metric: never a sweep axis, never a
    # default. It is retained because it costs nothing (BF16 needs the same staged-is-None
    # branch) and because it still reproduces historical deepep-v2/uccl-ep numbers for regression
    # checks -- 302.0us against 302.5us in run 30177021271 at T=1. It does not reproduce MoRI fp8
    # (that stage now casts only the rows dispatch filled) or any pre-hoist BF16 roundtrip (the
    # hatch is fp8-only by design).
    #
    # A second measured mode is unnecessary: `dequant roundtrip ~= roundtrip + stage` holds to
    # within a few percent, so the mismatched-config cost is derivable from what every run
    # already emits. Measure native and derive dequant, NEVER the reverse -- reconstructing
    # native as `dequant - stage` is worst exactly in the decode regime the headline reports.
    # The derivation-accuracy ladder is in docs/methodology.md (search "fp8_consume").
    #
    # It matters because for deepep-v2 and uccl `stage` IS the fp8 conversion (both set
    # stage_device_work = self._fp8), so charging it to the chained roundtrip compares fp8 and
    # bf16 through structurally different pipelines: on run 30177021271 that inverted the
    # fp8-vs-bf16 verdict in 39 of 51 comparisons.
    fp8_consume = os.environ.get("CX_FP8_CONSUME", "native")
    if fp8_consume not in ("native", "dequant"):
        raise ValueError(f"CX_FP8_CONSUME must be 'native' or 'dequant', got {fp8_consume!r}")

    @property
    def stage_excluded_from_roundtrip(self) -> bool:
        """Whether the chained roundtrip skips the per-iteration `stage()`.

        `roundtrip` must mean dispatch -> combine -- the transport, staging excluded -- in every
        row or it cannot be compared across backends, so the answer is yes whenever `stage()`
        does device work, regardless of precision. Gating on precision as well left staging
        inside the roundtrip for MoRI BF16 scale-up and FlashInfer BF16 alone, against 800+
        transport-only rows.

        Gated on `stage_device_work` rather than applied blanket: where `stage()` is a bare
        pointer assignment there is nothing to lift, and hoisting anyway would hand the
        low-latency backends a view into their double-buffered receive, whose parity flips on
        each timed re-dispatch -- combine would then read the stale-parity buffer.

        `CX_FP8_CONSUME=dequant` opts an fp8 run back into the inline stage, to model a stack
        that really does dequantise between the two collectives (see `fp8_consume`).
        """
        if not self.stage_device_work:
            return False
        return not (self.precision == "fp8" and self.fp8_consume == "dequant")

    def fused_quantize(self, eager):
        """The fp8 quantize the TIMED dispatch should call, keyed on mode.

        Production quantises bf16->fp8 once per forward pass, fused, just before dispatch, so
        the harness compiles it once outside the timed window: charging the eager 9-launch
        composite (19.2us H100, 53.6us MI300X, against ~1.5-4.9us compiled) would publish this
        harness's kernel count rather than production's cost and flip fp8-vs-bf16 verdicts on
        that basis. Low-latency keeps the eager form -- its dispatch quantises internally so that
        cost is already in-window, and the oracle's payload gate must keep matching the eager
        helper's bits. `dynamic=False` because a dynamic build measured 6.3x slower; the cache
        limit is raised because ~20+ shapes exceed the default of 8 and overflow falls back to
        eager SILENTLY.
        """
        if self.mode == "low-latency":
            return eager
        import torch

        torch._dynamo.config.cache_size_limit = 64
        if hasattr(torch._dynamo.config, "fail_on_recompile_limit_hit"):
            # Prefer a loud failure over a silent eager fallback if the limit is ever hit.
            torch._dynamo.config.fail_on_recompile_limit_hit = True
        return torch.compile(eager, dynamic=False)

    def assert_quantize_identity(self, eager, fused, x) -> None:
        """Fail loudly, untimed, if the compiled quantize is not the eager one bit-for-bit.

        The oracle's payload gate is a `torch.equal` between the sender's [T, hidden] quantize
        and the oracle's [receive_count, hidden] one, so identity has to hold per row across
        batch sizes, not merely deterministically. Both properties were verified on-metal for
        e4m3fn and e4m3fnuz; this check names a future toolchain regression here instead of
        leaving an unexplained fleet-wide payload mismatch.
        """
        if fused is eager:
            return
        import torch

        def bits(pair):
            values, scales = pair
            return values.view(torch.uint8), scales

        eager_values, eager_scales = bits(eager(x))
        fused_values, fused_scales = bits(fused(x))
        if not (torch.equal(eager_values, fused_values)
                and torch.equal(eager_scales, fused_scales)):
            raise RuntimeError(
                "compiled fp8 quantize is not bitwise identical to the eager helper; the "
                "oracle payload gate would fail fleet-wide"
            )
        rows = min(int(x.shape[0]), 3)
        if rows:
            part_values, part_scales = bits(fused(x[:rows]))
            whole_values, whole_scales = fused_values[:rows], fused_scales[:rows]
            if not (torch.equal(part_values, whole_values)
                    and torch.equal(part_scales, whole_scales)):
                raise RuntimeError(
                    "compiled fp8 quantize is not per-row invariant across batch sizes; the "
                    "oracle compares a different row count than the sender quantised"
                )

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "name", ""):
            raise TypeError(
                f"{cls.__name__} must declare a non-empty class-level `name`"
            )

    def __init__(self, args, rank, world_size, local_rank, device):
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.device = device
        self.mode = args.mode
        if self.mode not in self.SUPPORTED_MODES:
            raise ValueError(f"{self.name} does not support mode {self.mode!r}")
        self.precision = args.precision
        if self.precision not in self.SUPPORTED_PRECISIONS:
            raise ValueError(
                f"{self.name} does not support precision {self.precision!r}"
            )

    # ---- Abstract transport contract -------------------------------------------------

    @abc.abstractmethod
    def create_buffer(self, spec: WorkloadSpec):
        """Size the communicator from spec before the first dispatch."""

    @abc.abstractmethod
    def dispatch(self, problem):
        """Scatter tokens to their experts; return an opaque per-call handle."""

    @abc.abstractmethod
    def stage(self, problem, handle):
        """Prepare the combine input on handle (copy into place)."""

    @abc.abstractmethod
    def combine(self, problem, handle):
        """Gather the staged tokens back to their source rank; return combined activations."""

    @abc.abstractmethod
    def recv_tokens(self, handle):
        """Number of tokens this rank received in dispatch (stable for a fixed trace)."""

    @abc.abstractmethod
    def inspect_dispatch(self, problem, handle):
        """Normalized post-dispatch view for the token-rank correctness oracle."""

    @abc.abstractmethod
    def combine_transformed(self, problem, handle, transformed):
        """Combine an oracle-transformed payload in place of the staged input."""

    # ---- Input generation (shared) ---------------------------------------------------

    def buffer_cap(self, args):
        """Max tokens/rank the communicator can serve, or None when unbounded."""
        return None

    def make_inputs(self, args) -> WorkloadSpec:
        """Resolve the token ladder and materialise per-rank inputs for the sweep.

        Buffer sizing needs the ladder *numbers* (not the input tensors), so this
        runs before create_buffer. Returns a tensor-free spec with ok=False
        when the ladder is empty.
        """
        ep_size = self.world_size
        experts_per_rank = args.experts // ep_size
        cap = self.buffer_cap(args)
        ladder, dropped = token_ladder(args.tokens_ladder, cap)
        if not ladder:
            return WorkloadSpec(
                ok=False, rc=2,
                message=f"empty token ladder (phase={args.phase}, cap={cap})",
            )
        spec = WorkloadSpec(
            ep_size=ep_size,
            experts_per_rank=experts_per_rank,
            cap=cap,
            dropped=list(dropped),
            max_tokens_per_rank=max(ladder),
            ladder=list(ladder),
        )
        for tokens_per_rank in ladder:
            spec.points[tokens_per_rank] = self._build_rank_inputs(args, tokens_per_rank)
        return spec

    def _build_rank_inputs(self, args, tokens_per_rank) -> RankInputs:
        """Build one rank's deterministic inputs for a tokens-per-rank shape."""
        import torch
        import routing

        ep_size = self.world_size
        global_tokens = tokens_per_rank * ep_size
        idx_g, w_g = routing.build_global_routing(
            global_tokens, args.experts, args.topk, args.routing, args.seed
        )
        idx_s, w_s = routing.rank_slice(idx_g, w_g, self.rank, tokens_per_rank)
        activations = routing.rank_activations(
            tokens_per_rank, args.hidden, args.seed, self.rank, self.device, torch.bfloat16
        )
        return RankInputs(
            tokens_per_rank=tokens_per_rank,
            topk_idx=idx_s.contiguous(),
            topk_weights=w_s.contiguous(),
            activations=activations,
            global_idx=idx_g,
            global_weights=w_g,
        )

    def semantic_payload(self, x):
        """The BF16 values the oracle should expect for a dispatched payload.

        Identity for a backend that sends x unchanged. An FP8 backend overrides this
        to apply the exact quant->dequant round-trip the kernel transports, so the
        dispatched-payload compare stays bit-exact and the combine gate stays tight.
        """
        return x

    def _validate_quantizer(self, x) -> None:
        """Per-shape hook, run untimed from make_problem. An FP8 adapter whose timed
        dispatch calls a COMPILED quantizer overrides this to assert the compiled form
        is bit-identical to the eager one (see assert_quantize_identity). The base has
        no quantizer, and low-latency adapters keep the eager form, so neither checks.
        """

    def make_problem(self, T, idx, weights, x):
        """Assemble the per-shape problem namespace.

        dispatch_x is always x: every adapter quantizes INSIDE dispatch, where production
        pays it, so nothing is prequantized by the caller. oracle_x is semantic_payload(x)
        -- identity in BF16, and the exact quant->dequant round-trip the wire performs under
        FP8, so the combine gate stays tight without a tolerance change. Computing it here
        also compiles this rung's quantizer shape outside the timed window.
        """
        import torch

        self._validate_quantizer(x)
        return types.SimpleNamespace(
            T=T,
            x=x,
            dispatch_x=x,
            oracle_x=self.semantic_payload(x),
            topk_idx=idx.to(self._topk_idx_dtype()),
            topk_weights=weights.to(torch.float32),
        )

    def _topk_idx_dtype(self):
        """Integer dtype the backend's kernels expect for top-k routing indices."""
        import torch
        return torch.int64

    # ---- Timing template methods -----------------------------------------------------

    def timed_components(self):
        """Components measured for this backend: roundtrip, dispatch and combine
        always; stage only when it launches device work."""
        components = ["roundtrip", "dispatch", "combine"]
        if self.stage_device_work:
            components.append("stage")
        return components

    def warm(self, problem, count, stage_every=False):
        """Untimed synchronized full round trips (fabric/clock warm-up; cold-jump-safe).

        Caches the dynamic receive cardinality once so adapters never read a device
        scalar during a timed trial (the count is stable for a fixed routing trace).

        `stage_every` re-materialises the combine input on every iteration; the default hoists it
        after the first, mirroring `benchmark_roundtrip`. Where staging is excluded from the chain
        the timed region stages nothing, so warming it warms work the measurement never performs
        -- ~247us per FP8 dequant against a 61us roundtrip, the leg's largest single cost.
        `benchmark_stage` opts in, because there staging is the timed operation.
        """
        import torch

        staged = None
        for _ in range(count):
            handle = self.dispatch(problem)
            if not hasattr(problem, "recv_tokens"):
                problem.recv_tokens = self.recv_tokens(handle)
            if staged is None:
                self.stage(problem, handle)
                if not stage_every and self.stage_excluded_from_roundtrip:
                    staged = handle.combine_input
            else:
                handle.combine_input = staged
            self.combine(problem, handle)
            torch.cuda.synchronize()

    def run_roundtrip(self, problem, staged=None):
        """One chained round trip; returns combined activations.

        `staged` supplies a pre-materialised combine input so staging stays out of the timed
        region -- the default wherever `stage()` does device work (see
        `stage_excluded_from_roundtrip`). It is None where `stage()` is a bare pointer
        assignment, or under the `CX_FP8_CONSUME=dequant` hatch that wants it back in the chain.
        """
        handle = self.dispatch(problem)
        if staged is None:
            self.stage(problem, handle)
        else:
            handle.combine_input = staged
        return self.combine(problem, handle)

    def benchmark_chain(self, problem, warmup, iters, drop):
        """Free-running dispatch->combine pairs, no host sync: a floors chain, then a period chain.

        This is what a serving stack pays: a decode loop never stops between layers, so entry
        skew amortises across the chain instead of landing on one op the way `roundtrip`'s
        drained windows charge it. The pairing is `run_roundtrip`'s, so paired-API backends stay
        in contract; every backend is measured.

        Two chains, because per-op events inside a chained pair execute immediately on an idle
        stream, landing the host's record() cost in the pair window: six events per pair
        published a flat +10-30us host constant on every vendor (+20-38% at T=1, decaying with
        T). So the floors chain carries op-window events only, the period chain one outer pair
        with nothing between its two collectives, and `chain_health.interpair_gap_us`
        (start-to-start median minus window median) guards that defect in-artifact.

        Only the pair period and the per-op minima are publishable: each rank's inter-rank wait
        parks in whichever op window it blocks in while the period is conserved, so `run_sweep`
        enforces pair -> cross-rank median, per-op -> cross-rank minimum, never a chained per-op
        median or p99.

        `drop` discards each chain's head (pipeline fill, not period). The chain's own final
        combined output is returned under `combined` -- cloned after the closing synchronize, so
        the copy is untimed and detached from any double-buffered receive the next dispatch would
        overwrite. `run_sweep` checks it against a drained pair through this same code path and
        separately reruns the full expert oracle against the state the chain leaves behind; both
        fold into the point's verdict. Interior pairs stay unvalidated by design -- each pair
        overwrites its predecessor's output, and holding or reducing every output would put
        device work inside the timed loops (see methodology, Correctness).
        Free-running is safe fleet-wide: every backend double-buffers per dispatch or completes
        each op on a reusable handle, and deepep-v2 NORMAL probed clean with 256 un-synced pairs
        (T=128, EP8+EP16, both precisions, 2026-08-06, pin 01dc3aaa). Returns post-`drop` series
        in microseconds: `pair` and `start_to_start` from the period chain (the latter one
        element shorter), `dispatch` and `combine` from the floors chain.
        """
        import torch

        self.warm(problem, warmup)
        staged = None
        if self.stage_excluded_from_roundtrip:
            # The same hoist `benchmark_roundtrip` performs, so the chain is dispatch -> combine
            # and nothing else. The `CX_FP8_CONSUME=dequant` hatch leaves `staged` None, putting
            # the conversion inside the pair period and inside neither per-op window -- where
            # work between the two collectives belongs.
            handle = self.dispatch(problem)
            self.stage(problem, handle)
            staged = handle.combine_input
            self.combine(problem, handle)  # drain the pair backends require
            torch.cuda.synchronize()
        # Events are allocated BEFORE the loops: an allocation between two record() calls is host
        # work inside a window meant to belong to the stream, a measurable fraction of the period
        # at the bottom of the ladder.
        def events():
            return [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

        dispatch_start, dispatch_end = events(), events()
        combine_start, combine_end = events(), events()
        pair_start, pair_end = events(), events()

        # ---- Floors chain: op windows only, pair boundaries uninstrumented. ----
        for i in range(iters):
            dispatch_start[i].record()
            handle = self.dispatch(problem)
            dispatch_end[i].record()
            if staged is None:
                self.stage(problem, handle)
            else:
                handle.combine_input = staged
            combine_start[i].record()
            self.combine(problem, handle)
            combine_end[i].record()
        torch.cuda.synchronize()

        # ---- Period chain: nothing between the pair's collectives but the pair itself. ----
        for i in range(iters):
            pair_start[i].record()
            handle = self.dispatch(problem)
            if staged is None:
                self.stage(problem, handle)
            else:
                handle.combine_input = staged
            combined = self.combine(problem, handle)
            pair_end[i].record()
        torch.cuda.synchronize()

        def series(starts, ends):
            return [
                start.elapsed_time(end) * 1000.0  # ms -> us
                for start, end in zip(starts[drop:], ends[drop:])
            ]

        return {
            "pair": series(pair_start, pair_end),
            "start_to_start": series(pair_start[:-1], pair_start[1:]),
            "dispatch": series(dispatch_start, dispatch_end),
            "combine": series(combine_start, combine_end),
            # The period chain's final combined output, produced IN the free-running regime.
            # Cloned post-sync (untimed, stream-ordered ahead of any later dispatch) so the
            # caller can compare it against a drained pair without racing the buffers.
            "combined": combined.clone(),
        }

    def benchmark_component(self, component, problem, warmup, iters):
        """Measure one named component; every component gets the same warm-up first."""
        if component == "roundtrip":
            return self.benchmark_roundtrip(problem, warmup, iters)
        if component == "dispatch":
            return self.benchmark_dispatch(problem, warmup, iters)
        if component == "stage":
            return self.benchmark_stage(problem, warmup, iters)
        if component == "combine":
            return self.benchmark_combine(problem, warmup, iters)
        raise RuntimeError(f"unknown timed component {component!r}")

    def benchmark_roundtrip(self, problem, warmup, iters):
        import torch

        self.warm(problem, warmup)
        staged = None
        if self.stage_excluded_from_roundtrip:
            # Materialise the expert-output stand-in ONCE, untimed, so the chained measurement is
            # dispatch -> combine and nothing else. Routing is fixed for a ladder point, so the
            # same staged tensor is valid for every iteration -- MoRI's is the dispatch output at
            # BF16 or a `[:rows]` BF16 cast under FP8, FlashInfer's the workspace combine region,
            # which sits past the end of every dispatch receive plane. Read back through
            # `handle.combine_input` rather than constructed, so an adapter's non-torch payload
            # (nccl-ep) would round-trip unchanged if one ever reached here.
            handle = self.dispatch(problem)
            self.stage(problem, handle)
            staged = handle.combine_input
            self.combine(problem, handle)  # drain the pair backends require
            torch.cuda.synchronize()
        return time_us(torch, lambda p=problem: self.run_roundtrip(p, staged), 0, iters)

    def benchmark_dispatch(self, problem, warmup, iters):
        import torch

        self.warm(problem, warmup)

        def finish_dispatch(hh, p=problem):
            self.stage(p, hh)
            self.combine(p, hh)

        return time_us(
            torch, lambda p=problem: self.dispatch(p), 0, iters,
            post=finish_dispatch if self.requires_fresh_pair else None,
        )

    def benchmark_stage(self, problem, warmup, iters):
        import torch

        # Staging is the timed operation here, so it must be warmed on every iteration.
        self.warm(problem, warmup, stage_every=True)

        def prep_stage(p=problem):
            return self.dispatch(p)

        def stage_op(hh, p=problem):
            self.stage(p, hh)
            return hh

        # Drain each timed stage's dispatch with an untimed combine where the
        # backend requires the pair (same rule as benchmark_dispatch).
        return time_us(
            torch, stage_op, 0, iters, pre=prep_stage,
            post=(lambda hh, p=problem: self.combine(p, hh))
            if self.requires_fresh_pair else None,
        )

    def benchmark_combine(self, problem, warmup, iters):
        import torch

        self.warm(problem, warmup)

        def prep_combine(p=problem):
            hh = self.dispatch(p)
            self.stage(p, hh)
            return hh

        if self.requires_fresh_pair:
            return time_us(
                torch, lambda hh, p=problem: self.combine(p, hh), 0, iters, pre=prep_combine,
            )
        hh = prep_combine()
        torch.cuda.synchronize()
        return time_us(torch, lambda p=problem, hx=hh: self.combine(p, hx), 0, iters)

    def finalize(self, rc):
        """Barrier and tear down the process group; returns rc."""
        import torch.distributed as dist

        try:
            dist.barrier()
            dist.destroy_process_group()
        except Exception:
            pass
        return rc
