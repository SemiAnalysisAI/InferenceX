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

from ep_oracle import combine_model
from ep_timing import EagerTiming, GraphTiming


def token_ladder(spec: str, cap: int | None) -> tuple[list[int], list[int]]:
    """(ladder, dropped) from an explicit spec: unique positive ints, clamped to `cap` with the
    dropped points reported rather than silently truncated."""
    want = sorted({t for t in (int(t) for t in spec.replace(",", " ").split() if t) if t > 0})
    if cap is not None:
        return [t for t in want if t <= cap], [t for t in want if t > cap]
    return want, []


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
    Combine is always BF16; an adapter that supports FP8 dispatch widens
    SUPPORTED_PRECISIONS and adopts its codec with `_enable_fp8`.
    """

    name: str = ""
    # "production" = an engine can select this transport today (vLLM `--all2all-backend`,
    # SGLang `--moe-a2a-backend`); "candidate" = a real transport we benchmark that no engine
    # ships a selector for, so its numbers describe the library, not a deployable config.
    # Mirrored by hand in configs/platform_config.json `backend_maturity`; nothing checks the two agree.
    maturity: str = ""
    SUPPORTED_MODES: tuple = ("normal",)
    # Dispatch precisions the adapter realizes. BF16 is the universal control; an
    # adapter that also sends an FP8-quantized dispatch payload widens this.
    SUPPORTED_PRECISIONS: tuple = ("bf16",)
    stage_device_work = False
    # Modes whose fixed-shape dispatch -> stage -> combine roundtrip is safe to capture and
    # replay. Graph-capable modes use replay by default; COLLX_CUDA_GRAPH=0 restores eager timing.
    CUDA_GRAPH_MODES: tuple = ()
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
    # (eager quantize, dequantize) of an adopted blockwise FP8 codec; see `_enable_fp8`.
    _fp8_codec = None
    # Low-latency receives are pre-allocated at a fixed per-rank slot count, so this bounds the
    # measured ladder: every backend's LL ladder ends at the same rung. 256 is also vLLM's
    # DEFAULT_MAX_NUM_BATCHED_TOKENS_FOR_BATCHED_DP.
    LL_LADDER_CAP = 256
    # The expected-combine reduction the oracle holds the kernel to (see ep_oracle.combine_model);
    # published, because a backend may pick it per installed library version.
    combine_reduction = "domain-fp32"
    kernel_generation: "str | None" = None
    library_version: "str | None" = None
    mode: "str | None" = None
    # Handle contract, not an attribute of this class: every adapter's stage() sets
    # handle.combine_input to the tensor its combine() reads. The value need not be a torch
    # tensor -- nccl-ep stores its own nccl.ep wrapper -- because the shared paths below
    # only ever pass it through.

    # Which production FP8 consumption path the chained roundtrip models (methodology, "fp8_consume").
    # native (default): the expert consumes the dispatched fp8 + scales directly, as SGLang and
    # vLLM do for this workload's 128-block shape, so no conversion sits between the collectives.
    # dequant: vLLM's quant-format-mismatch fallback, which materialises and re-quantises. It is a
    # verification hatch, never a sweep axis: `dequant roundtrip ~= roundtrip + stage` holds within
    # a few percent, so measure native and derive dequant. Charging stage to the fp8 roundtrip
    # inverted the fp8-vs-bf16 verdict in 39 of 51 comparisons.
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

    @property
    def cuda_graph_supported(self) -> bool:
        """Whether this realized backend/mode has a graph-safe fixed-shape roundtrip."""
        return self.mode in self.CUDA_GRAPH_MODES

    @property
    def cuda_graph_enabled(self) -> bool:
        """Use CUDA graph replay unless the external eager switch disables it."""
        setting = os.environ.get("COLLX_CUDA_GRAPH", "1")
        if setting not in ("0", "1"):
            raise ValueError(f"COLLX_CUDA_GRAPH must be '0' or '1', got {setting!r}")
        return self.cuda_graph_supported and setting == "1"

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

    @staticmethod
    def init_process_group(dist, rank, world_size, device):
        """Form the default NCCL group, device-only so scale-out does not also depend on a host Gloo
        fabric. MoRI registers it with its SHMEM runtime, UCCL-EP bootstraps its Buffer and CPU-proxy
        ranks from it, and NCCL EP uses it only to broadcast its own communicator's unique id."""
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, device_id=device)

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
        return self.LL_LADDER_CAP if self.mode == "low-latency" else None

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

    def _enable_fp8(self, dispatch_dtype, quantize, dequantize):
        """Adopt a blockwise FP8 dispatch codec: 1 byte/value plus one FP32 scale per 128-value
        block. The timed dispatch calls `self._quant`, compiled outside the window except in
        low-latency (see fused_quantize)."""
        self.dispatch_dtype = dispatch_dtype
        self.dispatch_value_bytes = 1
        self.dispatch_scale_bytes_per_copy = ((self.args.hidden + 127) // 128) * 4
        self._fp8_codec = (quantize, dequantize)
        self._quant = self.fused_quantize(quantize)

    def semantic_payload(self, x):
        """The BF16 values the oracle should expect for a dispatched payload.

        Identity for a backend that sends x unchanged. Under an FP8 codec it is the exact
        quant->dequant round-trip the kernel transports, through the same callable the wire uses,
        so the dispatched-payload compare stays bit-exact and the combine gate stays tight.
        """
        if self._fp8_codec is None:
            return x
        return self._fp8_codec[1](*self._quant(x))

    def _validate_quantizer(self, x) -> None:
        """Per-shape hook, run untimed from make_problem: under an FP8 codec, assert the compiled
        quantizer the timed dispatch calls is bit-identical to the eager one (a no-op where
        fused_quantize kept the eager form)."""
        if self._fp8_codec is not None:
            self.assert_quantize_identity(self._fp8_codec[0], self._quant, x)

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

    # ---- Oracle views (shared) -------------------------------------------------------

    def _local_id_view(self, payload, local_idx, weights, experts_per_rank):
        """Token-rank view of a receive whose top-k ids are LOCAL expert indices, -1 where the
        expert is not this rank's: rebase the valid ones to the GLOBAL ids the oracle compares."""
        import torch

        local_idx = local_idx.to(torch.int64)
        valid = local_idx >= 0
        return types.SimpleNamespace(
            payload=payload,
            expert_ids=torch.where(valid, local_idx + self.rank * experts_per_rank, local_idx),
            weights=weights.to(torch.float32).masked_fill(~valid, 0),
            local_expert_counts=torch.bincount(local_idx[valid], minlength=experts_per_rank),
        )

    def _global_id_view(self, payload, ids, weights, experts_per_rank):
        """Token-rank view of a receive whose top-k ids come back GLOBAL and cover the token's
        whole top-k, including other ranks' experts: those are masked to -1 with zero weight, as
        the oracle's expectation has them."""
        import torch

        lo = self.rank * experts_per_rank
        local = (ids >= lo) & (ids < lo + experts_per_rank)
        return types.SimpleNamespace(
            payload=payload,
            expert_ids=torch.where(local, ids, torch.full_like(ids, -1)),
            weights=weights.masked_fill(~local, 0.0),
            local_expert_counts=torch.bincount(ids[local] - lo, minlength=experts_per_rank),
        )

    def _expert_major_view(self, h, recv, recv_count):
        """Flat per-slot view over an expert-major padded receive `[num_local_experts, slots, hidden]`.

        Each local expert's valid tokens are packed at the front `[0:recv_count[e]]` of its slot
        dimension. Flatten to the oracle's compact contract in `(expert, slot)` row-major order --
        nonzero yields C-order indices, e ascending then j ascending -- keeping the coordinates on
        the handle so combine_transformed can scatter the transformed rows back 1:1.
        """
        import torch

        counts = recv_count.to(torch.int64)
        slot_valid = (
            torch.arange(recv.shape[1], device=recv.device).unsqueeze(0) < counts.unsqueeze(1)
        )
        h.slot_expert, h.slot_j = slot_valid.nonzero(as_tuple=True)
        return types.SimpleNamespace(
            payload=recv[h.slot_expert, h.slot_j],
            expert_ids=self.rank * self.num_local_experts + h.slot_expert.to(torch.int64),
            local_expert_counts=counts,
        )

    # ---- Timing: EagerTiming / GraphTiming (ep_timing.py) run the windows -----------------

    @property
    def combine_model(self):
        """The expected-combine model for this backend's declared semantics and reduction."""
        return combine_model(self.combine_weight_semantics, self.combine_reduction)

    @property
    def timing(self):
        """The timing strategy for this backend's regime, kept for the backend's lifetime so graph
        alignment state persists; rebuilt if COLLX_CUDA_GRAPH flips the regime."""
        graph = self.cuda_graph_enabled
        timing = self.__dict__.get("_timing")
        if timing is None or timing.graph != graph:
            timing = self._timing = (GraphTiming if graph else EagerTiming)(self)
        return timing

    def timed_components(self):
        return self.timing.components()

    def warm(self, problem, count, stage_every=False):
        """Untimed synchronized full round trips (fabric/clock warm-up; cold-jump-safe).

        Caches the receive cardinality once so no adapter reads a device scalar during a timed
        trial. `stage_every` re-stages every iteration; the default hoists after the first, as the
        timed roundtrip does, because warming staging the measurement never performs cost ~247us
        per FP8 dequant against a 61us roundtrip. Stage timing opts in, since staging is timed there.
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

    def stage_or_reuse(self, problem, handle, staged):
        # Graph capture only. The eager timed loops inline this branch so no Python call sits
        # between the dispatch and combine launches inside a timed window.
        if staged is None:
            self.stage(problem, handle)
        else:
            handle.combine_input = staged

    def run_roundtrip(self, problem, staged=None):
        """One dispatch -> combine; `staged` supplies a pre-materialised combine input so staging
        stays out of the timed region (see `stage_excluded_from_roundtrip`)."""
        handle = self.dispatch(problem)
        if staged is None:
            self.stage(problem, handle)
        else:
            handle.combine_input = staged
        return self.combine(problem, handle)

    def warm_and_hoist_stage(self, problem, warmup):
        """Warm, then materialise the combine input ONCE, untimed, where staging is excluded from
        the roundtrip. Routing is fixed per ladder point, so one staged tensor serves every pair;
        it is read back through `handle.combine_input`, so a non-torch payload (nccl-ep) passes
        through unchanged. Returns None where each pair stages inline."""
        import torch

        self.warm(problem, warmup)
        if not self.stage_excluded_from_roundtrip:
            return None
        handle = self.dispatch(problem)
        self.stage(problem, handle)
        staged = handle.combine_input
        self.combine(problem, handle)  # drain the pair backends require
        torch.cuda.synchronize()
        return staged

    def benchmark_component(self, component, problem, warmup, iters):
        """Measure one named component; every component gets the same warm-up first."""
        return self.timing.component(component, problem, warmup, iters)

    def benchmark_chain(self, problem, warmup, iters, drop):
        """Free-running dispatch->combine pairs, no host sync between them: what a decode loop pays.

        Entry skew amortises across the chain instead of landing on one op. Only the pair period and
        the per-op minima are publishable: each rank's wait parks in whichever op window it blocks
        in while the period is conserved (run_sweep reduces pair -> median, per-op -> minimum).
        `drop` discards each chain's head (pipeline fill). The final combined output is returned
        under `combined` for run_sweep's chained-vs-drained check; interior pairs stay unvalidated
        by design (holding their outputs would put device work inside the timed loops).
        Free-running is safe fleet-wide: every backend double-buffers per dispatch or completes each
        op on a reusable handle (deepep-v2 NORMAL probed clean with 256 un-synced pairs).
        """
        staged = self.warm_and_hoist_stage(problem, warmup)
        return self.timing.chain(problem, staged, iters, drop)

    def graph_replay_output(self, problem):
        """Graph replay's value check (GraphTiming.replay_output)."""
        return self.timing.replay_output(problem)

    def finalize(self, rc):
        """Barrier and tear down the process group; returns rc."""
        import torch.distributed as dist

        try:
            dist.barrier()
            dist.destroy_process_group()
        except Exception:
            pass
        return rc
