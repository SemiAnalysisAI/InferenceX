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
    SUPPORTED_PRECISIONS and the semantic_payload/_encode_dispatch hooks.
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
    combine_needs_redispatch = False
    dispatch_needs_combine_cleanup = False
    # Adapters that reduce activations and top-k weights independently must carry
    # the complete local weighted expert sum in the activation tensor.
    combine_weight_semantics = "unweighted-rank-sum"
    roundtrip_only = False
    # Realized wire formats recorded in the artifact. Combine is always BF16;
    # dispatch_dtype is overridden per-run by an FP8 adapter (e.g. "fp8-e4m3fn").
    dispatch_dtype = "bf16"
    combine_dtype = "bf16"
    # Logical byte model for one dispatched copy: bytes per activation value and
    # per-copy scale bytes. BF16 moves 2 bytes/value with no scale payload; an FP8
    # adapter sends 1 byte/value plus (for a blockwise codec) per-block FP32 scales.
    dispatch_value_bytes = 2
    dispatch_scale_bytes_per_copy = 0
    # Handle attribute stage() populates with the tensor combine sends. Every adapter must
    # point this at the tensor its combine() reads (NCCL EP names it differently).
    combine_input_attr = "combine_input"
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
    # branch) and because it reproduces historical numbers for regression checks on the
    # backends whose stage this repo has not since changed -- measured 302.0us against 302.5us
    # in run 30177021271 at T=1 for deepep-v2/uccl-ep. It no longer reproduces MoRI fp8, whose
    # stage now casts only the rows dispatch filled rather than the whole padded plane; and no
    # env combination reproduces pre-hoist BF16 MoRI/FlashInfer roundtrips, because the hatch
    # is fp8-only by design.
    #
    # A second measured mode is unnecessary because the mismatched-config cost is DERIVABLE
    # from what every run already emits:
    #
    #     dequant roundtrip  ~=  roundtrip + stage        (+2.4% .. -0.1%, b200 LL fp8 ladder)
    #
    # slightly high because chaining amortises launch overhead (median rt/(d+s+c) = 0.93
    # across the corpus). The reverse does NOT hold -- reconstructing native as
    # `dequant - stage` errs by -11.6% at T=1, -5% at T=64, and only converges by T=256,
    # i.e. it is worst exactly in the decode regime the headline reports. So measure native
    # and derive dequant, never the other way round.
    #
    # It matters because for deepep-v2 and uccl `stage` is precisely the fp8 conversion
    # (both set stage_device_work = self._fp8), so charging it to the chained roundtrip
    # compares fp8 and bf16 through structurally different pipelines. On run 30177021271
    # that inverted the fp8-vs-bf16 verdict in 39 of 51 comparisons. dispatch and combine
    # were always measured stage-free; only the chained roundtrip mixed it in.
    fp8_consume = os.environ.get("CX_FP8_CONSUME", "native")

    @property
    def stage_excluded_from_roundtrip(self) -> bool:
        """Whether the chained roundtrip skips the per-iteration `stage()`.

        `roundtrip` must mean the same thing in every row, or it cannot be compared across
        backends. It means dispatch -> combine: the transport, staging excluded. So the
        answer is yes whenever `stage()` does device work, regardless of precision.

        This was previously gated on precision as well, which left `stage` inside the
        roundtrip for exactly two configurations -- MoRI BF16 scale-up and FlashInfer BF16 --
        and transport-only for all 800+ other rows, so the headline compared two different
        quantities and penalised those two.

        Gated on `stage_device_work` rather than applied blanket: where `stage()` is a bare
        pointer assignment there is nothing to lift, and hoisting anyway would hand the
        low-latency backends a VIEW into their double-buffered receive, whose parity flips on
        each timed re-dispatch -- combine would then read the stale-parity buffer.

        `CX_FP8_CONSUME=dequant` still opts an fp8 run back into the inline stage, because
        that switch exists to model a stack that really does dequantise between the two
        collectives (see `fp8_consume`).
        """
        if not self.stage_device_work:
            return False
        return not (self.precision == "fp8" and self.fp8_consume == "dequant")

    def fused_quantize(self, eager):
        """The fp8 quantize the TIMED dispatch should call, keyed on mode.

        Production quantises bf16->fp8 once per forward pass, with a single fused kernel, just
        before the dispatch collective. This benchmark used to do it once per SHAPE in
        `make_problem` -- outside every timed window -- so it omitted a real cost. Moving the
        eager helper inside the window would have been worse than omitting it: the eager form is
        a 9-launch composite (measured 19.2us on H100, 53.6us on MI300X) against ~1.5-4.9us for
        the compiled single kernel, so charging it would publish this harness's kernel count
        rather than production's cost, and flip fp8-vs-bf16 verdicts on that basis.

        LOW-LATENCY GETS THE EAGER FORM, unchanged and deliberately. Its dispatch kernel
        quantises internally (`use_fp8`), so the cost is already inside the timed window, and
        the oracle's payload gate compares the received bytes against `semantic_payload` -- which
        must therefore keep matching the kernel's arithmetic, i.e. the eager helper's bits.
        Swapping this globally would red every low-latency fp8 cell without touching LL timing.

        Compiled with `dynamic=False`: a dynamic-shape build of this same math measured 6.3x
        slower once, and here that lands inside the timed window as silently inflated numbers.
        The cache limit is raised because the quantize legitimately sees ~20+ shapes (one per
        ladder rung, plus the oracle's receive-count shapes) against a default of 8, and
        exceeding it makes dynamo fall back to EAGER SILENTLY -- the same failure class.
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

        The correctness oracle's payload gate is a `torch.equal`, and it compares the SENDER's
        [T, hidden] quantize against the ORACLE's [receive_count, hidden] one -- so identity has
        to hold per row, across batch sizes, not merely deterministically. Both properties were
        verified on-metal for e4m3fn (H100) and e4m3fnuz (MI300X/MI325X) before this was enabled;
        this check is what turns a future toolchain regression into a named failure here instead
        of an unexplained payload mismatch across the whole fleet.
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

    def _encode_dispatch(self, x):
        """Return (dispatch_payload, oracle_semantic) for the source activations x.

        Base identity: send x, no separate oracle payload (BF16). An FP8 adapter
        returns the caller-prequantized dispatch payload and the dequantized BF16 the
        oracle must expect after the backend's own dequant.
        """
        return x, None

    def make_problem(self, T, idx, weights, x):
        """Assemble the per-shape problem namespace.

        dispatch_x is the payload actually sent (x itself in BF16; the caller-
        prequantized encoding under FP8). oracle_x, when set, is the dequantized BF16
        the combine oracle must expect, so the tight gate needs no tolerance change.
        """
        import torch

        dispatch_x, oracle_semantic = self._encode_dispatch(x)
        problem = types.SimpleNamespace(
            T=T,
            x=x,
            dispatch_x=dispatch_x,
            topk_idx=idx.to(self._topk_idx_dtype()),
            topk_weights=weights.to(torch.float32),
        )
        if oracle_semantic is not None:
            problem.oracle_x = oracle_semantic
        return problem

    def _topk_idx_dtype(self):
        """Integer dtype the backend's kernels expect for top-k routing indices."""
        import torch
        return torch.int64

    # ---- Timing template methods -----------------------------------------------------

    # Pairs issued back-to-back per `period` sample; 0 disables the component. A serving decode
    # loop runs dispatch->combine->dispatch->combine without stopping, so its cost per layer is
    # the pipeline's PERIOD, not the sum of separately-drained stages -- `roundtrip` measures the
    # latter and overstates the former. Opt-in rather than default because the overlap is only
    # sound where the backend tolerates rank drift: dispatch is a peer WRITE into another rank's
    # buffer, and stream order on the receiver does not order the sender's remote writes. The
    # collective bounds that drift to roughly one iteration (a dispatch cannot complete until
    # every rank enters it), so a backend whose receive buffer is double-buffered per dispatch is
    # safe and one with a single shared buffer is not. Enabling it for a backend that is not
    # means a fast number over corrupted data, which is why the default is off.
    pipeline_pairs = 0

    def timed_components(self):
        """Components measured for this backend: roundtrip always; the rest unless
        the backend exposes only a stateful paired round trip."""
        components = ["roundtrip"]
        if not self.roundtrip_only:
            components.extend(["dispatch", "combine"])
            if self.stage_device_work:
                components.append("stage")
        if self.pipeline_pairs > 1:
            components.append("period")
        return components

    def warm(self, problem, count, stage_every=False):
        """Untimed synchronized full round trips (fabric/clock warm-up; cold-jump-safe).

        Caches the dynamic receive cardinality once so adapters never read a device
        scalar during a timed trial (the count is stable for a fixed routing trace).

        `stage_every` re-materialises the combine input on every iteration. The default hoists it
        after the first, mirroring `benchmark_roundtrip` on the same reasoning: where staging is
        excluded from the chain the timed region stages nothing at all, so staging once per warm
        iteration warms work the measurement never performs. For an FP8 dequant that is not free --
        it is ~247us against a 61us roundtrip, and at 32 warm iterations per component per trial it
        was the largest single cost in the leg. `benchmark_stage` opts in, because there staging IS
        the timed operation and its warm-up has to match.
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
                    staged = getattr(handle, self.combine_input_attr)
            else:
                setattr(handle, self.combine_input_attr, staged)
            self.combine(problem, handle)
            torch.cuda.synchronize()

    def run_roundtrip(self, problem, staged=None):
        """One chained round trip; returns combined activations.

        `staged` supplies a pre-materialised combine input so staging stays out of the timed
        region, which is the default for every backend that does device work there (see
        `stage_excluded_from_roundtrip`). It is None only when `stage()` is a bare pointer
        assignment -- deepep-v2, uccl-ep and nccl-ep at BF16, where there is nothing to lift --
        or under the `CX_FP8_CONSUME=dequant` hatch, which wants the conversion back in the chain.
        """
        handle = self.dispatch(problem)
        if staged is None:
            self.stage(problem, handle)
        else:
            setattr(handle, self.combine_input_attr, staged)
        return self.combine(problem, handle)

    def benchmark_period(self, problem, warmup, iters):
        """Steady-state cost per dispatch->combine pair, pairs issued back-to-back.

        `roundtrip` drains the GPU around every pair, so it reports the latency of an idle
        pipeline and charges inter-rank entry stagger to whichever component the ranks entered
        unevenly. A decode loop never stops between layers, so what it pays per layer is this
        period. The two are different quantities, not competing estimates of one: quote
        `roundtrip` for how long a single collective takes and `period` for what a continuous
        stream costs, and never sum or compare them across backends -- only backends that opt in
        via `pipeline_pairs` report it at all.
        """
        import torch

        self.warm(problem, warmup)
        pairs = max(2, int(self.pipeline_pairs))
        samples = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(pairs):
                handle = self.dispatch(problem)
                self.stage(problem, handle)
                self.combine(problem, handle)
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end) * 1000.0 / pairs)
        return samples

    # Whether this backend needs its ranks re-aligned between chained pairs. `benchmark_chain`
    # runs pairs back-to-back with no host sync -- that IS the measurement -- but the same rank
    # drift `pipeline_pairs` reasons about applies: a receive buffer that cannot absorb roughly
    # one iteration of drift would be read half-written. A backend that wedges sets this and gets
    # a one-element on-stream all_reduce between pairs.
    #
    # SETTING IT SUPPRESSES PUBLICATION. The barrier adds its own ~10us and removes the cross-pair
    # overlap, so what the chain then measures is a different quantity from the free-running period
    # this suite defines -- and the published period has to have exactly ONE definition, or two
    # cells ranked against each other may not be answering the same question. So `run_sweep` emits
    # `pair_period`, `chain_floor_us` and `chain_health` as unavailable in barrier mode and sends
    # the measured numbers to rank-0 stdout instead, with `implementation.chain_barrier` left as
    # the record of why. The valve is a BRING-UP DIAGNOSTIC for a backend that cannot be chained
    # free: a backend needing it is a bug to fix, not a variant to compare. Defaults live here,
    # not in the adapters, so the decision is auditable in one place.
    #
    # NOTHING SETS IT TODAY, on evidence rather than optimism. deepep-v2's NORMAL mode was the one
    # genuinely unknown cell -- ElasticBuffer's inter-iteration flow control had not been audited
    # for un-synced chaining -- so it was hand-probed on b200 (2026-08-06, pin 01dc3aaa) with 256
    # un-synchronized pairs at T=128 across EP8 and EP16 (hybrid GIN over RoCE) at both precisions:
    # all four passed, with finite outputs, timed inputs unchanged, and cross-rank period agreement
    # within 1us. Every other backend either double-buffers per dispatch (deepep-v2/uccl-ep
    # low-latency), enforces strict pairing through its own contract flags (MoRI, FlashInfer EP),
    # or completes each op on a reusable handle (nccl-ep). The valve stays implemented because the
    # next backend or the next pin is not covered by that probe.
    chain_barrier = False

    def benchmark_chain(self, problem, warmup, iters, drop):
        """Free-running dispatch->combine pairs, no host sync: a floors chain, then a period chain.

        This is the number a serving stack pays. `roundtrip` drains the GPU around every pair, so
        it reports an idle pipeline's latency and charges inter-rank entry stagger to whichever
        component the ranks happened to enter unevenly; a decode loop never stops between layers,
        so the collectives phase-lock the ranks and entry skew amortises across the chain instead
        of landing on one op. Both are real -- see `benchmark_period` for the same argument at the
        level of the opt-in `period` component -- but only this one is measured for every backend,
        because here the pairing is preserved exactly as `run_roundtrip` orders it (dispatch, then
        stage-or-staged, then combine) and the paired-API backends therefore stay in contract.

        WHY TWO PASSES. This method's first version timed one chain with six record() calls per
        pair -- pair start/end plus both op windows. When the device drains faster than the host
        enqueues (the bottom of the ladder), every event's timestamp is set the moment the idle
        stream reaches it, so the pair window degenerates to HOST elapsed time and the four inner
        record() calls (plus the glue between them) are charged into the published period. The
        fleet said so before a profiler did: period minus the sum of the op floors sat at a
        roughly T-independent 10-30us on every vendor and fabric at once -- a host constant, not
        transport -- inflating the T=1 period by 20-38%. So the statistics are now collected from
        two sibling chains, each carrying only the events its statistic needs:

          * FLOORS chain: op-window events only. Its per-op minima are the publishable floors,
            and the records sit at the window edges, where they always were -- the floor was
            never the contaminated statistic (it tracks profiler kernel time to ~10%).
          * PERIOD chain: one pair-window event pair and nothing between the two collectives.
            Both records' host cost lands in the inter-pair gap, OUTSIDE the window, so the
            published period carries what any uninstrumented caller would pay -- launches, glue,
            kernels -- and none of this harness's measurement apparatus.

        The floors chain runs first: it doubles as extra settling for the period chain, and
        `drop` re-discards the refill after the synchronize between them. The period chain also
        yields a start-to-start series (pair i's start to pair i+1's start); its median minus the
        pair-window median is the per-pair apparatus + stall residual, which `run_sweep` publishes
        as `chain_health.interpair_gap_us` so a regression of this exact kind is visible in every
        artifact instead of needing six of them side by side.

        WHY ONLY THE PAIR PERIOD AND THE PER-OP MINIMA ARE PUBLISHABLE. Without a host sync the
        ranks are free to arrive at each collective at slightly different times, and the resulting
        wait has to be spent somewhere: it PARKS in whichever op window a given rank happens to
        block in. That parking is stable per rank (so a single rank's per-op numbers look clean and
        convincing), arbitrary across ranks (rank 3's dispatch is long exactly where rank 5's
        combine is), and bistable across runs of the identical configuration -- while the sum, the
        pair period, is conserved. So a chained per-op MEDIAN measures where the wait sat on that
        run, not what the op cost, and the floors chain's per-op series is published only after a
        cross-rank MIN: the last rank into a collective is the one that waited least, so its window
        is the op's floor. `run_sweep` enforces that -- pair to a cross-rank median, dispatch and
        combine to a cross-rank minimum, and never a chained per-op median or p99.

        `drop` discards the head of each chain: the first pairs run into an empty pipeline and
        measure fill, not period.

        This is also a regime the correctness oracle covers. `run_sweep` runs the full expert
        oracle once per ladder point against the state the last chain leaves behind, on top of the
        drained checks it already ran before and after timing, and folds the result into the
        point's verdict -- so a backend that transports correctly when drained but corrupts under
        free-running pairs fails the case instead of publishing the suite's fastest period. The
        method ends synchronized, so that check sees a settled communicator.

        Under `chain_barrier` both chains still run but `run_sweep` publishes nothing from them
        (see the attribute's comment), and skips the chained oracle with it: there is no chained
        field left in the artifact for it to stand behind.

        Returns a dict of post-`drop` series in microseconds: `pair` and `start_to_start` from
        the period chain (the latter one element shorter), `dispatch` and `combine` from the
        floors chain.
        """
        import torch

        self.warm(problem, warmup)
        staged = None
        if self.stage_excluded_from_roundtrip:
            # The same hoist `benchmark_roundtrip` performs, for the same reason and read back
            # through the same attribute: the chain must be dispatch -> combine and nothing else.
            # The `CX_FP8_CONSUME=dequant` hatch leaves `staged` None and so keeps the conversion
            # inline, where it lands inside the pair period and inside NEITHER per-op window --
            # which is the honest placement: it is work between the two collectives, not part of
            # either one.
            handle = self.dispatch(problem)
            self.stage(problem, handle)
            staged = getattr(handle, self.combine_input_attr)
            self.combine(problem, handle)  # drain the pair backends require
            torch.cuda.synchronize()
        barrier = torch.zeros(1, device=self.device) if self.chain_barrier else None
        # Every event and the barrier tensor are allocated BEFORE the loops. An allocation between
        # two record() calls is host work inside a window that is supposed to belong to the stream,
        # and at the bottom of the ladder that is a measurable fraction of the period -- the same
        # arithmetic that split this method into two passes.
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
                setattr(handle, self.combine_input_attr, staged)
            combine_start[i].record()
            self.combine(problem, handle)
            combine_end[i].record()
            if barrier is not None:
                # Between PAIRS, never between the two collectives of one pair, and outside every
                # window so its cost sits in no published number. The serialization still shows up
                # -- as overlap the period no longer gets -- which is exactly why the artifact
                # records that this row ran with it.
                torch.distributed.all_reduce(barrier)
        torch.cuda.synchronize()

        # ---- Period chain: nothing between the pair's collectives but the pair itself. ----
        for i in range(iters):
            pair_start[i].record()
            handle = self.dispatch(problem)
            if staged is None:
                self.stage(problem, handle)
            else:
                setattr(handle, self.combine_input_attr, staged)
            self.combine(problem, handle)
            pair_end[i].record()
            if barrier is not None:
                torch.distributed.all_reduce(barrier)
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
        }

    def benchmark_component(self, component, problem, warmup, iters):
        """Measure one named component; every component gets the same warm-up first."""
        if component == "roundtrip":
            return self.benchmark_roundtrip(problem, warmup, iters)
        if component == "period":
            return self.benchmark_period(problem, warmup, iters)
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
            # Materialise the expert-output stand-in ONCE, untimed, so the chained
            # measurement is dispatch -> combine and nothing else. Routing is fixed for a
            # ladder point, so the same staged tensor is valid for every iteration (for MoRI it is
            # the dispatch output itself at BF16, or a fresh `[:rows]` BF16 cast under FP8; for
            # FlashInfer it is the workspace combine region, which dispatch cannot clobber because
            # that region sits past the end of every dispatch receive plane).
            #
            # Read the staged payload back through `combine_input_attr` rather than
            # constructing one, so whatever the adapter put there round-trips unchanged. No
            # current backend needs that -- nccl-ep is the one whose attribute holds a
            # non-torch wrapper, and its `stage()` does no device work so it never reaches
            # here -- but constructing a tensor instead would silently break it if it did.
            handle = self.dispatch(problem)
            self.stage(problem, handle)
            staged = getattr(handle, self.combine_input_attr)
            self.combine(problem, handle)  # drain the pair backends require
            torch.cuda.synchronize()
        return time_us(torch, lambda p=problem: self.run_roundtrip(p, staged), 0, iters)

    def benchmark_dispatch(self, problem, warmup, iters):
        import torch

        self.warm(problem, warmup)

        def finish_dispatch(hh, p=problem):
            self.stage(p, hh)
            self.combine(p, hh)

        dispatch_needs_cleanup = self.dispatch_needs_combine_cleanup
        return time_us(
            torch, lambda p=problem: self.dispatch(p), 0, iters,
            post=finish_dispatch if dispatch_needs_cleanup else None,
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
            if self.dispatch_needs_combine_cleanup else None,
        )

    def benchmark_combine(self, problem, warmup, iters):
        import torch

        self.warm(problem, warmup)

        def prep_combine(p=problem):
            hh = self.dispatch(p)
            self.stage(p, hh)
            return hh

        if self.combine_needs_redispatch:
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
