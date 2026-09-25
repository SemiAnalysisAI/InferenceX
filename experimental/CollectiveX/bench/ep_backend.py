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

from ep_case import token_ladder
from ep_timing import EPTiming


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


class EPBackend(EPTiming, abc.ABC):
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

    def finalize(self, rc):
        """Barrier and tear down the process group; returns rc."""
        import torch.distributed as dist

        try:
            dist.barrier()
            dist.destroy_process_group()
        except Exception:
            pass
        return rc
