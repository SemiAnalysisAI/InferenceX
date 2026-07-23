#!/usr/bin/env python3
"""Shared lifecycle and input generation for EP backends."""
from __future__ import annotations

import abc
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
        num_logical = getattr(args, "num_logical_experts", args.experts)
        global_tokens = tokens_per_rank * ep_size
        idx_g, w_g = routing.build_global_routing(
            global_tokens, num_logical, args.topk, args.routing, args.seed
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

    def timed_components(self):
        """Components measured for this backend: roundtrip always; the rest unless
        the backend exposes only a stateful paired round trip."""
        components = ["roundtrip"]
        if not self.roundtrip_only:
            components.extend(["dispatch", "combine"])
            if self.stage_device_work:
                components.append("stage")
        return components

    def warm(self, problem, count):
        """Untimed synchronized full round trips (fabric/clock warm-up; cold-jump-safe).

        Caches the dynamic receive cardinality once so adapters never read a device
        scalar during a timed trial (the count is stable for a fixed routing trace).
        """
        import torch

        for _ in range(count):
            handle = self.dispatch(problem)
            if not hasattr(problem, "recv_tokens"):
                problem.recv_tokens = self.recv_tokens(handle)
            self.stage(problem, handle)
            self.combine(problem, handle)
            torch.cuda.synchronize()

    def run_roundtrip(self, problem):
        """One full dispatch -> stage -> combine round trip; returns combined activations."""
        handle = self.dispatch(problem)
        self.stage(problem, handle)
        return self.combine(problem, handle)

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
        return time_us(torch, lambda p=problem: self.run_roundtrip(p), 0, iters)

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

        self.warm(problem, warmup)

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
