#!/usr/bin/env python3
"""DeepEP PR #605 adapter with the exact upstream PR #630 and #640 fixes."""

from __future__ import annotations

import inspect
import os
import sys
import types
from pathlib import Path

import torch
import torch.distributed as dist
from ep_backend import EPBackend

try:
    import deep_ep
    from deep_ep import ElasticBuffer  # type: ignore
except Exception as exc:  # pragma: no cover - requires the benchmark image
    print(f"ERROR: DeepEP V2 import failed: {exc!r}", file=sys.stderr)
    raise


# Source pins (PR #605 head + #630/#640 fixes) live in runtime/common.sh;
# the launcher fetches and builds them from that checkout. This adapter no longer
# verifies the wheel's commit tag against the pin — it checks only that the loaded
# deep_ep exposes ElasticBuffer (the from-source PR #605 capability).


def _jit_cache_directory(
    args,
    world_size: int,
    max_tokens: int,
    allow_hybrid_mode: bool,
    realized: dict[str, int | bool],
) -> str:
    values = (
        args.runner, world_size, args.hidden, args.topk, args.experts,
        getattr(args, "num_logical_experts", args.experts), max_tokens,
        int(allow_hybrid_mode), realized["allocated_qps"], realized["num_sms"],
    )
    return "jit-" + "-".join(str(value) for value in values)


# GIN/GDAKI allocates num_allocated_qps device QPs per peer rank on the local NIC
# (contexts x world_size QPs, before NCCL's own connection QPs). Upstream's hybrid
# default (129, or 65 with fast RDMA atomics) exhausts the per-NIC QP budget at
# EP16: construction dies in ncclDevCommCreate with ibv_create_qp ENOMEM once
# NCCL's regular QPs land on top (identical on H200 bare-metal and B200 pods; on
# CX-7 the budget sits between 784 and 1040 QPs — 49x16 initializes, 65x16 does
# not). Spending a fixed ~512-QP budget keeps every EP size inside that limit
# with headroom: EP8 resolves to 65 (the allocation CX-8 racks already run
# successfully), EP16 to 33 and EP32 to 17 (33 and 49 verified on the failing
# H200 pair). An explicit value also skips upstream's rank-local ibstat probe,
# which is not guaranteed to resolve identically across ranks.
_GIN_QP_BUDGET = 512


def _hybrid_num_allocated_qps(world_size: int) -> int:
    return max(9, 1 + _GIN_QP_BUDGET // world_size)


def _configure_gin_mode(args, world_size: int) -> bool:
    scale_up_domain = int(args.scale_up_domain)
    allow_hybrid_mode = world_size > scale_up_domain
    if allow_hybrid_mode:
        os.environ.pop("EP_DISABLE_GIN", None)
    else:
        os.environ["EP_DISABLE_GIN"] = "1"
    return allow_hybrid_mode


def _require_runtime() -> None:
    """Capability check only: the loaded deep_ep must expose ElasticBuffer (still
    catches the b300 image-bundled deep_ep 1.2.1 shadowing the from-source build,
    which lacks the class)."""
    if not inspect.isclass(ElasticBuffer) or ElasticBuffer.__name__ != "ElasticBuffer":
        raise RuntimeError("invalid DeepEP V2 runtime: deep_ep.ElasticBuffer is absent")


class DeepEPV2Backend(EPBackend):
    name = "deepep-v2"
    # Invariant by identity contract: this backend IS the PR #605 ElasticBuffer
    # implementation; LSA vs hybrid GIN are transport paths, not kernel families.
    kernel_generation = "v2-elastic-buffer"
    stage_device_work = False
    combine_needs_redispatch = False
    combine_weight_semantics = "unweighted-rank-sum"

    def __init__(self, args, rank, world_size, local_rank, device):
        # deepep-v2 is normal-mode only; base SUPPORTED_MODES=("normal",) enforces it.
        super().__init__(args, rank, world_size, local_rank, device)
        self.group = dist.group.WORLD

    def create_buffer(self, spec):
        # max_tokens is the measured-ladder maximum; the historical values (which
        # also folded in the conditioning ramp) are identical because the ramp
        # never exceeded the measured maximum, so the JIT directory stays stable.
        args, world_size = self.args, self.world_size
        self.max_tokens = spec.max_tokens_per_rank
        _require_runtime()
        jit_root = Path(os.environ["EP_JIT_CACHE_DIR"])
        allow_hybrid_mode = _configure_gin_mode(args, world_size)
        self.buffer = ElasticBuffer(
            self.group,
            num_max_tokens_per_rank=self.max_tokens,
            hidden=args.hidden,
            num_topk=args.topk,
            use_fp8_dispatch=False,  # BF16 communication path.
            deterministic=False,
            allow_hybrid_mode=allow_hybrid_mode,
            allow_multiple_reduction=True,
            prefer_overlap_with_compute=True,
            num_gpu_timeout_secs=100,
            explicitly_destroy=True,
            # 0 is upstream's use-the-default sentinel; only hybrid (GIN) mode
            # needs the explicit budget-derived allocation.
            num_allocated_qps=(
                _hybrid_num_allocated_qps(world_size) if allow_hybrid_mode else 0
            ),
        )
        tuning_num_experts = int(getattr(args, "num_logical_experts", args.experts))
        self.num_sms = int(
            self.buffer.get_theoretical_num_sms(tuning_num_experts, args.topk)
        )
        self.num_qps = int(self.buffer.get_theoretical_num_qps(self.num_sms))
        realized = {
            "num_sms": self.num_sms,
            "allocated_qps": int(self.buffer.num_allocated_qps),
        }
        jit_cache_directory = _jit_cache_directory(
            args,
            world_size,
            self.max_tokens,
            allow_hybrid_mode,
            realized,
        )
        os.environ["EP_JIT_CACHE_DIR"] = str(jit_root / jit_cache_directory)

    def _topk_idx_dtype(self):
        # DeepEP V2's kernels key routing indices on deep_ep.topk_idx_t, not int64.
        return deep_ep.topk_idx_t

    def dispatch(self, p):
        recv_x, recv_topk_idx, recv_topk_weights, handle, _ = self.buffer.dispatch(
            p.dispatch_x,
            topk_idx=p.topk_idx,
            topk_weights=p.topk_weights,
            num_experts=self.args.experts,
            num_max_tokens_per_rank=self.max_tokens,
            expert_alignment=1,
            num_sms=self.num_sms,
            num_qps=self.num_qps,
            async_with_compute_stream=False,
            do_handle_copy=True,
            do_cpu_sync=True,
            do_expand=False,
        )
        return types.SimpleNamespace(
            recv_x=recv_x,
            recv_topk_idx=recv_topk_idx,
            recv_topk_weights=recv_topk_weights,
            handle=handle,
        )

    def stage(self, p, h):
        # BF16: the received buffer is already the semantic payload to combine.
        h.combine_input = h.recv_x

    def combine(self, p, h):
        combined_x, _, _ = self.buffer.combine(
            h.combine_input,
            handle=h.handle,
            num_sms=self.num_sms,
            num_qps=self.num_qps,
            async_with_compute_stream=False,
        )
        return combined_x

    def inspect_dispatch(self, p, h):
        count = self.recv_tokens(h)
        local_idx = h.recv_topk_idx[:count]
        valid = local_idx >= 0
        expert_ids = torch.where(
            valid,
            local_idx + self.rank * (self.args.experts // self.world_size),
            local_idx,
        )
        local = local_idx[valid].to(torch.int64)
        return types.SimpleNamespace(
            payload=h.recv_x[:count],
            expert_ids=expert_ids,
            weights=h.recv_topk_weights[:count].masked_fill(~valid, 0),
            local_expert_counts=torch.bincount(
                local, minlength=self.args.experts // self.world_size
            ),
        )

    def combine_transformed(self, p, h, transformed):
        combine_input = torch.zeros_like(h.recv_x)
        combine_input[: transformed.shape[0]].copy_(transformed.to(combine_input.dtype))
        combined, _, _ = self.buffer.combine(
            combine_input,
            handle=h.handle,
            num_sms=self.num_sms,
            num_qps=self.num_qps,
            async_with_compute_stream=False,
        )
        return combined

    def recv_tokens(self, h):
        return int(h.handle.psum_num_recv_tokens_per_scaleup_rank[-1].item())

    def finalize(self, rc):
        try:
            dist.barrier()
            self.buffer.destroy()
            dist.barrier()
            dist.destroy_process_group()
        except Exception:
            return 1
        return rc
