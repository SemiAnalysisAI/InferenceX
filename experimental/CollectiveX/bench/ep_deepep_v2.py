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


def _fp8_cast_helpers():
    """The pinned per-token FP8 cast pair (blockwise e4m3fn, per-128-block FP32 scale).

    Imported lazily so a BF16-only run never depends on deep_ep.utils.math, and so
    the quantization the oracle models is byte-identical to what dispatch transports.
    """
    from deep_ep.utils.math import per_token_cast_to_fp8, per_token_cast_back
    return per_token_cast_to_fp8, per_token_cast_back


def _jit_cache_directory(
    args,
    world_size: int,
    max_tokens: int,
    allow_hybrid_mode: bool,
    realized: dict[str, int | bool],
    use_fp8: bool,
) -> str:
    values = (
        args.runner, world_size, args.hidden, args.topk, args.experts,
        getattr(args, "num_logical_experts", args.experts), max_tokens,
        int(allow_hybrid_mode), realized["allocated_qps"], realized["num_sms"],
        int(use_fp8),
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
    # Two kernel families under one adapter, selected by mode:
    #   normal      -> PR #605 ElasticBuffer (LSA vs hybrid GIN are transport paths, not
    #                  kernel families); rank-deduplicated unweighted-rank-sum combine.
    #   low-latency -> the legacy deep_ep.Buffer IBGDA decode kernels
    #                  (low_latency_dispatch/combine); per-expert padded layout with a
    #                  source-side weighted-kernel-sum combine. kernel_generation and the
    #                  combine semantics are switched to their LL values in __init__.
    kernel_generation = "v2-elastic-buffer"
    SUPPORTED_MODES = ("normal", "low-latency")
    SUPPORTED_PRECISIONS = ("bf16", "fp8")
    stage_device_work = False
    combine_needs_redispatch = False
    combine_weight_semantics = "unweighted-rank-sum"

    def __init__(self, args, rank, world_size, local_rank, device):
        # Mode picks the kernel family (normal ElasticBuffer vs low-latency legacy
        # Buffer); base SUPPORTED_MODES enforces the allowed set.
        super().__init__(args, rank, world_size, local_rank, device)
        self.group = dist.group.WORLD
        self._fp8 = self.precision == "fp8"
        # FP8 dispatch dequantizes the received (e4m3fn, per-128-block FP32 scale) payload
        # back to the BF16 combine sends, which is real device work and so a separately-
        # timed component. (Normal mode prequantizes on the host; low-latency casts inside
        # the dispatch kernel — either way the dequant lands in stage().)
        self.stage_device_work = self._fp8
        self._to_fp8 = self._cast_back = None
        if self._fp8:
            self.dispatch_dtype = "fp8-e4m3fn"
            self.dispatch_value_bytes = 1
            self.dispatch_scale_bytes_per_copy = ((args.hidden + 127) // 128) * 4
            # Resolve the pinned cast pair once (lazily, so a BF16-only run never imports
            # deep_ep.utils.math) so the timed stage() does no module lookup in the
            # measured region.
            self._to_fp8, self._cast_back = _fp8_cast_helpers()
        if self.mode == "low-latency":
            # Legacy Buffer IBGDA decode path: a distinct kernel family whose combine
            # multiplies by the gate at the source (weighted), not an unweighted rank sum.
            self.kernel_generation = "legacy-buffer-ll"
            self.combine_weight_semantics = "weighted-kernel-sum"
            # LL result tensors are double-buffered and single-use per dispatch (upstream:
            # "you cannot hold more than 2 low-latency kernels' result tensors at a single
            # moment"), so every timed combine needs a fresh dispatch and every timed
            # dispatch must be drained by its combine.
            self.combine_needs_redispatch = True
            self.dispatch_needs_combine_cleanup = True

    def buffer_cap(self, args):
        if self.mode == "low-latency":
            # LL pre-allocates a fixed [num_local_experts, cap * num_ranks, hidden] receive
            # buffer, so cap is a hard per-rank dispatch-slot bound (the harness clamps the
            # decode ladder to it and reports the dropped point). 256 sits well under the
            # default NVSHMEM_QP_DEPTH ceiling ((cap + 1) * 2 <= 1024 => cap <= 511 with
            # NVSHMEM_QP_DEPTH=1024) and is adjustable if the decode ladder needs more.
            return 256
        return None

    def create_buffer(self, spec):
        # max_tokens is the measured-ladder maximum; the historical values (which
        # also folded in the conditioning ramp) are identical because the ramp
        # never exceeded the measured maximum, so the JIT directory stays stable.
        args, world_size = self.args, self.world_size
        self.max_tokens = spec.max_tokens_per_rank
        if self.mode == "low-latency":
            self._create_ll_buffer(spec)
            return
        _require_runtime()
        jit_root = Path(os.environ["EP_JIT_CACHE_DIR"])
        allow_hybrid_mode = _configure_gin_mode(args, world_size)
        self.buffer = ElasticBuffer(
            self.group,
            num_max_tokens_per_rank=self.max_tokens,
            hidden=args.hidden,
            num_topk=args.topk,
            use_fp8_dispatch=self._fp8,  # FP8 sizes the buffer for the (e4m3fn, scale) tuple.
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
            self._fp8,
        )
        os.environ["EP_JIT_CACHE_DIR"] = str(jit_root / jit_cache_directory)

    def _create_ll_buffer(self, spec):
        """Construct the legacy low-latency deep_ep.Buffer (IBGDA decode kernels).

        Distinct from the ElasticBuffer path: LL always allocates the NVSHMEM RDMA
        buffer and forces IBGDA even for single-node EP8, so there is no NVLink-only
        fallback. `allow_nvlink_for_low_latency_mode` lets NVLink carry intranode
        traffic alongside IBGDA; it does not remove the IBGDA requirement.
        """
        args, world_size = self.args, self.world_size
        assert args.experts % world_size == 0, (
            "low-latency EP requires num_experts divisible by the EP size"
        )
        self.num_local_experts = args.experts // world_size
        # LL requires the QP-per-rank count to equal the number of local experts.
        num_qps_per_rank = self.num_local_experts
        assert num_qps_per_rank == self.num_local_experts
        if not hasattr(deep_ep.Buffer, "low_latency_dispatch"):
            raise RuntimeError(
                "invalid DeepEP LL runtime: deep_ep.Buffer.low_latency_dispatch is absent"
            )
        # Verified pinned signatures (commit fa8a9b16, deep_ep/buffers/legacy.py):
        #   Buffer.get_low_latency_rdma_size_hint(num_max_dispatch_tokens_per_rank,
        #       hidden, num_ranks, num_experts) -> int   (staticmethod, line 175)
        #   Buffer(group, num_nvl_bytes=0, num_rdma_bytes=0, low_latency_mode=False,
        #       num_qps_per_rank=24, allow_nvlink_for_low_latency_mode=True, ...)  (line 33)
        num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
            self.max_tokens, args.hidden, world_size, args.experts
        )
        self.buffer = deep_ep.Buffer(
            self.group,
            num_rdma_bytes=num_rdma_bytes,
            low_latency_mode=True,
            num_qps_per_rank=num_qps_per_rank,
            allow_nvlink_for_low_latency_mode=True,
            explicitly_destroy=True,
        )

    def _ll_recv_bf16(self, recv_x):
        """The padded per-expert receive as BF16 `[num_local_experts, cap*num_ranks, hidden]`.

        BF16 dispatch already returns that tensor; FP8 dispatch returns an (e4m3fn, per-128-
        block FP32 scale) tuple, dequantized here with the pinned cast-back (matches the
        upstream test's `recv_x[0].view(-1, hidden)` / `recv_x[1].view(-1, hidden//128)`).
        """
        if not self._fp8:
            return recv_x
        fp8, scales = recv_x
        num_experts, num_slots, hidden = fp8.shape
        bf16 = self._cast_back(fp8.view(-1, hidden), scales.view(-1, hidden // 128))
        return bf16.view(num_experts, num_slots, hidden)

    def _topk_idx_dtype(self):
        # DeepEP V2's kernels key routing indices on deep_ep.topk_idx_t, not int64.
        return deep_ep.topk_idx_t

    def semantic_payload(self, x):
        if not self._fp8:
            return x
        return self._cast_back(*self._to_fp8(x))

    def _encode_dispatch(self, x):
        if not self._fp8:
            return x, None
        if self.mode == "low-latency":
            # low_latency_dispatch takes BF16 x and casts to e4m3fn inside the kernel, so
            # send x unquantized; expose the host round-trip as the oracle semantic so the
            # combine expectation models the FP8 transport (same as semantic_payload).
            return x, self._cast_back(*self._to_fp8(x))
        quantized = self._to_fp8(x)
        return quantized, self._cast_back(*quantized)

    def _ll_dispatch(self, p):
        # Verified pinned signature (legacy.py:553):
        #   low_latency_dispatch(x[bf16, num_tokens, hidden], topk_idx,
        #       num_max_dispatch_tokens_per_rank, num_experts, use_fp8=True, ...)
        #   -> (recv_x | (fp8, scales), recv_count[num_local_experts], handle, event, hook)
        # Defaults async_finish=False / return_recv_hook=False => the kernel ensures the
        # data has arrived, so the hook/event are inert and unused here.
        recv_x, recv_count, ll_handle, _event, _hook = self.buffer.low_latency_dispatch(
            p.dispatch_x,
            p.topk_idx,
            self.max_tokens,
            self.args.experts,
            use_fp8=self._fp8,
        )
        return types.SimpleNamespace(
            recv_x=recv_x,
            recv_count=recv_count,
            ll_handle=ll_handle,
        )

    def dispatch(self, p):
        if self.mode == "low-latency":
            return self._ll_dispatch(p)
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
        if self.mode == "low-latency":
            # The timed combine sends the padded per-expert receive back as BF16 (dequant
            # under FP8). Value correctness is exercised by the oracle's combine_transformed
            # path; this only has to move the right shape for timing.
            h.combine_input = self._ll_recv_bf16(h.recv_x)
            return
        if self._fp8:
            # Dequantize the received (fp8, scale) tuple to the BF16 combine sends.
            h.combine_input = self._cast_back(h.recv_x[0], h.recv_x[1])
        else:
            # BF16: the received buffer is already the semantic payload to combine.
            h.combine_input = h.recv_x

    def combine(self, p, h):
        if self.mode == "low-latency":
            # Verified pinned signature (legacy.py:624):
            #   low_latency_combine(x[bf16, num_local_experts, cap*num_ranks, hidden],
            #       topk_idx, topk_weights, handle, ...) -> (combined_x[num_combined, hidden],
            #       event, hook). The kernel applies topk_weights internally (weighted).
            combined_x, _event, _hook = self.buffer.low_latency_combine(
                h.combine_input, p.topk_idx, p.topk_weights, h.ll_handle
            )
            return combined_x[: p.T]
        combined_x, _, _ = self.buffer.combine(
            h.combine_input,
            handle=h.handle,
            num_sms=self.num_sms,
            num_qps=self.num_qps,
            async_with_compute_stream=False,
        )
        return combined_x

    def _ll_inspect_dispatch(self, p, h):
        """Flat per-slot view over the padded per-expert LL receive.

        LL delivers `[num_local_experts, cap*num_ranks, hidden]` with each expert's valid
        tokens packed at the front `[0:recv_count[e]]` of its slot dimension. Flatten to the
        oracle's compact contract in `(expert, slot)` row-major order — for e: for j in
        range(recv_count[e]) — keeping the (expert, slot) coordinates on the handle so
        combine_transformed can scatter the transformed rows back 1:1.
        """
        recv_bf16 = self._ll_recv_bf16(h.recv_x)  # [E, S, hidden] BF16
        num_slots = recv_bf16.shape[1]
        counts = h.recv_count.to(torch.int64)  # [E]
        # Front-packed mask: slot j is valid for expert e iff j < counts[e]. nonzero yields
        # C-order indices, i.e. (e ascending, then j ascending) — the required slot order.
        slot_valid = (
            torch.arange(num_slots, device=recv_bf16.device).unsqueeze(0) < counts.unsqueeze(1)
        )
        slot_expert, slot_j = slot_valid.nonzero(as_tuple=True)
        h.slot_expert = slot_expert  # local expert index per slot (for the combine scatter)
        h.slot_j = slot_j
        local_lo = self.rank * self.num_local_experts
        return types.SimpleNamespace(
            payload=recv_bf16[slot_expert, slot_j],
            expert_ids=local_lo + slot_expert.to(torch.int64),
            local_expert_counts=counts,
        )

    def inspect_dispatch(self, p, h):
        if self.mode == "low-latency":
            return self._ll_inspect_dispatch(p, h)
        count = self.recv_tokens(h)
        local_idx = h.recv_topk_idx[:count]
        valid = local_idx >= 0
        expert_ids = torch.where(
            valid,
            local_idx + self.rank * (self.args.experts // self.world_size),
            local_idx,
        )
        local = local_idx[valid].to(torch.int64)
        if self._fp8:
            # Dequantize the sliced (fp8, scale) recv tuple so the payload the oracle
            # inspects is BF16 [count, hidden] (source-ID decode + bit-exact compare).
            payload = self._cast_back(h.recv_x[0][:count], h.recv_x[1][:count])
        else:
            payload = h.recv_x[:count]
        return types.SimpleNamespace(
            payload=payload,
            expert_ids=expert_ids,
            weights=h.recv_topk_weights[:count].masked_fill(~valid, 0),
            local_expert_counts=torch.bincount(
                local, minlength=self.args.experts // self.world_size
            ),
        )

    def _ll_combine_transformed(self, p, h, transformed):
        """Scatter the oracle-transformed rows back into a zeroed padded combine buffer at
        the exact `(expert, slot)` coordinates inspect_dispatch read them from, then run the
        weighted LL combine. `transformed` is `[N, hidden]` in that same slot order; the
        kernel applies p.topk_weights internally, so the staged transform is unweighted."""
        if self._fp8:
            fp8 = h.recv_x[0]
            combine_buf = torch.zeros(
                fp8.shape, dtype=torch.bfloat16, device=fp8.device
            )
        else:
            combine_buf = torch.zeros_like(h.recv_x)
        combine_buf[h.slot_expert, h.slot_j] = transformed.to(combine_buf.dtype)
        combined_x, _event, _hook = self.buffer.low_latency_combine(
            combine_buf, p.topk_idx, p.topk_weights, h.ll_handle
        )
        return combined_x[: p.T]

    def combine_transformed(self, p, h, transformed):
        if self.mode == "low-latency":
            return self._ll_combine_transformed(p, h, transformed)
        # Combine always sends BF16. Under FP8, recv_x is an (fp8, scale) tuple, so the
        # BF16 combine buffer is shaped from the fp8 payload rather than zeros_like it.
        if self._fp8:
            combine_input = torch.zeros_like(h.recv_x[0], dtype=torch.bfloat16)
        else:
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
        if self.mode == "low-latency":
            return int(h.recv_count.sum().item())
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
