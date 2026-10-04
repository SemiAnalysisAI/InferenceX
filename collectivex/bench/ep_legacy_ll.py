"""The DeepEP-legacy `Buffer` low-latency decode path, shared by deepep-v2 and uccl-ep.

UCCL-EP's `Buffer` is API-identical to DeepEP's legacy one, so both adapters drive the same
low_latency_dispatch/low_latency_combine calls over a per-expert padded receive and differ only
in how they construct the Buffer (`_create_ll_buffer`) and in their normal mode.
"""
from __future__ import annotations

import types

import torch
import torch.distributed as dist


@torch.compile(dynamic=False)
def _ll_dequant_static(fp8, scales):
    """Static-shape FP32-accumulate dequant of the low-latency FP8 (e4m3, per-128-block FP32
    scale) receive buffer to BF16.

    deep_ep's ``per_token_cast_back`` is ``@torch.compile(dynamic=True)``, which emits a
    generic near-eager kernel (~3.2 ms measured on the fixed low-latency recv shape
    ``[num_local_experts, cap*num_ranks, hidden]`` = (32, 2048, 7168) at EP8). The low-latency
    padded shape is constant on every dispatch, so a static (``dynamic=False``) compile fuses
    to one FP32 pass (~0.5 ms, 6.3x, bit-identical to the dynamic kernel on valid slots). The
    dequant runs in every timed `stage` sample and once per other component's warm-up, so the
    call count is large enough that the dynamic kernel's per-call overhead overran the leg's
    wall-clock budget (all ranks SIGKILLed ~22 min in, no result); the static form brings FP8
    low-latency inside the budget BF16 already meets. Padding slots decode to NaN in both
    forms (FP8 padding bytes) — harmless, because combine is handle-indexed and never reads
    padding. Only the padded low-latency recv uses this; normal-mode and the oracle's
    source-payload cast keep the pinned ``per_token_cast_back``.
    """
    e, s, h = fp8.shape
    values = fp8.to(torch.float32).view(e, s, h // 128, 128)
    block_scales = scales.to(torch.float32).view(e, s, h // 128, 1)
    return (values * block_scales).to(torch.bfloat16).view(e, s, h)


class LegacyBufferLL:
    """Mixin (listed before EPBackend) for the legacy `Buffer` low-latency kernels. The adapter
    keeps the mode branches of its timed methods inline and calls these `_ll_*` pieces; it provides
    `_fp8`, `buffer` (from its `_create_ll_buffer`), `max_tokens` and `num_local_experts`, and calls
    `_enable_ll` from `__init__`."""

    def _enable_ll(self, kernel_generation):
        # A distinct kernel family whose combine multiplies by the gate at the source (weighted),
        # not an unweighted rank sum, over a per-expert padded receive.
        self.kernel_generation = kernel_generation
        self.receive_layout = "token-expert"
        self.combine_weight_semantics = "weighted-kernel-sum"
        # LL result tensors are double-buffered and single-use per dispatch (upstream: "you
        # cannot hold more than 2 low-latency kernels' result tensors at a single moment"), so
        # every timed combine needs a fresh dispatch and every timed dispatch must be drained by
        # its combine.
        self.requires_fresh_pair = True

    def _ll_recv_bf16(self, recv_x):
        """The padded per-expert receive as BF16 `[num_local_experts, cap*num_ranks, hidden]`.

        BF16 dispatch already returns that tensor; FP8 dispatch returns an (e4m3, per-128-block
        FP32 scale) tuple, dequantized here with `_ll_dequant_static`. The low-latency fp8 scales
        come back column-major in their last two dims (TMA compatibility), i.e. non-contiguous,
        so they are made contiguous before the per-block view — a plain `.view()` raises "view
        size is not compatible with ... stride" on the transposed layout (the fp8 tensor itself
        is row-major contiguous, so only the scales need it). Mirrors the upstream low-latency
        test, which calls `.contiguous()` on the scales before dequant.
        """
        if not self._fp8:
            return recv_x
        fp8, scales = recv_x
        return _ll_dequant_static(fp8, scales.contiguous())

    def _ll_dispatch(self, p):
        # Defaults async_finish=False / return_recv_hook=False => the kernel ensures the data has
        # arrived, so the hook/event are inert and unused here.
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

    def _ll_inspect_dispatch(self, p, h):
        return self._expert_major_view(h, self._ll_recv_bf16(h.recv_x), h.recv_count)

    def _ll_combine_transformed(self, p, h, transformed):
        """Scatter the oracle-transformed rows back into a zeroed padded combine buffer at the
        exact `(expert, slot)` coordinates inspect_dispatch read them from, then run the weighted
        LL combine. `transformed` is `[N, hidden]` in that same slot order; the kernel applies
        p.topk_weights internally, so the staged transform is unweighted."""
        if self._fp8:
            fp8 = h.recv_x[0]
            combine_buf = torch.zeros(fp8.shape, dtype=torch.bfloat16, device=fp8.device)
        else:
            combine_buf = torch.zeros_like(h.recv_x)
        combine_buf[h.slot_expert, h.slot_j] = transformed.to(combine_buf.dtype)
        combined_x, _event, _hook = self.buffer.low_latency_combine(
            combine_buf, p.topk_idx, p.topk_weights, h.ll_handle
        )
        return combined_x[: p.T]

    def finalize(self, rc):
        try:
            dist.barrier()
            self.buffer.destroy()
            dist.barrier()
            dist.destroy_process_group()
        except Exception:
            return 1
        return rc
