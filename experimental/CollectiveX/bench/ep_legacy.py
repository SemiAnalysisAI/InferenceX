"""Operations shared by DeepEP and its API-compatible UCCL legacy Buffer.

Vendor construction, normal-mode transport, and quantizers stay in their adapters. Shared
methods retain the same dispatch/stage/combine call depth inside the measured windows.
"""
from __future__ import annotations

import types

import torch
import torch.distributed as dist


@torch.compile(dynamic=False)
def _ll_dequant_static(fp8, scales):
    """Static-shape FP32-accumulate dequant of the low-latency FP8 (e4m3fn, per-128-block
    FP32 scale) receive buffer to BF16.

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


class LegacyBufferOperations:
    """Legacy Buffer low-latency operations and FP8 staging for both library adapters."""

    def _ll_recv_bf16(self, recv_x):
        """The padded per-expert receive as BF16 `[num_local_experts, cap*num_ranks, hidden]`.

        BF16 dispatch already returns that tensor; FP8 dispatch returns an (e4m3fn, per-128-
        block FP32 scale) tuple, dequantized here with the pinned cast-back. The low-latency
        fp8 scales come back column-major in their last two dims (TMA compatibility), i.e.
        non-contiguous, so they are made contiguous before the per-block view — a plain
        `.view()` raises "view size is not compatible with ... stride" on the transposed
        layout (the fp8 tensor itself is row-major contiguous, so only the scales need it).
        Mirrors the upstream low-latency test, which calls `.contiguous()` on the scales
        before dequant. The dequant itself uses a static-shape compile (_ll_dequant_static)
        rather than deep_ep's dynamic-shape per_token_cast_back — 6.3x faster on the fixed LL
        recv shape and bit-identical, which is what keeps the FP8 leg inside its wall-clock
        budget (the dynamic kernel overran it).
        """
        if not self._fp8:
            return recv_x
        fp8, scales = recv_x
        return _ll_dequant_static(fp8, scales.contiguous())

    def semantic_payload(self, x):
        if not self._fp8:
            return x
        # Same callable the wire uses, so oracle and sender cannot disagree by construction.
        return self._cast_back(*self._quant(x))

    def _validate_quantizer(self, x):
        # Low-latency keeps the eager quantize (fused_quantize returns it unchanged), so
        # _quant IS _to_fp8 there and there is nothing to cross-check.
        if self._fp8 and self.mode != "low-latency":
            self.assert_quantize_identity(self._to_fp8, self._quant, x)

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

    def finalize(self, rc):
        try:
            dist.barrier()
            # UCCL's Buffer.destroy also tears its CPU proxies down via destroy_uccl.
            self.buffer.destroy()
            dist.barrier()
            dist.destroy_process_group()
        except Exception:
            return 1
        return rc
