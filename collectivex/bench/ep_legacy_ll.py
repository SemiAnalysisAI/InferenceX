"""The DeepEP-legacy `Buffer` low-latency decode path, shared by deepep-v2 and uccl-ep.

UCCL-EP's `Buffer` is API-identical to DeepEP's legacy one, so both adapters drive the same
low_latency_dispatch/low_latency_combine calls over a per-expert padded receive and differ only
in how they construct the Buffer (`_create_ll_buffer`) and in their normal mode.

The calls follow vLLM's DeepEPLLPrepareAndFinalize (prepare_finalize/deepep_ll.py): FP8 dispatch
rounds its scales to powers of two and packs them as UE8M0 exactly where vLLM does, and an adapter
that sets `ll_recv_hook` splits both collectives into send and receive kernels as vLLM's default
(non-DBO) path does.
"""
from __future__ import annotations

import types

import torch
import torch.distributed as dist

# The kernels' per-128-channel cast constants (DeepEP csrc/kernels/legacy/utils.cuh: kFP8Margin,
# kFinfoAmaxInvE4M3). The reciprocal is a float32 constant the kernel multiplies by; dividing by
# 448 instead can land one ulp away and flip the power-of-two rounding at an exact boundary.
_FP8_MARGIN = 1e-4
_E4M3_MAX_INV = 1.0 / 448.0


def _packed_ue8m0_scales(device) -> bool:
    """Whether vLLM dispatches low-latency FP8 with power-of-two scales packed as UE8M0.

    deepep_ll.py passes round_scale=use_ue8m0=True exactly when its expert kernel takes packed UE8M0
    activation scales (post_init_setup). Only BatchedDeepGemmExperts does, and only with DeepGEMM
    E8M0 in use (on by default) on a Blackwell-family GPU, CUDA capability 10.x or 12.x
    (supports_packed_ue8m0_act_scales). Hopper and ROCm keep plain FP32 scales.
    """
    if getattr(torch.version, "hip", None):
        return False
    return torch.cuda.get_device_capability(device)[0] in (10, 12)


def ue8m0_cast_to_fp8(x):
    """The low-latency kernels' round_scale cast, bit for bit (calculate_fp8_scales).

    Per 128-channel block: amax floored at 1e-4, times the float32 reciprocal of 448, rounded UP
    to a power of two by exponent arithmetic (fast_log2_ceil); the values are cast at exactly that
    power. Returns (e4m3fn [m, n], FP32 power-of-two scales [m, n/128]) -- the FP32 form the wire
    carries, which the receive side only repacks as UE8M0 exponent bytes.
    """
    m, n = x.shape
    blocks = x.float().view(m, n // 128, 128)
    amax = blocks.abs().amax(dim=2).clamp(min=_FP8_MARGIN)
    bits = (amax * torch.tensor(_E4M3_MAX_INV, dtype=torch.float32)).view(torch.int32)
    exponent = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    scale = ((127 - exponent) << 23).view(torch.float32)
    values = (blocks * scale.unsqueeze(2)).to(torch.float8_e4m3fn).view(m, n)
    return values, ((exponent + 127) << 23).view(torch.float32)


def _ue8m0_to_fp32(packed):
    """Packed UE8M0 receive scales (int32 [..., hidden/512], four exponent bytes each, column-major
    in the last two dims) as the FP32 powers of two they encode: byte b reads as 2^(b - 127)."""
    exponent = packed.contiguous().view(torch.uint8).to(torch.int32)
    return (exponent << 23).view(torch.float32)


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

    # vLLM's default (non-DBO) path issues each collective as a send kernel plus a receive hook
    # (deepep_ll.py: dispatch always passes return_recv_hook=True; finalize_async, the path the
    # modular kernel takes for this prepare/finalize, passes it to combine). With nothing to overlap
    # here the hook runs straight after the send, as it does when no shared expert is scheduled in
    # between. Opt-in per adapter, because the adapter's own combine must route through
    # `_ll_combine` for dispatch and combine to share the contract.
    ll_recv_hook = False
    _ll_ue8m0 = False

    def _enable_ll(self, kernel_generation):
        # A distinct kernel family whose combine multiplies by the gate at the source (weighted),
        # not an unweighted rank sum, over a per-expert padded receive.
        self.receive_layout = "token-expert"
        self.combine_weight_semantics = "weighted-kernel-sum"
        # LL result tensors are double-buffered and single-use per dispatch (upstream: "you
        # cannot hold more than 2 low-latency kernels' result tensors at a single moment"), so
        # every timed combine needs a fresh dispatch and every timed dispatch must be drained by
        # its combine.
        self.requires_fresh_pair = True
        self._ll_ue8m0 = bool(self._fp8) and _packed_ue8m0_scales(self.device)
        if self._ll_ue8m0:
            # The kernel now casts at power-of-two scales, so the oracle's payload model follows
            # it; the wire still carries one FP32 scale per 128-value block.
            self._enable_fp8(self.dispatch_dtype, ue8m0_cast_to_fp8, self._fp8_codec[1])
        # "-hook" marks the split send/receive kernels and "-ue8m0" the power-of-two FP8 cast.
        self.kernel_generation = (
            kernel_generation + ("-hook" if self.ll_recv_hook else "")
            + ("-ue8m0" if self._ll_ue8m0 else "")
        )

    def implementation_details(self):
        """Low-latency launch facts, shared by both adapters (see EPBackend.implementation_details)."""
        details = super().implementation_details()
        if self.mode == "low-latency":
            details.update({
                "receive_capacity": getattr(self, "max_tokens", None),
                "num_qps_per_rank": getattr(self, "num_local_experts", None),
                "return_recv_hook": bool(self.ll_recv_hook),
                "zero_copy": False,
                "fp8_scales": ("ue8m0-packed" if self._ll_ue8m0 else "fp32") if self._fp8 else None,
            })
        return details

    def _ll_recv_bf16(self, recv_x):
        """The padded per-expert receive as BF16 `[num_local_experts, cap*num_ranks, hidden]`.

        BF16 dispatch already returns that tensor; FP8 dispatch returns an (e4m3, per-128-block
        scale) tuple, dequantized here with `_ll_dequant_static`. The low-latency fp8 scales
        come back column-major in their last two dims (TMA compatibility), i.e. non-contiguous,
        so they are made contiguous before the per-block view — a plain `.view()` raises "view
        size is not compatible with ... stride" on the transposed layout (the fp8 tensor itself
        is row-major contiguous, so only the scales need it). Mirrors the upstream low-latency
        test, which calls `.contiguous()` on the scales before dequant. Packed UE8M0 scales are
        first read back as the FP32 powers of two they encode.
        """
        if not self._fp8:
            return recv_x
        fp8, scales = recv_x
        scales = _ue8m0_to_fp32(scales) if self._ll_ue8m0 else scales.contiguous()
        return _ll_dequant_static(fp8, scales)

    def _ll_dispatch(self, p):
        # async_finish=False: the compute stream waits on the kernels, so the event is inert. Without
        # the hook the one kernel ensures arrival; with it the receive kernel does, and both phases
        # launch on the compute stream (DeepEP's hook mode skips the comm-stream hand-off).
        recv_x, recv_count, ll_handle, _event, hook = self.buffer.low_latency_dispatch(
            p.dispatch_x,
            p.topk_idx,
            self.max_tokens,
            self.args.experts,
            use_fp8=self._fp8,
            round_scale=self._ll_ue8m0,
            use_ue8m0=self._ll_ue8m0,
            async_finish=False,
            return_recv_hook=self.ll_recv_hook,
        )
        if self.ll_recv_hook:
            hook()
        return types.SimpleNamespace(
            recv_x=recv_x,
            recv_count=recv_count,
            ll_handle=ll_handle,
        )

    def _ll_combine(self, p, h, x):
        """The weighted low-latency combine of `x`, the padded per-expert BF16 plane (the kernel
        applies p.topk_weights). vLLM passes zero_copy=False and async_finish=False as here; its
        `out=` only names the destination tensor."""
        combined_x, _event, hook = self.buffer.low_latency_combine(
            x, p.topk_idx, p.topk_weights, h.ll_handle,
            zero_copy=False, async_finish=False, return_recv_hook=self.ll_recv_hook,
        )
        if self.ll_recv_hook:
            hook()
        return combined_x[: p.T]

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
        return self._ll_combine(p, h, combine_buf)

    def finalize(self, rc):
        try:
            dist.barrier()
            self.buffer.destroy()
            dist.barrier()
            dist.destroy_process_group()
        except Exception:
            return 1
        return rc
