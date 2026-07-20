#!/usr/bin/env python3
"""UCCL-EP adapter: the drop-in DeepEP-legacy `Buffer` API over UCCL's CPU-proxy transport.

UCCL-EP (https://github.com/uccl-project/uccl) is an API-identical DeepEP replacement whose
CPU proxies issue GPUDirect RDMA over plain libibverbs (no NVSHMEM/IBGDA); scale-up is
single-node cudaIpc over NVLink/XGMI (never MNNVL). Its Python `Buffer` — installed via UCCL's
`deep_ep_wrapper`, so `import deep_ep` in the isolated UCCL venv resolves to UCCL, NOT DeepSeek's
DeepEP — manages the CPU proxy threads internally (spun up in `__init__`'s `initialize_uccl`,
torn down in `destroy()`'s `destroy_uccl`), so this adapter never calls those functions directly;
it just constructs the Buffer and calls `.destroy()`.

Both modes use the legacy `Buffer` surface (mirroring bench/ep_deepep_v2.py's legacy-Buffer LL
path, which is ~1:1 reusable here):
  normal      -> get_dispatch_layout + dispatch + combine; per-token multi-expert recv layout
                 (expanded per (token, local-expert) for the oracle, reduced back before combine);
                 activation-only unweighted rank-sum combine.
  low-latency -> low_latency_dispatch/low_latency_combine; per-expert padded recv, source-side
                 weighted-kernel-sum combine.

FP8 dispatch is caller-prequantized (blockwise e4m3fn, e4m3fnuz on gfx942), combine is BF16 — the
oracle applies the identical per-token cast round-trip via semantic_payload/oracle_x, so the tight
combine gate (COMBINE_REL_TOL = 8*2^-8) is preserved, not loosened.
"""
from __future__ import annotations

import sys
import types

import torch
import torch.distributed as dist

from ep_backend import EPBackend

try:
    # In the isolated UCCL venv `deep_ep` is UCCL's deep_ep_wrapper (a drop-in DeepEP API backed
    # by uccl.ep's CPU-proxy runtime), not DeepSeek's DeepEP. Buffer/Config come from it.
    import deep_ep  # noqa: F401  (UCCL deep_ep_wrapper)
    from deep_ep import Buffer, Config  # type: ignore
except Exception as exc:  # pragma: no cover - requires the benchmark image
    print(f"ERROR: UCCL-EP import failed: {exc!r}", file=sys.stderr)
    raise


# ---- Vendored UCCL FP8 helpers (ep/bench/utils.py) --------------------------------------
# These live in UCCL's bench dir, not the installed package, so they are vendored VERBATIM so
# the quantization the oracle models is byte-identical to what dispatch transports. Keep in
# lockstep with upstream ep/bench/utils.py if the pinned UCCL commit moves.

def _fp8_e4m3_dtype() -> "torch.dtype":
    """UCCL's arch-keyed FP8 E4M3 dtype: e4m3fnuz on gfx942 (MI300X/MI325X), e4m3fn elsewhere."""
    if hasattr(torch.version, "hip") and torch.version.hip is not None:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        arch = getattr(props, "gcnArchName", "")
        if arch.startswith("gfx942"):
            return torch.float8_e4m3fnuz
    return torch.float8_e4m3fn


def per_token_cast_to_fp8(x: "torch.Tensor"):
    """Blockwise (per-128-channel) FP8 quantization: returns (e4m3 [m, n], scales [m, n//128])."""
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    fp8_dtype = _fp8_e4m3_dtype()
    fp8_max = 240.0 if fp8_dtype == torch.float8_e4m3fnuz else 448.0
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_view * (fp8_max / x_amax.unsqueeze(2))).to(fp8_dtype).view(m, n), (
        x_amax / fp8_max
    ).view(m, -1)


def per_token_cast_back(x_fp8: "torch.Tensor", x_scales: "torch.Tensor"):
    """Blockwise FP8 -> BF16 dequant mirroring per_token_cast_to_fp8."""
    if x_scales.dtype == torch.int:
        x_scales = x_scales.view(dtype=torch.uint8).to(torch.int) << 23
        x_scales = x_scales.view(dtype=torch.float)
    x_fp32 = x_fp8.to(torch.float32).view(x_fp8.size(0), -1, 128)
    x_scales = x_scales.view(x_fp8.size(0), -1, 1)
    return (x_fp32 * x_scales).view(x_fp8.shape).to(torch.bfloat16)


@torch.compile(dynamic=False)
def _ll_dequant_static(fp8, scales):
    """Static-shape FP32-accumulate dequant of the padded low-latency FP8 recv to BF16.

    Mirror of ep_deepep_v2._ll_dequant_static: the low-latency padded recv shape
    ``[num_local_experts, cap*num_ranks, hidden]`` is constant per dispatch, so a static
    (dynamic=False) compile fuses to one FP32 pass and stays inside the wall-clock budget that
    deep_ep's dynamic-shape per_token_cast_back would overrun. Padding slots decode to NaN
    (FP8 padding bytes) — harmless because combine is handle-indexed and never reads padding.
    """
    e, s, h = fp8.shape
    values = fp8.to(torch.float32).view(e, s, h // 128, 128)
    block_scales = scales.to(torch.float32).view(e, s, h // 128, 1)
    return (values * block_scales).to(torch.bfloat16).view(e, s, h)


# Normal-mode legacy Config launch parameters (DeepEP-legacy Config(num_sms, chunk, nvl_buffer)).
# Bring-up-tunable; these mirror UCCL's own intranode bench (nvl_buffer_size=256) and the DeepEP
# normal default SM budget. num_nvl_bytes is a generous fixed reservation as in UCCL's bench.
_NORMAL_NUM_SMS = 24
_NORMAL_NVL_BUFFER_SIZE = 256
_NORMAL_NVL_BYTES = int(2e9)
# Internode (EP16) buffer-sizing Config, straight from UCCL's own test_internode bench
# compute_buffer_sizes (nvl_chunk=8/512, rdma_chunk=16/512). Used only to size the NVLink+RDMA
# staging; dispatch/combine themselves run on the per-world-size recommended configs.
_INTERNODE_SIZE_CFG = (8, 512, 16, 512)


def _align_buffer_bytes(size, margin=1.2, alignment=128):
    """Safety margin + alignment for a buffer-size hint (mirrors the UCCL bench helper)."""
    return ((int(size * margin) + alignment - 1) // alignment) * alignment


class UCCLEPBackend(EPBackend):
    name = "uccl-ep"
    # One legacy Buffer under two modes, selected by args.mode:
    #   normal      -> get_dispatch_layout/dispatch/combine; unweighted rank-sum combine.
    #   low-latency -> low_latency_dispatch/combine decode kernels; source-side weighted combine.
    kernel_generation = "uccl-legacy-buffer"
    SUPPORTED_MODES = ("normal", "low-latency")
    SUPPORTED_PRECISIONS = ("bf16", "fp8")
    stage_device_work = False
    combine_needs_redispatch = False
    dispatch_needs_combine_cleanup = False
    combine_weight_semantics = "unweighted-rank-sum"

    def __init__(self, args, rank, world_size, local_rank, device):
        super().__init__(args, rank, world_size, local_rank, device)
        self.group = dist.group.WORLD
        self.experts_per_rank = args.experts // world_size
        self._internode = world_size > int(args.scale_up_domain)
        self._fp8 = self.precision == "fp8"
        # FP8 dispatch dequantizes the received (e4m3, per-128-block scale) payload back to the
        # BF16 combine sends — real device work, hence a separately-timed stage component.
        self.stage_device_work = self._fp8
        self._fp8_dtype = None
        if self._fp8:
            self._fp8_dtype = _fp8_e4m3_dtype()
            self.dispatch_dtype = (
                "fp8-e4m3fnuz" if self._fp8_dtype == torch.float8_e4m3fnuz else "fp8-e4m3fn"
            )
            self.dispatch_value_bytes = 1
            self.dispatch_scale_bytes_per_copy = ((args.hidden + 127) // 128) * 4
        if self.mode == "low-latency":
            # Legacy low-latency decode path: a distinct kernel family whose combine multiplies
            # by the gate at the source (weighted), not an unweighted rank sum. LL result tensors
            # are double-buffered and single-use per dispatch, so every timed combine needs a
            # fresh dispatch and every timed dispatch must be drained by its combine.
            self.kernel_generation = "uccl-legacy-buffer-ll"
            self.combine_weight_semantics = "weighted-kernel-sum"
            self.combine_needs_redispatch = True
            self.dispatch_needs_combine_cleanup = True

    def buffer_cap(self, args):
        if self.mode == "low-latency":
            # LL pre-allocates a fixed [num_local_experts, cap*num_ranks, hidden] receive buffer,
            # so cap is a hard per-rank dispatch-slot bound (same as ep_deepep_v2's legacy LL).
            return 256
        return None

    # ---- buffer construction ---------------------------------------------------------------

    def create_buffer(self, spec):
        self.max_tokens = spec.max_tokens_per_rank
        if self.mode == "low-latency":
            self._create_ll_buffer(spec)
            return
        args, world_size = self.args, self.world_size
        if self._internode:
            # Internode (EP16) scale-out: the RDMA combine kernel asserts
            # num_max_rdma_chunked_send_tokens >= num_warps_per_forwarder, which a hand-rolled
            # Config does NOT satisfy (its rdma-chunked-send default is 6). Mirror UCCL's own
            # internode bench: size NVLink+RDMA from a generous sizing Config, give each rank
            # num_sms QPs, and drive dispatch/combine with the per-world-size RECOMMENDED configs
            # (these set rdma-chunked-send to 20/12 for EP16 and satisfy the kernel constraints).
            num_sms = Buffer.num_sms
            self.dispatch_config = Buffer.get_dispatch_config(world_size)
            self.combine_config = Buffer.get_combine_config(world_size)
            self.config = self.combine_config
            hidden_bytes = args.hidden * 2
            size_config = Config(num_sms, *_INTERNODE_SIZE_CFG)
            num_nvl_bytes = _align_buffer_bytes(
                size_config.get_nvl_buffer_size_hint(hidden_bytes, world_size)
            )
            num_rdma_bytes = _align_buffer_bytes(
                size_config.get_rdma_buffer_size_hint(hidden_bytes, world_size)
            )
            self.buffer = Buffer(
                self.group,
                num_nvl_bytes,
                num_rdma_bytes,
                low_latency_mode=False,
                num_qps_per_rank=num_sms,
                allow_nvlink_for_low_latency_mode=True,
                allow_mnnvl=False,
                explicitly_destroy=True,
                is_intranode=False,
            )
            return
        # Intranode (EP8) scale-up: validated recipe — one fixed ~2 GB NVLink buffer, no RDMA, a
        # single QP, and the legacy 3-arg Config (rdma-chunked params unused with no RDMA path).
        self.config = Config(_NORMAL_NUM_SMS, 8, _NORMAL_NVL_BUFFER_SIZE)
        self.dispatch_config = self.config
        self.combine_config = self.config
        self.buffer = Buffer(
            self.group,
            _NORMAL_NVL_BYTES,
            0,
            low_latency_mode=False,
            num_qps_per_rank=1,
            allow_nvlink_for_low_latency_mode=True,
            allow_mnnvl=False,
            explicitly_destroy=True,
            is_intranode=True,
        )

    def _create_ll_buffer(self, spec):
        """Construct the legacy low-latency Buffer (decode kernels over the CPU-proxy transport).

        Distinct from normal mode: LL always allocates the RDMA staging buffer and forces the
        proxy path even for single-node EP8. Mirrors ep_deepep_v2._create_ll_buffer.
        """
        args, world_size = self.args, self.world_size
        assert args.experts % world_size == 0, (
            "low-latency EP requires num_experts divisible by the EP size"
        )
        self.num_local_experts = args.experts // world_size
        # LL requires the QP-per-rank count to equal the number of local experts.
        num_qps_per_rank = self.num_local_experts
        if not hasattr(Buffer, "low_latency_dispatch") and not hasattr(
            Buffer, "get_low_latency_rdma_size_hint"
        ):
            raise RuntimeError(
                "invalid UCCL-EP LL runtime: Buffer.low_latency_dispatch is absent"
            )
        num_rdma_bytes = Buffer.get_low_latency_rdma_size_hint(
            self.max_tokens, args.hidden, world_size, args.experts
        )
        self.buffer = Buffer(
            self.group,
            0,
            num_rdma_bytes,
            low_latency_mode=True,
            num_qps_per_rank=num_qps_per_rank,
            allow_nvlink_for_low_latency_mode=True,
            explicitly_destroy=True,
            is_intranode=not self._internode,
        )

    # ---- FP8 encode/dequant hooks ----------------------------------------------------------

    def _topk_idx_dtype(self):
        return torch.int64

    def semantic_payload(self, x):
        if not self._fp8:
            return x
        return per_token_cast_back(*per_token_cast_to_fp8(x))

    def _encode_dispatch(self, x):
        if not self._fp8:
            return x, None
        if self.mode == "low-latency":
            # low_latency_dispatch takes BF16 x and casts to e4m3 inside the kernel, so send x
            # unquantized; expose the host round-trip as the oracle semantic.
            return x, per_token_cast_back(*per_token_cast_to_fp8(x))
        fp8, scales = per_token_cast_to_fp8(x)
        # Column-major (TMA-compatible) scale layout the dispatch kernel expects, matching UCCL's
        # own bench (`scales.T.contiguous().T`) and the LL scale-contiguity note below.
        quantized = (fp8, scales.T.contiguous().T)
        return quantized, per_token_cast_back(fp8, scales)

    def _ll_recv_bf16(self, recv_x):
        """The padded per-expert receive as BF16 [num_local_experts, cap*num_ranks, hidden].

        BF16 dispatch already returns that tensor; FP8 returns an (e4m3, per-128-block scale)
        tuple, dequantized here with the static-shape compile (the LL fp8 scales come back
        column-major / non-contiguous for TMA, so they are made contiguous before the per-block
        view). Mirror of ep_deepep_v2._ll_recv_bf16.
        """
        if not self._fp8:
            return recv_x
        fp8, scales = recv_x
        return _ll_dequant_static(fp8, scales.contiguous())

    # ---- transport contract ----------------------------------------------------------------

    def _ll_dispatch(self, p):
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
        # Legacy normal dispatch: compute the layout, then scatter tokens to their experts.
        # num_tokens_per_rdma_rank is None intranode (EP8) and populated internode (EP16); pass
        # it through so the same call serves both scopes.
        (num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert,
         is_token_in_rank, _) = self.buffer.get_dispatch_layout(p.topk_idx, self.args.experts)
        recv_x, recv_topk_idx, recv_topk_weights, _counts, handle, _event = self.buffer.dispatch(
            x=p.dispatch_x,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            topk_idx=p.topk_idx,
            topk_weights=p.topk_weights,
            config=self.dispatch_config,
            async_finish=False,
        )
        return types.SimpleNamespace(
            recv_x=recv_x,
            recv_topk_idx=recv_topk_idx,
            recv_topk_weights=recv_topk_weights,
            handle=handle,
        )

    def stage(self, p, h):
        if self.mode == "low-latency":
            # The timed combine sends the padded per-expert receive back as BF16 (dequant under
            # FP8). Value correctness is exercised by the oracle's combine_transformed path.
            h.combine_input = self._ll_recv_bf16(h.recv_x)
            return
        if self._fp8:
            h.combine_input = per_token_cast_back(h.recv_x[0], h.recv_x[1])
        else:
            h.combine_input = h.recv_x

    def combine(self, p, h):
        if self.mode == "low-latency":
            combined_x, _event, _hook = self.buffer.low_latency_combine(
                h.combine_input, p.topk_idx, p.topk_weights, h.ll_handle
            )
            return combined_x[: p.T]
        # Normal combine is the activation-only unweighted rank-sum: topk_weights are intentionally
        # NOT passed so the kernel sums the per-token expert aggregates across ranks without
        # applying the gate (matches combine_weight_semantics and the two-level oracle).
        combined_x, _weights, _event = self.buffer.combine(
            x=h.combine_input,
            handle=h.handle,
            config=self.combine_config,
            async_finish=False,
        )
        return combined_x

    # ---- correctness-oracle views ----------------------------------------------------------

    def _ll_inspect_dispatch(self, p, h):
        """Flat per-slot view over the padded per-expert LL receive (mirror of
        ep_deepep_v2._ll_inspect_dispatch)."""
        recv_bf16 = self._ll_recv_bf16(h.recv_x)  # [E, S, hidden] BF16
        num_slots = recv_bf16.shape[1]
        counts = h.recv_count.to(torch.int64)  # [E]
        slot_valid = (
            torch.arange(num_slots, device=recv_bf16.device).unsqueeze(0) < counts.unsqueeze(1)
        )
        slot_expert, slot_j = slot_valid.nonzero(as_tuple=True)
        h.slot_expert = slot_expert
        h.slot_j = slot_j
        local_lo = self.rank * self.num_local_experts
        return types.SimpleNamespace(
            payload=recv_bf16[slot_expert, slot_j],
            expert_ids=local_lo + slot_expert.to(torch.int64),
            local_expert_counts=counts,
        )

    def _normal_recv_payload(self, h):
        """The received tokens as BF16 [num_recv_tokens, hidden] (dequant under FP8)."""
        if self._fp8:
            return per_token_cast_back(h.recv_x[0], h.recv_x[1])
        return h.recv_x

    def inspect_dispatch(self, p, h):
        if self.mode == "low-latency":
            return self._ll_inspect_dispatch(p, h)
        # Legacy normal recv: recv_x is [num_recv, hidden] (each received token once) and
        # recv_topk_idx/recv_topk_weights are [num_recv, topk] — the oracle's per-received-token
        # 2-D contract (it sorts each row over the topk axis and sums the per-expert transforms,
        # so token order is free and no per-(token,expert) expansion is needed). recv_topk_idx
        # holds LOCAL expert indices [0, experts_per_rank) with non-local masked to -1 (verified
        # against UCCL's own test_intranode: every entry is -1 or < epr), so rebase the valid
        # locals to the GLOBAL ids the oracle compares by rank*experts_per_rank.
        local_idx = h.recv_topk_idx.to(torch.int64)  # [num_recv, topk] local ids, -1 non-local
        valid = local_idx >= 0
        expert_ids = torch.where(
            valid, local_idx + self.rank * self.experts_per_rank, local_idx
        )
        return types.SimpleNamespace(
            payload=self._normal_recv_payload(h),  # [num_recv, hidden] BF16
            expert_ids=expert_ids,
            weights=h.recv_topk_weights.to(torch.float32),
            local_expert_counts=torch.bincount(
                local_idx[valid], minlength=self.experts_per_rank
            ),
        )

    def _ll_combine_transformed(self, p, h, transformed):
        """Scatter the oracle-transformed rows back into a zeroed padded combine buffer at the
        exact (expert, slot) coordinates inspect read them from, then run the weighted LL combine
        (mirror of ep_deepep_v2._ll_combine_transformed)."""
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

    def combine_transformed(self, p, h, transformed):
        if self.mode == "low-latency":
            return self._ll_combine_transformed(p, h, transformed)
        # `transformed` is the oracle's per-received-token combine input [num_recv, hidden]
        # (already summed over the top-k axis) — exactly the per-token buffer legacy combine
        # consumes; combine then sums those per-token aggregates across ranks (unweighted).
        combined, _weights, _event = self.buffer.combine(
            x=transformed.to(torch.bfloat16),
            handle=h.handle,
            config=self.combine_config,
            async_finish=False,
        )
        return combined

    def recv_tokens(self, h):
        if self.mode == "low-latency":
            return int(h.recv_count.sum().item())
        recv = h.recv_x[0] if self._fp8 else h.recv_x
        return int(recv.shape[0])

    def finalize(self, rc):
        try:
            dist.barrier()
            self.buffer.destroy()  # tears the CPU proxies down via destroy_uccl internally
            dist.barrier()
            dist.destroy_process_group()
        except Exception:
            return 1
        return rc
