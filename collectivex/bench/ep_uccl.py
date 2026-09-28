#!/usr/bin/env python3
"""UCCL-EP adapter: the drop-in DeepEP-legacy `Buffer` API over UCCL's CPU-proxy RDMA.

Scale-out only: at the scoped single-node EP8 the adapter passes `is_intranode=True`, UCCL
never starts its proxies, and both modes move data over `cudaIpc`/NVLink instead.

UCCL-EP (https://github.com/uccl-project/uccl) is an API-identical DeepEP replacement whose
CPU proxies issue GPUDirect RDMA over plain libibverbs (no NVSHMEM/IBGDA); scale-up is
single-node cudaIpc over NVLink/XGMI (never MNNVL). Its Python `Buffer` — installed via UCCL's
`deep_ep_wrapper`, so `import deep_ep` in the isolated UCCL venv resolves to UCCL, NOT DeepSeek's
DeepEP — manages the CPU proxy threads internally (spun up in `__init__`'s `initialize_uccl`,
torn down in `destroy()`'s `destroy_uccl`), so this adapter never calls those functions directly;
it just constructs the Buffer and calls `.destroy()`.

Both modes use the legacy `Buffer` surface (low-latency shares bench/ep_legacy_ll.py with
deepep-v2's legacy-Buffer path):
  normal      -> get_dispatch_layout + dispatch + combine; per-token multi-expert recv layout;
                 activation-only unweighted rank-sum combine.
  low-latency -> low_latency_dispatch/low_latency_combine; per-expert padded recv, source-side
                 weighted-kernel-sum combine.

FP8 dispatch quantizes inside the timed dispatch in normal mode (blockwise e4m3fn, e4m3fnuz on
gfx942); in low-latency mode the caller sends BF16 and the decode kernel quantizes to e4m3
internally (``use_fp8``). Combine is always BF16 — the oracle applies the identical per-token cast round-trip
via semantic_payload/oracle_x in both modes, so the tight combine gate (COMBINE_REL_TOL = 8*2^-8)
is preserved, not loosened.
"""
from __future__ import annotations

import os
import sys
import types

import torch
import torch.distributed as dist

from ep_backend import EPBackend
from ep_legacy_ll import LegacyBufferLL

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


# Normal-mode legacy Config launch parameters (DeepEP-legacy Config(num_sms, chunk, nvl_buffer)).
# These mirror UCCL's own intranode bench (nvl_buffer_size=256); num_nvl_bytes is a generous fixed
# reservation as in that bench. The SM budget is vendor-keyed (see _normal_num_sms).
_NORMAL_NVL_BUFFER_SIZE = 256
_NORMAL_NVL_BYTES = int(2e9)
# Internode (EP16) buffer-sizing Config, straight from UCCL's own test_internode bench
# compute_buffer_sizes (nvl_chunk=8/512, rdma_chunk=16/512). Used only to size the NVLink+RDMA
# staging; dispatch/combine themselves run on the per-world-size recommended configs.
_INTERNODE_SIZE_CFG = (8, 512, 16, 512)


def _align_buffer_bytes(size, margin=1.2, alignment=128):
    """Safety margin + alignment for a buffer-size hint (mirrors the UCCL bench helper)."""
    return ((int(size * margin) + alignment - 1) // alignment) * alignment


def _normal_num_sms() -> int:
    """Intranode normal-mode SM budget, keyed by vendor exactly as UCCL's own bench does
    (ep/bench/test_intranode.py: ``num_sms = 24 if torch.version.cuda else 64``): 24 on CUDA,
    64 on HIP/ROCm. A flat 24 would materially understate AMD, whose wider CU count wants the
    larger grid — the same reason upstream branches on the vendor."""
    return 24 if torch.version.cuda else 64


class UCCLEPBackend(LegacyBufferLL, EPBackend):
    name = "uccl-ep"
    maturity = "candidate"  # no engine exposes a UCCL-EP all-to-all selector
    # One legacy Buffer under two modes, selected by args.mode:
    #   normal      -> get_dispatch_layout/dispatch/combine; unweighted rank-sum combine.
    #   low-latency -> low_latency_dispatch/combine decode kernels; source-side weighted combine.
    kernel_generation = "uccl-legacy-buffer"
    SUPPORTED_MODES = ("normal", "low-latency")
    SUPPORTED_PRECISIONS = ("bf16", "fp8")
    # LL's only host state is the double-buffer toggle; captures hold whole pairs, so it returns
    # to where it started. Normal mode host-syncs on its receive counters.
    CUDA_GRAPH_MODES = ("low-latency",)

    # (product, precision) cases measured faster eager than graphed: b200 FP8 low-latency pair
    # period 0.98x baseline eager vs 1.04x graphed, every rung.
    _EAGER_CASES = frozenset({("b200", "fp8")})

    @property
    def cuda_graph_supported(self) -> bool:
        if not super().cuda_graph_supported:
            return False
        args = self.args
        # Intranode only: scale-out LL runs through the CPU proxy, unvalidated under capture.
        if self.world_size > int(getattr(args, "scale_up_domain", self.world_size)):
            return False
        if os.environ.get("UCCL_RDMA_ADAPTIVE_SLEEP", "0") not in ("", "0"):
            return False
        product = str(getattr(args, "runner", "")).split("-")[0]
        return (product, self.precision) not in self._EAGER_CASES

    def __init__(self, args, rank, world_size, local_rank, device):
        super().__init__(args, rank, world_size, local_rank, device)
        self.group = dist.group.WORLD
        self.experts_per_rank = args.experts // world_size
        self._internode = world_size > int(args.scale_up_domain)
        self._fp8 = self.precision == "fp8"
        # FP8 dispatch dequantizes the received (e4m3, per-128-block scale) payload back to the
        # BF16 combine sends — real device work, hence a separately-timed stage component.
        self.stage_device_work = self._fp8
        if self._fp8:
            # Low-latency keeps the eager quantize, whose bits its in-kernel cast matches.
            fnuz = _fp8_e4m3_dtype() == torch.float8_e4m3fnuz
            self._enable_fp8(
                "fp8-e4m3fnuz" if fnuz else "fp8-e4m3fn", per_token_cast_to_fp8, per_token_cast_back
            )
        if self.mode == "low-latency":
            self._enable_ll("uccl-legacy-buffer-ll")

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
        # SM budget is vendor-keyed (24 CUDA / 64 HIP), matching UCCL's intranode bench.
        self.config = Config(_normal_num_sms(), 8, _NORMAL_NVL_BUFFER_SIZE)
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
        """Construct the legacy low-latency Buffer (the decode kernels).

        Distinct from normal mode only in allocating the RDMA staging buffer unconditionally.
        It does NOT force the proxy path: `is_intranode` is passed below, so at EP8 UCCL
        leaves the proxies stopped and the kernel takes its IPC branch.
        """
        args, world_size = self.args, self.world_size
        assert args.experts % world_size == 0, (
            "low-latency EP requires num_experts divisible by the EP size"
        )
        self.num_local_experts = args.experts // world_size
        # LL requires the QP-per-rank count to equal the number of local experts.
        num_qps_per_rank = self.num_local_experts
        if not hasattr(Buffer, "low_latency_dispatch") or not hasattr(
            Buffer, "get_low_latency_rdma_size_hint"
        ):
            raise RuntimeError(
                "invalid UCCL-EP LL runtime: Buffer.low_latency_dispatch / "
                "get_low_latency_rdma_size_hint absent"
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

    # ---- transport contract (the shared low-latency pieces live in ep_legacy_ll) ---------

    def dispatch(self, p):
        if self.mode == "low-latency":
            return self._ll_dispatch(p)
        # Legacy normal dispatch: compute the layout, then scatter tokens to their experts.
        # num_tokens_per_rdma_rank is None intranode (EP8) and populated internode (EP16); pass
        # it through so the same call serves both scopes.
        (num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert,
         is_token_in_rank, _) = self.buffer.get_dispatch_layout(p.topk_idx, self.args.experts)
        # Quantise here, not in make_problem: production runs one fused bf16->fp8 kernel per forward
        # pass right before this collective. The scales need UCCL's column-major (TMA-compatible)
        # layout, which production's kernel emits directly -- so timing the transpose over-states
        # by one small copy.
        dispatch_x = p.dispatch_x
        if self._fp8:
            fp8, scales = self._quant(dispatch_x)
            dispatch_x = (fp8, scales.T.contiguous().T)
        recv_x, recv_topk_idx, recv_topk_weights, _counts, handle, _event = self.buffer.dispatch(
            x=dispatch_x,
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

    def inspect_dispatch(self, p, h):
        if self.mode == "low-latency":
            return self._ll_inspect_dispatch(p, h)
        # Legacy normal recv: recv_x is [num_recv, hidden] (each received token once) and
        # recv_topk_idx/recv_topk_weights are [num_recv, topk] — the oracle's per-received-token
        # 2-D contract (it sorts each row over the topk axis and sums the per-expert transforms,
        # so token order is free and no per-(token,expert) expansion is needed). recv_topk_idx
        # holds LOCAL expert indices [0, experts_per_rank) with non-local masked to -1 (verified
        # against UCCL's own test_intranode: every entry is -1 or < epr).
        payload = per_token_cast_back(h.recv_x[0], h.recv_x[1]) if self._fp8 else h.recv_x
        return self._local_id_view(
            payload, h.recv_topk_idx, h.recv_topk_weights, self.experts_per_rank
        )

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
