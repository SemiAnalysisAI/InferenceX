#!/usr/bin/env python3
import argparse
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import matplotlib.pyplot as plt

from vllm.attention.ops.flashmla import (
    get_mla_metadata,
    flash_mla_with_kvcache,
    is_flashmla_dense_supported,
)

from vllm.utils import deep_gemm


# -----------------------------
# Timing helpers
# -----------------------------
@torch.inference_mode()
def time_cuda_ms(fn, warmup: int, iters: int) -> float:
    """Return average milliseconds per call for fn()."""
    assert iters > 0
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def ms_to_cost_per_1m_tokens(ms_per_tok: float, gpu_hour_price: float) -> float:
    # $/1M = (ms * 1e-3 sec) / 3600 * price * 1e6 = ms * price / 3.6
    return ms_per_tok * gpu_hour_price / 3.6


def round_up(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


# -----------------------------
# Cache builders (FAST init)
# -----------------------------
def _float32_to_u8_bytes(x: float, device: torch.device) -> torch.Tensor:
    t = torch.tensor([x], dtype=torch.float32, device=device)
    return t.view(torch.uint8)  # [4]


@torch.inference_mode()
def make_indexer_kv_cache_fp8_paged_fast(
    num_blocks_total: int,
    block_size: int,
    d: int,
    device: torch.device,
) -> torch.Tensor:
    """
    KV layout expected by fp8_paged_mqa_logits:
      [num_blocks_total, block_size, 1, d+4] uint8
      last 4 bytes per token are float32 scale bytes

    We fill FP8 payload with zeros (valid) and set scale=1.0.
    """
    kv = torch.zeros((num_blocks_total, block_size, 1, d + 4),
                     dtype=torch.uint8, device=device)
    scale_bytes = _float32_to_u8_bytes(1.0, device=device)  # [4]
    kv[..., d:] = scale_bytes
    return kv


@torch.inference_mode()
def make_flashmla_fp8_ds_mla_kvcache_fast(
    num_blocks_total: int,
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    FlashMLA sparse decoding KV cache packed bytes:
      656 bytes/token:
        512 bytes fp8 (NoPE)
        16 bytes scales (4 fp32)
        128 bytes rope (64 bf16)

    Fill with zeros and set scales=1.0.
    """
    BYTES_PER_TOKEN = 656
    kv = torch.zeros((num_blocks_total, block_size, 1, BYTES_PER_TOKEN),
                     dtype=torch.uint8, device=device)

    # 16 bytes scale field
    one = _float32_to_u8_bytes(1.0, device=device)   # [4]
    scales16 = one.repeat(4)                         # [16]
    kv[..., 512:528] = scales16

    return kv


# -----------------------------
# Benchmark config
# -----------------------------
@dataclass(frozen=True)
class BenchCfg:
    min_len: int
    max_len: int
    step: int
    gpu_hour_price: float
    warmup: int
    iters: int

    batch_size: int

    # Indexer (DeepSeek V3.2 config)
    index_heads: int
    index_head_dim: int
    topk: int
    next_n: int

    # FlashMLA dims
    mla_q_heads: int
    mla_kv_heads: int
    mla_d_qk: int
    mla_d_v: int
    block_size: int

    device: str


# -----------------------------
# FLOP / Byte models (per decoded token, per sequence)
# -----------------------------
def flops_indexer_per_token(L: int, H: int, D: int) -> int:
    # dot: 2*D, plus (weight mul + accumulate) ~2 per head per key
    return H * L * (2 * D + 2)


def flops_mla_per_token(n_ctx: int, h_q: int, d_qk: int, d_v: int) -> int:
    # QK dot: 2*d_qk, PV accumulate: 2*d_v
    return h_q * n_ctx * 2 * (d_qk + d_v)


def bytes_indexer_topk_per_token(L: int, topk: int, d_idx: int, logits_elem_size: int,
                                q_bytes: int, w_bytes: int) -> int:
    # KV read: L*(d_idx+4) bytes
    # logits: write + read
    # topk output indices: write topk*4
    kv_bytes = L * (d_idx + 4)
    logits_bytes = L * logits_elem_size
    topk_bytes = topk * 4
    return q_bytes + w_bytes + kv_bytes + 2 * logits_bytes + topk_bytes


def bytes_sparse_mla_per_token(topk: int, h_q: int, d_qk: int, d_v: int,
                              q_elem_size: int, out_elem_size: int) -> int:
    # Q read: h_q*d_qk
    # indices read: topk*4
    # KV read: topk*656 bytes
    # output write: h_q*d_v
    q_bytes = h_q * d_qk * q_elem_size
    idx_bytes = topk * 4
    kv_bytes = topk * 656
    out_bytes = h_q * d_v * out_elem_size
    return q_bytes + idx_bytes + kv_bytes + out_bytes


def bytes_dense_mla_per_token(L: int, h_q: int, d_qk: int, d_v: int,
                             q_elem_size: int, out_elem_size: int,
                             dense_supported: bool, baseline_topk: int) -> int:
    q_bytes = h_q * d_qk * q_elem_size
    kv_bytes = L * 656
    out_bytes = h_q * d_v * out_elem_size
    idx_bytes = 0 if dense_supported else baseline_topk * 4
    return q_bytes + kv_bytes + out_bytes + idx_bytes


def to_tflops(flops: float, ms: float) -> float:
    # TFLOP/s = flops / (ms*1e-3) / 1e12
    return flops / (ms * 1e-3) / 1e12


def to_tbps(bytes_: float, ms: float) -> float:
    # TB/s = bytes / (ms*1e-3) / 1e12
    return bytes_ / (ms * 1e-3) / 1e12


# -----------------------------
# Plotting
# -----------------------------
def plot_latency_cost(
    lengths: List[int],
    ms_indexer: List[float],
    ms_sparse: List[float],
    ms_total: List[float],
    ms_base: List[float],
    cost_indexer: List[float],
    cost_sparse: List[float],
    cost_total: List[float],
    cost_base: List[float],
    topk: int,
    gpu_name: str,
    output_path: str,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    xk = [L / 1000 for L in lengths]

    ax1 = axes[0]
    ax1.plot(xk, ms_indexer, 'o-', label='Indexer + Top-k (V3.2 DSA)', linewidth=2, markersize=5)
    ax1.plot(xk, ms_sparse,  's-', label='Sparse MLA Decode (V3.2 DSA)', linewidth=2, markersize=5)
    ax1.plot(xk, ms_total,   '^-', label='Total DSA Decode (V3.2)', linewidth=2, markersize=5)
    ax1.plot(xk, ms_base,    'x--', label='Baseline MLA Decode (V3)', linewidth=2, markersize=6)
    ax1.set_xlabel('Context Length (K tokens)')
    ax1.set_ylabel('Latency per token (ms)')
    ax1.set_title(f'Decode Attention Latency (per token)\n{gpu_name}')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)

    ax2 = axes[1]
    ax2.plot(xk, cost_indexer, 'o-', label='Indexer + Top-k (V3.2 DSA)', linewidth=2, markersize=5)
    ax2.plot(xk, cost_sparse,  's-', label='Sparse MLA Decode (V3.2 DSA)', linewidth=2, markersize=5)
    ax2.plot(xk, cost_total,   '^-', label='Total DSA Decode (V3.2)', linewidth=2, markersize=5)
    ax2.plot(xk, cost_base,    'x--', label='Baseline MLA Decode (V3)', linewidth=2, markersize=6)
    ax2.set_xlabel('Context Length (K tokens)')
    ax2.set_ylabel('Cost ($/1M tokens)')
    ax2.set_title(f'Decode Attention Cost (top-k={topk})\n$2/GPU-hour')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nLatency/Cost plot saved to: {output_path}")


def plot_tflops_tbps(
    lengths: List[int],
    tflops_indexer: List[float],
    tflops_sparse: List[float],
    tflops_total: List[float],
    tflops_base: List[float],
    tbps_indexer: List[float],
    tbps_sparse: List[float],
    tbps_total: List[float],
    tbps_base: List[float],
    gpu_name: str,
    output_path: str,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    xk = [L / 1000 for L in lengths]

    ax1 = axes[0]
    ax1.plot(xk, tflops_indexer, 'o-', label='Indexer + Top-k (V3.2 DSA)', linewidth=2, markersize=5)
    ax1.plot(xk, tflops_sparse,  's-', label='Sparse MLA Decode (V3.2 DSA)', linewidth=2, markersize=5)
    ax1.plot(xk, tflops_total,   '^-', label='Total DSA Decode (V3.2)', linewidth=2, markersize=5)
    ax1.plot(xk, tflops_base,    'x--', label='Baseline MLA Decode (V3)', linewidth=2, markersize=6)
    ax1.set_xlabel('Context Length (K tokens)')
    ax1.set_ylabel('Effective TFLOP/s (matmul-equivalent)')
    ax1.set_title(f'Effective Compute Throughput\n{gpu_name}')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)

    ax2 = axes[1]
    ax2.plot(xk, tbps_indexer, 'o-', label='Indexer + Top-k (V3.2 DSA)', linewidth=2, markersize=5)
    ax2.plot(xk, tbps_sparse,  's-', label='Sparse MLA Decode (V3.2 DSA)', linewidth=2, markersize=5)
    ax2.plot(xk, tbps_total,   '^-', label='Total DSA Decode (V3.2)', linewidth=2, markersize=5)
    ax2.plot(xk, tbps_base,    'x--', label='Baseline MLA Decode (V3)', linewidth=2, markersize=6)
    ax2.set_xlabel('Context Length (K tokens)')
    ax2.set_ylabel('Effective TB/s (algorithmic bytes)')
    ax2.set_title(f'Effective Memory Throughput\n{gpu_name}')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"TFLOP/TB plot saved to: {output_path}")


# -----------------------------
# Main
# -----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--min-len", type=int, default=2048)
    p.add_argument("--max-len", type=int, default=131072)
    p.add_argument("--step", type=int, default=2048)
    p.add_argument("--topk", type=int, default=2048)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--gpu-hour-price", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Number of independent rows (sequences) to run per step. "
                        "Use >=16 or >=32 to remove the small-L top-k spike (then times are divided by B).")
    p.add_argument("--output-plot", type=str, default="dsa_vs_mla_latency_cost.png")
    p.add_argument("--output-throughput-plot", type=str, default="dsa_vs_mla_tflops_tbps.png")
    args = p.parse_args()

    cfg = BenchCfg(
        min_len=args.min_len,
        max_len=args.max_len,
        step=args.step,
        gpu_hour_price=args.gpu_hour_price,
        warmup=args.warmup,
        iters=args.iters,
        batch_size=args.batch_size,
        index_heads=64,
        index_head_dim=128,
        topk=args.topk,
        next_n=1,
        mla_q_heads=128,
        mla_kv_heads=1,
        mla_d_qk=576,
        mla_d_v=512,
        block_size=64,
        device=args.device,
    )

    if cfg.min_len < cfg.topk:
        raise ValueError(f"min_len ({cfg.min_len}) must be >= topk ({cfg.topk})")

    device = torch.device(cfg.device)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    torch.manual_seed(args.seed)
    torch.cuda.set_device(device)

    # DeepGEMM availability
    try:
        num_sms = deep_gemm.get_num_sms()
    except Exception as e:
        raise RuntimeError(
            "vllm.utils.deep_gemm / DeepGEMM is not available.\n"
            "You need DeepGEMM installed to benchmark the Lightning indexer."
        ) from e

    gpu_name = torch.cuda.get_device_name(device)
    cc = torch.cuda.get_device_capability(device)
    print(f"GPU: {gpu_name} (cc={cc[0]}.{cc[1]})")
    print(f"DeepGEMM num_sms: {num_sms}")
    print(f"block_size={cfg.block_size}, DSA topk={cfg.topk}, batch_size={cfg.batch_size}")

    dense_ok, dense_reason = is_flashmla_dense_supported()
    if dense_ok:
        print("FlashMLA dense baseline: supported (will use indices=None).")
    else:
        print(f"FlashMLA dense baseline: NOT supported (will emulate dense via sparse topk=L). Reason: {dense_reason}")

    print("All reported latencies/costs are PER TOKEN (divide by batch_size*next_n).")
    print()

    B = cfg.batch_size
    tokens_per_call = B * cfg.next_n
    BASELINE_TOPK_MULTIPLE = 64

    # Fixed queries
    q_indexer = torch.zeros((B, cfg.next_n, cfg.index_heads, cfg.index_head_dim),
                            device=device, dtype=torch.float8_e4m3fn)
    weights = torch.ones((B * cfg.next_n, cfg.index_heads),
                         device=device, dtype=torch.float32)
    q_mla = torch.zeros((B, 1, cfg.mla_q_heads, cfg.mla_d_qk),
                        device=device, dtype=torch.bfloat16)

    # Buffers
    num_rows = B * cfg.next_n
    topk_buf = torch.empty((num_rows, cfg.topk), device=device, dtype=torch.int32)

    # Allocate KV pools ONCE for max_len, with unique blocks per batch element
    max_blocks_per_seq = math.ceil(cfg.max_len / cfg.block_size)
    total_blocks_pool = B * max_blocks_per_seq
    print(f"Allocating KV pools: max_blocks_per_seq={max_blocks_per_seq}, total_blocks_pool={total_blocks_pool}")
    kv_indexer_pool = make_indexer_kv_cache_fp8_paged_fast(
        num_blocks_total=total_blocks_pool,
        block_size=cfg.block_size,
        d=cfg.index_head_dim,
        device=device,
    )
    kv_mla_pool = make_flashmla_fp8_ds_mla_kvcache_fast(
        num_blocks_total=total_blocks_pool,
        block_size=cfg.block_size,
        device=device,
    )

    # Infer element sizes once (logits, flashmla output)
    logits_elem_size: Optional[int] = None
    mla_out_elem_size: Optional[int] = None
    mla_q_elem_size = q_mla.element_size()

    # Per-token q/weights bytes for indexer model
    # q_indexer is FP8 -> 1 byte/element
    q_indexer_bytes_per_token = cfg.index_heads * cfg.index_head_dim * q_indexer.element_size()
    weights_bytes_per_token = cfg.index_heads * weights.element_size()

    lengths: List[int] = []

    ms_indexer_tok: List[float] = []
    ms_sparse_tok: List[float] = []
    ms_total_tok: List[float] = []
    ms_base_tok: List[float] = []

    cost_indexer: List[float] = []
    cost_sparse: List[float] = []
    cost_total: List[float] = []
    cost_base: List[float] = []

    tflops_indexer: List[float] = []
    tflops_sparse: List[float] = []
    tflops_total: List[float] = []
    tflops_base: List[float] = []

    tbps_indexer: List[float] = []
    tbps_sparse: List[float] = []
    tbps_total: List[float] = []
    tbps_base: List[float] = []

    num_q_tokens_per_head_k = 1 * cfg.mla_q_heads // cfg.mla_kv_heads

    for L in range(cfg.min_len, cfg.max_len + 1, cfg.step):
        num_blocks = math.ceil(L / cfg.block_size)

        # cache lengths per seq
        context_lens = torch.full((B,), L, device=device, dtype=torch.int32)

        # block tables: each sequence gets its own contiguous block range in the pool
        base_blocks = torch.arange(num_blocks, device=device, dtype=torch.int32).view(1, num_blocks)
        offsets = (torch.arange(B, device=device, dtype=torch.int32) * max_blocks_per_seq).view(B, 1)
        block_tables = base_blocks + offsets  # [B, num_blocks]

        schedule_md = deep_gemm.get_paged_mqa_logits_metadata(
            context_lens, cfg.block_size, num_sms
        )

        # Sparse MLA metadata (DSA)
        tile_md_sparse, num_splits_sparse = get_mla_metadata(
            cache_seqlens=context_lens,
            num_q_tokens_per_head_k=num_q_tokens_per_head_k,
            num_heads_k=cfg.mla_kv_heads,
            num_heads_q=cfg.mla_q_heads,
            is_fp8_kvcache=True,
            topk=cfg.topk,
        )

        indices_view = topk_buf.view(B, cfg.next_n, cfg.topk)[:, :1, :]

        # Baseline MLA metadata + indices
        baseline_indices = None
        baseline_topk = None
        if dense_ok:
            tile_md_base, num_splits_base = get_mla_metadata(
                cache_seqlens=context_lens,
                num_q_tokens_per_head_k=num_q_tokens_per_head_k,
                num_heads_k=cfg.mla_kv_heads,
                num_heads_q=None,
                is_fp8_kvcache=True,
                topk=None,
            )
        else:
            baseline_topk = round_up(L, BASELINE_TOPK_MULTIPLE)
            base = torch.arange(baseline_topk, device=device, dtype=torch.int32).view(1, 1, baseline_topk)
            baseline_indices = base.repeat(B, 1, 1)
            if baseline_topk > L:
                baseline_indices[..., L:] = -1

            tile_md_base, num_splits_base = get_mla_metadata(
                cache_seqlens=context_lens,
                num_q_tokens_per_head_k=num_q_tokens_per_head_k,
                num_heads_k=cfg.mla_kv_heads,
                num_heads_q=cfg.mla_q_heads,
                is_fp8_kvcache=True,
                topk=baseline_topk,
            )

        # One-time: determine logits dtype/elem_size and mla output elem_size
        if logits_elem_size is None:
            logits_tmp = deep_gemm.fp8_paged_mqa_logits(
                q_indexer, kv_indexer_pool, weights, context_lens,
                block_tables, schedule_md, max_model_len=L,
            )
            logits_elem_size = logits_tmp.element_size()
            del logits_tmp
            torch.cuda.synchronize()

        if mla_out_elem_size is None:
            res = flash_mla_with_kvcache(
                q=q_mla,
                k_cache=kv_mla_pool,
                block_table=block_tables,
                cache_seqlens=context_lens,
                head_dim_v=cfg.mla_d_v,
                tile_scheduler_metadata=tile_md_sparse,
                num_splits=num_splits_sparse,
                softmax_scale=None,
                causal=False,
                is_fp8_kvcache=True,
                indices=indices_view,
            )
            out = res[0] if isinstance(res, tuple) else res
            mla_out_elem_size = out.element_size()
            del res, out
            torch.cuda.synchronize()

        # -----------------------------
        # Timed kernels
        # -----------------------------
        def run_indexer_and_topk():
            logits = deep_gemm.fp8_paged_mqa_logits(
                q_indexer, kv_indexer_pool, weights, context_lens,
                block_tables, schedule_md, max_model_len=L,
            )
            torch.ops._C.top_k_per_row_decode(
                logits,
                cfg.next_n,
                context_lens,
                topk_buf,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                cfg.topk,
            )

        def run_sparse_mla_decode_only():
            flash_mla_with_kvcache(
                q=q_mla,
                k_cache=kv_mla_pool,
                block_table=block_tables,
                cache_seqlens=context_lens,
                head_dim_v=cfg.mla_d_v,
                tile_scheduler_metadata=tile_md_sparse,
                num_splits=num_splits_sparse,
                softmax_scale=None,
                causal=False,
                is_fp8_kvcache=True,
                indices=indices_view,
            )

        def run_total_dsa():
            logits = deep_gemm.fp8_paged_mqa_logits(
                q_indexer, kv_indexer_pool, weights, context_lens,
                block_tables, schedule_md, max_model_len=L,
            )
            torch.ops._C.top_k_per_row_decode(
                logits,
                cfg.next_n,
                context_lens,
                topk_buf,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                cfg.topk,
            )
            flash_mla_with_kvcache(
                q=q_mla,
                k_cache=kv_mla_pool,
                block_table=block_tables,
                cache_seqlens=context_lens,
                head_dim_v=cfg.mla_d_v,
                tile_scheduler_metadata=tile_md_sparse,
                num_splits=num_splits_sparse,
                softmax_scale=None,
                causal=False,
                is_fp8_kvcache=True,
                indices=indices_view,
            )

        def run_mla_baseline_only():
            flash_mla_with_kvcache(
                q=q_mla,
                k_cache=kv_mla_pool,
                block_table=block_tables,
                cache_seqlens=context_lens,
                head_dim_v=cfg.mla_d_v,
                tile_scheduler_metadata=tile_md_base,
                num_splits=num_splits_base,
                softmax_scale=None,
                causal=False,
                is_fp8_kvcache=True,
                indices=baseline_indices,  # None => dense baseline, tensor => sparse-emulated dense
            )

        ms_idx_call = time_cuda_ms(run_indexer_and_topk, warmup=cfg.warmup, iters=cfg.iters)
        ms_sp_call  = time_cuda_ms(run_sparse_mla_decode_only, warmup=cfg.warmup, iters=cfg.iters)
        ms_tot_call = time_cuda_ms(run_total_dsa, warmup=cfg.warmup, iters=cfg.iters)
        ms_bas_call = time_cuda_ms(run_mla_baseline_only, warmup=cfg.warmup, iters=cfg.iters)

        # Convert to per-token
        ms_idx = ms_idx_call / tokens_per_call
        ms_sp  = ms_sp_call  / tokens_per_call
        ms_tot = ms_tot_call / tokens_per_call
        ms_bas = ms_bas_call / tokens_per_call

        # Costs per 1M tokens
        c_idx = ms_to_cost_per_1m_tokens(ms_idx, cfg.gpu_hour_price)
        c_sp  = ms_to_cost_per_1m_tokens(ms_sp,  cfg.gpu_hour_price)
        c_tot = ms_to_cost_per_1m_tokens(ms_tot, cfg.gpu_hour_price)
        c_bas = ms_to_cost_per_1m_tokens(ms_bas, cfg.gpu_hour_price)

        # FLOPs/bytes per token (single sequence)
        f_idx = flops_indexer_per_token(L, cfg.index_heads, cfg.index_head_dim)
        f_sp  = flops_mla_per_token(cfg.topk, cfg.mla_q_heads, cfg.mla_d_qk, cfg.mla_d_v)
        f_tot = f_idx + f_sp
        f_bas = flops_mla_per_token(L, cfg.mla_q_heads, cfg.mla_d_qk, cfg.mla_d_v)

        b_idx = bytes_indexer_topk_per_token(
            L=L,
            topk=cfg.topk,
            d_idx=cfg.index_head_dim,
            logits_elem_size=logits_elem_size,
            q_bytes=q_indexer_bytes_per_token,
            w_bytes=weights_bytes_per_token,
        )
        b_sp  = bytes_sparse_mla_per_token(
            topk=cfg.topk,
            h_q=cfg.mla_q_heads,
            d_qk=cfg.mla_d_qk,
            d_v=cfg.mla_d_v,
            q_elem_size=mla_q_elem_size,
            out_elem_size=mla_out_elem_size,
        )
        b_tot = b_idx + b_sp
        b_bas = bytes_dense_mla_per_token(
            L=L,
            h_q=cfg.mla_q_heads,
            d_qk=cfg.mla_d_qk,
            d_v=cfg.mla_d_v,
            q_elem_size=mla_q_elem_size,
            out_elem_size=mla_out_elem_size,
            dense_supported=dense_ok,
            baseline_topk=(baseline_topk or L),
        )

        # Effective throughput (per token)
        tf_idx = to_tflops(f_idx, ms_idx)
        tf_sp  = to_tflops(f_sp,  ms_sp)
        tf_tot = to_tflops(f_tot, ms_tot)
        tf_bas = to_tflops(f_bas, ms_bas)

        tb_idx = to_tbps(b_idx, ms_idx)
        tb_sp  = to_tbps(b_sp,  ms_sp)
        tb_tot = to_tbps(b_tot, ms_tot)
        tb_bas = to_tbps(b_bas, ms_bas)

        # Store
        lengths.append(L)

        ms_indexer_tok.append(ms_idx)
        ms_sparse_tok.append(ms_sp)
        ms_total_tok.append(ms_tot)
        ms_base_tok.append(ms_bas)

        cost_indexer.append(c_idx)
        cost_sparse.append(c_sp)
        cost_total.append(c_tot)
        cost_base.append(c_bas)

        tflops_indexer.append(tf_idx)
        tflops_sparse.append(tf_sp)
        tflops_total.append(tf_tot)
        tflops_base.append(tf_bas)

        tbps_indexer.append(tb_idx)
        tbps_sparse.append(tb_sp)
        tbps_total.append(tb_tot)
        tbps_base.append(tb_bas)

        base_note = ""
        if not dense_ok:
            base_note = f" (sparse-emulated topk={baseline_topk})"

        print(
            f"L={L:7d} | "
            f"idx+topk={ms_idx:8.4f} ms/tok ({c_idx:7.3f} $/1M, {tf_idx:6.1f} TF, {tb_idx:5.2f} TB/s) | "
            f"sparse_mla={ms_sp:8.4f} ms/tok ({c_sp:7.3f} $/1M, {tf_sp:6.1f} TF, {tb_sp:5.2f} TB/s) | "
            f"TOTAL={ms_tot:8.4f} ms/tok ({c_tot:7.3f} $/1M, {tf_tot:6.1f} TF, {tb_tot:5.2f} TB/s) | "
            f"V3 baseline={ms_bas:8.4f} ms/tok ({c_bas:7.3f} $/1M, {tf_bas:6.1f} TF, {tb_bas:5.2f} TB/s){base_note}"
        )

    # Plots
    plot_latency_cost(
        lengths=lengths,
        ms_indexer=ms_indexer_tok,
        ms_sparse=ms_sparse_tok,
        ms_total=ms_total_tok,
        ms_base=ms_base_tok,
        cost_indexer=cost_indexer,
        cost_sparse=cost_sparse,
        cost_total=cost_total,
        cost_base=cost_base,
        topk=cfg.topk,
        gpu_name=gpu_name,
        output_path=args.output_plot,
    )

    plot_tflops_tbps(
        lengths=lengths,
        tflops_indexer=tflops_indexer,
        tflops_sparse=tflops_sparse,
        tflops_total=tflops_total,
        tflops_base=tflops_base,
        tbps_indexer=tbps_indexer,
        tbps_sparse=tbps_sparse,
        tbps_total=tbps_total,
        tbps_base=tbps_base,
        gpu_name=gpu_name,
        output_path=args.output_throughput_plot,
    )


if __name__ == "__main__":
    main()
