#!/usr/bin/env python3
"""Vendor-neutral microbenchmark for DeepSeek-R1 MoE kernels via SGLang's FusedMoE layer.

Auto-dispatches to the correct backend:
  - NVIDIA (B200/H200): flashinfer_cutlass / triton / flashinfer_trtllm
  - AMD (MI355X):       aiter CK MXFP4 / triton

Usage (inside SGLang container, single GPU):
    python3 bench_deepseek_moe.py [--batch-sizes 1,8,32,64,128] [--warmup 20] [--iters 100]

Multi-GPU (for allreduce benchmarks):
    torchrun --nproc_per_node=4 bench_deepseek_moe.py --allreduce

Containers:
  B200:   lmsysorg/sglang:v0.5.9-cu130
  MI355X: lmsysorg/sglang:v0.5.9-rocm700-mi35x
"""
import argparse
import time
import sys

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Detect platform
# ---------------------------------------------------------------------------
def get_platform():
    if hasattr(torch, "hip") or torch.version.hip is not None:
        return "hip"
    return "cuda"


PLATFORM = get_platform()
DEVICE = "cuda" if PLATFORM == "cuda" else "hip:0"
SYNC = torch.cuda.synchronize if PLATFORM == "cuda" else lambda: torch.hip.synchronize()


# ---------------------------------------------------------------------------
# DeepSeek-R1 model constants
# ---------------------------------------------------------------------------
HIDDEN_SIZE = 7168
N_ROUTED_EXPERTS = 256
N_SHARED_EXPERTS = 2
MOE_INTERMEDIATE_SIZE = 2048  # full size per expert (before TP shard)
TOPK = 8  # routed experts per token
TOPK_GROUP = 4
N_GROUP = 8
ROUTED_SCALING_FACTOR = 2.5


# ---------------------------------------------------------------------------
# 1. FusedMoE via SGLang (vendor-neutral)
# ---------------------------------------------------------------------------
def bench_sglang_fused_moe(batch_sizes, tp_size, warmup, iters):
    """Benchmark SGLang's FusedMoE layer — auto-dispatches per platform."""
    print("\n" + "=" * 72)
    print("BENCHMARK: SGLang FusedMoE (vendor-neutral auto-dispatch)")
    print("=" * 72)

    try:
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
        from sglang.srt.layers.moe.topk import TopKOutput
    except ImportError:
        print("ERROR: SGLang not available. Install via: pip install sglang")
        print("Falling back to pure-torch benchmarks.")
        return False

    intermediate_per_shard = MOE_INTERMEDIATE_SIZE // tp_size
    num_experts = N_ROUTED_EXPERTS + N_SHARED_EXPERTS  # 258 total (256 routed + 2 shared)

    try:
        layer = FusedMoE(
            num_experts=num_experts,
            hidden_size=HIDDEN_SIZE,
            intermediate_size=intermediate_per_shard,
            layer_id=0,
            top_k=TOPK,
            params_dtype=torch.bfloat16,
            reduce_results=False,
            activation="silu",
        ).to(DEVICE)
    except Exception as e:
        print(f"ERROR creating FusedMoE layer: {e}")
        print("This may require quantization config or model weights.")
        return False

    print(f"Platform: {PLATFORM} | TP={tp_size} | Experts={num_experts} "
          f"| hidden={HIDDEN_SIZE} | intermediate/shard={intermediate_per_shard}")
    print(f"w13 shape: {list(layer.w13_weight.shape)} dtype={layer.w13_weight.dtype}")
    print(f"w2  shape: {list(layer.w2_weight.shape)} dtype={layer.w2_weight.dtype}")
    print()

    for bs in batch_sizes:
        x = torch.randn(bs, HIDDEN_SIZE, dtype=torch.bfloat16, device=DEVICE)
        topk_weights = torch.randn(bs, TOPK, dtype=torch.float32, device=DEVICE).softmax(dim=-1)
        topk_ids = torch.randint(0, N_ROUTED_EXPERTS, (bs, TOPK), dtype=torch.int32, device=DEVICE)
        topk_output = TopKOutput(
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )

        try:
            # Warmup
            for _ in range(warmup):
                layer(x, topk_output)
            SYNC()

            t0 = time.perf_counter()
            for _ in range(iters):
                layer(x, topk_output)
            SYNC()
            elapsed_us = (time.perf_counter() - t0) / iters * 1e6

            # FLOPS: gate_up = 2*M*topk*intermediate*2*hidden, down = 2*M*topk*hidden*intermediate
            flops = 2 * bs * TOPK * (2 * intermediate_per_shard * HIDDEN_SIZE + HIDDEN_SIZE * intermediate_per_shard)
            tflops = flops / elapsed_us / 1e6
            print(f"  BS={bs:>4d}: FusedMoE avg={elapsed_us:>8.1f}us  ({tflops:>6.1f} TFLOPS)")
        except Exception as e:
            print(f"  BS={bs:>4d}: FusedMoE FAILED: {e}")

    return True


# ---------------------------------------------------------------------------
# 2. Dense GEMMs (attention projections) — pure torch.mm
# ---------------------------------------------------------------------------
def bench_dense_gemms(batch_sizes, tp_size, warmup, iters):
    """Benchmark dense GEMMs matching DeepSeek-R1 attention projections.
    Uses torch.mm which dispatches to hipBLASLt (AMD) or cuBLAS (NVIDIA).
    """
    print("\n" + "=" * 72)
    print("BENCHMARK: Dense GEMMs (attention projections) — torch.mm")
    print("=" * 72)

    # DeepSeek-R1 MLA dims (TP-sharded)
    num_heads_per_shard = 128 // tp_size
    q_head_dim = 192  # 128 nope + 64 rope
    kv_lora_rank = 512
    qk_rope_dim = 64
    v_head_dim = 128

    for bs in batch_sizes:
        print(f"\n  Batch={bs}:")
        gemm_configs = [
            # (name, M, N, K)
            (f"q_a_proj",        bs, kv_lora_rank + qk_rope_dim, HIDDEN_SIZE),  # 992x1536 <- 992x7168
            (f"q_b_proj",        bs, num_heads_per_shard * q_head_dim, kv_lora_rank),  # 992x6144 <- 992x1536
            (f"kv_a_proj",       bs, kv_lora_rank + qk_rope_dim, HIDDEN_SIZE),  # 992x576 <- 992x7168
            (f"o_proj",          bs, HIDDEN_SIZE, num_heads_per_shard * v_head_dim),  # 992x7168 <- 992x4096
        ]

        for name, M, N, K in gemm_configs:
            a = torch.randn(M, K, dtype=torch.bfloat16, device=DEVICE)
            b = torch.randn(N, K, dtype=torch.bfloat16, device=DEVICE)

            for _ in range(warmup):
                torch.mm(a, b.t())
            SYNC()

            t0 = time.perf_counter()
            for _ in range(iters):
                torch.mm(a, b.t())
            SYNC()
            elapsed_us = (time.perf_counter() - t0) / iters * 1e6

            flops = 2 * M * N * K
            tflops = flops / elapsed_us / 1e6
            print(f"    {name:20s}: [{M}x{K}] x [{K}x{N}] = {elapsed_us:>7.1f}us ({tflops:>5.1f} TFLOPS)")


# ---------------------------------------------------------------------------
# 3. MoE GEMMs decomposed — pure torch.mm (no fused kernel)
# ---------------------------------------------------------------------------
def bench_moe_gemms_decomposed(batch_sizes, tp_size, warmup, iters):
    """Benchmark MoE GEMMs as individual torch.mm calls (gate_up + down_proj).
    Shows the raw GEMM performance without MoE dispatch overhead.
    """
    print("\n" + "=" * 72)
    print("BENCHMARK: MoE GEMMs decomposed — torch.mm (per-expert shapes)")
    print("=" * 72)

    intermediate_per_shard = MOE_INTERMEDIATE_SIZE // tp_size

    for bs in batch_sizes:
        # Tokens per expert after routing: bs * topk / num_experts
        # For bs=128, topk=8, 256 experts => ~4 tokens/expert avg
        tokens_per_expert = max(1, (bs * TOPK) // N_ROUTED_EXPERTS)
        M = tokens_per_expert

        print(f"\n  Batch={bs} (~{M} tokens/expert):")

        gemm_configs = [
            # gate_up: [M, hidden] x [2*intermediate, hidden].T -> [M, 2*intermediate]
            (f"gate_up",  M, 2 * intermediate_per_shard, HIDDEN_SIZE),
            # down_proj: [M, intermediate] x [hidden, intermediate].T -> [M, hidden]
            (f"down_proj", M, HIDDEN_SIZE, intermediate_per_shard),
        ]

        for name, m, n, k in gemm_configs:
            a = torch.randn(m, k, dtype=torch.bfloat16, device=DEVICE)
            b = torch.randn(n, k, dtype=torch.bfloat16, device=DEVICE)

            for _ in range(warmup):
                torch.mm(a, b.t())
            SYNC()

            t0 = time.perf_counter()
            for _ in range(iters):
                torch.mm(a, b.t())
            SYNC()
            elapsed_us = (time.perf_counter() - t0) / iters * 1e6

            flops = 2 * m * n * k
            tflops = flops / elapsed_us / 1e6
            print(f"    {name:12s}: [{m}x{k}] x [{k}x{n}] = {elapsed_us:>7.1f}us ({tflops:>5.1f} TFLOPS)")


# ---------------------------------------------------------------------------
# 4. RMSNorm — pure PyTorch
# ---------------------------------------------------------------------------
def bench_rmsnorm(batch_sizes, warmup, iters):
    """Benchmark RMSNorm (fused add+norm+quant dispatches to platform-specific kernel)."""
    print("\n" + "=" * 72)
    print("BENCHMARK: RMSNorm (add + normalize)")
    print("=" * 72)

    eps = 1e-6

    # Try SGLang's fused kernel first
    use_sglang = False
    try:
        from sglang.srt.layers.layernorm import RMSNorm
        norm = RMSNorm(HIDDEN_SIZE, eps=eps).to(dtype=torch.bfloat16, device=DEVICE)
        use_sglang = True
        print("  Using: SGLang RMSNorm (fused kernel)")
    except ImportError:
        print("  Using: PyTorch manual RMSNorm")

    for bs in batch_sizes:
        x = torch.randn(bs, HIDDEN_SIZE, dtype=torch.bfloat16, device=DEVICE)
        residual = torch.randn(bs, HIDDEN_SIZE, dtype=torch.bfloat16, device=DEVICE)

        if use_sglang:
            # SGLang path: fused add_rmsnorm
            for _ in range(warmup):
                norm(x, residual=residual)
            SYNC()

            t0 = time.perf_counter()
            for _ in range(iters):
                norm(x, residual=residual)
            SYNC()
        else:
            # Pure PyTorch fallback
            weight = torch.ones(HIDDEN_SIZE, dtype=torch.bfloat16, device=DEVICE)
            for _ in range(warmup):
                combined = x + residual
                variance = combined.pow(2).mean(-1, keepdim=True)
                combined * torch.rsqrt(variance + eps) * weight
            SYNC()

            t0 = time.perf_counter()
            for _ in range(iters):
                combined = x + residual
                variance = combined.pow(2).mean(-1, keepdim=True)
                combined * torch.rsqrt(variance + eps) * weight
            SYNC()

        elapsed_us = (time.perf_counter() - t0) / iters * 1e6
        bw = 4 * bs * HIDDEN_SIZE * 2 / elapsed_us * 1e6 / 1e9  # 2 reads + 2 writes, bf16
        print(f"  BS={bs:>4d}: [{bs}x{HIDDEN_SIZE}] add+rmsnorm avg={elapsed_us:>6.1f}us, BW={bw:>5.0f}GB/s")


# ---------------------------------------------------------------------------
# 5. Dtype copies — pure PyTorch
# ---------------------------------------------------------------------------
def bench_dtype_copies(batch_sizes, warmup, iters):
    """Benchmark dtype conversion kernels (bf16<->fp32, bf16->fp8)."""
    print("\n" + "=" * 72)
    print("BENCHMARK: Dtype copies (conversion kernels)")
    print("=" * 72)

    conversions = [
        ("bf16 -> fp32", torch.bfloat16, torch.float32),
        ("fp32 -> bf16", torch.float32, torch.bfloat16),
    ]

    # FP8 support check
    try:
        _ = torch.zeros(1, dtype=torch.float8_e4m3fn, device=DEVICE)
        conversions.append(("bf16 -> fp8_e4m3", torch.bfloat16, torch.float8_e4m3fn))
        conversions.append(("fp8_e4m3 -> bf16", torch.float8_e4m3fn, torch.bfloat16))
    except Exception:
        print("  (FP8 not supported on this platform, skipping)")

    for bs in batch_sizes:
        print(f"\n  Batch={bs}:")
        for name, src_dtype, dst_dtype in conversions:
            x = torch.randn(bs, HIDDEN_SIZE, dtype=torch.float32, device=DEVICE).to(src_dtype)

            for _ in range(warmup):
                x.to(dst_dtype)
            SYNC()

            t0 = time.perf_counter()
            for _ in range(iters):
                x.to(dst_dtype)
            SYNC()
            elapsed_us = (time.perf_counter() - t0) / iters * 1e6
            read_bytes = bs * HIDDEN_SIZE * x.element_size()
            write_bytes = bs * HIDDEN_SIZE * torch.tensor([], dtype=dst_dtype).element_size()
            bw = (read_bytes + write_bytes) / elapsed_us * 1e6 / 1e9
            print(f"    {name:20s}: [{bs}x{HIDDEN_SIZE}] avg={elapsed_us:>6.1f}us, BW={bw:>5.0f}GB/s")


# ---------------------------------------------------------------------------
# 6. AllReduce — torch.distributed (dispatches to custom AR or NCCL/RCCL)
# ---------------------------------------------------------------------------
def bench_allreduce(batch_sizes, warmup, iters):
    """Benchmark allreduce. On AMD dispatches to cross_device_reduce_2stage,
    on NVIDIA dispatches to custom allreduce or NCCL.
    Requires: torchrun --nproc_per_node=N
    """
    print("\n" + "=" * 72)
    print("BENCHMARK: AllReduce (torch.distributed)")
    print("=" * 72)

    if not dist.is_initialized():
        dist.init_process_group("nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}" if PLATFORM == "cuda" else f"hip:{rank}"

    # Try SGLang custom allreduce (what the profile actually uses)
    use_custom = False
    ca = None
    try:
        # AMD path: try aiter first
        if PLATFORM == "hip":
            try:
                from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce
                ca = CustomAllreduce(dist.group.WORLD, torch.device(device))
                use_custom = True
                source = "aiter"
            except ImportError:
                pass

        if not use_custom:
            from sglang.srt.distributed.device_communicators.custom_all_reduce import CustomAllreduce
            ca = CustomAllreduce(dist.group.WORLD, torch.device(device))
            use_custom = True
            source = "sglang"
    except Exception:
        pass

    if rank == 0:
        if use_custom:
            print(f"  Using: {source} CustomAllreduce (P2P, bypasses NCCL/RCCL)")
        else:
            print(f"  Using: torch.distributed.all_reduce (NCCL/RCCL)")
        print(f"  World size: {world_size}")

    for bs in batch_sizes:
        x = torch.randn(bs, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)

        if use_custom:
            out = torch.empty_like(x)
            for _ in range(warmup):
                ca.all_reduce_unreg(x, out)
            torch.cuda.synchronize()

            t0 = time.perf_counter()
            for _ in range(iters):
                ca.all_reduce_unreg(x, out)
            torch.cuda.synchronize()
        else:
            for _ in range(warmup):
                dist.all_reduce(x)
            torch.cuda.synchronize()

            t0 = time.perf_counter()
            for _ in range(iters):
                dist.all_reduce(x)
            torch.cuda.synchronize()

        elapsed_us = (time.perf_counter() - t0) / iters * 1e6
        nbytes = x.numel() * x.element_size()
        algbw = 2 * nbytes * (world_size - 1) / world_size / elapsed_us * 1e6 / 1e9

        if rank == 0:
            print(f"  [{bs:>4d}, {HIDDEN_SIZE}] bf16  {nbytes/1e3:>7.1f}KB  avg={elapsed_us:>7.1f}us  algBW={algbw:>5.1f}GB/s")

    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# 7. MoE Routing (topk + sort)
# ---------------------------------------------------------------------------
def bench_moe_routing(batch_sizes, warmup, iters):
    """Benchmark MoE routing: grouped_topk selection + token sorting."""
    print("\n" + "=" * 72)
    print("BENCHMARK: MoE Routing (grouped_topk + sort)")
    print("=" * 72)

    # Try SGLang's topk implementation
    use_sglang = False
    try:
        from sglang.srt.layers.moe.topk import select_experts
        use_sglang = True
        print("  Using: SGLang select_experts (auto-dispatch)")
    except ImportError:
        print("  Using: PyTorch torch.topk fallback")

    for bs in batch_sizes:
        router_logits = torch.randn(bs, N_ROUTED_EXPERTS, dtype=torch.bfloat16, device=DEVICE)

        if use_sglang:
            try:
                for _ in range(warmup):
                    select_experts(
                        router_logits,
                        top_k=TOPK,
                        use_grouped_topk=True,
                        topk_group=TOPK_GROUP,
                        num_expert_group=N_GROUP,
                        renormalize=True,
                    )
                SYNC()

                t0 = time.perf_counter()
                for _ in range(iters):
                    select_experts(
                        router_logits,
                        top_k=TOPK,
                        use_grouped_topk=True,
                        topk_group=TOPK_GROUP,
                        num_expert_group=N_GROUP,
                        renormalize=True,
                    )
                SYNC()
                elapsed_us = (time.perf_counter() - t0) / iters * 1e6
                print(f"  BS={bs:>4d}: select_experts avg={elapsed_us:>6.1f}us")
                continue
            except Exception as e:
                print(f"  BS={bs:>4d}: SGLang select_experts failed ({e}), falling back to torch.topk")

        # PyTorch fallback
        for _ in range(warmup):
            torch.topk(router_logits.float(), k=TOPK, dim=-1)
        SYNC()

        t0 = time.perf_counter()
        for _ in range(iters):
            torch.topk(router_logits.float(), k=TOPK, dim=-1)
        SYNC()
        elapsed_us = (time.perf_counter() - t0) / iters * 1e6
        print(f"  BS={bs:>4d}: torch.topk avg={elapsed_us:>6.1f}us")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Vendor-neutral DeepSeek-R1 MoE microbenchmark (SGLang auto-dispatch)")
    parser.add_argument("--batch-sizes", type=str, default="1,4,8,16,32,64,128",
                        help="Comma-separated batch sizes")
    parser.add_argument("--tp", type=int, default=4, help="Tensor parallel size")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--allreduce", action="store_true",
                        help="Benchmark allreduce (requires torchrun)")
    parser.add_argument("--skip-fused-moe", action="store_true",
                        help="Skip FusedMoE benchmark (if SGLang not available)")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    print("=" * 72)
    print(f"DeepSeek-R1 MoE Microbenchmark")
    print(f"Platform: {PLATFORM.upper()} | Device: {DEVICE}")
    print(f"Batch sizes: {batch_sizes} | TP={args.tp}")
    print(f"Hidden={HIDDEN_SIZE} | Experts={N_ROUTED_EXPERTS}+{N_SHARED_EXPERTS} | TopK={TOPK}")
    print(f"MoE intermediate (full)={MOE_INTERMEDIATE_SIZE} | per-shard={MOE_INTERMEDIATE_SIZE // args.tp}")
    print("=" * 72)

    if args.allreduce:
        bench_allreduce(batch_sizes, args.warmup, args.iters)
        return

    # 1. FusedMoE via SGLang
    if not args.skip_fused_moe:
        bench_sglang_fused_moe(batch_sizes, args.tp, args.warmup, args.iters)

    # 2. Dense GEMMs (attention projections)
    bench_dense_gemms(batch_sizes, args.tp, args.warmup, args.iters)

    # 3. MoE GEMMs decomposed
    bench_moe_gemms_decomposed(batch_sizes, args.tp, args.warmup, args.iters)

    # 4. RMSNorm
    bench_rmsnorm(batch_sizes, args.warmup, args.iters)

    # 5. Dtype copies
    bench_dtype_copies(batch_sizes, args.warmup, args.iters)

    # 6. MoE Routing
    bench_moe_routing(batch_sizes, args.warmup, args.iters)


if __name__ == "__main__":
    main()
