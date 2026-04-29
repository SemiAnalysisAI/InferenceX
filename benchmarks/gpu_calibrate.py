#!/usr/bin/env python3
"""GPU Matmul & AllReduce Calibration Suite.

A portable benchmark for comparing GPU compute and communication performance
across machines. Auto-detects and tests all available Tensor Core precisions
(BF16, FP16, TF32, FP8-e4m3, MXFP8, NVFP4) and NCCL AllReduce bandwidth.
Designed for B300/H100/H200 SXM nodes running LLM inference workloads.

Dependencies:
    PyTorch (with CUDA + NCCL). No other packages required.

Usage:
    If `torchrun` is not on PATH, use `python -m torch.distributed.run` instead.

    1) Full suite -- compute + communication (recommended):
       torchrun --nproc_per_node=auto gpu_calibrate.py
       # rank 0 runs matmul, all ranks run allreduce, rank 0 prints summary

    2) Matmul only -- single process, no distributed:
       python gpu_calibrate.py --matmul-only

    3) AllReduce only:
       torchrun --nproc_per_node=auto gpu_calibrate.py --allreduce-only
       # or specify GPU count explicitly:
       torchrun --nproc_per_node=4 gpu_calibrate.py --allreduce-only

    4) Export CSV for cross-machine comparison:
       torchrun --nproc_per_node=auto gpu_calibrate.py --output results.csv

    5) Tune iteration count for timing stability:
       python gpu_calibrate.py --matmul-only --iters 200 --warmup 20
       torchrun --nproc_per_node=8 gpu_calibrate.py --ar-iters 50 --warmup 10

Arguments:
    --matmul-only      Only run matmul benchmark (no distributed needed)
    --allreduce-only   Only run allreduce benchmark
    --iters N          Timed iterations for matmul          [default: 100]
    --ar-iters N       Timed iterations for allreduce       [default: 50]
    --warmup N         Warmup iterations                    [default: 10]
    --output PATH      Save results to CSV file

Tested dtypes (auto-detected per GPU):
    BF16      -- torch.matmul, bfloat16 Tensor Cores
    FP16      -- torch.matmul, float16 Tensor Cores
    TF32      -- torch.matmul, float32 input with TF32 Tensor Cores
    FP8-e4m3  -- torch._scaled_mm, tensorwise scaling
    MXFP8     -- torch._scaled_mm, blockwise 1x32 with float8_e8m0fnu scales
    NVFP4     -- torch._scaled_mm, blockwise 1x16 with float8_e4m3fn scales

AllReduce message sizes: 1MB, 4MB, 32MB, 128MB, 256MB, 512MB, 1GB, 2GB, 4GB, 8GB

Matmul shapes:
    Square:      256..16384 (powers of 2)
    Decode:      M=1,   N=K in {4096, 8192, 16384}
    Prefill:     M=128, N=K in {4096, 8192, 16384}
    Med-batch:   M=1024, N=K in {4096, 8192, 16384}
"""

import argparse
import csv
import os
import socket
import subprocess
import sys
from datetime import datetime

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# System info helpers
# ---------------------------------------------------------------------------

def get_system_info():
    info = {}
    info["hostname"] = socket.gethostname()
    info["gpu_name"] = torch.cuda.get_device_name(0)
    info["gpu_count"] = torch.cuda.device_count()
    cap = torch.cuda.get_device_capability(0)
    info["cuda_capability"] = f"{cap[0]}.{cap[1]}"
    info["torch_version"] = torch.__version__
    info["cuda_version"] = torch.version.cuda or "N/A"
    try:
        nccl_ver = torch.cuda.nccl.version()
        info["nccl_version"] = ".".join(str(x) for x in nccl_ver)
    except Exception:
        info["nccl_version"] = "N/A"
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).strip().split("\n")[0]
        info["driver"] = out
    except Exception:
        info["driver"] = "N/A"
    info["timestamp"] = datetime.now().isoformat()
    return info


def print_sysinfo(info):
    print(f"Hostname: {info['hostname']}")
    print(f"GPU: {info['gpu_name']} x {info['gpu_count']}")
    print(
        f"Driver: {info['driver']} | CUDA: {info['cuda_version']} "
        f"| PyTorch: {info['torch_version']} | NCCL: {info['nccl_version']}"
    )


# ---------------------------------------------------------------------------
# Dtype capability detection
# ---------------------------------------------------------------------------

def detect_dtypes():
    """Detect which matmul dtypes are supported on this GPU."""
    dtypes = []

    # Always available
    dtypes.append(("bfloat16", "tc_bf16"))
    dtypes.append(("float16", "tc_fp16"))
    dtypes.append(("tf32", "tc_tf32"))

    # FP8 e4m3 tensorwise (Hopper+, sm_89+)
    has_fp8 = hasattr(torch, "float8_e4m3fn") and hasattr(torch, "_scaled_mm")
    if has_fp8:
        try:
            a = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
            b = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
            s = torch.tensor(1.0, device="cuda", dtype=torch.float32)
            torch._scaled_mm(a, b.t(), scale_a=s, scale_b=s, out_dtype=torch.bfloat16)
            dtypes.append(("fp8_e4m3", "tc_fp8"))
            del a, b, s
        except Exception:
            pass

    # MXFP8: FP8 with blockwise 1x32 microscaling (Blackwell, sm_100+)
    has_mxfp8 = has_fp8 and hasattr(torch, "float8_e8m0fnu")
    if has_mxfp8:
        try:
            M_t, K_t = 256, 256
            a = torch.randn(M_t, K_t, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
            b = torch.randn(M_t, K_t, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
            n_scales = M_t * (K_t // 32)
            sa = torch.ones(n_scales, device="cuda", dtype=torch.float8_e8m0fnu)
            sb = torch.ones(n_scales, device="cuda", dtype=torch.float8_e8m0fnu)
            torch._scaled_mm(a, b.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
            dtypes.append(("mxfp8", "tc_mxfp8"))
            del a, b, sa, sb
        except Exception:
            pass

    # FP4 e2m1 blockwise 1x16 (Blackwell, sm_100+)
    has_fp4 = hasattr(torch, "float4_e2m1fn_x2") and hasattr(torch, "float8_e4m3fn")
    if has_fp4:
        try:
            M_t, K_t = 256, 256
            a = torch.randint(0, 256, (M_t, K_t // 2), device="cuda", dtype=torch.uint8).view(
                dtype=torch.float4_e2m1fn_x2
            )
            b = torch.randint(0, 256, (M_t, K_t // 2), device="cuda", dtype=torch.uint8).view(
                dtype=torch.float4_e2m1fn_x2
            )
            n_scales = M_t * (K_t // 16)
            sa = torch.ones(n_scales, device="cuda", dtype=torch.float8_e4m3fn)
            sb = torch.ones(n_scales, device="cuda", dtype=torch.float8_e4m3fn)
            torch._scaled_mm(a, b.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
            dtypes.append(("nvfp4", "tc_nvfp4"))
            del a, b, sa, sb
        except Exception:
            pass

    torch.cuda.empty_cache()
    return dtypes


# ---------------------------------------------------------------------------
# Matmul benchmark
# ---------------------------------------------------------------------------

SQUARE_SIZES = [256, 512, 1024, 2048, 4096, 8192, 16384]
RECT_K_SIZES = [4096, 8192, 16384]
RECT_M_SIZES = [1, 128, 1024]


def build_matmul_shapes():
    shapes = []
    for s in SQUARE_SIZES:
        shapes.append((s, s, s))
    for m in RECT_M_SIZES:
        for k in RECT_K_SIZES:
            shapes.append((m, k, k))
    return shapes


def _create_inputs_and_fn(M, N, K, dtype_name, device):
    """Create input tensors and a benchmark callable for the given dtype.

    Returns (run_fn, cleanup_tensors) where run_fn() executes one matmul
    and cleanup_tensors is a list of tensors to delete afterwards.
    """
    if dtype_name == "bfloat16":
        A = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        B = torch.randn(K, N, dtype=torch.bfloat16, device=device)
        fn = lambda: torch.matmul(A, B)
        return fn, [A, B]

    elif dtype_name == "float16":
        A = torch.randn(M, K, dtype=torch.float16, device=device)
        B = torch.randn(K, N, dtype=torch.float16, device=device)
        fn = lambda: torch.matmul(A, B)
        return fn, [A, B]

    elif dtype_name == "tf32":
        A = torch.randn(M, K, dtype=torch.float32, device=device)
        B = torch.randn(K, N, dtype=torch.float32, device=device)
        prev_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        def fn():
            return torch.matmul(A, B)
        def cleanup():
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32
        return fn, [A, B], cleanup

    elif dtype_name == "fp8_e4m3":
        A = torch.randn(M, K, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
        B = torch.randn(N, K, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
        sa = torch.tensor(1.0, device=device, dtype=torch.float32)
        sb = torch.tensor(1.0, device=device, dtype=torch.float32)
        Bt = B.t()
        fn = lambda: torch._scaled_mm(A, Bt, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
        return fn, [A, B, Bt, sa, sb]

    elif dtype_name == "mxfp8":
        # MXFP8: FP8 e4m3 data with blockwise 1x32 microscaling (e8m0fnu scales)
        A = torch.randn(M, K, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
        B = torch.randn(N, K, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
        sa = torch.ones(M * (K // 32), device=device, dtype=torch.float8_e8m0fnu)
        sb = torch.ones(N * (K // 32), device=device, dtype=torch.float8_e8m0fnu)
        Bt = B.t()
        fn = lambda: torch._scaled_mm(A, Bt, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
        return fn, [A, B, Bt, sa, sb]

    elif dtype_name == "nvfp4":
        # FP4 packed: 2 values per byte, so last dim is K//2
        # Scales: blockwise 1x16, one float8_e4m3fn scale per 16 elements
        A = torch.randint(0, 256, (M, K // 2), device=device, dtype=torch.uint8).view(
            dtype=torch.float4_e2m1fn_x2
        )
        B = torch.randint(0, 256, (N, K // 2), device=device, dtype=torch.uint8).view(
            dtype=torch.float4_e2m1fn_x2
        )
        sa = torch.ones(M * (K // 16), device=device, dtype=torch.float8_e4m3fn)
        sb = torch.ones(N * (K // 16), device=device, dtype=torch.float8_e4m3fn)
        Bt = B.t()
        fn = lambda: torch._scaled_mm(A, Bt, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
        return fn, [A, B, Bt, sa, sb]

    else:
        raise ValueError(f"Unknown dtype: {dtype_name}")


def bench_matmul(M, N, K, dtype_name, iters, warmup, device="cuda:0"):
    result = _create_inputs_and_fn(M, N, K, dtype_name, device)
    if len(result) == 3:
        fn, tensors, extra_cleanup = result
    else:
        fn, tensors = result
        extra_cleanup = None

    # warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize(device)

    elapsed_ms = start.elapsed_time(end)
    avg_ms = elapsed_ms / iters
    flops = 2.0 * M * N * K
    tflops = flops / (avg_ms / 1000.0) / 1e12

    if extra_cleanup:
        extra_cleanup()
    for t in tensors:
        del t
    torch.cuda.empty_cache()

    return avg_ms, tflops


def _shape_valid_for_dtype(M, N, K, dtype_name):
    """Check if a shape is valid for the given dtype.

    FP4/MXFP8 use blockwise scaling with CUTLASS kernels that require minimum
    tile sizes. M < 128 causes scale dimension mismatches due to internal padding.
    """
    if dtype_name == "nvfp4":
        # Blockwise 1x16: needs K divisible by 32, M >= 128 for tile alignment
        if K < 256 or K % 32 != 0 or M < 128:
            return False
        return True
    elif dtype_name == "mxfp8":
        # Blockwise 1x32: needs K divisible by 32, M >= 128 for tile alignment
        if K < 256 or K % 32 != 0 or M < 128:
            return False
        return True
    elif dtype_name == "fp8_e4m3":
        if K < 16 or K % 16 != 0:
            return False
        return True
    return True


def run_matmul_benchmark(args, sysinfo):
    available_dtypes = detect_dtypes()
    shapes = build_matmul_shapes()
    results = []

    print(f"\n  Detected Tensor Core dtypes: {', '.join(d[0] for d in available_dtypes)}")

    for dtype_name, tc_type in available_dtypes:
        print(f"\n--- {dtype_name} ({tc_type}) ---")
        print(f"  {'M':>7s} {'N':>7s} {'K':>7s} {'Time(ms)':>10s} {'TFLOPS':>10s}")

        for M, N, K in shapes:
            if not _shape_valid_for_dtype(M, N, K, dtype_name):
                continue
            try:
                avg_ms, tflops = bench_matmul(M, N, K, dtype_name, args.iters, args.warmup)
                print(f"  {M:>7d} {N:>7d} {K:>7d} {avg_ms:>10.4f} {tflops:>10.2f}")
                results.append({
                    "test_type": "matmul",
                    "dtype": dtype_name,
                    "M": M, "N": N, "K": K,
                    "size_bytes": "",
                    "time_ms": f"{avg_ms:.4f}",
                    "tflops": f"{tflops:.2f}",
                    "busbw_gbps": "",
                    "algobw_gbps": "",
                    **sysinfo,
                    "mode": "matmul",
                    "iters": args.iters,
                    "warmup": args.warmup,
                })
            except Exception as e:
                print(f"  {M:>7d} {N:>7d} {K:>7d}   FAILED: {e}")

    return results


# ---------------------------------------------------------------------------
# Per-GPU matmul benchmark (all GPUs in parallel)
# ---------------------------------------------------------------------------

# Representative shapes for per-GPU comparison (keep it fast)
PER_GPU_SHAPES = [
    (4096, 4096, 4096),
    (8192, 8192, 8192),
]
PER_GPU_DTYPES = ["bfloat16", "fp8_e4m3", "nvfp4"]


def run_per_gpu_matmul(args, sysinfo, rank, local_rank, world_size):
    """Each rank benchmarks its own GPU, then rank 0 collects and prints comparison."""
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)

    available = {d[0] for d in detect_dtypes()}
    my_results = []

    for dtype_name in PER_GPU_DTYPES:
        if dtype_name not in available:
            continue
        for M, N, K in PER_GPU_SHAPES:
            if not _shape_valid_for_dtype(M, N, K, dtype_name):
                continue
            try:
                avg_ms, tflops = bench_matmul(M, N, K, dtype_name, args.iters, args.warmup, device=device)
                my_results.append({
                    "gpu_id": local_rank,
                    "dtype": dtype_name,
                    "M": M, "N": N, "K": K,
                    "time_ms": avg_ms,
                    "tflops": tflops,
                })
            except Exception:
                my_results.append({
                    "gpu_id": local_rank,
                    "dtype": dtype_name,
                    "M": M, "N": N, "K": K,
                    "time_ms": -1,
                    "tflops": -1,
                })

    # Gather all results to rank 0
    all_gpu_results = [None] * world_size
    dist.all_gather_object(all_gpu_results, my_results)

    csv_results = []
    if rank == 0:
        # Flatten
        flat = []
        for gpu_results in all_gpu_results:
            flat.extend(gpu_results)

        # Print comparison table per (dtype, shape)
        print("\n" + "=" * 60)
        print("=== Per-GPU Matmul Comparison ===")
        print(f"    iters={args.iters}, warmup={args.warmup}")
        print("=" * 60)

        for dtype_name in PER_GPU_DTYPES:
            dtype_rows = [r for r in flat if r["dtype"] == dtype_name and r["tflops"] > 0]
            if not dtype_rows:
                continue
            shapes_in_dtype = sorted(set((r["M"], r["N"], r["K"]) for r in dtype_rows))
            for M, N, K in shapes_in_dtype:
                shape_rows = [r for r in dtype_rows if r["M"] == M and r["N"] == N and r["K"] == K]
                shape_rows.sort(key=lambda r: r["gpu_id"])
                tflops_vals = [r["tflops"] for r in shape_rows]
                mean_t = sum(tflops_vals) / len(tflops_vals)
                min_t = min(tflops_vals)
                max_t = max(tflops_vals)
                spread_pct = (max_t - min_t) / mean_t * 100 if mean_t > 0 else 0

                print(f"\n--- {dtype_name} M={M} N={N} K={K} ---")
                header = "  " + "  ".join(f"{'GPU'+str(r['gpu_id']):>10s}" for r in shape_rows)
                values = "  " + "  ".join(f"{r['tflops']:>10.1f}" for r in shape_rows)
                print(header)
                print(values + "  TFLOPS")
                flag = " <<<" if spread_pct > 5 else ""
                print(f"  mean={mean_t:.1f}  min={min_t:.1f}  max={max_t:.1f}  spread={spread_pct:.1f}%{flag}")

                # CSV rows
                for r in shape_rows:
                    csv_results.append({
                        "test_type": "matmul_per_gpu",
                        "dtype": dtype_name,
                        "M": M, "N": N, "K": K,
                        "size_bytes": "",
                        "time_ms": f"{r['time_ms']:.4f}",
                        "tflops": f"{r['tflops']:.2f}",
                        "busbw_gbps": "",
                        "algobw_gbps": "",
                        **sysinfo,
                        "gpu_id": r["gpu_id"],
                        "mode": "matmul_per_gpu",
                        "iters": args.iters,
                        "warmup": args.warmup,
                    })

    return csv_results


# ---------------------------------------------------------------------------
# AllReduce benchmark
# ---------------------------------------------------------------------------

MB = 1024 * 1024
GB = 1024 * MB
ALLREDUCE_SIZES = [
    (1 * MB, "1MB"),
    (4 * MB, "4MB"),
    (32 * MB, "32MB"),
    (128 * MB, "128MB"),
    (256 * MB, "256MB"),
    (512 * MB, "512MB"),
    (1 * GB, "1GB"),
    (2 * GB, "2GB"),
    (4 * GB, "4GB"),
    (8 * GB, "8GB"),
]


def run_allreduce_benchmark(args, sysinfo, rank, world_size, local_rank):
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    dtype = torch.bfloat16
    elem_size = 2  # bf16 = 2 bytes

    results = []

    if rank == 0:
        print(f"\n{'Size':>10s} {'Time(us)':>12s} {'BusBW(GB/s)':>14s} {'AlgoBW(GB/s)':>14s}")

    ar_iters = args.ar_iters if args.ar_iters else args.iters
    ar_warmup = args.warmup

    for nbytes, label in ALLREDUCE_SIZES:
        numel = nbytes // elem_size
        buf = torch.randn(numel, dtype=dtype, device=device)

        # synchronize all ranks
        dist.barrier()

        # warmup
        for _ in range(ar_warmup):
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(device)

        # timed iterations
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        dist.barrier()
        start.record()
        for _ in range(ar_iters):
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        end.record()
        torch.cuda.synchronize(device)

        elapsed_ms = start.elapsed_time(end)
        avg_ms = elapsed_ms / ar_iters
        avg_s = avg_ms / 1000.0
        avg_us = avg_ms * 1000.0

        algobw = nbytes / avg_s / 1e9
        busbw = nbytes * 2.0 * (world_size - 1) / world_size / avg_s / 1e9

        if rank == 0:
            print(f"  {label:>8s} {avg_us:>12.1f} {busbw:>14.2f} {algobw:>14.2f}")
            results.append({
                "test_type": "allreduce",
                "dtype": "bfloat16",
                "M": "", "N": "", "K": "",
                "size_bytes": nbytes,
                "time_ms": f"{avg_ms:.4f}",
                "tflops": "",
                "busbw_gbps": f"{busbw:.2f}",
                "algobw_gbps": f"{algobw:.2f}",
                **sysinfo,
                "mode": "allreduce",
                "iters": ar_iters,
                "warmup": ar_warmup,
            })

        del buf
        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

# Dtype display order and short names (highest perf first)
DTYPE_DISPLAY = [
    ("nvfp4", "NVFP4"),
    ("mxfp8", "MXFP8"),
    ("fp8_e4m3", "FP8-e4m3"),
    ("bfloat16", "BF16"),
    ("float16", "FP16"),
    ("tf32", "TF32"),
]


def print_summary(matmul_results, allreduce_results, sysinfo):
    print("\n" + "=" * 60)
    print("=== Machine Summary ===")
    print("=" * 60)
    print_sysinfo(sysinfo)
    print("-" * 50)

    if matmul_results:
        print("  Tensor Core Peak TFLOPS:")
        for dtype_name, short in DTYPE_DISPLAY:
            dtype_rows = [r for r in matmul_results if r["dtype"] == dtype_name]
            if dtype_rows:
                peak = max(float(r["tflops"]) for r in dtype_rows)
                print(f"    {short:>12s}:  {peak:>10.2f}")

        print()

        # Decode M=1 average (bf16 baseline)
        decode_rows = [
            r for r in matmul_results
            if r["dtype"] == "bfloat16" and int(r["M"]) == 1
        ]
        if decode_rows:
            avg_tflops = sum(float(r["tflops"]) for r in decode_rows) / len(decode_rows)
            avg_lat = sum(float(r["time_ms"]) for r in decode_rows) / len(decode_rows)
            print(f"  Decode  (M=1,bf16)    avg TFLOPS: {avg_tflops:>8.2f}  | latency: {avg_lat:.4f} ms")

        # Prefill M=128 average (bf16 baseline)
        prefill_rows = [
            r for r in matmul_results
            if r["dtype"] == "bfloat16" and int(r["M"]) == 128
        ]
        if prefill_rows:
            avg_tflops = sum(float(r["tflops"]) for r in prefill_rows) / len(prefill_rows)
            avg_lat = sum(float(r["time_ms"]) for r in prefill_rows) / len(prefill_rows)
            print(f"  Prefill (M=128,bf16)  avg TFLOPS: {avg_tflops:>8.2f}  | latency: {avg_lat:.4f} ms")

        # FP8/FP4 decode/prefill if available
        for dtype_tag, dtype_short in [("fp8_e4m3", "fp8"), ("nvfp4", "nvfp4")]:
            for label, m_val in [("Decode", 1), ("Prefill", 128)]:
                rows = [
                    r for r in matmul_results
                    if r["dtype"] == dtype_tag and r["M"] != "" and int(r["M"]) == m_val
                ]
                if rows:
                    avg_tflops = sum(float(r["tflops"]) for r in rows) / len(rows)
                    avg_lat = sum(float(r["time_ms"]) for r in rows) / len(rows)
                    pad = " " if label == "Decode" else ""
                    print(f"  {label}{pad} (M={m_val},{dtype_short})    avg TFLOPS: {avg_tflops:>8.2f}  | latency: {avg_lat:.4f} ms")

    print("-" * 50)

    if allreduce_results:
        peak_busbw = max(float(r["busbw_gbps"]) for r in allreduce_results)
        print(f"  Peak AllReduce BusBW:    {peak_busbw:>10.2f} GB/s")

        for target_label, target_bytes in [("1MB", MB), ("128MB", 128 * MB), ("1GB", GB), ("4GB", 4 * GB), ("8GB", 8 * GB)]:
            row = [r for r in allreduce_results if int(r["size_bytes"]) == target_bytes]
            if row:
                print(f"  AllReduce {target_label:>4s} BusBW:   {float(row[0]['busbw_gbps']):>10.2f} GB/s")

    print("=" * 60)


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "test_type", "dtype", "M", "N", "K", "size_bytes",
    "time_ms", "tflops", "busbw_gbps", "algobw_gbps",
    "hostname", "gpu_name", "gpu_id", "gpu_count", "cuda_capability",
    "torch_version", "cuda_version", "nccl_version", "driver",
    "mode", "iters", "warmup", "timestamp",
]


def write_csv(results, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"\nCSV saved to: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GPU Matmul & AllReduce Calibration Suite")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--matmul-only", action="store_true", help="Run matmul benchmark only (no distributed)")
    group.add_argument("--allreduce-only", action="store_true", help="Run allreduce benchmark only")
    parser.add_argument("--iters", type=int, default=100, help="Timed iterations for matmul (default: 100)")
    parser.add_argument("--ar-iters", type=int, default=None, help="Timed iterations for allreduce (default: 50)")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations (default: 10)")
    parser.add_argument("--output", type=str, default=None, help="Path to save CSV results")
    args = parser.parse_args()

    if args.ar_iters is None:
        args.ar_iters = 50

    is_distributed = "RANK" in os.environ
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Initialize distributed if needed
    if is_distributed and not args.matmul_only:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))

    sysinfo = get_system_info()
    sysinfo["gpu_count"] = world_size if is_distributed else torch.cuda.device_count()

    all_results = []

    # --- Matmul ---
    if not args.allreduce_only:
        if rank == 0:
            print("=" * 60)
            print(f"=== Matmul Benchmark (GPU: {sysinfo['gpu_name']}) ===")
            print(f"    iters={args.iters}, warmup={args.warmup}")
            print("=" * 60)
            matmul_results = run_matmul_benchmark(args, sysinfo)
            all_results.extend(matmul_results)
        else:
            matmul_results = []

        # Other ranks wait for rank 0 to finish matmul
        if is_distributed:
            dist.barrier()
    else:
        matmul_results = []

    # --- Per-GPU Matmul (all ranks in parallel) ---
    if not args.allreduce_only and is_distributed:
        per_gpu_results = run_per_gpu_matmul(args, sysinfo, rank, local_rank, world_size)
        all_results.extend(per_gpu_results)

    # --- AllReduce ---
    if not args.matmul_only:
        if rank == 0:
            print("\n" + "=" * 60)
            print(f"=== AllReduce Benchmark ({world_size} GPUs, NCCL {sysinfo['nccl_version']}) ===")
            print(f"    iters={args.ar_iters}, warmup={args.warmup}")
            print("=" * 60)

        ar_results = run_allreduce_benchmark(args, sysinfo, rank, world_size, local_rank)
        all_results.extend(ar_results)
    else:
        ar_results = []

    # --- Summary & CSV (rank 0 only) ---
    if rank == 0:
        print_summary(matmul_results, ar_results, sysinfo)
        if args.output:
            write_csv(all_results, args.output)

    # Cleanup
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
