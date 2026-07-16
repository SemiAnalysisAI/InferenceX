#!/usr/bin/env python3
"""Run a single (shape, dtype) sglang MoE forward inside torch.profiler so we
can see per-CUDA-kernel FLOPs and time. The dashboard reports >100% bf16 SOL
for some sglang MoE shapes and we want to understand whether the kernel is
actually doing the work the op_spec counts.

Usage (inside the sglang container):
    OPERATORX_BACKENDS=sglang OPERATORX_CLUSTER=b200_dgx_8x \
        WORLD_SIZE=8 RANK=$SLURM_PROCID LOCAL_RANK=$SLURM_LOCALID \
        torchrun --nnodes=1 --nproc-per-node=8 scripts/profile_sglang_moe.py
"""
from __future__ import annotations

import os
import sys
import torch

# Force import path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from operatorx.core import Op
from operatorx.runners.nvidia.backends import sglang as sglang_be

# The shape we're suspicious of (b300 sglang showed 231% bf16 SOL)
ARGS = {
    "num_tokens": 16384, "hidden": 4096, "intermediate": 2048,
    "num_experts": 128, "top_k": 6, "world_size": 8, "tp": 1, "ep": 8,
    "dtype_act": "bf16", "dtype_weight": "bf16",
    "expert_distribution": "uniform",
}

op = Op(type="moe_forward", args=ARGS, backend="sglang")
ctx = sglang_be._prepare_moe_forward(op)

# Warmup
for _ in range(3):
    sglang_be._kernel_moe_forward(ctx)
torch.cuda.synchronize()

# Profile a single forward
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
    record_shapes=True,
    with_flops=True,
) as prof:
    sglang_be._kernel_moe_forward(ctx)
    torch.cuda.synchronize()

if int(os.environ.get("RANK", "0")) == 0:
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=20, max_name_column_width=80))
    # Also dump aggregate flops
    total_flops = sum(getattr(evt, "flops", 0) or 0 for evt in prof.key_averages())
    total_cuda_us = sum(evt.self_cuda_time_total for evt in prof.key_averages()) / 1.0
    print()
    print(f"== aggregate ==")
    print(f"profiler flops: {total_flops/1e12:.2f} TFLOP")
    print(f"profiler cuda_us: {total_cuda_us:.1f}")
    if total_cuda_us > 0:
        print(f"profiler tflops/s: {total_flops/(total_cuda_us*1e-6)/1e12:.1f}")
    # Theoretical
    th_flops = 6 * ARGS["num_tokens"] * ARGS["top_k"] * ARGS["hidden"] * ARGS["intermediate"]
    print(f"theoretical (our formula): {th_flops/1e12:.2f} TFLOP")
