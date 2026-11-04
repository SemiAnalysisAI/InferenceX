"""ROCm operator timing with HIP events exposed through torch.cuda."""

from __future__ import annotations

import os
import time
from importlib import import_module

import torch

from operatorx.core import Op, Result, UnsupportedOpError
from operatorx.runners.common import profiling, ranks, telemetry, timing

_COOLDOWN_RATIO = float(os.environ.get("OPERATORX_COOLDOWN_RATIO", "4"))
_COOLDOWN_MAX_S = float(os.environ.get("OPERATORX_COOLDOWN_MAX_S", "1.0"))
# ROCm reports only the per-XCD L2 slice: size the flush buffer for the whole hierarchy
_FLUSH_MB = int(os.environ.get("OPERATORX_FLUSH_MB", "512"))

_L2_BUF: dict[int, torch.Tensor] = {}


def run(op: Op) -> Result:
    if op.backend not in ("torch", "vllm"):
        raise UnsupportedOpError(f"unknown AMD backend: {op.backend}")
    backend = import_module(f"operatorx.runners.amd.backends.{op.backend}")
    impl = next((item for item in backend.IMPLS if item.op_type == op.type), None)
    if impl is None:
        raise UnsupportedOpError(f"amd/{op.backend} has no impl for {op.type!r}")
    if not torch.version.hip:
        raise RuntimeError("AMD measurements require a ROCm PyTorch build")
    device = torch.cuda.current_device()
    if device not in _L2_BUF:
        size = torch.cuda.get_device_properties(device).L2_cache_size
        if size <= 0:
            raise RuntimeError("ROCm did not report a positive L2 cache size")
        size = max(size, _FLUSH_MB << 20)
        _L2_BUF[device] = torch.empty(size, dtype=torch.int8, device="cuda")
    ctx = impl.prepare(op)

    fn, cuda_graph = impl.launcher(ctx) if impl.launcher else ((lambda: impl.kernel(ctx)), False)
    median_us, telem = telemetry.measure(
        op, lambda sleep_s: timing.time_op(fn, _L2_BUF[device].zero_, sleep_s))

    if _COOLDOWN_RATIO > 0.0:
        time.sleep(min(median_us * 1e-6 * (timing.ITERS + timing.WARMUP) * _COOLDOWN_RATIO,
                       _COOLDOWN_MAX_S))
    metrics = {"latency_us": median_us, "cuda_graph": cuda_graph, "telemetry": telem}
    if isinstance(ctx, dict) and ctx.get("meta"):
        metrics["backend_meta"] = ctx["meta"]
    prof = profiling.profile_op(fn)
    if prof is not None:
        metrics["profile"] = prof
    per_rank = ranks.summary(metrics)  # every rank's timeline and telemetry, on rank 0
    if per_rank is not None:
        metrics["ranks"] = per_rank
        if per_rank["capped_ranks"] and metrics.get("telemetry"):
            metrics["telemetry"]["capped"] = True  # any rank capped caps the op
    return Result(op=op, metrics=metrics)
