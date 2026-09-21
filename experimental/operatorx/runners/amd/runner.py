"""ROCm operator timing with HIP events exposed through torch.cuda."""

from __future__ import annotations

import os
import time
from importlib import import_module

import torch

from operatorx.runners import profiling
from operatorx.core import Op, Result, UnsupportedOpError

_L2_BUF: dict[int, torch.Tensor] = {}
_WARMUP = 5
_ITERS = 10
# Timing robustness, validated on the NVIDIA runner: a wall-clock warmup floor
# so clocks settle, a GPU spin enqueued ahead of the timed loop so the event
# brackets never include host launch gaps, and an inter-case cooldown.
_WARMUP_MIN_S = float(os.environ.get("OPERATORX_WARMUP_MIN_S", "0.025"))
_SHIELD_CYCLES = int(os.environ.get("OPERATORX_SHIELD_CYCLES", "4000000"))
_COOLDOWN_RATIO = float(os.environ.get("OPERATORX_COOLDOWN_RATIO", "4"))
_COOLDOWN_MAX_S = float(os.environ.get("OPERATORX_COOLDOWN_MAX_S", "1.0"))


def run(op: Op) -> Result:
    if op.backend not in {"torch", "aiter", "vllm"}:
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
        _L2_BUF[device] = torch.empty(size, dtype=torch.int8, device="cuda")
    ctx = impl.prepare(op)
    for _ in range(_WARMUP):
        impl.kernel(ctx)
    torch.cuda.synchronize()
    if _WARMUP_MIN_S > 0.0:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < _WARMUP_MIN_S:
            impl.kernel(ctx)
            torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(_ITERS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(_ITERS)]
    if _SHIELD_CYCLES > 0:
        torch.cuda._sleep(_SHIELD_CYCLES)
    for start, end in zip(starts, ends):
        _L2_BUF[device].zero_()
        start.record()
        impl.kernel(ctx)
        end.record()
    torch.cuda.synchronize()
    times = sorted(start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends))
    median_us = times[_ITERS // 2]
    if _COOLDOWN_RATIO > 0.0:
        time.sleep(min(median_us * 1e-6 * (_ITERS + _WARMUP) * _COOLDOWN_RATIO,
                       _COOLDOWN_MAX_S))
    metrics = {"latency_us": median_us}
    prof = profiling.profile_op(lambda: impl.kernel(ctx))
    if prof is not None:
        metrics["profile"] = prof
    return Result(op=op, metrics=metrics)
