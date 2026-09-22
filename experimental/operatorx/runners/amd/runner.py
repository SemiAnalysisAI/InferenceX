"""ROCm operator timing with HIP events exposed through torch.cuda."""

from __future__ import annotations

import os
import time
from importlib import import_module

import torch

from operatorx.core import Op, Result, UnsupportedOpError
from operatorx.runners import profiling, telemetry

_WARMUP = 5
_ITERS = 10
# Same timing protocol as the NVIDIA runner: wall-clock warmup floor, GPU
# spin ahead of the timed loop, per-iteration cache flush, inter-op cooldown.
_WARMUP_MIN_S = float(os.environ.get("OPERATORX_WARMUP_MIN_S", "0.025"))
_SHIELD_CYCLES = int(os.environ.get("OPERATORX_SHIELD_CYCLES", "4000000"))
_COOLDOWN_RATIO = float(os.environ.get("OPERATORX_COOLDOWN_RATIO", "4"))
_COOLDOWN_MAX_S = float(os.environ.get("OPERATORX_COOLDOWN_MAX_S", "1.0"))
# ROCm reports only the per-XCD L2 slice, so the flush buffer is sized to
# cover the whole cache hierarchy.
_FLUSH_MB = int(os.environ.get("OPERATORX_FLUSH_MB", "512"))

_L2_BUF: dict[int, torch.Tensor] = {}


def _time_op(impl, ctx, device: int, sleep_s: float) -> float:
    """Median of _ITERS cold, event-timed iterations in us."""
    for _ in range(_WARMUP):
        impl.kernel(ctx)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < _WARMUP_MIN_S:
        impl.kernel(ctx)
        torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(_ITERS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(_ITERS)]
    if sleep_s <= 0.0 and _SHIELD_CYCLES > 0:
        torch.cuda._sleep(_SHIELD_CYCLES)
    for start, end in zip(starts, ends):
        if sleep_s > 0.0:
            time.sleep(sleep_s)
            torch.cuda._sleep(max(_SHIELD_CYCLES // 4, 500000))
        _L2_BUF[device].zero_()
        start.record()
        impl.kernel(ctx)
        end.record()
        if sleep_s > 0.0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends))
    return times[_ITERS // 2]


def run(op: Op) -> Result:
    if op.backend != "torch":
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

    median_us, telem = telemetry.measure(
        op, lambda sleep_s: _time_op(impl, ctx, device, sleep_s))

    if _COOLDOWN_RATIO > 0.0:
        time.sleep(min(median_us * 1e-6 * (_ITERS + _WARMUP) * _COOLDOWN_RATIO,
                       _COOLDOWN_MAX_S))
    metrics = {"latency_us": median_us, "telemetry": telem}
    prof = profiling.profile_op(lambda: impl.kernel(ctx))
    if prof is not None:
        metrics["profile"] = prof
    return Result(op=op, metrics=metrics)
