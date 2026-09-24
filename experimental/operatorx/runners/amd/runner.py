"""ROCm operator timing with HIP events exposed through torch.cuda."""

from __future__ import annotations

import os
import time
from importlib import import_module

import torch

from operatorx.core import Op, Result, UnsupportedOpError
from operatorx.runners.common import profiling, telemetry

_WARMUP = 5
_ITERS = 10
# Same timing protocol as the NVIDIA runner: wall-clock warmup floor, GPU
# spin ahead of the timed loop, per-iteration cache flush, inter-op cooldown.
_WARMUP_MIN_S = float(os.environ.get("OPERATORX_WARMUP_MIN_S", "0.025"))
# The first op of a process starts from an idle GPU; give it a longer ramp.
_FIRST_WARMUP_S = float(os.environ.get("OPERATORX_FIRST_WARMUP_S", "1.0"))
_FIRST = True
_SHIELD_CYCLES = int(os.environ.get("OPERATORX_SHIELD_CYCLES", "4000000"))
_COOLDOWN_RATIO = float(os.environ.get("OPERATORX_COOLDOWN_RATIO", "4"))
_COOLDOWN_MAX_S = float(os.environ.get("OPERATORX_COOLDOWN_MAX_S", "1.0"))
# ROCm reports only the per-XCD L2 slice, so the flush buffer is sized to
# cover the whole cache hierarchy.
_FLUSH_MB = int(os.environ.get("OPERATORX_FLUSH_MB", "512"))

_L2_BUF: dict[int, torch.Tensor] = {}


def _time_op(fn, device: int, sleep_s: float) -> float:
    """Median of _ITERS cold, event-timed iterations in us."""
    for _ in range(_WARMUP):
        fn()
    torch.cuda.synchronize()
    global _FIRST
    floor, _FIRST = (max(_WARMUP_MIN_S, _FIRST_WARMUP_S) if _FIRST else _WARMUP_MIN_S), False
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < floor:
        fn()
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
        fn()
        end.record()
        if sleep_s > 0.0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends))
    return times[_ITERS // 2]


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

    fn = impl.launcher(ctx) if impl.launcher else (lambda: impl.kernel(ctx))
    median_us, telem = telemetry.measure(
        op, lambda sleep_s: _time_op(fn, device, sleep_s))

    if _COOLDOWN_RATIO > 0.0:
        time.sleep(min(median_us * 1e-6 * (_ITERS + _WARMUP) * _COOLDOWN_RATIO,
                       _COOLDOWN_MAX_S))
    metrics = {"latency_us": median_us, "telemetry": telem}
    if isinstance(ctx, dict) and ctx.get("meta"):
        metrics["backend_meta"] = ctx["meta"]
    prof = profiling.profile_op(fn)
    if prof is not None:
        metrics["profile"] = prof
    return Result(op=op, metrics=metrics)
