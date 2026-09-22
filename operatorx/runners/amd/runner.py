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
_TELEMETRY = None


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
        # ROCm reports the PER-XCD slice (4 MB on gfx950), not the aggregate
        # cache hierarchy; size the flush to cover all of it (override with
        # OPERATORX_FLUSH_MB).
        size = max(size, int(os.environ.get("OPERATORX_FLUSH_MB", "512")) << 20)
        _L2_BUF[device] = torch.empty(size, dtype=torch.int8, device="cuda")
    ctx = impl.prepare(op)

    def _time_once(sleep_s: float) -> list[float]:
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
        if sleep_s <= 0.0:
            if _SHIELD_CYCLES > 0:
                torch.cuda._sleep(_SHIELD_CYCLES)
            for start, end in zip(starts, ends):
                _L2_BUF[device].zero_()
                start.record()
                impl.kernel(ctx)
                end.record()
            torch.cuda.synchronize()
        else:
            # throttle retry: space iterations so the power budget recovers;
            # per-iteration sync forces a per-iteration mini-shield
            for start, end in zip(starts, ends):
                time.sleep(sleep_s)
                if _SHIELD_CYCLES > 0:
                    torch.cuda._sleep(max(_SHIELD_CYCLES // 4, 500000))
                _L2_BUF[device].zero_()
                start.record()
                impl.kernel(ctx)
                end.record()
                torch.cuda.synchronize()
        return sorted(start.elapsed_time(end) * 1000.0
                      for start, end in zip(starts, ends))

    from operatorx.runners import telemetry as _telemetry_mod
    global _TELEMETRY
    if _TELEMETRY is None:
        _TELEMETRY = _telemetry_mod.get_provider("amd", device)
    retries = int(os.environ.get("OPERATORX_THROTTLE_RETRIES", "3"))
    sleep_base_ms = float(os.environ.get("OPERATORX_RETRY_SLEEP_MS", "2"))
    sleep_s = 0.0
    best = None
    attempts = 0
    for attempt in range(1 + max(retries, 0)):
        attempts += 1
        _TELEMETRY.start()
        times = _time_once(sleep_s)
        rep = _TELEMETRY.stop()
        median_us = times[_ITERS // 2]
        rep.dump(f"{op.type}/{op.backend}/attempt{attempt}")
        if best is None or median_us < best[0]:
            best = (median_us, rep, sleep_s)
        if not rep.capped:
            break
        sleep_s = (sleep_base_ms * (2 ** attempt)) / 1e3

    median_us, rep, used_sleep = best
    if _COOLDOWN_RATIO > 0.0:
        time.sleep(min(median_us * 1e-6 * (_ITERS + _WARMUP) * _COOLDOWN_RATIO,
                       _COOLDOWN_MAX_S))
    metrics = {"latency_us": median_us}
    telemetry_summary = rep.summary()
    telemetry_summary["attempts"] = attempts
    telemetry_summary["inter_kernel_sleep_ms"] = used_sleep * 1e3
    metrics["telemetry"] = telemetry_summary
    prof = profiling.profile_op(lambda: impl.kernel(ctx))
    if prof is not None:
        metrics["profile"] = prof
    return Result(op=op, metrics=metrics)
