from __future__ import annotations

import os
import pkgutil
import time
from importlib import import_module

import torch

from operatorx.core import BackendImpl, Op, Result, UnsupportedOpError
from operatorx.runners.common import profiling, telemetry

_WARMUP = 5
_ITERS = 10
# Wall-clock warmup floor: iteration-count warmup alone is far shorter than
# the SM clock ramp after the inter-op cooldown.
_WARMUP_MIN_S = float(os.environ.get("OPERATORX_WARMUP_MIN_S", "0.025"))
# The first op of a process starts from an idle GPU; give it a longer ramp.
_FIRST_WARMUP_S = float(os.environ.get("OPERATORX_FIRST_WARMUP_S", "1.0"))
_FIRST = True
# GPU spin enqueued ahead of the timed loop so the CPU queues every timed
# iteration before the first one runs; event brackets then exclude host
# launch overhead.
_SHIELD_CYCLES = int(os.environ.get("OPERATORX_SHIELD_CYCLES", "4000000"))
# Idle time between ops as a multiple of the GPU-busy time, capped per op;
# keeps the duty cycle low enough that every op starts at boost clocks.
_COOLDOWN_RATIO = float(os.environ.get("OPERATORX_COOLDOWN_RATIO", "4"))
_COOLDOWN_MAX_S = float(os.environ.get("OPERATORX_COOLDOWN_MAX_S", "1.0"))

_DISPATCH: dict[tuple[str, str], BackendImpl] = {}
_L2_BUF: dict[int, torch.Tensor] = {}


def _load() -> None:
    if _DISPATCH:
        return
    from operatorx.runners.nvidia import backends
    for info in pkgutil.iter_modules(backends.__path__):
        if info.name.startswith("_"):
            continue
        try:
            mod = import_module(f"operatorx.runners.nvidia.backends.{info.name}")
        except ImportError:
            continue
        for impl in getattr(mod, "IMPLS", []):
            _DISPATCH[(impl.op_type, info.name)] = impl


def _clear_l2() -> None:
    dev = torch.cuda.current_device()
    if dev not in _L2_BUF:
        size = torch.cuda.get_device_properties(dev).L2_cache_size
        _L2_BUF[dev] = torch.empty(size, dtype=torch.int8, device=dev)
    _L2_BUF[dev].zero_()


def _time_op(fn, sleep_s: float) -> float:
    """Median of _ITERS cold, event-timed iterations in us.

    sleep_s > 0 spaces iterations with a host sleep (throttle retry), which
    forces a sync per iteration, so each gets its own smaller shield."""
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
    if sleep_s <= 0.0:
        torch.cuda._sleep(_SHIELD_CYCLES)
    for start, end in zip(starts, ends):
        if sleep_s > 0.0:
            time.sleep(sleep_s)
            torch.cuda._sleep(max(_SHIELD_CYCLES // 4, 500000))
        _clear_l2()
        start.record()
        fn()
        end.record()
        if sleep_s > 0.0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends))
    return times[_ITERS // 2]


def run(op: Op) -> Result:
    _load()
    impl = _DISPATCH.get((op.type, op.backend))
    if impl is None:
        raise UnsupportedOpError(f"nvidia/{op.backend} has no impl for op_type={op.type!r}")
    ctx = impl.prepare(op)

    fn, cuda_graph = impl.launcher(ctx) if impl.launcher else ((lambda: impl.kernel(ctx)), False)
    median_us, telem = telemetry.measure(op, lambda sleep_s: _time_op(fn, sleep_s))

    if _COOLDOWN_RATIO > 0.0:
        busy_s = median_us * 1e-6 * (_ITERS + _WARMUP)
        time.sleep(min(busy_s * _COOLDOWN_RATIO, _COOLDOWN_MAX_S))

    metrics = {"latency_us": median_us, "cuda_graph": cuda_graph, "telemetry": telem}
    if isinstance(ctx, dict) and ctx.get("meta"):
        metrics["backend_meta"] = ctx["meta"]
    prof = profiling.profile_op(fn)
    if prof is not None:
        metrics["profile"] = prof
    return Result(op=op, metrics=metrics)
