from __future__ import annotations

import os
import time
from importlib import import_module

import torch

from operatorx.runners import profiling
from operatorx.core import BackendImpl, Op, Result, UnsupportedOpError

# Backends are DISCOVERED, not hardcoded: every module under
# operatorx/runners/nvidia/backends/ is a backend (matches main.py), which
# also covers upstream's list (torch/deepgemm/flashinfer/.../vllm).
def _discover() -> list[str]:
    import pkgutil
    from operatorx.runners.nvidia import backends as _pkg
    return sorted(i.name for i in pkgutil.iter_modules(_pkg.__path__)
                  if not i.name.startswith("_"))
_DISPATCH: dict[tuple[str, str], BackendImpl] = {}
_L2_BUF: dict[int, torch.Tensor] = {}


def _load() -> None:
    if _DISPATCH:
        return
    for name in _discover():
        try:
            mod = import_module(f"operatorx.runners.nvidia.backends.{name}")
        except ImportError:
            continue
        for impl in getattr(mod, "IMPLS", []):
            _DISPATCH[(impl.op_type, name)] = impl


def _clear_l2() -> None:
    """Flush L2 by writing zeros to a buffer sized to the device's L2 cache."""
    dev = torch.cuda.current_device()
    buf = _L2_BUF.get(dev)
    if buf is None:
        l2 = torch.cuda.get_device_properties(dev).L2_cache_size
        buf = torch.empty(l2, dtype=torch.int8, device=dev)
        _L2_BUF[dev] = buf
    buf.zero_()


_WARMUP = 5
# Minimum wall-clock warmup per op (seconds); see the ramp note in _time_op.
_WARMUP_MIN_S = float(os.environ.get("OPERATORX_WARMUP_MIN_S", "0.025"))
_ITERS = 10
# GPU spin cycles enqueued ahead of the timed loop (see the shield note in
# _time_op); ~4M cycles is a few ms at boost clocks.
_SHIELD_CYCLES = int(os.environ.get("OPERATORX_SHIELD_CYCLES", "4000000"))
_NUM_BUFFER_SETS = 1

# Idle time between test cases, as a multiple of the GPU-busy time just spent.
# Without it a long sweep drives average power into the board's software power
# cap (SwPowerCap) and the SM clock drops well below boost -- measurements taken
# in that state are NOT peak. Sleeping proportionally keeps the duty cycle low
# enough that every op starts at full boost clocks. Per-op sleep is capped so a
# handful of very large shapes can't stretch the sweep unbounded.
# Tune/disable with OPERATORX_COOLDOWN_RATIO (0 = off).
_COOLDOWN_RATIO = float(os.environ.get("OPERATORX_COOLDOWN_RATIO", "4"))
_COOLDOWN_MAX_S = float(os.environ.get("OPERATORX_COOLDOWN_MAX_S", "1.0"))

# Throttle-retry policy: if telemetry observes actual power/thermal capping
# (throttle reason active while the SM clock is depressed below rated boost)
# during the timed window, re-measure with increased inter-kernel sleeps so
# the power budget recovers between iterations. At most this many retries;
# then the best (lowest-median) attempt is kept and reported.
_THROTTLE_RETRIES = int(os.environ.get("OPERATORX_THROTTLE_RETRIES", "3"))
_RETRY_SLEEP_BASE_MS = float(os.environ.get("OPERATORX_RETRY_SLEEP_MS", "2"))

_TELEMETRY = None


def _telemetry():
    global _TELEMETRY
    if _TELEMETRY is None:
        from operatorx.runners import telemetry as _t
        _TELEMETRY = _t.get_provider("nvidia", torch.cuda.current_device())
    return _TELEMETRY


def _time_op(impl, ctxs, sleep_s: float) -> list[float]:
    """Warmup + timed loop; returns sorted per-iteration times in us.

    With sleep_s == 0 all iterations are enqueued behind one GPU spin
    (shield) so event brackets exclude host launch gaps. With sleep_s > 0
    (throttle retry) each iteration is spaced by a host sleep to let the
    power budget recover, which forces a sync per iteration -- so each
    iteration gets its own small shield before its start event.
    """
    for i in range(_WARMUP):
        impl.kernel(ctxs[i % _NUM_BUFFER_SETS])
    torch.cuda.synchronize()
    # Iteration-count warmup is microseconds for small ops -- far less than
    # the SM clock ramp after the inter-case cooldown idle -- so their timed
    # iterations can land mid-ramp and record isolated 2-4x outliers. Keep
    # warming until a minimum wall time has passed so clocks settle first.
    if _WARMUP_MIN_S > 0.0:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < _WARMUP_MIN_S:
            impl.kernel(ctxs[0])
            torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(_ITERS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(_ITERS)]
    if sleep_s <= 0.0:
        # Head-start shield: for microsecond kernels the start/end event pair
        # is only tight if the GPU is still busy when the CPU enqueues it --
        # otherwise the bracket times the CPU's launch path (tens of us, and
        # unbounded under driver-lock contention, e.g. a concurrent telemetry
        # poll). A few ms of enqueued GPU spin lets the CPU queue ALL timed
        # iterations before the first one starts executing.
        torch.cuda._sleep(_SHIELD_CYCLES)
        for i in range(_ITERS):
            _clear_l2()
            starts[i].record()
            impl.kernel(ctxs[i % _NUM_BUFFER_SETS])
            ends[i].record()
        torch.cuda.synchronize()
    else:
        for i in range(_ITERS):
            time.sleep(sleep_s)
            torch.cuda._sleep(max(_SHIELD_CYCLES // 4, 500000))
            _clear_l2()
            starts[i].record()
            impl.kernel(ctxs[i % _NUM_BUFFER_SETS])
            ends[i].record()
            torch.cuda.synchronize()

    return sorted(starts[i].elapsed_time(ends[i]) * 1000.0
                  for i in range(_ITERS))


def run(op: Op) -> Result:
    _load()
    impl = _DISPATCH.get((op.type, op.backend))
    if impl is None:
        raise UnsupportedOpError(
            f"nvidia/{op.backend} has no impl for op_type={op.type!r}"
        )

    ctxs = [impl.prepare(op) for _ in range(_NUM_BUFFER_SETS)]

    sleep_s = 0.0
    best = None  # (median_us, telemetry report)
    attempts = 0
    for attempt in range(1 + max(_THROTTLE_RETRIES, 0)):
        attempts += 1
        tel = _telemetry()
        tel.start()
        times = _time_op(impl, ctxs, sleep_s)
        rep = tel.stop()
        median_us = times[_ITERS // 2]
        rep.dump(f"{op.type}/{op.backend}/attempt{attempt}")
        if best is None or median_us < best[0]:
            best = (median_us, rep, sleep_s)
        if not rep.capped:
            break
        # capping observed: space kernels out and try again
        sleep_s = (_RETRY_SLEEP_BASE_MS * (2 ** attempt)) / 1e3

    median_us, rep, used_sleep = best

    # Let the board shed the power it just drew before the next test case, so
    # the next measurement also starts at boost clocks (see _COOLDOWN_RATIO).
    if _COOLDOWN_RATIO > 0.0:
        busy_s = median_us * 1e-6 * (_ITERS + _WARMUP)
        time.sleep(min(busy_s * _COOLDOWN_RATIO, _COOLDOWN_MAX_S))

    metrics = {"latency_us": median_us}
    telemetry_summary = rep.summary()
    telemetry_summary["attempts"] = attempts
    telemetry_summary["inter_kernel_sleep_ms"] = used_sleep * 1e3
    metrics["telemetry"] = telemetry_summary
    prof = profiling.profile_op(lambda: impl.kernel(ctxs[0]))
    if prof is not None:
        metrics["profile"] = prof
    return Result(op=op, metrics=metrics)
