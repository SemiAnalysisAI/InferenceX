"""Event timing of one op on one or many devices.

Per iteration: flush, rank align, start, op, end. A spin ahead of the loop lets the host
enqueue every iteration first; multi-device iterations start behind a device-side barrier
+ spin (CollectiveX _graph_align) so the cross-rank max is the op, not launch skew.
"""
from __future__ import annotations

import os
import time

import torch

from operatorx.runners.common import ranks

WARMUP = 5
ITERS = 10
_WARMUP_MIN_S = float(os.environ.get("OPERATORX_WARMUP_MIN_S", "0.025"))
_FIRST_WARMUP_S = float(os.environ.get("OPERATORX_FIRST_WARMUP_S", "1.0"))
_SHIELD_CYCLES = int(os.environ.get("OPERATORX_SHIELD_CYCLES", "100000000"))  # ~50 ms
_ALIGN_CYCLES = int(os.environ.get("OPERATORX_ALIGN_CYCLES", "10000000"))  # ~5 ms
_RETRY_SPIN_CYCLES = 2_000_000
_FIRST = True


def time_op(fn, flush, sleep_s: float) -> float:
    """Median over ITERS cold iterations of fn, in us (each iteration's slowest rank)."""
    global _FIRST
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    floor, _FIRST = (max(_WARMUP_MIN_S, _FIRST_WARMUP_S) if _FIRST else _WARMUP_MIN_S), False
    t0 = time.perf_counter()
    while ranks.any_(time.perf_counter() - t0 < floor):  # same round count on every rank
        fn()
        torch.cuda.synchronize()

    align = _ALIGN_CYCLES if ranks.world() > 1 else 0
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    if sleep_s <= 0.0:
        torch.cuda._sleep(_SHIELD_CYCLES)
    for start, end in zip(starts, ends):
        if sleep_s > 0.0:  # throttle retry
            time.sleep(sleep_s)
            torch.cuda._sleep(_RETRY_SPIN_CYCLES)
        flush()
        ranks.align(align)
        start.record()
        fn()
        end.record()
        if sleep_s > 0.0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends)]
    return sorted(ranks.iterations(times)["max"])[ITERS // 2]
