"""Chrome-trace arithmetic shared by the per-platform profilers.

Platforms differ in how they capture (CUPTI, rocprofiler)
and in how a replay is delimited, but once a replay is a list of device events
with a start, a duration and a lane, the structure of a measurement - which
kernels ran, how long the device was busy, where it was idle, what overlapped -
is the same arithmetic everywhere.
"""
from __future__ import annotations

import json
from typing import Callable

TIMELINE_MAX = 64

def load_events(path: str) -> list[dict]:
    return json.loads(open(path).read()).get("traceEvents", [])


def ts(e: dict) -> float:
    return float(e.get("ts", 0.0))


def dur(e: dict) -> float:
    return float(e.get("dur", 0.0))


def busy_us(events: list[dict]) -> float:
    """Union of the events' time, so concurrent lanes are not counted twice."""
    total, end = 0.0, None
    for a, b in sorted((ts(e), ts(e) + dur(e)) for e in events):
        if end is None or a > end:
            total += b - a
            end = b
        elif b > end:
            total += b - end
            end = b
    return total


def replay_stats(replays: list[list[dict]], stream_of: Callable[[dict], object]) -> dict | None:
    """Timing structure of the replays, reported from the median one by span.

    span is first start to last end, busy the union of device time across lanes,
    gap the device idle inside the span, and overlap the time counted twice
    because lanes ran concurrently.
    """
    rows = []
    for r in replays:
        if not r:
            continue
        busy = busy_us(r)
        span = max(ts(e) + dur(e) for e in r) - min(ts(e) for e in r)
        rows.append((span, busy, span - busy, sum(dur(e) for e in r) - busy,
                     len({stream_of(e) for e in r}), r))
    if not rows:
        return None
    span, busy, gap, overlap, streams, r = sorted(rows, key=lambda x: x[0])[len(rows) // 2]
    t0 = min(ts(e) for e in r)
    timeline = [{"name": e.get("name", "")[:120], "stream": stream_of(e),
                 "start_us": round(ts(e) - t0, 3), "dur_us": round(dur(e), 3)}
                for e in sorted(r, key=ts)[:TIMELINE_MAX]]
    # + 0.0 so a value that is zero to rounding is never reported as -0.0
    return {"span_us": round(span, 3) + 0.0, "busy_us": round(busy, 3) + 0.0,
            "gap_us": round(gap, 3) + 0.0, "overlap_us": round(overlap, 3) + 0.0,
            "streams": streams, "timeline": timeline}


def aggregate(events: list[dict], calls: int, arg_fields: tuple,
              cat_of: Callable[[dict], str], name_of: Callable[[dict], str]) -> list[dict]:
    """Per-kernel totals over all replays, divided down to one call, longest first."""
    kernels: dict[str, dict] = {}
    for e in events:
        name = name_of(e)
        a = e.get("args") or {}
        k = kernels.setdefault(name, {"name": name, "cat": cat_of(e), "count": 0, "total_us": 0.0})
        k["count"] += 1
        k["total_us"] += dur(e)
        for src, dst in arg_fields:
            if src in a and dst not in k:
                k[dst] = a[src]
    out = []
    for k in kernels.values():
        k["count_per_call"] = round(k.pop("count") / calls, 2)
        k["us_per_call"] = round(k.pop("total_us") / calls, 3)
        out.append(k)
    out.sort(key=lambda x: -x["us_per_call"])
    return out
