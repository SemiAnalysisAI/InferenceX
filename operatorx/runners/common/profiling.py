"""Per-op kernel decomposition, run after timing so it cannot affect latency_us.

Each op is replayed under torch.profiler (kineto: CUPTI on CUDA, rocprofiler
on ROCm) with the cache hierarchy flushed before every replay, and the
per-kernel breakdown is attached to the result:

  metrics["profile"] = {
    "iters": N, "op_index": i,
    "kernels": [{"name", "cat", "count_per_call", "us_per_call",
                 "grid", "block", "regs", "smem", "blocks_per_sm",
                 "warps_per_sm", "occupancy_pct"}, ...],   # sorted by time
    "gpu_us_per_call": ...,          # sum of kernel durations
    "span_us", "busy_us", "gap_us", "overlap_us", "streams",
                                     # of the median-span replay: first start -> last end,
                                     # union of kernel time across streams, span - busy,
                                     # sum of durations - busy (concurrent kernels)
    "timeline": [{"name", "stream", "start_us", "dur_us"}, ...],  # that replay, op-relative
    "flush_kernels_excluded": ...,
    "trace": ...,                    # when a chrome trace was kept
  }

Launch-config fields are present only where the platform reports them.

OPERATORX_PROFILE=0             disable
OPERATORX_PROFILE_ITERS         replays per op (default 3)
OPERATORX_PROFILE_FLUSH_MB      flush size before each replay (default 512, 0 = warm)
OPERATORX_PROFILE_TRACE_DIR     keep chrome traces here
OPERATORX_PROFILE_TRACE_EVERY   keep every Nth op's trace (default 200)
OPERATORX_PROFILE_MARKERS=1     instead of torch.profiler, wrap the replays in an
                                nvtx/roctx range "opx<op_index>" for an external
                                profiler (ncu, rocprofv3); op_index is the join key.
                                Latencies from such runs are not timing data.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile

import torch

PROFILE = os.environ.get("OPERATORX_PROFILE", "1") == "1"
_ITERS = int(os.environ.get("OPERATORX_PROFILE_ITERS", "3"))
_FLUSH_MB = int(os.environ.get("OPERATORX_PROFILE_FLUSH_MB", "512"))
_TRACE_DIR = os.environ.get("OPERATORX_PROFILE_TRACE_DIR") or None
_TRACE_EVERY = int(os.environ.get("OPERATORX_PROFILE_TRACE_EVERY", "200"))
_MARKERS = os.environ.get("OPERATORX_PROFILE_MARKERS", "") == "1"
# name of the kernel the int8 zero_() flush dispatches
_FLUSH_KERNEL_MARKER = "FillFunctor"

_ARG_FIELDS = (
    ("grid", "grid"),
    ("block", "block"),
    ("registers per thread", "regs"),
    ("shared memory", "smem"),
    ("blocks per SM", "blocks_per_sm"),
    ("warps per SM", "warps_per_sm"),
    ("est. achieved occupancy %", "occupancy_pct"),
)

_TIMELINE_MAX = 64

_flush_buf = None
_counter = 0


def _replay_stats(events: list[dict]) -> dict | None:
    """Timing structure of the replays; the per-replay flush kernel separates them."""
    if _FLUSH_MB <= 0:
        return None
    replays, cur = [], []
    for e in sorted(events, key=lambda e: float(e.get("ts", 0.0))):
        if _FLUSH_KERNEL_MARKER in e.get("name", ""):
            if cur:
                replays.append(cur)
            cur = []
        else:
            cur.append(e)
    if cur:
        replays.append(cur)
    if not replays:
        return None
    rows = []
    for r in replays:
        iv = sorted((float(e["ts"]), float(e["ts"]) + float(e.get("dur", 0.0))) for e in r)
        busy, end = 0.0, None
        for a, b in iv:
            if end is None or a > end:
                busy += b - a
                end = b
            elif b > end:
                busy += b - end
                end = b
        span = max(b for _, b in iv) - iv[0][0]
        total = sum(b - a for a, b in iv)
        rows.append((span, busy, span - busy, total - busy,
                     len({(e.get("args") or {}).get("stream") for e in r})))
    # all fields from one replay: the median by span
    i = sorted(range(len(rows)), key=lambda j: rows[j][0])[len(rows) // 2]
    span, busy, gap, overlap, streams = rows[i]
    t0 = min(float(e["ts"]) for e in replays[i])
    timeline = [{"name": e.get("name", "")[:120], "stream": (e.get("args") or {}).get("stream"),
                 "start_us": round(float(e["ts"]) - t0, 3), "dur_us": round(float(e.get("dur", 0.0)), 3)}
                for e in sorted(replays[i], key=lambda e: float(e["ts"]))[:_TIMELINE_MAX]]
    return {"span_us": round(span, 3), "busy_us": round(busy, 3), "gap_us": round(gap, 3),
            "overlap_us": round(overlap, 3), "streams": streams, "timeline": timeline}


def _flush_caches() -> None:
    global _flush_buf
    if _FLUSH_MB <= 0:
        return
    if _flush_buf is None:
        _flush_buf = torch.empty(_FLUSH_MB << 20, dtype=torch.int8, device="cuda")
    _flush_buf.zero_()


def _markers_pass(kernel_fn) -> dict:
    marker = f"opx{_counter:06d}"
    _flush_caches()  # outside the range so the flush is not attributed
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(marker)
    try:
        for _ in range(_ITERS):
            kernel_fn()
        torch.cuda.synchronize()
    finally:
        torch.cuda.nvtx.range_pop()
    return {"iters": _ITERS, "op_index": _counter, "marker": marker}


def profile_op(kernel_fn) -> dict | None:
    global _counter
    if not PROFILE:
        return None
    _counter += 1
    if _MARKERS:
        try:
            return _markers_pass(kernel_fn)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"[:200], "op_index": _counter}

    acts = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    try:
        with torch.profiler.profile(activities=acts) as prof:
            for _ in range(_ITERS):
                _flush_caches()
                kernel_fn()
            torch.cuda.synchronize()
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"[:200]}

    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        prof.export_chrome_trace(path)
        events = json.load(open(path)).get("traceEvents", [])
    except Exception as e:
        os.unlink(path)
        return {"error": f"trace: {type(e).__name__}: {e}"[:200]}

    kernels: dict[str, dict] = {}
    flush_excluded = 0
    gpu_events = [e for e in events
                  if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    for e in gpu_events:
        name = e.get("name", "")[:200]
        if _FLUSH_MB > 0 and _FLUSH_KERNEL_MARKER in name:
            flush_excluded += 1
            continue
        a = e.get("args") or {}
        k = kernels.setdefault(name, {"name": name, "cat": e["cat"], "count": 0, "total_us": 0.0})
        k["count"] += 1
        k["total_us"] += float(e.get("dur", 0.0))
        for src, dst in _ARG_FIELDS:
            if src in a and dst not in k:
                k[dst] = a[src]

    out = []
    for k in kernels.values():
        k["count_per_call"] = round(k.pop("count") / _ITERS, 2)
        k["us_per_call"] = round(k.pop("total_us") / _ITERS, 3)
        out.append(k)
    out.sort(key=lambda x: -x["us_per_call"])
    summary = {"iters": _ITERS, "kernels": out,
               "gpu_us_per_call": round(sum(k["us_per_call"] for k in out), 3),
               "op_index": _counter}
    if flush_excluded:
        summary["flush_kernels_excluded"] = flush_excluded
    stats = _replay_stats(gpu_events)
    if stats:
        summary.update(stats)

    try:
        # "== 1 % N" so that TRACE_EVERY=1 keeps every trace
        if _TRACE_DIR and _counter % _TRACE_EVERY == 1 % _TRACE_EVERY:
            os.makedirs(_TRACE_DIR, exist_ok=True)
            dest = os.path.join(_TRACE_DIR, f"op{_counter:06d}.json")
            shutil.move(path, dest)  # tmp may be on another filesystem
            summary["trace"] = dest
        else:
            os.unlink(path)
    except OSError as e:
        summary["trace_error"] = str(e)[:120]
    return summary
