"""Per-op kernel profiling — part of the standard measurement methodology.

Every measured op gets, in this order:
  1. the event-timed latency (warmup + wall-clock warmup floor, a
     pre-enqueued GPU spin so event brackets exclude host launch gaps,
     a full L2 flush before every timed iteration, median of the timed
     iterations) — that is the recorded latency_us, and nothing in this
     module runs before or during it;
  2. a profiling replay under torch.profiler (CUPTI on CUDA, rocprofiler
     on ROCm — both through kineto) whose per-kernel decomposition is
     attached to the result metrics. Replays are back-to-back and hence
     warm-cache unless OPERATORX_PROFILE_FLUSH_MB is set; use them for
     kernel attribution and launch-gap analysis, never as cold timing.

Disable with OPERATORX_PROFILE=0 (timing-only runs). Latencies from
counter/marker runs must never be ingested as timing ground truth.

The per-kernel summary attached to the result metrics:

  metrics["profile"] = {
    "iters": N,                       # profiled replays
    "kernels": [                      # one entry per distinct kernel/memop,
      {"name": ..., "cat": "kernel",  #   sorted by time, per-CALL numbers
       "count_per_call": 2.0, "us_per_call": 31.2,
       "grid": [..], "block": [..], "regs": 255, "smem": 0,
       "blocks_per_sm": ..., "warps_per_sm": ..., "occupancy_pct": ...},
      ...],
    "gpu_us_per_call": ...,           # sum over kernels (compare to latency_us:
                                      #   the gap is launch/idle time)
    "trace": "traces/op000123.json",  # present when a full chrome trace was kept
  }

OPERATORX_PROFILE_TRACE_DIR   keep full chrome traces here (default: off)
OPERATORX_PROFILE_TRACE_EVERY keep every Nth op's trace (default 200)
OPERATORX_PROFILE_ITERS       replay count under the profiler (default 3)
OPERATORX_PROFILE_METRICS     comma-separated hardware-counter metrics
                              (CUDA only, CUPTI range-profiler names such
                              as dram__bytes_read.sum); adds a second
                              replay pass and attaches
                              metrics["profile"]["counters"] =
                              {kernel_name: {metric: value_per_call}}
OPERATORX_PROFILE_MARKERS     "1": skip torch.profiler and instead bracket
                              the replay in an nvtx/roctx range named
                              "opx<op_index>" so an EXTERNAL profiler
                              (e.g. rocprofv3 --pmc --marker-trace) can
                              attribute per-dispatch counters to ops; the
                              summary then carries iters/op_index/marker
                              only. Every summary carries "op_index",
                              which is also the join key for such
                              externally collected counters.

Launch-config fields depend on what the platform's kineto backend reports;
missing fields are simply absent. This is measurement-side instrumentation
only -- it runs after timing and cannot affect recorded latencies.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile

import torch

PROFILE = os.environ.get("OPERATORX_PROFILE", "1") == "1"
_TRACE_DIR = os.environ.get("OPERATORX_PROFILE_TRACE_DIR") or None
_TRACE_EVERY = int(os.environ.get("OPERATORX_PROFILE_TRACE_EVERY", "200"))
_ITERS = int(os.environ.get("OPERATORX_PROFILE_ITERS", "3"))
_METRICS = [m.strip() for m in
            os.environ.get("OPERATORX_PROFILE_METRICS", "").split(",")
            if m.strip()]
_MARKERS = os.environ.get("OPERATORX_PROFILE_MARKERS", "") == "1"
# Cache flush before the marked replay (MB; 0 = off). External profilers
# without their own cache control (rocprofv3) otherwise measure the replay
# against caches warmed by the preceding timed loop. Size it to cover the
# FULL cache hierarchy (incl. any memory-side cache), not just L2. The
# flush runs OUTSIDE the marker range so its dispatch is not attributed.
_FLUSH_MB = int(os.environ.get("OPERATORX_PROFILE_FLUSH_MB", "0"))
_FLUSH_BUF = None


def _flush_caches() -> None:
    global _FLUSH_BUF
    if _FLUSH_MB <= 0:
        return
    if _FLUSH_BUF is None:
        _FLUSH_BUF = torch.empty(_FLUSH_MB << 20, dtype=torch.int8,
                                 device="cuda")
    _FLUSH_BUF.zero_()

_ARG_FIELDS = (
    ("grid", "grid"),
    ("block", "block"),
    ("registers per thread", "regs"),
    ("shared memory", "smem"),
    ("blocks per SM", "blocks_per_sm"),
    ("warps per SM", "warps_per_sm"),
    ("est. achieved occupancy %", "occupancy_pct"),
)

_counter = 0


def _markers_pass(kernel_fn) -> dict:
    """Replay inside an nvtx/roctx range for an external profiler to catch."""
    marker = f"opx{_counter:06d}"
    _flush_caches()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(marker)
    try:
        for _ in range(_ITERS):
            kernel_fn()
        torch.cuda.synchronize()
    finally:
        torch.cuda.nvtx.range_pop()
    return {"iters": _ITERS, "op_index": _counter, "marker": marker}


def _counters_pass(kernel_fn) -> dict:
    """Replay once more under the kineto range profiler for HW counters.

    Returns {kernel_name: {metric: value_per_call}} (values summed over
    the replays, then divided by _ITERS), or {"error": ...}.
    """
    try:
        from torch._C._profiler import _ExperimentalConfig
        exp = _ExperimentalConfig(profiler_metrics=_METRICS,
                                  profiler_measure_per_kernel=True)
        acts = [torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(activities=acts,
                                    experimental_config=exp) as prof:
            for _ in range(_ITERS):
                kernel_fn()
            torch.cuda.synchronize()
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            prof.export_chrome_trace(path)
            events = json.load(open(path)).get("traceEvents", [])
        finally:
            os.unlink(path)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"[:200]}

    out: dict[str, dict] = {}
    counts: dict[str, int] = {}
    for e in events:
        a = e.get("args") or {}
        vals = {m: a[m] for m in _METRICS if m in a}
        if not vals:
            continue
        name = e.get("name", "")[:200]
        k = out.setdefault(name, {})
        counts[name] = counts.get(name, 0) + 1
        for m, v in vals.items():
            try:
                k[m] = k.get(m, 0.0) + float(v)
            except (TypeError, ValueError):
                k[m] = v
    for name, k in out.items():
        for m, v in list(k.items()):
            if isinstance(v, float):
                k[m] = v / _ITERS
        k["_ranges_per_call"] = round(counts[name] / _ITERS, 2)
    return out


def profile_op(kernel_fn) -> dict | None:
    """Replay kernel_fn under torch.profiler; return the summary dict."""
    global _counter
    if not PROFILE:
        return None
    _counter += 1
    if _MARKERS:
        try:
            return _markers_pass(kernel_fn)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"[:200],
                    "op_index": _counter}
    acts = [torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA]
    try:
        with torch.profiler.profile(activities=acts) as prof:
            for _ in range(_ITERS):
                # optional per-replay cache flush (OPERATORX_PROFILE_FLUSH_MB):
                # makes each replay cold; the flush shows up in the trace as a
                # memset-style kernel and is excluded by name downstream
                _flush_caches()
                kernel_fn()
            torch.cuda.synchronize()
    except Exception as e:  # profiling must never fail the measurement
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
    for e in events:
        if e.get("ph") != "X" or e.get("cat") not in (
                "kernel", "gpu_memcpy", "gpu_memset"):
            continue
        name = e.get("name", "")[:200]
        a = e.get("args") or {}
        k = kernels.setdefault(name, {"name": name, "cat": e["cat"],
                                      "count": 0, "total_us": 0.0})
        k["count"] += 1
        k["total_us"] += float(e.get("dur", 0.0))
        for src, dst in _ARG_FIELDS:
            if src in a and dst not in k:
                k[dst] = a[src]

    out = []
    gpu_us = 0.0
    for k in kernels.values():
        k["count_per_call"] = round(k.pop("count") / _ITERS, 2)
        k["us_per_call"] = round(k.pop("total_us") / _ITERS, 3)
        gpu_us += k["us_per_call"]
        out.append(k)
    out.sort(key=lambda x: -x["us_per_call"])

    summary = {"iters": _ITERS, "kernels": out,
               "gpu_us_per_call": round(gpu_us, 3),
               "op_index": _counter}
    if _METRICS:
        summary["counters"] = _counters_pass(kernel_fn)
    try:
        # "== 1 % N" (not "== 1") so TRACE_EVERY=1 keeps every trace
        if _TRACE_DIR and (_counter % _TRACE_EVERY) == (1 % _TRACE_EVERY):
            os.makedirs(_TRACE_DIR, exist_ok=True)
            dest = os.path.join(_TRACE_DIR, f"op{_counter:06d}.json")
            # shutil.move, not os.replace: the temp file lives on a different
            # filesystem than the (typically bind-mounted) trace dir.
            shutil.move(path, dest)
            summary["trace"] = dest
        else:
            os.unlink(path)
    except OSError as e:
        summary["trace_error"] = str(e)[:120]
    return summary
