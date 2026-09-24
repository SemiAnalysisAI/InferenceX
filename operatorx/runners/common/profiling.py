"""Per-op kernel decomposition, run after timing so it cannot affect latency_us.

Each op is replayed under torch.profiler - kineto with CUPTI on CUDA,
rocprofiler on ROCm - and the per-kernel breakdown is attached to the result:

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

import os
import shutil
import tempfile

import torch

from operatorx.runners.common import trace

PROFILE = os.environ.get("OPERATORX_PROFILE", "1") == "1"
_ITERS = int(os.environ.get("OPERATORX_PROFILE_ITERS", "3"))
_FLUSH_MB = int(os.environ.get("OPERATORX_PROFILE_FLUSH_MB", "512"))
_TRACE_DIR = os.environ.get("OPERATORX_PROFILE_TRACE_DIR") or None
_TRACE_EVERY = int(os.environ.get("OPERATORX_PROFILE_TRACE_EVERY", "200"))
_MARKERS = os.environ.get("OPERATORX_PROFILE_MARKERS", "") == "1"
# name of the kernel the int8 zero_() flush dispatches
_FLUSH_KERNEL_MARKER = "FillFunctor"
# GPU spin enqueued after each flush so the replay's launches queue up behind it and
# the timeline shows device time, not host launch gaps; torch.cuda._sleep's kernel
_SHIELD_CYCLES = int(os.environ.get("OPERATORX_SHIELD_CYCLES", "4000000"))
_SHIELD_KERNEL_MARKER = "spin_kernel"

_counter = 0


class _Cuda:
    """CUDA and ROCm: kineto kernel events, replays delimited by the harness."""

    iters = _ITERS
    arg_fields = (
        ("grid", "grid"),
        ("block", "block"),
        ("registers per thread", "regs"),
        ("shared memory", "smem"),
        ("blocks per SM", "blocks_per_sm"),
        ("warps per SM", "warps_per_sm"),
        ("est. achieved occupancy %", "occupancy_pct"),
    )
    _flush_buf = None

    def activities(self):
        return [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]

    def experimental_config(self):
        return None

    def flush(self) -> None:
        if _FLUSH_MB <= 0:
            return
        if _Cuda._flush_buf is None:
            _Cuda._flush_buf = torch.empty(_FLUSH_MB << 20, dtype=torch.int8, device="cuda")
        _Cuda._flush_buf.zero_()

    def pre_replay(self) -> None:
        self.flush()
        if _SHIELD_CYCLES > 0:
            torch.cuda._sleep(_SHIELD_CYCLES)

    def sync(self) -> None:
        torch.cuda.synchronize()

    def stream_of(self, e):
        return (e.get("args") or {}).get("stream")

    def cat_of(self, e):
        return e["cat"]

    def name_of(self, e):
        return e.get("name", "")[:200]

    def harness_events(self, events: list[dict]) -> tuple[set[int], set[int]]:
        """(flush, spin) event ids among the sorted device events. The op itself may launch
        fill kernels, so the flush is identified by position: the last fill before each spin
        (the loop issues flush, spin, replay). Without a spin, every fill counts as a flush."""
        flush, spin, last_fill = set(), set(), None
        for i, e in enumerate(events):
            name = e.get("name", "")
            if _SHIELD_KERNEL_MARKER in name:
                spin.add(i)
                if last_fill is not None:
                    flush.add(last_fill)
                last_fill = None
            elif _FLUSH_MB > 0 and _FLUSH_KERNEL_MARKER in name:
                last_fill = i
                if _SHIELD_CYCLES <= 0:
                    flush.add(i)
        return flush, spin

    def decompose(self, events: list[dict], iters: int) -> tuple[list[dict], list[list[dict]], int]:
        device = sorted((e for e in events
                         if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")),
                        key=trace.ts)
        flush, spin = self.harness_events(device)
        kernels = [e for i, e in enumerate(device) if i not in flush and i not in spin]
        replays = None
        if _FLUSH_MB > 0:
            bounds = spin if spin else flush
            replays, cur = [], []
            for i, e in enumerate(device):
                if i in bounds:
                    if cur:
                        replays.append(cur)
                    cur = []
                elif i not in flush and i not in spin:
                    cur.append(e)
            if cur:
                replays.append(cur)
        return kernels, (replays or []), len(flush)


def _platform():
    """The capture for the device in front of us: CUDA and ROCm share one."""
    return _Cuda()


def _markers_pass(kernel_fn, plat) -> dict:
    marker = f"opx{_counter:06d}"
    plat.flush()  # outside the range so the flush is not attributed
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
    plat = _platform()
    iters = plat.iters
    if _MARKERS:
        try:
            return _markers_pass(kernel_fn, plat)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"[:200], "op_index": _counter}

    try:
        with torch.profiler.profile(activities=plat.activities(),
                                    experimental_config=plat.experimental_config()) as prof:
            for _ in range(iters):
                plat.pre_replay()
                kernel_fn()
            plat.sync()
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"[:200]}

    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        prof.export_chrome_trace(path)
        events = trace.load_events(path)
    except Exception as e:
        os.unlink(path)
        return {"error": f"trace: {type(e).__name__}: {e}"[:200]}

    kernels, replays, flush_excluded = plat.decompose(events, iters)
    out = trace.aggregate(kernels, iters, plat.arg_fields, plat.cat_of, plat.name_of)
    summary = {"iters": iters, "kernels": out,
               "gpu_us_per_call": round(sum(k["us_per_call"] for k in out), 3),
               "op_index": _counter}
    if flush_excluded:
        summary["flush_kernels_excluded"] = flush_excluded
    stats = trace.replay_stats(replays, plat.stream_of)
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
