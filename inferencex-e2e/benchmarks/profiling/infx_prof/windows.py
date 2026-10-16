"""Profiler windows and the graph-capture profile."""

import json
import os
import time

from . import graphs, modules, routing, runtime

CAPTURE_GRAPHS_PER_FILE = int(os.environ.get("INFX_PROF_CAPTURE_GRAPHS_PER_FILE") or 0)


def log_profiler_event(rank, event):
    try:
        runtime.sink("profiler", rank).write({"event": event, "t_ns": time.time_ns()})
    except Exception:
        runtime.write_error(f"profiler {event} log")


def window_opened(rank):
    """Module markers on for an open window."""
    log_profiler_event(rank, "start")
    try:
        modules.enable()
    except Exception:
        runtime.write_error("enable module markers")


def window_closing(rank):
    """Before the profiler stops: log the stop, write the window's routing, markers off."""
    log_profiler_event(rank, "stop")
    try:
        routing.flush()
    except Exception:
        runtime.write_error("routing flush")
    try:
        modules.disable()
    except Exception:
        runtime.write_error("disable module markers")


def _quiesce():
    """Wait for the device's queued work before the profiler starts or stops."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        runtime.write_error("synchronize")


def _new_profiler():
    from torch.profiler import ProfilerActivity, profile

    return profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True,
                   with_stack=False)


class FileWindows:
    """Torch profiler windows driven by control/windows.jsonl, opened and closed at step boundaries."""

    def __init__(self):
        self.path = os.path.join(runtime.DIR, "control", "windows.jsonl")
        self.offset = 0
        self.pending = []
        self.stops = set()
        self.active = None
        self.mtime = None

    def _poll(self):
        try:
            st = os.stat(self.path)
            mtime = (st.st_mtime_ns, st.st_size)
        except OSError:
            return
        if mtime == self.mtime:
            return
        self.mtime = mtime
        with open(self.path) as f:
            f.seek(self.offset)
            while line := f.readline():
                if not line.endswith("\n"):
                    break
                self.offset = f.tell()
                request = json.loads(line)
                if "stop" in request:
                    self.stops.add(request["stop"])
                else:
                    self.pending.append(request)

    def on_step_begin(self, rank, step):
        try:
            self._poll()
            if self.active is not None:
                window, _, start, steps = self.active
                if step >= start + steps or window in self.stops:
                    self._stop(rank)
            if self.active is None and self.pending:
                request = self.pending[0]
                if step >= request.get("start_at", {}).get(rank, 0):
                    self.pending.pop(0)
                    if request["window"] not in self.stops:
                        self._start(rank, request, step)
        except Exception:
            runtime.write_error("window control")

    def _start(self, rank, request, step):
        prof = _new_profiler()
        _quiesce()
        prof.start()
        self.active = (request["window"], prof, step, int(request["steps"]))
        window_opened(rank)

    def _stop(self, rank):
        window, prof, _, _ = self.active
        self.active = None
        window_closing(rank)
        _quiesce()
        prof.stop()
        path = os.path.join(runtime.DIR, "torch", f"{rank}.{time.time_ns()}.pt.trace.json.gz")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        prof.export_chrome_trace(path)

    def close(self, rank):
        if self.active is not None:
            try:
                self._stop(rank)
            except Exception:
                runtime.write_error("window close")


_retired = []


class capture_session:
    """Profile CUDA graph capture on capture ranks: capture/<rank>[.<n>].pt.trace.json.gz."""

    def __init__(self, rank):
        self.rank = rank
        self.prof = None
        self.graphs = 0

    def __enter__(self):
        if not runtime.is_capture_rank(self.rank) or runtime.profiling():
            return self
        try:
            self.prof = _new_profiler()
            self.prof.__enter__()
            modules.enable()
            graphs.after_capture.append(self._graph_captured)
        except Exception:
            runtime.write_error("capture profiler start")
            self.prof = None
        return self

    def _graph_captured(self):
        """Every CAPTURE_GRAPHS_PER_FILE graphs, export this trace and continue in a new one."""
        self.graphs += 1
        if self.prof is None or CAPTURE_GRAPHS_PER_FILE <= 0 or self.graphs % CAPTURE_GRAPHS_PER_FILE:
            return
        self._export()
        _retired.append(self.prof)
        self.prof = _new_profiler()
        self.prof.__enter__()

    def _export(self):
        try:
            self.prof.__exit__(None, None, None)
            base = os.path.join(runtime.DIR, "capture", self.rank)
            path, n = f"{base}.pt.trace.json.gz", 1
            while os.path.exists(path):
                path, n = f"{base}.{n}.pt.trace.json.gz", n + 1
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.prof.export_chrome_trace(path)
        except Exception:
            runtime.write_error("capture profiler export")

    def __exit__(self, *exc):
        if self.prof is not None:
            try:
                graphs.after_capture.remove(self._graph_captured)
            except ValueError:
                pass
            modules.disable()
            self._export()
            self.prof = None
        runtime.flush_all()
        return False
