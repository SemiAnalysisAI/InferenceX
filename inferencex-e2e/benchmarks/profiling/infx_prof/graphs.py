"""CUDA graph ordinals and stream markers."""

import itertools
import os
import threading
import time

from . import runtime

_graph_ids = itertools.count()
capture_context = None
after_capture = []
_replays = threading.local()


def replays():
    """CUDA graph replays launched from this thread so far (adapters diff it around a step)."""
    return getattr(_replays, "n", 0)


def replayed(graph, call):
    """Run call() as a replay of the captured torch graph `graph`."""
    import torch

    _replays.n = getattr(_replays, "n", 0) + 1
    gid = getattr(graph, "_infx_id", None)
    if gid is None or not runtime.profiling():
        return call()
    with torch.autograd.profiler.record_function(f"infx_graph_replay#{gid}"):
        return call()


def patch_cuda_graph(graphs_module):
    import torch

    cls = graphs_module.CUDAGraph
    if getattr(cls, "_infx_patched", False):
        return
    orig_begin, orig_end, orig_replay = cls.capture_begin, cls.capture_end, cls.replay

    def capture_begin(self, *args, **kwargs):
        try:
            gid = next(_graph_ids)
            self._infx_id = gid
            rf = torch.autograd.profiler.record_function(f"infx_graph_capture#{gid}")
            rf.__enter__()
            self._infx_rf = rf
            context = None
            if capture_context is not None:
                try:
                    context = capture_context()
                except Exception:
                    context = None
            runtime.sink("graphs", f"pid{os.getpid()}").write({
                "graph": gid, "event": "capture_begin", "t_ns": time.time_ns(),
                "device": torch.cuda.current_device(), "context": context,
            })
        except Exception:
            runtime.write_error("capture_begin")
        return orig_begin(self, *args, **kwargs)

    def capture_end(self, *args, **kwargs):
        try:
            return orig_end(self, *args, **kwargs)
        finally:
            try:
                rf = getattr(self, "_infx_rf", None)
                if rf is not None:
                    rf.__exit__(None, None, None)
                    self._infx_rf = None
                runtime.sink("graphs", f"pid{os.getpid()}").write({
                    "graph": getattr(self, "_infx_id", None), "event": "capture_end", "t_ns": time.time_ns(),
                })
            except Exception:
                runtime.write_error("capture_end")
            for fn in list(after_capture):
                try:
                    fn()
                except Exception:
                    runtime.write_error("after capture")

    def replay(self, *args, **kwargs):
        return replayed(self, lambda: orig_replay(self, *args, **kwargs))

    cls.capture_begin = capture_begin
    cls.capture_end = capture_end
    cls.replay = replay
    cls._infx_patched = True


def patch_stream_context(cuda_module):
    """Mark `with torch.cuda.stream(s)` blocks: infx_stream#<handle>."""
    import torch

    cls = getattr(cuda_module, "StreamContext", None)
    if cls is None or getattr(cls, "_infx_patched", False):
        return
    orig_enter, orig_exit = cls.__enter__, cls.__exit__

    def __enter__(self):
        out = orig_enter(self)
        stream = getattr(self, "stream", None)
        if stream is not None and runtime.profiling():
            try:
                rf = torch.autograd.profiler.record_function(f"infx_stream#{int(stream.cuda_stream)}")
                rf.__enter__()
                self._infx_rf = rf
            except Exception:
                pass
        return out

    def __exit__(self, *exc):
        rf = getattr(self, "_infx_rf", None)
        if rf is not None:
            self._infx_rf = None
            try:
                rf.__exit__(None, None, None)
            except Exception:
                pass
        return orig_exit(self, *exc)

    cls.__enter__, cls.__exit__ = __enter__, __exit__
    cls._infx_patched = True


def install(registry):
    registry.on("torch.cuda.graphs", patch_cuda_graph)
    registry.on("torch.cuda", patch_stream_context)
