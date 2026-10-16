"""Process-wide state shared by the core and the engine adapters."""

import json
import os
import threading
import time
import traceback

DIR = os.environ.get("INFX_PROF_DIR")


def write_error(where):
    """Append the current exception to errors/<pid>.log."""
    try:
        path = os.path.join(DIR, "errors", f"{os.getpid()}.log")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            f.write(f"--- {time.time():.3f} {where}\n{traceback.format_exc()}\n")
    except Exception:
        pass


class JsonlSink:
    """Line-buffered: engines are killed at teardown, so nothing may sit in a buffer."""

    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._f = open(path, "a", buffering=1)
        self._lock = threading.Lock()

    def write(self, record):
        line = json.dumps(record, separators=(",", ":"), default=str)
        with self._lock:
            self._f.write(line + "\n")

    def flush(self):
        with self._lock:
            self._f.flush()


_sinks = {}


def sink(kind, tag):
    """The JSONL file <kind>/<tag>.jsonl."""
    key = (kind, tag)
    if key not in _sinks:
        _sinks[key] = JsonlSink(os.path.join(DIR, kind, f"{tag}.jsonl"))
    return _sinks[key]


def flush_all():
    for s in list(_sinks.values()):
        try:
            s.flush()
        except Exception:
            pass


_notes = set()


def note_once(kind, event, detail):
    """One record per (event, detail) in <kind>/pid<pid>.jsonl, to see which paths ran."""
    if (kind, event, detail) in _notes:
        return
    _notes.add((kind, event, detail))
    try:
        sink(kind, f"pid{os.getpid()}").write({"event": event, "detail": detail, "t_ns": time.time_ns()})
    except Exception:
        pass


def profiling():
    """Whether a torch profiler is recording in this process."""
    try:
        import torch

        return torch._C._autograd._profiler_enabled()
    except Exception:
        return True


def compiling():
    """Whether Dynamo is tracing (hooks must then be no-ops so compiled graphs are unchanged)."""
    try:
        import torch

        return torch.compiler.is_compiling()
    except Exception:
        return False


def capture_ranks():
    return os.environ.get("INFX_PROF_CAPTURE_RANKS", "dp0_tp0").split(",")


def is_capture_rank(tag):
    ranks = capture_ranks()
    return tag in ranks or "all" in ranks


def keep_signature(wrapper, fn):
    """Give `wrapper` the signature of `fn` with string annotations resolved in fn's globals."""
    import inspect

    try:
        wrapper.__signature__ = inspect.signature(fn, eval_str=True)
    except Exception:
        pass
    return wrapper


def write_json(relpath, data):
    path = os.path.join(DIR, relpath)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=1, default=str)
