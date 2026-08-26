"""Attach a torch.profiler.ExecutionTraceObserver to engine profiler sessions.

Kineto traces (what /start_profile produces) record ops, shapes, and kernel
timing, but no dataflow edges. PyTorch's ExecutionTraceObserver records the
operator DAG with tensor ids as JSON; its nodes carry an "rf_id" that joins
the Kineto trace's cpu_op "Record function id" args, so the pair yields a true
dataflow graph aligned with the timeline (this is the join HTA's trace_linker
and Chakra use).

Neither engine exposes an execution-trace option (sglang's ProfileReq has no
field for it; vllm's ProfilerConfig doesn't either), and both construct their
torch profiler inside the process that runs the model forward -- sglang in
each scheduler subprocess (SchedulerProfilerManager / scheduler_profiler_mixin,
depending on version), vllm in each worker via TorchProfilerWrapper. Their
profiler internals also move across releases, so instead of patching engine
code, this shim wraps torch.profiler.profile.start/stop, which every version
of both engines calls for its capture sessions. Import-hook based, so
processes that never import torch.profiler pay nothing.

Delivery: this file keeps its runtime name (sitecustomize.py) in the repo, so
arming it is just putting this directory on PYTHONPATH before the server
launches -- single-node runs export it in setup_profiling_env
(benchmarks/benchmark_lib.sh), multinode srt-slurm runs get it injected into
each worker's environment block by utils/profile_recipe_inject.py (worker
containers mount the checkout at /infmax-workspace). Spawned engine
subprocesses inherit the environment, so the shim runs in every python
interpreter that matters.

Gate: PROFILE_EXECUTION_TRACE_DIR must be set (see PROFILE_EXECUTION_TRACE in
benchmark_lib.sh). Additionally, a `.et-disabled` marker file in that dir (or
its parent -- multinode workers write to /logs/profiles/<mode> while the
benchmark-stage trigger manages /logs/profiles) suppresses the observer for
sessions that start while it exists. Env is fixed per process but the marker
is re-checked at every profile.start(), which is how the agentic dual-window
trigger (launch_agentic_profile_trigger in benchmark_lib.sh) applies its
per-window policy: kineto stacks on the ramp window without ET, ET on the
stackless steady window. sglang's with_stack python tracer has a teardown
race at profiler stop under the overlap scheduler that an active ET observer
can expose, so the two never ride the same session there. Absent marker =
observer on (the single-session fixed-seq path has no trigger and relies on
this default). Output per capture session in that dir:
et-<unix_ts>-<host>-pid<pid>[-rank<global_rank>]-s<session>.json -- host+pid
keep names collision-free when many workers share one output dir (multinode
/logs/profiles), the rank tag is added when torch.distributed is initialized.
Files are written as .json.tmp and renamed on a clean stop, so crashed
sessions never stage a truncated JSON. All failure paths warn and fall
through: profiling itself must never break because of this shim.
"""

import os
import sys

_ENV_DIR = "PROFILE_EXECUTION_TRACE_DIR"
_DISABLE_MARKER = ".et-disabled"
_TARGET = "torch.profiler"

# Process-globals: the ExecutionTraceObserver callback is a singleton in
# libtorch, so at most one observer may be registered per process.
_active_observer = None
_session_idx = 0
_warned = set()


def _warn_once(key, msg):
    if key in _warned:
        return
    _warned.add(key)
    print(f"[execution-trace] {msg}", file=sys.stderr, flush=True)


def _et_disabled_by_marker(out_dir):
    """Per-session control: a .et-disabled marker in the output dir (or its
    parent, for multinode /logs/profiles/<mode> layouts) suppresses the
    observer for sessions starting while it exists. Re-checked at every
    profile.start() so the agentic trigger can toggle ET per capture window
    without touching process env."""
    clean = out_dir.rstrip("/") or "/"
    for d in (clean, os.path.dirname(clean) or "/"):
        try:
            if os.path.exists(os.path.join(d, _DISABLE_MARKER)):
                return True
        except OSError:
            pass
    return False


def _attach(prof):
    """Register + start an ExecutionTraceObserver alongside a started session."""
    global _active_observer, _session_idx
    out_dir = os.environ.get(_ENV_DIR)
    if not out_dir:
        return
    if _et_disabled_by_marker(out_dir):
        print(
            f"[execution-trace] {_DISABLE_MARKER} marker present; "
            "skipping this session",
            file=sys.stderr,
            flush=True,
        )
        return
    if _active_observer is not None:
        # Overlapping sessions in one process (not the engines' pattern, but
        # be safe): only the first gets an observer.
        _warn_once("overlap", "observer already active; skipping nested session")
        return
    try:
        import socket
        import time

        from torch.profiler import ExecutionTraceObserver

        os.makedirs(out_dir, exist_ok=True)
        _session_idx += 1
        host = socket.gethostname().split(".")[0] or "unknown"
        tmp_path = os.path.join(
            out_dir,
            f"et-{int(time.time())}-{host}-pid{os.getpid()}-s{_session_idx}.json.tmp",
        )
        observer = ExecutionTraceObserver()
        observer.register_callback(tmp_path)
        observer.start()
    except Exception as exc:  # noqa: BLE001 - never break the kineto capture
        _warn_once("attach", f"failed to attach observer: {exc!r}")
        return
    prof.__dict__["_infx_et_state"] = (observer, tmp_path)
    _active_observer = observer
    print(
        f"[execution-trace] recording execution trace -> {tmp_path}",
        file=sys.stderr,
        flush=True,
    )


def _finalize(prof):
    """Stop the observer, close the JSON, and rename it to its final name."""
    global _active_observer
    observer, tmp_path = prof.__dict__.pop("_infx_et_state", (None, None))
    if observer is None:
        return
    try:
        observer.stop()
        # unregister_callback writes the JSON epilogue and closes the file.
        observer.unregister_callback()
    except Exception as exc:  # noqa: BLE001
        _warn_once("finalize", f"failed to finalize observer: {exc!r}")
        _active_observer = None
        return
    _active_observer = None

    rank_tag = ""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            rank_tag = f"-rank{dist.get_rank()}"
    except Exception:  # noqa: BLE001
        pass
    # tmp name is et-<ts>-<host>-pid<pid>-s<idx>.json.tmp; splice the rank tag
    # (when known) in front of the session suffix.
    stem = os.path.basename(tmp_path)[: -len(".json.tmp")]
    prefix, _, session = stem.rpartition("-")
    final_path = os.path.join(
        os.path.dirname(tmp_path), f"{prefix}{rank_tag}-{session}.json"
    )
    try:
        os.replace(tmp_path, final_path)
        print(
            f"[execution-trace] execution trace saved: {final_path}",
            file=sys.stderr,
            flush=True,
        )
    except OSError as exc:
        _warn_once("rename", f"failed to rename {tmp_path}: {exc!r}")


def _patch_profiler_module(module):
    """Wrap torch.profiler.profile.start/stop (idempotent)."""
    profile_cls = getattr(module, "profile", None)
    if profile_cls is None or getattr(profile_cls, "_infx_et_patched", False):
        return
    orig_start = profile_cls.start
    orig_stop = profile_cls.stop

    def start(self):
        result = orig_start(self)
        # Attach only after the kineto session started cleanly (sglang treats
        # a start() RuntimeError as session failure), so the ET window is a
        # subset of the kineto window and rf_ids line up.
        _attach(self)
        return result

    def stop(self):
        _finalize(self)
        return orig_stop(self)

    profile_cls.start = start
    profile_cls.stop = stop
    profile_cls._infx_et_patched = True


class _ProfilerImportHook:
    """Meta-path finder that patches torch.profiler right after it executes.

    sitecustomize runs before torch is importable state-wise (and importing
    torch in every python process would be prohibitively slow), so the patch
    is deferred until the process actually imports torch.profiler.
    """

    _busy = False

    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET or _ProfilerImportHook._busy:
            return None
        import importlib.util

        _ProfilerImportHook._busy = True
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            _ProfilerImportHook._busy = False
        if spec is None or spec.loader is None:
            return None
        orig_exec_module = spec.loader.exec_module

        def exec_module(module):
            orig_exec_module(module)
            try:
                _patch_profiler_module(module)
            except Exception as exc:  # noqa: BLE001
                _warn_once("patch", f"failed to patch torch.profiler: {exc!r}")

        # FileFinder builds a fresh loader instance per spec, so shadowing
        # exec_module on this instance only affects this one import.
        spec.loader.exec_module = exec_module
        return spec


def _install():
    if not os.environ.get(_ENV_DIR):
        return
    existing = sys.modules.get(_TARGET)
    if existing is not None:
        _patch_profiler_module(existing)
        return
    sys.meta_path.insert(0, _ProfilerImportHook())


_install()
