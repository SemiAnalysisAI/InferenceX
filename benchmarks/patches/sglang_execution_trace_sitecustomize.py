"""Attach a torch.profiler.ExecutionTraceObserver to sglang profiler sessions.

Kineto traces (what sglang's /start_profile produces) record ops, shapes, and
kernel timing, but no dataflow edges. PyTorch's ExecutionTraceObserver records
the operator DAG with tensor ids as JSON; its nodes carry an "rf_id" that joins
the Kineto trace's cpu_op "Record function id" args, so the pair yields a true
dataflow graph aligned with the timeline (this is the join HTA's trace_linker
and Chakra use).

sglang's ProfileReq has no execution-trace option and the torch profiler is
constructed inside each scheduler subprocess (SchedulerProfilerManager /
scheduler_profiler_mixin, depending on version), so this file is applied as a
`sitecustomize.py` PYTHONPATH shim (same mechanism as utils/evals/patches/
lm_eval_sitecustomize.py): setup_profiling_env in benchmarks/benchmark_lib.sh
copies it into a temp dir prepended to PYTHONPATH before the server launches.
Every python interpreter in the container then imports it at startup --
including the spawned scheduler processes where the model forward and the
Kineto capture actually run.

Rather than patching sglang internals (whose profiler code has moved across
releases: scheduler.py -> scheduler_profiler_mixin.py ->
scheduler_components/profiler_manager.py), it wraps
torch.profiler.profile.start/stop, which every sglang version calls for its
capture sessions. Import-hook based, so processes that never import
torch.profiler pay nothing.

Gate: SGLANG_EXECUTION_TRACE_DIR must be set (exported by setup_profiling_env
when PROFILE=1, FRAMEWORK is sglang-based, and PROFILE_EXECUTION_TRACE=1).
Output: et-<unix_ts>-rank<global_rank>-s<session>.json per capture session in
that dir (pid<pid> when torch.distributed is not initialized). Files are
written as .json.tmp and renamed on a clean stop, so crashed sessions never
stage a truncated JSON. All failure paths warn and fall through: profiling
itself must never break because of this shim.
"""

import os
import sys

_ENV_DIR = "SGLANG_EXECUTION_TRACE_DIR"
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
    print(f"[sglang-execution-trace] {msg}", file=sys.stderr, flush=True)


def _attach(prof):
    """Register + start an ExecutionTraceObserver alongside a started session."""
    global _active_observer, _session_idx
    out_dir = os.environ.get(_ENV_DIR)
    if not out_dir:
        return
    if _active_observer is not None:
        # Overlapping sessions in one process (not sglang's pattern, but be
        # safe): only the first gets an observer.
        _warn_once("overlap", "observer already active; skipping nested session")
        return
    try:
        import time

        from torch.profiler import ExecutionTraceObserver

        os.makedirs(out_dir, exist_ok=True)
        _session_idx += 1
        tmp_path = os.path.join(
            out_dir,
            f"et-{int(time.time())}-pid{os.getpid()}-s{_session_idx}.json.tmp",
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
        f"[sglang-execution-trace] recording execution trace -> {tmp_path}",
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

    rank_tag = f"pid{os.getpid()}"
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            rank_tag = f"rank{dist.get_rank()}"
    except Exception:  # noqa: BLE001
        pass
    final_path = tmp_path[: -len(".json.tmp")]
    # tmp name is et-<ts>-pid<pid>-s<idx>; swap pid for the global rank when
    # known (session index disambiguates by-stage sessions within a second).
    base = os.path.basename(final_path).split("-")
    base[2] = rank_tag
    final_path = os.path.join(os.path.dirname(final_path), "-".join(base) + ".json")
    try:
        os.replace(tmp_path, final_path)
        print(
            f"[sglang-execution-trace] execution trace saved: {final_path}",
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
