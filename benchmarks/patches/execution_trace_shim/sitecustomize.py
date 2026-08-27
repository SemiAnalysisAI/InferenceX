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
_ENV_DISPATCH = "PROFILE_ET_DISPATCH"  # "0" disables the launcher trampoline
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
    _write_dispatch_report(os.path.dirname(tmp_path), _session_idx)


# ---------------------------------------------------------------------------
# Launcher dispatch trampoline.
#
# Kernels launched outside the torch dispatcher (tilelang JIT kernels,
# deep_gemm's python entrypoints) are invisible to both Kineto op attribution
# and the ExecutionTraceObserver: nothing records their tensor arguments, so
# the captured operator DAG has holes at every such launch. While an ET
# session is active, calls to those launchers are routed through dynamically
# registered torch.library custom ops (namespace `etshim`) whose only impl is
# the original callable — the dispatcher then records the call with its full
# tensor argument list like any other op. Outside ET sessions every call goes
# straight through (one flag check of overhead). Any failure permanently
# falls back to the raw callable for that call signature.
# ---------------------------------------------------------------------------

_dispatch_lib = None
_dispatch_idx = 0
_proxy_registry = []  # every _LauncherProxy, for the per-session report


def collections_counter():
    import collections

    return collections.Counter()


def _dispatch_enabled():
    return os.environ.get(_ENV_DISPATCH, "1") != "0"


def _write_dispatch_report(out_dir, session_idx):
    """Sidecar accounting of what each proxied launcher did this session.

    Lands in the profile artifact next to the ET json, so failures to
    dispatch are diagnosable from the artifact instead of worker stderr.
    """
    stats = {}
    for p in _proxy_registry:
        s = p._infx_stats
        if s:
            stats[p._infx_qualname] = dict(s)
            s.clear()
    if not stats:
        return
    try:
        import json
        import socket

        host = socket.gethostname().split(".")[0] or "unknown"
        path = os.path.join(
            out_dir, f"et-dispatch-report-{host}-pid{os.getpid()}-s{session_idx}.json"
        )
        with open(path, "w") as fh:
            json.dump(stats, fh, indent=1, sort_keys=True)
        print(f"[execution-trace] dispatch report saved: {path}",
              file=sys.stderr, flush=True)
    except Exception as exc:  # noqa: BLE001
        _warn_once("report", f"failed to write dispatch report: {exc!r}")


def _type_key(a):
    # bool must be tested before int (bool subclasses int)
    if a is None:
        return "n"  # schema'd as Tensor? and always passed None
    if isinstance(a, bool):
        return "b"
    if isinstance(a, int):
        return "i"
    if isinstance(a, float):
        return "f"
    type_name = type(a).__name__
    if type_name == "Tensor":
        return "T"
    return None


_SCHEMA_OF_KEY = {
    "T": "Tensor(a{i}!) a{i}",  # assume mutable: out-params are the norm here
    "n": "Tensor? a{i}",
    "b": "bool a{i}",
    "i": "int a{i}",
    "f": "float a{i}",
}


def _ret_schema(result):
    """Return-type schema for a first-call result, or None if not schema-able.

    None -> "()", Tensor -> "Tensor", tuple/list of Tensors -> "(Tensor, ...)"
    (kernels like compress_forward / moe_fused_gate / mhc_pre return tuples).
    """
    if result is None:
        return "()"
    import torch

    if isinstance(result, torch.Tensor):
        return "Tensor"
    if (isinstance(result, (tuple, list)) and result
            and all(isinstance(r, torch.Tensor) for r in result)):
        return "(" + ", ".join(["Tensor"] * len(result)) + ")"
    return None


def _register_dispatch_op(shortname, fn, args, ret_schema):
    """Define etshim::<shortname>_<n> with a schema derived from this call."""
    global _dispatch_lib, _dispatch_idx
    from torch.library import Library

    parts = []
    for i, a in enumerate(args):
        k = _type_key(a)
        tmpl = _SCHEMA_OF_KEY.get(k)
        if tmpl is None:
            return None
        parts.append(tmpl.format(i=i))
    if _dispatch_lib is None:
        _dispatch_lib = Library("etshim", "DEF")
    _dispatch_idx += 1
    safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in shortname)
    opname = f"{safe}_{_dispatch_idx}"
    _dispatch_lib.define(f"{opname}({', '.join(parts)}) -> {ret_schema}")
    if ret_schema.startswith("(") and ret_schema != "()":
        # multi-return ops must hand the dispatcher a tuple
        _dispatch_lib.impl(opname, lambda *a: tuple(fn(*a)), "CompositeExplicitAutograd")
    else:
        _dispatch_lib.impl(opname, lambda *a: fn(*a), "CompositeExplicitAutograd")
    import torch

    return getattr(torch.ops.etshim, opname).default


class _LauncherProxy:
    """Callable proxy that dispatches through etshim ops during ET sessions.

    Attribute access forwards to the wrapped callable so decorated objects
    that expose helper methods keep working.
    """

    def __init__(self, qualname, fn):
        object.__setattr__(self, "_infx_fn", fn)
        object.__setattr__(self, "_infx_qualname", qualname)
        object.__setattr__(self, "_infx_ops", {})
        object.__setattr__(self, "_infx_stats", collections_counter())
        _proxy_registry.append(self)

    def _target(self, split):
        """A positional-only callable equivalent to fn for a flattened call.

        Keyword arguments are flattened by CALL SHAPE (sorted names appended
        after the positionals) and re-applied by name here — no
        inspect.signature involved, so pybind builtins, **kwargs forwarders,
        and keyword-only parameters all work.
        """
        if split is None:
            return self._infx_fn
        npos, names = split

        def call(*a, _fn=self._infx_fn, _names=names, _k=npos):
            return _fn(*a[:_k], **dict(zip(_names, a[_k:])))

        return call

    def __call__(self, *args, **kwargs):
        fn = self._infx_fn
        if _active_observer is None:
            return fn(*args, **kwargs)
        stats = self._infx_stats
        if kwargs:
            names = tuple(sorted(kwargs))
            flat = args + tuple(kwargs[n] for n in names)
            split = (len(args), names)
        else:
            flat, split = args, None
        try:
            key = (tuple(_type_key(a) for a in flat), split)
        except Exception:  # noqa: BLE001
            stats["typekey_failed"] += 1
            return fn(*args, **kwargs)
        ops = self._infx_ops
        op = ops.get(key)
        if op is False:
            stats["raw_fallback"] += 1
            return fn(*args, **kwargs)
        if op is None:
            # First call for this signature runs raw to learn the return
            # kind; anything but None / Tensor / tuple-of-Tensors is not
            # schema-able.
            target = self._target(split)
            result = target(*flat)
            stats["raw_learn"] += 1
            try:
                ret_schema = _ret_schema(result)
                if ret_schema is None:
                    stats["ret_not_schemable"] += 1
                    ops[key] = False
                else:
                    new_op = _register_dispatch_op(
                        self._infx_qualname.rsplit(".", 1)[-1],
                        target,
                        flat,
                        ret_schema,
                    )
                    if new_op is None:
                        stats["args_not_schemable"] += 1
                        ops[key] = False
                    else:
                        if isinstance(result, list):
                            # dispatcher returns tuples; preserve raw shape
                            base_op = new_op
                            new_op = lambda *a: list(base_op(*a))  # noqa: E731
                        ops[key] = new_op
            except Exception as exc:  # noqa: BLE001
                stats["define_failed"] += 1
                _warn_once(
                    f"dispatch-def-{self._infx_qualname}",
                    f"cannot register dispatch op for {self._infx_qualname}: {exc!r}",
                )
                ops[key] = False
            return result
        try:
            result = op(*flat)
            stats["dispatched"] += 1
            return result
        except Exception as exc:  # noqa: BLE001
            stats["call_failed"] += 1
            _warn_once(
                f"dispatch-call-{self._infx_qualname}",
                f"dispatch call failed for {self._infx_qualname}, "
                f"falling back to raw: {exc!r}",
            )
            ops[key] = False
            return self._infx_fn(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(object.__getattribute__(self, "_infx_fn"), item)

    def __repr__(self):
        return f"<etshim launcher proxy for {self._infx_fn!r}>"


def _maybe_proxy(qualname, obj):
    if not callable(obj) or isinstance(obj, _LauncherProxy) or isinstance(obj, type):
        return obj
    return _LauncherProxy(qualname, obj)


def _hook_tilelang(module):
    """Wrap tilelang.jit so decorated kernels dispatch through etshim ops."""
    orig = getattr(module, "jit", None)
    if orig is None or getattr(orig, "_infx_et_patched", False):
        return
    import functools

    @functools.wraps(orig)
    def jit(*args, **kwargs):
        # bare form: @tilelang.jit
        if len(args) == 1 and callable(args[0]) and not kwargs:
            fn = args[0]
            return _maybe_proxy(
                f"tilelang.{getattr(fn, '__name__', 'kernel')}", orig(fn)
            )
        deco = orig(*args, **kwargs)
        if not callable(deco):
            return deco

        @functools.wraps(deco)
        def wrapped_deco(fn):
            return _maybe_proxy(
                f"tilelang.{getattr(fn, '__name__', 'kernel')}", deco(fn)
            )

        return wrapped_deco

    jit._infx_et_patched = True
    module.jit = jit
    print("[execution-trace] tilelang.jit launcher trampoline armed",
          file=sys.stderr, flush=True)


def _hook_module_functions(root):
    """Patcher proxying a module's public module-level callables.

    Only callables DEFINED under `root` (by __module__) are proxied, so
    re-exports of aten/foreign functions are left alone.
    """

    def patch(module):
        if getattr(module, "_infx_et_patched", False):
            return
        wrapped = 0
        for attr in dir(module):
            if attr.startswith("_"):
                continue
            try:
                val = getattr(module, attr)
            except Exception:  # noqa: BLE001
                continue
            if callable(val) and not isinstance(val, type) and getattr(
                val, "__module__", ""
            ).startswith(root):
                setattr(module, attr, _maybe_proxy(f"{module.__name__}.{attr}", val))
                wrapped += 1
        module._infx_et_patched = True
        if wrapped:
            print(f"[execution-trace] launcher trampoline armed on "
                  f"{module.__name__} ({wrapped} callables)",
                  file=sys.stderr, flush=True)

    return patch


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


class _ModuleImportHook:
    """Meta-path finder that patches target modules right after they execute.

    sitecustomize runs before torch is importable state-wise (and importing
    torch in every python process would be prohibitively slow), so patches
    are deferred until the process actually imports each target.
    """

    _busy = False

    def __init__(self, patchers, prefix_patchers=()):
        self._patchers = patchers  # fullname -> patch fn
        # (prefix, patch fn): applies to the prefix module AND its submodules
        self._prefix_patchers = tuple(prefix_patchers)

    def _patcher_for(self, fullname):
        p = self._patchers.get(fullname)
        if p is not None:
            return p
        for prefix, patcher in self._prefix_patchers:
            if fullname == prefix or fullname.startswith(prefix + "."):
                return patcher
        return None

    def find_spec(self, fullname, path=None, target=None):
        patcher = self._patcher_for(fullname)
        if patcher is None or _ModuleImportHook._busy:
            return None
        import importlib.util

        _ModuleImportHook._busy = True
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            _ModuleImportHook._busy = False
        if spec is None or spec.loader is None:
            return None
        orig_exec_module = spec.loader.exec_module

        def exec_module(module):
            orig_exec_module(module)
            try:
                patcher(module)
            except Exception as exc:  # noqa: BLE001
                _warn_once("patch", f"failed to patch {fullname}: {exc!r}")

        # FileFinder builds a fresh loader instance per spec, so shadowing
        # exec_module on this instance only affects this one import.
        spec.loader.exec_module = exec_module
        return spec


# Module families whose python launchers bypass the torch dispatcher; every
# public callable defined under each root gets the etshim trampoline. Roots
# and their submodules are matched by prefix (sglang.jit_kernel.dsv4.*, the
# flashinfer wrappers around cute-dsl/trtllm/triton kernels, ...).
# sglang.srt.layers.mhc and deep_gemm_wrapper are hooked at the python-wrapper
# level because their inner launches evade lower hooks: tilelang kernels load
# straight from the on-disk JIT cache on warm starts (never passing through
# tilelang.jit), and deep_gemm's pybind builtins have no python signature to
# bind keyword arguments against.
_DISPATCH_PREFIXES = (
    "deep_gemm",
    "sglang.jit_kernel",
    "sglang.srt.layers.mhc",
    "sglang.srt.layers.deep_gemm_wrapper",
    "flashinfer",
)


def _install():
    if not os.environ.get(_ENV_DIR):
        return
    patchers = {_TARGET: _patch_profiler_module}
    prefix_patchers = []
    if _dispatch_enabled():
        patchers["tilelang"] = _hook_tilelang
        prefix_patchers = [(p, _hook_module_functions(p)) for p in _DISPATCH_PREFIXES]
    pending = {}
    for fullname, patcher in patchers.items():
        existing = sys.modules.get(fullname)
        if existing is not None:
            try:
                patcher(existing)
            except Exception as exc:  # noqa: BLE001
                _warn_once("patch", f"failed to patch {fullname}: {exc!r}")
        else:
            pending[fullname] = patcher
    if pending or prefix_patchers:
        sys.meta_path.insert(0, _ModuleImportHook(pending, prefix_patchers))


_install()
