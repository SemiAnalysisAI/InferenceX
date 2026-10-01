"""Op-attribution profiling hooks for vLLM, loaded through PYTHONPATH.

Inactive unless INFX_PROF_DIR is set. When active, in every vLLM process:

- Each CUDA graph gets an ordinal. Its capture runs inside an
  ``infx_graph_capture#<n>`` record_function and each replay inside
  ``infx_graph_replay#<n>``. A replayed kernel's ``graph node id`` then pairs
  with the n-th launch recorded while graph n was captured.
- The V2 model runner's ``capture_model`` runs under a torch profiler
  (record_shapes, with_stack) on the selected ranks, so every launch captured
  into a graph is recorded with its CPU op, input shapes and Python stack.
- Every ``execute_model`` step is logged (per-request scheduled and computed
  tokens, spec tokens, the dispatched cudagraph descriptor and the DP-synced
  token counts) and runs inside an ``infx_step#<k>`` record_function, so a
  profiled step joins its batch composition.

Outputs go under $INFX_PROF_DIR: env/, capture/, graphs/, steps/, errors/.
Nothing here may break serving: every hook degrades to the original call.
"""

import importlib.abc
import importlib.machinery
import importlib.util
import itertools
import json
import os
import sys
import threading
import time
import traceback

_DIR = os.environ.get("INFX_PROF_DIR")


def _chain_next_sitecustomize():
    """Run the sitecustomize this file shadows, if the image has one."""
    here = os.path.dirname(os.path.abspath(__file__))
    for entry in sys.path:
        if not entry or os.path.abspath(entry) == here:
            continue
        spec = importlib.machinery.PathFinder.find_spec("sitecustomize", [entry])
        if spec is not None and spec.origin and spec.loader is not None:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return


def _write_error(where):
    try:
        path = os.path.join(_DIR, "errors", f"{os.getpid()}.log")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            f.write(f"--- {time.time():.3f} {where}\n{traceback.format_exc()}\n")
    except Exception:
        pass


def _rank_tag(runner=None):
    """dp<d>_tp<t> for the calling worker; pid when ranks are not set up."""
    try:
        from vllm.distributed import parallel_state as ps

        tp = ps.get_tensor_model_parallel_rank()
        dp = 0
        if runner is not None:
            dp = runner.parallel_config.data_parallel_rank
        else:
            dp = ps.get_dp_group().rank_in_group
        return f"dp{dp}_tp{tp}"
    except Exception:
        return f"pid{os.getpid()}"


class _JsonlSink:
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


def _sink(kind, tag):
    key = (kind, tag)
    if key not in _sinks:
        _sinks[key] = _JsonlSink(os.path.join(_DIR, kind, f"{tag}.jsonl"))
    return _sinks[key]


def _flush_all():
    for sink in list(_sinks.values()):
        try:
            sink.flush()
        except Exception:
            pass


# --- CUDA graph ordinals -----------------------------------------------------

_graph_ids = itertools.count()


def _profiling():
    try:
        import torch

        return torch._C._autograd._profiler_enabled()
    except Exception:
        return True


def _forward_context_desc():
    try:
        from vllm.forward_context import get_forward_context, is_forward_context_available

        if not is_forward_context_available():
            return None
        ctx = get_forward_context()
        return {
            "cudagraph_runtime_mode": str(getattr(ctx, "cudagraph_runtime_mode", None)),
            "batch_descriptor": repr(getattr(ctx, "batch_descriptor", None)),
        }
    except Exception:
        return None


def _patch_cuda_graph(graphs_module):
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
            _sink("graphs", f"pid{os.getpid()}").write({
                "graph": gid,
                "event": "capture_begin",
                "t_ns": time.time_ns(),
                "device": torch.cuda.current_device(),
                "context": _forward_context_desc(),
            })
        except Exception:
            _write_error("capture_begin")
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
                _sink("graphs", f"pid{os.getpid()}").write({
                    "graph": getattr(self, "_infx_id", None),
                    "event": "capture_end",
                    "t_ns": time.time_ns(),
                })
            except Exception:
                _write_error("capture_end")

    def replay(self, *args, **kwargs):
        gid = getattr(self, "_infx_id", None)
        if gid is None or not _profiling():
            return orig_replay(self, *args, **kwargs)
        with torch.autograd.profiler.record_function(f"infx_graph_replay#{gid}"):
            return orig_replay(self, *args, **kwargs)

    cls.capture_begin = capture_begin
    cls.capture_end = capture_end
    cls.replay = replay
    cls._infx_patched = True


# --- model runner: capture profile, step log --------------------------------

def _capture_ranks():
    return os.environ.get("INFX_PROF_CAPTURE_RANKS", "dp0_tp0").split(",")


def _write_env(runner, tag):
    try:
        import torch

        info = {"pid": os.getpid(), "rank": tag, "t_ns": time.time_ns()}
        for name in ("vllm", "torch", "triton", "flashinfer", "deep_gemm", "cuda"):
            try:
                module = __import__(name)
                info[f"{name}_version"] = getattr(module, "__version__", None)
            except Exception as e:
                info[f"{name}_version"] = f"unavailable: {e}"
        try:
            import cuda.bindings

            info["cuda_bindings_version"] = cuda.bindings.__version__
        except Exception as e:
            info["cuda_bindings_version"] = f"unavailable: {e}"
        info["cuda_runtime"] = torch.version.cuda
        try:
            info["cuda_driver"] = torch._C._cuda_getDriverVersion()
        except Exception as e:
            info["cuda_driver"] = f"unavailable: {e}"
        try:
            info["device_name"] = torch.cuda.get_device_name()
        except Exception as e:
            info["device_name"] = f"unavailable: {e}"
        try:
            # Joins this rank to the window client's per-GPU clock samples (NVML UUIDs).
            info["device_uuid"] = str(torch.cuda.get_device_properties(
                torch.cuda.current_device()).uuid)
        except Exception as e:
            info["device_uuid"] = f"unavailable: {e}"
        info["runner_class"] = f"{type(runner).__module__}.{type(runner).__qualname__}"
        info["env"] = {k: v for k, v in os.environ.items()
                       if k.startswith(("VLLM_", "INFX_", "TORCH", "CUDA", "NCCL", "INDUCTOR"))}
        info["vllm_config"] = str(getattr(runner, "vllm_config", None))
        path = os.path.join(_DIR, "env", f"{tag}.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(info, f, indent=1, default=str)
    except Exception:
        _write_error("write_env")


def _batch_desc_record(result):
    try:
        batch_desc, dp_sync = result
        record = {"batch_desc": repr(batch_desc)}
        for field in ("cg_mode", "num_tokens", "num_reqs", "uniform_token_count"):
            if hasattr(batch_desc, field):
                record[field] = str(getattr(batch_desc, field))
        if dp_sync is not None:
            record["dp_sync"] = repr(dp_sync)
        return record
    except Exception:
        return {"batch_desc": "unparsed"}


_dispatch_log = threading.local()


def _patch_dp_utils(module):
    orig = module.dispatch_cg_and_sync_dp
    if getattr(orig, "_infx_patched", False):
        return

    def dispatch_cg_and_sync_dp(*args, **kwargs):
        result = orig(*args, **kwargs)
        try:
            calls = getattr(_dispatch_log, "calls", None)
            if calls is not None:
                calls.append(_batch_desc_record(result))
        except Exception:
            pass
        return result

    dispatch_cg_and_sync_dp._infx_patched = True
    module.dispatch_cg_and_sync_dp = dispatch_cg_and_sync_dp


def _step_record(scheduler_output):
    reqs = []
    computed = {}
    prompt_len = {}
    for new in scheduler_output.scheduled_new_reqs:
        computed[new.req_id] = new.num_computed_tokens
        ids = getattr(new, "prompt_token_ids", None)
        prompt_len[new.req_id] = len(ids) if ids is not None else None
    cached = scheduler_output.scheduled_cached_reqs
    for req_id, n in zip(cached.req_ids, cached.num_computed_tokens):
        computed[req_id] = n
    spec = scheduler_output.scheduled_spec_decode_tokens or {}
    for req_id, n in scheduler_output.num_scheduled_tokens.items():
        reqs.append({
            "req": req_id,
            "scheduled": n,
            "computed": computed.get(req_id),
            "spec": len(spec.get(req_id, ())),
            "new": req_id in prompt_len,
            "prompt_len": prompt_len.get(req_id),
        })
    return {"total_tokens": scheduler_output.total_num_scheduled_tokens, "reqs": reqs}


# --- module markers ---------------------------------------------------------
# Replay windows run without Python stacks (their export stalls the engine), so
# eager modules mark themselves: infx_mod#<qualified name>. Global module hooks
# are only safe where Dynamo never traces (compilation mode NONE); compiled
# pieces mark themselves through the piecewise backend instead.

_module_names = None  # weakref.WeakKeyDictionary once a runner registers its models
_module_hooks = []
_module_tls = threading.local()
_module_hooks_allowed = False


def _register_module_names(runner):
    import weakref

    global _module_names, _module_hooks_allowed
    names = weakref.WeakKeyDictionary()
    roots = [("model", getattr(runner, "model", None))]
    speculator = getattr(runner, "speculator", None)
    if speculator is not None:
        roots.append(("draft", getattr(speculator, "model", None)))
    for prefix, root in roots:
        if root is None or not hasattr(root, "named_modules"):
            continue
        for name, mod in root.named_modules(prefix=prefix):
            names.setdefault(mod, name)
    _module_names = names
    try:
        mode = runner.vllm_config.compilation_config.mode
        _module_hooks_allowed = int(mode) == 0
    except Exception:
        _module_hooks_allowed = False


def _tensor_signature(args):
    """'7x7168:bfloat16;1x2:int64' for the tensors among a call's (one-level nested) args.

    Marker names must stay free of quotes and backslashes: Kineto writes event
    names into the trace JSON without escaping them.
    """
    import torch

    sig = []
    for arg in args:
        items = arg if isinstance(arg, (list, tuple)) else (arg,)
        for item in items:
            if isinstance(item, torch.Tensor):
                dims = "x".join(str(d) for d in item.shape)
                sig.append(f"{dims}:{str(item.dtype).removeprefix('torch.')}")
    return ";".join(sig)


def _module_pre_hook(module, args):
    try:
        import torch

        name = (_module_names.get(module) if _module_names is not None else None) or type(module).__name__
        # Python-launched kernels (Triton, TileLang, DeepGEMM) have no op
        # shapes of their own; the module's input shapes stand in for them.
        rf = torch.autograd.profiler.record_function(f"infx_mod#{name}#{_tensor_signature(args)}")
        rf.__enter__()
        stack = getattr(_module_tls, "stack", None)
        if stack is None:
            stack = _module_tls.stack = []
        stack.append((module, rf))
    except Exception:
        pass


def _module_post_hook(module, args, output):
    stack = getattr(_module_tls, "stack", None)
    while stack:
        mod, rf = stack.pop()
        try:
            rf.__exit__(None, None, None)
        except Exception:
            pass
        if mod is module:
            break


def _enable_module_markers():
    if not _module_hooks_allowed or _module_hooks:
        return
    from torch.nn.modules import module as nn_module

    _module_hooks.append(nn_module.register_module_forward_pre_hook(_module_pre_hook))
    _module_hooks.append(nn_module.register_module_forward_hook(_module_post_hook, always_call=True))


def _disable_module_markers():
    while _module_hooks:
        _module_hooks.pop().remove()
    stack = getattr(_module_tls, "stack", None)
    while stack:
        try:
            stack.pop()[1].__exit__(None, None, None)
        except Exception:
            pass


def _patch_profiler_wrapper(module):
    cls = module.WorkerProfiler
    if getattr(cls, "_infx_patched", False):
        return
    orig_start, orig_stop = cls._call_start, cls._call_stop

    # Each rank's window, as its profiler saw it: the window client samples GPU
    # clocks until every rank that started has stopped (the stop is logged
    # before the trace export, which runs inside it).
    def _log(event):
        try:
            _sink("profiler", _rank_tag()).write({"event": event, "t_ns": time.time_ns()})
        except Exception:
            _write_error(f"profiler {event} log")

    def _call_start(self, *args, **kwargs):
        result = orig_start(self, *args, **kwargs)
        _log("start")
        try:
            _enable_module_markers()
        except Exception:
            _write_error("enable module markers")
        return result

    def _call_stop(self, *args, **kwargs):
        _log("stop")
        try:
            _routing_flush()
        except Exception:
            _write_error("routing flush")
        try:
            _disable_module_markers()
        except Exception:
            _write_error("disable module markers")
        return orig_stop(self, *args, **kwargs)

    cls._call_start = _call_start
    cls._call_stop = _call_stop
    cls._infx_patched = True


def _patch_piecewise_backend(module):
    import torch

    cls = module.PiecewiseBackend
    if getattr(cls, "_infx_patched", False):
        return
    orig_call = cls.__call__

    def __call__(self, *args):
        if not _profiling():
            return orig_call(self, *args)
        index = getattr(self, "piecewise_compile_index", "?")
        with torch.autograd.profiler.record_function(f"infx_piece#{index}"):
            return orig_call(self, *args)

    cls.__call__ = __call__
    cls._infx_patched = True


_current_step = None  # the scheduled step executing on this worker


# --- MoE routing ----------------------------------------------------------------
# vLLM's RoutedExpertsCapturer writes each MoE layer's top-k expert ids into a
# device buffer from the layer's capture_fn; the write is a GPU copy, so CUDA
# graphs capture it and every replay records its routing. It is bound to the
# target model before graph capture. While a window is open, each step's rows
# are copied to pinned host memory asynchronously; at the window's stop every
# step's per-layer expert token counts are written, and its per-token ids when
# it has at most ROUTING_IDS_MAX_TOKENS tokens (decode steps; a 4k-token prefill
# step's ids are ~3 MB per DP rank).

ROUTING_IDS_MAX_TOKENS = 1024
_routing = {"capturer": None, "pending": [], "rank": None, "write": False}


def _bind_routing(runner, tag):
    """Attach a routed-experts capturer to every MoE layer of the target model."""
    from functools import partial

    from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
        RoutedExpertsCaptureSource,
        RoutedExpertsCapturer,
    )

    capturer = RoutedExpertsCapturer(runner.scheduler_config.max_num_batched_tokens,
                                     runner.vllm_config)
    try:
        from vllm.model_executor.layers.fused_moe.layer import MoERunner
    except Exception:
        MoERunner = ()
    bound, failed = {}, {}
    for name, module in runner.model.named_modules(prefix="model"):
        try:
            if isinstance(module, RoutedExpertsCaptureSource):
                module.capture_fn = partial(capturer.capture, module.layer_id)
            elif MoERunner and isinstance(module, MoERunner):
                fn = partial(capturer.capture, module.layer_id)
                quant_method = module._quant_method
                if quant_method.is_monolithic:
                    impl = getattr(getattr(quant_method, "moe_kernel", None), "impl", None)
                    getattr(impl, "fused_experts").set_capture_fn(fn)
                else:
                    module.router.set_capture_fn(fn)
            else:
                continue
            bound[module.layer_id] = name
        except Exception as e:
            failed[name] = repr(e)
    try:
        from vllm.distributed import parallel_state as ps

        write = ps.get_tensor_model_parallel_rank() == 0  # TP ranks route the same tokens
    except Exception:
        write = True
    meta = {"rank": tag, "write": write, "layers": capturer.device_buffer.shape[1],
            "topk": capturer.device_buffer.shape[2],
            "num_experts": runner.vllm_config.model_config.get_num_experts(),
            "max_tokens": capturer.device_buffer.shape[0],
            "bound": {str(k): v for k, v in sorted(bound.items())}, "failed": failed}
    path = os.path.join(_DIR, "routing", tag, "meta.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=1)
    if bound:
        _routing.update(capturer=capturer, rank=tag, write=write)


def _routing_snapshot(step, num_tokens, t0_ns):
    """Queue this step's routing rows for the host (asynchronous)."""
    import torch

    capturer = _routing["capturer"]
    if capturer is None or not _routing["write"] or not num_tokens:
        return
    with torch.autograd.profiler.record_function("infx_routing_copy"):
        rows = capturer.device_buffer[:num_tokens].to(torch.int16)
        host = torch.empty(rows.shape, dtype=torch.int16, pin_memory=True)
        host.copy_(rows, non_blocking=True)
    _routing["pending"].append((step, num_tokens, t0_ns, host, rows))


def _routing_flush():
    """Write the window's queued routing: per-layer expert counts, and ids for small steps."""
    pending, _routing["pending"] = _routing["pending"], []
    if not pending:
        return
    import numpy as np
    import torch

    if torch.cuda.is_available():
        torch.cuda.current_stream().synchronize()  # the queued host copies
    num_experts = None
    try:
        with open(os.path.join(_DIR, "routing", _routing["rank"], "meta.json")) as f:
            num_experts = json.load(f)["num_experts"]
    except Exception:
        pass
    out_dir = os.path.join(_DIR, "routing", _routing["rank"])
    for step, num_tokens, t0_ns, host, _ in pending:
        ids = host.numpy().astype(np.int32)  # [tokens, layers, topk]; -1 marks no expert
        experts = num_experts or int(ids.max()) + 1
        counts = np.zeros((ids.shape[1], experts), dtype=np.int32)
        for layer in range(ids.shape[1]):
            valid = ids[:, layer][ids[:, layer] >= 0]
            counts[layer] = np.bincount(valid, minlength=experts)[:experts]
        arrays = {"counts": counts, "t0_ns": np.array(t0_ns), "tokens": np.array(num_tokens)}
        if num_tokens <= ROUTING_IDS_MAX_TOKENS:
            arrays["ids"] = ids.astype(np.int16)
        np.savez_compressed(os.path.join(out_dir, f"step{step:06d}.npz"), **arrays)


def _patch_model_runner(module):
    import torch

    cls = module.GPUModelRunner
    if getattr(cls, "_infx_patched", False):
        return
    orig_capture, orig_execute = cls.capture_model, cls.execute_model
    orig_sample = getattr(cls, "sample_tokens", None)
    step_counter = itertools.count()

    def capture_model(self, *args, **kwargs):
        tag = _rank_tag(self)
        _write_env(self, tag)
        try:
            _register_module_names(self)
        except Exception:
            _write_error("register module names")
        try:
            _bind_routing(self, tag)
        except Exception:
            _write_error("bind routing")
        if tag not in _capture_ranks() and "all" not in _capture_ranks():
            try:
                return orig_capture(self, *args, **kwargs)
            finally:
                _flush_all()
        try:
            from torch.profiler import ProfilerActivity, profile
            prof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_stack=True,
            )
            prof.__enter__()
            _enable_module_markers()
        except Exception:
            _write_error("capture profiler start")
            return orig_capture(self, *args, **kwargs)
        try:
            return orig_capture(self, *args, **kwargs)
        finally:
            try:
                _disable_module_markers()
                prof.__exit__(None, None, None)
                path = os.path.join(_DIR, "capture", f"{tag}.pt.trace.json.gz")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                prof.export_chrome_trace(path)
            except Exception:
                _write_error("capture profiler export")
            _flush_all()

    def execute_model(self, scheduler_output, *args, **kwargs):
        global _current_step
        if kwargs.get("dummy_run") or (args and args[1:2] == (True,)):
            return orig_execute(self, scheduler_output, *args, **kwargs)
        k = next(step_counter)
        _current_step = k
        record = None
        try:
            record = _step_record(scheduler_output)
        except Exception:
            _write_error("step record")
        _dispatch_log.calls = []
        t0 = time.time_ns()
        try:
            if not _profiling():
                return orig_execute(self, scheduler_output, *args, **kwargs)
            with torch.autograd.profiler.record_function(f"infx_step#{k}"):
                result = orig_execute(self, scheduler_output, *args, **kwargs)
            try:
                _routing_snapshot(k, record.get("total_tokens") if record else None, t0)
            except Exception:
                _write_error("routing snapshot")
            return result
        finally:
            try:
                if record is not None:
                    record.update(step=k, t0_ns=t0, t1_ns=time.time_ns(),
                                  dispatch=_dispatch_log.calls)
                    _sink("steps", _rank_tag(self)).write(record)
            except Exception:
                _write_error("step write")
            _dispatch_log.calls = None

    def sample_tokens(self, *args, **kwargs):
        if not _profiling():
            return orig_sample(self, *args, **kwargs)
        with torch.autograd.profiler.record_function("infx_sample"):
            return orig_sample(self, *args, **kwargs)

    cls.capture_model = capture_model
    cls.execute_model = execute_model
    if orig_sample is not None:
        cls.sample_tokens = sample_tokens
    cls._infx_patched = True


# --- Python launcher markers --------------------------------------------------
# Kernels launched straight from Python (Triton, TileLang, CuTe DSL, DeepGEMM
# and FlashInfer bindings) have no torch op. Their launch entry points mark
# themselves while profiling: infx_py#<launcher>#<vllm frame>|<vllm frame>...,
# the frames being the vLLM callers above the launch, outermost first.

_CALLER_FRAMES = 6
_WRAP_SKIP_PREFIXES = ("_", "is_", "has_", "should_", "supports")


def _compiling():
    try:
        import torch

        return torch.compiler.is_compiling()
    except Exception:
        return False


_OWN_FILE = os.path.abspath(__file__)


def _vllm_callers(skip):
    frame = sys._getframe(skip)
    chain = []
    while frame is not None and len(chain) < _CALLER_FRAMES:
        path = frame.f_code.co_filename
        cut = path.rfind(f"{os.sep}vllm{os.sep}")
        if cut >= 0 and os.path.abspath(path) != _OWN_FILE:
            chain.append(f"{path[cut + 1:]}:{frame.f_lineno}:{frame.f_code.co_name}")
        frame = frame.f_back
    return chain[::-1]


def _launcher_marker(label):
    import torch

    callers = "|".join(_vllm_callers(3))
    return torch.autograd.profiler.record_function(f"infx_py#{label}#{callers}")


def _wrap_launcher(fn, label):
    import functools

    @functools.wraps(fn)
    def launcher(*args, **kwargs):
        if _compiling() or not _profiling():
            return fn(*args, **kwargs)
        with _launcher_marker(label):
            return fn(*args, **kwargs)

    launcher._infx_launcher = True
    return launcher


def _patch_launcher_module(module):
    """Mark every public Python function a launcher module defines."""
    import types

    for name, obj in list(vars(module).items()):
        if (isinstance(obj, types.FunctionType) and obj.__module__ == module.__name__
                and not name.startswith(_WRAP_SKIP_PREFIXES)
                and not getattr(obj, "_infx_launcher", False)):
            setattr(module, name, _wrap_launcher(obj, f"{module.__name__}.{name}"))


def _patch_launcher_calls(module, attr, label_of):
    """Mark calls of every class in `module` that defines `attr` itself."""
    for cls in list(vars(module).values()):
        if not isinstance(cls, type) or cls.__module__ != module.__name__ or attr not in vars(cls):
            continue
        orig = vars(cls)[attr]
        if getattr(orig, "_infx_launcher", False):
            continue

        def call(self, *args, _orig=orig, **kwargs):
            if _compiling() or not _profiling():
                return _orig(self, *args, **kwargs)
            try:
                label = label_of(self)
            except Exception:
                label = type(self).__qualname__
            with _launcher_marker(label):
                return _orig(self, *args, **kwargs)

        call._infx_launcher = True
        setattr(cls, attr, call)


def _triton_label(jit_function):
    fn = getattr(jit_function, "fn", None)
    return f"triton:{getattr(fn, '__qualname__', None) or type(jit_function).__qualname__}"


def _named_label(prefix):
    def label(obj):
        for attr in ("kernel_name", "name", "__name__", "func_name"):
            value = getattr(obj, attr, None)
            if isinstance(value, str) and value:
                return f"{prefix}:{value}"
        return f"{prefix}:{type(obj).__qualname__}"

    return label


_LAUNCHER_HOOKS = {
    "vllm.utils.deep_gemm": _patch_launcher_module,
    "vllm.utils.flashinfer": _patch_launcher_module,
    "vllm.third_party.deep_gemm.mega": _patch_launcher_module,
    "triton.runtime.jit": lambda m: _patch_launcher_calls(m, "run", _triton_label),
    "tilelang.jit.kernel": lambda m: _patch_launcher_calls(m, "__call__", _named_label("tilelang")),
    "cutlass.cutlass_dsl.tvm_ffi_provider": lambda m: _patch_launcher_calls(
        m, "__call__", _named_label("cute")),
}


def _patch_gpu_worker(module):
    """Idle DP ranks run a dummy forward each step to keep EP collectives in step."""
    import torch

    cls = module.Worker
    if getattr(cls, "_infx_patched", False) or not hasattr(cls, "execute_dummy_batch"):
        return
    orig = cls.execute_dummy_batch
    counter = itertools.count()

    def execute_dummy_batch(self, *args, **kwargs):
        k = next(counter)
        t0 = time.time_ns()
        # vLLM advances its profiler only for scheduled steps, so an idle DP
        # rank's window never reaches max_iterations and records until the
        # window is stopped. Its dummy forwards are its steps: count them.
        profiler = getattr(self, "profiler", None)
        if profiler is not None and hasattr(profiler, "step"):
            try:
                profiler.step()
            except Exception:
                _write_error("dummy profiler step")
        try:
            if not _profiling():
                return orig(self, *args, **kwargs)
            with torch.autograd.profiler.record_function(f"infx_dummy#{k}"):
                return orig(self, *args, **kwargs)
        finally:
            try:
                _sink("steps", _rank_tag(getattr(self, "model_runner", None))).write(
                    {"dummy": k, "t0_ns": t0, "t1_ns": time.time_ns()})
            except Exception:
                _write_error("dummy step write")

    cls.execute_dummy_batch = execute_dummy_batch
    cls._infx_patched = True


# --- CPU KV offload copies ----------------------------------------------------
# SimpleCPUOffload moves KV blocks with cuMemcpyBatchAsync, a driver call Kineto
# does not record, issued from its own copy thread, so those memcpys have no CPU
# launch to join. Each launch_copy queues one copy_blocks, one memcpy; the log
# of them lets the extractor assign the memcpys in queue order.

def _patch_copy_backend(module):
    cls = getattr(module, "DmaCopyBackend", None)
    if cls is None or getattr(cls, "_infx_patched", False) or not hasattr(cls, "launch_copy"):
        return
    orig = cls.launch_copy
    block_bytes = {}  # id(params) -> bytes of one block across all layers

    def launch_copy(self, src_blocks, dst_blocks, is_store, *args, **kwargs):
        try:
            params = self._store_params if is_store else self._load_params
            if id(params) not in block_bytes:
                block_bytes[id(params)] = int(sum(int(b) for b in params.bpb))
            _sink("copies", _rank_tag()).write({
                "t_ns": time.time_ns(), "store": bool(is_store), "blocks": len(src_blocks),
                "bytes": block_bytes[id(params)] * len(src_blocks), "step": _current_step,
                "callers": _vllm_callers(2),
            })
        except Exception:
            _write_error("copy log")
        return orig(self, src_blocks, dst_blocks, is_store, *args, **kwargs)

    cls.launch_copy = launch_copy
    cls._infx_patched = True


# --- post-import hooks ------------------------------------------------------

_HOOKS = {
    "torch.cuda.graphs": _patch_cuda_graph,
    "vllm.v1.worker.gpu.dp_utils": _patch_dp_utils,
    "vllm.v1.worker.gpu.model_runner": _patch_model_runner,
    "vllm.profiler.wrapper": _patch_profiler_wrapper,
    "vllm.compilation.piecewise_backend": _patch_piecewise_backend,
    "vllm.v1.worker.gpu_worker": _patch_gpu_worker,
    "vllm.v1.simple_kv_offload.copy_backend": _patch_copy_backend,
    **_LAUNCHER_HOOKS,
}


class _PostImportFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in _HOOKS:
            return None
        for finder in sys.meta_path:
            if finder is self:
                continue
            find = getattr(finder, "find_spec", None)
            if find is None:
                continue
            spec = find(name, path, target)
            if spec is not None and spec.loader is not None:
                break
        else:
            return None
        loader = spec.loader
        exec_module = loader.exec_module

        def exec_and_patch(module, _exec=exec_module, _name=name):
            _exec(module)
            try:
                _HOOKS[_name](module)
            except Exception:
                _write_error(f"patch {_name}")

        loader.exec_module = exec_and_patch
        return spec


if _DIR:
    for _name, _hook in _HOOKS.items():
        if _name in sys.modules:
            try:
                _hook(sys.modules[_name])
            except Exception:
                _write_error(f"patch {_name}")
    sys.meta_path.insert(0, _PostImportFinder())
    import atexit

    atexit.register(_flush_all)

try:
    _chain_next_sitecustomize()
except Exception:
    if _DIR:
        _write_error("chain sitecustomize")
