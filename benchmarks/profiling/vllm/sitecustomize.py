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
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._f = open(path, "a", buffering=1 << 16)
        self._lock = threading.Lock()
        self._n = 0

    def write(self, record):
        line = json.dumps(record, separators=(",", ":"), default=str)
        with self._lock:
            self._f.write(line + "\n")
            self._n += 1
            if self._n % 256 == 0:
                self._f.flush()

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
        if tag not in _capture_ranks() and "all" not in _capture_ranks():
            return orig_capture(self, *args, **kwargs)
        try:
            from torch.profiler import ProfilerActivity, profile
            prof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_stack=True,
            )
            prof.__enter__()
        except Exception:
            _write_error("capture profiler start")
            return orig_capture(self, *args, **kwargs)
        try:
            return orig_capture(self, *args, **kwargs)
        finally:
            try:
                prof.__exit__(None, None, None)
                path = os.path.join(_DIR, "capture", f"{tag}.pt.trace.json.gz")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                prof.export_chrome_trace(path)
            except Exception:
                _write_error("capture profiler export")
            _flush_all()

    def execute_model(self, scheduler_output, *args, **kwargs):
        if kwargs.get("dummy_run") or (args and args[1:2] == (True,)):
            return orig_execute(self, scheduler_output, *args, **kwargs)
        k = next(step_counter)
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
                return orig_execute(self, scheduler_output, *args, **kwargs)
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


# --- post-import hooks ------------------------------------------------------

_HOOKS = {
    "torch.cuda.graphs": _patch_cuda_graph,
    "vllm.v1.worker.gpu.dp_utils": _patch_dp_utils,
    "vllm.v1.worker.gpu.model_runner": _patch_model_runner,
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
