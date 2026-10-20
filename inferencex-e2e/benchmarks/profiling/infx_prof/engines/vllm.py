"""vLLM adapter (V1 and V2 GPU model runners)."""

import functools
import inspect
import itertools
import os
import threading
import time

from .. import envinfo, graphs, launchers, modules, routing, runtime, steps, windows
from .base import Engine

_dispatch_log = threading.local()
_moe_layer = threading.local()
_current_step = None


def rank_tag(runner=None):
    """dp<d>_tp<t> for the calling worker; pid when ranks are not set up."""
    try:
        from vllm.distributed import parallel_state as ps

        tp = ps.get_tensor_model_parallel_rank()
        dp = runner.parallel_config.data_parallel_rank if runner is not None else ps.get_dp_group().rank_in_group
        return f"dp{dp}_tp{tp}"
    except Exception:
        return f"pid{os.getpid()}"


def _forward_context():
    from vllm.forward_context import get_forward_context, is_forward_context_available

    if not is_forward_context_available():
        return None
    ctx = get_forward_context()
    return {"cudagraph_runtime_mode": str(getattr(ctx, "cudagraph_runtime_mode", None)),
            "batch_descriptor": repr(getattr(ctx, "batch_descriptor", None))}


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


def patch_dp_utils(module):
    orig = module.dispatch_cg_and_sync_dp
    if getattr(orig, "_infx_patched", False):
        return

    def dispatch_cg_and_sync_dp(*args, **kwargs):
        result = orig(*args, **kwargs)
        calls = getattr(_dispatch_log, "calls", None)
        if calls is not None:
            calls.append(_batch_desc_record(result))
        return result

    dispatch_cg_and_sync_dp._infx_patched = True
    module.dispatch_cg_and_sync_dp = dispatch_cg_and_sync_dp


def patch_cudagraph_dispatcher(module):
    """V1 dispatches before and again after the DP batch sync; the last call is the step's mode."""
    cls = getattr(module, "CudagraphDispatcher", None)
    if cls is None or getattr(cls, "_infx_patched", False):
        return
    orig = cls.dispatch

    def dispatch(self, *args, **kwargs):
        result = orig(self, *args, **kwargs)
        try:
            calls = getattr(_dispatch_log, "calls", None)
            if calls is not None and getattr(_dispatch_log, "v1", False):
                mode, batch_desc = result
                record = {"batch_desc": repr(batch_desc), "cg_mode": str(getattr(mode, "name", mode))}
                for field in ("num_tokens", "num_reqs", "uniform"):
                    if hasattr(batch_desc, field):
                        record[field] = str(getattr(batch_desc, field))
                calls.append(record)
        except Exception:
            pass
        return result

    cls.dispatch = dispatch
    cls._infx_patched = True


def step_record(scheduler_output):
    reqs, computed, prompt_len = [], {}, {}
    for new in scheduler_output.scheduled_new_reqs:
        computed[new.req_id] = new.num_computed_tokens
        ids = getattr(new, "prompt_token_ids", None)
        prompt_len[new.req_id] = len(ids) if ids is not None else None
    cached = scheduler_output.scheduled_cached_reqs
    for req_id, n in zip(cached.req_ids, cached.num_computed_tokens):
        computed[req_id] = n
    spec = scheduler_output.scheduled_spec_decode_tokens or {}
    for req_id, n in scheduler_output.num_scheduled_tokens.items():
        reqs.append({"req": req_id, "scheduled": n, "computed": computed.get(req_id),
                     "spec": len(spec.get(req_id, ())), "new": req_id in prompt_len,
                     "prompt_len": prompt_len.get(req_id)})
    return {"total_tokens": scheduler_output.total_num_scheduled_tokens, "reqs": reqs}


def register_models(runner):
    modules.register_root("model", getattr(runner, "model", None))
    speculator = getattr(runner, "speculator", None) or getattr(runner, "drafter", None)
    if speculator is not None:
        modules.register_root("draft", getattr(speculator, "model", None))


def bind_routing(runner, tag):
    """Route every MoE layer's top-k ids of the target model to the core capturer."""
    model_config = runner.vllm_config.model_config
    try:
        from vllm.model_executor.layers.fused_moe.layer import MoERunner
    except Exception:
        MoERunner = ()
    bound, failed, sources = {}, {}, []
    for name, module in runner.model.named_modules(prefix="model"):
        try:
            fn = functools.partial(routing.capture, module.layer_id) if hasattr(module, "layer_id") else None
            if MoERunner and isinstance(module, MoERunner):
                quant_method = module._quant_method
                if quant_method.is_monolithic:
                    impl = getattr(getattr(quant_method, "moe_kernel", None), "impl", None)
                    experts = getattr(impl, "fused_experts")
                    supports = getattr(experts, "supports_routing_replay_capture", None)
                    if supports is not None and not supports():
                        bind_aiter_routing(experts, module.layer_id)
                    else:
                        sources.append((experts.set_capture_fn, fn))
                else:
                    sources.append((module.router.set_capture_fn, fn))
            elif hasattr(module, "capture_fn") and fn is not None:
                sources.append((functools.partial(setattr, module, "capture_fn"), fn))
            else:
                continue
            bound[module.layer_id] = name
        except Exception as e:
            failed[name] = repr(e)
    try:
        from vllm.distributed import parallel_state as ps

        write = ps.get_tensor_model_parallel_rank() == 0
    except Exception:
        write = True
    routing.setup(tag, runner.scheduler_config.max_num_batched_tokens, model_config.get_total_num_hidden_layers(),
                  model_config.get_num_experts_per_tok(), model_config.get_num_experts(), write=write,
                  bound=bound, failed=failed)
    for set_fn, fn in sources:
        set_fn(fn)


def _aiter_routing_module():
    import importlib

    for name in ("aiter.ops.triton.moe.moe_routing.routing", "aiter.ops.triton.moe_routing.routing"):
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    return None


def _recompute_aiter_ids(gating_output, topk, score_mode="softmax", bias=None):
    """AITER routing()'s expert selection: top-k of the scores (plus the correction bias)."""
    import torch

    logits = gating_output.float()
    if score_mode == "sigmoid":
        scores = torch.sigmoid(logits)
    elif score_mode == "sqrtsoftplus":
        scores = torch.sqrt(torch.nn.functional.softplus(logits))
    elif score_mode == "softmax":
        scores = logits
    else:
        return None
    if bias is not None:
        scores = scores + bias.float()
    return torch.topk(scores, int(topk), dim=-1).indices.int()


def patch_aiter_routing():
    mod = _aiter_routing_module()
    if mod is None:
        raise RuntimeError("AITER triton routing module not found")
    orig = mod.routing
    if getattr(orig, "_infx_patched", False):
        return

    @functools.wraps(orig)
    def routing_(gating_output, topk, *args, **kwargs):
        out = orig(gating_output, topk, *args, **kwargs)
        layer = getattr(_moe_layer, "id", None)
        if layer is not None and routing.active():
            try:
                ids = getattr(out[0] if isinstance(out, tuple) else out, "topk_ids", None)
                if ids is None:
                    with launchers.marker("infx_prof.routing.recompute", skip=2):
                        ids = _recompute_aiter_ids(gating_output, topk, kwargs.get("score_mode", "softmax"),
                                                   kwargs.get("bias"))
                if ids is not None:
                    routing.capture(int(layer), ids)
            except Exception:
                runtime.write_error("aiter routing capture")
        return out

    routing_._infx_patched = True
    mod.routing = routing_


def bind_aiter_routing(experts, layer_id):
    """Name the layer around a monolithic AITER MoE kernel's apply; its routing() call captures."""
    patch_aiter_routing()
    orig = experts.apply
    if getattr(orig, "_infx_patched", False):
        return

    @functools.wraps(orig)
    def apply(*args, **kwargs):
        _moe_layer.id = layer_id
        try:
            return orig(*args, **kwargs)
        finally:
            _moe_layer.id = None

    apply._infx_patched = True
    experts.apply = apply


def patch_model_runner(module, v1=False):
    import torch

    cls = module.GPUModelRunner
    if vars(cls).get("_infx_patched", False):
        return
    orig_capture, orig_execute = cls.capture_model, cls.execute_model
    orig_sample = getattr(cls, "sample_tokens", None)
    step_counter = itertools.count()

    def capture_model(self, *args, **kwargs):
        tag = rank_tag(self)
        envinfo.write(tag, ("vllm",), ("VLLM_",),
                      {"runner_class": f"{type(self).__module__}.{type(self).__qualname__}",
                       "vllm_config": str(getattr(self, "vllm_config", None))})
        for what, fn in (("register module names", register_models), ("bind routing", lambda r: bind_routing(r, tag))):
            try:
                fn(self)
            except Exception:
                runtime.write_error(what)
        with windows.capture_session(tag):
            return orig_capture(self, *args, **kwargs)

    def execute_model(self, scheduler_output, *args, **kwargs):
        global _current_step
        if kwargs.get("dummy_run") or (args and args[1:2] == (True,)):
            return orig_execute(self, scheduler_output, *args, **kwargs)
        k = next(step_counter)
        _current_step = k
        try:
            record = step_record(scheduler_output)
        except Exception:
            runtime.write_error("step record")
            record = {}
        _dispatch_log.calls, _dispatch_log.v1 = [], v1
        try:
            with steps.Step(rank_tag(self), k, record) as rec:
                try:
                    return orig_execute(self, scheduler_output, *args, **kwargs)
                finally:
                    calls = _dispatch_log.calls or []
                    rec["dispatch"] = calls
                    if calls:
                        rec["graph"] = calls[-1].get("cg_mode")
        finally:
            _dispatch_log.calls = None

    def sample_tokens(self, *args, **kwargs):
        if not runtime.profiling():
            return orig_sample(self, *args, **kwargs)
        with torch.autograd.profiler.record_function("infx_sample"):
            return orig_sample(self, *args, **kwargs)

    cls.capture_model = capture_model
    cls.execute_model = execute_model
    if orig_sample is not None:
        cls.sample_tokens = sample_tokens
    cls._infx_patched = True


def patch_gpu_worker(module):
    """Idle DP ranks run a dummy forward each step to keep EP collectives in step."""
    cls = module.Worker
    if getattr(cls, "_infx_patched", False) or not hasattr(cls, "execute_dummy_batch"):
        return
    orig = cls.execute_dummy_batch
    counter = itertools.count()

    def execute_dummy_batch(self, *args, **kwargs):
        profiler = getattr(self, "profiler", None)
        if profiler is not None and hasattr(profiler, "step"):
            try:
                profiler.step()
            except Exception:
                runtime.write_error("dummy profiler step")
        with steps.Step(rank_tag(getattr(self, "model_runner", None)), next(counter), kind="dummy"):
            return orig(self, *args, **kwargs)

    cls.execute_dummy_batch = execute_dummy_batch
    cls._infx_patched = True


def patch_profiler_wrapper(module):
    cls = module.WorkerProfiler
    if getattr(cls, "_infx_patched", False):
        return
    orig_start, orig_stop = cls._call_start, cls._call_stop

    def _call_start(self, *args, **kwargs):
        result = orig_start(self, *args, **kwargs)
        windows.window_opened(rank_tag())
        return result

    def _call_stop(self, *args, **kwargs):
        windows.window_closing(rank_tag())
        return orig_stop(self, *args, **kwargs)

    cls._call_start = _call_start
    cls._call_stop = _call_stop
    cls._infx_patched = True


def patch_piecewise_backend(module):
    import torch

    cls = module.PiecewiseBackend
    if getattr(cls, "_infx_patched", False):
        return
    orig_call = cls.__call__

    def __call__(self, *args):
        if not runtime.profiling():
            return orig_call(self, *args)
        tag = str(getattr(getattr(self, "vllm_backend", None), "prefix", "") or "")
        modules.set_compiled_root("draft" if any(k in tag for k in ("eagle", "draft", "mtp", "spec", "dspark"))
                                  else "model")
        try:
            with torch.autograd.profiler.record_function(f"infx_piece#{getattr(self, 'piecewise_compile_index', '?')}"):
                return orig_call(self, *args)
        finally:
            modules.close_compiled_module()

    cls.__call__ = __call__
    cls._infx_patched = True


def _mark_layer_op(op_func):
    """An op called with layer_name runs that layer without calling it: mark the launches as its."""
    try:
        params = inspect.signature(op_func).parameters
    except (TypeError, ValueError):
        return op_func
    if "layer_name" not in params or getattr(op_func, "_infx_patched", False):
        return op_func
    position = list(params).index("layer_name")

    @functools.wraps(op_func)
    def op(*args, **kwargs):
        if not runtime.profiling() or runtime.compiling():
            return op_func(*args, **kwargs)
        name = kwargs.get("layer_name", args[position] if len(args) > position else None)
        with modules.marked(modules.resolve(str(name)) if name else op_func.__name__, args):
            return op_func(*args, **kwargs)

    op._infx_patched = True
    return runtime.keep_signature(op, op_func)


def patch_custom_op_registry(module):
    orig = getattr(module, "direct_register_custom_op", None)
    if orig is None or getattr(orig, "_infx_patched", False):
        return

    @functools.wraps(orig)
    def direct_register_custom_op(op_name, op_func, *args, **kwargs):
        return orig(op_name, _mark_layer_op(op_func), *args, **kwargs)

    direct_register_custom_op._infx_patched = True
    module.direct_register_custom_op = direct_register_custom_op


def patch_copy_backend(module):
    """SimpleCPUOffload copies are driver memcpys from its copy thread: log each queued copy."""
    cls = getattr(module, "DmaCopyBackend", None)
    if cls is None or getattr(cls, "_infx_patched", False) or not hasattr(cls, "launch_copy"):
        return
    orig = cls.launch_copy
    block_bytes = {}

    def launch_copy(self, src_blocks, dst_blocks, is_store, *args, **kwargs):
        try:
            params = self._store_params if is_store else self._load_params
            if id(params) not in block_bytes:
                block_bytes[id(params)] = int(sum(int(b) for b in params.bpb))
            runtime.sink("copies", rank_tag()).write({
                "t_ns": time.time_ns(), "store": bool(is_store), "blocks": len(src_blocks),
                "bytes": block_bytes[id(params)] * len(src_blocks), "step": _current_step,
                "callers": launchers.callers(2)})
        except Exception:
            runtime.write_error("copy log")
        return orig(self, src_blocks, dst_blocks, is_store, *args, **kwargs)

    cls.launch_copy = launch_copy
    cls._infx_patched = True


class VllmEngine(Engine):
    name = "vllm"
    caller_roots = ("vllm",)
    launcher_modules = ("vllm.utils.deep_gemm", "vllm.utils.flashinfer")
    launcher_packages = ("vllm.third_party", "flashinfer")
    version_packages = ("vllm",)
    env_prefixes = ("VLLM_",)
    window_control = "engine"

    def install(self, registry):
        graphs.capture_context = _forward_context
        for name, fn in {
            "vllm.v1.worker.gpu.dp_utils": patch_dp_utils,
            "vllm.v1.worker.gpu.model_runner": patch_model_runner,
            "vllm.v1.worker.gpu_model_runner": functools.partial(patch_model_runner, v1=True),
            "vllm.v1.cudagraph_dispatcher": patch_cudagraph_dispatcher,
            "vllm.profiler.wrapper": patch_profiler_wrapper,
            "vllm.compilation.piecewise_backend": patch_piecewise_backend,
            "vllm.v1.worker.gpu_worker": patch_gpu_worker,
            "vllm.v1.simple_kv_offload.copy_backend": patch_copy_backend,
            "vllm.utils.torch_utils": patch_custom_op_registry,
            "vllm.utils": patch_custom_op_registry,
        }.items():
            registry.on(name, fn)


ENGINE = VllmEngine()
