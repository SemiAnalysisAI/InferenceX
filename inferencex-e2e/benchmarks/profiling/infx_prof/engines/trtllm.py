"""TensorRT-LLM PyTorch-backend adapter (trtllm-serve; one MPI process per rank)."""

import itertools
import os
import sys
import threading

from .. import envinfo, graphs, launchers, modules, routing, runtime, steps, windows
from .base import Engine

_rank = {"tag": None}
_tls = threading.local()


def rank_tag():
    return _rank["tag"] or f"pid{os.getpid()}"


def _tag(mapping):
    tp = getattr(mapping, "tp_rank", 0) or 0
    pp = getattr(mapping, "pp_rank", 0) or 0
    return f"dp0_tp{tp}" if not pp else f"dp0_pp{pp}_tp{tp}"


def setup_routing(engine):
    """Allocate the routing buffer once, before the first warmup captures graphs that write it."""
    if routing.active():
        return
    config = getattr(getattr(engine, "model", None), "config", None)
    config = getattr(config, "text_config", None) or config
    num_layers = getattr(config, "num_hidden_layers", None)
    topk = getattr(config, "num_experts_per_tok", None) or getattr(config, "moe_topk", None)
    experts = (getattr(config, "n_routed_experts", None) or getattr(config, "num_experts", None)
               or getattr(config, "num_local_experts", None))
    if not (num_layers and topk and experts):
        return
    mapping = getattr(engine, "mapping", None)
    adp = bool(getattr(mapping, "enable_attention_dp", False))
    write = adp or (getattr(mapping, "tp_rank", 0) or 0) == 0
    max_tokens = int(getattr(engine, "max_num_tokens", 0) or 8192)
    routing.setup(rank_tag(), max_tokens, int(num_layers), int(topk), int(experts), write=write,
                  extra={"source": "TRT-LLM routing_method.apply, else recomputed from router logits"})


def patch_py_executor(module):
    cls = getattr(module, "PyExecutor", None)
    if cls is None or getattr(cls, "_infx_patched", False):
        return
    orig_init, orig_step = cls.__init__, cls._forward_step
    counter = itertools.count()

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        try:
            engine = self.model_engine
            _rank["tag"] = _tag(getattr(engine, "mapping", None))
            modules.register_root("model", getattr(engine, "model", None))
            draft = getattr(self, "draft_model_engine", None)
            if draft is not None:
                modules.register_root("draft", getattr(draft, "model", None))
            envinfo.write(rank_tag(), ("tensorrt_llm",), ("TLLM_", "TRTLLM_"),
                          {"model_engine_class": f"{type(engine).__module__}.{type(engine).__qualname__}",
                           "llm_args": str(getattr(engine, "llm_args", None))})
        except Exception:
            runtime.write_error("trtllm executor init hooks")

    def _forward_step(self, scheduled_requests, *args, **kwargs):
        if getattr(self, "is_warmup", False):
            return orig_step(self, scheduled_requests, *args, **kwargs)
        try:
            record = requests_record(scheduled_requests)
        except Exception:
            runtime.write_error("trtllm step record")
            record = {}
        kind = "dummy" if record.get("phase") == "idle" else "step"
        _tls.full = False
        replays = graphs.replays()
        with steps.Step(rank_tag(), next(counter), record, kind=kind) as rec:
            try:
                return orig_step(self, scheduled_requests, *args, **kwargs)
            finally:
                rec["graph"] = ("FULL" if _tls.full else "PIECEWISE" if graphs.replays() > replays else "NONE")
                rec["iter"] = getattr(self, "iter_counter", None)

    cls.__init__ = __init__
    cls._forward_step = _forward_step
    cls._infx_patched = True


def _draft_len(req):
    for attr in ("draft_tokens", "py_draft_tokens"):
        try:
            tokens = getattr(req, attr, None)
            if tokens is not None:
                return len(tokens)
        except Exception:
            continue
    return 0


def requests_record(scheduled):
    rows, total = [], 0
    context = list(getattr(scheduled, "context_requests", None) or ())
    generation = list(getattr(scheduled, "generation_requests", None) or ())
    dummies = 0
    for req, is_context in [(r, True) for r in context] + [(r, False) for r in generation]:
        if getattr(req, "is_dummy", False) or getattr(req, "is_attention_dp_dummy", False) \
                or getattr(req, "is_cuda_graph_dummy", False):
            dummies += 1
            continue
        if is_context:
            draft = 0
            scheduled_tokens = int(getattr(req, "context_chunk_size", 0) or 0)
            computed = int(getattr(req, "context_current_position", 0) or 0)
        else:
            draft = _draft_len(req)
            scheduled_tokens = 1 + draft
            try:
                computed = int(req.get_num_tokens(0))
            except Exception:
                computed = None
        total += scheduled_tokens
        rows.append({"req": getattr(req, "py_request_id", None), "scheduled": scheduled_tokens, "computed": computed,
                     "prompt_len": getattr(req, "py_prompt_len", None), "new": is_context and not computed,
                     "spec": draft})
    phase = ("idle" if not rows and dummies else "mixed" if context and generation
             else "prefill" if context else "decode")
    return {"phase": phase, "total_tokens": total, "reqs": rows, "dummies": dummies}


def patch_model_engine(module):
    cls = getattr(module, "PyTorchModelEngine", None)
    if cls is None or getattr(cls, "_infx_patched", False) or not hasattr(cls, "warmup"):
        return
    orig = cls.warmup

    def warmup(self, *args, **kwargs):
        if getattr(self, "is_draft_model", False):
            return orig(self, *args, **kwargs)
        try:
            _rank["tag"] = _tag(getattr(self, "mapping", None))
            setup_routing(self)
        except Exception:
            runtime.write_error("trtllm routing setup")
        with windows.capture_session(rank_tag()):
            return orig(self, *args, **kwargs)

    cls.warmup = warmup
    cls._infx_patched = True


def patch_cuda_graph_runner(module):
    cls = getattr(module, "CUDAGraphRunner", None)
    if cls is None or getattr(cls, "_infx_patched", False) or not hasattr(cls, "replay"):
        return
    orig = cls.replay

    def replay(self, *args, **kwargs):
        _tls.full = True
        return orig(self, *args, **kwargs)

    cls.replay = replay
    cls._infx_patched = True


def _recompute_ids(method, logits):
    """Top-k ids of routing methods with a closed form, else None. Selection only: weights are not needed."""
    import torch

    name = type(method).__name__
    k = getattr(method, "top_k", None)
    if not k:
        return None
    bias = getattr(method, "e_score_correction_bias", None)
    if bias is not None and getattr(method, "n_group", 1) in (None, 1):
        scores = torch.sigmoid(logits) + bias
    elif name in ("DefaultMoeRoutingMethod", "RenormalizeMoeRoutingMethod", "RenormalizeNaiveMoeRoutingMethod"):
        scores = logits.float()
    else:
        return None
    return torch.topk(scores, int(k), dim=-1).indices.int()


def patch_moe_backends(module):
    """Every MoE class's forward_impl in this module names its layer and fills in fused routing."""
    interface = __import__("sys").modules.get("tensorrt_llm._torch.modules.fused_moe.interface")
    base = getattr(interface, "MoE", None)
    for cls in list(vars(module).values()):
        if not isinstance(cls, type) or cls.__module__ != module.__name__ or "forward_impl" not in vars(cls):
            continue
        if base is not None and not issubclass(cls, base) and cls.__name__ != "ConfigurableMoE":
            continue
        orig = vars(cls)["forward_impl"]
        if getattr(orig, "_infx_patched", False):
            continue

        def forward_impl(self, x, router_logits, *args, _orig=orig, **kwargs):
            layer = getattr(self, "layer_idx", None)
            if layer is None or getattr(_tls, "layer", None) is not None:
                return _orig(self, x, router_logits, *args, **kwargs)
            _tls.layer, _tls.routed = layer, False
            try:
                result = _orig(self, x, router_logits, *args, **kwargs)
                if not _tls.routed and routing.active() and not runtime.compiling():
                    try:
                        with launchers.marker("infx_prof.routing.recompute", skip=2):
                            ids = _recompute_ids(getattr(self, "routing_method", None), router_logits)
                            if ids is not None:
                                routing.capture(int(layer), ids)
                    except Exception:
                        runtime.write_error("trtllm routing recompute")
                return result
            finally:
                _tls.layer = None

        forward_impl._infx_patched = True
        cls.forward_impl = forward_impl


def patch_routing_methods(module):
    base = getattr(module, "BaseMoeRoutingMethod", None)
    if base is None:
        return
    for cls in [base] + [c for c in vars(module).values() if isinstance(c, type) and issubclass(c, base)]:
        orig = vars(cls).get("apply")
        if orig is None or getattr(orig, "_infx_patched", False):
            continue

        def apply(self, *args, _orig=orig, **kwargs):
            result = _orig(self, *args, **kwargs)
            layer = getattr(_tls, "layer", None)
            if layer is not None and routing.active():
                try:
                    routing.capture(int(layer), result[0])
                    _tls.routed = True
                except Exception:
                    pass
            return result

        apply._infx_patched = True
        cls.apply = apply


def patch_thop(module):
    """The C++ attention entry points TRT-LLM's FMHA libraries call as nanobind functions, not torch ops."""
    internal = getattr(getattr(module, "bindings", None), "internal", None)
    thop = getattr(internal, "thop", None)
    if thop is not None:
        launchers.wrap_builtins(thop, "tensorrt_llm.bindings.internal.thop")
    kv_utils = getattr(getattr(internal, "batch_manager", None), "kv_cache_manager_v2_utils", None)
    if kv_utils is not None:
        launchers.wrap_builtins(kv_utils, "tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2_utils")


def patch_kv_copies(module):
    """KV cache manager v2's block-offset and host-tier copies (nanobind functions bound by name)."""
    launchers.wrap_builtins(module, module.__name__, names={
        "copy_batch_block_offsets_to_device", "copy_host_to_device", "copy_device_to_host",
        "copy_device_to_device"})


class TrtllmEngine(Engine):
    name = "trtllm"
    caller_roots = ("tensorrt_llm",)
    launcher_packages = ("flashinfer", "fmha_sm100", "tensorrt_llm.deep_gemm", "tensorrt_llm._torch.cute_dsl_kernels",
                         "tensorrt_llm._torch.cuda_tile_kernels", "tensorrt_llm._torch.attention_backend.fmha")
    version_packages = ("tensorrt_llm",)
    env_prefixes = ("TLLM_", "TRTLLM_")
    window_control = "file"

    def install(self, registry):
        registry.on("tensorrt_llm._torch.pyexecutor.py_executor", patch_py_executor)
        registry.on("tensorrt_llm._torch.pyexecutor.model_engine", patch_model_engine)
        registry.on("tensorrt_llm._torch.pyexecutor.cuda_graph_runner", patch_cuda_graph_runner)
        registry.on_package("tensorrt_llm._torch.modules.fused_moe", patch_moe_backends)
        registry.on("tensorrt_llm._torch.modules.fused_moe.routing", patch_routing_methods)
        registry.on("tensorrt_llm", patch_thop)
        registry.on("tensorrt_llm._torch.attention_backend.trtllm", lambda m: patch_thop(sys.modules["tensorrt_llm"]))
        registry.on("tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2", patch_kv_copies)
        registry.on("tensorrt_llm.runtime.kv_cache_manager_v2._copy_engine", patch_kv_copies)


ENGINE = TrtllmEngine()
