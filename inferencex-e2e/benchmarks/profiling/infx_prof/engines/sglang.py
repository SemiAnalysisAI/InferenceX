"""SGLang adapter (one scheduler process per TP rank; v0.5.12 through the runner/ refactor and main)."""

import functools
import inspect
import itertools
import os
import re
import sys
import threading

from .. import envinfo, graphs, launchers, modules, routing, runtime, steps, windows
from .base import Engine

_rank = {"tag": None, "dp": 0}
_tls = threading.local()


def rank_tag():
    return _rank["tag"] or f"pid{os.getpid()}"


def patch_scheduler_module(module):
    run_process = getattr(module, "run_scheduler_process", None)
    if run_process is not None and not getattr(run_process, "_infx_patched", False):
        signature = inspect.signature(run_process)

        @functools.wraps(run_process)
        def run_scheduler_process(*args, **kwargs):
            try:
                bound = signature.bind_partial(*args, **kwargs).arguments
                dp = bound.get("dp_rank")
                if dp is None:
                    dp = os.environ.get("SGLANG_DP_RANK")
                _rank["dp"] = int(dp or 0)
                _rank["tag"] = f"dp{_rank['dp']}_tp{bound.get('tp_rank') or 0}"
            except Exception:
                runtime.write_error("sglang rank")
            return run_process(*args, **kwargs)

        run_scheduler_process._infx_patched = True
        module.run_scheduler_process = run_scheduler_process

    cls = getattr(module, "Scheduler", None)
    if cls is None or getattr(cls, "_infx_patched", False) or not hasattr(cls, "run_batch"):
        return
    orig = cls.run_batch
    counter = itertools.count()

    def run_batch(self, batch, *args, **kwargs):
        try:
            record = batch_record(batch)
        except Exception:
            runtime.write_error("sglang step record")
            record = {}
        kind = "dummy" if record.get("phase") == "idle" else "step"
        _tls.graph, _tls.forward = "NONE", None
        with steps.Step(rank_tag(), next(counter), record, kind=kind) as rec:
            try:
                return orig(self, batch, *args, **kwargs)
            finally:
                rec["graph"] = _tls.graph
                rec["forward_ct"] = getattr(self, "forward_ct", None)
                try:
                    apply_forward(rec, _tls.forward)
                except Exception:
                    runtime.write_error("sglang step forward")

    cls.run_batch = run_batch
    cls._infx_patched = True


def batch_record(batch):
    mode = getattr(batch, "forward_mode", None)
    mode_name = getattr(mode, "name", str(mode))
    phase = {"EXTEND": "prefill", "DECODE": "decode", "MIXED": "mixed", "IDLE": "idle",
             "TARGET_VERIFY": "decode"}.get(mode_name, mode_name.lower() if mode_name else None)
    reqs = list(getattr(batch, "reqs", None) or ())
    extend_lens = list(getattr(batch, "extend_lens", None) or ())
    prefix_lens = list(getattr(batch, "prefix_lens", None) or ())
    seq_lens = getattr(batch, "seq_lens_cpu", None)
    seq_lens = seq_lens.tolist() if hasattr(seq_lens, "tolist") else list(seq_lens or ())
    rows, total = [], 0
    for i, req in enumerate(reqs):
        prompt = getattr(req, "origin_input_ids", None)
        if phase in ("prefill", "mixed") and i < len(extend_lens):
            scheduled, computed = int(extend_lens[i]), int(prefix_lens[i]) if i < len(prefix_lens) else None
        else:
            scheduled = 1
            computed = int(seq_lens[i]) if i < len(seq_lens) else None
        total += scheduled
        rows.append({"req": getattr(req, "rid", None), "scheduled": scheduled, "computed": computed,
                     "prompt_len": len(prompt) if prompt is not None else None,
                     "new": phase in ("prefill", "mixed") and not computed, "spec": 0})
    extend_tokens = getattr(batch, "extend_num_tokens", None)
    if phase in ("prefill", "mixed") and extend_tokens:
        total = int(extend_tokens)
    return {"phase": phase, "mode": mode_name, "total_tokens": total, "reqs": rows}


def apply_forward(rec, forward):
    """Token counts from the target model's forward batch (the verify pass under speculation)."""
    if not forward:
        return
    rec["total_tokens"] = forward["tokens"]
    rec["forward_mode"] = forward["mode"]
    draft = forward.get("draft_tokens")
    if forward["mode"] == "TARGET_VERIFY" and draft:
        for row in rec.get("reqs") or ():
            row["scheduled"], row["spec"] = int(draft), int(draft) - 1


def patch_model_runner(module):
    cls = getattr(module, "ModelRunner", None)
    if cls is None or getattr(cls, "_infx_patched", False):
        return
    orig_load, orig_forward = cls.load_model, cls.forward

    def load_model(self, *args, **kwargs):
        result = orig_load(self, *args, **kwargs)
        draft = bool(getattr(self, "is_draft_worker", False))
        try:
            model = getattr(self, "model", None)
            modules.register_root("draft" if draft else "model", model)
            mark_draft(model, draft)
            if not draft:
                bound = tag_experts_layers(model)
                envinfo.write(rank_tag(), ("sglang", "sgl_kernel"), ("SGLANG_", "SGL_"),
                              {"runner_class": f"{type(self).__module__}.{type(self).__qualname__}",
                               "server_args": str(getattr(self, "server_args", None))})
                setup_routing(self, bound)
        except Exception:
            runtime.write_error("sglang load_model hooks")
        return result

    def forward(self, forward_batch, *args, **kwargs):
        if not getattr(self, "is_draft_worker", False):
            try:
                mode = getattr(getattr(forward_batch, "forward_mode", None), "name", "")
                ids = getattr(forward_batch, "input_ids", None)
                spec = getattr(forward_batch, "spec_info", None)
                _tls.forward = {"mode": mode, "tokens": int(ids.shape[0]) if ids is not None else None,
                                "draft_tokens": getattr(spec, "draft_token_num", None)}
            except Exception:
                pass
        return orig_forward(self, forward_batch, *args, **kwargs)

    cls.load_model = load_model
    cls.forward = forward
    cls._infx_patched = True


def mark_draft(model, draft):
    """Draft-model modules: their MoE layers reuse the target's layer numbers, so they record no routing."""
    if not draft or model is None or not hasattr(model, "named_modules"):
        return
    for _, mod in model.named_modules():
        try:
            mod._infx_draft = True
        except Exception:
            pass


_EXPERTS_NAME = re.compile(r"layers\.(\d+)\.mlp(?:\.experts)?$")


def tag_experts_layers(model):
    """Target-model experts modules that are not FusedMoE (Muse Spark calls FlashInfer's routed TRT-LLM MoE itself): their layer number, set while they run, for the routed-MoE hook."""
    layer_module = sys.modules.get("sglang.srt.layers.moe.fused_moe_triton.layer")
    fused = getattr(layer_module, "FusedMoE", ())
    bound = {}
    for name, mod in getattr(model, "named_modules", lambda: ())():
        match = _EXPERTS_NAME.search(name)
        if match is None or (fused and isinstance(mod, fused)):
            continue
        mod._infx_layer = int(match.group(1))
        if name.endswith(".experts") or mod._infx_layer not in bound:
            bound[mod._infx_layer] = name
        cls = type(mod)
        if vars(cls).get("_infx_experts_patched"):
            continue
        orig = cls.forward

        def forward(self, *args, _orig=orig, **kwargs):
            layer = getattr(self, "_infx_layer", None)
            if layer is None:
                return _orig(self, *args, **kwargs)
            outer, _tls.experts_layer = getattr(_tls, "experts_layer", None), layer
            try:
                return _orig(self, *args, **kwargs)
            finally:
                _tls.experts_layer = outer

        cls.forward = runtime.keep_signature(forward, orig)
        cls._infx_experts_patched = True
    return bound


ROUTED_MOE = ("trtllm_fp4_block_scale_routed_moe", "trtllm_fp8_block_scale_routed_moe",
              "trtllm_bf16_routed_moe", "trtllm_mxint4_block_scale_routed_moe")


def capture_routed_moe(args, kwargs):
    """FlashInfer's routed MoE takes the top-k ids it was routed with: one int32 tensor packed (expert id << 16 | weight), as SGLang's PackedTopKOutput carries them, or a (topk_ids, topk_weights) tuple."""
    import torch

    layer = getattr(_tls, "experts_layer", None)
    if layer is None or not routing.active():
        return
    ids = kwargs.get("topk_ids", args[0] if args else None)
    if isinstance(ids, tuple):
        ids = ids[0] if ids else None
    elif isinstance(ids, torch.Tensor) and ids.dtype == torch.int32:
        ids = ids >> 16
    if isinstance(ids, torch.Tensor) and ids.dim() == 2:
        routing.capture(layer, ids)


def patch_muse_routing(module):
    """Muse Spark's fused router returns (topk_weights, topk_ids, ...) with global expert ids."""
    orig = vars(module).get("fused_muse_spark_routing")
    if orig is None or getattr(orig, "_infx_routed", False):
        return

    @functools.wraps(orig)
    def wrapped(*args, **kwargs):
        result = orig(*args, **kwargs)
        try:
            layer = getattr(_tls, "experts_layer", None)
            if layer is not None and routing.active() and isinstance(result, tuple) and len(result) > 1:
                routing.capture(layer, result[1])
        except Exception:
            runtime.write_error("sglang muse routing capture")
        return result

    wrapped._infx_routed = True
    module.fused_muse_spark_routing = wrapped


def patch_routed_moe(module):
    """FlashInfer's routed TRT-LLM MoE entry points, in the module that defines them and the package that re-exports them; the experts layer running them names the layer."""
    for target in (module, sys.modules.get("flashinfer.fused_moe")):
        for name in ROUTED_MOE:
            orig = getattr(target, name, None) if target is not None else None
            if orig is None or getattr(orig, "_infx_routed", False):
                continue

            @functools.wraps(orig)
            def wrapped(*args, _orig=orig, **kwargs):
                try:
                    capture_routed_moe(args, kwargs)
                except Exception:
                    runtime.write_error("sglang routed moe capture")
                return _orig(*args, **kwargs)

            wrapped = runtime.keep_signature(wrapped, orig)
            wrapped._infx_routed = True
            setattr(target, name, wrapped)


def _graph_label(runner):
    name = type(runner).__name__
    if not any(k in name for k in ("Prefill", "Piecewise", "Breakable")):
        return "FULL"
    backend = type(getattr(runner, "backend", None)).__name__ if getattr(runner, "backend", None) is not None else name
    for key, label in (("Breakable", "BREAKABLE"), ("Piecewise", "PIECEWISE"), ("Full", "FULL")):
        if key in backend or key in name:
            return label
    return "PIECEWISE"


def _is_target(runner):
    model_runner = getattr(runner, "model_runner", None)
    return model_runner is not None and not getattr(model_runner, "is_draft_worker", False)


def patch_deduped_graphs(module):
    """Deduplicated CUDA graphs launch through cuda-python, past CUDAGraph.replay."""
    cls = getattr(module, "DedupedCudaGraphRegistry", None)
    orig = vars(cls).get("replay") if cls is not None else None
    if orig is None or getattr(orig, "_infx_patched", False):
        return

    @functools.wraps(orig)
    def replay(self, graph, *args, **kwargs):
        return graphs.replayed(getattr(graph, "original_graph", None), lambda: orig(self, graph, *args, **kwargs))

    replay._infx_patched = True
    cls.replay = replay


def patch_graph_runners(module):
    """Graph runner classes: capture inside the core capture session; replays label the step."""
    for name, cls in list(vars(module).items()):
        if not isinstance(cls, type) or cls.__module__ != module.__name__ or not name.endswith("CudaGraphRunner"):
            continue
        for method in ("capture", "_capture_all"):
            orig = vars(cls).get(method)
            if orig is None or getattr(orig, "_infx_patched", False):
                continue

            def capture(self, *args, _orig=orig, **kwargs):
                with windows.capture_session(rank_tag()):
                    return _orig(self, *args, **kwargs)

            capture._infx_patched = True
            setattr(cls, method, capture)
            if method == "capture":
                break
        for method in ("replay", "execute"):
            orig = vars(cls).get(method)
            if orig is None or getattr(orig, "_infx_patched", False):
                continue

            def run(self, *args, _orig=orig, **kwargs):
                before = graphs.replays()
                result = _orig(self, *args, **kwargs)
                if graphs.replays() > before and _is_target(self):
                    _tls.graph = _graph_label(self)
                return result

            run._infx_patched = True
            setattr(cls, method, run)


def setup_routing(runner, bound=None):
    config = getattr(getattr(runner, "model_config", None), "hf_config", None)
    text = getattr(config, "text_config", None) or config
    num_layers = getattr(text, "num_hidden_layers", None)
    topk = getattr(text, "num_experts_per_tok", None)
    experts = (getattr(text, "n_routed_experts", None) or getattr(text, "num_experts", None)
               or getattr(text, "num_local_experts", None))
    if not (num_layers and topk and experts):
        return
    args = getattr(runner, "server_args", None)
    max_tokens = max(int(getattr(args, "chunked_prefill_size", 0) or 0), int(getattr(args, "max_prefill_tokens", 0) or 0),
                     8192)
    try:
        from sglang.srt.distributed.parallel_state import get_attn_tp_group

        attn_tp_rank = get_attn_tp_group().rank_in_group
    except Exception:
        runtime.write_error("sglang attention TP rank")
        attn_tp_rank = int(getattr(runner, "tp_rank", 0) or 0)
    extra = {"source": "sglang TopK outputs; bypassed routing recomputed with select_experts"}
    write = attn_tp_rank == 0
    if getattr(args, "enable_layernorm_sp", False):
        write = True
        extra["token_shards"] = True
    gathered = bool(getattr(args, "enable_dp_attention", False)) and getattr(args, "moe_a2a_backend", "none") in (
        None, "none")
    if gathered:
        write = False
        extra["unavailable"] = ("DP attention with a gathered MoE: routing rows are every DP rank's tokens "
                                "in a padded layout, not recorded")
    routing.setup(rank_tag(), max_tokens, int(num_layers), int(topk), int(experts), write=write,
                  bound=bound or None, extra=extra)


def _ids(result):
    ids = getattr(result, "topk_ids", None)
    if ids is None:
        packed = getattr(result, "packed_topk_ids", None)
        if packed is not None:
            ids = packed >> 16
    return ids


def _topk_output(args, kwargs):
    return args[1] if len(args) > 1 else kwargs.get("topk_output")


def capture_moe_routing(moe, topk_output):
    """Record the top-k ids an experts layer is handed; recompute them when routing is left to its kernel."""
    import torch

    layer = getattr(moe, "layer_id", None)
    if not isinstance(layer, int) or getattr(moe, "_infx_draft", False) or topk_output is None:
        return
    if torch.compiler.is_compiling():
        routing.capture(layer, _ids(topk_output))
        return
    if not routing.active() or getattr(_tls, "routed", None) is topk_output:
        return
    _tls.routed = topk_output
    try:
        ids = _ids(topk_output)
        if ids is None and type(topk_output).__name__ == "BypassedTopKOutput":
            with launchers.marker("infx_prof.routing.recompute", skip=2):
                ids = _ids(topk_output.to_standard(layer))
        routing.capture(layer, ids)
    except Exception:
        runtime.write_error("sglang moe routing capture")


def capture_topk(topk, result):
    """TopK / HashTopK outputs, for MoE paths that do not go through FusedMoE.forward_impl (Mega-MoE). Eager only: compiled paths are captured at the experts layer, which skips an output already captured here."""
    import torch

    if torch.compiler.is_compiling() or not routing.active():
        return
    layer = getattr(topk, "layer_id", None)
    ids = _ids(result)
    if not isinstance(layer, int) or getattr(topk, "_infx_draft", False) or ids is None:
        return
    _tls.routed = result
    try:
        routing.capture(layer, ids)
    except Exception:
        runtime.write_error("sglang topk capture")


def patch_topk(module):
    """TopK's leaf forward_* methods (its forward dispatches to one) and HashTopK.forward."""
    for cls_name, methods in (("TopK", ("forward_native", "forward_cuda", "forward_cpu", "forward_npu",
                                        "forward_xpu", "forward_musa")),
                              ("HashTopK", ("forward",))):
        cls = getattr(module, cls_name, None)
        if cls is None or getattr(cls, "_infx_patched", False):
            continue
        for method in methods:
            orig = vars(cls).get(method)
            if orig is None:
                continue

            def wrapped(self, *args, _orig=orig, **kwargs):
                result = _orig(self, *args, **kwargs)
                capture_topk(self, result)
                return result

            setattr(cls, method, wrapped)
        cls._infx_patched = True


def patch_fused_moe(module):
    """FusedMoE and its subclasses: every MoE layer's routing arrives at forward_impl (or forward_deferred_finalize), whichever producer built it (TopK, HashTopK, a model's own routing such as DeepSeek-V4.1's vision_topk) and whichever path called it (eager, piecewise custom op, dual-stream graph)."""
    layer_module = sys.modules.get("sglang.srt.layers.moe.fused_moe_triton.layer")
    base = getattr(layer_module, "FusedMoE", None)
    if base is None:
        return
    for cls in list(vars(module).values()):
        if not isinstance(cls, type) or cls.__module__ != module.__name__ or not issubclass(cls, base):
            continue
        for method in ("forward_impl", "forward_deferred_finalize"):
            orig = vars(cls).get(method)
            if orig is None or getattr(orig, "_infx_patched", False):
                continue

            def wrapped(self, *args, _orig=orig, **kwargs):
                capture_moe_routing(self, _topk_output(args, kwargs))
                return _orig(self, *args, **kwargs)

            wrapped._infx_patched = True
            setattr(cls, method, wrapped)


class SglangEngine(Engine):
    name = "sglang"
    caller_roots = ("sglang", "sgl_kernel")
    launcher_packages = ("flashinfer", "sgl_kernel", "sglang.jit_kernel", "sglang.kernels", "deep_gemm",
                         "flash_mla", "flash_attn", "sglang.srt.distributed.device_communicators", "aiter")
    version_packages = ("sglang", "sgl_kernel")
    env_prefixes = ("SGLANG_", "SGL_")
    window_control = "file"

    def install(self, registry):
        registry.on("sglang.srt.managers.scheduler", patch_scheduler_module)
        registry.on("sglang.srt.model_executor.model_runner", patch_model_runner)
        registry.on_package("sglang.srt.layers.moe", patch_fused_moe)
        registry.on("sglang.srt.layers.moe.topk", patch_topk)
        registry.on("sglang.srt.layers.moe.hash_topk", patch_topk)
        registry.on("flashinfer.fused_moe.core", patch_routed_moe)
        for name in ("sglang.kernels.ops.muse_spark_v12.routing", "sglang.kernels.ops.muse_spark_v12.fused_muse_spark_routing",
                     "sglang.srt.rocm_muse_spark.fused_muse_spark_routing"):
            registry.on(name, patch_muse_routing)
        registry.on("sglang.srt.model_executor.runner_backend.cuda_graph_dedup_mixin", patch_deduped_graphs)
        registry.on_package("sglang.srt.model_executor", patch_graph_runners)
        registry.on_package("sglang.srt.speculative", patch_graph_runners)


ENGINE = SglangEngine()
