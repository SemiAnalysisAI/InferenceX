"""Dense GEMM as a real vLLM linear layer, so vLLM picks the kernel.

Each op builds a vLLM ReplicatedLinear under the quantization config its
weight scheme implies (the same config classes a checkpoint's
quantization_config selects), loads synthetic weights in checkpoint format,
runs vLLM's process_weights_after_loading, and times layer(x) on a bf16
activation - so activation quantization, kernel selection and any weight
repacking are vLLM's own - replayed as a CUDA graph where vLLM would capture
one. The kernel vLLM chose and the graph decision are reported per op.
"""
from __future__ import annotations

import json
import os
import tempfile

import torch

from operatorx.core import BackendImpl, Op, UnsupportedOpError, lookup_versions

PER_TENSOR, PER_TOKEN, PER_CHANNEL = [-1, -1], [1, -1], [-1, 1]

_MX_FP4 = {"dtype": "fp4", "qscheme": "per_group", "ch_axis": -1, "group_size": 32, "symmetric": None,
           "round_method": "half_even", "scale_type": "float", "scale_format": "e8m0",
           "scale_calculation_mode": "even", "mx_element_dtype": None, "observer_cls": "PerBlockMXObserver",
           "is_scale_quant": False}
# The quantization_config AMD's MXFP4 checkpoints ship (e.g. amd/Kimi-K2.5-MXFP4), exclusions dropped.
_QUARK_MXFP4 = {
    "quant_method": "quark", "quant_mode": "eager_mode", "exclude": [], "algo_config": None,
    "global_quant_config": {"input_tensors": {**_MX_FP4, "is_dynamic": True},
                            "weight": {**_MX_FP4, "is_dynamic": False},
                            "output_tensors": None, "bias": None, "target_device": None},
    "layer_type_quant_config": {}, "layer_quant_config": {}, "kv_cache_quant_config": {},
    "export": {"kv_cache_group": [], "min_kv_scale": 0.0, "pack_method": "reorder",
               "weight_format": "real_quantized", "weight_merge_groups": None},
}


def _scheme(qa: dict, qb: dict) -> tuple[str, dict] | None:
    """Operand descriptors -> (vLLM quant method, checkpoint quantization_config), as a
    checkpoint with that scheme would declare it; None for unquantized."""
    if "scale" not in qa and "scale" not in qb:
        return None if qa["dtype"] == qb["dtype"] == "bf16" else ()
    sa, sb = qa.get("scale"), qb.get("scale")
    if sa is None or sb is None or not sb["static"] or not qa.get("symmetric", True) or not qb.get("symmetric", True):
        return ()
    if qa["dtype"] == qb["dtype"] == "e4m3" and "scale2" not in qa and "scale2" not in qb:
        ga, gb = sa["group"], sb["group"]
        dyn = "static" if sa["static"] else "dynamic"
        if ga == gb == PER_TENSOR and sa["dtype"] == sb["dtype"] == "fp32":
            return "fp8", {"quant_method": "fp8", "activation_scheme": dyn}
        if gb[0] >= 1 and gb[1] > 1 and ga == [1, gb[1]] and not sa["static"] and sa["dtype"] == sb["dtype"]:
            cfg = {"quant_method": "fp8", "activation_scheme": "dynamic", "fmt": "e4m3", "weight_block_size": gb}
            if sb["dtype"] == "fp32":
                return "fp8", cfg
            if sb["dtype"] == "ue8m0":
                # DeepSeek-V4: vLLM remaps the checkpoint's fp8 config to deepseek_v4_fp8 (ue8m0 scales).
                return "deepseek_v4_fp8", {**cfg, "scale_fmt": "ue8m0"}
        if ga == PER_TOKEN and gb == PER_CHANNEL and not sa["static"]:
            return "compressed-tensors", {
                "quant_method": "compressed-tensors", "format": "float-quantized", "ignore": [],
                "config_groups": {"group_0": {
                    "targets": ["Linear"],
                    "weights": {"num_bits": 8, "type": "float", "strategy": "channel", "dynamic": False,
                                "symmetric": True},
                    "input_activations": {"num_bits": 8, "type": "float", "strategy": "token", "dynamic": True,
                                          "symmetric": True}}}}
    if (qa["dtype"] == qb["dtype"] == "e2m1" and sa["group"] == sb["group"] == [1, 32] and not sa["static"]
            and sa["dtype"] == sb["dtype"] == "ue8m0" and "scale2" not in qa and "scale2" not in qb):
        return "quark", _QUARK_MXFP4
    if (qa["dtype"] == qb["dtype"] == "e2m1" and sa["group"] == sb["group"] == [1, 16] and not sa["static"]
            and sa["dtype"] == sb["dtype"] == "e4m3" and qa.get("scale2") and qb.get("scale2")):
        return "modelopt_fp4", {"quantization": {
            "quant_algo": "NVFP4", "group_size": 16, "kv_cache_quant_algo": None, "exclude_modules": []}}
    return ()


# Env that steers vLLM's linear-kernel choice; recorded with every result.
_ENV_KEYS = ("VLLM_USE_DEEP_GEMM", "VLLM_USE_DEEP_GEMM_E8M0", "VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER",
             "VLLM_ROCM_USE_AITER", "VLLM_ROCM_USE_AITER_LINEAR")

_READY = None


def versions() -> dict[str, str]:
    return lookup_versions("vllm", "torch")


def _vllm_context():
    """Enter a minimal vLLM config + single-rank parallel state once per process."""
    global _READY
    if _READY is not None:
        return _READY
    from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel

    cfg_dir = tempfile.mkdtemp(prefix="opx_vllm_cfg_")
    with open(os.path.join(cfg_dir, "config.json"), "w") as f:
        json.dump({"architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": 256,
                   "intermediate_size": 512, "num_attention_heads": 4, "num_key_value_heads": 4,
                   "num_hidden_layers": 1, "vocab_size": 1024, "max_position_embeddings": 2048,
                   "torch_dtype": "bfloat16"}, f)
    vcfg = VllmConfig()
    vcfg.model_config = ModelConfig(model=cfg_dir, dtype="bfloat16", skip_tokenizer_init=True)
    try:
        vcfg._set_cudagraph_sizes()  # vLLM's default capture sizes for this config
    except Exception:
        pass
    ctx = set_current_vllm_config(vcfg)
    ctx.__enter__()
    init_distributed_environment(world_size=1, rank=0, local_rank=torch.cuda.current_device(),
                                 distributed_init_method=f"tcp://127.0.0.1:{29500 + os.getpid() % 1000}",
                                 backend="nccl")
    initialize_model_parallel(1, 1)
    _READY = ctx
    return ctx


def _quant_config(args):
    sch = _scheme(args["a"], args["b"])
    if sch is None:
        return None
    if not sch:
        raise UnsupportedOpError(f"no vLLM quantization config for a={args['a']} b={args['b']}")
    from vllm.model_executor.layers.quantization import get_quantization_config
    method, cfg = sch
    return get_quantization_config(method).from_config(cfg)


def _set_quant_fp8_op(qc) -> None:
    """VllmConfig turns on the CUDA quant_fp8 custom op for checkpoints with blocked
    weights; mirror that per layer (CustomOp reads it when the layer is built)."""
    from vllm.config import get_current_vllm_config
    ops = get_current_vllm_config().compilation_config.custom_ops
    blocked = getattr(qc, "weight_block_size", None) is not None
    if blocked and "+quant_fp8" not in ops:
        ops.append("+quant_fp8")
    elif not blocked and "+quant_fp8" in ops:
        ops.remove("+quant_fp8")


def _is_fault(e: BaseException) -> bool:
    """Allocation failures and device faults are errors, never 'unsupported'."""
    msg = str(e)
    return isinstance(e, torch.OutOfMemoryError) or any(
        s in msg for s in ("CUDA error", "HIP error", "illegal memory", "out of memory", "device-side assert"))


def _fill(layer: torch.nn.Module) -> None:
    """Synthetic checkpoint-format values for whatever parameters the method registered."""
    for name, p in layer.named_parameters(recurse=False):
        with torch.no_grad():
            if p.dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e4m3fnuz):
                p.copy_((torch.randn(p.shape, device=p.device) * 0.5).to(p.dtype))
            elif p.dtype in (torch.uint8, torch.int8, torch.int32):
                p.copy_(torch.randint(0, 127, p.shape, device=p.device, dtype=p.dtype))
            elif "scale" in name:
                p.fill_(0.01)
            else:
                p.copy_((torch.randn(p.shape, device=p.device) * 0.02).to(p.dtype))


def _kernel_names(layer) -> dict[str, str]:
    """Kernel objects vLLM attached to the quant method or its scheme."""
    qm = layer.quant_method
    out = {}
    for owner in (qm, getattr(qm, "scheme", None), getattr(layer, "scheme", None)):
        if owner is None:
            continue
        if owner is not qm:
            out["scheme"] = type(owner).__name__
        for k, v in vars(owner).items():
            t = type(v)
            if t.__module__.startswith("vllm.model_executor.kernels") or t.__name__.endswith("Kernel"):
                out[k] = t.__name__
    return out


def _param_dtypes(layer) -> dict[str, str]:
    """Post-processing dtype of every weight/scale tensor, as the kernel sees it."""
    return {n: str(p.dtype).removeprefix("torch.") for n, p in layer.named_parameters(recurse=False)}


def _prepare_gemm(op: Op) -> dict:
    a = op.args
    if a.get("activation") is not None:
        raise UnsupportedOpError(f"vllm linear has no fused activation; got {a['activation']!r}")
    m, n, k = a["m"], a["n"], a["k"]
    if min(m, n, k) <= 0:
        raise UnsupportedOpError(f"degenerate gemm shape m={m} n={n} k={k}")
    if a.get("out", "bf16") != "bf16" or a["a"].get("input", "bf16") != "bf16":
        raise UnsupportedOpError("vLLM linear layers take a bf16 activation and return bf16")
    _vllm_context()
    import vllm.envs as envs
    from vllm.model_executor.layers.linear import ReplicatedLinear
    qc = _quant_config(a)
    _set_quant_fp8_op(qc)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)  # as vLLM's model loader does while building layers
    try:
        layer = ReplicatedLinear(k, n, bias=bool(a.get("bias")), quant_config=qc, params_dtype=torch.bfloat16,
                                 prefix="model.layers.0.mlp.down_proj", disable_tp=True).cuda()
        # set by the column/row-parallel linears real layers use; some kernels read them
        for attr, v in (("input_size_per_partition", k), ("output_size_per_partition", n)):
            if not hasattr(layer, attr):
                setattr(layer, attr, v)
        _fill(layer)
        loaded = _param_dtypes(layer)
        qm = layer.quant_method
        if hasattr(qm, "process_weights_after_loading"):
            qm.process_weights_after_loading(layer)
    except (NotImplementedError, AssertionError, ValueError, RuntimeError) as e:
        if _is_fault(e):
            raise
        raise UnsupportedOpError(f"vLLM rejected {a}: {type(e).__name__}: {e}"[:400]) from e
    finally:
        torch.set_default_dtype(prev)
    kernels = _kernel_names(layer)
    if any(v.startswith("Emulation") for v in kernels.values()):
        raise UnsupportedOpError(f"vLLM has only an emulation kernel for {a} here: {kernels}")
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    ctx = {"layer": layer, "x": x,
           "meta": {"vllm_quant_method": type(qm).__name__, "vllm_kernels": kernels,
                    "param_dtypes_loaded": loaded, "param_dtypes": _param_dtypes(layer),
                    "vllm_env": {k: getattr(envs, k) for k in _ENV_KEYS if hasattr(envs, k)}}}
    try:
        _kernel_gemm(ctx)
        torch.cuda.synchronize()
    except (NotImplementedError, AssertionError, RuntimeError, ValueError) as e:
        if _is_fault(e):
            raise
        raise UnsupportedOpError(f"vLLM kernel failed for {a}: {type(e).__name__}: {e}"[:400]) from e
    return ctx


def _kernel_gemm(ctx: dict) -> None:
    ctx["out"] = ctx["layer"](ctx["x"])


def _capture_sizes() -> list[int]:
    from vllm.config import get_current_vllm_config
    sizes = get_current_vllm_config().compilation_config.cudagraph_capture_sizes
    if sizes:
        return sorted(sizes)
    # VllmConfig._set_cudagraph_sizes' default candidates
    from vllm.platforms import current_platform
    top = 1024 if current_platform.is_device_capability_family(100) else 512
    return [1, 2, 4] + list(range(8, 256, 8)) + list(range(256, top + 1, 16))


def _launcher(ctx: dict):
    """As vLLM serves it: batches up to the largest capture size replay a full CUDA
    graph captured at the next capture size (the batch is padded up to it); larger
    batches run eagerly. OPERATORX_CUDA_GRAPHS=0 forces eager."""
    x, meta = ctx["x"], ctx["meta"]
    eager = lambda: _kernel_gemm(ctx)  # noqa: E731
    if os.environ.get("OPERATORX_CUDA_GRAPHS", "1") != "1":
        meta["graph"] = {"mode": "eager", "reason": "OPERATORX_CUDA_GRAPHS=0"}
        return eager
    sizes = _capture_sizes()
    m = x.shape[0]
    size = next((s for s in sizes if s >= m), None)
    if size is None:
        meta["graph"] = {"mode": "eager", "reason": f"m > max capture size {sizes[-1]}"}
        return eager
    xp = torch.randn(size, x.shape[1], device=x.device, dtype=x.dtype)
    try:
        for _ in range(2):
            ctx["layer"](xp)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=torch.cuda.graph_pool_handle()):
            ctx["graph_out"] = ctx["layer"](xp)
        torch.cuda.synchronize()
    except Exception as e:
        if _is_fault(e):
            raise
        torch.cuda.synchronize()
        meta["graph"] = {"mode": "eager", "reason": f"capture failed: {type(e).__name__}: {e}"[:200]}
        return eager
    ctx["graph"], ctx["x_padded"] = g, xp
    meta["graph"] = {"mode": "full", "capture_size": size, "tokens": m, "max_capture_size": sizes[-1]}
    return g.replay


IMPLS = [BackendImpl(op_type="gemm", prepare=_prepare_gemm, kernel=_kernel_gemm, launcher=_launcher)]
