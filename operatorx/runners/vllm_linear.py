"""Dense GEMM as a real vLLM linear layer, so vLLM picks the kernel.

Each op builds a vLLM ReplicatedLinear under the quantization config its
weight scheme implies (the same config classes a checkpoint's
quantization_config selects), loads synthetic weights in checkpoint format,
runs vLLM's process_weights_after_loading, and times layer(x) on a bf16
activation - so activation quantization, kernel selection and any weight
repacking are vLLM's own. The kernel vLLM chose is reported per op.
"""
from __future__ import annotations

import json
import os
import tempfile

import torch

from operatorx.core import BackendImpl, Op, UnsupportedOpError, lookup_versions

PER_TENSOR, PER_TOKEN, PER_CHANNEL = [-1, -1], [1, -1], [-1, 1]


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
    if a.get("out", "bf16") != "bf16":
        raise UnsupportedOpError("vLLM linear layers here are built with bf16 activations/outputs")
    _vllm_context()
    import vllm.envs as envs
    from vllm.model_executor.layers.linear import ReplicatedLinear
    qc = _quant_config(a)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)  # as vLLM's model loader does while building layers
    try:
        layer = ReplicatedLinear(k, n, bias=bool(a.get("bias")), quant_config=qc, params_dtype=torch.bfloat16,
                                 prefix="model.layers.0.mlp.down_proj", disable_tp=True).cuda()
        _fill(layer)
        loaded = _param_dtypes(layer)
        qm = layer.quant_method
        if hasattr(qm, "process_weights_after_loading"):
            qm.process_weights_after_loading(layer)
    except (NotImplementedError, AssertionError, ValueError) as e:
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
        raise UnsupportedOpError(f"vLLM kernel failed for {a}: {type(e).__name__}: {e}"[:400]) from e
    return ctx


def _kernel_gemm(ctx: dict) -> None:
    ctx["out"] = ctx["layer"](ctx["x"])


IMPLS = [BackendImpl(op_type="gemm", prepare=_prepare_gemm, kernel=_kernel_gemm)]
