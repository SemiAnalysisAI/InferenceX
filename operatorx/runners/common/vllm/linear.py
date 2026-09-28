"""Dense GEMM as a vLLM RowParallelLinear (K split over tp, all-reduced), so vLLM picks the kernel.

Built under the quantization config a checkpoint with the operands' scheme would declare, with
synthetic checkpoint-format weights and process_weights_after_loading; layer(x) on a bf16
activation, as a CUDA graph where vLLM would capture one. The chosen kernel is recorded.
"""
from __future__ import annotations

import os
import sys

import torch

from operatorx.core import BackendImpl, Op, UnsupportedOpError, lookup_versions
from operatorx.runners.common import ranks
from operatorx.runners.common.vllm import engine

PER_TENSOR, PER_TOKEN, PER_CHANNEL = [-1, -1], [1, -1], [-1, 1]


def device() -> "torch.device":
    """cuda on NVIDIA and ROCm."""
    from vllm.platforms import current_platform
    return torch.device(current_platform.device_type)


def sync() -> None:
    getattr(torch, device().type).synchronize()

_MX_FP4 = {"dtype": "fp4", "qscheme": "per_group", "ch_axis": -1, "group_size": 32, "symmetric": None,
           "round_method": "half_even", "scale_type": "float", "scale_format": "e8m0",
           "scale_calculation_mode": "even", "mx_element_dtype": None, "observer_cls": "PerBlockMXObserver",
           "is_scale_quant": False}
# the quantization_config AMD's MXFP4 checkpoints ship (e.g. amd/Kimi-K2.5-MXFP4), exclusions dropped
_QUARK_MXFP4 = {
    "quant_method": "quark", "quant_mode": "eager_mode", "exclude": [], "algo_config": None,
    "global_quant_config": {"input_tensors": {**_MX_FP4, "is_dynamic": True},
                            "weight": {**_MX_FP4, "is_dynamic": False},
                            "output_tensors": None, "bias": None, "target_device": None},
    "layer_type_quant_config": {}, "layer_quant_config": {}, "kv_cache_quant_config": {},
    "export": {"kv_cache_group": [], "min_kv_scale": 0.0, "pack_method": "reorder",
               "weight_format": "real_quantized", "weight_merge_groups": None},
}


# Kimi-K3's routed experts: MXFP4 weights, bf16 activations (compressed-tensors mxfp4-pack)
_CT_MXFP4_W4A16 = {
    "quant_method": "compressed-tensors", "format": "mxfp4-pack-quantized", "ignore": [],
    "config_groups": {"group_0": {
        "format": "mxfp4-pack-quantized", "targets": ["Linear"], "input_activations": None,
        "output_activations": None,
        "weights": {"actorder": None, "block_structure": None, "dynamic": False, "group_size": 32, "num_bits": 4,
                    "observer": "minmax", "observer_kwargs": {}, "scale_dtype": "torch.uint8",
                    "strategy": "group", "symmetric": True, "type": "float", "zp_dtype": None}}}}


def _scheme(qa: dict, qb: dict) -> tuple[str, dict] | None:
    """Operands -> (quant method, checkpoint quantization_config); None: unquantized; (): unsupported."""
    if "scale" not in qa and "scale" not in qb:
        return None if qa["dtype"] == qb["dtype"] == "bf16" else ()
    sa, sb = qa.get("scale"), qb.get("scale")
    if (qa == {"dtype": "bf16"} and qb["dtype"] == "e2m1" and sb == {"dtype": "ue8m0", "static": True, "group": [1, 32]}
            and "scale2" not in qb):
        return "compressed-tensors", _CT_MXFP4_W4A16
    if sa is None or sb is None or not sb["static"] or not qa.get("symmetric", True) or not qb.get("symmetric", True):
        return ()
    if qa["dtype"] == qb["dtype"] == "e4m3" and "scale2" not in qa and "scale2" not in qb:
        ga, gb = sa["group"], sb["group"]
        dyn = "static" if sa["static"] else "dynamic"
        if ga == gb == PER_TENSOR and sa["dtype"] == sb["dtype"] == "fp32":
            return "fp8", {"quant_method": "fp8", "activation_scheme": dyn}
        if ga == gb == [1, 32] and sa["dtype"] == sb["dtype"] == "ue8m0" and not sa["static"]:
            return "mxfp8", {"quant_method": "mxfp8", "activation_scheme": "dynamic", "weight_block_size": [1, 32],
                             "ignored_layers": []}
        if gb[0] >= 1 and gb[1] > 1 and ga == [1, gb[1]] and not sa["static"] and (
                sa["dtype"] == sb["dtype"] or (sa["dtype"] == "fp32" and sb["dtype"] == "bf16")):
            cfg = {"quant_method": "fp8", "activation_scheme": "dynamic", "fmt": "e4m3", "weight_block_size": gb}
            if sb["dtype"] in ("fp32", "bf16"):  # bf16 block scales load like fp32 ones
                return "fp8", cfg
            if sb["dtype"] == "ue8m0":
                # DeepSeek-V4: vLLM remaps the checkpoint's fp8 config to deepseek_v4_fp8
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


# env that steers vLLM's linear-kernel choice; recorded with every result
_ENV_KEYS = ("VLLM_USE_DEEP_GEMM", "VLLM_USE_DEEP_GEMM_E8M0", "VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER",
             "VLLM_ROCM_USE_AITER", "VLLM_ROCM_USE_AITER_LINEAR")

def versions() -> dict[str, str]:
    return lookup_versions("vllm", "torch")


def _quant_config(args):
    sch = _scheme(args["a"], args["b"])
    if sch is None:
        return None
    if not sch:
        raise UnsupportedOpError(f"no vLLM quantization config for a={args['a']} b={args['b']}")
    return quant_config(*sch)


def quant_config(method: str, cfg: dict):
    from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS, get_quantization_config
    if method not in QUANTIZATION_METHODS:
        raise UnsupportedOpError(f"vLLM has no {method!r} quantization method")
    return get_quantization_config(method).from_config(cfg)


def _reject_fallback(qc, method) -> None:
    """An unquantized fallback method would record a bf16 matmul under the quantized row
    (vLLM's mxfp4 does this for linear layers; only its MoE experts have a kernel)."""
    if qc is not None and "Unquantized" in type(method).__name__:
        raise UnsupportedOpError(
            f"{type(qc).__name__} has no quantized linear method here; the layer fell back "
            f"to {type(method).__name__}")


def _set_quant_fp8_op(qc) -> None:
    """VllmConfig enables +quant_fp8 for blocked-weight checkpoints; mirror it per layer
    (CustomOp reads it at build)."""
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
    return {n: str(p.dtype).removeprefix("torch.") for n, p in layer.named_parameters(recurse=False)}


def _prepare_gemm(op: Op) -> dict:
    a = op.args
    if a.get("activation") is not None:
        raise UnsupportedOpError(f"vllm linear has no fused activation; got {a['activation']!r}")
    m, n, k = a["m"], a["n"], a["k"]
    if min(m, n, k) <= 0:
        raise UnsupportedOpError(f"degenerate gemm shape m={m} n={n} k={k}")
    if a["a"].get("input", "bf16") != "bf16":
        raise UnsupportedOpError("vLLM linear layers take a bf16 activation")
    if a.get("out", "bf16") == "fp32" or a["b"]["dtype"] == "fp32":
        return _prepare_fp32_out(op)
    if a.get("out", "bf16") != "bf16":
        raise UnsupportedOpError(f"vLLM linear layers return bf16 or fp32; got {a['out']!r}")
    tp = (a.get("parallel") or {}).get("tp", 1)
    if k % tp:
        raise UnsupportedOpError(f"k={k} does not split {tp} ways")
    engine.context(a.get("parallel"))
    import vllm.envs as envs
    from vllm.model_executor.layers.linear import RowParallelLinear
    with ranks.together("build this layer"):
        qc = _quant_config(a)
        _set_quant_fp8_op(qc)
        prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)  # as vLLM's model loader does while building layers
        try:
            layer = RowParallelLinear(k, n, bias=bool(a.get("bias")), quant_config=qc, params_dtype=torch.bfloat16,
                                      reduce_results=True, prefix="model.layers.0.mlp.down_proj").to(device())
            _fill(layer)
            loaded = _param_dtypes(layer)
            qm = layer.quant_method
            _reject_fallback(qc, qm)
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
    x = torch.randn(m, layer.input_size_per_partition, device=device(), dtype=torch.bfloat16)
    ctx = {"layer": layer, "x": x,
           "meta": {"vllm_quant_method": type(qm).__name__, "vllm_kernels": kernels,
                    "param_dtypes_loaded": loaded, "param_dtypes": _param_dtypes(layer),
                    "vllm_env": {k: getattr(envs, k) for k in _ENV_KEYS if hasattr(envs, k)},
                    "vllm_engine": engine.meta()}}
    with ranks.together("run this layer"):
        try:
            _kernel_gemm(ctx)
            sync()
        except (NotImplementedError, AssertionError, RuntimeError, ValueError) as e:
            if _is_fault(e):
                raise
            raise UnsupportedOpError(f"vLLM kernel failed for {a}: {type(e).__name__}: {e}"[:400]) from e
    return ctx


class _TorchMM(torch.nn.Module):
    """vLLM's fp32-output gate/router matmul (GateLinear's cuBLAS tier)."""

    def __init__(self, k: int, n: int, dtype: torch.dtype, bias: bool):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(n, k, device=device()).mul_(0.02).to(dtype),
                                         requires_grad=False)
        self.bias = torch.nn.Parameter(torch.zeros(n, device=device(), dtype=dtype),
                                       requires_grad=False) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight.dtype == torch.bfloat16 and self.bias is None:
            return torch.mm(x, self.weight.T, out_dtype=torch.float32)
        return torch.nn.functional.linear(x.to(self.weight.dtype), self.weight, self.bias).float()


def _prepare_fp32_out(op: Op) -> dict:
    a = op.args
    qa, qb = a["a"], a["b"]
    if "scale" in qa or "scale" in qb or qa["dtype"] != "bf16" or qb["dtype"] not in ("bf16", "fp32") \
            or a.get("out", "bf16") != "fp32":
        raise UnsupportedOpError(f"no fp32-output vLLM matmul for a={qa} b={qb} out={a.get('out')}")
    if (a.get("parallel") or {}).get("tp", 1) > 1:
        raise UnsupportedOpError("vLLM's fp32-output gates and routers are replicated, never split")
    engine.context(a.get("parallel"))
    layer = _TorchMM(a["k"], a["n"], getattr(torch, {"bf16": "bfloat16", "fp32": "float32"}[qb["dtype"]]),
                     bool(a.get("bias")))
    ctx = {"layer": layer, "x": torch.randn(a["m"], a["k"], device=device(), dtype=torch.bfloat16),
           "meta": {"vllm_quant_method": "torch.mm", "vllm_kernels": {},
                    "param_dtypes_loaded": _param_dtypes(layer), "param_dtypes": _param_dtypes(layer)}}
    _kernel_gemm(ctx)
    sync()
    return ctx


def _kernel_gemm(ctx: dict) -> None:
    ctx["out"] = ctx["layer"](ctx["x"])


def _capture_sizes() -> list[int]:
    from vllm.config import get_current_vllm_config
    return sorted(get_current_vllm_config().compilation_config.cudagraph_capture_sizes or [])


def _launcher(ctx: dict):
    """(callable, cuda_graph): as served, a batch up to the largest capture size replays the next
    size's graph, larger ones run eagerly. OPERATORX_CUDA_GRAPH_MAX_TOKENS caps it (0 = never)."""
    from vllm.distributed.parallel_state import graph_capture
    x = ctx["x"]
    eager = (lambda: _kernel_gemm(ctx)), False  # noqa: E731
    sizes = _capture_sizes()
    cap = int(os.environ.get("OPERATORX_CUDA_GRAPH_MAX_TOKENS", sizes[-1] if sizes else 0))
    size = next((s for s in sizes if s >= x.shape[0]), None)
    if cap <= 0 or size is None or size > cap:
        return eager
    xp = torch.randn(size, *x.shape[1:], device=x.device, dtype=x.dtype)
    err = None
    try:
        for _ in range(2):
            ctx["layer"](xp)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with graph_capture(x.device) as gc, torch.cuda.graph(g, pool=torch.cuda.graph_pool_handle(),
                                                             stream=gc.stream):
            ctx["graph_out"] = ctx["layer"](xp)
        torch.cuda.synchronize()
    except Exception as e:
        if _is_fault(e):
            raise
        torch.cuda.synchronize()
        err = e
    if not ranks.agree(err is None):  # every rank replays, or every rank runs eagerly
        print(f"[vllm.linear] CUDA-graph capture failed, timing eagerly: {type(err).__name__}: {err}"[:300]
              if err else "[vllm.linear] another rank's CUDA-graph capture failed, timing eagerly",
              file=sys.stderr)
        return eager
    ctx["graph"], ctx["x_padded"] = g, xp
    return g.replay, True


IMPLS = [BackendImpl(op_type="gemm", prepare=_prepare_gemm, kernel=_kernel_gemm, launcher=_launcher)]
