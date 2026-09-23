"""Per-checkpoint GEMM quantization schemes, keyed by HF model id and layer role.

Derived from each checkpoint's config.json / hf_quant_config.json and
safetensors headers, resolved to the scheme vLLM runs at load time (e.g. AMD
DeepSeek-R1 MXFP4 checkpoints ship bf16 attention, which vLLM's Quark loader
re-quantizes to MXFP4).

Role classes:
  - "attn":   every attention projection (MHA q/k/v/qkv/o, MLA q_a/q_b/kv_a/kv_b/o)
  - "attn_o": optional o_proj override (nvidia DeepSeek-R1 NVFP4 quantizes only o_proj)
  - "mlp":    what the enumerator emits as the dense MLP. That is the first-k
              dense layers for DeepSeek/GLM/Kimi/MiniMax-M3, the shared expert
              for Qwen3.5, and a routed-expert-shaped GEMM for MiniMax-M2.5
              (which has no dense layers), so it takes the expert scheme there.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GemmScheme:
    """Activation (a) and weight (b) operand descriptors, as in operatorx/ops/gemm.py:
    scale groups are [rows, cols] of A [M, K] / B [N, K]; -1 spans the dimension."""
    a: tuple
    b: tuple

    def args(self) -> dict[str, Any]:
        return {"a": _operand(*self.a), "b": _operand(*self.b), "out": "bf16"}


def _operand(dtype: str, scale: tuple | None = None, scale2: tuple | None = None) -> dict[str, Any]:
    d: dict[str, Any] = {"dtype": dtype}
    for key, sc in (("scale", scale), ("scale2", scale2)):
        if sc is not None:
            d[key] = {"dtype": sc[0], "static": sc[1], "group": list(sc[2])}
    return d


_TENSOR = ("fp32", True, (-1, -1))
BF16 = GemmScheme(("bf16",), ("bf16",))
FP8_BLOCK = GemmScheme(("e4m3", ("fp32", False, (1, 128))), ("e4m3", ("fp32", True, (128, 128))))
FP8_BLOCK_UE8M0 = GemmScheme(("e4m3", ("ue8m0", False, (1, 128))), ("e4m3", ("ue8m0", True, (128, 128))))
FP8_BLOCK32_UE8M0 = GemmScheme(("e4m3", ("ue8m0", False, (1, 32))), ("e4m3", ("ue8m0", True, (32, 32))))
FP8_PER_TENSOR = GemmScheme(("e4m3", _TENSOR), ("e4m3", _TENSOR))
FP8_PER_CHANNEL = GemmScheme(("e4m3", ("fp32", False, (1, -1))), ("e4m3", ("fp32", True, (-1, 1))))
MXFP8 = GemmScheme(("e4m3", ("ue8m0", False, (1, 32))), ("e4m3", ("ue8m0", True, (1, 32))))
NVFP4 = GemmScheme(("e2m1", ("e4m3", False, (1, 16)), _TENSOR), ("e2m1", ("e4m3", True, (1, 16)), _TENSOR))
MXFP4 = GemmScheme(("e2m1", ("ue8m0", False, (1, 32))), ("e2m1", ("ue8m0", True, (1, 32))))


def _roles(attn: GemmScheme, mlp: GemmScheme | None = None, attn_o: GemmScheme | None = None) -> dict:
    out = {"attn": attn, "mlp": attn if mlp is None else mlp}
    if attn_o is not None:
        out["attn_o"] = attn_o
    return out


_DSR1_NVFP4 = _roles(BF16, NVFP4, attn_o=NVFP4)

MODEL_SCHEMES: dict[str, dict[str, GemmScheme]] = {
    # DeepSeek-R1
    "deepseek-ai/DeepSeek-R1-0528": _roles(FP8_BLOCK),
    "nvidia/DeepSeek-R1-0528-FP4-V2": _DSR1_NVFP4,
    "nvidia/DeepSeek-R1-0528-NVFP4-v2": _DSR1_NVFP4,
    "deepseek-r1-fp4": _DSR1_NVFP4,  # local copy of nvidia/DeepSeek-R1-0528-NVFP4-v2
    "amd/DeepSeek-R1-0528-MXFP4": _roles(MXFP4),
    "amd/DeepSeek-R1-0528-MXFP4-Preview": _roles(MXFP4),
    "amd/DeepSeek-R1-0528-MXFP4-v2": _roles(MXFP4),
    # DeepSeek-V4 (no dense layers; compressor/indexer GEMMs are not enumerated)
    "deepseek-ai/DeepSeek-V4-Pro": _roles(FP8_BLOCK_UE8M0),
    "deepseek-ai/DeepSeek-V4-Pro-0813": _roles(FP8_BLOCK_UE8M0),
    "sgl-project/DeepSeek-V4-Pro-FP8": _roles(FP8_BLOCK_UE8M0),
    # V4.1-Flash: 32x32 weight blocks (config weight_block_size, confirmed by the scale shapes)
    "deepseek-ai/DeepSeek-V4.1-Flash": _roles(FP8_BLOCK32_UE8M0),
    # GLM-5.x
    "zai-org/GLM-5-FP8": _roles(FP8_BLOCK),
    "zai-org/GLM-5.1-FP8": _roles(FP8_BLOCK),
    "zai-org/GLM-5.2-FP8": _roles(FP8_BLOCK),
    "nvidia/GLM-5-NVFP4": _roles(BF16, NVFP4),
    "nvidia/GLM-5.1-NVFP4": _roles(BF16),
    "nvidia/GLM-5.2-NVFP4": _roles(BF16),
    "amd/GLM-5.1-MXFP4": _roles(BF16),
    "amd/GLM-5.2-MXFP4": _roles(BF16),
    # Kimi (K2.5 INT4 quantizes routed experts only)
    "moonshotai/Kimi-K2.5": _roles(BF16),
    "moonshotai/Kimi-K3": _roles(BF16),
    "amd/Kimi-K2.5-MXFP4": _roles(BF16, MXFP4),
    "nvidia/Kimi-K2.5-NVFP4": _roles(BF16, NVFP4),
    "nvidia/Kimi-K2.6-NVFP4": _roles(BF16),
    # MiniMax ("mlp" is expert-shaped for M2.5, dense layers for M3)
    "MiniMaxAI/MiniMax-M2.5": _roles(FP8_BLOCK),
    "nvidia/MiniMax-M2.5-NVFP4": _roles(BF16, NVFP4),
    "amd/MiniMax-M2.5-MXFP4": _roles(BF16, MXFP4),
    "MiniMaxAI/MiniMax-M3-MXFP8": _roles(MXFP8),
    "nvidia/MiniMax-M3-NVFP4": _roles(MXFP8),
    "amd/MiniMax-M3-MXFP4": _roles(BF16),
    # gpt-oss (MXFP4 routed experts only)
    "openai/gpt-oss-120b": _roles(BF16),
    "amd/gpt-oss-120b-w-mxfp4-a-fp8": _roles(BF16),
    # Qwen ("mlp" is the shared expert)
    "Qwen/Qwen3.5-397B-A17B": _roles(BF16),
    "Qwen/Qwen3.5-397B-A17B-FP8": _roles(FP8_BLOCK),
    "nvidia/Qwen3.5-397B-A17B-NVFP4": _roles(BF16),
    "nvidia/Qwen3.5-397B-A17B-NVFP4-V2": _roles(FP8_PER_TENSOR),
    "amd/Qwen3.5-397B-A17B-MXFP4": _roles(BF16),
    "amd/Qwen3.5-397B-A17B-MXFP4-AttnFP8-V2": _roles(FP8_PER_CHANNEL, MXFP4),
    "Qwen/Qwen3.8-Flash-Next-FP8": _roles(BF16),
    "RadixArk/Qwen3.8-Flash-Next-NVFP4": _roles(BF16),
}


def _role_class(role: str) -> str:
    if role.startswith(("attn_o_proj", "mla_o_proj")):
        return "attn_o"
    if role.startswith(("attn_", "mla_")):
        return "attn"
    if role.startswith("mlp_"):
        return "mlp"
    raise ValueError(f"unknown GEMM role {role!r}")


def gemm_scheme(model: str, role: str) -> GemmScheme:
    """Scheme for one GEMM of `model` (HF id) in the given enumerator role tag."""
    roles = MODEL_SCHEMES.get(model)
    if roles is None:
        raise ValueError(f"no GEMM scheme for model {model!r}; add it to schemes.MODEL_SCHEMES")
    cls = _role_class(role)
    if cls == "attn_o":
        return roles.get("attn_o", roles["attn"])
    return roles[cls]
