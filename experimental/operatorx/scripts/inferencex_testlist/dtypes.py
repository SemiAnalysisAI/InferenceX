"""Per-op-class dtype resolver.

Real InferenceX invocations use different (weight, activation) dtype pairs
for attention/dense/MoE GEMMs within the same workload — e.g. DSv4 Instruct
ships FP4 MoE experts + FP8 attention/dense; AMD's gpt-oss `-w-mxfp4-a-fp8`
checkpoint runs W4A8 MoE. The single `precision` tag in InferenceX is too
coarse to capture this, so we keep a lookup table keyed on
(precision, framework, runner-family, model-family) → per-op-class dtypes.

The table was derived by reading benchmarks/single_node/*.sh launch scripts
plus the SGLang/TRT-LLM cookbooks for each model. When a new (framework,
precision) tuple is added to InferenceX, add an entry here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .matrix import WorkloadRow


@dataclass(frozen=True)
class OpDtype:
    """A single op's dtype triple."""
    weight: str        # storage dtype of the weight tensor
    activation: str    # dtype of the activation tensor input to the GEMM
    output: str        # dtype of the GEMM output (usually bf16)


@dataclass(frozen=True)
class RowDtypes:
    """Per-op-class dtypes for one workload row."""
    attn: OpDtype      # attention block GEMMs (q/k/v/o projections)
    dense: OpDtype     # dense MLP GEMMs (when first_k_dense_replace > 0)
    moe: OpDtype       # MoE expert MLP GEMMs
    routing: str       # topk_routing op dtype (always high precision)
    kv: str            # KV cache storage dtype


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_BF16 = "bf16"
_FP16 = "fp16"


def _runner_family(runner: str) -> str:
    """Reduce a specific runner tag (e.g. "b200-dsv4", "b200-multinode") to a
    GPU family ("b200", "b300", "h200", "h100", "gb200", "gb300", "mi300x",
    "mi325x", "mi355x")."""
    r = runner.lower()
    # order matters: longest match first
    for fam in ("b300", "b200", "gb300", "gb200", "h200", "h100",
                "mi355x", "mi325x", "mi300x", "trn", "tpu"):
        if fam in r:
            return fam
    return r


def _is_blackwell(runner: str) -> bool:
    return _runner_family(runner) in ("b200", "b300", "gb200", "gb300")


def _is_hopper(runner: str) -> bool:
    return _runner_family(runner) in ("h100", "h200")


def _is_amd(runner: str) -> bool:
    return _runner_family(runner).startswith("mi")


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


def _resolve_attn_dense(precision: str, runner: str, model_prefix: str) -> OpDtype:
    """Attention + dense MLP dtypes. These follow the model's "non-MoE" quant
    regime, which for mixed-precision checkpoints (DSv4) is FP8 even when MoE
    is FP4."""
    if precision == "bf16":
        return OpDtype(_BF16, _BF16, _BF16)
    if precision == "fp16":
        return OpDtype(_FP16, _FP16, _FP16)
    if precision == "fp8":
        # FP8 block-128 weight + per-token dynamic FP8 activation
        return OpDtype("fp8", "fp8", _BF16)
    if precision in ("fp4", "nvfp4", "mxfp4"):
        if _is_amd(runner):
            # AMD: only MoE layers are quantized to MXFP4 weights. From SGLang
            # docs (quark_int4fp8_moe): "Other layers (e.g. projections in the
            # attention layers) have their weights quantized online to float8
            # directly." So attention/dense GEMMs run W=fp8, A=fp8.
            return OpDtype("fp8", "fp8", _BF16)
        if model_prefix == "dsv4":
            # DSv4 Instruct: attention/dense stay FP8 even on FP4 row
            return OpDtype("fp8", "fp8", _BF16)
        if _is_blackwell(runner):
            # NVFP4 weight + NVFP4 activation (Blackwell native FP4 tensor cores).
            # Verified against SGLang's CompressedTensorsW4A4 / modelopt_fp4 path.
            return OpDtype("nvfp4", "nvfp4", _BF16)
        # Hopper FP4 uses W4A16 (marlin / cutlass W4A16). Verified against
        # FlashInfer SM90 cutlass MXFP4 MoE backend (PR #24816, explicit W4A16).
        return OpDtype("mxfp4", _BF16, _BF16)
    if precision == "int4":
        # Weight-only quantization (e.g. Kimi K2.5 int4 compressed-tensors).
        # SGLang's CompressedTensorsWNA16 path: int4 weight × bf16 activation.
        return OpDtype("int4", _BF16, _BF16)
    if precision == "int8":
        return OpDtype("int8", "int8", _BF16)
    raise ValueError(f"unknown precision {precision!r}")


def _resolve_moe(precision: str, runner: str, framework: str, model_prefix: str) -> OpDtype:
    """MoE expert MLP dtypes. Depends on the moe-runner-backend chosen by the
    launch script for each (framework, precision, hardware) combination."""
    if precision == "bf16":
        return OpDtype(_BF16, _BF16, _BF16)
    if precision == "fp16":
        return OpDtype(_FP16, _FP16, _FP16)
    if precision == "fp8":
        return OpDtype("fp8", "fp8", _BF16)
    if precision in ("fp4", "nvfp4", "mxfp4"):
        if _is_amd(runner):
            # AMD MoE runs as W=mxfp4, A=fp8 across MI300x/MI325x/MI355x
            # (matches the published `amd/*-w-mxfp4-a-fp8` checkpoint contract
            # for both vLLM and ATOM frameworks).
            return OpDtype("mxfp4", "fp8", _BF16)
        if model_prefix == "dsv4":
            # DSv4 Instruct ships MXFP4 MoE experts. On Blackwell SGLang's
            # flashinfer_mxfp4 MoE backend runs FP4 weight × NVFP4 activation.
            # On Hopper it falls back to W4A16 marlin.
            if _is_hopper(runner):
                return OpDtype("mxfp4", _BF16, _BF16)
            return OpDtype("mxfp4", "nvfp4", _BF16)
        # Standard FP4 rows: NVFP4 weight from modelopt_fp4 quantization.
        if _is_blackwell(runner):
            return OpDtype("nvfp4", "nvfp4", _BF16)
        # Hopper FP4 marlin: W4A16
        return OpDtype("mxfp4", _BF16, _BF16)
    if precision == "int4":
        return OpDtype("int4", _BF16, _BF16)
    if precision == "int8":
        return OpDtype("int8", "int8", _BF16)
    raise ValueError(f"unknown precision {precision!r}")


def _resolve_kv(precision: str, runner: str, model_prefix: str) -> str:
    """KV cache storage dtype.

    Most quantized configurations (FP4/FP8) keep the KV cache in FP8 to save
    bandwidth + memory; BF16/FP16 paths leave it native. InferenceX scripts
    consistently pass `--kv-cache-dtype fp8_e4m3` on FP4 launches.
    """
    if precision in ("bf16", "fp16"):
        return precision
    if precision == "int4":
        # Weight-only quant; KV cache is BF16
        return _BF16
    # fp8/fp4/nvfp4/mxfp4 → fp8 KV cache
    return "fp8"


def _resolve_routing(precision: str) -> str:
    """topk_routing scoring is always done in a high-precision dtype because
    softmax/sigmoid need numerical headroom. SGLang/TRT-LLM run it in BF16
    (or FP32 for the gate matmul, then BF16 for softmax)."""
    if precision == "fp16":
        return _FP16
    return _BF16


def resolve_dtypes(row: WorkloadRow) -> RowDtypes:
    """Resolve per-op-class dtypes for one InferenceX matrix row."""
    return RowDtypes(
        attn=_resolve_attn_dense(row.precision, row.runner, row.model_prefix),
        dense=_resolve_attn_dense(row.precision, row.runner, row.model_prefix),
        moe=_resolve_moe(row.precision, row.runner, row.framework, row.model_prefix),
        routing=_resolve_routing(row.precision),
        kv=_resolve_kv(row.precision, row.runner, row.model_prefix),
    )
