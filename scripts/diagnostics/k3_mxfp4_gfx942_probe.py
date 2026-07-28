#!/usr/bin/env python3
"""Launch the serving-faithful Kimi-K3 TP8 MXFP4 MoE kernel on gfx942.

This is a temporary cluster diagnostic, not production InferenceX code.  It
uses Kimi-K3's exact TP8 per-rank routed-expert shape and compares the fused
kernel output with AITER's torch reference.  The current ROCm image defaults
to separated A16W4 on gfx942; do not relabel this probe as A8W4.
"""

from __future__ import annotations

import gc
import importlib.metadata
import os
import time

import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import (
    fused_moe,
    get_2stage_cfgs,
    get_padded_M,
    torch_moe_stage1,
    torch_moe_stage2,
)
from aiter.jit.utils.chip_info import get_cu_num, get_gfx
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
from aiter.test_common import checkAllclose


TOKENS = 1
MODEL_DIM = 3584
INTER_DIM = 384
EXPERTS = 896
TOPK = 16
BETA = 4.0
LINEAR_BETA = 25.0
MAX_LOGITS_DIFF = 0.01


def stage_name(stage: object) -> str:
    func = getattr(stage, "func", stage)
    keywords = getattr(stage, "keywords", {})
    module = getattr(func, "__module__", "")
    name = getattr(func, "__name__", type(func).__name__)
    return f"{module}.{name} {keywords}"


def cuda_memory(label: str) -> None:
    free, total = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated()
    peak = torch.cuda.max_memory_allocated()
    gib = 1024**3
    print(
        f"K3_MXFP4_MEMORY label={label}"
        f" free_gib={free / gib:.3f} total_gib={total / gib:.3f}"
        f" allocated_gib={allocated / gib:.3f} peak_gib={peak / gib:.3f}",
        flush=True,
    )


def quantize_weight(
    shape: tuple[int, int, int],
    packed_shape: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    source = torch.randn(shape, dtype=dtypes.bf16, device="cuda")
    quantize = aiter.get_torch_quant(QuantType.per_1x32)
    packed, scale = quantize(source, quant_dtype=dtypes.fp4x2)
    del source
    return packed.view(packed_shape), scale


def normalized_logits_diff(reference: torch.Tensor, actual: torch.Tensor) -> float:
    reference64 = reference.double()
    actual64 = actual.double()
    denominator = (reference64.square() + actual64.square()).sum()
    if denominator.item() == 0:
        return 0.0 if torch.equal(reference, actual) else float("inf")
    similarity = 2 * (reference64 * actual64).sum() / denominator
    return float((1 - similarity).item())


def main() -> None:
    if os.environ.get("AITER_SITUV2_A8W4", "0").lower() in {"1", "true"}:
        raise RuntimeError(
            "primary gfx942 probe requires the image-default "
            "AITER_SITUV2_A8W4=0 contract"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is unavailable")
    if get_gfx() != "gfx942":
        raise RuntimeError(f"expected gfx942, got {get_gfx()}")

    try:
        aiter_version = importlib.metadata.version("amd-aiter")
    except importlib.metadata.PackageNotFoundError:
        aiter_version = getattr(aiter, "__version__", "unknown")

    torch.manual_seed(20260728)
    torch.cuda.reset_peak_memory_stats()
    print(
        "K3_MXFP4_KERNEL_ENV"
        f" gfx={get_gfx()} cu={get_cu_num()} aiter={aiter_version}"
        f" torch={torch.__version__} hip={torch.version.hip}"
        " mode=a16w4-separated"
        f" shape=M{TOKENS}xD{MODEL_DIM}xI{INTER_DIM}xE{EXPERTS}xK{TOPK}",
        flush=True,
    )
    cuda_memory("start")

    hidden = torch.randn((TOKENS, MODEL_DIM), dtype=dtypes.bf16, device="cuda")
    topk_ids = torch.arange(TOPK, dtype=dtypes.i32, device="cuda").view(TOKENS, TOPK)
    topk_weights = torch.full(
        (TOKENS, TOPK), 1.0 / TOPK, dtype=dtypes.fp32, device="cuda"
    )

    started = time.perf_counter()
    w1, w1_scale = quantize_weight(
        (EXPERTS, INTER_DIM * 2, MODEL_DIM),
        (EXPERTS, INTER_DIM * 2, MODEL_DIM // 2),
    )
    cuda_memory("w1-quantized")
    w2, w2_scale = quantize_weight(
        (EXPERTS, MODEL_DIM, INTER_DIM),
        (EXPERTS, MODEL_DIM, INTER_DIM // 2),
    )
    torch.cuda.synchronize()
    print(
        f"K3_MXFP4_QUANTIZED elapsed_seconds={time.perf_counter() - started:.3f}",
        flush=True,
    )
    cuda_memory("w2-quantized")

    # The deterministic routing above touches experts [0, TOPK).  Restricting
    # the torch reference to those experts preserves the exact result without
    # dequantizing all 896 experts to fp32 at once.
    ref_w1 = w1[:TOPK].contiguous()
    ref_w2 = w2[:TOPK].contiguous()
    ref_w1_scale = w1_scale.view(EXPERTS, INTER_DIM * 2, MODEL_DIM // 32)[
        :TOPK
    ].contiguous()
    ref_w2_scale = w2_scale.view(EXPERTS, MODEL_DIM, INTER_DIM // 32)[
        :TOPK
    ].contiguous()

    started = time.perf_counter()
    stage1_reference = torch_moe_stage1(
        hidden,
        ref_w1,
        ref_w2,
        topk_weights,
        topk_ids,
        dtype=dtypes.bf16,
        activation=ActivationType.Situv2,
        quant_type=QuantType.per_1x32,
        a1_scale=None,
        w1_scale=ref_w1_scale,
        w1_bias=None,
        doweight=False,
        situ_beta=BETA,
        situ_linear_beta=LINEAR_BETA,
    )
    reference = torch_moe_stage2(
        stage1_reference.view(TOKENS, TOPK, INTER_DIM),
        ref_w1,
        ref_w2,
        topk_weights,
        topk_ids,
        dtype=dtypes.bf16,
        quant_type=QuantType.per_1x32,
        w2_scale=ref_w2_scale,
        a2_scale=None,
        w2_bias=None,
        doweight=True,
    )
    torch.cuda.synchronize()
    print(
        f"K3_MXFP4_REFERENCE elapsed_seconds={time.perf_counter() - started:.3f}",
        flush=True,
    )
    if not torch.isfinite(reference).all().item():
        raise AssertionError("torch reference contains NaN or Inf")
    del stage1_reference, ref_w1, ref_w2, ref_w1_scale, ref_w2_scale
    gc.collect()
    torch.cuda.empty_cache()
    cuda_memory("reference-complete")

    # env=0 matches this image's vLLM contract: GGUU/separated stage-1 weights.
    shuffled_w1 = shuffle_weight_a16w4(w1, 16, False)
    shuffled_w1_scale = shuffle_scale_a16w4(w1_scale, EXPERTS, False)
    shuffled_w2 = shuffle_weight_a16w4(w2, 16, False)
    shuffled_w2_scale = shuffle_scale_a16w4(w2_scale, EXPERTS, False)
    del w1, w2, w1_scale, w2_scale
    gc.collect()
    torch.cuda.empty_cache()
    cuda_memory("weights-shuffled")

    metadata = get_2stage_cfgs(
        get_padded_M(TOKENS),
        MODEL_DIM,
        INTER_DIM,
        EXPERTS,
        TOPK,
        dtypes.bf16,
        dtypes.bf16,
        dtypes.fp4x2,
        QuantType.per_1x32,
        True,
        ActivationType.Situv2,
        False,
        0,
        0,
        True,
        GateMode.SEPARATED,
    )
    dispatch = f"{stage_name(metadata.stage1)} {stage_name(metadata.stage2)}"
    if dispatch.count("abf16_wfp4") < 2:
        raise AssertionError(f"unexpected non-A16W4 dispatch: {dispatch}")
    print(
        "K3_MXFP4_KERNEL_DISPATCH"
        f" block_m={metadata.block_m}"
        f" stage1={stage_name(metadata.stage1)}"
        f" stage2={stage_name(metadata.stage2)}",
        flush=True,
    )

    started = time.perf_counter()
    actual = fused_moe(
        hidden,
        shuffled_w1,
        shuffled_w2,
        topk_weights,
        topk_ids,
        activation=ActivationType.Situv2,
        quant_type=QuantType.per_1x32,
        doweight_stage1=False,
        w1_scale=shuffled_w1_scale,
        w2_scale=shuffled_w2_scale,
        beta=BETA,
        linear_beta=LINEAR_BETA,
        gate_mode=GateMode.SEPARATED.value,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    finite = bool(torch.isfinite(actual).all().item())
    diff = normalized_logits_diff(reference, actual) if finite else float("inf")
    allclose_error = (
        checkAllclose(reference, actual, msg="K3 gfx942 A16W4 exact-shape probe")
        if finite
        else 1
    )
    passed = finite and not (allclose_error != 0 and diff > MAX_LOGITS_DIFF)
    max_abs = (
        float((reference.float() - actual.float()).abs().max().item())
        if finite
        else float("inf")
    )
    print(
        "K3_MXFP4_KERNEL_RESULT"
        f" status={'pass' if passed else 'fail'}"
        f" finite={str(finite).lower()}"
        f" allclose_error={allclose_error}"
        f" logits_diff={diff:.9g} max_abs={max_abs:.9g}"
        f" threshold={MAX_LOGITS_DIFF} elapsed_seconds={elapsed:.6f}",
        flush=True,
    )
    cuda_memory("complete")

    if not passed:
        raise AssertionError(
            "official strict accuracy gate failed:"
            f" finite={finite} allclose_error={allclose_error}"
            f" logits_diff={diff} threshold={MAX_LOGITS_DIFF}"
        )


if __name__ == "__main__":
    main()
