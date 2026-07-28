#!/usr/bin/env python3
"""Launch the serving-faithful Kimi-K3 TP8 MXFP4 MoE kernel on gfx942.

This is a temporary cluster diagnostic, not production InferenceX code.  It
uses Kimi-K3's exact TP8 per-rank routed-expert shape and compares the fused
kernel output with AITER's torch reference.  The current ROCm image defaults
to separated A16W4 on gfx942; do not relabel this probe as A8W4.
"""

from __future__ import annotations

import gc
import importlib
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
    pattern: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if pattern == "constant":
        source = torch.full(shape, 0.03125, dtype=dtypes.bf16, device="cuda")
    else:
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
    pattern = os.environ.get("K3_PROBE_PATTERN", "random").strip().lower()
    if pattern not in {"constant", "random"}:
        raise ValueError(f"unsupported K3_PROBE_PATTERN={pattern!r}")
    torch.cuda.reset_peak_memory_stats()
    print(
        "K3_MXFP4_KERNEL_ENV"
        f" gfx={get_gfx()} cu={get_cu_num()} aiter={aiter_version}"
        f" torch={torch.__version__} hip={torch.version.hip}"
        " mode=a16w4-separated"
        f" pattern={pattern}"
        f" shape=M{TOKENS}xD{MODEL_DIM}xI{INTER_DIM}xE{EXPERTS}xK{TOPK}",
        flush=True,
    )
    cuda_memory("start")

    hidden = (
        torch.full((TOKENS, MODEL_DIM), 0.5, dtype=dtypes.bf16, device="cuda")
        if pattern == "constant"
        else torch.randn((TOKENS, MODEL_DIM), dtype=dtypes.bf16, device="cuda")
    )
    topk_ids = torch.arange(TOPK, dtype=dtypes.i32, device="cuda").view(TOKENS, TOPK)
    topk_weights = torch.full(
        (TOKENS, TOPK), 1.0 / TOPK, dtype=dtypes.fp32, device="cuda"
    )

    started = time.perf_counter()
    w1, w1_scale = quantize_weight(
        (EXPERTS, INTER_DIM * 2, MODEL_DIM),
        (EXPERTS, INTER_DIM * 2, MODEL_DIM // 2),
        pattern,
    )
    cuda_memory("w1-quantized")
    w2, w2_scale = quantize_weight(
        (EXPERTS, MODEL_DIM, INTER_DIM),
        (EXPERTS, MODEL_DIM, INTER_DIM // 2),
        pattern,
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
    del ref_w1, ref_w2, ref_w1_scale, ref_w2_scale
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

    moe_kernels = importlib.import_module("aiter.ops.flydsl.moe_kernels")
    original_compile_stage1 = moe_kernels.compile_flydsl_moe_stage1

    def traced_compile_stage1(*args, **kwargs):
        print(
            "K3_MXFP4_STAGE1_COMPILE"
            f" act={kwargs.get('act')}"
            f" situ_beta={kwargs.get('situ_beta')}"
            f" situ_linear_beta={kwargs.get('situ_linear_beta')}"
            f" tile_k={kwargs.get('tile_k')}"
            f" k_batch={kwargs.get('k_batch')}",
            flush=True,
        )
        return original_compile_stage1(*args, **kwargs)

    moe_kernels.compile_flydsl_moe_stage1 = traced_compile_stage1
    started = time.perf_counter()
    fused_moe_module = importlib.import_module("aiter.fused_moe")
    captured_calls: list[tuple[str, object]] = []
    fused_moe_module.kernel_bench_callable = captured_calls
    try:
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
    finally:
        fused_moe_module.kernel_bench_callable = None
        moe_kernels.compile_flydsl_moe_stage1 = original_compile_stage1
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    calls_by_stage = dict(captured_calls)
    if set(calls_by_stage) != {"stage1", "stage2"}:
        raise AssertionError(
            f"expected captured stage1/stage2 calls, got {list(calls_by_stage)}"
        )

    stage1_actual = calls_by_stage["stage1"]()
    torch.cuda.synchronize()
    stage1_reference_view = stage1_reference.view_as(stage1_actual)
    stage1_finite = bool(torch.isfinite(stage1_actual).all().item())
    stage1_diff = (
        normalized_logits_diff(stage1_reference_view, stage1_actual)
        if stage1_finite
        else float("inf")
    )
    stage1_error = (
        checkAllclose(
            stage1_reference_view,
            stage1_actual,
            msg="K3 gfx942 A16W4 stage1 isolation",
        )
        if stage1_finite
        else 1
    )
    print(
        "K3_MXFP4_STAGE1_RESULT"
        f" finite={str(stage1_finite).lower()}"
        f" allclose_error={stage1_error}"
        f" logits_diff={stage1_diff:.9g}",
        flush=True,
    )

    stage2_call = calls_by_stage["stage2"]
    stage2_args = list(stage2_call.args)
    if len(stage2_args) < 8:
        raise AssertionError(f"unexpected stage2 partial ABI: {len(stage2_args)} args")
    stage2_args[0] = stage1_reference_view
    stage2_isolated = torch.zeros_like(actual)
    stage2_args[6] = stage2_isolated
    stage2_call.func(*stage2_args, **stage2_call.keywords)
    torch.cuda.synchronize()
    stage2_finite = bool(torch.isfinite(stage2_isolated).all().item())
    stage2_diff = (
        normalized_logits_diff(reference, stage2_isolated)
        if stage2_finite
        else float("inf")
    )
    stage2_error = (
        checkAllclose(
            reference,
            stage2_isolated,
            msg="K3 gfx942 A16W4 stage2 isolation",
        )
        if stage2_finite
        else 1
    )
    print(
        "K3_MXFP4_STAGE2_RESULT"
        f" finite={str(stage2_finite).lower()}"
        f" allclose_error={stage2_error}"
        f" logits_diff={stage2_diff:.9g}",
        flush=True,
    )

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
