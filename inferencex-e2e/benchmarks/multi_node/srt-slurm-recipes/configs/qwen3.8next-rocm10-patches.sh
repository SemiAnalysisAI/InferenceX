#!/usr/bin/env bash
# Qwen3.8-Flash-Next kernels on lmsysorg/sglang-rocm:v0.5.23-rocm10-mi35x-20261102,
# applied to its editable AITER (285579b6) and SGLang (1ced9a9eb0) checkouts. The
# image already has the monokernel (--enable-amd-monokernel), the AITER topk_gating
# router (sglang#40708), ratio-3 GDN packing (sglang#44302), the 128-row hc_mix
# (sglang#41880) and ROCm/aiter#6225, #5804, #6050, #6165, #6467, so none of them is patched.
#   - ROCm/aiter#5992: FlyDSL causal conv1d prefill and GDN decode/verify
#   - sgl-project/sglang#41879: opt-in SGLANG_AITER_GDN_CONV/_DECODE hooks
#   - fused Triton QK Gemma-RMSNorm + M-RoPE + gate on HIP
#   - ROCm/aiter#5993 FlyDSL hc_mix gates and the single-kernel hc_combine on ROCm
#     (SGLANG_HC_COMBINE_SPLIT=0)
#   - opt-in FlyDSL paged decode for QSA attention (SGLANG_AITER_QSA_PA_DECODE) on
#     AITER mainline pa_decode
#   - fused FlyDSL QSA sparse decode that reads KV from the pool
#     (SGLANG_ENABLE_QSA_FLYDSL_DECODE, default on, tried before the paged decode)
#   - fused GDN decode projection + conv1d on HIP and off-by-default GDN prefill
#     opt-ins (SGLANG_GDN_FUSED_QKV_NORM, _WIDE_QKV_SPLIT, _PACKED_QWEN4,
#     _HIP_PREFILL_VIEWS)
#   - tuned BF16 / A8W8 bpreshuffle GEMM tables for Qwen3.8 on gfx950 (ROCm/aiter#6769)
#   - off-by-default decode optimizations: shared per-token FP8 attention input
#     quant, fused GDN gated norm + out_proj quant, shared-expert gate folded into
#     the router, BF16 lm_head through tgemm (SGLANG_OPT_QWEN4_SHARED_ATTN_QUANT,
#     _QWEN4_FUSED_GDN_NORM_QUANT, _QWEN_MOE_FUSED_SHARED_TOPK, _AITER_LM_HEAD_GEMM)
#   - opt-in AITER GDN chunked prefill (SGLANG_AITER_GDN_PREFILL; only
#     SGLANG_AITER_GDN_PREFILL_H=fused works on this AITER)
#   - faster FlyDSL GDN decode/verify (used with SGLANG_AITER_GDN_DECODE)
#   - exact chunk fill (default on for gfx95) fixes: unfinished prefill chunks end
#     on a QSA page, padded QSA group writes read an inert source, and off-page
#     prefixes use the absolute mamba checkpoint grid
#   - fused shared-expert loading for Qwen3.8, strict for the MTP draft
#   - an all-floating-point MTP branch loads unquantized under a Quark target
#   - sgl-project/sglang#46329: breakable prefill graphs replay multimodal batches
#     (--cuda-graph-backend-prefill breakable), and a shared-quantized attention
#     input (SGLANG_OPT_QWEN4_SHARED_ATTN_QUANT) goes through the QSA indexer break
#     as its two tensors instead of crashing the prefill graph capture
# All but the upstream PRs come from amd-nprotaso/{aiter,sglang} qwen_3_8_improvement.
# All Python. A patch that no longer applies fails the worker.
set -euo pipefail
# srtctl only looks up setup_script here; the patches live with the single-node
# recipe and are read through the workspace mount.
patches=/infmax-workspace/benchmarks/single_node/srt-slurm-recipes/qwen3.8next/sglang/mi355x-fp4-mtp/patches
[[ -d $patches ]] || { echo "qwen3.8next: $patches not found" >&2; exit 1; }
# Later patches touch files of earlier ones, so a per-patch reverse check cannot
# tell a re-run apart; the marker records the patch set a container already has.
marker=/sgl-workspace/.qwen3.8next-rocm10-patches
digest="$(cat "${BASH_SOURCE[0]}" "$patches"/*.patch | sha256sum | cut -d' ' -f1)"
if [[ -f $marker ]]; then
    if [[ "$(cat "$marker")" == "$digest" ]]; then
        echo "qwen3.8next patches already applied"
        exit 0
    fi
    echo "qwen3.8next: $marker is from a different patch set" >&2
    exit 1
fi

apply() {
    git -c safe.directory="*" -C "$1" apply "$patches/$2"
    echo "qwen3.8next patch applied: $2"
}

apply /sgl-workspace/aiter 01-aiter-flydsl-gdn-conv-decode.patch
apply /sgl-workspace/aiter 02-aiter-flydsl-hc-mix.patch
apply /sgl-workspace/aiter 03-aiter-flydsl-gdn-fused-prefill.patch
apply /sgl-workspace/aiter 04-aiter-flydsl-gdn-decode-verify-perf.patch
apply /sgl-workspace/aiter 05-aiter-qwen38-tuned-gemm-tables.patch
apply /sgl-workspace/sglang 01-sglang-aiter-gdn-conv-decode-hooks.patch
apply /sgl-workspace/sglang 02-sglang-fused-qk-rope-gate-hip.patch
apply /sgl-workspace/sglang 03-sglang-hc-mix-env-gates.patch
apply /sgl-workspace/sglang 04-sglang-qsa-pa-decode.patch
apply /sgl-workspace/sglang 05-sglang-aiter-gdn-prefill.patch
apply /sgl-workspace/sglang 06-sglang-qsa-padded-group-mtp-shared-expert.patch
apply /sgl-workspace/sglang 07-sglang-prefill-chunk-qsa-alignment.patch
apply /sgl-workspace/sglang 08-sglang-qsa-flydsl-sparse-decode.patch
apply /sgl-workspace/sglang 09-sglang-gdn-hip-fused-proj-conv.patch
apply /sgl-workspace/sglang 10-sglang-qwen38-decode-opts.patch
apply /sgl-workspace/sglang 11-sglang-mtp-quant-resolve.patch
apply /sgl-workspace/sglang 13-sglang-prefill-breakable-multimodal.patch
apply /sgl-workspace/sglang 14-sglang-qsa-breakable-shared-quant.patch

echo "$digest" > "$marker"
