#!/usr/bin/env bash
# Qwen3.8-Flash-Next kernels on lmsysorg/sglang-rocm:v0.5.22-rocm10-mi35x-20261018,
# applied to its editable AITER and SGLang checkouts. Already in the image and not
# patched: AITER topk_gating router (sglang#40708), ratio-3 GDN packing (sglang#44302),
# the 128-row hc_mix, and ROCm/aiter#6225, #5804, #6050:
#   - ROCm/aiter#5992: FlyDSL causal conv1d prefill and GDN decode/verify
#   - sgl-project/sglang#41879: opt-in SGLANG_AITER_GDN_CONV/_DECODE hooks
#   - fused Triton QK Gemma-RMSNorm + M-RoPE + gate on HIP
#   - FlyDSL hc_mix gates and the single-kernel hc_combine on ROCm
#     (SGLANG_HC_COMBINE_SPLIT=0); hc_mix itself (sglang#41880, up to 128 rows) is in the image
#   - opt-in FlyDSL paged decode for QSA attention (SGLANG_AITER_QSA_PA_DECODE) on
#     the image's AITER pa_decode
#   - fused FlyDSL QSA sparse decode that reads KV from the pool
#     (SGLANG_ENABLE_QSA_FLYDSL_DECODE, default on, tried before the paged decode)
#   - fused GDN decode projection + conv1d on HIP and off-by-default GDN prefill
#     opt-ins (SGLANG_GDN_FUSED_QKV_NORM, _WIDE_QKV_SPLIT, _PACKED_QWEN4,
#     _HIP_PREFILL_VIEWS)
#   - tuned BF16 / A8W8 bpreshuffle GEMM tables for Qwen3.8 on gfx950, added to the
#     image's gfx942 ones (AITER loads model_configs/ by default)
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
#   - opt-in fused HC for decode-sized rows (SGLANG_OPT_QWEN4_FUSED_HC): the
#     per-branch norm and the combine's inject GEMV fold into the FlyDSL HC mix
#   - sgl-project/sglang#46329: breakable prefill graphs replay multimodal batches
#     when speculative decoding is off (--cuda-graph-backend-prefill breakable)
#   - fused decode MoE route-sort + MXFP4 activation quant (ROCm/aiter#6165, #6467),
#     instantiated for Qwen3.8-Flash-Next's 513 experts / top-11 / hidden 2560
# All but the upstream PRs come from amd-nprotaso/{aiter,sglang} qwen_3_8_improvement.
# All Python except the MoE sort + quant, whose prebuilt module is removed so AITER's
# JIT rebuilds it on first use. A patch that no longer applies fails the worker.
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

apply /sgl-workspace/aiter aiter-flydsl-gdn-conv-decode.patch
apply /sgl-workspace/aiter aiter-flydsl-hc-mix.patch
apply /sgl-workspace/aiter aiter-flydsl-gdn-fused-prefill.patch
apply /sgl-workspace/aiter aiter-flydsl-gdn-decode-verify-perf.patch
apply /sgl-workspace/aiter aiter-qwen38-tuned-gemm-tables.patch
apply /sgl-workspace/aiter aiter-moe-sort-quant-2560.patch
rm -rf /sgl-workspace/aiter/aiter/jit/module_moe_sorting_opus.so \
    /sgl-workspace/aiter/aiter/jit/build/module_moe_sorting_opus
apply /sgl-workspace/sglang sglang-aiter-gdn-conv-decode-hooks.patch
apply /sgl-workspace/sglang sglang-fused-qk-rope-gate-hip.patch
apply /sgl-workspace/sglang sglang-flydsl-hc-mix.patch
apply /sgl-workspace/sglang sglang-flydsl-qsa-pa-decode.patch
apply /sgl-workspace/sglang sglang-aiter-gdn-prefill.patch
apply /sgl-workspace/sglang sglang-qsa-padded-group-mtp-shared-expert.patch
apply /sgl-workspace/sglang sglang-prefill-chunk-qsa-alignment.patch
apply /sgl-workspace/sglang sglang-qsa-flydsl-sparse-decode.patch
apply /sgl-workspace/sglang sglang-gdn-hip-fused-proj-conv.patch
apply /sgl-workspace/sglang sglang-qsa-pa-decode-mainline.patch
apply /sgl-workspace/sglang sglang-qwen38-decode-opts.patch
apply /sgl-workspace/sglang sglang-mtp-quant-resolve.patch
apply /sgl-workspace/sglang sglang-qwen38-fused-hc.patch
apply /sgl-workspace/sglang sglang-prefill-breakable-multimodal.patch

echo "$digest" > "$marker"
