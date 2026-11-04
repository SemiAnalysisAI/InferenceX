#!/usr/bin/env bash
# Qwen3.8-Flash-Next kernels on lmsysorg/sglang-rocm:v0.5.23-rocm10-mi35x-20261102,
# applied to its editable AITER (285579b6) and SGLang (1ced9a9eb0) checkouts. The
# patch set is the "v2" stack (2026-11-04, amd-cgilli/sglang qwen38-submission-v2),
# 2 AITER and 5 SGLang patches that apply cleanly on these two checkouts:
#   AITER
#   - 03 FlyDSL varlen causal conv1d prefill kernel
#   - 04 FlyDSL single-kernel GDN fused chunked prefill (ROCm/aiter#6496)
#   SGLang
#   - 05 fused FlyDSL QSA sparse decode on gfx950 (sgl-project/sglang#45196),
#     SGLANG_ENABLE_QSA_FLYDSL_DECODE, default on
#   - 07 GDN prefill on the AITER conv (03) and fused chunked delta rule (04)
#     (sgl-project/sglang#44103, #41879): SGLANG_AITER_GDN_CONV, SGLANG_AITER_GDN_PREFILL
#   - 08 fused decode-sized hyper-connection norm + inject into the FlyDSL mix,
#     opt-in with SGLANG_OPT_QWEN4_FUSED_HC
#   - 11 pad the decode index Q to the ROCm MFMA width
#   - 13 zero-copy GDN prefill projection views on HIP (sgl-project/sglang#47128, #41683)
# 07 needs AITER 03 and 04, so the AITER patches go first. The image already has the
# monokernel (--enable-amd-monokernel), the AITER topk_gating router (sglang#40708),
# ratio-3 GDN packing (sglang#44302), the 128-row hc_mix (sglang#41880) and
# ROCm/aiter#6225, #5804, #6050, #6165, #6467, so none of them is patched.
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

apply /sgl-workspace/aiter 03-aiter-causal-conv1d-prefill.patch
apply /sgl-workspace/aiter 04-aiter-gdn-fused-prefill.patch
apply /sgl-workspace/sglang 05-qsa-flydsl-decode.patch
apply /sgl-workspace/sglang 07-gdn-aiter-prefill.patch
apply /sgl-workspace/sglang 08-fused-hc-decode.patch
apply /sgl-workspace/sglang 11-qsa-indexer-q-pad.patch
apply /sgl-workspace/sglang 13-gdn-prefill-views.patch

echo "$digest" > "$marker"
