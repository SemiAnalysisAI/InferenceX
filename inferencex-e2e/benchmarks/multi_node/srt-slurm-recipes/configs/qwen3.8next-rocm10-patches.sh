#!/usr/bin/env bash
# Qwen3.8-Flash-Next kernels on lmsysorg/sglang-rocm:v0.5.23-rocm10-mi35x-20261102,
# applied to its editable AITER (285579b6) and SGLang (1ced9a9eb0) checkouts. Each
# patch is the source-only diff of an open upstream PR (tests, benchmarks and AITER's
# AOT registry dropped; the header of each file names the PR and its head commit).
#   AITER
#   - 01 ROCm/aiter#5992  FlyDSL causal conv1d prefill, GDN decode/verify kernels
#   - 02 ROCm/aiter#6670  FlyDSL fused QSA sparse decode (needed by sglang#45196)
#   - 03 ROCm/aiter#6496  FlyDSL single-kernel GDN fused chunked prefill
#   SGLang
#   - 04 sglang#48648  fused decode-sized hyper-connection norm + inject, SGLANG_OPT_QWEN4_FUSED_HC
#   - 05 sglang#45196  QSA sparse decode on the AITER FlyDSL kernel (on with SGLANG_USE_AITER)
#   - 06 sglang#48649  faster QSA decode MQA logits (padded Q, group-strided launch)
#   - 07 sglang#45529  fused QK norm + RoPE + gate for Qwen3.5 self-attention on gfx950
#   - 08 sglang#41683  strided GDN prefill projection views on HIP
#   - 09 sglang#41879  GDN prefill conv on AITER FlyDSL, SGLANG_AITER_GDN_CONV
#   - 10 sglang#44103  GDN chunked prefill on AITER, SGLANG_AITER_GDN_PREFILL[_H]
# AITER 01 is needed by 09, 02 by 05 and 03 by 10 with SGLANG_AITER_GDN_PREFILL_H=fused,
# so the AITER patches go first. The image already has the monokernel
# (--enable-amd-monokernel), the AITER topk_gating router (sglang#40708), ratio-3 GDN
# packing (sglang#44302), the 128-row hc_mix (sglang#41880) and ROCm/aiter#6225, #5804,
# #6050, #6165, #6467, so none of them is patched. All Python. A patch that no longer
# applies fails the worker.
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

apply /sgl-workspace/aiter 01-aiter-5992-flydsl-causal-conv1d-prefill.patch
apply /sgl-workspace/aiter 02-aiter-6670-flydsl-qsa-sparse-decode.patch
apply /sgl-workspace/aiter 03-aiter-6496-flydsl-gdn-fused-prefill.patch
apply /sgl-workspace/sglang 04-sglang-48648-qwen4-fused-hc-decode.patch
apply /sgl-workspace/sglang 05-sglang-45196-qsa-flydsl-sparse-decode.patch
apply /sgl-workspace/sglang 06-sglang-48649-qsa-decode-mqa-launch.patch
apply /sgl-workspace/sglang 07-sglang-45529-qwen35-fuse-selfattn-prepare.patch
apply /sgl-workspace/sglang 08-sglang-41683-gdn-prefill-strided-views.patch
apply /sgl-workspace/sglang 09-sglang-41879-aiter-gdn-conv-prefill.patch
apply /sgl-workspace/sglang 10-sglang-44103-aiter-gdn-chunked-prefill.patch

echo "$digest" > "$marker"
