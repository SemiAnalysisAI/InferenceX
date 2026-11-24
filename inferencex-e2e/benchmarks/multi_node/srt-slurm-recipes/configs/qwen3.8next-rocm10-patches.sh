#!/usr/bin/env bash
# Qwen3.8-Flash-Next kernels on lmsysorg/sglang-rocm:v0.5.23-rocm10-mi35x-20261124 (SGLang
# 88dc7501f5 and AITER 581579ab 0.1.29.dev156 with FlyDSL 0.4.2, both editable checkouts under
# /sgl-workspace).
#   AITER
#   - Not touched. The image's AITER has what the AITER release wheel gave this recipe before:
#     ROCm/aiter#5992 (FlyDSL causal conv1d prefill, GDN decode/verify kernels), #6670 (FlyDSL
#     fused QSA sparse decode), #6496 (FlyDSL single-kernel GDN fused chunked prefill) and #5993
#     (hc_mix kernels). The script only checks that the FlyDSL modules are importable.
#   SGLang (each patch is the source-only diff of an open upstream PR, rebased onto the image's
#   SGLang; tests, benchmarks and AITER's AOT registry dropped; the header of each file names the
#   PR and its head commit)
#   - 01 sglang#48648  fused decode-sized hyper-connection norm + inject, SGLANG_OPT_QWEN4_FUSED_HC
#   - 02 sglang#45529  fused QK norm + RoPE + gate for Qwen3.5 self-attention on gfx950
#   - 03 sglang#41879  GDN prefill conv on AITER FlyDSL, SGLANG_AITER_GDN_CONV
#   - 04 sglang#44103  GDN chunked prefill on AITER, SGLANG_AITER_GDN_PREFILL[_H]
#   - 05 sglang#48090  MTP verify target probs on the top-k support instead of a vocabulary-wide
#                      sort (on with SGLANG_USE_AITER under rejection sampling; requests with
#                      min-p keep the dense path)
# Already in the image, so not patched: sglang#45196 (FlyDSL QSA sparse decode) and #48649 (faster
# QSA decode MQA logits), both merged 2026-11-23, sglang#41683 (strided GDN prefill projection
# views), the monokernel (--enable-amd-monokernel), the AITER topk_gating router (sglang#40708),
# ratio-3 GDN packing (sglang#44302) and the 128-row hc_mix (sglang#41880). All Python. A patch
# that no longer applies, or a missing FlyDSL module, fails the worker.
set -euo pipefail
# srtctl only looks up setup_script here; the patches live with the single-node
# recipe and are read through the workspace mount.
patches=/infmax-workspace/benchmarks/single_node/srt-slurm-recipes/qwen3.8next/sglang/mi355x-fp4-mtp/patches
[[ -d $patches ]] || { echo "qwen3.8next: $patches not found" >&2; exit 1; }
# Later patches touch files of earlier ones, so a per-patch reverse check cannot
# tell a re-run apart; the marker records the script and patch set a container
# already has.
marker=/sgl-workspace/.qwen3.8next-rocm10-patches
digest="$( { cat "${BASH_SOURCE[0]}" "$patches"/*.patch; } | sha256sum | cut -d' ' -f1)"
if [[ -f $marker ]]; then
    if [[ "$(cat "$marker")" == "$digest" ]]; then
        echo "qwen3.8next patches already applied"
        exit 0
    fi
    echo "qwen3.8next: $marker is from a different patch set" >&2
    exit 1
fi

# The image's AITER must be the editable checkout with the FlyDSL kernels of the four PRs.
python3 - <<'PY'
import importlib.metadata as metadata
import importlib.util
import sys

for module in (
    "aiter.ops.flydsl.causal_conv1d_flydsl",
    "aiter.ops.flydsl.qsa_sparse_decode",
    "aiter.ops.flydsl.gdn_fused_prefill",
    "aiter.ops.flydsl.hc_mix",
):
    if importlib.util.find_spec(module) is None:
        sys.exit(f"qwen3.8next: {module} is missing from the image's AITER")
print(
    f"qwen3.8next image aiter: amd-aiter {metadata.version('amd-aiter')}, "
    f"flydsl {metadata.version('flydsl')}, from {importlib.util.find_spec('aiter').origin}"
)
PY

apply() {
    git -c safe.directory="*" -C "$1" apply "$patches/$2"
    echo "qwen3.8next patch applied: $2"
}

apply /sgl-workspace/sglang 01-sglang-48648-qwen4-fused-hc-decode.patch
apply /sgl-workspace/sglang 02-sglang-45529-qwen35-fuse-selfattn-prepare.patch
apply /sgl-workspace/sglang 03-sglang-41879-aiter-gdn-conv-prefill.patch
apply /sgl-workspace/sglang 04-sglang-44103-aiter-gdn-chunked-prefill.patch
apply /sgl-workspace/sglang 05-sglang-48090-mtp-sparse-rejection-target-probs.patch

echo "$digest" > "$marker"
