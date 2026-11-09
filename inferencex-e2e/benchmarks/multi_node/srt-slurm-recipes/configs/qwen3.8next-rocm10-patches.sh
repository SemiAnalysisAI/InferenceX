#!/usr/bin/env bash
# Qwen3.8-Flash-Next kernels on lmsysorg/sglang-rocm:v0.5.23-rocm10-mi35x-20261102 (SGLang
# 1ced9a9eb0 and AITER 285579b6, both editable checkouts under /sgl-workspace).
#   AITER
#   - The image's AITER is replaced by a release wheel, AITER_WHEEL_URL (the recipe pins the
#     v0.1.29 ROCm 7.14 / cp312 wheel and its sha256; that is the build whose prebuilt kernels
#     work with the image's torch, built for HIP 7.15, while the ROCm 10.0 and 10.1 wheels
#     fail on any tensor call), with AITER_FLYDSL_VERSION (the wheel requires flydsl==0.4.2,
#     the image has 0.4.0). v0.1.29 already has what three open PRs
#     gave this recipe before: ROCm/aiter#5992 (FlyDSL causal conv1d prefill, GDN decode/verify
#     kernels), #6670 (FlyDSL fused QSA sparse decode, needed by sglang#45196) and #6496
#     (FlyDSL single-kernel GDN fused chunked prefill), and the hc_mix kernels (#5993).
#   SGLang (each patch is the source-only diff of an open upstream PR; tests, benchmarks and
#   AITER's AOT registry dropped; the header of each file names the PR and its head commit)
#   - 01 sglang#48648  fused decode-sized hyper-connection norm + inject, SGLANG_OPT_QWEN4_FUSED_HC
#   - 02 sglang#45196  QSA sparse decode on the AITER FlyDSL kernel (on with SGLANG_USE_AITER)
#   - 03 sglang#48649  faster QSA decode MQA logits (padded Q, group-strided launch)
#   - 04 sglang#45529  fused QK norm + RoPE + gate for Qwen3.5 self-attention on gfx950
#   - 05 sglang#41683  strided GDN prefill projection views on HIP
#   - 06 sglang#41879  GDN prefill conv on AITER FlyDSL, SGLANG_AITER_GDN_CONV
#   - 07 sglang#44103  GDN chunked prefill on AITER, SGLANG_AITER_GDN_PREFILL[_H]
#   - 08 sglang#48090  MTP verify target probs on the top-k support instead of a vocabulary-wide
#                      sort (on with SGLANG_USE_AITER under rejection sampling; the image lacks
#                      sglang#41916, so the patch carries its top-p rescale helper)
# The image already has the monokernel (--enable-amd-monokernel), the AITER topk_gating router
# (sglang#40708), ratio-3 GDN packing (sglang#44302) and the 128-row hc_mix (sglang#41880), so
# none of them is patched. All Python. A wheel that does not install, or a patch that no longer
# applies, fails the worker.
set -euo pipefail
source /infmax-workspace/benchmarks/check_env.sh
check_env_vars AITER_WHEEL_URL AITER_FLYDSL_VERSION
# srtctl only looks up setup_script here; the patches live with the single-node
# recipe and are read through the workspace mount.
patches=/infmax-workspace/benchmarks/single_node/srt-slurm-recipes/qwen3.8next/sglang/mi355x-fp4-mtp/patches
[[ -d $patches ]] || { echo "qwen3.8next: $patches not found" >&2; exit 1; }
# Later patches touch files of earlier ones, so a per-patch reverse check cannot
# tell a re-run apart; the marker records the script, patch set and AITER wheel a
# container already has.
marker=/sgl-workspace/.qwen3.8next-rocm10-patches
digest="$( { cat "${BASH_SOURCE[0]}" "$patches"/*.patch; echo "$AITER_WHEEL_URL $AITER_FLYDSL_VERSION"; } | sha256sum | cut -d' ' -f1)"
if [[ -f $marker ]]; then
    if [[ "$(cat "$marker")" == "$digest" ]]; then
        echo "qwen3.8next patches already applied"
        exit 0
    fi
    echo "qwen3.8next: $marker is from a different patch set" >&2
    exit 1
fi

pip_install=(python3 -m pip install)
if python3 -m pip install --help 2>/dev/null | grep -q -- --break-system-packages; then
    pip_install+=(--break-system-packages)
fi
# --no-deps: torch, triton and the other ROCm builds of the image must stay as they are;
# the wheel's own requirements are installed already (checked below).
"${pip_install[@]}" --quiet --no-deps --force-reinstall "$AITER_WHEEL_URL"
"${pip_install[@]}" --quiet --no-deps --index-url https://pypi.org/simple "flydsl==$AITER_FLYDSL_VERSION"
# Fail only on a conflict this install caused; the image may carry others.
if conflicts="$(python3 -m pip check 2>&1 | grep -iE 'amd-aiter|flydsl' || true)" && [[ -n $conflicts ]]; then
    echo "qwen3.8next: pip check reports $conflicts" >&2
    exit 1
fi
# The wheel must now be the AITER that python imports, not the editable checkout.
python3 - <<'PY'
import importlib.metadata as metadata
import importlib.util
import os
import sys

wheel = os.environ["AITER_WHEEL_URL"].rsplit("/", 1)[-1].replace("%2B", "+")
want = wheel.split("-")[1].split("+")[0]
have = metadata.version("amd-aiter")
spec = importlib.util.find_spec("aiter")
if not have.startswith(want) or spec is None or spec.origin.startswith("/sgl-workspace/aiter"):
    sys.exit(f"qwen3.8next: amd-aiter {have} from {spec and spec.origin}, expected the {want} wheel")
for module in (
    "aiter.ops.flydsl.causal_conv1d_flydsl",
    "aiter.ops.flydsl.qsa_sparse_decode",
    "aiter.ops.flydsl.gdn_fused_prefill",
    "aiter.ops.flydsl.hc_mix",
):
    if importlib.util.find_spec(module) is None:
        sys.exit(f"qwen3.8next: {module} is missing from amd-aiter {have}")
print(f"qwen3.8next aiter wheel installed: amd-aiter {have}, flydsl {metadata.version('flydsl')}, from {spec.origin}")
PY

apply() {
    git -c safe.directory="*" -C "$1" apply "$patches/$2"
    echo "qwen3.8next patch applied: $2"
}

apply /sgl-workspace/sglang 01-sglang-48648-qwen4-fused-hc-decode.patch
apply /sgl-workspace/sglang 02-sglang-45196-qsa-flydsl-sparse-decode.patch
apply /sgl-workspace/sglang 03-sglang-48649-qsa-decode-mqa-launch.patch
apply /sgl-workspace/sglang 04-sglang-45529-qwen35-fuse-selfattn-prepare.patch
apply /sgl-workspace/sglang 05-sglang-41683-gdn-prefill-strided-views.patch
apply /sgl-workspace/sglang 06-sglang-41879-aiter-gdn-conv-prefill.patch
apply /sgl-workspace/sglang 07-sglang-44103-aiter-gdn-chunked-prefill.patch
apply /sgl-workspace/sglang 08-sglang-48090-mtp-sparse-rejection-target-probs.patch

echo "$digest" > "$marker"
