#!/usr/bin/env bash
# Qwen3.8-Flash-Next kernels on lmsysorg/sglang-rocm:v0.5.23-rocm10-mi35x-20261117 (SGLang
# c67db7c762 and AITER 581579ab 0.1.29.dev156 with FlyDSL 0.4.2, both editable checkouts under
# /sgl-workspace).
#   AITER
#   - Not replaced. The image's AITER has what the AITER release wheel gave this recipe before:
#     ROCm/aiter#5992 (FlyDSL causal conv1d prefill, GDN decode/verify kernels), #6670 (FlyDSL
#     fused QSA sparse decode, needed by sglang#45196), #6496 (FlyDSL single-kernel GDN fused
#     chunked prefill) and #5993 (hc_mix kernels). The script checks that the FlyDSL modules are
#     importable.
#   SGLang (each patch is the source-only diff of an open upstream PR, rebased onto the image's
#   SGLang; tests, benchmarks and AITER's AOT registry dropped; the header of each file names the
#   PR and its head commit)
#   - 01 sglang#48648  fused decode-sized hyper-connection norm + inject, SGLANG_OPT_QWEN4_FUSED_HC
#   - 02 sglang#45196  QSA sparse decode on the AITER FlyDSL kernel (on with SGLANG_USE_AITER)
#   - 03 sglang#48649  faster QSA decode MQA logits (padded Q, group-strided launch)
#   - 04 sglang#45529  fused QK norm + RoPE + gate for Qwen3.5 self-attention on gfx950
#   - 05 sglang#41879  GDN prefill conv on AITER FlyDSL, SGLANG_AITER_GDN_CONV
#   - 06 sglang#44103  GDN chunked prefill on AITER, SGLANG_AITER_GDN_PREFILL[_H]
#   - 07 sglang#48090  MTP verify target probs on the top-k support instead of a vocabulary-wide
#                      sort (on with SGLANG_USE_AITER under rejection sampling; the image lacks
#                      sglang#41916, so the patch carries its top-p rescale helper)
#   - 09 sglang#50024  one FP8 quant + one GEMM for the GDN in_proj_qkvz / in_proj_ba of up to 64 rows
#                      (on by default; the monokernel points set SGLANG_QWEN35_PACKED_IN_PROJ=0)
#   sgl-amd-mono (/sgl-workspace/sgl-amd-mono, an editable install of commit a04ceab159d9)
#   - 10 amd-asalykov/sgl-amd-mono asalykov/qfn-mono-tp (afbb25a457009d): the updated Qwen3.8-Flash-Next
#                      monokernel (QSA index chain, stage placement, TP 1 / 2 / 4 launches), the
#                      qwen38_flash_next package and the common/ops.py and common/sync.py it needs
#                      (the diff a04ceab..afbb25a; a04ceab already has afbb25a's common/ops.py and sync.py)
# Already in the image, so not patched: sglang#41683 (strided GDN prefill projection views, merged
# 2026-11-17), the original monokernel (--enable-amd-monokernel; patch 10 updates it), the AITER
# topk_gating router (sglang#40708), ratio-3 GDN packing (sglang#44302) and the 128-row hc_mix
# (sglang#41880). All Python. A patch that no longer applies, or a missing FlyDSL module, fails the
# worker.
set -eo pipefail
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
# Patches 12 and 22 write into the AITER that python imports, which must be the image's editable checkout.
aiter_root="$(python3 -c 'import aiter, pathlib; print(pathlib.Path(aiter.__file__).resolve().parents[1])')"
[[ $aiter_root == /sgl-workspace/aiter ]] || { echo "qwen3.8next: aiter imports from $aiter_root, expected /sgl-workspace/aiter" >&2; exit 1; }

apply() {
    git -c safe.directory="*" -C "$1" apply "$patches/$2"
    echo "qwen3.8next patch applied: $2"
}

apply /sgl-workspace/sglang 01-sglang-48648-qwen4-fused-hc-decode.patch
apply /sgl-workspace/sglang 02-sglang-45196-qsa-flydsl-sparse-decode.patch
apply /sgl-workspace/sglang 03-sglang-48649-qsa-decode-mqa-launch.patch
apply /sgl-workspace/sglang 04-sglang-45529-qwen35-fuse-selfattn-prepare.patch
apply /sgl-workspace/sglang 05-sglang-41879-aiter-gdn-conv-prefill.patch
apply /sgl-workspace/sglang 06-sglang-44103-aiter-gdn-chunked-prefill.patch
apply /sgl-workspace/sglang 07-sglang-48090-mtp-sparse-rejection-target-probs.patch
apply /sgl-workspace/sglang 09-sglang-50024-qwen4-packed-fp8-gdn-in-proj.patch
apply /sgl-workspace/sgl-amd-mono 10-sgl-amd-mono-qfn-mono-tp-afbb25a.patch
apply /sgl-workspace/sglang 11-sglang-48625-qsa-grouped-verify.patch
apply "$aiter_root" 12-aiter-7556-qsa-flydsl-sparse-prefill.patch
apply /sgl-workspace/sglang 13-sglang-51179-qsa-flydsl-prefill.patch
apply /sgl-workspace/sglang 14-sglang-49852-mxfp4-moe-pad128.patch
apply /sgl-workspace/sglang 15-sglang-50863-hc-prefill-triton.patch
apply /sgl-workspace/sglang 16-sglang-50871-51263-qsa-prefill-topk-v2.patch
apply /sgl-workspace/sglang 21-sglang-50871-51242-qsa-decode-select.patch
apply /sgl-workspace/sglang 24-sglang-51180-gdn-verify.patch
apply "$aiter_root" 22-aiter-8174-moe-stage2-fp8.patch
apply /sgl-workspace/sglang 23-sglang-51198-moe-stage2-fp8.patch
# 26 brings patch 15's two files to sglang#50863's head, so 15 stays and 26 applies after it.
apply /sgl-workspace/sglang 26-sglang-50863-hc-prefill-triton-head.patch
apply /sgl-workspace/sglang 27-sglang-51023-decode-fp8-quant-fuse.patch

# The monokernel patch must be the code python imports (an editable install of the checkout).
python3 - <<'PY'
import importlib.util
import sys

spec = importlib.util.find_spec("sgl_amd_mono.qwen38_flash_next.gfx950.draft_head")
if spec is None or not spec.origin.startswith("/sgl-workspace/sgl-amd-mono/"):
    sys.exit(f"qwen3.8next: sgl_amd_mono imports from {spec and spec.origin}, expected /sgl-workspace/sgl-amd-mono")
print(f"qwen3.8next sgl-amd-mono patched: {spec.origin}")
PY

echo "$digest" > "$marker"
