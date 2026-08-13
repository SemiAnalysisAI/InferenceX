#!/usr/bin/env bash
# Which of the vendor-only aiter files exist at tag v0.1.19.post2 (a63ede724b15)?
#
# post2 already contains the PR #4269 merge (compare says ahead=0/behind=24),
# so anything #4269 shipped should be there. Anything still missing came from a
# different source -- either a later upstream PR, or nothing upstream at all.
# That split is exactly what decides "bump the pin" vs "we must open a PR".
set -u
REF="${1:-a63ede724b15}"   # v0.1.19.post2
TAG_NAME="${2:-v0.1.19.post2}"

check() { # $1 = repo path
    # `gh api --jq .size` prints the 404 JSON body on a miss, which is a
    # non-empty string -- test for an all-digits size instead of non-emptiness.
    local p="$1" s
    s=$(gh api "/repos/ROCm/aiter/contents/$p?ref=$REF" --jq '.size' 2>/dev/null)
    if [[ "$s" =~ ^[0-9]+$ ]]; then printf "  %-8s %-9s %s\n" "PRESENT" "${s}B" "$p"
    else                            printf "  %-8s %-9s %s\n" "ABSENT"  "-"    "$p"; fi
}

echo "=== $TAG_NAME ($REF) ==="
echo "-- FHMoE / FSE (PR #4269) --"
check aiter/fhmoe.py
check aiter/ops/flydsl/fhmoe.py
check aiter/aot/flydsl/fhmoe.py
check aiter/ops/flydsl/kernels/fhmoe.py
check aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py
check aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage_common.py

echo "-- MegaMoE kernel tree --"
for f in __init__ dispatch gemm1 gemm2 gemm_util mega_moe_config \
         mega_moe_stage1 mega_moe_stage2 mega_moe_v2 quant; do
    check "aiter/ops/flydsl/kernels/mega_moe/$f.py"
done

echo "-- tuned-GEMM data --"
check aiter/configs/model_configs/a8w8_blockscale_tuned_gemm_dsv4.csv

echo "-- Gluon sparse attention (PR #4382) --"
check aiter/ops/triton/_gluon_kernels/gfx950/attention/pa_decode_sparse.py
