#!/usr/bin/env bash
# Does the vendor image's MegaMoE tree match what aiter PR #4439 merged, or is
# the vendor carrying a private variant? Sizes differ enough between vendor and
# aiter/main that this needs checking at the merge commit itself (97d0c6e4cb7a),
# not at main -- main has 87 commits of drift past post2.
#
# If sizes match at 97d0c6e4, the port is a clean cherry-pick from upstream.
# If they don't, the vendor delta is real and needs its own PR.
set -u
REF="${1:-97d0c6e4cb7a}"
V=$(find /home/jiacao/3way-20260812-2214/vendor -maxdepth 9 -type d -name aiter -path '*dist-packages*' | head -1)

printf "%-24s %10s %10s %10s  %s\n" FILE VENDOR "@$REF" MAIN VERDICT
for f in __init__ dispatch gemm1 gemm2 gemm_util mega_moe_config \
         mega_moe_stage1 mega_moe_stage2 mega_moe_v2 quant; do
    p="aiter/ops/flydsl/kernels/mega_moe/$f.py"
    lv=$(stat -c %s "$V/ops/flydsl/kernels/mega_moe/$f.py" 2>/dev/null || echo -)
    sr=$(gh api "/repos/ROCm/aiter/contents/$p?ref=$REF" --jq '.size' 2>/dev/null)
    [[ "$sr" =~ ^[0-9]+$ ]] || sr=-
    sm=$(gh api "/repos/ROCm/aiter/contents/$p?ref=main" --jq '.size' 2>/dev/null)
    [[ "$sm" =~ ^[0-9]+$ ]] || sm=-
    if   [ "$lv" = "$sr" ]; then v="EXACT@ref"
    elif [ "$lv" = "$sm" ]; then v="EXACT@main"
    elif [ "$sr" = "-" ];   then v="UPSTREAM-MISSING"
    else                         v="DIFFERS"
    fi
    printf "%-24s %10s %10s %10s  %s\n" "$f.py" "$lv" "$sr" "$sm" "$v"
done
