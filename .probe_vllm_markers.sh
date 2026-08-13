#!/usr/bin/env bash
# For each vendor-only vLLM change, pick a marker string that only exists if the
# feature is present, and check it in ref / target / vendor.
#
# "DIFFERS" from the file-level probe is not evidence of a missing feature --
# 465 upstream commits touch these files anyway. Markers are.
#   ref     absent, vendor present, target absent  -> we must add it
#   ref     absent, vendor present, target present -> upstream already landed it
set -u
D=/home/jiacao/3way-20260812-2214
R="$D/ref/usr/local/lib/python3.12/dist-packages/vllm"
V="$D/vendor/src/vllm/vllm"
T="$D/target/usr/local/lib/python3.12/dist-packages/vllm"

row() { # $1=marker  $2=relpath-or-TREE  $3=label
    local m="$1" p="$2" lbl="$3" out=""
    for side in R V T; do
        local base; base=$(eval echo "\$$side")
        local n
        if [ "$p" = "TREE" ]; then
            n=$(grep -rl -- "$m" "$base" --include='*.py' 2>/dev/null | wc -l)
        elif [ -f "$base/$p" ]; then
            n=$(grep -c -- "$m" "$base/$p" 2>/dev/null || echo 0)
        else
            n="-"
        fi
        out="$out$(printf '%6s' "$n")"
    done
    printf "%s  %-34s %s\n" "$out" "$lbl" "$m"
}

echo "   ref vendor target  FEATURE                            MARKER"
row "VLLM_ROCM_DSV4_SPARSE_GLUON"        TREE "gluon env knob"
row "pa_decode_sparse"                   TREE "gluon sparse decode call"
row "mla_gluon"                          TREE "gluon MLA"
row "flydsl_mega_moe"                    TREE "MegaMoE backend name"
row "mega_moe"                           TREE "MegaMoE (any)"
row "MegaMoE"                            TREE "MegaMoE (class)"
row "FUSION_SHARED_EXPERTS"              TREE "FSE env knob"
row "fhmoe"                              TREE "FHMoE"
row "AITER_MXFP4_BF16"                   TREE "PR #51473 marker"
row "tuned_gemm"                         TREE "tuned GEMM"
echo
echo "-- new files --"
for f in models/deepseek_v4/amd/mega_moe_experts.py models/deepseek_v4/amd/mega_moe_runtime.py; do
    printf "  ref=%-9s target=%-9s %s\n" \
        "$([ -f "$R/$f" ] && echo yes || echo no)" \
        "$([ -f "$T/$f" ] && echo yes || echo no)" "$f"
done
