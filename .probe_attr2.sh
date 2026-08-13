#!/usr/bin/env bash
# Attribute the still-unattributed vendor hunks: cudagraph_warmup_context,
# the tgemm call sites (#51713), and the #51473 MXFP4 TP8 shard. For each,
# print which vendor files carry it and whether the base image (target) has it.
set -u
D=/home/jiacao/3way-20260812-2214
R="$D/ref/usr/local/lib/python3.12/dist-packages/vllm"
V="$D/vendor/src/vllm/vllm"
T="$D/target/usr/local/lib/python3.12/dist-packages/vllm"

show() { # $1=marker
    echo "### $1"
    echo "  vendor files:"
    grep -rl -- "$1" "$V" --include='*.py' 2>/dev/null | sed "s|$V/|    |"
    echo "  target files:"
    grep -rl -- "$1" "$T" --include='*.py' 2>/dev/null | sed "s|$T/|    |"
    echo "  ref files:"
    grep -rl -- "$1" "$R" --include='*.py' 2>/dev/null | sed "s|$R/|    |"
    echo
}

show "cudagraph_warmup_context"
show "tgemm.mm"
show "AITER_MXFP4_BF16"

echo "### #51473: which MoEActivation branch guards AITER_MXFP4_BF16"
for t in "$T" "$V"; do
    echo "--- ${t##*/dist-packages/}${t##*/src/vllm/}"
    grep -rn -B4 -A4 "AITER_MXFP4_BF16" "$t" --include='*.py' 2>/dev/null | head -40
done
