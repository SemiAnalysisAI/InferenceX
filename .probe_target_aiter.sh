#!/usr/bin/env bash
# What does the BASE image's aiter (v0.1.19, target/) already carry, and what
# must the container patch add? Anything MISSING here is a real gap for the
# corresponding upstream aiter PR.
set -u
T=/home/jiacao/3way-20260812-2214/target/usr/local/lib/python3.12/dist-packages/aiter

p() { # $1=path-relative-to-aiter  $2=label
    if [ -e "$T/$1" ]; then echo "  PRESENT  $2  ($1)"; else echo "  MISSING  $2  ($1)"; fi
}
m() { # $1=marker  $2=file  $3=label
    if grep -qF -- "$1" "$T/$2" 2>/dev/null; then echo "  PRESENT  $3"; else echo "  MISSING  $3"; fi
}

echo "== #4382 gfx950 gluon sparse decode =="
p "ops/triton/_gluon_kernels/gfx950/attention/pa_decode_sparse.py" "gluon kernel"
m "_pa_decode_sparse_gfx950_gluon" "ops/triton/attention/pa_decode_sparse.py" "gfx950 routing entry"

echo "== #4673 buffer_load span fix =="
m "max_addressable_bytes" "ops/triton/utils/common_utils.py" "max_addressable_bytes helper"

echo "== #4439 MegaMoE =="
p "ops/flydsl/kernels/mega_moe/mega_moe_v2.py" "mega_moe package"
p "ops/flydsl/kernels/mega_moe/__init__.py" "mega_moe __init__"

echo "== #4269 FSE / heterogeneous shared expert (FHMoE) =="
grep -rlF "shared_expert_id" "$T" --include='*.py' 2>/dev/null | sed 's|^|  hit: |' | head
grep -rlF "fhmoe" "$T" --include='*.py' 2>/dev/null | sed 's|^|  hit: |' | head

echo "== #4664 tuned GEMM CSV rows =="
p "configs/model_configs/dsv4_a8w8_blockscale_tuned_gemm.csv" "dsv4 tuned-gemm csv"
if [ -e "$T/configs/model_configs/dsv4_a8w8_blockscale_tuned_gemm.csv" ]; then
    echo "    lines: $(wc -l < "$T/configs/model_configs/dsv4_a8w8_blockscale_tuned_gemm.csv")"
    grep -c '^6144,7168\|^65536,1536\|^7168,3072' "$T/configs/model_configs/dsv4_a8w8_blockscale_tuned_gemm.csv" \
        | sed 's|^|    dsv4 shapes present: |'
fi
