#!/usr/bin/env bash
# 444/538 shared aiter files drift between v0.1.19 and the MegaMoE merge, so a
# wholesale sync is out. The narrow question instead: if we copy ONLY the
# feature-carrying files from main onto the base image, what do they import that
# the base image does not have? That set is the true blast radius.
set -u
W=/tmp/aitermain/aiter
T=/home/jiacao/3way-20260812-2214/target/usr/local/lib/python3.12/dist-packages/aiter

FEATURE_FILES="
ops/flydsl/kernels/mega_moe/__init__.py
ops/flydsl/kernels/mega_moe/dispatch.py
ops/flydsl/kernels/mega_moe/gemm1.py
ops/flydsl/kernels/mega_moe/gemm2.py
ops/flydsl/kernels/mega_moe/gemm_util.py
ops/flydsl/kernels/mega_moe/mega_moe_config.py
ops/flydsl/kernels/mega_moe/mega_moe_stage1.py
ops/flydsl/kernels/mega_moe/mega_moe_stage2.py
ops/flydsl/kernels/mega_moe/mega_moe_v2.py
ops/flydsl/kernels/mega_moe/quant.py
ops/triton/_gluon_kernels/gfx950/attention/pa_decode_sparse.py
ops/triton/attention/pa_decode_sparse.py
ops/triton/utils/common_utils.py
fhmoe.py
ops/flydsl/fhmoe.py
ops/flydsl/kernels/fhmoe.py
"

echo "=== what these files import from aiter, and whether the base image has it ==="
for f in $FEATURE_FILES; do
    [ -f "$W/$f" ] || { echo "  !! not in main: $f"; continue; }
    grep -hoE '^\s*(from|import)\s+aiter[.a-zA-Z0-9_]*' "$W/$f"
done | sed -E 's/^\s*(from|import)\s+//' | sort -u > /tmp/imports.txt

while IFS= read -r mod; do
    rel="${mod#aiter}"; rel="${rel#.}"; rel="${rel//./\/}"
    if [ -z "$rel" ]; then continue; fi
    if [ -f "$T/$rel.py" ] || [ -d "$T/$rel" ]; then
        # exists -- but does it differ from main?
        if [ -f "$T/$rel.py" ] && [ -f "$W/$rel.py" ] && ! cmp -s "$T/$rel.py" "$W/$rel.py"; then
            echo "  DRIFTED  aiter.$(echo "$rel" | tr '/' '.')"
        else
            echo "  ok       aiter.$(echo "$rel" | tr '/' '.')"
        fi
    else
        echo "  MISSING  aiter.$(echo "$rel" | tr '/' '.')"
    fi
done < /tmp/imports.txt

echo
echo "=== do the FlyDSL kernels depend on the compiled csrc that changed? ==="
grep -l "import aiter.jit\|from aiter.jit\|compile_ops\|get_module" $(for f in $FEATURE_FILES; do [ -f "$W/$f" ] && echo "$W/$f"; done) 2>/dev/null | sed "s|$W/|  |"
