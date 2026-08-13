#!/usr/bin/env bash
# Which of the four DSv4 features are already in each image's aiter tree?
#
# The question this answers: aiter ships as its own wheel inside each image, so
# "the upstream PR is merged" does not imply "this image's aiter has it". Check
# the trees, not the PR states.
set -u
D="${1:-/home/jiacao/3way-20260812-2214}"

for side in ref vendor target; do
    A=$(find "$D/$side" -maxdepth 9 -type d -name aiter -path '*dist-packages*' 2>/dev/null | head -1)
    echo "=== $side"
    echo "    root: ${A:-<absent>}"
    [ -z "$A" ] && continue

    n=$(ls "$A/ops/flydsl/kernels/mega_moe"/*.py 2>/dev/null | wc -l)
    echo "    MegaMoE kernels (ops/flydsl/kernels/mega_moe/*.py) : ${n}"

    for f in \
        "ops/triton/_gluon_kernels/gfx950/attention/pa_decode_sparse.py" \
        "fhmoe.py" \
        "ops/flydsl/fhmoe.py" \
        "aot/flydsl/fhmoe.py" \
        "ops/flydsl/kernels/mixed_moe_gemm_2stage_common.py" \
    ; do
        [ -f "$A/$f" ] && echo "    PRESENT $f" || echo "    absent  $f"
    done

    echo -n "    tgemm: "
    ls "$A"/ops/*tgemm* "$A"/*tgemm* 2>/dev/null | head -3 | tr '\n' ' '
    grep -rl "tgemm" "$A" --include='*.py' 2>/dev/null | wc -l | xargs echo "files mentioning tgemm:"
done
