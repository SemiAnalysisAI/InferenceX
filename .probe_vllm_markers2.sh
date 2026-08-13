#!/usr/bin/env bash
# Second pass: the coarse markers were too coarse. FUSION_SHARED_EXPERTS and
# AITER_MXFP4_BF16 appear on all three sides, so their mere presence proves
# nothing -- the vendor delta is in the specific condition / call site.
set -u
D=/home/jiacao/3way-20260812-2214
R="$D/ref/usr/local/lib/python3.12/dist-packages/vllm"
V="$D/vendor/src/vllm/vllm"
T="$D/target/usr/local/lib/python3.12/dist-packages/vllm"

echo "=== PR #51473 exact condition (oracle/mxfp4.py) ==="
for s in R V T; do b=$(eval echo "\$$s")
    printf "  %-7s " "$s"
    grep -n "AITER_MXFP4_BF16" "$b/model_executor/layers/fused_moe/oracle/mxfp4.py" 2>/dev/null \
        | head -4 | tr '\n' '|' ; echo
done

echo
echo "=== FSE: where do the extra 7 vendor hits live? ==="
echo "--- vendor files mentioning FUSION_SHARED_EXPERTS ---"
grep -rl "FUSION_SHARED_EXPERTS" "$V" --include='*.py' 2>/dev/null | sed "s|$V/|    |"
echo "--- target files mentioning FUSION_SHARED_EXPERTS ---"
grep -rl "FUSION_SHARED_EXPERTS" "$T" --include='*.py' 2>/dev/null | sed "s|$T/|    |"

echo
echo "=== MegaMoE: which files, each side ==="
for s in R V T; do b=$(eval echo "\$$s"); echo "--- $s ---"
    grep -rl "mega_moe\|MegaMoE" "$b" --include='*.py' 2>/dev/null | sed "s|$b/|    |"
done

echo
echo "=== tuned_gemm call sites ==="
for s in R V T; do b=$(eval echo "\$$s"); echo "--- $s ---"
    grep -rn "tuned_gemm" "$b" --include='*.py' 2>/dev/null | sed "s|$b/|    |" | head -6
done
