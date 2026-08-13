#!/usr/bin/env bash
# Classify each of the 19 vendor-changed vLLM files by which feature it carries.
# Print the ref->vendor diff for the small ones in full and a keyword census for
# the large ones, so every hunk can be attributed to gluon / MegaMoE / FSE /
# #51473 / other before it goes in a waiver table.
set -u
D=/home/jiacao/3way-20260812-2214
R="$D/ref/usr/local/lib/python3.12/dist-packages/vllm"
V="$D/vendor/src/vllm/vllm"

FILES="
config/kernel.py
model_executor/layers/fused_moe/experts/rocm_aiter_moe.py
model_executor/model_loader/utils.py
models/deepseek_v4/attention.py
platforms/interface.py
platforms/rocm.py
v1/worker/gpu/cudagraph_utils.py
v1/worker/gpu_ubatch_wrapper.py
models/deepseek_v4/amd/dspark.py
models/deepseek_v4/amd/mtp.py
v1/worker/gpu_model_runner.py
"
for f in $FILES; do
    echo "################ $f"
    diff -u "$R/$f" "$V/$f" | tail -n +3
done

echo
echo "############ keyword census for the large files ############"
for f in _aiter_ops.py models/deepseek_v4/amd/model.py models/deepseek_v4/amd/rocm.py \
         v1/attention/ops/rocm_aiter_mla_sparse.py; do
    echo "--- $f"
    diff -u "$R/$f" "$V/$f" | grep '^+' | grep -oiE \
      "gluon|mega_?moe|flydsl|fhmoe|fusion_shared|tgemm|tuned_gemm|sparse|mxfp4|dspark" \
      | tr 'A-Z' 'a-z' | sort | uniq -c | sort -rn | sed 's/^/     /'
done
