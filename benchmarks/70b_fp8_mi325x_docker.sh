#!/usr/bin/env bash

# ========= Required Env Vars =========
# HF_TOKEN
# HF_HUB_CACHE
# MODEL
# PORT
# TP
# CONC
# MAX_MODEL_LEN

# Reference
# https://rocm.docs.amd.com/en/docs-7.0-rc1/preview/benchmark-docker/inference-vllm-llama-3.3-70b-fp8.html#run-the-inference-benchmark

if [[ "$ISL" == "1024" && "$OSL" == "1024" ]]; then
    export VLLM_ROCM_USE_AITER_MHA=0
elif [[ "$ISL" == "1024" && "$OSL" == "8192" ]]; then
    export VLLM_ROCM_USE_AITER_MHA=0
elif [[ "$ISL" == "8192" && "$OSL" == "1024" ]]; then
    if [[ "$CONC" -gt "16" ]]; then
        export VLLM_ROCM_USE_AITER_MHA=1
    fi
fi

# Patch the aiter config script to deal
# with weird strings reported by /opt/rocm/llvm/bin/amdgpu-arch.
file_to_patch='/opt/venv/lib/python3.10/site-packages/aiter_meta/csrc/cpp_itfs/utils.py'
sed -i'' -e 's#archs = \[arch.strip() for arch in archs\]#archs = \[arch.strip().split(":")\[0\] for arch in archs\]#'  $file_to_patch


# In this specific case, float16 performs better than the datatype
# picked by vllm when using auto for --dtype (bfloat16).
set -x
vllm serve $MODEL --port=$PORT \
--swap-space=64 \
--gpu-memory-utilization=0.94 \
--dtype=float16 --kv-cache-dtype=fp8 \
--distributed-executor-backend=mp --tensor-parallel-size=$TP \
--max-model-len=$MAX_MODEL_LEN \
--max-seq-len-to-capture=$MAX_MODEL_LEN \
--max-num-seqs=$CONC \
--max-num-batched-tokens=131072 \
--no-enable-prefix-caching \
--async-scheduling \
--disable-log-requests
