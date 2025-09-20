#!/usr/bin/env bash

# ========= Required Env Vars =========
# HF_TOKEN
# HF_HUB_CACHE
# MODEL
# MAX_MODEL_LEN
# TP
# CONC
# PORT

export VLLM_ROCM_USE_AITER=1
export VLLM_USE_AITER_UNIFIED_ATTENTION=1 
export VLLM_ROCM_USE_AITER_MHA=0
export VLLM_USE_AITER_TRITON_FUSED_SPLIT_QKV_ROPE=1 
export VLLM_USE_AITER_TRITON_FUSED_ADD_RMSNORM_PAD=1
export TRITON_HIP_PRESHUFFLE_SCALES=1
export VLLM_USE_AITER_TRITON_GEMM=1

set -x
vllm serve $MODEL --port $PORT \
--tensor-parallel-size=$TP \
--compilation-config='{"compile_sizes": [1, 2, 4, 8, 16, 24, 32, 64], "full_cuda_graph": true}' \
--block-size=64 \
--no-enable-prefix-caching \
--disable-log-requests
