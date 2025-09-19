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

# export HF_HUB_OFFLINE=1
export VLLM_USE_V1=1
export VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1
export AMDGCN_USE_BUFFER_OPS=1
export VLLM_USE_AITER_TRITON_ROPE=1
export TRITON_HIP_ASYNC_COPY_BYPASS_PERMUTE=1
export TRITON_HIP_USE_ASYNC_COPY=1
export TRITON_HIP_USE_BLOCK_PINGPONG=1
export TRITON_HIP_ASYNC_FAST_SWIZZLE=1
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_RMSNORM=1

set -x
vllm serve $MODEL --port=$PORT \
--swap-space=64 \
--gpu-memory-utilization=0.94 \
--dtype=auto --kv-cache-dtype=fp8 \
--distributed-executor-backend=mp --tensor-parallel-size=$TP \
--max-model-len=$MAX_MODEL_LEN \
--max-seq-len-to-capture=$MAX_MODEL_LEN \
--max-num-seqs=$CONC \
--max-num-batched-tokens=131072 \
--no-enable-prefix-caching \
--async-scheduling \
--disable-log-requests
