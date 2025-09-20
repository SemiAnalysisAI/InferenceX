#!/usr/bin/bash

# === Required Env Vars === 
# HF_TOKEN
# HF_HUB_CACHE
# IMAGE
# MODEL
# ISL
# OSL
# MAX_MODEL_LEN
# RANDOM_RANGE_RATIO
# TP
# CONC
# RESULT_FILENAME

# Reference
# https://rocm.docs.amd.com/en/docs-7.0-rc1/preview/benchmark-docker/inference-vllm-llama-3.1-405b-fp4.html#run-the-inference-benchmark

pip install amd-quark

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
export VLLM_TRITON_FP4_GEMM_USE_ASM=1
export VLLM_TRITON_FP4_GEMM_SPLITK_USE_BF16=1
export TRITON_HIP_PRESHUFFLE_SCALES=0
export VLLM_TRITON_FP4_GEMM_BPRESHUFFLE=0

set -x
vllm serve $MODEL --port=$PORT \
--tensor-parallel-size=$TP \
--distributed-executor-backend=mp \
--dtype=auto \
--kv-cache-dtype=fp8 \
--gpu-memory-utilization=0.92 \
--swap-space=64 \
--max-model-len=$MAX_MODEL_LEN \
--max-seq-len-to-capture=$MAX_MODEL_LEN \
--max-num-seqs=$CONC \
--max-num-batched-tokens=$(( $CONC * 128 )) \
--no-enable-prefix-caching \
--async-scheduling \
--disable-log-requests
