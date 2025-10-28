#!/usr/bin/env bash

# ========= Required Env Vars =========
# HF_TOKEN
# HF_HUB_CACHE
# MODEL
# PORT
# TP
# CONC
# MAX_MODEL_LEN


# If the machine runs a MEC FW older than 177, RCCL
# cannot reclaim some memory.
# Disable that features to avoid crashes.
# This is related to the changes in the driver at:
# https://rocm.docs.amd.com/en/docs-6.4.3/about/release-notes.html#amdgpu-driver-updates
version=`rocm-smi --showfw | grep MEC | head -n 1 |  awk '{print $NF}'`
if [[ "$version" == "" || $version -lt 177 ]]; then
  export HSA_NO_SCRATCH_RECLAIM=1
fi


max_model_len=16384            # Must be >= the input + output length
max_seq_len_to_capture=10240   # Beneficial to set this to max_model_len
max_num_seqs=1024
max_num_batched_tokens=131072  # Smaller values may result in better TTFT but worse TPOT / Throughput

export VLLM_USE_V1=1
export VLLM_USE_AITER_TRITON_ROPE=1
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_RMSNORM=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4


export VLLM_ROCM_USE_AITER_TRITON_FUSED_RMSNORM_FP8_QUANT=1
export VLLM_ROCM_USE_AITER_TRITON_FUSED_MUL_ADD=1
export VLLM_ROCM_USE_AITER_TRITON_FUSED_SHARED_EXPERTS=0

set -x
vllm serve ${MODEL} \
    --host localhost \
    --port $PORT \
    --swap-space 64 \
    --tensor-parallel-size $TP \
    --max-num-seqs ${max_num_seqs} \
    --no-enable-prefix-caching \
    --max-num-batched-tokens ${max_num_batched_tokens} \
    --max-model-len ${max_model_len} \
    --block-size 1 \
    --gpu-memory-utilization 0.95 \
    --max-seq-len-to-capture ${max_seq_len_to_capture} \
    --async-scheduling \
    --kv-cache-dtype auto
