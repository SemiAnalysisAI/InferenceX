#!/usr/bin/env bash

# ========= Required Env Vars =========
# HF_TOKEN
# HF_HUB_CACHE
# MODEL
# PORT
# TP
# CONC
# MAX_MODEL_LEN
max_model_len=16384            # Must be >= the input + output length
max_seq_len_to_capture=10240   # Beneficial to set this to max_model_len
max_num_seqs=1024
max_num_batched_tokens=131072  # Smaller values may result in better TTFT but worse TPOT / Throughput

export VLLM_USE_V1=1
export VLLM_USE_AITER_TRITON_ROPE=1
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_RMSNORM=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4



vllm serve ${MODEL} \
    --host localhost \
    --port 6000 \
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