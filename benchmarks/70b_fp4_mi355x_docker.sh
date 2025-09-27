#!/usr/bin/env bash

# ========= Required Env Vars =========
# HF_TOKEN
# HF_HUB_CACHE
# MODEL
# PORT
# TP
# CONC
# MAX_MODEL_LEN

export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4

if [[ "$ISL" == "1024" && "$OSL" == "1024" ]]; then
	export VLLM_ROCM_USE_AITER_MHA=0
	if [[ "$CONC" -lt "16" ]]; then
		export VLLM_ROCM_USE_AITER_TRITON_BF16_GEMM=0
	else
		export VLLM_TRITON_FP4_GEMM_USE_ASM=1
	fi
elif [[ "$ISL" == "1024" && "$OSL" == "8192" ]]; then
	export VLLM_ROCM_USE_AITER_MHA=0
	if [[ "$CONC" -lt "16" ]]; then
		export VLLM_ROCM_USE_AITER_TRITON_BF16_GEMM=0
	else
		export VLLM_TRITON_FP4_GEMM_USE_ASM=1
	fi
elif [[ "$ISL" == "8192" && "$OSL" == "1024" ]]; then
	if [[ "$CONC" -gt "16" ]]; then
		export VLLM_ROCM_USE_AITER_MHA=1
	fi
fi

set -x
vllm serve $MODEL \
--host=0.0.0.0 \
--port $PORT \
--swap-space 64 \
--max-model-len $MAX_MODEL_LEN \
--tensor-parallel-size $TP \
--max-num-seqs 1024 \
--kv-cache-dtype fp8 \
--gpu-memory-utilization 0.94 \
--max-seq-len-to-capture $MAX_MODEL_LEN \
--max-num-batched-tokens 131072 \
--no-enable-prefix-caching \
--disable-log-requests \
--async-scheduling
