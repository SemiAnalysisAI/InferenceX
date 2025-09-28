#!/usr/bin/env bash

# ========= Required Env Vars =========
# HF_TOKEN
# HF_HUB_CACHE
# MODEL
# ISL
# OSL
# MAX_MODEL_LEN
# RANDOM_RANGE_RATIO
# TP
# CONC
# PORT
# RESULT_FILENAME

SERVER_LOG=$(mktemp /tmp/server-XXXXXX.log)
PORT=8888

export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4

if [[ "$ISL" == "1024" && "$OSL" == "1024" ]]; then
        export VLLM_ROCM_USE_AITER_MHA=0
        if [[ "$CONC" -le "16" ]]; then
                export VLLM_ROCM_USE_AITER_TRITON_BF16_GEMM=0
        else
                export VLLM_TRITON_FP4_GEMM_USE_ASM=1
        fi
elif [[ "$ISL" == "1024" && "$OSL" == "8192" ]]; then
        export VLLM_ROCM_USE_AITER_MHA=0
        if [[ "$CONC" -le "16" ]]; then
                export VLLM_ROCM_USE_AITER_TRITON_BF16_GEMM=0
        else
                export VLLM_TRITON_FP4_GEMM_USE_ASM=1
        fi
elif [[ "$ISL" == "8192" && "$OSL" == "1024" ]]; then
	if [[ "$CONC" -gt "16" ]]; then
		export VLLM_ROCM_USE_AITER_MHA=1
	else
		export VLLM_ROCM_USE_AITER_MHA=0
	fi
	if [[ "$CONC" -lt "16" && "$TP" -gt "1" ]]; then
		export VLLM_ROCM_USE_AITER_TRITON_BF16_GEMM=0
	else
		export VLLM_TRITON_FP4_GEMM_USE_ASM=1
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
--async-scheduling > $SERVER_LOG 2>&1 &

set +x
while IFS= read -r line; do
    printf '%s\n' "$line"
    if [[ "$line" == *"Application startup complete"* ]]; then
        break
    fi
done < <(tail -F -n0 "$SERVER_LOG")

set -x
git clone https://github.com/kimbochen/bench_serving.git
python3 bench_serving/benchmark_serving.py \
--model $MODEL --backend vllm \
--base-url "http://0.0.0.0:$PORT" \
--dataset-name random \
--random-input-len $ISL --random-output-len $OSL --random-range-ratio $RANDOM_RANGE_RATIO \
--num-prompts $(( $CONC * 10 )) --max-concurrency $CONC \
--request-rate inf --ignore-eos \
--save-result --percentile-metrics "ttft,tpot,itl,e2el" \
--result-dir /workspace/ --result-filename $RESULT_FILENAME.json

