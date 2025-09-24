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

# Reference
# https://rocm.docs.amd.com/en/docs-7.0-rc1/preview/benchmark-docker/inference-vllm-llama-3.3-70b-fp8.html#run-the-inference-benchmark

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

if [[ "$ISL" == "1024" && "$OSL" == "1024" ]]; then
    export VLLM_ROCM_USE_AITER_MHA=0
elif [[ "$ISL" == "1024" && "$OSL" == "8192" ]]; then
    export VLLM_ROCM_USE_AITER_MHA=0
elif [[ "$ISL" == "8192" && "$OSL" == "1024" ]]; then
    export VLLM_ROCM_USE_AITER_MHA=1
fi

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
--disable-log-requests \
> $SERVER_LOG 2>&1 &

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

exit
