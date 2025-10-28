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

export HF_MODULES_CACHE="/tmp/hf_modules_cache/"

SERVER_LOG=$(mktemp /tmp/server-XXXXXX.log)

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
export VLLM_ROCM_USE_AITER_TRITON_FUSED_SHARED_EXPERTS=1


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
    --kv-cache-dtype auto > $SERVER_LOG 2>&1 &

set +x
while IFS= read -r line; do
    printf '%s\n' "$line"
    if [[ "$line" =~ Application\ startup\ complete ]]; then
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
