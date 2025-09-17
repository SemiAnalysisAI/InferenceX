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

echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"

huggingface-cli download $MODEL

SERVER_LOG=$(mktemp /tmp/server-XXXXXX.log)
PORT=8888

# Reference
# https://rocm.docs.amd.com/en/docs-7.0-rc1/preview/benchmark-docker/inference-vllm-gpt-oss-120b.html#run-the-inference-benchmark

export VLLM_USE_AITER_TRITON_FUSED_SPLIT_QKV_ROPE=1
export VLLM_USE_AITER_TRITON_FUSED_ADD_RMSNORM_PAD=1
export VLLM_USE_AITER_TRITON_GEMM=1
export VLLM_ROCM_USE_AITER=1
export VLLM_USE_AITER_UNIFIED_ATTENTION=1
export VLLM_ROCM_USE_AITER_MHA=0
export TRITON_HIP_PRESHUFFLE_SCALES=1
export VLLM_DISABLE_COMPILE_CACHE=1
export HSA_NO_SCRATCH_RECLAIM=1

set -x
vllm serve $MODEL --port=$PORT \
--tensor-parallel-size=$TP \
--gpu-memory-utilization=0.95 \
--compilation-config='{"full_cuda_graph": true}' \
--block-size=64 \
--swap-space=16 \
--no-enable-prefix-caching \
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
--model=$MODEL --backend=vllm \
--base-url="http://0.0.0.0:$PORT" \
--dataset-name=random \
--random-input-len=$ISL --random-output-len=$OSL --random-range-ratio=$RANDOM_RANGE_RATIO \
--num-prompts=$(( $CONC * 10 )) --max-concurrency=$CONC \
--request-rate=inf --ignore-eos \
--save-result --percentile-metrics='ttft,tpot,itl,e2el' \
--result-dir=/workspace/ \
--result-filename=$RESULT_FILENAME.json
