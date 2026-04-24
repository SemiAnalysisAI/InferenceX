#!/usr/bin/env bash

# NOTE: https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4
# lists H200 (not B300) as the FP8 target, and only the Flash-FP8
# checkpoint is live (Pro-FP8 is still pending upload). This script
# reuses the H200 Flash Max-Throughput recipe (DP + DeepEP, no MTP) on
# B300 using the Blackwell image with SGLANG_DSV4_FP4_EXPERTS=0 to swap
# the experts to FP8. Prefix caching is disabled.

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME \
    EP_SIZE \
    DP_ATTENTION

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

hf download "$MODEL"

nvidia-smi

export SGLANG_DSV4_FP4_EXPERTS=0
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

echo "TP: $TP, EP_SIZE: $EP_SIZE, DP_ATTENTION: $DP_ATTENTION, CONC: $CONC, ISL: $ISL, OSL: $OSL"

DP_ATTN_ARGS=""
if [ "$DP_ATTENTION" = "true" ]; then
    DP_ATTN_ARGS="--data-parallel-size $TP --enable-dp-attention"
fi

EVAL_CONTEXT_ARGS=""
if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    EVAL_CONTEXT_ARGS="--context-length $EVAL_MAX_MODEL_LEN"
fi

start_gpu_monitor

set -x
PYTHONNOUSERSITE=1 python3 -m sglang.launch_server --model-path=$MODEL --host=0.0.0.0 --port=$PORT \
--trust-remote-code \
--tensor-parallel-size=$TP --ep-size $EP_SIZE $DP_ATTN_ARGS \
--moe-a2a-backend deepep \
--deepep-config '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}' \
--cuda-graph-max-bs 128 \
--max-running-requests 256 \
--disable-radix-cache $EVAL_CONTEXT_ARGS > $SERVER_LOG 2>&1 &

SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

pip install -q datasets pandas

run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts "$((CONC * 10))" \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

stop_gpu_monitor
set +x
