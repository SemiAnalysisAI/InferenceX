#!/usr/bin/env bash

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

export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

if [[ $CONC -ge 16 ]]; then
  SCHEDULER_RECV_INTERVAL=30
else
  SCHEDULER_RECV_INTERVAL=10
fi
echo "SCHEDULER_RECV_INTERVAL: $SCHEDULER_RECV_INTERVAL, CONC: $CONC, ISL: $ISL, OSL: $OSL, TP: $TP, EP_SIZE: $EP_SIZE, DP_ATTENTION: $DP_ATTENTION"

DP_ATTN_ARGS=""
if [[ "$DP_ATTENTION" == "true" ]]; then
  DP_ATTN_ARGS="--enable-dp-attention --dp-size $TP"
fi

EVAL_CONTEXT_ARGS=""
if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    EVAL_CONTEXT_ARGS="--context-length $EVAL_MAX_MODEL_LEN"
fi

start_gpu_monitor

set -x
PYTHONNOUSERSITE=1 python3 -m sglang.launch_server --model-path $MODEL --host 0.0.0.0 --port $PORT --trust-remote-code \
--tensor-parallel-size=$TP --ep-size $EP_SIZE $DP_ATTN_ARGS \
--moe-runner-backend flashinfer_mxfp4 --moe-a2a-backend deepep \
--mem-fraction-static 0.82 \
--chunked-prefill-size 4096 --disable-flashinfer-autotune \
--scheduler-recv-interval $SCHEDULER_RECV_INTERVAL \
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
    --num-prompts $((CONC * 10)) \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

stop_gpu_monitor
set +x
