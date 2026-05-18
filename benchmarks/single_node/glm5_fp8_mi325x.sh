#!/usr/bin/env bash

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}
CONTEXT_LENGTH=$((ISL + OSL + 20))
MAX_PREFILL_TOKENS=32768

EVAL_CONTEXT_ARGS=""
if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    EVAL_CONTEXT_ARGS="--context-length $EVAL_MAX_MODEL_LEN"
else EVAL_CONTEXT_ARGS="--context-length $CONTEXT_LENGTH"
fi

start_gpu_monitor

# Follows the qwen3.5-fp8-mi325x recipe (AMD Andy's LinkedIn-recommended
# args for sglang ROCm on mi325x) with GLM-5-specific parsers.
python3 -m sglang.launch_server \
    --attention-backend aiter \
    --model-path $MODEL \
    --host=0.0.0.0 \
    --port $PORT \
    --tensor-parallel-size $TP \
    --data-parallel-size 1 \
    --trust-remote-code \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --tokenizer-worker-num 6 \
    --enable-aiter-allreduce-fusion \
    --cuda-graph-max-bs $CONC \
    --disable-radix-cache \
    --max-prefill-tokens $MAX_PREFILL_TOKENS \
    --scheduler-recv-interval 30 \
    --mem-fraction-static 0.75 $EVAL_CONTEXT_ARGS > $SERVER_LOG 2>&1 &

SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

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
