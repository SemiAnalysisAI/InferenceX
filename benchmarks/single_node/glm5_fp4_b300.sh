#!/usr/bin/env bash

# NOTE: At the time of submission, https://cookbook.sglang.io/autoregressive/GLM/GLM-5
# does not have a B300-specific recipe, so this script reuses the existing
# GLM-5 FP4 B200 SGLang recipe as-is until B300-specific tuning is available.

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME \
    EP_SIZE

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

nvidia-smi

hf download "$MODEL"

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

echo "EP_SIZE: $EP_SIZE, CONC: $CONC, ISL: $ISL, OSL: $OSL"

EVAL_CONTEXT_ARGS=""
if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    EVAL_CONTEXT_ARGS="--context-length $EVAL_MAX_MODEL_LEN"
fi
# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor

set -x
PYTHONNOUSERSITE=1 python3 -m sglang.launch_server --model-path=$MODEL --host=0.0.0.0 --port=$PORT \
--trust-remote-code \
--tensor-parallel-size=$TP \
--data-parallel-size 1 --expert-parallel-size $EP_SIZE \
--disable-radix-cache \
--quantization modelopt_fp4 \
--kv-cache-dtype fp8_e4m3 \
--nsa-decode-backend trtllm \
--nsa-prefill-backend trtllm \
--moe-runner-backend flashinfer_trtllm \
--enable-flashinfer-allreduce-fusion \
--cuda-graph-max-bs 256 \
--max-prefill-tokens 32768 \
--chunked-prefill-size 32768 \
--mem-fraction-static 0.9 \
--stream-interval 30 \
--scheduler-recv-interval 10 \
--tokenizer-worker-num 6 \
--tokenizer-path $MODEL $EVAL_CONTEXT_ARGS > $SERVER_LOG 2>&1 &

SERVER_PID=$!

# Wait for server to be ready
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

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

# Stop GPU monitoring
stop_gpu_monitor
set +x
