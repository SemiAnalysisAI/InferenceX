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
    EP_SIZE

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

nvidia-smi

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

CONTEXT_LENGTH=$((ISL + OSL + 20))
if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    CONTEXT_LENGTH="$EVAL_MAX_MODEL_LEN"
fi

# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor

set -x
SGLANG_ENABLE_SPEC_V2=1 PYTHONNOUSERSITE=1 python3 -m sglang.launch_server --model-path=$MODEL --host=0.0.0.0 --port=$PORT \
--trust-remote-code \
--tensor-parallel-size=$TP --data-parallel-size=1 --expert-parallel-size=$EP_SIZE \
--enable-symm-mem \
--disable-radix-cache \
--quantization fp8 \
--kv-cache-dtype fp8_e4m3 \
--mamba-ssm-dtype bfloat16 \
--attention-backend trtllm_mha \
--moe-runner-backend flashinfer_trtllm \
--cuda-graph-max-bs $CONC \
--max-running-requests $CONC \
--max-prefill-tokens 16384 \
--chunked-prefill-size 16384 \
--mem-fraction-static 0.8 \
--stream-interval 50 \
--scheduler-recv-interval 10 \
--tokenizer-worker-num 6 \
--tokenizer-path $MODEL \
--speculative-algorithm EAGLE \
--speculative-num-steps 3 \
--speculative-eagle-topk 1 \
--speculative-num-draft-tokens 4 \
--context-length $CONTEXT_LENGTH > $SERVER_LOG 2>&1 &

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
    --result-dir /workspace/ \
    --use-chat-template

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

# Stop GPU monitoring
stop_gpu_monitor
set +x
