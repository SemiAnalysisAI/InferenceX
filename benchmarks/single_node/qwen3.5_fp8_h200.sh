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

hf download "$MODEL"

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}
MAX_SEQ_LEN=$((ISL + OSL + 20))

echo "CONC: $CONC, ISL: $ISL, OSL: $OSL, MAX_SEQ_LEN: $MAX_SEQ_LEN"

# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor

set -x
python3 -m sglang.launch_server \
  --model "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tp "$TP" \
  --expert-parallel-size "$EP_SIZE" \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --enable-flashinfer-allreduce-fusion \
  --max-running-requests 128 \
  --chunked-prefill-size 16384 \
  --decode-log-interval 1 \
  --mem-fraction-static 0.8 \
  --cuda-graph-max-bs "$CONC" \
  --context-length "$MAX_SEQ_LEN" \
  --kv-cache-dtype fp8_e4m3 \
  --quantization fp8 \
  --attention-backend flashinfer \
  --stream-interval 50 \
  --tokenizer-worker-num 6 \
  --mamba-ssm-dtype bfloat16 \
  --disable-radix-cache \
  --trust-remote-code \
  > "$SERVER_LOG" 2>&1 &

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
    run_eval --framework lm-eval --port "$PORT" --concurrent-requests $CONC
    append_lm_eval_summary
fi

# Stop GPU monitoring
stop_gpu_monitor
set +x
