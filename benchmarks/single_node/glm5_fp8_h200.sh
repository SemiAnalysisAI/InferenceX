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

nvidia-smi

hf download "$MODEL"

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor

set -x
python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tp-size "$TP" \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --mem-fraction-static 0.85 \
  --served-model-name glm-5-fp8 \
  > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!

# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# If your --model is not on Hugging Face (e.g. glm-5-fp8), set BENCH_TOKENIZER
# to a local path or a public HF model id, e.g. export BENCH_TOKENIZER=THUDM/glm-4-9b-chat
TOKENIZER_ARGS=""
if [ -n "${BENCH_TOKENIZER:-}" ]; then
  TOKENIZER_ARGS="--tokenizer $BENCH_TOKENIZER"
fi

num_prompts=$((CONC * 5))
SGLANG_URL="http://0.0.0.0:$PORT"

python3 utils/bench_serving/benchmark_serving.py \
    --backend openai-chat \
    --base-url "$SGLANG_URL" \
    --endpoint /v1/chat/completions \
    --model glm-5-fp8 \
    $TOKENIZER_ARGS \
    --dataset-name random \
    --num-prompts "$num_prompts" \
    --random-input-len "$ISL" \
    --random-output-len "$OSL" \
    --random-range-ratio "${RANDOM_RANGE_RATIO:-0.8}" \
    --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el \
    --max-concurrency "$CONC" \
    --save-result \
    --result-dir /workspace \
    --result-filename "$RESULT_FILENAME.json"

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT" --concurrent-requests $CONC
    append_lm_eval_summary
fi

# Stop GPU monitoring
stop_gpu_monitor
set +x
