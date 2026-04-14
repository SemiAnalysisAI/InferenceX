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
    MAX_MODEL_LEN

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

nvidia-smi

hf download "$MODEL"

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

# MTP (Multi-Token Prediction) Config - EAGLE speculative decoding
SPECULATIVE_NUM_STEPS=3
SPECULATIVE_DRAFT_TOKENS=4
SPECULATIVE_EAGLE_TOPK=1

echo "CONC: $CONC, ISL: $ISL, OSL: $OSL, MAX_MODEL_LEN: $MAX_MODEL_LEN"

# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor

set -x
SGLANG_ENABLE_SPEC_V2=1 python3 -m sglang.launch_server \
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
  --mem-fraction-static 0.8 \
  --cuda-graph-max-bs "$CONC" \
  --context-length "$MAX_MODEL_LEN" \
  --kv-cache-dtype fp8_e4m3 \
  --quantization fp8 \
  --attention-backend flashinfer \
  --stream-interval 50 \
  --tokenizer-worker-num 6 \
  --mamba-ssm-dtype bfloat16 \
  --disable-radix-cache \
  --trust-remote-code \
  --speculative-algorithm EAGLE \
  --speculative-num-steps "$SPECULATIVE_NUM_STEPS" \
  --speculative-num-draft-tokens "$SPECULATIVE_DRAFT_TOKENS" \
  --speculative-eagle-topk "$SPECULATIVE_EAGLE_TOPK" \
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
    --use-chat-template \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    export EVAL_CONCURRENT_REQUESTS="${EVAL_CONCURRENT_REQUESTS:-$CONC}"
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

# Stop GPU monitoring
stop_gpu_monitor
set +x
