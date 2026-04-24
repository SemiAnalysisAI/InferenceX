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

hf download "$MODEL"

nvidia-smi

# The deepseek-v4-blackwell image bakes CUDA_VISIBLE_DEVICES=4,5,6,7 into its ENV,
# which masks half of the 8 GPUs Slurm allocates us. Clear it so TP=8 can bind to
# all ranks.
unset CUDA_VISIBLE_DEVICES

SERVER_LOG="$PWD/server.log"
PORT=${PORT:-30000}

echo "TP: $TP, CONC: $CONC, ISL: $ISL, OSL: $OSL"

start_gpu_monitor --output "$PWD/gpu_metrics.csv"

set -x
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256 \
SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 \
sglang serve \
  --trust-remote-code \
  --model-path $MODEL \
  --tp 8 \
  --dp 8 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --speculative-algo EAGLE \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 2 \
  --mem-fraction-static 0.82 \
  --cuda-graph-max-bs 64 \
  --max-running-requests 128 \
  --deepep-config '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}' \
  --host 0.0.0.0 \
  --port $PORT > $SERVER_LOG 2>&1 &

SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

pip install -q datasets pandas

run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend sglang \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts $((CONC * 10)) \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir "$PWD/"

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

stop_gpu_monitor
set +x
