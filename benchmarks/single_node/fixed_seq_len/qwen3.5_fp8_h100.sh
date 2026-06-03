#!/usr/bin/env bash

# Qwen-3.5-397B-A17B FP8 on H100 via sglang.
# Mirrors qwen3.5_fp8_h200.sh but with tighter memory accommodations:
# H100 has 80GB HBM3 vs H200's 141GB HBM3e, so weights + KV cache fit
# more snugly. Mem-fraction-static lowered from 0.8 → 0.75 and
# chunked-prefill-size from 16384 → 8192 to leave more headroom.
# Sweep tops out at conc=32 instead of 64 for the same reason.

source "$(dirname "$0")/../../benchmark_lib.sh"

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

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi

SERVER_LOG=/workspace/server.log
MAX_SEQ_LEN=$((ISL + OSL + 20))
if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_SEQ_LEN="$EVAL_MAX_MODEL_LEN"
fi

echo "CONC: $CONC, ISL: $ISL, OSL: $OSL, MAX_SEQ_LEN: $MAX_SEQ_LEN"

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
  --max-running-requests 64 \
  --chunked-prefill-size 8192 \
  --decode-log-interval 1 \
  --mem-fraction-static 0.75 \
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
