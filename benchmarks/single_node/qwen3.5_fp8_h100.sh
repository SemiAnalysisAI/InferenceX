#!/usr/bin/env bash

# Qwen-3.5-397B-A17B FP8 on H100 via sglang.
# Uses TP8/EP1 at conc 1-8, TP8/EP8 at conc 16-64,
# and TP8/EP8 with DP attention at conc 128-256.

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    DP_ATTENTION \
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
PORT=${PORT:-8888}
MAX_SEQ_LEN=$((ISL + OSL + 20))
if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_SEQ_LEN="$EVAL_MAX_MODEL_LEN"
fi

PARALLEL_ARGS=(--tp "$TP")
if [ "${EP_SIZE}" -gt 1 ]; then
    PARALLEL_ARGS+=(--expert-parallel-size "$EP_SIZE")
fi

SCHEDULER_RECV_INTERVAL=
if [ "${DP_ATTENTION}" != "true" ]; then
    case "$CONC" in
      1|2|4)
        SCHEDULER_RECV_INTERVAL=2
        ;;
      8)
        SCHEDULER_RECV_INTERVAL=60
        ;;
      16)
        SCHEDULER_RECV_INTERVAL=30
        ;;
      32)
        SCHEDULER_RECV_INTERVAL=1200
        ;;
      64)
        SCHEDULER_RECV_INTERVAL=600
        ;;
      *)
        echo "Unsupported CONC=$CONC for qwen3.5 FP8 H100 SGLang recipe" >&2
        exit 1
        ;;
    esac
fi

SCHEDULER_ARGS=()
if [ -n "$SCHEDULER_RECV_INTERVAL" ]; then
    SCHEDULER_ARGS=(--scheduler-recv-interval "$SCHEDULER_RECV_INTERVAL")
fi
if [ "${DP_ATTENTION}" = "true" ]; then
    PARALLEL_ARGS+=(--dp-size "$TP" --enable-dp-attention)
fi

echo "TP: $TP, EP_SIZE: $EP_SIZE, DP_ATTENTION: $DP_ATTENTION, CONC: $CONC, ISL: $ISL, OSL: $OSL, MAX_SEQ_LEN: $MAX_SEQ_LEN"
echo "SCHEDULER_RECV_INTERVAL: ${SCHEDULER_RECV_INTERVAL:-none}"
echo "SCHEDULER_ARGS: ${SCHEDULER_ARGS[*]}"

start_gpu_monitor

set -x
python3 -m sglang.launch_server \
  --model "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  "${PARALLEL_ARGS[@]}" \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --enable-flashinfer-allreduce-fusion \
  --max-running-requests 256 \
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
  --enable-symm-mem \
  --trust-remote-code \
  "${SCHEDULER_ARGS[@]}" \
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
