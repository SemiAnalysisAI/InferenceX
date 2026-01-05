#!/usr/bin/env bash

# === Required Env Vars ===
# MODEL
# PORT
# TP
# CONC
# ISL
# OSL
# RANDOM_RANGE_RATIO
# RESULT_FILENAME
# EP_SIZE
# NUM_PROMPTS

nvidia-smi

# To improve CI stability, we patch this helper function to prevent a race condition that
# happens 1% of the time. ref: https://github.com/flashinfer-ai/flashinfer/pull/1779
sed -i '102,108d' /usr/local/lib/python3.12/dist-packages/flashinfer/jit/cubin_loader.py

export SGL_ENABLE_JIT_DEEPGEMM=false
export SGLANG_ENABLE_FLASHINFER_GEMM=true
SERVER_LOG=$(mktemp /tmp/server-XXXXXX.log)

# Default: recv every ~10 requests; if CONC ≥ 16, relax to ~30 requests between scheduler recv polls.
if [[ $CONC -ge 16 ]]; then
  SCHEDULER_RECV_INTERVAL=30
else
  SCHEDULER_RECV_INTERVAL=10
fi
echo "SCHEDULER_RECV_INTERVAL: $SCHEDULER_RECV_INTERVAL, CONC: $CONC, ISL: $ISL, OSL: $OSL"

ps aux

set -x
PYTHONNOUSERSITE=1 python3 -m sglang.launch_server --model-path=$MODEL --host=0.0.0.0 --port=$PORT \
--tensor-parallel-size=$TP --data-parallel-size=1 \
--cuda-graph-max-bs 128 --max-running-requests 128 \
--mem-fraction-static 0.82 --kv-cache-dtype fp8_e4m3 --chunked-prefill-size 32768 --max-prefill-tokens 32768 \
--enable-flashinfer-allreduce-fusion --scheduler-recv-interval $SCHEDULER_RECV_INTERVAL --disable-radix-cache \
--attention-backend trtllm_mla --stream-interval 30 --ep-size $EP_SIZE --moe-runner-backend flashinfer_trtllm --quantization fp8 > $SERVER_LOG 2>&1 &

SERVER_PID=$!

# Source benchmark utilities
source "$(dirname "$0")/benchmark_lib.sh"

# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

pip install -q datasets pandas

if [[ "${PROFILE:-}" == "1" ]]; then
  SGLANG_TORCH_PROFILER_DIR="${SGLANG_TORCH_PROFILER_DIR:-/workspace}"
  mkdir -p "$SGLANG_TORCH_PROFILER_DIR"
  echo "[PROFILE] Using benchmark_serving managed profiling (--profile); dir=$SGLANG_TORCH_PROFILER_DIR"
fi

run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts "$NUM_PROMPTS" \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/ \
    &
BENCH_PID=$!

wait "$BENCH_PID"

ls -lt "$SGLANG_TORCH_PROFILER_DIR"

if [[ "${PROFILE:-}" == "1" ]]; then
  # Wait briefly for the file to appear (auto-stop writes it)
  TRACE_FILE=""
  for _ in {1..180}; do
    TRACE_FILE=$(ls -t "$SGLANG_TORCH_PROFILER_DIR"/*.trace.json* 2>/dev/null | head -n1)
    [[ -n "$TRACE_FILE" ]] && break
    sleep 1
  done

  if [[ -n "$TRACE_FILE" ]]; then
    DEST_TRACE="/workspace/profile_${RESULT_FILENAME}.trace.json.gz"
    # If a merged profile exists, run MFU analyzer on it before copying
    MERGED_TRACE=$(ls -t "$SGLANG_TORCH_PROFILER_DIR"/merged-*.trace.json* 2>/dev/null | head -n1)
    if [[ -n "$MERGED_TRACE" ]]; then
      echo "[PROFILE] Running MFU analyzer on merged trace: $MERGED_TRACE"
      PYTHONNOUSERSITE=1 python3 utils/mfu_trace_analyzer.py "$MERGED_TRACE" "$MERGED_TRACE" --gpu B200 --tp $TP --decode-batch-size $CONC || echo "[PROFILE] MFU analyzer failed; continuing without modification"
    fi
    echo "[PROFILE] Found trace: $TRACE_FILE -> $DEST_TRACE"
    cp "$TRACE_FILE" "$DEST_TRACE"
  else
    echo "[PROFILE] No trace found under $SGLANG_TORCH_PROFILER_DIR" >&2
  fi
fi
