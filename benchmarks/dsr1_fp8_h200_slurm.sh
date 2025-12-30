#!/usr/bin/env bash

# === Required Env Vars ===
# MODEL
# TP
# CONC
# ISL
# OSL
# RANDOM_RANGE_RATIO
# RESULT_FILENAME
# PORT_OFFSET

echo "JOB \$SLURM_JOB_ID running on \$SLURMD_NODENAME"

pip3 install --user sentencepiece
hf download $MODEL
PORT=$(( 8888 + $PORT_OFFSET ))
SERVER_LOG=$(mktemp /tmp/server-XXXXXX.log)

export TORCH_CUDA_ARCH_LIST="9.0"

set -x
if [[ $ISL -eq 1024 && $OSL -eq 1024 ]]; then
    PYTHONNOUSERSITE=1 python3 -m sglang.launch_server --model-path $MODEL --tokenizer-path $MODEL \
    --host 0.0.0.0 --port $PORT --trust-remote-code \
    --tensor-parallel-size=$TP --data-parallel-size=1 \
    --disable-radix-cache --max-running-requests 512 --cuda-graph-max-bs 512 \
    --chunked-prefill-size 32768 --max-prefill-tokens 32768 --mem-fraction-static 0.82 \
    --attention-backend flashinfer --stream-interval 10 \
    --decode-log-interval 1 \
    > $SERVER_LOG 2>&1 &
else
    PYTHONNOUSERSITE=1 python3 -m sglang.launch_server --model-path $MODEL --tokenizer-path $MODEL \
    --host 0.0.0.0 --port $PORT --trust-remote-code \
    --tensor-parallel-size=$TP --data-parallel-size=1 \
    --disable-radix-cache --max-running-requests 256 --cuda-graph-max-bs 256 \
    --chunked-prefill-size 32768 --max-prefill-tokens 32768 --mem-fraction-static 0.82 \
    --attention-backend flashinfer --stream-interval 10 \
    --decode-log-interval 1 \
    > $SERVER_LOG 2>&1 &
fi

SERVER_PID=$!

# Source benchmark utilities
source "$(dirname "$0")/benchmark_lib.sh"

# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# If profiling is enabled, start profiling via SGLang HTTP API
if [[ "${PROFILE:-}" == "1" ]]; then
    SGLANG_TORCH_PROFILER_DIR="${SGLANG_TORCH_PROFILER_DIR:-/workspace/profiles}"
    mkdir -p "$SGLANG_TORCH_PROFILER_DIR"
fi

run_benchmark_serving \
  --model "$MODEL" \
  --port "$PORT" \
  --backend vllm \
  --input-len "$ISL" \
  --output-len "$OSL" \
  --random-range-ratio "$RANDOM_RANGE_RATIO" \
  --num-prompts $((CONC * 2)) \
  --max-concurrency "$CONC" \
  --result-filename "$RESULT_FILENAME" \
  --result-dir /workspace/ \
  &
BENCH_PID=$!

if [[ "${PROFILE:-}" == "1" ]]; then
  echo "[PROFILE] Will start mid-run; dir=$SGLANG_TORCH_PROFILER_DIR"

  # Wait until the run has ramped up (tune this)
  #sleep "${PROFILE_DELAY_SECS:-60}"

  # Start a SMALL bounded capture (this auto-stops; do NOT call stop_profile)
  curl -sf -X POST "http://127.0.0.1:$PORT/start_profile" \
    -H "Content-Type: application/json" \
    -d "{
      \"output_dir\": \"$SGLANG_TORCH_PROFILER_DIR\",
      \"num_steps\": 5,
      \"start_step\": 0,
      \"activities\": [\"GPU\", \"CPU\"],
      \"merge_profiles\": true,
      \"profile_by_stage\": true,
      \"record_shapes\": true
    }" || true
fi

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
      PYTHONNOUSERSITE=1 python3 utils/mfu_trace_analyzer.py "$MERGED_TRACE" "$MERGED_TRACE" --gpu H200 --tp $TP --decode-batch-size $CONC || echo "[PROFILE] MFU analyzer failed; continuing without modification"
    fi
    echo "[PROFILE] Found trace: $TRACE_FILE -> $DEST_TRACE"
    cp "$TRACE_FILE" "$DEST_TRACE"
  else
    echo "[PROFILE] No trace found under $SGLANG_TORCH_PROFILER_DIR" >&2
  fi
fi
