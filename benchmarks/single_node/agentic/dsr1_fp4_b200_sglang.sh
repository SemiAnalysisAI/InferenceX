#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for DSR1 FP4 on B200 using SGLang.
#
# Required env vars:
#   MODEL, TP, USERS, RESULT_DIR

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP USERS RESULT_DIR

PORT=${PORT:-8888}
DURATION=${DURATION:-1800}
MAX_DELAY=${MAX_DELAY:-60}
ADVANCE_MIN=${ADVANCE_MIN:-0.0}
ADVANCE_MAX=${ADVANCE_MAX:-0.7}
EP_SIZE=${EP_SIZE:-1}
SCHEDULER_RECV_INTERVAL=${SCHEDULER_RECV_INTERVAL:-5}

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi
nvidia-smi

# ---- Resolve traces and install deps ----------------------------------------
resolve_trace_source
install_agentic_deps

# ---- Start SGLang server ----------------------------------------------------
SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

echo "Starting SGLang server..."
export TORCH_CUDA_ARCH_LIST="10.0"
export PYTHONNOUSERSITE=1

python3 -m sglang.launch_server \
--model-path $MODEL \
--host 0.0.0.0 \
--port $PORT \
--trust-remote-code \
--tensor-parallel-size=$TP \
--data-parallel-size=1 \
--cuda-graph-max-bs $USERS \
--max-running-requests $USERS \
--mem-fraction-static 0.85 \
--kv-cache-dtype fp8_e4m3 \
--chunked-prefill-size 16384 \
--ep-size $EP_SIZE \
--quantization modelopt_fp4 \
--enable-flashinfer-allreduce-fusion \
--scheduler-recv-interval $SCHEDULER_RECV_INTERVAL \
--enable-symm-mem \
--attention-backend trtllm_mla \
--moe-runner-backend flashinfer_trtllm \
--stream-interval 10 > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# ---- Run benchmark ----------------------------------------------------------
start_agentic_metrics_collector "$RESULT_DIR"
build_replay_cmd "$RESULT_DIR"

echo "$REPLAY_CMD" > "$RESULT_DIR/benchmark_command.txt"

set -x
$REPLAY_CMD 2>&1 | tee "$RESULT_DIR/benchmark.log" || true
set +x

check_agentic_success "$RESULT_DIR"

# ---- Post-processing --------------------------------------------------------
python3 "$AGENTIC_DIR/scripts/analyze_benchmark_distributions.py" \
    "$RESULT_DIR/trace_replay" -o "$RESULT_DIR" 2>&1 || true

stop_agentic_metrics_collector
trim_idle_metrics "$RESULT_DIR"

# ---- Cleanup ----------------------------------------------------------------
echo "Stopping server..."
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true

echo "Experiment finished at $(date)"
