#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for GPT-OSS 120B FP4 on H200 using vLLM.
#
# Required env vars:
#   MODEL, TP, USERS, RESULT_DIR

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP USERS OFFLOAD_MODE TOTAL_CPU_DRAM_GB RESULT_DIR

PORT=${PORT:-8888}
DURATION=${DURATION:-1800}
MAX_DELAY=${MAX_DELAY:-60}
ADVANCE_MIN=${ADVANCE_MIN:-0.0}
ADVANCE_MAX=${ADVANCE_MAX:-0.7}
# Agentic matrix entries don't set max-model-len, so the workflow passes 0.
# ${:-DEFAULT} only fires on unset/empty, so handle 0 explicitly.
if [ -z "${MAX_MODEL_LEN:-}" ] || [ "$MAX_MODEL_LEN" = "0" ]; then
    MAX_MODEL_LEN=131072
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi
nvidia-smi

# ---- Resolve traces and install deps ----------------------------------------
resolve_trace_source
install_agentic_deps

# ---- Server config ----------------------------------------------------------
SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

cat > "$RESULT_DIR/config.yaml" << EOF
async-scheduling: true
max-cudagraph-capture-size: 2048
max-model-len: $MAX_MODEL_LEN
EOF

OFFLOAD_ARGS=""
if [ "$OFFLOAD_MODE" = "on" ]; then
    export VLLM_USE_SIMPLE_KV_OFFLOAD=1
    OFFLOAD_ARGS="--kv_offloading_backend native --kv_offloading_size $TOTAL_CPU_DRAM_GB --no-disable-hybrid-kv-cache-manager"
fi

echo "Starting vllm server..."
export TORCH_CUDA_ARCH_LIST="9.0"
export PYTHONNOUSERSITE=1
export VLLM_MXFP4_USE_MARLIN=1

vllm serve $MODEL \
--host 0.0.0.0 \
--port $PORT \
--config "$RESULT_DIR/config.yaml" \
--gpu-memory-utilization 0.9 \
--tensor-parallel-size $TP \
--max-num-seqs $USERS \
$OFFLOAD_ARGS > "$SERVER_LOG" 2>&1 &
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

write_agentic_result_json "$RESULT_DIR"

# ---- Post-processing --------------------------------------------------------
python3 "$AGENTIC_DIR/scripts/analyze_benchmark_distributions.py" \
    "$RESULT_DIR/trace_replay" -o "$RESULT_DIR" 2>&1 || true

stop_agentic_metrics_collector
trim_idle_metrics "$RESULT_DIR"
