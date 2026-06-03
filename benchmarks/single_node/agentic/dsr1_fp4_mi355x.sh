#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for DSR1 FP4 on MI355X using SGLang.
#
# Required env vars:
#   MODEL, TP, CONC, RESULT_DIR

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC RESULT_DIR DURATION


if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi
rocm-smi
amd-smi || true

# ---- Resolve traces and install deps ----------------------------------------
resolve_trace_source
install_agentic_deps

# ---- Start SGLang server ----------------------------------------------------
SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

echo "Starting SGLang server..."
export SGLANG_USE_AITER=1
export ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export PYTHONNOUSERSITE=1

python3 -m sglang.launch_server \
--model-path=$MODEL \
--host=0.0.0.0 \
--port=$PORT \
--trust-remote-code \
--tensor-parallel-size=$TP \
--chunked-prefill-size=16384 \
--mem-fraction-static=0.8 \
--num-continuous-decode-steps=4 \
--cuda-graph-max-bs=$CONC \
--max-running-requests=$CONC \
--attention-backend aiter \
--kv-cache-dtype fp8_e4m3 \
--enable-metrics > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# ---- Run benchmark ----------------------------------------------------------
build_replay_cmd "$RESULT_DIR"

run_agentic_replay_and_write_outputs "$RESULT_DIR"
