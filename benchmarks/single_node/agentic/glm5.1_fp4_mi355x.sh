#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for GLM-5.1 FP4 on MI355X using SGLang.
#
# Required env vars:
#   MODEL, TP, CONC, RESULT_DIR

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC RESULT_DIR DURATION

if [ -z "${MAX_MODEL_LEN:-}" ] || [ "$MAX_MODEL_LEN" = "0" ]; then
    MAX_MODEL_LEN=131072
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi
rocm-smi || true
amd-smi || true

# ---- Resolve traces and install deps ----------------------------------------
resolve_trace_source
install_agentic_deps

# ROCm / SGLang performance tuning for MI355X
export SGLANG_ROCM_FUSED_DECODE_MLA=0
export ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export SAFETENSORS_FAST_GPU=1

# ---- Start SGLang server ----------------------------------------------------
SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

pip install -U transformers

echo "Starting SGLang server..."
export PYTHONNOUSERSITE=1

python3 -m sglang.launch_server \
    --model-path $MODEL \
    --host=0.0.0.0 \
    --port $PORT \
    --tensor-parallel-size $TP \
    --trust-remote-code \
    --cuda-graph-max-bs $CONC \
    --max-running-requests $CONC \
    --context-length $MAX_MODEL_LEN \
    --mem-fraction-static 0.85 \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}' \
    --nsa-prefill-backend tilelang \
    --nsa-decode-backend tilelang \
    --kv-cache-dtype fp8_e4m3 \
    --tokenizer-worker-num $((TP*2)) \
    --enable-metrics > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# ---- Run benchmark ----------------------------------------------------------
build_replay_cmd "$RESULT_DIR"

run_agentic_replay_and_write_outputs "$RESULT_DIR"
