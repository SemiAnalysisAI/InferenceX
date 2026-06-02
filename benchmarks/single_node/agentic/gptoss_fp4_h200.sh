#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for GPT-OSS 120B FP4 on H200 using vLLM.
#
# Required env vars:
#   MODEL, TP, CONC, RESULT_DIR

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION

# Agentic matrix entries don't set max-model-len, so the workflow passes 0.
# ${:-DEFAULT} only fires on unset/empty, so handle 0 explicitly.
if [ -z "${MAX_MODEL_LEN:-}" ] || [ "$MAX_MODEL_LEN" = "0" ]; then
    MAX_MODEL_LEN=131072
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

# `hf download` creates the target dir if missing and is itself idempotent.
# When MODEL_PATH is unset (stand-alone runs), fall back to the HF_HUB_CACHE
# Either way, MODEL_PATH is what the server is launched with.
if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi
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
case "$OFFLOADING" in
    none)
        ;;
    cpu)
        export VLLM_USE_SIMPLE_KV_OFFLOAD=1
        OFFLOAD_ARGS="--kv_offloading_backend native --kv_offloading_size $TOTAL_CPU_DRAM_GB --disable-hybrid-kv-cache-manager"
        ;;
    *)
        echo "Error: unsupported OFFLOADING value '$OFFLOADING' (expected one of: none, cpu)" >&2
        exit 1
        ;;
esac

echo "Starting vllm server..."
export TORCH_CUDA_ARCH_LIST="9.0"
export PYTHONNOUSERSITE=1
export VLLM_MXFP4_USE_MARLIN=1

vllm serve "$MODEL_PATH" --served-model-name "$MODEL" \
--host 0.0.0.0 \
--port $PORT \
--config "$RESULT_DIR/config.yaml" \
--gpu-memory-utilization 0.9 \
--tensor-parallel-size $TP \
--max-num-seqs $CONC \
$OFFLOAD_ARGS > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# ---- Run benchmark ----------------------------------------------------------
build_replay_cmd "$RESULT_DIR"

run_agentic_replay_and_write_outputs "$RESULT_DIR"
