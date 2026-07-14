#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for Kimi-K2.5 INT4 on H200 using vLLM.
#
# Required env vars:
#   MODEL, TP, CONC, KV_OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION


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

OFFLOAD_ARGS=""
if require_agentic_kv_offload_backend vllm-simple; then
    # Kimi K2.5 is pure TP (no DP-attn): single engine, world_size=TP.
    # SimpleCPUOffloadConnector internally divides cpu_bytes_to_use by
    # world_size, so pass the full TOTAL_CPU_DRAM_GB; TP-shared mmap
    # keeps the aggregate at TOTAL.
    PER_ENGINE_BYTES=$((TOTAL_CPU_DRAM_GB * 1024 * 1024 * 1024))
    # JSON form (rather than --kv_offloading_backend native shortcut) so we can
    # pass lazy_offload=true. Eager mode hits a popleft_n AssertionError at
    # low/mid CONC on DSv4 + SimpleCPUOffloadConnector.
    export VLLM_USE_SIMPLE_KV_OFFLOAD=1
    OFFLOAD_ARGS="--kv-transfer-config {\"kv_connector\":\"SimpleCPUOffloadConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":$PER_ENGINE_BYTES,\"lazy_offload\":true}}"
fi

echo "Starting vllm server..."
export PYTHONNOUSERSITE=1
export VLLM_USE_FLASHINFER_MOE_INT4=1

vllm serve "$MODEL_PATH" --served-model-name "$MODEL" \
--host 0.0.0.0 \
--port $PORT \
--gpu-memory-utilization 0.95 \
--tensor-parallel-size $TP \
--max-num-seqs $CONC \
--reasoning-parser kimi_k2 \
--tool-call-parser kimi_k2 \
--compilation_config.pass_config.fuse_allreduce_rms true \
--trust-remote-code \
$OFFLOAD_ARGS > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# ---- Run benchmark ----------------------------------------------------------
build_replay_cmd "$RESULT_DIR"

run_agentic_replay_and_write_outputs "$RESULT_DIR"
