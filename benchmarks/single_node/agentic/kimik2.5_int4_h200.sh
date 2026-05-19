#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for Kimi-K2.5 INT4 on H200 using vLLM.
#
# Required env vars:
#   MODEL, TP, CONC, OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR

PORT=${PORT:-8888}
DURATION=${DURATION:-1800}
MAX_DELAY=${MAX_DELAY:-60}
ADVANCE_MIN=${ADVANCE_MIN:-0.0}
ADVANCE_MAX=${ADVANCE_MAX:-0.7}

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

OFFLOAD_ARGS=""
case "$OFFLOADING" in
    none) ;;
    cpu)
        # H200 nodes — reserve up to 1 TB for the simple CPU offload
        # connector. Conservative because Hopper-class clusters typically
        # have smaller host RAM envelopes than the Blackwell side.
        TOTAL_CPU_DRAM_GB=1000
        # Kimi K2.5 is pure TP (no DP-attn): single engine, world_size=TP.
        # SimpleCPUOffloadConnector internally divides cpu_bytes_to_use by
        # world_size, so pass the full TOTAL_CPU_DRAM_GB; TP-shared mmap
        # keeps the aggregate at TOTAL.
        PER_ENGINE_BYTES=$((TOTAL_CPU_DRAM_GB * 1024 * 1024 * 1024))
        # JSON form (rather than --kv_offloading_backend native shortcut) so
        # we can pass lazy_offload=true. Eager mode (the shortcut default)
        # hits a popleft_n AssertionError in
        # vllm/v1/core/kv_cache_utils.py at low/mid CONC on DSv4 + SimpleCPUOffloadConnector;
        # lazy defers the store path and clears low/mid CONC reliably. See
        # SimpleCPUOffloadConnector PR #37160.
        export VLLM_USE_SIMPLE_KV_OFFLOAD=1
        OFFLOAD_ARGS="--kv-transfer-config {\"kv_connector\":\"SimpleCPUOffloadConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":$PER_ENGINE_BYTES,\"lazy_offload\":true}}"
        ;;
    *) echo "Error: unsupported OFFLOADING value '$OFFLOADING'" >&2; exit 1 ;;
esac

echo "Starting vllm server..."
export PYTHONNOUSERSITE=1
export VLLM_USE_FLASHINFER_MOE_INT4=1

vllm serve $MODEL \
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

echo "$REPLAY_CMD" > "$RESULT_DIR/benchmark_command.txt"

set -x
$REPLAY_CMD 2>&1 | tee "$RESULT_DIR/benchmark.log" || true
set +x

write_agentic_result_json "$RESULT_DIR"

# ---- Post-processing --------------------------------------------------------
python3 "$AGENTIC_DIR/scripts/analyze_benchmark_distributions.py" \
    "$RESULT_DIR/aiperf_artifacts" -o "$RESULT_DIR" 2>&1 || true
