#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for Kimi-K2.5 NVFP4 on B300 using vLLM.
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
        # B300 NV nodes: RealMemory ~3.02 TiB; Slurm AllocMem cgroup caps each
        # job at ~2.82 TiB (2965 GB). At 3 TB the offload pool + worker RSS
        # exceeded the cgroup and workers OOM-killed silently (R2 saw 10/10
        # cpu jobs die ~4 min into init). 2.5 TB leaves ~465 GB headroom
        # inside the cgroup for vLLM worker RSS + page cache.
        TOTAL_CPU_DRAM_GB=2500
        export VLLM_USE_SIMPLE_KV_OFFLOAD=1
        OFFLOAD_ARGS="--kv_offloading_backend native --kv_offloading_size $TOTAL_CPU_DRAM_GB --disable-hybrid-kv-cache-manager"
        ;;
    *) echo "Error: unsupported OFFLOADING value '$OFFLOADING'" >&2; exit 1 ;;
esac

echo "Starting vllm server..."
export PYTHONNOUSERSITE=1

vllm serve $MODEL \
--host 0.0.0.0 \
--port $PORT \
--tensor-parallel-size=$TP \
--gpu-memory-utilization 0.90 \
--max-num-seqs $CONC \
--reasoning-parser kimi_k2 \
--tool-call-parser kimi_k2 \
--compilation_config.pass_config.fuse_allreduce_rms true \
--kv-cache-dtype fp8 \
--max-cudagraph-capture-size 2048 \
--stream-interval 20 \
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
