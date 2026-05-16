#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for MiniMax-M2.5 FP8 on MI325X using vLLM.
# Supports LMCache CPU DRAM offloading for KV cache.
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
# Agentic matrix entries don't set max-model-len, so the workflow passes 0.
# ${:-DEFAULT} only fires on unset/empty, so handle 0 explicitly.
if [ -z "${MAX_MODEL_LEN:-}" ] || [ "$MAX_MODEL_LEN" = "0" ]; then
    MAX_MODEL_LEN=131072
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi
rocm-smi

# If the machine runs a MEC FW older than 177, RCCL cannot reclaim some memory.
# See https://rocm.docs.amd.com/en/docs-6.4.3/about/release-notes.html#amdgpu-driver-updates
version=`rocm-smi --showfw | grep MEC | head -n 1 | awk '{print $NF}'`
if [[ "$version" == "" || $version -lt 177 ]]; then
  export HSA_NO_SCRATCH_RECLAIM=1
fi

# Ray compatibility in vLLM 0.14+ needs HIP_VISIBLE_DEVICES to match ROCR_VISIBLE_DEVICES
if [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

export AMDGCN_USE_BUFFER_OPS=0
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export PYTHONNOUSERSITE=1

# ---- Resolve traces and install deps ----------------------------------------
resolve_trace_source
install_agentic_deps

# ---- Server config ----------------------------------------------------------
SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

OFFLOAD_ARGS=""
PREFIX_CACHE_FLAG="--no-enable-prefix-caching"

case "$OFFLOADING" in
    none)
        ;;
    cpu)
        OFFLOAD_ARGS="--kv_offloading_backend native --kv_offloading_size $TOTAL_CPU_DRAM_GB --disable-hybrid-kv-cache-manager"
        ;;
    lmcache_cpu)
        # LMCache CPU DRAM offloading via LMCacheConnectorV1.
        # Critical: PYTHONHASHSEED=0 is mandatory for cache key consistency
        # across TP workers. Without it, hit rate is 0%.
        install_lmcache_hip
        export PYTHONHASHSEED=0
        export LMCACHE_LOCAL_CPU=true
        export LMCACHE_CHUNK_SIZE=256
        export LMCACHE_MAX_LOCAL_CPU_SIZE=$((TOTAL_CPU_DRAM_GB / TP))
        # LMCache reuses vLLM's prefix cache hash function, so prefix caching
        # must be enabled (unlike native CPU offloading).
        PREFIX_CACHE_FLAG="--enable-prefix-caching"
        OFFLOAD_ARGS="--kv-transfer-config {\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_both\"}"
        ;;
    *)
        echo "Error: unsupported OFFLOADING value '$OFFLOADING' (expected one of: none, cpu, lmcache_cpu)" >&2
        exit 1
        ;;
esac

echo "Starting vllm server..."

vllm serve $MODEL \
--host 0.0.0.0 \
--port $PORT \
--trust-remote-code \
--tool-call-parser minimax_m2 \
--reasoning-parser minimax_m2 \
--enable-auto-tool-choice \
--attention-backend ROCM_AITER_UNIFIED_ATTN \
--tensor-parallel-size=$TP \
--gpu-memory-utilization 0.85 \
--max-model-len $MAX_MODEL_LEN \
--max-num-seqs $CONC \
--block-size=64 \
--kv-cache-dtype fp8 \
$PREFIX_CACHE_FLAG \
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
    "$RESULT_DIR/trace_replay" -o "$RESULT_DIR" 2>&1 || true
