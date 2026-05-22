#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for Qwen3.5 FP8 on MI355X using SGLang.
#
# Required env vars:
#   MODEL, TP, CONC, OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR
#
# OFFLOADING values:
#   none    - SGLang GPU KV only with radix cache disabled.
#   hicache - SGLang HiCache with local CPU hierarchical cache.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR

PORT=${PORT:-8888}
DURATION=${DURATION:-1800}
MAX_DELAY=${MAX_DELAY:-60}
ADVANCE_MIN=${ADVANCE_MIN:-0.0}
ADVANCE_MAX=${ADVANCE_MAX:-0.7}
EP_SIZE=${EP_SIZE:-1}
SCHEDULER_RECV_INTERVAL=${SCHEDULER_RECV_INTERVAL:-30}
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

# ---- Server config ----------------------------------------------------------
SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

CACHE_ARGS=()
WARMUP_ARGS=()
CUDA_GRAPH_MAX_BS="$CONC"
case "$OFFLOADING" in
    none)
        CACHE_ARGS=(--disable-radix-cache)
        ;;
    hicache)
        # HiCache extends RadixAttention, so do not pass --disable-radix-cache.
        # MI355X nodes have about 3 TB of host DRAM, but Qwen3.5's hybrid
        # GDN/Mamba path allocates two HiCache host pools per TP rank: one for
        # hierarchical KV cache and one for hierarchical Mamba cache. A 2 TB
        # node-total target at TP=8 is therefore 2000 / (8 * 2) = 125 GB per
        # host pool, not 250 GB. Keep overrides for one-off tuning.
        TOTAL_CPU_DRAM_GB="${HICACHE_TOTAL_CPU_DRAM_GB:-2000}"
        HICACHE_HOST_POOL_COUNT="${HICACHE_HOST_POOL_COUNT:-2}"
        HICACHE_MAX_SIZE_GB_PER_RANK_POOL="${HICACHE_MAX_SIZE_GB_PER_RANK_POOL:-${HICACHE_MAX_SIZE_GB_PER_RANK:-180}}"
        HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_through_selective}"
        # SGLang --hicache-size is per rank per host pool, while the workflow
        # input is a node-total DRAM budget. Divide by TP and the number of
        # host pools unless HICACHE_SIZE_GB is set directly for one-off tuning.
        HICACHE_SIZE_GB="${HICACHE_SIZE_GB:-$((TOTAL_CPU_DRAM_GB / TP / HICACHE_HOST_POOL_COUNT))}"
        if [ "$HICACHE_SIZE_GB" -gt "$HICACHE_MAX_SIZE_GB_PER_RANK_POOL" ]; then
            HICACHE_SIZE_GB="$HICACHE_MAX_SIZE_GB_PER_RANK_POOL"
        fi
        if [ "$HICACHE_SIZE_GB" -lt 1 ]; then
            echo "Error: computed HICACHE_SIZE_GB=$HICACHE_SIZE_GB from TOTAL_CPU_DRAM_GB=$TOTAL_CPU_DRAM_GB, TP=$TP, HICACHE_HOST_POOL_COUNT=$HICACHE_HOST_POOL_COUNT" >&2
            exit 1
        fi
        echo "HiCache CPU pool: ${HICACHE_SIZE_GB} GB per rank per host pool across TP=${TP}, host_pool_count=${HICACHE_HOST_POOL_COUNT}"
        CACHE_ARGS=(
            --page-size 64
            --enable-hierarchical-cache
            --hicache-size "$HICACHE_SIZE_GB"
            --hicache-io-backend kernel
            --hicache-mem-layout page_first
            --hicache-write-policy "$HICACHE_WRITE_POLICY"
        )
        # HiCache startup reaches API readiness, but SGLang's internal warmup
        # request has timed out after 600s on this Qwen MI355X path. Let aiperf
        # own benchmark traffic instead of blocking server readiness on it.
        WARMUP_ARGS=(--skip-server-warmup)
        # Keep request concurrency as the swept variable, but do not force
        # HiCache runs to capture ROCm graphs at every high concurrency point.
        # The conc=32 HiCache job crashed after startup readiness, before any
        # aiperf traffic, while conc=16 is the highest known-good capture size
        # for this model/server path. Requests above the capture size can still
        # run; they just do not require a larger captured graph at startup.
        HICACHE_CUDA_GRAPH_MAX_BS="${HICACHE_CUDA_GRAPH_MAX_BS:-16}"
        if [ "$HICACHE_CUDA_GRAPH_MAX_BS" -lt "$CUDA_GRAPH_MAX_BS" ]; then
            CUDA_GRAPH_MAX_BS="$HICACHE_CUDA_GRAPH_MAX_BS"
        fi
        ;;
    *)
        echo "Error: unsupported OFFLOADING value '$OFFLOADING' (expected one of: none, hicache)" >&2
        exit 1
        ;;
esac

echo "Starting SGLang server..."
export PYTHONNOUSERSITE=1

{ set +x; } 2>/dev/null
SGLANG_CMD=(
    python3 -m sglang.launch_server
    --attention-backend triton
    --model-path "$MODEL"
    --host=0.0.0.0
    --port "$PORT"
    --tensor-parallel-size "$TP"
    --ep-size "$EP_SIZE"
    --trust-remote-code
    --tokenizer-worker-num 6
    --enable-aiter-allreduce-fusion
    --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS"
    --max-running-requests "$CONC"
    --max-prefill-tokens 32768
    --scheduler-recv-interval "$SCHEDULER_RECV_INTERVAL"
    --mem-fraction-static 0.8
    --context-length "$MAX_MODEL_LEN"
    --enable-metrics
    "${CACHE_ARGS[@]}"
    "${WARMUP_ARGS[@]}"
)
printf '%q ' "${SGLANG_CMD[@]}" | tee "$RESULT_DIR/sglang_command.txt"
printf '\n' | tee -a "$RESULT_DIR/sglang_command.txt"
"${SGLANG_CMD[@]}" > "$SERVER_LOG" 2>&1 &
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
