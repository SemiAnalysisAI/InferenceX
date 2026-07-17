#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for Qwen3.5 FP8 on MI355X using SGLang.
#
# Required env vars:
#   MODEL, TP, CONC, KV_OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR
#
# KV_OFFLOADING=dram requires KV_OFFLOAD_BACKEND=hicache.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE

SCHEDULER_RECV_INTERVAL=${SCHEDULER_RECV_INTERVAL:-30}

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
if require_agentic_kv_offload_backend hicache; then
    # Qwen3.5 allocates one KV and one Mamba host pool per TP rank.
    REQUESTED_HICACHE_TOTAL_GB="${HICACHE_TOTAL_CPU_DRAM_GB:-$TOTAL_CPU_DRAM_GB}"
    if [ "$REQUESTED_HICACHE_TOTAL_GB" -gt "$TOTAL_CPU_DRAM_GB" ]; then
        echo "Error: requested HiCache pool ${REQUESTED_HICACHE_TOTAL_GB} GB exceeds configured capacity ${TOTAL_CPU_DRAM_GB} GB" >&2
        exit 1
    fi
    TOTAL_CPU_DRAM_GB="$REQUESTED_HICACHE_TOTAL_GB"
    HICACHE_HOST_POOL_COUNT="${HICACHE_HOST_POOL_COUNT:-2}"
    HICACHE_MAX_SIZE_GB_PER_RANK_POOL="${HICACHE_MAX_SIZE_GB_PER_RANK_POOL:-${HICACHE_MAX_SIZE_GB_PER_RANK:-180}}"
    HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_through_selective}"
    # Qwen3.5's hybrid Mamba path runs SGLang's no_buffer scheduler on MI355X,
    # which requires page_size=1. The kernel/page_first HiCache transfer path
    # faults on first prefill in this mode on ROCm, so keep the default on the
    # safer direct/layer_first copy path. These remain env-overridable.
    HICACHE_PAGE_SIZE="${HICACHE_PAGE_SIZE:-1}"
    HICACHE_IO_BACKEND="${HICACHE_IO_BACKEND:-direct}"
    HICACHE_MEM_LAYOUT="${HICACHE_MEM_LAYOUT:-layer_first}"
    # SGLang --hicache-size is per rank per host pool, while the workflow
    # input is a node-total DRAM budget. Divide by TP and the number of
    # host pools unless HICACHE_SIZE_GB is set directly for one-off tuning.
    MAX_HICACHE_SIZE_GB=$((TOTAL_CPU_DRAM_GB / TP / HICACHE_HOST_POOL_COUNT))
    HICACHE_SIZE_GB="${HICACHE_SIZE_GB:-$MAX_HICACHE_SIZE_GB}"
    if [ "$HICACHE_SIZE_GB" -gt "$MAX_HICACHE_SIZE_GB" ]; then
        echo "Error: HICACHE_SIZE_GB=$HICACHE_SIZE_GB exceeds configured per-pool limit $MAX_HICACHE_SIZE_GB" >&2
        exit 1
    fi
    if [ "$HICACHE_SIZE_GB" -gt "$HICACHE_MAX_SIZE_GB_PER_RANK_POOL" ]; then
        HICACHE_SIZE_GB="$HICACHE_MAX_SIZE_GB_PER_RANK_POOL"
    fi
    if [ "$HICACHE_SIZE_GB" -lt 1 ]; then
        echo "Error: computed HICACHE_SIZE_GB=$HICACHE_SIZE_GB from TOTAL_CPU_DRAM_GB=$TOTAL_CPU_DRAM_GB, TP=$TP, HICACHE_HOST_POOL_COUNT=$HICACHE_HOST_POOL_COUNT" >&2
        exit 1
    fi
    echo "HiCache CPU pool: ${HICACHE_SIZE_GB} GB per rank per host pool across TP=${TP}, host_pool_count=${HICACHE_HOST_POOL_COUNT}"
    CACHE_ARGS=(
        --page-size "$HICACHE_PAGE_SIZE"
        --enable-hierarchical-cache
        --hicache-size "$HICACHE_SIZE_GB"
        --hicache-io-backend "$HICACHE_IO_BACKEND"
        --hicache-mem-layout "$HICACHE_MEM_LAYOUT"
        --hicache-write-policy "$HICACHE_WRITE_POLICY"
    )
    # HiCache startup reaches API readiness, but SGLang's internal warmup
    # request has timed out after 600s on this Qwen MI355X path. Let aiperf
    # own benchmark traffic instead of blocking server readiness on it.
    WARMUP_ARGS=(--skip-server-warmup)
    # Keep request concurrency as the swept variable, but do not force HiCache
    # runs to capture ROCm graphs at every high concurrency point.
    HICACHE_CUDA_GRAPH_MAX_BS="${HICACHE_CUDA_GRAPH_MAX_BS:-16}"
    if [ "$HICACHE_CUDA_GRAPH_MAX_BS" -lt "$CUDA_GRAPH_MAX_BS" ]; then
        CUDA_GRAPH_MAX_BS="$HICACHE_CUDA_GRAPH_MAX_BS"
    fi
fi

echo "Starting SGLang server..."
export PYTHONNOUSERSITE=1

{ set +x; } 2>/dev/null
SGLANG_CMD=(
    python3 -m sglang.launch_server
    --attention-backend triton
    --model-path "$MODEL_PATH" --served-model-name "$MODEL"
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

if [ "${EVAL_ONLY}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
