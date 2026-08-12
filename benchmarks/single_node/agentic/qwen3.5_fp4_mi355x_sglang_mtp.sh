#!/usr/bin/env bash
set -euo pipefail
set -x

# AgentX trace replay for Qwen3.5-397B-A17B MXFP4 on MI355X with SGLang
# native EAGLE MTP. Throughput uses the committed golden synthetic
# acceptance length; evaluation retains real target-model verification.

source "$(dirname "$0")/../../benchmark_lib.sh"

export EVAL_FRAMEWORK="lm-eval"

check_env_vars \
    MODEL TP CONC EP_SIZE KV_OFFLOADING \
    TOTAL_CPU_DRAM_GB RESULT_DIR DURATION

SCHEDULER_RECV_INTERVAL=${SCHEDULER_RECV_INTERVAL:-30}

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

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

export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126_256k
resolve_trace_source
install_agentic_deps

export AIPERF_SERVER_METRICS_URLS="http://localhost:${PORT}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="sglang:"

SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

SERVER_PID=""
cleanup_agentic_services() {
    local exit_code=$?
    trap - EXIT INT TERM
    set +e
    stop_background_process_tree "$SERVER_PID" "SGLang server" 60
    exit "$exit_code"
}
trap cleanup_agentic_services EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

CACHE_ARGS=()
WARMUP_ARGS=()
PAGE_SIZE=16
if require_agentic_kv_offload_backend hicache; then
    # Qwen3.5 creates target KV and Mamba host pools per rank. NEXTN adds a
    # one-layer draft KV pool, or 1/15 of target KV. Convert the node-level
    # workflow budget to SGLang's per-rank --hicache-size and retain the
    # established MI355X 180 GB per-pool ceiling.
    HICACHE_ALIGNMENT_RESERVE_GB=$TP
    HICACHE_USABLE_TOTAL_GB=$((TOTAL_CPU_DRAM_GB - HICACHE_ALIGNMENT_RESERVE_GB))
    if [ "$HICACHE_USABLE_TOTAL_GB" -lt 1 ]; then
        echo "Error: insufficient DRAM after HiCache alignment reserve." >&2
        exit 1
    fi
    MAX_HICACHE_SIZE_GB=$((HICACHE_USABLE_TOTAL_GB * 15 / TP / 31))
    HICACHE_MAX_SIZE_GB_PER_RANK_POOL=${HICACHE_MAX_SIZE_GB_PER_RANK_POOL:-180}
    if [ "$MAX_HICACHE_SIZE_GB" -gt "$HICACHE_MAX_SIZE_GB_PER_RANK_POOL" ]; then
        MAX_HICACHE_SIZE_GB="$HICACHE_MAX_SIZE_GB_PER_RANK_POOL"
    fi
    HICACHE_SIZE_GB="${HICACHE_SIZE_GB:-$MAX_HICACHE_SIZE_GB}"
    if [ "$HICACHE_SIZE_GB" -lt 1 ] || [ "$HICACHE_SIZE_GB" -gt "$MAX_HICACHE_SIZE_GB" ]; then
        echo "Error: HICACHE_SIZE_GB=$HICACHE_SIZE_GB outside 1..$MAX_HICACHE_SIZE_GB." >&2
        exit 1
    fi
    PROJECTED_HICACHE_TOTAL_GB=$(((HICACHE_SIZE_GB * TP * 31 + 14) / 15 + HICACHE_ALIGNMENT_RESERVE_GB))
    if [ "$PROJECTED_HICACHE_TOTAL_GB" -gt "$TOTAL_CPU_DRAM_GB" ]; then
        echo "Error: projected HiCache use ${PROJECTED_HICACHE_TOTAL_GB} GB exceeds configured ${TOTAL_CPU_DRAM_GB} GB." >&2
        exit 1
    fi
    echo "HiCache pools: ${HICACHE_SIZE_GB} GB per rank; projected node total ${PROJECTED_HICACHE_TOTAL_GB} GB."

    # Qwen3.5's hybrid attention/Mamba pools require the page-first kernel
    # transfer path on ROCm. Page size 64 also matches SGLang's MI355X EAGLE
    # recipe and avoids page-size-1 verify-graph compiler failures.
    PAGE_SIZE=64
    CACHE_ARGS=(
        --enable-hierarchical-cache
        --hicache-size "$HICACHE_SIZE_GB"
        --hicache-io-backend kernel
        --hicache-mem-layout page_first
        --hicache-write-policy write_through_selective
    )
    WARMUP_ARGS=(--skip-server-warmup)
fi

PARALLEL_ARGS=(
    --tp "$TP"
    --dp 1
    --ep-size "$EP_SIZE"
)

TOKENIZER_ARGS=()
if [ "$TP" -ge 4 ]; then
    TOKENIZER_ARGS=(--tokenizer-worker-num 6)
fi

MAX_RUNNING_REQUESTS=$((2 * CONC))
CUDA_GRAPH_MAX_BS="$CONC"
[ "$CUDA_GRAPH_MAX_BS" -gt 64 ] && CUDA_GRAPH_MAX_BS=64
if agentic_kv_offload_enabled && [ "$CUDA_GRAPH_MAX_BS" -gt 16 ]; then
    CUDA_GRAPH_MAX_BS=16
fi

export PYTHONNOUSERSITE=1
export SGLANG_USE_AITER=1
export SGLANG_USE_AITER_UNIFIED_ATTN=1
export AITER_FLYDSL_FORCE=1
export SGLANG_MAMBA_SSM_DTYPE=bfloat16
export SGLANG_TIMEOUT_KEEP_ALIVE=1800

if [ "${EVAL_ONLY:-false}" != "true" ]; then
    export SGLANG_SIMULATE_ACC_LEN=3.39
    export SGLANG_SIMULATE_ACC_METHOD=match-expected
    export SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
fi

SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    "${PARALLEL_ARGS[@]}"
    --attention-backend aiter
    --mem-fraction-static 0.80
    --model-loader-extra-config '{"enable_multithread_load": true}'
    --watchdog-timeout 1200
    --page-size "$PAGE_SIZE"
    --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS"
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --max-prefill-tokens 32768
    --chunked-prefill-size 32768
    --scheduler-recv-interval "$SCHEDULER_RECV_INTERVAL"
    --stream-interval 50
    "${TOKENIZER_ARGS[@]}"
    --tokenizer-path "$MODEL"
    --reasoning-parser qwen3
    --tool-call-parser qwen3_coder
    --speculative-algorithm EAGLE
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
    --enable-metrics
    --enable-cache-report
    "${CACHE_ARGS[@]}"
    "${WARMUP_ARGS[@]}"
)

printf '%q ' "${SGLANG_CMD[@]}" | tee "$RESULT_DIR/sglang_command.txt"
printf '\n' | tee -a "$RESULT_DIR/sglang_command.txt"
"${SGLANG_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [ "${EVAL_ONLY:-false}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    REPLAY_CMD+=" --apply-chat-template"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
