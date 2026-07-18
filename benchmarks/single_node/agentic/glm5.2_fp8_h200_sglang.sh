#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for GLM-5.2 FP8 on one 8xH200 node using
# aggregate SGLang serving. This is intentionally STP: no speculative decoding
# flags are passed even though the checkpoint includes an MTP head.
#
# The engine uses the upstream H200 cookbook's TP8 low-latency topology minus
# its EAGLE flags, swept HBM-only and with a local-DRAM HiCache tier. Keeping
# attention tensor-parallel is required for full-context AgentX replay. Match
# the established DeepSeek-V4 AgentX pattern by using FP8 KV on H200; this
# keeps every unfiltered trace intact while the model retains its native
# 1,048,576-token context window.
#
# Required env vars:
#   MODEL, TP, CONC, KV_OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR, DURATION,
#   EP_SIZE, DP_ATTENTION, SPEC_DECODING

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE DP_ATTENTION SPEC_DECODING

if [[ "$SPEC_DECODING" != "none" ]]; then
    echo "Error: this is an STP recipe; expected SPEC_DECODING=none, got '$SPEC_DECODING'" >&2
    exit 1
fi
if [[ "$TP" != "8" || "$DP_ATTENTION" != "false" ]]; then
    echo "Error: full-context GLM-5.2 AgentX requires TP=8 and DP_ATTENTION=false" >&2
    exit 1
fi

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
nvidia-smi

resolve_trace_source
install_agentic_deps

SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

# Match the production SGLang AgentX launchers: retain inactive radix-tree
# entries in HiCache so multi-turn session prefixes can be reused from L2.
export SGLANG_ENABLE_UNIFIED_RADIX_TREE=1
export SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS=1

# HiCache allocates its host pool per rank. GLM-5.2 also allocates a separate
# DSA indexer in host memory (observed at 20.1% of the KV pool), so reserve 21%
# for it before dividing the workflow-provided aggregate budget across TP8.
CACHE_ARGS=()
if require_agentic_kv_offload_backend hicache; then
    REQUESTED_HICACHE_TOTAL_GB="${HICACHE_TOTAL_CPU_DRAM_GB:-$TOTAL_CPU_DRAM_GB}"
    if [ "$REQUESTED_HICACHE_TOTAL_GB" -gt "$TOTAL_CPU_DRAM_GB" ]; then
        echo "Error: requested HiCache pool ${REQUESTED_HICACHE_TOTAL_GB} GB exceeds configured capacity ${TOTAL_CPU_DRAM_GB} GB" >&2
        exit 1
    fi
    HICACHE_HOST_POOL_COUNT="${HICACHE_HOST_POOL_COUNT:-1}"
    HICACHE_DSA_INDEXER_OVERHEAD_PERCENT="${HICACHE_DSA_INDEXER_OVERHEAD_PERCENT:-21}"
    if ! [[ "$HICACHE_DSA_INDEXER_OVERHEAD_PERCENT" =~ ^[0-9]+$ ]]; then
        echo "Error: HICACHE_DSA_INDEXER_OVERHEAD_PERCENT must be a non-negative integer" >&2
        exit 1
    fi
    MAX_HICACHE_SIZE_GB=$((
        REQUESTED_HICACHE_TOTAL_GB * 100
        / (TP * HICACHE_HOST_POOL_COUNT * (100 + HICACHE_DSA_INDEXER_OVERHEAD_PERCENT))
    ))
    HICACHE_SIZE_GB="${HICACHE_SIZE_GB:-$MAX_HICACHE_SIZE_GB}"
    if [ "$HICACHE_SIZE_GB" -gt "$MAX_HICACHE_SIZE_GB" ]; then
        echo "Error: HICACHE_SIZE_GB=$HICACHE_SIZE_GB exceeds configured per-pool limit $MAX_HICACHE_SIZE_GB" >&2
        exit 1
    fi
    if [ "$HICACHE_SIZE_GB" -lt 1 ]; then
        echo "Error: computed HICACHE_SIZE_GB=$HICACHE_SIZE_GB from TOTAL_CPU_DRAM_GB=$TOTAL_CPU_DRAM_GB and TP=$TP" >&2
        exit 1
    fi

    HICACHE_PAGE_SIZE="${HICACHE_PAGE_SIZE:-64}"
    HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_through}"
    # Kernel + page_first is SGLang's recommended local L2 path. Keep both
    # overridable so the H200 qualification can compare direct +
    # page_first_direct if GLM-5.2's DSA element shape misses the JIT fast path.
    HICACHE_IO_BACKEND="${HICACHE_IO_BACKEND:-kernel}"
    HICACHE_MEM_LAYOUT="${HICACHE_MEM_LAYOUT:-page_first}"
    ESTIMATED_HICACHE_TOTAL_GB=$((
        HICACHE_SIZE_GB * TP * HICACHE_HOST_POOL_COUNT
        * (100 + HICACHE_DSA_INDEXER_OVERHEAD_PERCENT) / 100
    ))
    CACHE_ARGS=(
        --page-size "$HICACHE_PAGE_SIZE"
        --enable-hierarchical-cache
        --hicache-size "$HICACHE_SIZE_GB"
        --hicache-write-policy "$HICACHE_WRITE_POLICY"
        --hicache-io-backend "$HICACHE_IO_BACKEND"
        --hicache-mem-layout "$HICACHE_MEM_LAYOUT"
    )
    echo "HiCache GLM-5.2 CPU tier: ${HICACHE_SIZE_GB} GB KV per rank across TP=${TP}; reserving ${HICACHE_DSA_INDEXER_OVERHEAD_PERCENT}% DSA indexer overhead, estimated aggregate=${ESTIMATED_HICACHE_TOTAL_GB} GB/${REQUESTED_HICACHE_TOTAL_GB} GB; page_size=$HICACHE_PAGE_SIZE, write_policy=$HICACHE_WRITE_POLICY, io_backend=$HICACHE_IO_BACKEND, mem_layout=$HICACHE_MEM_LAYOUT"
fi

# The TP8 engine serves $PORT directly. Do not enable attention DP here: it
# limits an individual request to one H200's KV pool (~135K BF16-KV tokens),
# whereas TP8 plus FP8 KV provides enough L1 capacity for the complete trace.
SGLANG_BACKEND_PORT="$PORT"

# AgentX concurrency counts live session trees, not individual HTTP requests.
# Allow subagent fan-out while retaining the upstream H200 engine cap.
MAX_RUNNING_REQUESTS=$((2 * CONC))
[ "$MAX_RUNNING_REQUESTS" -gt 256 ] && MAX_RUNNING_REQUESTS=256
CUDA_GRAPH_MAX_BS="$CONC"
[ "$CUDA_GRAPH_MAX_BS" -gt 64 ] && CUDA_GRAPH_MAX_BS=64

export PYTHONNOUSERSITE=1
export TORCH_CUDA_ARCH_LIST=9.0
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
export SGLANG_TIMEOUT_KEEP_ALIVE=900

PARALLEL_ARGS=(--tp "$TP" --ep-size "$EP_SIZE")
# Keep the cookbook's default whole-engine prefill budget.
CHUNKED_PREFILL_SIZE=8192

SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$SGLANG_BACKEND_PORT"
    --trust-remote-code
    "${PARALLEL_ARGS[@]}"
    # The unfiltered AgentX corpus reaches 488,209 input tokens. The Hopper
    # default BF16 KV pool holds only ~297K tokens in TP8, so use the same FP8
    # KV strategy as the production DeepSeek-V4 AgentX launchers.
    --kv-cache-dtype fp8_e4m3
    --tool-call-parser glm47
    --reasoning-parser glm45
    # Pin the checkpoint's native context explicitly. Truncation is forbidden:
    # an input-length rejection is a failed qualification, not a usable sample.
    --context-length 1048576
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
    --mem-fraction-static 0.88
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS"
    --watchdog-timeout 1800
    --enable-metrics
    "${CACHE_ARGS[@]}"
)

printf '%q ' "${SGLANG_CMD[@]}" | tee "$RESULT_DIR/sglang_command.txt"
printf '\n' | tee -a "$RESULT_DIR/sglang_command.txt"

echo "Starting aggregate SGLang server for GLM-5.2 FP8 on H200..."
"${SGLANG_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

capture_cache_metrics() {
    {
        echo "=== SGLang cache metrics snapshot $(date --iso-8601=seconds) ==="
        curl -fsS "http://localhost:$SGLANG_BACKEND_PORT/metrics" 2>/dev/null \
            | grep -E '^(sglang:(cache_hit_rate|cached_tokens_total|prompt_tokens_total|hicache_host_used_tokens|hicache_host_total_tokens|token_usage|num_requests_running|num_requests_waiting))' \
            || true
        echo "============================================================"
    } >> "$SERVER_LOG"
}

wait_for_server_ready --port "$SGLANG_BACKEND_PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# A model can advertise a 1M context while a topology exposes a much smaller
# per-request KV pool. Qualify the native context, truncation policy, and enough
# L1 capacity for the complete unfiltered AgentX corpus independently.
SERVER_CONTEXT_LENGTH=$(sed -nE 's/.*context_len=([0-9]+).*/\1/p' "$SERVER_LOG" | tail -1)
SERVER_KV_TOKEN_CAPACITY=$(sed -nE 's/.*max_total_num_tokens=([0-9]+).*/\1/p' "$SERVER_LOG" | tail -1)
MIN_AGENTX_KV_TOKEN_CAPACITY=524288
if [[ "$SERVER_CONTEXT_LENGTH" != "1048576" ]]; then
    echo "Error: SGLang reported context_len=${SERVER_CONTEXT_LENGTH:-unknown}; expected 1048576" >&2
    exit 1
fi
if ! grep -q 'allow_auto_truncate=False' "$SERVER_LOG"; then
    echo "Error: could not verify that SGLang automatic truncation is disabled" >&2
    exit 1
fi
if [[ -z "$SERVER_KV_TOKEN_CAPACITY" || "$SERVER_KV_TOKEN_CAPACITY" -lt "$MIN_AGENTX_KV_TOKEN_CAPACITY" ]]; then
    echo "Error: SGLang reported max_total_num_tokens=${SERVER_KV_TOKEN_CAPACITY:-unknown}; untruncated AgentX replay requires at least $MIN_AGENTX_KV_TOKEN_CAPACITY" >&2
    exit 1
fi
echo "Full-context qualification passed: context_len=$SERVER_CONTEXT_LENGTH, max_total_num_tokens=$SERVER_KV_TOKEN_CAPACITY, truncation=disabled"

capture_cache_metrics
trap capture_cache_metrics EXIT

if [ "${EVAL_ONLY}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    REPLAY_CMD+=" --server-metrics http://localhost:$SGLANG_BACKEND_PORT/metrics"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
