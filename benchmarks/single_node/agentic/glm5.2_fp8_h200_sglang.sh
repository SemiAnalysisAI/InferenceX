#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for GLM-5.2 FP8 on one 8xH200 node using
# aggregate SGLang serving. This is intentionally STP: no speculative decoding
# flags are passed even though the checkpoint includes an MTP head.
#
# The base engine follows the upstream H200 high-throughput recipe: TP8/DP8,
# attention DP, EP8/DeepEP, BF16 KV selected automatically on Hopper, and a
# 0.85 static-memory fraction. The optional DRAM tier uses SGLang HiCache.
#
# Required env vars:
#   MODEL, TP, CONC, KV_OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR, DURATION,
#   EP_SIZE, DP_ATTENTION, SPEC_DECODING

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE DP_ATTENTION SPEC_DECODING

if [[ "$TP" != "8" || "$EP_SIZE" != "8" || "$DP_ATTENTION" != "true" ]]; then
    echo "Error: this H200 recipe requires TP=8, EP_SIZE=8, and DP_ATTENTION=true" >&2
    exit 1
fi
if [[ "$SPEC_DECODING" != "none" ]]; then
    echo "Error: this is an STP recipe; expected SPEC_DECODING=none, got '$SPEC_DECODING'" >&2
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
ROUTER_LOG="$RESULT_DIR/router.log"
mkdir -p "$RESULT_DIR"

# HiCache allocates its host pool per rank. GLM-5.2 has one DSA KV pool per
# rank, so divide the workflow-provided aggregate host budget across TP8.
CACHE_ARGS=()
if require_agentic_kv_offload_backend hicache; then
    REQUESTED_HICACHE_TOTAL_GB="${HICACHE_TOTAL_CPU_DRAM_GB:-$TOTAL_CPU_DRAM_GB}"
    if [ "$REQUESTED_HICACHE_TOTAL_GB" -gt "$TOTAL_CPU_DRAM_GB" ]; then
        echo "Error: requested HiCache pool ${REQUESTED_HICACHE_TOTAL_GB} GB exceeds configured capacity ${TOTAL_CPU_DRAM_GB} GB" >&2
        exit 1
    fi
    HICACHE_HOST_POOL_COUNT="${HICACHE_HOST_POOL_COUNT:-1}"
    MAX_HICACHE_SIZE_GB=$((REQUESTED_HICACHE_TOTAL_GB / TP / HICACHE_HOST_POOL_COUNT))
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
    CACHE_ARGS=(
        --page-size "$HICACHE_PAGE_SIZE"
        --enable-hierarchical-cache
        --hicache-size "$HICACHE_SIZE_GB"
        --hicache-write-policy "$HICACHE_WRITE_POLICY"
        --hicache-io-backend "$HICACHE_IO_BACKEND"
        --hicache-mem-layout "$HICACHE_MEM_LAYOUT"
    )
    echo "HiCache GLM-5.2 CPU tier: ${HICACHE_SIZE_GB} GB per rank across TP=${TP}; page_size=$HICACHE_PAGE_SIZE, write_policy=$HICACHE_WRITE_POLICY, io_backend=$HICACHE_IO_BACKEND, mem_layout=$HICACHE_MEM_LAYOUT"
fi

# DP-aware consistent hashing keeps every multi-turn AgentX session on the DP
# rank that owns its Radix/HiCache state.
export AIPERF_HTTP_X_SMG_ROUTING_KEY_FROM_CORRELATION_ID=true
SGLANG_BACKEND_PORT=$((PORT + 1))
SGLANG_ROUTER_METRICS_PORT=$((PORT + 10000))

# AgentX concurrency counts live session trees, not individual HTTP requests.
# Allow subagent fan-out while retaining the upstream H200 engine cap.
MAX_RUNNING_REQUESTS=$((2 * CONC))
[ "$MAX_RUNNING_REQUESTS" -gt 256 ] && MAX_RUNNING_REQUESTS=256

export PYTHONNOUSERSITE=1
export TORCH_CUDA_ARCH_LIST=9.0
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
export SGLANG_TIMEOUT_KEEP_ALIVE=900

SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$SGLANG_BACKEND_PORT"
    --trust-remote-code
    --tp "$TP"
    --dp "$TP"
    --ep-size "$EP_SIZE"
    --enable-dp-attention
    --moe-a2a-backend deepep
    --tokenizer-worker-num "$TP"
    --dist-init-addr "127.0.0.1:$((PORT + 2000))"
    # Hopper uses BF16 KV for GLM-5.2 DSA; let SGLang select it rather than
    # copying the Blackwell-only fp8_e4m3 setting from the NVFP4 recipe.
    --tool-call-parser glm47
    --reasoning-parser glm45
    --chunked-prefill-size 32768
    --mem-fraction-static 0.85
    --max-running-requests "$MAX_RUNNING_REQUESTS"
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

echo "Starting SGLang router on port $PORT for $TP DP ranks..."
python3 -m sglang_router.launch_router \
    --worker-urls "http://localhost:$SGLANG_BACKEND_PORT" \
    --policy consistent_hashing \
    --request-id-headers x-correlation-id \
    --dp-aware \
    --host 0.0.0.0 \
    --port "$PORT" \
    --prometheus-host 127.0.0.1 \
    --prometheus-port "$SGLANG_ROUTER_METRICS_PORT" \
    --connect-timeout-secs 900 \
    --request-timeout-secs 14400 \
    --disable-health-check \
    --disable-retries > "$ROUTER_LOG" 2>&1 &
ROUTER_PID=$!
echo "Router PID: $ROUTER_PID"
wait_for_server_ready --port "$PORT" --server-log "$ROUTER_LOG" --server-pid "$ROUTER_PID"

capture_cache_metrics
trap capture_cache_metrics EXIT

if [ "${EVAL_ONLY}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    REPLAY_CMD+=" --server-metrics http://localhost:$SGLANG_BACKEND_PORT/metrics"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
