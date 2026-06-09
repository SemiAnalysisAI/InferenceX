#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for DeepSeek-V4-Pro FP4 on Blackwell using
# SGLang. B200 and B300 use the same current upstream DSv4 recipes.
#
# OFFLOADING values:
#   none    - SGLang GPU KV cache with RadixAttention prefix caching.
#   hicache - SGLang HiCache local CPU tier with DSv4 UnifiedRadixCache.

source "$(dirname "${BASH_SOURCE[0]}")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE DP_ATTENTION

if [ -z "${MAX_MODEL_LEN:-}" ] || [ "$MAX_MODEL_LEN" = "0" ]; then
    MAX_MODEL_LEN=1000000
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

if [ "$DP_ATTENTION" = "true" ]; then
    echo "Error: current SGLang nightly self-collides on internal IPC ports during single-node DP-attention startup; use pure TP until upstream fixes PortArgs initialization." >&2
    exit 1
fi

CACHE_ARGS=()
case "$OFFLOADING" in
    none)
        ;;
    hicache)
        # DeepSeek V4 HiCache currently rejects --hicache-size and supports
        # capacity control only through a host/device token-capacity ratio.
        # DSv4 allocates several physical host sub-pools for each logical host
        # token. At TP8, ratio=4 consumes about 237 GB/rank (1.9 TB total) while
        # model loading/page cache is still resident and the OS kills a rank.
        # TP4 ratio=4 works but fills its roughly 500 GB host tier during the
        # C48/C64 focused tests and useful host hits collapse. Ratio=8 doubles
        # that logical capacity while remaining below the node's host budget.
        # Use ratio=2 at TP8 to leave enough transient headroom during startup.
        if [ "$TP" -ge 8 ]; then
            DEFAULT_HICACHE_RATIO=2
        else
            DEFAULT_HICACHE_RATIO=8
        fi
        HICACHE_RATIO="${HICACHE_RATIO:-$DEFAULT_HICACHE_RATIO}"
        HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_through}"
        HICACHE_IO_BACKEND="${HICACHE_IO_BACKEND:-direct}"
        HICACHE_MEM_LAYOUT="${HICACHE_MEM_LAYOUT:-page_first_direct}"
        export SGLANG_ENABLE_UNIFIED_RADIX_TREE=1
        CACHE_ARGS=(
            --enable-hierarchical-cache
            --hicache-ratio "$HICACHE_RATIO"
            --hicache-write-policy "$HICACHE_WRITE_POLICY"
            --hicache-io-backend "$HICACHE_IO_BACKEND"
            --hicache-mem-layout "$HICACHE_MEM_LAYOUT"
        )
        echo "HiCache DSv4 CPU tier: ratio=$HICACHE_RATIO, write_policy=$HICACHE_WRITE_POLICY, io_backend=$HICACHE_IO_BACKEND, mem_layout=$HICACHE_MEM_LAYOUT"
        ;;
    *)
        echo "Error: unsupported OFFLOADING value '$OFFLOADING' (expected one of: none, hicache)" >&2
        exit 1
        ;;
esac

PARALLEL_ARGS=(--tp "$TP")
METRICS_ARGS=(--enable-metrics)
MEM_FRACTION_STATIC=0.88
CHUNKED_PREFILL_SIZE=8192
PARALLEL_ARGS+=(
    --moe-runner-backend flashinfer_mxfp4
    --disable-flashinfer-autotune
)

PER_ENGINE_MAX_RUNNING=$CONC
[ "$PER_ENGINE_MAX_RUNNING" -lt 1 ] && PER_ENGINE_MAX_RUNNING=1
CUDA_GRAPH_MAX_BS=$PER_ENGINE_MAX_RUNNING
[ "$CUDA_GRAPH_MAX_BS" -gt 64 ] && CUDA_GRAPH_MAX_BS=64

export PYTHONNOUSERSITE=1
export TORCH_CUDA_ARCH_LIST=10.0
# Agentic warmup dispatches hundreds of large prompts at once. SGLang's
# tokenizer process can leave request bytes unacknowledged for longer than
# AIPerf's 30-second TCP_USER_TIMEOUT while it admits that initial burst,
# causing Linux to abort otherwise-live localhost connections. Keep the
# six-hour request timeout unchanged, but allow up to 15 minutes for TCP
# progress before declaring the connection dead.
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT=1
export SGLANG_OPT_USE_JIT_NORM=1
export SGLANG_OPT_USE_JIT_INDEXER_METADATA=1
export SGLANG_OPT_USE_TOPK_V2=1
export SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=1

SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    "${PARALLEL_ARGS[@]}"
    --attention-backend compressed
    --page-size 256
    --mem-fraction-static "$MEM_FRACTION_STATIC"
    --swa-full-tokens-ratio 0.1
    --max-running-requests "$PER_ENGINE_MAX_RUNNING"
    --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS"
    --context-length "$MAX_MODEL_LEN"
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
    --disable-shared-experts-fusion
    --tool-call-parser deepseekv4
    --reasoning-parser deepseek-v4
    --chat-template "$(dirname "${BASH_SOURCE[0]}")/../chat_templates/deepseek_v4_thinking.jinja"
    --watchdog-timeout 1800
    "${METRICS_ARGS[@]}"
    "${CACHE_ARGS[@]}"
)

printf '%q ' "${SGLANG_CMD[@]}" | tee "$RESULT_DIR/sglang_command.txt"
printf '\n' | tee -a "$RESULT_DIR/sglang_command.txt"

{
    echo "=== SGLANG_* env vars at launch ==="
    env | grep -E '^SGLANG_' | sort
    echo "==================================="
} | tee "$SERVER_LOG"

echo "Starting SGLang server for ${DSV4_SGLANG_PLATFORM:-Blackwell}..."
"${SGLANG_CMD[@]}" >> "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

capture_cache_metrics() {
    {
        echo "=== SGLang cache metrics snapshot $(date --iso-8601=seconds) ==="
        curl -fsS "http://localhost:$PORT/metrics" 2>/dev/null \
            | grep -E '^(sglang:(cache_hit_rate|cached_tokens_total|prompt_tokens_total|hicache_host_used_tokens|hicache_host_total_tokens|token_usage|num_requests_running|num_requests_waiting))' \
            || true
        echo "============================================================"
    } >> "$SERVER_LOG"
}

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"
if [ "${#METRICS_ARGS[@]}" -gt 0 ]; then
    capture_cache_metrics
    trap capture_cache_metrics EXIT
fi

build_replay_cmd "$RESULT_DIR"
run_agentic_replay_and_write_outputs "$RESULT_DIR"
