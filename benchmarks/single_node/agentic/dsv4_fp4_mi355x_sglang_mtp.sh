#!/usr/bin/env bash
set -eo pipefail
set -x

# Agentic trace replay benchmark for DeepSeek-V4-Pro FP4 on MI355X using SGLang.
#
# KV_OFFLOADING=dram requires KV_OFFLOAD_BACKEND=hicache.
#
# Required env vars:
#   MODEL, TP, CONC, KV_OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR
#
# KV_OFFLOADING=dram requires one of these.
#   KV_OFFLOAD_BACKEND=hicache.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE DP_ATTENTION

if [[ -n "$SLURM_JOB_ID" ]]; then
    echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

# ROCR/HIP visibility under slurm cgroups.
if [ -n "$ROCR_VISIBLE_DEVICES" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

if [[ -n "$MODEL_PATH" ]]; then
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
if agentic_kv_offload_enabled; then
    # HiCache config — https://lmsysorg.mintlify.app/cookbook/autoregressive/DeepSeek/DeepSeek-V4
    case "$KV_OFFLOAD_BACKEND" in
        hicache)
            HICACHE_RATIO=4
            HICACHE_WRITE_POLICY="write_through"
            HICACHE_IO_BACKEND="direct"
            HICACHE_MEM_LAYOUT="page_first_direct"
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
            echo "Error: unsupported KV_OFFLOAD_BACKEND '$KV_OFFLOAD_BACKEND' (expected: hicache)" >&2
            exit 1
            ;;
    esac
fi
# ---- Client config ----------------------------------------------------------
export AIPERF_HTTP_TCP_USER_TIMEOUT=1000000

# ---- LLM server config ----------------------------------------------------------
USE_SGLANG_ROUTER=false
SGLANG_BACKEND_PORT="$PORT"
ROUTER_LOG="$RESULT_DIR/router.log"
MEM_FRACTION_STATIC=0.90
CHUNKED_PREFILL_SIZE=8192
PARALLEL_ARGS=(--tensor-parallel-size "$TP")
if [ "$DP_ATTENTION" = "true" ]; then
    USE_SGLANG_ROUTER=true
    # DPA + MTP needs additional runtime headroom for speculative decode and
    # communication buffers beyond SGLang's static KV pool.
    # 0719
    #MEM_FRACTION_STATIC=0.75
    export AIPERF_HTTP_X_SMG_ROUTING_KEY_FROM_CORRELATION_ID=true
    SGLANG_BACKEND_PORT=$((PORT + 1))
    SGLANG_ROUTER_METRICS_PORT=$((PORT + 10000))
    SGLANG_ROUTER_CMD=(python3 -m sglang_router.launch_router)

    export SGLANG_SHARED_EXPERT_TP1=1
    export SGLANG_DP_SHARED_EXPERT_LOCAL=1
    export SGLANG_DP_USE_GATHERV=1
    export SGLANG_DP_USE_REDUCE_SCATTER=1

    # SGLang divides the configured chunk across DP schedulers. Use a 16K
    # per-scheduler chunk so long agentic prefill tails drain within the
    # standard 600-second warmup grace period.
    CHUNKED_PREFILL_SIZE=$((16384 * TP))
    PARALLEL_ARGS+=(
        --dp "$TP"
        --enable-dp-attention
        --enable-prefill-delayer
    )
fi

if [ "$EP_SIZE" -gt 1 ]; then
    PARALLEL_ARGS+=(--ep-size "$EP_SIZE")
fi

# SGLang treats max-running-requests as a global DPA limit and partitions it
# internally. CUDA graph capture is per scheduler, so only its batch size is
# divided across DP ranks.
# 0719
MAX_RUNNING_REQUESTS=$((2 * CONC))
# 0719
CUDA_GRAPH_MAX_BS=$CONC
[ "$CUDA_GRAPH_MAX_BS" -gt 128 ] && CUDA_GRAPH_MAX_BS=128

# Simulated acceptance-length (AL) settings.
export SGLANG_DEFAULT_THINKING=1
export SGLANG_DSV4_REASONING_EFFORT=high
export SGLANG_SIMULATE_ACC_LEN=2.49
export SGLANG_SIMULATE_ACC_METHOD=match-expected
export SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token

export SGLANG_USE_ROCM700A=0
export SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton
export AITER_BF16_FP8_MOE_BOUND=0

export SGLANG_ENABLE_UNIFIED_RADIX_TREE=1
export SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS=1

METRICS_ARGS=(--enable-metrics)
SPEC_ARGS=(
    --speculative-algorithm EAGLE
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
)

SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$SGLANG_BACKEND_PORT"
    --trust-remote-code
    "${PARALLEL_ARGS[@]}"
    --attention-backend compressed
    --cuda-graph-max-bs-decode "$CUDA_GRAPH_MAX_BS"
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --mem-fraction-static "$MEM_FRACTION_STATIC"
    --swa-full-tokens-ratio 0.10
    --page-size 256
    --kv-cache-dtype fp8_e4m3
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
    --disable-shared-experts-fusion
    --tool-call-parser deepseekv4
    --reasoning-parser deepseek-v4
    --chat-template "$(dirname "$0")/../chat_templates/deepseek_v4_thinking.jinja"
    --watchdog-timeout 1800
    "${METRICS_ARGS[@]}"
    "${SPEC_ARGS[@]}"
    "${CACHE_ARGS[@]}"
)

printf '%q ' "${SGLANG_CMD[@]}" | tee "$RESULT_DIR/sglang_command.txt"
printf '\n' | tee -a "$RESULT_DIR/sglang_command.txt"

{
    echo "=== SGLANG_* env vars at launch ==="
    env | grep -E '^SGLANG_' | sort
    echo "==================================="
} | tee "$SERVER_LOG"

echo "Starting SGLang server for MI355X..."
"${SGLANG_CMD[@]}" >> "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$SGLANG_BACKEND_PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [ "$USE_SGLANG_ROUTER" = "true" ]; then
    echo "Starting SGLang router on port $PORT for $TP DP ranks..."
    "${SGLANG_ROUTER_CMD[@]}" \
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
fi

# ---- Run benchmark ----------------------------------------------------------
build_replay_cmd "$RESULT_DIR"
REPLAY_CMD+=" --server-metrics http://localhost:$SGLANG_BACKEND_PORT/metrics"

run_agentic_replay_and_write_outputs "$RESULT_DIR"
