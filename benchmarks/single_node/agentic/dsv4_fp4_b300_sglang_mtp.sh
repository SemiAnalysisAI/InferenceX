#!/usr/bin/env bash
set -eo pipefail
set -x

# DeepSeek-V4-Pro-0813 FP4 on B300 with SGLang DSpark K=6.
# KV_OFFLOADING=dram requires KV_OFFLOAD_BACKEND=hicache.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFERENCEX_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$INFERENCEX_ROOT/benchmarks/benchmark_lib.sh" --validation-only
check_env_vars INFMAX_CONTAINER_WORKSPACE RESULT_DIR

# The B200 DeepSeek-V4 image installs SGLang editable under /workspace, so its
# launcher mounts InferenceX at /ix. Resolve tooling and results against the
# actual repository mount.
if [[ "${RESULT_DIR:-}" == /workspace/* && "$INFMAX_CONTAINER_WORKSPACE" != /workspace ]]; then
    export RESULT_DIR="$INFMAX_CONTAINER_WORKSPACE/${RESULT_DIR#/workspace/}"
fi
source "$INFERENCEX_ROOT/benchmarks/benchmark_lib.sh"

export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="sglang:"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE DP_ATTENTION

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

# AIPerf's Transformers-main dependency would replace the Transformers build
# pinned by the B200 SGLang image; the server keeps the image interpreter and
# AIPerf runs from an isolated venv when InferenceX is mounted at /ix.
SGLANG_PYTHON="$(command -v python3)"
if [[ "$INFMAX_CONTAINER_WORKSPACE" != /workspace ]]; then
    AGENTIC_VENV="/tmp/inferencex-agentic-venv"
    "$SGLANG_PYTHON" -m venv "$AGENTIC_VENV"
    export PATH="$AGENTIC_VENV/bin:$PATH"
fi
install_agentic_deps

SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

export SGLANG_ENABLE_UNIFIED_RADIX_TREE=1
export SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS=1

CACHE_ARGS=()
WARMUP_ARGS=()
if require_agentic_kv_offload_backend hicache; then
    # DeepSeek V4 HiCache rejects --hicache-size; capacity is a host/device
    # token ratio, and host bytes scale with ratio AND mem-fraction-static.
    # TP8 ratio=4 at 0.93 left 5.84 GB free on a 2,964 GB node and the paged
    # pool failed to allocate; ratio=3 keeps the tier near 2 TB with room for
    # the paged pool, page cache, AIPerf and the router.
    if [ "$TP" -ge 8 ]; then
        HICACHE_RATIO=3
    else
        HICACHE_RATIO=8
    fi
    HICACHE_WRITE_POLICY="write_back"
    HICACHE_IO_BACKEND="direct"
    HICACHE_MEM_LAYOUT="page_first_direct"
    CACHE_ARGS=(
        --enable-hierarchical-cache
        --hicache-ratio "$HICACHE_RATIO"
        --hicache-write-policy "$HICACHE_WRITE_POLICY"
        --hicache-io-backend "$HICACHE_IO_BACKEND"
        --hicache-mem-layout "$HICACHE_MEM_LAYOUT"
    )
    # AIPerf owns the AgentX warmup; SGLang's per-DP warmup can time out after
    # the API is already healthy.
    WARMUP_ARGS=(--skip-server-warmup)
    echo "HiCache DSv4 CPU tier: ratio=$HICACHE_RATIO, capacity=${TOTAL_CPU_DRAM_GB} GB, write_policy=$HICACHE_WRITE_POLICY, io_backend=$HICACHE_IO_BACKEND, mem_layout=$HICACHE_MEM_LAYOUT"
fi

USE_SGLANG_ROUTER=false
SGLANG_BACKEND_PORT="$PORT"
ROUTER_LOG="$RESULT_DIR/router.log"
if [ "$DP_ATTENTION" = "true" ]; then
    USE_SGLANG_ROUTER=true
    export AIPERF_HTTP_X_SMG_ROUTING_KEY_FROM_CORRELATION_ID=true
    SGLANG_BACKEND_PORT=$((PORT + 1))
    SGLANG_ROUTER_METRICS_PORT=$((PORT + 10000))
    SGLANG_ROUTER_CMD=("$SGLANG_PYTHON" -m sglang_router.launch_router)
fi

PARALLEL_ARGS=(--tp "$TP")
METRICS_ARGS=(--enable-metrics --enable-cache-report)
MEM_FRACTION_STATIC=0.88
CHUNKED_PREFILL_SIZE=8192
if [ "$DP_ATTENTION" = "true" ]; then
    PARALLEL_ARGS+=(
        --dp "$TP"
        --tokenizer-worker-num "$TP"
        --enable-prefill-delayer
        --prefill-decode-interval 20
        --enable-dp-attention
        --enable-dp-lm-head
        --enable-dp-attention-local-control-broadcast
        --incremental-streaming-output
        --stream-interval 20
        --dist-init-addr "127.0.0.1:$((PORT + 2000))"
        --ep-size "$EP_SIZE"
        --moe-a2a-backend megamoe
        --enable-w4a4-mxfp4-megamoe
        --enable-deepseek-v4-fp4-indexer
        --disable-flashinfer-autotune
    )
    if [ "$TP" -ge 8 ]; then
        # Mega-MoE's transient workspace lives outside the static allocation
        # and needs one ~7 GB contiguous block, so headroom grows with
        # concurrency. At conc 256, 0.835 runs; 0.93 and 0.95 OOM one DP rank
        # and hang the engine in the MLP-sync collective.
        MEM_FRACTION_STATIC=0.93
        if [ "$CONC" -ge 512 ]; then
            MEM_FRACTION_STATIC=0.86
        elif [ "$CONC" -ge 384 ]; then
            MEM_FRACTION_STATIC=0.88
        elif [ "$CONC" -ge 32 ]; then
            MEM_FRACTION_STATIC=0.90
        fi
    else
        # DEP4 weights take ~90% of each GPU, so the engine refuses to start
        # below ~0.902, while megamoe still needs its ~7 GB workspace above
        # the static budget; 0.93 leaves ~16 GB for it.
        MEM_FRACTION_STATIC=0.93
    fi
    # --chunked-prefill-size is a global budget divided by dp_size (=TP).
    # Scale it so every DEP shape gets 8192 per rank; 16384/rank exceeds
    # MegaMoE's per-rank token cap (startup ValueError).
    CHUNKED_PREFILL_SIZE=$((8192 * TP))
else
    PARALLEL_ARGS+=(
        --moe-runner-backend flashinfer_mxfp4
        --disable-flashinfer-autotune
    )
fi

MODEL_ARGS=(
    --attention-backend compressed
    --page-size 256
    --disable-shared-experts-fusion
)

# AgentX concurrency counts live session trees, not individual requests.
# Allow subagent fan-out to exceed CONC without clipping request bursts.
MAX_RUNNING_REQUESTS=$((2 * CONC))
# Live requests exceed CONC under fan-out, so graphs sized at CONC would drop
# larger batches to eager decode; the runtime clamps to the request pool anyway.
CUDA_GRAPH_MAX_BS=$((CONC * 4))
[ "$CUDA_GRAPH_MAX_BS" -gt 64 ] && CUDA_GRAPH_MAX_BS=64

# --cuda-graph-max-bs is an alias whose dest is cuda_graph_max_bs_decode, so the
# two forms below are the same knob and must not both be passed.
CUDA_GRAPH_ARGS=(--cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS")
SWA_FULL_TOKENS_RATIO=0.1
if [ "$DP_ATTENTION" = "true" ]; then
    # Decode graphs must cover the padded speculative batch across all DP ranks, which
    # exceeds CONC; capping at 64 would fall back to eager decode.
    CUDA_GRAPH_ARGS=(--cuda-graph-max-bs-decode 544)
    SWA_FULL_TOKENS_RATIO=0.075
fi

export PYTHONNOUSERSITE=1
export TORCH_CUDA_ARCH_LIST=10.0
# Agentic warmup dispatches hundreds of large prompts at once and SGLang's
# tokenizer can leave bytes unacknowledged past AIPerf's default 30 s
# TCP_USER_TIMEOUT, so Linux aborts live localhost connections.
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
# Outlast AIPerf's pooled connections so an inter-turn idle gap cannot race
# Uvicorn's five-second keep-alive closure.
export SGLANG_TIMEOUT_KEEP_ALIVE=900
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
export SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT=1
export SGLANG_OPT_USE_JIT_NORM=1
export SGLANG_OPT_USE_JIT_INDEXER_METADATA=1
export SGLANG_OPT_USE_TOPK_V2=1
export SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=1
if [ "$DP_ATTENTION" = "true" ]; then
    # Must cover the per-rank prefill budget (8192) or startup raises; the
    # extra 128 is headroom over the exact-fit boundary.
    export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320
fi
if [ "${EVAL_ONLY}" != "true" ]; then
    export SGLANG_SIMULATE_ACC_LEN=3.77
    export SGLANG_SIMULATE_ACC_METHOD=match-expected
    export SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
fi
TRITON_PTXAS_PATH=$(find \
    /usr/local/cuda* \
    /usr/local/lib/python*/dist-packages/nvidia \
    /usr/local/lib/python*/site-packages/nvidia \
    -type f -name ptxas -perm -u+x -print -quit 2>/dev/null || true)
if [ -n "$TRITON_PTXAS_PATH" ]; then
    export TRITON_PTXAS_PATH
    echo "Using ptxas for Triton: $TRITON_PTXAS_PATH"
fi
SGLANG_CMD=(
    "$SGLANG_PYTHON" -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$SGLANG_BACKEND_PORT"
    --trust-remote-code
    "${PARALLEL_ARGS[@]}"
    --mem-fraction-static "$MEM_FRACTION_STATIC"
    --swa-full-tokens-ratio "$SWA_FULL_TOKENS_RATIO"
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    "${CUDA_GRAPH_ARGS[@]}"
    --allow-auto-truncate
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
    --tool-call-parser deepseekv4
    --reasoning-parser deepseek-v4
    --chat-template "$SCRIPT_DIR/../chat_templates/deepseek_v4_thinking.jinja"
    --watchdog-timeout 1800
    --speculative-algorithm DSPARK
    --speculative-dspark-block-size 6
    --speculative-num-steps 1
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 7
    "${MODEL_ARGS[@]}"
    "${METRICS_ARGS[@]}"
    "${CACHE_ARGS[@]}"
    "${WARMUP_ARGS[@]}"
)

write_command "$RESULT_DIR/sglang_command.txt" "${SGLANG_CMD[@]}"

{
    echo "=== SGLANG_* env vars at launch ==="
    env | grep -E '^SGLANG_' | sort
    echo "==================================="
} | tee "$SERVER_LOG"

echo "Starting SGLang server for B300..."
"${SGLANG_CMD[@]}" >> "$SERVER_LOG" 2>&1 &
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

wait_for_ready \
    --endpoint "http://localhost:$SGLANG_BACKEND_PORT/health" \
    --log "$SERVER_LOG" \
    --pid "$SERVER_PID"

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
        `# A single transient router->engine send failure would otherwise` \
        `# surface as a 500, and AgentX aborts the whole run when a root` \
        `# warmup request fails ("ProfileAborted"). Measured at conc 512:` \
        `# 22 such transients in one 3600s run, spread over all 8 DP` \
        `# workers, every one of them recovered by the retry; with retries` \
        `# disabled a single one killed a 2h15m arm.` \
        --retry-max-retries 8 \
        --retry-initial-backoff-ms 500 \
        --retry-max-backoff-ms 10000 \
        --retry-backoff-multiplier 2 > "$ROUTER_LOG" 2>&1 &
    ROUTER_PID=$!
    echo "Router PID: $ROUTER_PID"
    wait_for_ready \
        --endpoint "http://localhost:$PORT/health" \
        --log "$ROUTER_LOG" \
        --pid "$ROUTER_PID"
fi

if [ "${#METRICS_ARGS[@]}" -gt 0 ]; then
    capture_cache_metrics
    trap capture_cache_metrics EXIT
fi

if [ "${EVAL_ONLY}" = "true" ]; then
    git config --global --add safe.directory "$INFMAX_CONTAINER_WORKSPACE"
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    REPLAY_CMD+=" --server-metrics http://localhost:$SGLANG_BACKEND_PORT/metrics"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
