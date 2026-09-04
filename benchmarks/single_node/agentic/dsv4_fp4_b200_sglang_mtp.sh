#!/usr/bin/env bash
set -eo pipefail
set -x

# Agentic trace replay for DeepSeek-V4-Pro-0813 FP4 on B200 with DSpark K=6.
# Throughput uses the committed golden synthetic AL; eval retains real target
# verification.
#
# KV_OFFLOADING=dram requires KV_OFFLOAD_BACKEND=hicache.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFERENCEX_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
export INFMAX_CONTAINER_WORKSPACE="${INFMAX_CONTAINER_WORKSPACE:-/workspace}"

# The B200 DeepSeek-V4 Blackwell image installs SGLang editable under
# /workspace, so its launcher mounts InferenceX at /ix instead. Resolve the
# agentic tooling and results against the actual repository mount so the image
# can keep its /workspace install and GitHub Actions can collect the outputs.
if [[ ! -d "$INFMAX_CONTAINER_WORKSPACE/utils/aiperf" ]]; then
    export INFMAX_CONTAINER_WORKSPACE="$INFERENCEX_ROOT"
fi
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

# Keep AIPerf's Transformers-main dependency from replacing the older
# Transformers build pinned by the B200-specialized SGLang image. The server
# always launches with the image's original interpreter; AIPerf and result
# processing use the isolated environment when InferenceX is mounted at /ix.
SGLANG_PYTHON="$(command -v python3)"
if [[ "$INFMAX_CONTAINER_WORKSPACE" != /workspace ]]; then
    AGENTIC_VENV="${AGENTIC_VENV:-/tmp/inferencex-agentic-venv}"
    "$SGLANG_PYTHON" -m venv "$AGENTIC_VENV"
    export PATH="$AGENTIC_VENV/bin:$PATH"
fi
install_agentic_deps

SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

export SGLANG_ENABLE_UNIFIED_RADIX_TREE=1
export SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS=1

CACHE_ARGS=()
if require_agentic_kv_offload_backend hicache; then
    # DeepSeek V4 HiCache currently rejects --hicache-size and supports
    # capacity control only through a host/device token-capacity ratio.
    # DSv4 exposes capacity as a host/device token ratio rather than bytes.
    # DEP8 shards the host pools and fits ratio=8 on NScale. The replicated
    # TP8 pools need a lower ratio: 2.75 allocates about 121 GiB per rank and
    # leaves startup headroom on the 1.7 TiB NScale hosts.
    DEFAULT_HICACHE_RATIO=2.75
    if [ "$DP_ATTENTION" = "true" ]; then
        DEFAULT_HICACHE_RATIO=8
    fi
    HICACHE_RATIO="${HICACHE_RATIO:-$DEFAULT_HICACHE_RATIO}"
    if awk -v ratio="$HICACHE_RATIO" -v max="$DEFAULT_HICACHE_RATIO" 'BEGIN { exit !(ratio > max) }'; then
        echo "Error: HICACHE_RATIO=$HICACHE_RATIO exceeds configured limit $DEFAULT_HICACHE_RATIO" >&2
        exit 1
    fi
    HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_through}"
    HICACHE_IO_BACKEND="${HICACHE_IO_BACKEND:-direct}"
    HICACHE_MEM_LAYOUT="${HICACHE_MEM_LAYOUT:-page_first_direct}"
    CACHE_ARGS=(
        --enable-hierarchical-cache
        --hicache-ratio "$HICACHE_RATIO"
        --hicache-write-policy "$HICACHE_WRITE_POLICY"
        --hicache-io-backend "$HICACHE_IO_BACKEND"
        --hicache-mem-layout "$HICACHE_MEM_LAYOUT"
    )
    echo "HiCache DSv4 CPU tier: ratio=$HICACHE_RATIO, capacity=${TOTAL_CPU_DRAM_GB} GB, write_policy=$HICACHE_WRITE_POLICY, io_backend=$HICACHE_IO_BACKEND, mem_layout=$HICACHE_MEM_LAYOUT"
fi

USE_SGLANG_ROUTER=false
SGLANG_BACKEND_PORT="$PORT"
ROUTER_LOG="$RESULT_DIR/router.log"
if [ "$DP_ATTENTION" = "true" ]; then
    USE_SGLANG_ROUTER=true
    ROUTER_POLICY_ARGS=()
    export AIPERF_HTTP_X_SMG_ROUTING_KEY_FROM_CORRELATION_ID=true
    SGLANG_BACKEND_PORT=$((PORT + 1))
    SGLANG_ROUTER_METRICS_PORT=$((PORT + 10000))
    SGLANG_ROUTER_CMD=("$SGLANG_PYTHON" -m sglang_router.launch_router)
fi

PARALLEL_ARGS=(--tp "$TP")
METRICS_ARGS=(--enable-metrics --enable-cache-report)
CHUNKED_PREFILL_SIZE=8192
SWA_FULL_TOKENS_RATIO=0.1
MEM_FRACTION_STATIC=0.90
if [ "$DP_ATTENTION" = "true" ]; then
    export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320

    # Leave HBM headroom for the FP4 indexer's context-dependent workspace.
    MEM_FRACTION_STATIC=0.88
    PREFILL_DECODE_INTERVAL=24

    # Keep DP admission and session routing uniform across the DEP8 curve.
    PARALLEL_ARGS+=(--load-balance-method total_requests)
    METRICS_ARGS+=(--load-snapshot-publish-interval 1)
    export AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=true
    if [ "$CONC" -eq 160 ]; then
        PREFILL_DECODE_INTERVAL=20
        ROUTER_POLICY_ARGS+=(--balance-abs-threshold 32)
    fi

    PARALLEL_ARGS+=(
        --dp "$TP"
        --tokenizer-worker-num "$TP"
        --prefill-decode-interval "$PREFILL_DECODE_INTERVAL"
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
        --disable-shared-experts-fusion
        --disable-flashinfer-autotune
    )
    # SGLang divides this global budget by dp_size. Keep 6144 tokens per rank
    # for every DP-attention profile so the FP4 indexer retains HBM headroom.
    CHUNKED_PREFILL_SIZE=$((6144 * TP))
    SWA_FULL_TOKENS_RATIO=0.02
else
    PARALLEL_ARGS+=(
        --moe-runner-backend flashinfer_mxfp4
        --enable-deepseek-v4-fp4-indexer
        --disable-flashinfer-autotune
    )
fi

# The B200-specialized image deadlocks immediately after weight loading when
# forced through the B300 compressed-attention/page-size overrides.
# DeepGEMM's DSv4 indexer needs a multi-GiB temporary allocation at long
# contexts. The selected fractions preserve the measured indexer and CUDA
# graph headroom while HiCache spills to host memory.

# AgentX concurrency counts live session trees, not individual requests.
# Allow subagent fan-out to exceed CONC without clipping request bursts.
MAX_RUNNING_REQUESTS=$((2 * CONC))
CUDA_GRAPH_MAX_BS=$((2 * CONC))
if [ "$DP_ATTENTION" = "true" ]; then
    CUDA_GRAPH_MAX_BS=32
fi
CUDA_GRAPH_ARGS=(--cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS")

export PYTHONNOUSERSITE=1
export TORCH_CUDA_ARCH_LIST=10.0
# Agentic warmup dispatches hundreds of large prompts at once. SGLang's
# tokenizer process can leave request bytes unacknowledged for longer than
# AIPerf's 30-second TCP_USER_TIMEOUT while it admits that initial burst,
# causing Linux to abort otherwise-live localhost connections. Keep the
# six-hour request timeout unchanged, but allow up to 15 minutes for TCP
# progress before declaring the connection dead.
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
    # The B200 checkpoint lives on Lustre. Partition sequential prefetching
    # across local ranks so post-load weight repacking reads from page cache
    # instead of issuing redundant fragmented mmap faults from every rank.
    --weight-loader-prefetch-checkpoints
    --model-loader-extra-config '{"enable_multithread_load": true}'
    "${METRICS_ARGS[@]}"
    "${CACHE_ARGS[@]}"
)

write_command "$RESULT_DIR/sglang_command.txt" "${SGLANG_CMD[@]}"

{
    echo "=== SGLANG_* env vars at launch ==="
    env | grep -E '^SGLANG_' | sort
    echo "==================================="
} | tee "$SERVER_LOG"

echo "Starting SGLang server for B200..."
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
        --policy cache_aware \
        "${ROUTER_POLICY_ARGS[@]}" \
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
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    REPLAY_CMD+=" --server-metrics http://localhost:$SGLANG_BACKEND_PORT/metrics"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
