#!/usr/bin/env bash
set -eo pipefail
set -x

# Agentic trace replay for DeepSeek-V4-Pro FP4 on B300 with native EAGLE MTP.
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
WARMUP_ARGS=()
if require_agentic_kv_offload_backend hicache; then
    # DeepSeek V4 HiCache rejects --hicache-size and controls capacity only
    # through a host/device token ratio, so TOTAL_CPU_DRAM_GB cannot apply
    # directly. Host capacity scales with BOTH the ratio and device KV, so it
    # also grows with mem-fraction-static -- the two knobs multiply. Measured:
    # TP8 ratio=2 at mem-fraction 0.835 gives 999 GB. ratio=4 at mem-fraction
    # 0.93 overshoots: it left only 5.84 GB free on a 2,964 GB node and the
    # V4 paged pool failed to allocate. ratio=3 keeps the tier near 2 TB with
    # room for the paged pool, page cache, AIPerf and the router, while still
    # well above the old half-node rule that pinned TP8 to ratio=2.
    if [ "$TP" -ge 8 ]; then
        DEFAULT_HICACHE_RATIO=3
    else
        DEFAULT_HICACHE_RATIO=8
    fi
    HICACHE_RATIO="${HICACHE_RATIO:-$DEFAULT_HICACHE_RATIO}"
    if [ "$HICACHE_RATIO" -gt "$DEFAULT_HICACHE_RATIO" ]; then
        echo "Error: HICACHE_RATIO=$HICACHE_RATIO exceeds configured limit $DEFAULT_HICACHE_RATIO" >&2
        exit 1
    fi
    HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_back}"
    HICACHE_IO_BACKEND="${HICACHE_IO_BACKEND:-direct}"
    HICACHE_MEM_LAYOUT="${HICACHE_MEM_LAYOUT:-page_first_direct}"
    CACHE_ARGS=(
        --enable-hierarchical-cache
        --hicache-ratio "$HICACHE_RATIO"
        --hicache-write-policy "$HICACHE_WRITE_POLICY"
        --hicache-io-backend "$HICACHE_IO_BACKEND"
        --hicache-mem-layout "$HICACHE_MEM_LAYOUT"
    )
    # AIPerf owns the representative warmup for AgentX. Avoid SGLang's
    # redundant per-DP warmup timing out after the API is already healthy.
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
        --enable-dp-attention-local-control-broadcast
        --incremental-streaming-output
        --stream-interval 20
        --dist-init-addr "127.0.0.1:$((PORT + 2000))"
        --ep-size "$EP_SIZE"
        --moe-a2a-backend megamoe
        --enable-deepseek-v4-fp4-indexer
        --disable-flashinfer-autotune
    )
    # DEP4 shards the model over half the node, so per-rank weights roughly
    # double and the weights-only floor rises above 0.9 (the engine reports a
    # minimum viable 0.9013 and refuses to start). Keep upstream's 0.95 there.
    # DEP8 has room for the lower value, which leaves mega-MoE workspace
    # headroom.
    if [ "$TP" -ge 8 ]; then
        # Mega-MoE's transient workspace lives OUTSIDE the static allocation and
        # needs a single ~7 GB contiguous block, so headroom must grow with
        # concurrency. Measured at conc 256: 0.835 (~42 GB free) runs; 0.93
        # (~16 GB free) and 0.95 (~11 GB free) both die with a CUDA OOM on one
        # DP rank, which then hangs the whole engine in the MLP-sync collective.
        MEM_FRACTION_STATIC=0.93
        if [ "$CONC" -ge 512 ]; then
            MEM_FRACTION_STATIC=0.86
        elif [ "$CONC" -ge 384 ]; then
            MEM_FRACTION_STATIC=0.88
        elif [ "$CONC" -ge 256 ]; then
            MEM_FRACTION_STATIC=0.9
        fi
    else
        # DEP4 is squeezed from both sides: weights occupy ~90% of each GPU when
        # the model is sharded over half the node, so the engine refuses to start
        # below ~0.902 (no KV left), while megamoe still needs its ~7 GB
        # workspace above the static budget. 0.95 leaves only ~11 GB free -- the
        # same margin that OOM'd a rank at DEP8 conc 256 -- so use 0.93, which
        # gives the ~16 GB that DEP8 conc 128 runs with at the same per-rank load
        # (max-running-requests/dp = 32 in both cases).
        MEM_FRACTION_STATIC=0.93
    fi
    # --chunked-prefill-size is a GLOBAL budget: server_args.py divides it by
    # dp_size, and dp_size is TP here. Scale it so every DEP shape gets the
    # per-rank 8192 that was tuned, rather than 16384/rank at DEP4 -- which
    # exceeds MegaMoE's per-rank token cap (a startup ValueError) and measured
    # slower at DEP8 when tried directly.
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
# Subagent fan-out means live requests exceed CONC (see MAX_RUNNING_REQUESTS
# above), so sizing decode graphs at CONC would drop every larger batch to
# eager decode. Capture past the fan-out; the runtime clamps this down to the
# request pool size anyway.
CUDA_GRAPH_MAX_BS=$((CONC * 4))
[ "$CUDA_GRAPH_MAX_BS" -gt 64 ] && CUDA_GRAPH_MAX_BS=64

# --cuda-graph-max-bs is an alias whose dest is cuda_graph_max_bs_decode, so the
# two forms below are the same knob and must not both be passed.
CUDA_GRAPH_ARGS=(--cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS")
SWA_FULL_TOKENS_RATIO=0.1
if [ "$DP_ATTENTION" = "true" ]; then
    # Decode graphs must cover the padded MTP batch across all DP ranks, which
    # exceeds CONC; capping at 64 would fall back to eager decode.
    CUDA_GRAPH_ARGS=(--cuda-graph-max-bs-decode 544)
    SWA_FULL_TOKENS_RATIO=0.075
fi

# Profiled runs execute eager so every device kernel keeps its operator link;
# setup_profiling_env must run before the server launch reads the profiler dir.
setup_profiling_env
if profiling_cuda_graph_disabled; then
    CUDA_GRAPH_ARGS=(--disable-cuda-graph)
fi

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
if [ "$DP_ATTENTION" = "true" ]; then
    # MegaMoE's FP4/MXF4 activation path is opt-in -- both flags default False,
    # so --moe-a2a-backend megamoe alone runs a different kernel than the one
    # measured. DG_USE_FP4_ACTS / DG_USE_MXF4_KIND are forwarded to DeepGEMM
    # automatically from these two.
    export SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS=1
    export SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND=1
    # Must cover the per-rank prefill budget (8192) or startup raises; the
    # extra 128 is headroom over the exact-fit boundary.
    export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320
fi
if [ "${EVAL_ONLY}" != "true" ]; then
    export SGLANG_SIMULATE_ACC_LEN=2.49
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
    --speculative-algorithm EAGLE
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
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
    # Profiling endpoints live on the backend server, not the DP router.
    export PROFILE_SERVER_URL="http://localhost:$SGLANG_BACKEND_PORT"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
