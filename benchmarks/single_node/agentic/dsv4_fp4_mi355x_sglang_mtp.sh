#!/usr/bin/env bash
set -eo pipefail
set -x

# Agentic trace replay benchmark for DeepSeek-V4-Pro FP4 on MI355X using SGLang
# with EAGLE/MTP speculative decoding.

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

# A server killed on this node minutes earlier (previous job, crashed run)
# can still be draining its HBM: KFD reclaim takes minutes, and booting into a
# half-drained node fails RCCL init with HIP 'unhandled cuda error' /
# 'invalid argument'. DeepSeek-V4-Pro is an 805 GiB checkpoint, so the drain
# window here is at the long end. Wait for the GPUs to come back before
# launching. Per-GPU threshold: idle nodes hold a small driver/firmware VRAM
# baseline (observed up to ~4%/GPU), while a draining or occupied GPU sits at
# 50-90%. Require every GPU <= 10%.
GPU_CLEAN=false
for i in $(seq 1 90); do
    VRAM_MAX=$(rocm-smi --showmemuse 2>/dev/null | grep -oE "GPU Memory Allocated \(VRAM%\): [0-9]+" | awk '{if ($NF > m) m = $NF} END {print m+0}')
    if [ "${VRAM_MAX:-0}" -le 10 ]; then echo "GPUs clean (vram%max=$VRAM_MAX after $((i*10))s)"; GPU_CLEAN=true; break; fi
    echo "waiting for prior-job GPU memory reclaim: vram%max=$VRAM_MAX"; sleep 10
done
[ "$GPU_CLEAN" = "true" ] || { echo "Error: GPUs still draining prior job's memory after 15min" >&2; exit 1; }

# ---- Resolve traces and install deps ----------------------------------------
resolve_trace_source
install_agentic_deps

SERVER_LOG="$RESULT_DIR/server.log"
ROUTER_LOG="$RESULT_DIR/router.log"
mkdir -p "$RESULT_DIR"

# ---- Client config ----------------------------------------------------------
export PYTHONNOUSERSITE=1
# Agentic warmup dispatches hundreds of large prompts at once; allow up to
# 15 minutes of TCP progress before AIPerf declares a connection dead.
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
# AIPerf pins one pooled keep-alive connection per session (client-side
# keep-alive 300s) while uvicorn's default SGLANG_TIMEOUT_KEEP_ALIVE is 5s;
# inter-turn idle gaps can reuse a socket exactly as the server closes it.
# Outlast the client pool so the race cannot occur.
export SGLANG_TIMEOUT_KEEP_ALIVE=900

# ---- DSv4 kernel routing / thinking mode ------------------------------------
# Mirrors the deleted spec-none sibling plus the DSv4 block in
# benchmarks/multi_node/amd_utils/env.sh. AgentX measures the thinking-on
# regime, which is also the golden-AL curve committed for this model.
export SGLANG_DEFAULT_THINKING=1
export SGLANG_DSV4_REASONING_EFFORT=high
export SGLANG_USE_ROCM700A=0
export SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton
export AITER_BF16_FP8_MOE_BOUND=0
export TORCH_BLAS_PREFER_HIPBLASLT=1
export HSA_NO_SCRATCH_RECLAIM=0
# aiter batched GEMM for the absorbed MLA projections, carried by the v0.5.18
# image and off by default in environ.py.
export SGLANG_OPT_USE_AITER_BATCHED_GEMM=1

# Unified radix tree: per-component (full-attn / SWA) cache management for
# hybrid-attention models, plus proactive release of out-of-window SWA KV
# slots during chunked prefill. Without the latter, in-flight requests pin SWA
# KV for their whole context and the trailing window of cached sessions gets
# flushed under LRU, collapsing the effective prefix-cache hit rate on
# multi-turn agentic workloads.
export SGLANG_ENABLE_UNIFIED_RADIX_TREE=1
export SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS=1

# ---- HiCache (host DRAM KV tier) --------------------------------------------
# Per-arm L2 sizing: host pinned memory is roughly
# HICACHE_RATIO * (per-rank device KV pool) * TP, which must stay under the
# node's ~2.7 TB of DRAM. The deleted spec-none sibling used ratio 4 with a
# smaller device pool; at TP8 with mem-fraction-static 0.85 that would
# oversubscribe host DRAM, so this recipe starts from 1.5 (the value validated
# on this cluster by glm5.2_fp4_mi355x_sglang_mtp.sh) and leaves every knob
# overridable for tuning.
CACHE_ARGS=()
if agentic_kv_offload_enabled; then
    case "$KV_OFFLOAD_BACKEND" in
        hicache)
            HICACHE_RATIO="${HICACHE_RATIO:-1.5}"
            HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_through}"
            HICACHE_IO_BACKEND="${HICACHE_IO_BACKEND:-direct}"
            HICACHE_MEM_LAYOUT="${HICACHE_MEM_LAYOUT:-page_first_direct}"
            echo "HiCache DSv4 CPU tier: ratio=$HICACHE_RATIO, write_policy=$HICACHE_WRITE_POLICY, io_backend=$HICACHE_IO_BACKEND, mem_layout=$HICACHE_MEM_LAYOUT, dram_budget=${TOTAL_CPU_DRAM_GB} GB, tp=$TP"
            CACHE_ARGS=(
                --enable-hierarchical-cache
                --hicache-ratio "$HICACHE_RATIO"
                --hicache-write-policy "$HICACHE_WRITE_POLICY"
                --hicache-io-backend "$HICACHE_IO_BACKEND"
                --hicache-mem-layout "$HICACHE_MEM_LAYOUT"
            )
            ;;
        *)
            echo "Error: unsupported KV_OFFLOAD_BACKEND '$KV_OFFLOAD_BACKEND' (expected: hicache)" >&2
            exit 1
            ;;
    esac
fi

# ---- Parallelism ------------------------------------------------------------
# NOTE: the DP-attention path below is currently DORMANT (no dp-attn arms in
# amd-master.yaml for this key). It is kept so a future arm can enable it
# without rebuilding the router plumbing: sglang-router fronts the DP ranks
# with consistent hashing on the AIPerf correlation id, keeping multi-turn
# sessions on the DP rank that holds their radix/hicache prefix.
USE_SGLANG_ROUTER=false
SGLANG_BACKEND_PORT="$PORT"
# Small prefill chunks interleave long-context agentic prefills across
# requests instead of letting one ~100K-token prefill monopolize the engine
# (the conc>=16 queue-saturation / decode-stall failure mode). 8192 = 32*256,
# a page-size multiple well under the dsv4 compressor kernel's uint16 token
# cap; same value the multi-node DeepSeek-V4-Pro-AgentX no_dp profile uses.
if [ "$TP" -eq 8 ]; then
    CHUNKED_PREFILL_SIZE=16384
elif [ "$TP" -eq 4 ]; then
    CHUNKED_PREFILL_SIZE=8192
else
    echo "Error: unsupported TP '$TP' (expected: 4 or 8)" >&2
    exit 1
fi
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.86}"
PARALLEL_ARGS=(--tensor-parallel-size "$TP")
SHARED_EXPERTS_ARGS=(--enforce-shared-experts-fusion)
SWA_FULL_TOKENS_RATIO="${SWA_FULL_TOKENS_RATIO:-0.10}"
export GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES:-2}"
if [ "$DP_ATTENTION" = "true" ]; then
    USE_SGLANG_ROUTER=true
    export AIPERF_HTTP_X_SMG_ROUTING_KEY_FROM_CORRELATION_ID=true
    SGLANG_BACKEND_PORT=$((PORT + 1))
    SGLANG_ROUTER_METRICS_PORT=$((PORT + 10000))
    SGLANG_ROUTER_CMD=(python3 -m sglang_router.launch_router)

    export SGLANG_SHARED_EXPERT_TP1=1
    export SGLANG_DP_SHARED_EXPERT_LOCAL=1
    export SGLANG_DP_USE_GATHERV=1
    export SGLANG_DP_USE_REDUCE_SCATTER=1
    export GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES_DP:-5}"
    SHARED_EXPERTS_ARGS=(--disable-shared-experts-fusion)
    SWA_FULL_TOKENS_RATIO="${SWA_FULL_TOKENS_RATIO_DP:-0.15}"

    # Chunked prefill is a whole-engine budget, so widen it by the DP degree.
    CHUNKED_PREFILL_SIZE=$((CHUNKED_PREFILL_SIZE * TP))
    PARALLEL_ARGS+=(
        --dp "$TP"
        --enable-dp-attention
        --enable-prefill-delayer
        --enable-two-batch-overlap
        --enable-dp-attention-local-control-broadcast
        --tokenizer-worker-num "$TP"
        --stream-interval 20
        --prefill-decode-interval 10
    )
fi

if [ "$EP_SIZE" -gt 1 ]; then
    PARALLEL_ARGS+=(--ep-size "$EP_SIZE")
fi

# AgentX concurrency counts live session trees, not individual requests.
# Subagent fan-out can push instantaneous request concurrency above CONC, so
# leave 2x headroom rather than clipping those bursts at the scheduler.
MAX_RUNNING_REQUESTS=$((2 * CONC))
[ "$MAX_RUNNING_REQUESTS" -gt 256 ] && MAX_RUNNING_REQUESTS=256
CUDA_GRAPH_MAX_BS=$MAX_RUNNING_REQUESTS
[ "$CUDA_GRAPH_MAX_BS" -gt 128 ] && CUDA_GRAPH_MAX_BS=128

# Saturation arms carry a larger in-flight working set than the 30-minute
# default warmup drain allows.
if [ "$CONC" -ge 32 ]; then
    export AGENTIC_WARMUP_GRACE_PERIOD=3600
fi

# ---- Speculative decoding ---------------------------------------------------
# DeepSeek-V4 ships a built-in MTP head, loaded through the EAGLE spec path
# with eagle-topk 1 (a single MTP chain); NOT NEXTN, whose V3/R1 loader
# crashes on the V4 architecture. Depth 3 matches the vLLM agentic sibling
# (dsv4-fp4-mi355x-vllm-agentic-mtp) and the fixed-seq-len SGLang MTP recipe.
SPEC_ARGS=(
    --speculative-algorithm EAGLE
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
)

# Throughput runs pin acceptance to the committed golden AL for this model,
# thinking mode, and draft length (golden_al_distribution/dsv4_mtp.yaml:
# thinking_on, 3 -> 2.49). Eval-only runs keep real target verification so
# accuracy stays meaningful.
if [ "${EVAL_ONLY:-false}" != "true" ]; then
    export SGLANG_SIMULATE_ACC_LEN=2.49
    export SGLANG_SIMULATE_ACC_METHOD=match-expected
    export SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
fi

# ---- Launch -----------------------------------------------------------------
# No --chat-template: the AgentX traces are tool-heavy, and
# chat_templates/deepseek_v4_thinking.jinja renders only system/user/assistant
# (tool definitions and role: tool messages are silently dropped, which would
# truncate prompts and distort ISL). The multi-node DeepSeek-V4-Pro-AgentX
# profile and the vLLM agentic sibling both serve DSv4 without an override.
SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$SGLANG_BACKEND_PORT"
    --trust-remote-code
    "${PARALLEL_ARGS[@]}"
    --attention-backend dsv4
    --enable-deepseek-v4-fp4-indexer
    --page-size 256
    --swa-full-tokens-ratio "$SWA_FULL_TOKENS_RATIO"
    --kv-cache-dtype fp8_e4m3
    "${SHARED_EXPERTS_ARGS[@]}"
    --tool-call-parser deepseekv4
    --reasoning-parser deepseek-v4
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
    --mem-fraction-static "$MEM_FRACTION_STATIC"
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS"
    "${SPEC_ARGS[@]}"
    "${CACHE_ARGS[@]}"
    # MTP draft-token forward passes under long-context agentic load block the
    # scheduler long enough to trip the 1800s watchdog mid-warmup.
    --watchdog-timeout 3600
    --enable-metrics
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

if [ "${EVAL_ONLY}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    REPLAY_CMD+=" --server-metrics http://localhost:$SGLANG_BACKEND_PORT/metrics"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
