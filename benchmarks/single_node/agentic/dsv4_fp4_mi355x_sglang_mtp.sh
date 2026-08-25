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
UMBP_MASTER_LOG="$RESULT_DIR/umbp_master.log"
mkdir -p "$RESULT_DIR"

SERVER_PID=""
ROUTER_PID=""
UMBP_MASTER_PID=""

# The UMBP master owns fixed ports on the node; a leaked one breaks the next
# job's bind. Tear down every background service we started, in reverse order.
cleanup_services() {
    local exit_code=$?
    trap - EXIT INT TERM
    set +e
    stop_background_process_tree "$ROUTER_PID" "SGLang router"
    stop_background_process_tree "$SERVER_PID" "SGLang server" 60
    stop_background_process_tree "$UMBP_MASTER_PID" "UMBP master"
    exit "$exit_code"
}
trap cleanup_services EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

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
        umbp|mori)
            # HiCache L2 (host DRAM) + UMBP/MoRI as the L3 storage tier, in
            # UMBP "distributed" mode with a node-local master. Ported from the
            # validated single-node recipe in /apps/billhe/umbp_recipe
            # (rocm/sgl-dev v0.5.17-rocm720-mi35x-20260813, upstream sglang
            # PR #30762 / a34f81251f, which supplies umbp_store.py's
            # batch_*_v2 multi-pool path).
            #
            # Two single-node-specific requirements from that bring-up:
            #
            # 1. UMBP_DISABLE_ZERO_COPY_REGISTER=true is mandatory. UMBPStore
            #    pre-registers each side pool's whole host KV buffer for RDMA
            #    at startup, but PoolClient does not reuse that MR and calls
            #    RegisterRdmaMemoryRegionAuto again on the identical
            #    (ptr, size) during the first batch_put_from_ptr. The second
            #    ibv_reg_mr returns ENOMEM and mori aborts the whole server
            #    from C++ (not catchable in Python). Skipping the
            #    pre-registration hands registration to PoolClient exactly
            #    once; the cost is the staging-buffer path.
            # 2. hicache-ratio must be much smaller than the 1P1D disagg
            #    recipe's 3. All 8 ranks share one host here, and ibv_reg_mr
            #    pins + populates every page, so ratio 3 means a 243 GB/rank
            #    deepseek_v4_c4 side pool (1.94 TB across the node) on top of
            #    the 806 GB checkpoint. Ratio 1 measures ~88 GB/rank
            #    (c4 75.5 + c128 9.7 + swa 2.4), i.e. ~704 GB node-wide.
            #    DSv4 rejects --hicache-size outright
            #    ("DeepSeek V4 HiCache currently does not support
            #    --hicache-size"), so ratio is the only knob.
            HICACHE_RATIO="${HICACHE_RATIO:-1}"
            HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_through}"
            HICACHE_IO_BACKEND="${HICACHE_IO_BACKEND:-direct}"
            # page_first, not the hicache-only arm's page_first_direct: the
            # mori storage backend consumes the page-first host layout.
            HICACHE_MEM_LAYOUT="${HICACHE_MEM_LAYOUT:-page_first}"
            HICACHE_PREFETCH_POLICY="${HICACHE_PREFETCH_POLICY:-best_effort}"

            # MoRI IO engine (UMBP's RDMA transport) and process-level knobs.
            export MORI_IO_QP_MAX_SEND_WR="${MORI_IO_QP_MAX_SEND_WR:-32767}"
            export MORI_IO_SQ_BACKOFF_TIMEOUT_US="${MORI_IO_SQ_BACKOFF_TIMEOUT_US:-500000}"
            export MORI_SHMEM_MODE="${MORI_SHMEM_MODE:-ISOLATION}"
            export MORI_SHMEM_HEAP_SIZE="${MORI_SHMEM_HEAP_SIZE:-1G}"
            export UMBP_DISABLE_ZERO_COPY_REGISTER="${UMBP_DISABLE_ZERO_COPY_REGISTER:-true}"
            # These nodes run with HugePages_Total=0, and
            # UMBPHostTensorAllocator silently demotes to 4 KiB pages. Keep it
            # off explicitly; to enable, reserve pages before the server starts
            # (echo N > /proc/sys/vm/nr_hugepages) and set this to 1.
            export UMBP_DRAM_USE_HUGEPAGES="${UMBP_DRAM_USE_HUGEPAGES:-0}"

            # Port plan. Every family gets a 16-port stride keyed off this
            # runner's PORT so the per-rank io_engine / peer_service ports
            # (base + rank, rank < TP) of one runner cannot land on another's.
            UMBP_PORT_STRIDE=$(( (PORT % 100) * 16 ))
            UMBP_GRPC_PORT="${UMBP_GRPC_PORT:-$((21000 + UMBP_PORT_STRIDE))}"
            UMBP_HTTP_PORT="${UMBP_HTTP_PORT:-$((23000 + UMBP_PORT_STRIDE))}"
            UMBP_IO_ENGINE_PORT="${UMBP_IO_ENGINE_PORT:-$((25000 + UMBP_PORT_STRIDE))}"
            UMBP_PEER_SERVICE_PORT="${UMBP_PEER_SERVICE_PORT:-$((27000 + UMBP_PORT_STRIDE))}"
            UMBP_NODE_ADDR="${UMBP_NODE_ADDR:-127.0.0.1}"

            # L3 DRAM pool, sized per UMBP client (one client per TP rank).
            # Node accounting: L2 pinned (~88 GiB/rank at ratio 1) + L3
            # (UMBP_L3_PER_RANK_GB/rank) must stay inside the configured
            # TOTAL_CPU_DRAM_GB budget, alongside the 806 GB checkpoint's page
            # cache. 64 GiB/rank = 512 GiB node-wide leaves ample headroom.
            # The pool is sized in GiB; TOTAL_CPU_DRAM_GB from the matrix is
            # decimal GB, so convert before comparing them.
            UMBP_L3_PER_RANK_GB="${UMBP_L3_PER_RANK_GB:-96}"
            UMBP_L3_TOTAL_GB=$((UMBP_L3_PER_RANK_GB * TP * 1073741824 / 1000000000))
            if [ "$UMBP_L3_TOTAL_GB" -gt "$((TOTAL_CPU_DRAM_GB / 2))" ]; then
                echo "Error: UMBP L3 pool ${UMBP_L3_TOTAL_GB} GB (${UMBP_L3_PER_RANK_GB} GB x TP${TP}) exceeds half of TOTAL_CPU_DRAM_GB=${TOTAL_CPU_DRAM_GB}; the HiCache L2 host pool needs the rest" >&2
                exit 1
            fi
            UMBP_DRAM_BYTES=$((UMBP_L3_PER_RANK_GB * 1024 * 1024 * 1024))

            # umbp_master is not on the default rpath; point the loader at
            # mori's .so directory the same way the recipe does.
            MORI_HOME="${MORI_HOME:-/sgl-workspace/mori}"
            UMBP_MASTER_BIN="${UMBP_MASTER_BIN:-$MORI_HOME/build/src/umbp/umbp_master}"
            if [ ! -x "$UMBP_MASTER_BIN" ]; then
                echo "Error: umbp_master not found at '$UMBP_MASTER_BIN'. This image does not ship a built MoRI/UMBP; set MORI_HOME or UMBP_MASTER_BIN, or use KV_OFFLOAD_BACKEND=hicache." >&2
                exit 1
            fi
            export LD_LIBRARY_PATH="$MORI_HOME/python/mori:${LD_LIBRARY_PATH:-}"

            echo "Starting UMBP master (grpc=$UMBP_GRPC_PORT, http=$UMBP_HTTP_PORT)..."
            "$UMBP_MASTER_BIN" "0.0.0.0:${UMBP_GRPC_PORT}" "$UMBP_HTTP_PORT" > "$UMBP_MASTER_LOG" 2>&1 &
            UMBP_MASTER_PID=$!
            UMBP_MASTER_READY=false
            for _ in $(seq 1 60); do
                if ! kill -0 "$UMBP_MASTER_PID" 2>/dev/null; then
                    echo "UMBP master died during startup. Log follows:" >&2
                    cat "$UMBP_MASTER_LOG" >&2 || true
                    exit 1
                fi
                if curl -s --max-time 2 -o /dev/null "http://127.0.0.1:${UMBP_HTTP_PORT}/metrics"; then
                    UMBP_MASTER_READY=true
                    break
                fi
                sleep 1
            done
            [ "$UMBP_MASTER_READY" = "true" ] || { echo "Error: UMBP master metrics endpoint never came up on :$UMBP_HTTP_PORT" >&2; cat "$UMBP_MASTER_LOG" >&2 || true; exit 1; }
            echo "UMBP master PID: $UMBP_MASTER_PID"

            # cache_remote_fetches=false: a fetch served from a peer client is
            # not re-cached locally. ssd_enabled=false keeps L3 DRAM-only.
            UMBP_EXTRA_CONFIG=$(cat <<JSON
{"dram_capacity_bytes": ${UMBP_DRAM_BYTES},
 "ssd_enabled": false,
 "master_address": "127.0.0.1:${UMBP_GRPC_PORT}",
 "node_address": "${UMBP_NODE_ADDR}",
 "io_engine_port": "${UMBP_IO_ENGINE_PORT}",
 "peer_service_port": "${UMBP_PEER_SERVICE_PORT}",
 "cache_remote_fetches": ${UMBP_CACHE_REMOTE_FETCHES:-false}}
JSON
)
            echo "HiCache+UMBP DSv4: ratio=$HICACHE_RATIO, write_policy=$HICACHE_WRITE_POLICY, io_backend=$HICACHE_IO_BACKEND, mem_layout=$HICACHE_MEM_LAYOUT, prefetch=$HICACHE_PREFETCH_POLICY, l3_per_rank=${UMBP_L3_PER_RANK_GB} GiB (${UMBP_L3_TOTAL_GB} GB decimal node-wide), dram_budget=${TOTAL_CPU_DRAM_GB} GB, tp=$TP"
            CACHE_ARGS=(
                --enable-hierarchical-cache
                --hicache-ratio "$HICACHE_RATIO"
                --hicache-write-policy "$HICACHE_WRITE_POLICY"
                --hicache-io-backend "$HICACHE_IO_BACKEND"
                --hicache-mem-layout "$HICACHE_MEM_LAYOUT"
                --hicache-storage-prefetch-policy "$HICACHE_PREFETCH_POLICY"
                --hicache-storage-backend mori
                --hicache-storage-backend-extra-config "$UMBP_EXTRA_CONFIG"
                # Surfaces usage.prompt_tokens_details.cached_tokens, the only
                # per-request signal that a prefix came back from L2/L3.
                --enable-cache-report
            )
            ;;
        *)
            echo "Error: unsupported KV_OFFLOAD_BACKEND '$KV_OFFLOAD_BACKEND' (expected: hicache or umbp)" >&2
            exit 1
            ;;
    esac
fi

# Snapshot the UMBP master's Prometheus output so an L3 delta (objects,
# capacity, RPC counts) can be reconstructed from the artifacts after the run.
# No-op unless the UMBP arm is active.
snapshot_umbp_metrics() {
    [ -n "$UMBP_MASTER_PID" ] || return 0
    curl -s --max-time 10 "http://127.0.0.1:${UMBP_HTTP_PORT}/metrics" \
        > "$RESULT_DIR/umbp_master_metrics_$1.txt" 2>/dev/null || true
}

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
CHUNKED_PREFILL_SIZE=8192
# MTP adds a draft KV pool and extra graph captures on top of the spec-none
# footprint, which ran at 0.90. 0.89 recovers most of that: the DSv4 compressor
# state pools are sized from the full-attention pool and allocated after it,
# outside this budget, so the remainder has to stay large enough to cover them.
MEM_FRACTION_STATIC=0.89
PARALLEL_ARGS=(--tensor-parallel-size "$TP")
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
    export GPU_MAX_HW_QUEUES=5

    # Chunked prefill is a whole-engine budget, so widen it by the DP degree.
    CHUNKED_PREFILL_SIZE=$((8192 * TP))
    PARALLEL_ARGS+=(
        --dp "$TP"
        --enable-dp-attention
        --enable-prefill-delayer
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
    --page-size 256
    --swa-full-tokens-ratio 0.10
    --kv-cache-dtype fp8_e4m3
    --enforce-shared-experts-fusion
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
    snapshot_umbp_metrics before
    build_replay_cmd "$RESULT_DIR"
    REPLAY_CMD+=" --server-metrics http://localhost:$SGLANG_BACKEND_PORT/metrics"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
    snapshot_umbp_metrics after
fi
