#!/usr/bin/env bash
set -eo pipefail
set -x

# Agentic trace replay benchmark for GLM-5.2 MXFP4 (amd/GLM-5.2-MXFP4) on
# MI355X using SGLang.
#
# Motivation over the FP8 sibling (glm5.2_fp8_mi355x_sglang.sh): the FP8
# frontier is prefill-starved (TTFT p50 ~390 s at conc 16) because the ROCm
# DSA path defaults to bf16 KV and TP-mode DSA replicates the KV pool on
# every rank - 1.335M tokens aggregate against a 131k-token mean corpus.
# MXFP4 halves the weight bytes (438 GB vs 756 GB), growing the device pool
# to 1.96M tokens/rank (182.70 GB), and the DP-attention arm un-replicates
# it: each DP rank holds only its own sessions, so aggregate capacity is
# ~15.7M tokens at dp8 - enough to hold the conc-64 working set.
#
# Both arms run HiCache with the Mooncake store as the L3 storage tier
# (wait_complete prefetch), with per-arm L2 ratios (see the CACHE_ARGS
# block). Ratio 2 from the cookbook OOMs the 3 TB hosts outright: the main
# host pool alone would pin 2 x 182.7 GB x 8 = 2.9 TB, and GLM-5.2's DSA
# hicache additionally allocates a per-rank indexer-K sidecar host pool
# (40.33 GB/rank at ratio 2; observed "Requesting 40.33 GB but only have
# 0.87 GB free").
#
# Required env vars:
#   MODEL, TP, CONC, KV_OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR, DURATION,
#   EP_SIZE, DP_ATTENTION
#
# KV_OFFLOADING=dram requires KV_OFFLOAD_BACKEND=mooncake.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE DP_ATTENTION

if [[ -n "$SLURM_JOB_ID" ]]; then
    echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

# ROCR/HIP visibility under slurm cgroups.
if [ -n "$ROCR_VISIBLE_DEVICES" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

# Weights are pre-staged in the NFS HF hub cache (launch_mi355x-amds.sh mounts
# /it-share/hf-hub-cache for GLM-5.2 models); a warm cache makes this a no-op.
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
# can still be draining its ~1.4 TB of HBM: KFD reclaim takes minutes, and
# booting into a half-drained node fails RCCL init with HIP 'unhandled cuda
# error' / 'invalid argument' (observed as the mooncake-c64 CI failure).
# Wait for the GPUs to come back before launching.
GPU_CLEAN=false
for i in $(seq 1 90); do
    VRAM_BUSY=$(rocm-smi --showmemuse 2>/dev/null | grep -oE "GPU Memory Allocated \(VRAM%\): [0-9]+" | awk '{s+=$NF} END {print s+0}')
    if [ "${VRAM_BUSY:-0}" -le 8 ]; then echo "GPUs clean (vram%sum=$VRAM_BUSY after $((i*10))s)"; GPU_CLEAN=true; break; fi
    echo "waiting for prior-job GPU memory reclaim: vram%sum=$VRAM_BUSY"; sleep 10
done
[ "$GPU_CLEAN" = "true" ] || { echo "Error: GPUs still draining prior job's memory after 15min" >&2; exit 1; }

resolve_trace_source
install_agentic_deps

SERVER_LOG="$RESULT_DIR/server.log"
ROUTER_LOG="$RESULT_DIR/router.log"
mkdir -p "$RESULT_DIR"

export PYTHONNOUSERSITE=1
# Agentic warmup dispatches hundreds of large prompts at once; allow up to
# 15 minutes of TCP progress before AIPerf declares a connection dead.
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
# AIPerf pins one pooled keep-alive connection per session (client-side
# keep-alive 300s) while uvicorn's default SGLANG_TIMEOUT_KEEP_ALIVE is 5s;
# inter-turn idle gaps can reuse a socket exactly as the server closes it.
# Outlast the client pool so the race cannot occur.
export SGLANG_TIMEOUT_KEEP_ALIVE=900
# The DSA indexer's top-k v2 kernel (default since v0.5.14) is JIT-compiled
# from CUDA-only source (cooperative_groups.h) and cannot build for gfx950;
# v1 dispatches to the precompiled HIP op in sgl-kernel (upstream MI355X CI
# runs DSA models the same way).
export SGLANG_OPT_USE_TOPK_V2=false

# HiCache L2 + Mooncake L3 on every point (sizing rationale in the header).
# Per-arm L2 ratio, both measured on-node. TP arm (182.7 GB/rank device
# pool): the working set oversubscribes the device pool ~3x at conc 32, so
# the host tier is what carries the radix hits - ratio 1.5 (~2.9 TB pinned
# incl. sidecars) validates through the conc-24 long-context storm. The
# DP-attention arm (159.4 GB/rank) only runs at conc >= 32, where each DP
# rank's ~8 sessions nearly fit in its own device pool (~1.5-1.6M of 1.7M
# tokens at conc 64) and the host tier just absorbs overflow - ratio 1.5
# boots but the host OOM killer takes the server mid-storm at conc 48, so
# it runs ratio 0.5 (~1.2 TB pinned, ~1.8 TB of load headroom) at
# negligible hit-rate cost.
CACHE_ARGS=()
if require_agentic_kv_offload_backend mooncake; then
    if [ "$DP_ATTENTION" = "true" ]; then
        HICACHE_RATIO="${HICACHE_RATIO:-0.5}"
    else
        HICACHE_RATIO="${HICACHE_RATIO:-1.5}"
    fi
    HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_through}"
    HICACHE_IO_BACKEND="${HICACHE_IO_BACKEND:-direct}"
    HICACHE_MEM_LAYOUT="${HICACHE_MEM_LAYOUT:-page_first_direct}"
    L3_PER_RANK_GB="${L3_PER_RANK_GB:-40}"
    python3 -c "from mooncake.store import MooncakeDistributedStore" >/dev/null
    MOONCAKE_MASTER_PORT=$((PORT + 12000))
    MOONCAKE_MASTER_LOG="$RESULT_DIR/mooncake_master.log"
    MOONCAKE_CONFIG_PATH="$RESULT_DIR/mooncake_config.json"
    cat > "$MOONCAKE_CONFIG_PATH" <<EOF
{
  "local_hostname": "127.0.0.1",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "127.0.0.1:$MOONCAKE_MASTER_PORT",
  "global_segment_size": "${L3_PER_RANK_GB}gb",
  "local_buffer_size": "4gb",
  "protocol": "tcp",
  "device_name": ""
}
EOF
    export SGLANG_HICACHE_MOONCAKE_CONFIG_PATH="$MOONCAKE_CONFIG_PATH"
    mooncake_master --port "$MOONCAKE_MASTER_PORT" \
        --default_kv_lease_ttl=120s \
        --eviction_high_watermark_ratio=0.80 \
        --eviction_ratio=0.10 > "$MOONCAKE_MASTER_LOG" 2>&1 &
    MOONCAKE_MASTER_PID=$!
    sleep 2
    kill -0 "$MOONCAKE_MASTER_PID"
    echo "HiCache+Mooncake: ratio=$HICACHE_RATIO, l3_per_rank=${L3_PER_RANK_GB} GB, dram_budget=${TOTAL_CPU_DRAM_GB} GB"
    CACHE_ARGS=(
        --enable-hierarchical-cache
        --hicache-ratio "$HICACHE_RATIO"
        --hicache-size 0
        --hicache-write-policy "$HICACHE_WRITE_POLICY"
        --hicache-io-backend "$HICACHE_IO_BACKEND"
        --hicache-mem-layout "$HICACHE_MEM_LAYOUT"
        --hicache-storage-backend mooncake
        --hicache-storage-prefetch-policy wait_complete
    )
fi

# Arm selection. TP arm keeps the FP8 sibling's cookbook batch-shaping
# bands.
#
# NOTE: the DP-attention path below is currently DORMANT (no dp-attn arms
# in amd-master.yaml): DSA + dp-attention hangs a collective under
# long-context prefill on ROCm v0.5.14 (watchdog kills the scheduler with
# zero completions; reproduced with and without HiCache, with and without
# the DSv4 DP collective envs; short prompts are fine). Re-enable the
# config arm once upstream fixes the DSA DP prefill path.
#
# When active, the DP-attention (DEP) arm fronts the DP ranks with sglang-router
# using consistent hashing on the AIPerf correlation id so multi-turn
# sessions stay on the DP rank holding their radix/hicache prefix, and
# widens chunked-prefill (whole-engine, /dp ranks) like the B300 sibling.
USE_SGLANG_ROUTER=false
SGLANG_BACKEND_PORT="$PORT"
PARALLEL_ARGS=(--tp "$TP" --ep-size "$EP_SIZE")
if [ "$DP_ATTENTION" = "true" ]; then
    USE_SGLANG_ROUTER=true
    export AIPERF_HTTP_X_SMG_ROUTING_KEY_FROM_CORRELATION_ID=true
    SGLANG_BACKEND_PORT=$((PORT + 1))
    SGLANG_ROUTER_METRICS_PORT=$((PORT + 10000))
    SGLANG_ROUTER_CMD=(python3 -m sglang_router.launch_router)
    PARALLEL_ARGS+=(--dp "$TP" --enable-dp-attention)
    CHUNKED_PREFILL_SIZE=32768
    export AGENTIC_WARMUP_GRACE_PERIOD=3600
    # Swap the DP gather collectives to gatherv/reduce-scatter on ROCm
    # (dsv4_fp4_mi355x_sglang.sh precedent - the only green DP-attention
    # config on this cluster/image): with the defaults the DSA DP path
    # hangs a collective under long-context prefill load until the
    # watchdog kills the scheduler (0/96 storm completions, twice).
    export SGLANG_DP_USE_GATHERV=1
    export SGLANG_DP_USE_REDUCE_SCATTER=1
    export GPU_MAX_HW_QUEUES=5
elif [ "$CONC" -le 16 ]; then
    CHUNKED_PREFILL_SIZE=131072
else
    CHUNKED_PREFILL_SIZE=32768
    export AGENTIC_WARMUP_GRACE_PERIOD=3600
fi
MAX_RUNNING_REQUESTS=$((2 * CONC))
[ "$MAX_RUNNING_REQUESTS" -gt 256 ] && MAX_RUNNING_REQUESTS=256
CUDA_GRAPH_MAX_BS=$MAX_RUNNING_REQUESTS

SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$SGLANG_BACKEND_PORT"
    --trust-remote-code
    "${PARALLEL_ARGS[@]}"
    --dsa-prefill-backend tilelang
    --dsa-decode-backend tilelang
    # GLM-5.2 emits the GLM-4.7-style tool-call format; glm47 is required for
    # structured message.tool_calls (SWE-bench agentic evals die without it).
    # The glm45 reasoning parser keeps hybrid thinking in reasoning_content.
    --tool-call-parser glm47
    --reasoning-parser glm45
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
    --mem-fraction-static 0.85
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS"
    "${CACHE_ARGS[@]}"
    --watchdog-timeout 1800
    --enable-metrics
)

printf '%q ' "${SGLANG_CMD[@]}" | tee "$RESULT_DIR/sglang_command.txt"
printf '\n' | tee -a "$RESULT_DIR/sglang_command.txt"

echo "Starting SGLang server for MI355X..."
"${SGLANG_CMD[@]}" > "$SERVER_LOG" 2>&1 &
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
    # GLM-5.2's chat template defaults to reasoning_effort=Max when the
    # client passes no chat_template_kwargs (mini-swe-agent doesn't), and the
    # heavy thinking burns the default 75-step budget before submission.
    # Double the step budget for this recipe; others keep the shared default.
    export SWEBENCH_AGENT_STEP_LIMIT=150
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    REPLAY_CMD+=" --server-metrics http://localhost:$SGLANG_BACKEND_PORT/metrics"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
