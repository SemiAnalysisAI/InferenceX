#!/usr/bin/env bash
set -eo pipefail
set -x

# Agentic trace replay benchmark for GLM-5.2 FP8 on MI355X using SGLang.
#
# Server flags follow the SGLang cookbook MI355X FP8 single-node recipes
# (https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.2): TP8 with the
# DSA tilelang prefill/decode backends, no MTP (spec decoding is not yet
# validated on ROCm gfx950). The cookbook's low-latency / balanced /
# high-throughput strategies differ only in batch-shaping levers
# (chunked-prefill / mem-fraction / graph bs / max-running), which this
# script derives from CONC so one search-space arm traces the full frontier.
#
# Required env vars:
#   MODEL, TP, CONC, KV_OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR, DURATION,
#   EP_SIZE, DP_ATTENTION
#
# KV_OFFLOADING=dram requires KV_OFFLOAD_BACKEND=hicache.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE DP_ATTENTION

if [[ "$DP_ATTENTION" = "true" ]]; then
    echo "Error: DP-attention is not part of the GLM-5.2 MI355X cookbook recipe" >&2
    exit 1
fi

if [[ -n "$SLURM_JOB_ID" ]]; then
    echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

# ROCR/HIP visibility under slurm cgroups.
if [ -n "$ROCR_VISIBLE_DEVICES" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

# Weights are pre-staged in the NFS HF hub cache (launch_mi355x-amds.sh mounts
# /it-share/hf-hub-cache for this model); a warm cache makes this a no-op.
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

resolve_trace_source
install_agentic_deps

SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

# KV offloading: all backends layer on HiCache, which extends RadixAttention
# by spilling prefixes evicted from the HBM KV pool to a pinned host pool
# instead of recomputing them. GLM-5.2 is DSA-family, so every rank holds
# complete per-token KV (124.54 GB / 1.335M tokens per rank at mem-fraction
# 0.85, replicated on all 8 ranks) and the L2 host pool is sized through the
# host/device token-capacity ratio like the B300 sibling and DSv4 recipes,
# NOT a GB-based --hicache-size (which pins TOTAL_CPU_DRAM_GB/TP per rank
# and OOMs the node - see the B300 #2279 notes). Write policy / IO backend /
# memory layout follow the same-cluster dsv4-fp4-mi355x-sglang-agentic-hicache
# recipe (the ROCm-proven combo for DSA KV pools on this image).
#
#   hicache  - L2 host pool only, ratio 1.5 (~1.5 TB pinned on the 3 TB
#              nodes; the B300 sibling caps at 0.75 against its 170 GB
#              device pool, and the MI355X pool is a third smaller, so 1.5
#              doubles host token capacity in the same headroom class).
#   mooncake - L2 at ratio 0.5 (~0.5 TB staging tier) + Mooncake store as
#              the HiCache L3 storage backend: local mooncake_master,
#              embedded per-rank segments over TCP (single node - transfers
#              are host-local), 40% of the DRAM budget (~1 TB), keeping
#              total pinned memory in the same ~1.5 TB envelope.
#   mori     - L2 at ratio 0.5 + AMD MORI UMBP (Unified Memory Buffer Pool)
#              as the HiCache L3 storage backend: standalone in-process
#              per-rank DRAM pools (BUILD_UMBP=ON ships in this image),
#              same 40%-of-budget sizing, SSD tier disabled so the arm
#              measures the DRAM path against mooncake like-for-like.
CACHE_ARGS=()
if agentic_kv_offload_enabled; then
    HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_through}"
    HICACHE_IO_BACKEND="${HICACHE_IO_BACKEND:-direct}"
    HICACHE_MEM_LAYOUT="${HICACHE_MEM_LAYOUT:-page_first_direct}"
    # 40% of the node DRAM budget for the L3 tier, split per rank.
    L3_PER_RANK_GB=$((TOTAL_CPU_DRAM_GB * 2 / 5 / TP))
    case "$KV_OFFLOAD_BACKEND" in
        hicache)
            HICACHE_RATIO="${HICACHE_RATIO:-1.5}"
            ;;
        mooncake)
            HICACHE_RATIO="${HICACHE_RATIO:-0.5}"
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
            ;;
        mori)
            HICACHE_RATIO="${HICACHE_RATIO:-0.5}"
            python3 -c "import mori.umbp" >/dev/null
            ;;
        *)
            echo "Error: unsupported KV_OFFLOAD_BACKEND '$KV_OFFLOAD_BACKEND' (expected: hicache, mooncake, mori)" >&2
            exit 1
            ;;
    esac
    echo "HiCache tier: backend=$KV_OFFLOAD_BACKEND, ratio=$HICACHE_RATIO, dram_budget=${TOTAL_CPU_DRAM_GB} GB, l3_per_rank=${L3_PER_RANK_GB} GB, write_policy=$HICACHE_WRITE_POLICY, io_backend=$HICACHE_IO_BACKEND, mem_layout=$HICACHE_MEM_LAYOUT"
    CACHE_ARGS=(
        --enable-hierarchical-cache
        --hicache-ratio "$HICACHE_RATIO"
        --hicache-write-policy "$HICACHE_WRITE_POLICY"
        --hicache-io-backend "$HICACHE_IO_BACKEND"
        --hicache-mem-layout "$HICACHE_MEM_LAYOUT"
    )
    if [[ "$KV_OFFLOAD_BACKEND" == "mooncake" ]]; then
        CACHE_ARGS+=(--hicache-storage-backend mooncake)
    elif [[ "$KV_OFFLOAD_BACKEND" == "mori" ]]; then
        CACHE_ARGS+=(
            --hicache-storage-backend mori
            --hicache-storage-backend-extra-config
            "{\"dram_capacity_bytes\": $((L3_PER_RANK_GB * 1024 * 1024 * 1024)), \"ssd_enabled\": false}"
        )
    fi
fi

# Cookbook batch-shaping by concurrency band:
#   low-latency  (conc <= 16): chunked-prefill 131072, mem-fraction 0.80
#   balanced/high-throughput (conc >= 32): chunked-prefill 32768, mem 0.85
# AgentX concurrency counts live session trees, not individual requests, so
# max-running-requests is 2*CONC (subagent fan-out headroom); the CUDA-graph
# range covers it up to the cookbook high-throughput cap of 256.
if [ "$CONC" -le 16 ]; then
    CHUNKED_PREFILL_SIZE=131072
    MEM_FRACTION_STATIC=0.80
else
    CHUNKED_PREFILL_SIZE=32768
    MEM_FRACTION_STATIC=0.85
    # MI355X prefill is slow relative to the 1M-context agentic corpus; give
    # the warmup drain the same extended grace as the B300 saturation arm.
    export AGENTIC_WARMUP_GRACE_PERIOD=3600
fi
MAX_RUNNING_REQUESTS=$((2 * CONC))
CUDA_GRAPH_MAX_BS=$MAX_RUNNING_REQUESTS
[ "$CUDA_GRAPH_MAX_BS" -gt 256 ] && CUDA_GRAPH_MAX_BS=256

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
# from CUDA-only source (cooperative_groups.h) and cannot build for gfx950,
# killing decode CUDA-graph capture. v1 dispatches to the precompiled HIP op
# in sgl-kernel instead; upstream's own MI355X CI runs DSA models this way
# (scripts/ci/slurm/launch_mi355x.sh).
export SGLANG_OPT_USE_TOPK_V2=false

SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    --tp "$TP"
    --ep-size "$EP_SIZE"
    --dsa-prefill-backend tilelang
    --dsa-decode-backend tilelang
    # GLM-5.2 emits the GLM-4.7-style tool-call format; glm47 is required for
    # structured message.tool_calls (SWE-bench agentic evals die without it).
    # The glm45 reasoning parser keeps hybrid thinking in reasoning_content.
    --tool-call-parser glm47
    --reasoning-parser glm45
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
    --mem-fraction-static "$MEM_FRACTION_STATIC"
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

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [ "${EVAL_ONLY}" = "true" ]; then
    # GLM-5.2's chat template defaults to reasoning_effort=Max when the
    # client passes no chat_template_kwargs (mini-swe-agent doesn't), and the
    # heavy thinking burns the default 75-step budget before submission
    # (observed on the B300 sibling recipe: 12/23 trajectories exited
    # LimitsExceeded unsubmitted while 10 of the 11 that submitted resolved).
    # Double the step budget for this recipe; others keep the shared default.
    export SWEBENCH_AGENT_STEP_LIMIT=150
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    REPLAY_CMD+=" --server-metrics http://localhost:$PORT/metrics"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
