#!/usr/bin/env bash
set -eo pipefail
set -x
 
source "$(dirname "$0")/../../benchmark_lib.sh"

 export EVAL_FRAMEWORK="lm-eval"
 
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
# can still be draining its ~1.4 TB of HBM: KFD reclaim takes minutes, and
# booting into a half-drained node fails RCCL init with HIP 'unhandled cuda
# error' / 'invalid argument' (observed as the mooncake-c64 CI failure).
# Wait for the GPUs to come back before launching.
# Per-GPU threshold: idle nodes hold a small driver/firmware VRAM baseline
# (observed up to ~4%/GPU, node-dependent), while a draining or occupied
# GPU sits at 50-90%. Require every GPU <= 10%.
GPU_CLEAN=false
for i in $(seq 1 90); do
    VRAM_MAX=$(rocm-smi --showmemuse 2>/dev/null | grep -oE "GPU Memory Allocated \(VRAM%\): [0-9]+" | awk '{if ($NF > m) m = $NF} END {print m+0}')
    if [ "${VRAM_MAX:-0}" -le 10 ]; then echo "GPUs clean (vram%max=$VRAM_MAX after $((i*10))s)"; GPU_CLEAN=true; break; fi
    echo "waiting for prior-job GPU memory reclaim: vram%max=$VRAM_MAX"; sleep 10
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
# Optional quick-reduce quantisation (F-2 probe). Set in runner .env to activate;
# leave unset for baseline behaviour (falls back to NCCL/custom-AR).
#   ROCM_QUICK_REDUCE_QUANTIZATION=INT4  — INT4 quick-reduce (tensor >= 16 MB for TP4/bf16)
#   ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16=1 — use faster fp16 kernels on ROCm (recommended with INT4)
[[ -n "${ROCM_QUICK_REDUCE_QUANTIZATION:-}" ]]    && export ROCM_QUICK_REDUCE_QUANTIZATION
[[ -n "${ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16:-}" ]] && export ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16

# HiCache L2 (host DRAM), optionally extended with Mooncake L3.
# KV_OFFLOADING=dram requires KV_OFFLOAD_BACKEND=hicache or mooncake.
#
# Per-arm L2 ratio (sizing rationale below) applies to both backends unless
# overridden via HICACHE_RATIO. TP arm (182.7 GB/rank device pool): the
# agentic-coding corpus saturates any fixed DRAM pool at conc ≥ 10; ratio 1.5
# (~2.9 TB pinned) is the safe default for cluster:mi355x-amds nodes (~3.0 TB
# available DRAM per runners.yaml). ratio=2.5 (~4.8 TB) yields higher
# throughput at conc 10-12 but exceeds physical DRAM on these nodes and must
# be set via HICACHE_RATIO env-var override on nodes that can accommodate it.
# The DP-attention arm (159.4 GB/rank) only runs at conc >= 32, where the host
# tier just absorbs overflow - ratio 0.5 (~1.2 TB pinned, ~1.8 TB of load
# headroom) at negligible hit-rate cost (ratio 1.5 OOMs the host mid-storm at
# conc 48).
CACHE_ARGS=()
if agentic_kv_offload_enabled; then
    if [ "$DP_ATTENTION" = "true" ]; then
        HICACHE_RATIO="${HICACHE_RATIO:-0.5}"
    else
        # ratio=1.5 (~2.9 TB pinned): safe default within the ~3.0 TB DRAM
        # available on cluster:mi355x-amds nodes. Set HICACHE_RATIO=2.5 via
        # env-var override for maximum throughput on nodes with >4 TB DRAM.
        HICACHE_RATIO="${HICACHE_RATIO:-1.5}"
    fi
    # write_through_selective skips DRAM writes for non-reusable KV blocks,
    # reducing host-bus traffic without affecting the cache hit rate.
    HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_through_selective}"
    HICACHE_IO_BACKEND="${HICACHE_IO_BACKEND:-direct}"
    HICACHE_MEM_LAYOUT="${HICACHE_MEM_LAYOUT:-page_first_direct}"
    case "$KV_OFFLOAD_BACKEND" in
        hicache)
            echo "HiCache (GPU+host DRAM only): ratio=$HICACHE_RATIO, write_policy=$HICACHE_WRITE_POLICY, io_backend=$HICACHE_IO_BACKEND, mem_layout=$HICACHE_MEM_LAYOUT"
            CACHE_ARGS=(
                --enable-hierarchical-cache
                --hicache-ratio "$HICACHE_RATIO"
                --hicache-write-policy "$HICACHE_WRITE_POLICY"
                --hicache-io-backend "$HICACHE_IO_BACKEND"
                --hicache-mem-layout "$HICACHE_MEM_LAYOUT"
            )
            ;;
        mooncake)
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
            ;;
        *)
            echo "Error: unsupported KV_OFFLOAD_BACKEND '$KV_OFFLOAD_BACKEND' (expected: hicache or mooncake)" >&2
            exit 1
            ;;
    esac
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
MEM_FRACTION_STATIC=0.85
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
    # Chunked prefill 32k: smaller chunks let the scheduler interleave decode
    # steps between prefill chunks, reducing TPOT for concurrent sessions
    # (improved interactivity vs the original 131072-token chunk). The reduced
    # chunk size drops per-chunk activation headroom from ~7 GiB/rank to
    # ~1.7 GiB/rank, so mem-fraction 0.85 is safe (0.85 OOMed at 131k:
    # "Tried to allocate 6.86 GiB ... 5.15 GiB is free", run 29751563205).
    CHUNKED_PREFILL_SIZE=32768
    MEM_FRACTION_STATIC=0.85
else
    CHUNKED_PREFILL_SIZE=32768
    export AGENTIC_WARMUP_GRACE_PERIOD=3600
fi
# 2×CONC in-flight slots: MTP draft+verify transiently batches more tokens
# than CONC sessions; headroom prevents scheduler stalls under burst.
MAX_RUNNING_REQUESTS=$((2 * CONC))
[ "$MAX_RUNNING_REQUESTS" -gt 256 ] && MAX_RUNNING_REQUESTS=256
# SGLang interpolates a bs list [1..max_bs] automatically; cap at 64 to
# keep graph-capture memory bounded without giving up coverage.
CUDA_GRAPH_MAX_BS=$(( MAX_RUNNING_REQUESTS < 64 ? MAX_RUNNING_REQUESTS : 64 ))

if [ "${EVAL_ONLY:-false}" != "true" ]; then
    export SGLANG_SIMULATE_ACC_LEN=3.61
    export SGLANG_SIMULATE_ACC_METHOD=match-expected
    export SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
fi
 
SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$SGLANG_BACKEND_PORT"
    --trust-remote-code
    "${PARALLEL_ARGS[@]}"
    --kv-cache-dtype fp8_e4m3
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
    --speculative-algorithm EAGLE
    --speculative-num-steps 5
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 6
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
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    REPLAY_CMD+=" --server-metrics http://localhost:$SGLANG_BACKEND_PORT/metrics"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
