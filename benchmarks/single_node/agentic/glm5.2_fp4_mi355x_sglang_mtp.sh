#!/usr/bin/env bash
set -eo pipefail
set -x

# Agentic trace replay benchmark for GLM-5.2 MXFP4 on MI355X using SGLang with
# EAGLE/MTP speculative decoding.
#
# Spec-decode only, per the AgentX policy that agentic recipes are run and
# published with speculative decoding enabled rather than as an STP/MTP A/B
# (MODELS.md: GLM-5.2 agentic non-MTP is deprecated after 2026-08-03). The
# non-MTP arm is neither wired into the master config nor kept as a separate
# script.
#
# Sibling of agentic/glm5.2_fp4_b300_sglang_mtp.sh (NVFP4/B300): same nextn
# head, same draft length, same golden AL; the serve flags below are the ROCm
# ones (tilelang DSA prefill/decode, MXFP4 checkpoint, HiCache or Mooncake host
# offload).
#
# Required env vars:
#   MODEL, TP, CONC, KV_OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR, DURATION,
#   EP_SIZE, DP_ATTENTION

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
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
export SGLANG_TIMEOUT_KEEP_ALIVE=900
export SGLANG_OPT_USE_TOPK_V2=false

# AgentX pins acceptance to the committed golden AL so submissions are compared
# on system performance at a fixed acceptance target rather than on draft-head
# quality (golden_al_distribution/README.md). 2.99 is the GLM-5.2 curve at
# num_speculative_tokens=3, thinking_on (SPEED-Bench coding, speedbench-al.yml
# run 28058352479); the curve is committed as
# golden_al_distribution/glm5.2_mtp.yaml by the B300 sibling recipe (#2447).
# One curve per model: it was collected on the FP8 checkpoint, and the MXFP4
# checkpoint ships the same nextn head.
#
# SGLANG_SIMULATE_ACC_TOKEN_MODE is only read from SGLang v0.5.16 onward --
# ACC_LEN / ACC_METHOD exist further back, so an older image would silently
# honor two thirds of the contract. The pinned image is v0.5.16-rocm720.
#
# EVAL_ONLY leaves simulated acceptance off: it commits drafted tokens
# regardless of the target logits, so generated text is wrong and the eval
# would score ~0.
if [ "${EVAL_ONLY:-false}" != "true" ]; then
    export SGLANG_SIMULATE_ACC_LEN=2.99
    export SGLANG_SIMULATE_ACC_METHOD=match-expected
    export SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
fi

CACHE_ARGS=()
if agentic_kv_offload_enabled; then
    if [ "$DP_ATTENTION" = "true" ]; then
        HICACHE_RATIO="${HICACHE_RATIO:-0.5}"
    else
        HICACHE_RATIO="${HICACHE_RATIO:-1.5}"
    fi
    HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_through}"
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

# MTP: GLM-5.2 ships its own nextn head, so SGLang EAGLE runs off the
# checkpoint with no external draft model. num-steps 3 / eagle-topk 1 /
# num-draft-tokens 4 is 3 speculative tokens per verification step -- the same
# shape as the B300 sibling and the GLM-5.2 GB300 dynamo-sglang agentic
# recipes, and the draft length whose golden AL is pinned above.
#
# The draft MoE backend is left to SGLang: GLM-5.2's nextn layer is unquantized
# bf16, so it cannot inherit the target model's MXFP4 MoE runner once expert
# parallelism puts an all-to-all in the path. Upstream's
# _deepseek_spec_moe_resolution fixes that up automatically, and its hook is
# gated on is_hip() -- i.e. it fires here, unlike on the CUDA sibling where the
# same pair has to be passed explicitly.
SPEC_ARGS=(
    --speculative-algorithm EAGLE
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
)

USE_SGLANG_ROUTER=false
SGLANG_BACKEND_PORT="$PORT"
PARALLEL_ARGS=(--tp "$TP" --ep-size "$EP_SIZE")
# The nextn layer's weights and its own KV pool come out of the same static
# budget as the target model, and CUDA-graph capture is over 4-token
# verification batches rather than 1-token decodes. Hold the whole grid at the
# conservative 0.80 instead of taking the +0.05 the non-speculative serve shape
# could afford at conc <= 16: an OOM mid-warmup costs the whole sweep point,
# and HiCache's host tier (ratio relative to the device pool) absorbs the
# slightly smaller HBM KV pool.
MEM_FRACTION_STATIC=0.80
if [ "$DP_ATTENTION" = "true" ]; then
    USE_SGLANG_ROUTER=true
    export AIPERF_HTTP_X_SMG_ROUTING_KEY_FROM_CORRELATION_ID=true
    SGLANG_BACKEND_PORT=$((PORT + 1))
    SGLANG_ROUTER_METRICS_PORT=$((PORT + 10000))
    SGLANG_ROUTER_CMD=(python3 -m sglang_router.launch_router)
    PARALLEL_ARGS+=(--dp "$TP" --enable-dp-attention)
    CHUNKED_PREFILL_SIZE=32768
    export AGENTIC_WARMUP_GRACE_PERIOD=3600
    export SGLANG_DP_USE_GATHERV=1
    export SGLANG_DP_USE_REDUCE_SCATTER=1
    export GPU_MAX_HW_QUEUES=5
elif [ "$CONC" -le 16 ]; then
    CHUNKED_PREFILL_SIZE=131072
else
    CHUNKED_PREFILL_SIZE=32768
    export AGENTIC_WARMUP_GRACE_PERIOD=3600
fi
MAX_RUNNING_REQUESTS=$((1 * CONC))
[ "$MAX_RUNNING_REQUESTS" -gt 256 ] && MAX_RUNNING_REQUESTS=256
# --cuda-graph-max-bs counts requests, not verification tokens: SGLang's
# spec-decode graph runner scales each captured batch by
# --speculative-num-draft-tokens itself.
CUDA_GRAPH_MAX_BS=$MAX_RUNNING_REQUESTS

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
    --tool-call-parser glm47
    --reasoning-parser glm45
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
    --mem-fraction-static "$MEM_FRACTION_STATIC"
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS"
    "${SPEC_ARGS[@]}"
    "${CACHE_ARGS[@]}"
    --watchdog-timeout 1800
    --enable-metrics
)

printf '%q ' "${SGLANG_CMD[@]}" | tee "$RESULT_DIR/sglang_command.txt"
printf '\n' | tee -a "$RESULT_DIR/sglang_command.txt"

{
    echo "=== SGLANG_SIMULATE_ACC_* env vars at launch (empty => real verification) ==="
    env | grep -E '^SGLANG_SIMULATE_ACC_' | sort || true
    echo "============================================================================"
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
    export SWEBENCH_AGENT_STEP_LIMIT=150
    export SWEBENCH_AGENT_WORKERS="${SWEBENCH_AGENT_WORKERS:-32}"
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    REPLAY_CMD+=" --server-metrics http://localhost:$SGLANG_BACKEND_PORT/metrics"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
