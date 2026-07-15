#!/usr/bin/env bash
set -eo pipefail
set -x

# Agentic trace replay benchmark for DeepSeek-V4-Pro FP4 on B200 using vLLM.
# Mirrors the fixed-seq-len parallelism options (pure TP and DEP) so the
# agentic sweep can probe both interactivity and throughput regimes:
#   pure TP (DP_ATTENTION=false, EP_SIZE=1):  attention TP-sharded across
#       all $TP GPUs in a single engine. Lower TPOT, lower batch.
#   TP+EP   (DP_ATTENTION=false, EP_SIZE>1):  attention TP-sharded, MoE
#       experts EP-sharded within the TP group.
#   DEP     (DP_ATTENTION=true, EP_SIZE>1):   per-DP-rank attention with
#       experts EP-sharded across DP ranks (per the vLLM blog recipe).
#       Highest aggregate throughput at large CONC.
#
# Image is configured in nvidia-master.yaml. block_size=256,
# kv-cache-dtype=fp8, FLASHINFER_MLA_SPARSE_DSV4 attention with the FP4 indexer
# cache, FULL_DECODE_ONLY cudagraph capture, and (in EP tiers) mega-MoE backend.
#
# Required env vars:
#   MODEL, TP, CONC, KV_OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR
#
# Pure TP is GPU-resident (KV_OFFLOADING=none). DEP tiers offload KV to host
# DRAM: KV_OFFLOADING=dram requires KV_OFFLOAD_BACKEND=vllm-simple or mooncake.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE DP_ATTENTION

if [ -z "$DCP_SIZE" ]; then
    DCP_SIZE=1
fi
if [ -z "$PCP_SIZE" ]; then
    PCP_SIZE=1
fi
VLLM_CP_ARGS=()
if [ "$DCP_SIZE" -gt 1 ]; then
    VLLM_CP_ARGS+=(--decode-context-parallel-size "$DCP_SIZE")
fi
if [ "$PCP_SIZE" -gt 1 ]; then
    VLLM_CP_ARGS+=(--prefill-context-parallel-size "$PCP_SIZE")
fi

GPU_COUNT=$TP
if [[ ! "$GPU_COUNT" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: GPU_COUNT must be a positive integer, got '$GPU_COUNT'" >&2
    exit 1
fi
export GPU_COUNT

if [[ -n "$SLURM_JOB_ID" ]]; then
    echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

# `hf download` creates the target dir if missing and is itself idempotent.
# When MODEL_PATH is unset (stand-alone runs), fall back to the HF_HUB_CACHE
# Either way, MODEL_PATH is what the server is launched with.
if [[ -n "$MODEL_PATH" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi
nvidia-smi

# ---- Resolve traces and install deps ----------------------------------------
resolve_trace_source
install_agentic_deps

# vllm-project/router expands the one HTTP backend into one logical worker per
# DP rank and sends X-data-parallel-rank on forwarded requests. aiperf's
# X-Correlation-ID is stable for every turn of a conversation; alias it to the
# router's preferred X-Session-ID header.
USE_VLLM_ROUTER=false
VLLM_BACKEND_PORT="$PORT"
if [ "$DP_ATTENTION" = "true" ]; then
    USE_VLLM_ROUTER=true
    VLLM_BACKEND_PORT=$((PORT + 1))
    VLLM_ROUTER_VERSION=0.1.14
    VLLM_ROUTER_POLICY=consistent_hash
    VLLM_ROUTER_METRICS_PORT=$((PORT + 10000))
    export AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID=1
    agentic_pip_install --quiet "vllm-router==$VLLM_ROUTER_VERSION"
fi

# DeepSeek-V4-Pro weights are large; engine startup can exceed default 600s.
export VLLM_ENGINE_READY_TIMEOUT_S=3600

# vllm-project/vllm#43447 keeps local SWA prefix-cache tails sparsely, while
# vllm-project/vllm#44774 applies the same reachability policy to Mooncake's
# store mask. 32k matches the trace-replay tuning validated for this workload.
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=32768
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_RUST_FRONTEND=1
export VLLM_DSV4_MEGA_FP8_COMBINE=1
export VLLM_RPC_TIMEOUT=600000

# ---- Server config ----------------------------------------------------------
SERVER_LOG="$RESULT_DIR/server.log"
ROUTER_LOG="$RESULT_DIR/router.log"
MOONCAKE_MASTER_LOG="$RESULT_DIR/mooncake_master.log"
mkdir -p "$RESULT_DIR"

SERVER_PID=""
ROUTER_PID=""
MOONCAKE_MASTER_PID=""

OFFLOAD_ARGS=()
case "$KV_OFFLOAD_BACKEND" in
    "")
        require_agentic_kv_offload_none
        ;;
    vllm-simple)
        require_agentic_kv_offload_backend vllm-simple
        CPU_BYTES_PER_RANK=$(( TOTAL_CPU_DRAM_GB * 1000 * 1000 * 1000 / GPU_COUNT ))
        # Identical prefixes must hash to identical block keys across DP ranks.
        export PYTHONHASHSEED=42
        OFFLOAD_CONFIG=$(cat <<EOF
{
  "kv_connector": "SimpleCPUOffloadConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "cpu_bytes_to_use_per_rank": ${CPU_BYTES_PER_RANK},
    "lazy_offload": false,
    "enable_cross_layers_blocks": "true"
  }
}
EOF
)
        OFFLOAD_ARGS=(
            --kv-transfer-config
            "$OFFLOAD_CONFIG"
        )
        ;;
    mooncake)
        require_agentic_kv_offload_backend mooncake
        # Embedded mode contributes one segment per GPU rank to a shared
        # distributed store, so pre-divide the aggregate host-memory budget.
        PER_RANK_GB=$((TOTAL_CPU_DRAM_GB / GPU_COUNT))

        MOONCAKE_VERSION=0.3.11.post1
        agentic_pip_install --quiet --no-cache-dir --no-deps \
            --force-reinstall "mooncake-transfer-engine-cuda13==$MOONCAKE_VERSION"
        python3 -c "from mooncake.store import MooncakeDistributedStore" >/dev/null

        MOONCAKE_MASTER_PORT=$((PORT + 12000))
        MOONCAKE_CONFIG_PATH="$RESULT_DIR/mooncake_config.json"
        cat > "$MOONCAKE_CONFIG_PATH" <<EOF
{
  "mode": "embedded",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "127.0.0.1:$MOONCAKE_MASTER_PORT",
  "global_segment_size": "${PER_RANK_GB}GB",
  "local_buffer_size": "4GB",
  "protocol": "rdma",
  "device_name": "",
  "enable_offload": false
}
EOF
        export MOONCAKE_CONFIG_PATH
        export MC_ENABLE_DEST_DEVICE_AFFINITY=1
        # Identical prefixes must hash to identical store keys across DP ranks.
        export PYTHONHASHSEED=0
        export WITH_NVIDIA_PEERMEM=0
        export MC_SLICE_SIZE=1048576
        export MC_WORKERS_PER_CTX=4

        # Each rank contributes a separate segment. Evict early enough to
        # avoid an imbalanced rank exhausting its segment.
        MOONCAKE_EVICTION_HIGH_WATERMARK_RATIO=0.80
        MOONCAKE_EVICTION_RATIO=0.10
        # Mooncake's default 5s read lease is shorter than the observed
        # transfer latency for large DSv4 hybrid-KV loads on B200 TCP.
        MOONCAKE_KV_LEASE_TTL=60s

        echo "Starting Mooncake master on port $MOONCAKE_MASTER_PORT..."
        mooncake_master --port "$MOONCAKE_MASTER_PORT" \
            --eviction_high_watermark_ratio="$MOONCAKE_EVICTION_HIGH_WATERMARK_RATIO" \
            --eviction_ratio="$MOONCAKE_EVICTION_RATIO" \
            --default_kv_lease_ttl="$MOONCAKE_KV_LEASE_TTL" \
            > "$MOONCAKE_MASTER_LOG" 2>&1 &
        MOONCAKE_MASTER_PID=$!
        sleep 2
        if ! kill -0 "$MOONCAKE_MASTER_PID" 2>/dev/null; then
            echo "Mooncake master died during startup." >&2
            cat "$MOONCAKE_MASTER_LOG" >&2
            exit 1
        fi
        unset VLLM_USE_SIMPLE_KV_OFFLOAD
        OFFLOAD_ARGS=(
            --kv-transfer-config
            '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"load_async":true}}'
        )
        ;;
    *)
        echo "Error: unsupported B200 KV_OFFLOAD_BACKEND='$KV_OFFLOAD_BACKEND'" >&2
        exit 1
        ;;
esac

PARALLEL_ARGS=(--tensor-parallel-size "$TP" --data-parallel-size 1)
if [ "$DP_ATTENTION" = "true" ]; then
    PARALLEL_ARGS=(--tensor-parallel-size 1 --data-parallel-size "$TP")
fi

EP_ARGS=()
FAST_MOE_ARGS=()
if [ "$EP_SIZE" -gt 1 ]; then
    EP_ARGS=(--enable-expert-parallel)
    FAST_MOE_ARGS=(
        --moe-backend deep_gemm_amxf4_mega_moe
        --enable-ep-weight-filter
        --prefill-schedule-interval 16
    )
fi

# AgentX concurrency counts live session trees, not individual requests.
# Subagent fan-out can push instantaneous request concurrency above CONC, so
# leave 2x headroom rather than clipping those bursts at the scheduler.
MAX_NUM_SEQS=$((2 * CONC))

echo "Starting vllm server..."
export TORCH_CUDA_ARCH_LIST="10.0"
export PYTHONNOUSERSITE=1
export VLLM_FLOAT32_MATMUL_PRECISION=high

{ set +x; } 2>/dev/null
VLLM_CMD=(
    vllm serve "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$VLLM_BACKEND_PORT"
    --trust-remote-code
    --kv-cache-dtype fp8
    --block-size 256
    --max-model-len 1048576
    --gpu-memory-utilization 0.92
    --numa-bind
    --enable-cumem-allocator
    --no-enable-flashinfer-autotune
    --tokenizer-mode deepseek_v4
    --reasoning-parser deepseek_v4
    --attention-config '{"backend":"FLASHINFER_MLA_SPARSE_DSV4","use_prefill_query_quantization":true,"use_fp4_indexer_cache":true}'
    --no-disable-hybrid-kv-cache-manager
    --disable-uvicorn-access-log
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","mode":0}'
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-cudagraph-capture-size "$MAX_NUM_SEQS"
    "${PARALLEL_ARGS[@]}"
    "${VLLM_CP_ARGS[@]}"
    "${EP_ARGS[@]}"
    "${FAST_MOE_ARGS[@]}"
    "${OFFLOAD_ARGS[@]}"
)
printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"
printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$VLLM_BACKEND_PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [ "$USE_VLLM_ROUTER" = "true" ]; then
    echo "Starting native vLLM router on port $PORT for $TP DP ranks..."
    vllm-router \
        --worker-urls "http://localhost:$VLLM_BACKEND_PORT" \
        --policy "$VLLM_ROUTER_POLICY" \
        --intra-node-data-parallel-size "$TP" \
        --host 0.0.0.0 \
        --port "$PORT" \
        --prometheus-host 127.0.0.1 \
        --prometheus-port "$VLLM_ROUTER_METRICS_PORT" \
        --request-timeout-secs 14400 \
        --disable-retries > "$ROUTER_LOG" 2>&1 &
    ROUTER_PID=$!
    echo "Router PID: $ROUTER_PID"
    wait_for_server_ready --port "$PORT" --server-log "$ROUTER_LOG" --server-pid "$ROUTER_PID"
fi

# ---- Run benchmark ----------------------------------------------------------
build_replay_cmd "$RESULT_DIR"

run_agentic_replay_and_write_outputs "$RESULT_DIR"
