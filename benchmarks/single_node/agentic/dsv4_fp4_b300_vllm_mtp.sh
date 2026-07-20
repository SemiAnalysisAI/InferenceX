#!/usr/bin/env bash
set -eo pipefail
set -x

# Agentic trace replay benchmark for DeepSeek-V4-Pro FP4 on B300 using vLLM,
# with MTP speculative decoding (num_speculative_tokens=3): synthetic acceptance
# length 2.49 for throughput, real target verification for the EVAL_ONLY eval.
#
# Identical to dsv4_fp4_b300_vllm.sh (same image, engine args, offload, GPU
# topologies, and agentic aiperf rig) with exactly two MTP deltas:
#   --speculative-config: synthetic acceptance length 2.49 (throughput) vs real MTP (EVAL_ONLY); see the SPEC_CONFIG block
#   cudagraph capture sizes expressed in TOKENS (see the capture block below).
#
# Image is configured in nvidia-master.yaml. The recipe uses FP8 KV cache,
# sparse DeepSeek-V4 FlashInfer attention with an FP4 indexer cache, mega-MoE,
# and FULL_DECODE_ONLY CUDA graphs with every decode batch captured explicitly.
#
# Required env vars:
#   MODEL, TP, CONC, KV_OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR
#
# TP4, TP8, and DEP8 (TP8 + DP-attention) are GPU-resident (KV_OFFLOADING=none).
# DEP4 uses KV_OFFLOADING=dram with KV_OFFLOAD_BACKEND=vllm-simple or mooncake.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE DP_ATTENTION

DCP_SIZE="${DCP_SIZE:-1}"
PCP_SIZE="${PCP_SIZE:-1}"
VLLM_CP_ARGS=()
if [ "$DCP_SIZE" -gt 1 ]; then
    VLLM_CP_ARGS+=(--decode-context-parallel-size "$DCP_SIZE")
fi
if [ "$PCP_SIZE" -gt 1 ]; then
    VLLM_CP_ARGS+=(--prefill-context-parallel-size "$PCP_SIZE")
fi

GPU_COUNT=$((TP * PCP_SIZE))
if [[ ! "$GPU_COUNT" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: GPU_COUNT must be a positive integer, got '$GPU_COUNT'" >&2
    exit 1
fi
export GPU_COUNT

# Under DP-attention the DP world size equals TP, and the DEP recipe sizes
# per-rank batch as MAX_NUM_SEQS = 2*CONC/TP, which must be an integer.
if [ "$DP_ATTENTION" = "true" ] && [ $((2 * CONC % TP)) -ne 0 ]; then
    echo "Error: DEP requires 2*CONC divisible by TP, got CONC='$CONC' and TP='$TP'" >&2
    exit 1
fi

# DEP8 (TP8 + DP-attention) is a GPU-resident, high-concurrency arm that is
# tuned separately from the smaller DEP4 arm (larger prefill token budget,
# long-prefill chunking, and a lower GPU-memory-utilization headroom).
IS_DEP8=false
if [ "$DP_ATTENTION" = "true" ] && [ "$TP" -eq 8 ]; then
    IS_DEP8=true
fi

if [[ -n "$SLURM_JOB_ID" ]]; then
    echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

# `hf download` creates the target dir if missing and is itself idempotent.
# When MODEL_PATH is unset (stand-alone runs), fall back to the HF_HUB_CACHE.
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
# DP rank. Bind every turn of a conversation to the same rank by mapping
# AIPerf's stable correlation ID to the router's X-Session-ID header.
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

# Match the environment used by v4pro-b300.yaml.
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=32768
export VLLM_DSV4_MEGA_FP8_COMBINE=1
export NCCL_NVLS_ENABLE=1
export VLLM_USE_RUST_FRONTEND=1

# ---- Server config ----------------------------------------------------------
SERVER_LOG="$RESULT_DIR/server.log"
ROUTER_LOG="$RESULT_DIR/router.log"
MOONCAKE_MASTER_LOG="$RESULT_DIR/mooncake_master.log"
mkdir -p "$RESULT_DIR"

SERVER_PID=""
ROUTER_PID=""
MOONCAKE_MASTER_PID=""

# The generated TOTAL_CPU_DRAM_GB budget is proportional to allocated GPUs.
# On cluster:b300-nv, dram-utilization=0.80 and DEP4 resolve to roughly the
# source recipe's 280 GiB per DP rank. TP4 remains GPU-resident.
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
        # The plain-TP (non-DP-attention) offload ladder uses lazy offload;
        # DEP keeps eager offload for cross-rank block-hash stability.
        SIMPLE_LAZY_OFFLOAD=false
        if [ "$DP_ATTENTION" != "true" ]; then
            SIMPLE_LAZY_OFFLOAD=true
        fi
        OFFLOAD_CONFIG=$(cat <<EOF
{
  "kv_connector": "SimpleCPUOffloadConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "cpu_bytes_to_use": ${CPU_BYTES_PER_RANK},
    "enable_cross_layers_blocks": "true",
    "lazy_offload": ${SIMPLE_LAZY_OFFLOAD}
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
        # Embedded mode contributes one global segment per DP rank to the
        # shared store, so divide the aggregate host budget across ranks.
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

        # The store is shared, but each rank contributes a separate segment.
        # Start eviction before an imbalanced rank exhausts its segment, and
        # reclaim enough space for several concurrent multi-GB batch puts.
        MOONCAKE_EVICTION_HIGH_WATERMARK_RATIO=0.80
        MOONCAKE_EVICTION_RATIO=0.10

        echo "Starting Mooncake master on port $MOONCAKE_MASTER_PORT..."
        mooncake_master --port "$MOONCAKE_MASTER_PORT" \
            --eviction_high_watermark_ratio="$MOONCAKE_EVICTION_HIGH_WATERMARK_RATIO" \
            --eviction_ratio="$MOONCAKE_EVICTION_RATIO" \
            > "$MOONCAKE_MASTER_LOG" 2>&1 &
        MOONCAKE_MASTER_PID=$!
        sleep 2
        if ! kill -0 "$MOONCAKE_MASTER_PID" 2>/dev/null; then
            echo "Mooncake master died during startup." >&2
            cat "$MOONCAKE_MASTER_LOG" >&2
            exit 1
        fi

        unset VLLM_USE_SIMPLE_KV_OFFLOAD
        OFFLOAD_CONFIG='{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"load_async":true}}'
        OFFLOAD_ARGS=(--kv-transfer-config "$OFFLOAD_CONFIG")
        ;;
    *)
        echo "Error: unsupported B300 KV_OFFLOAD_BACKEND='$KV_OFFLOAD_BACKEND'" >&2
        exit 1
        ;;
esac

PARALLEL_ARGS=(--tensor-parallel-size "$TP" --data-parallel-size 1)
if [ "$DP_ATTENTION" = "true" ]; then
    PARALLEL_ARGS=(--tensor-parallel-size 1 --data-parallel-size "$TP")
fi

TP_ARGS=()
if [ "$DP_ATTENTION" = "true" ]; then
    export PYTORCH_ALLOC_CONF=expandable_segments:True
else
    export VLLM_ALLREDUCE_USE_FLASHINFER=1
    export VLLM_FLASHINFER_ALLREDUCE_BACKEND=auto
    TP_ARGS+=(--disable-custom-all-reduce)
fi

MODE_ARGS=()
if [ "$EP_SIZE" -gt 1 ]; then
    MODE_ARGS+=(
        --enable-expert-parallel
        --enable-ep-weight-filter
        --moe-backend deep_gemm_amxf4_mega_moe
    )
fi
if [ "$DP_ATTENTION" = "true" ]; then
    MODE_ARGS+=(--prefill-schedule-interval 8)
    if [ "$IS_DEP8" = "true" ]; then
        # GPU-resident DEP8 gets a larger prefill token budget and chunks long
        # prefills so decode latency stays bounded at high concurrency.
        MODE_ARGS+=(
            --max-num-batched-tokens 16384
            --long-prefill-token-threshold 4096
        )
    else
        MODE_ARGS+=(--max-num-batched-tokens 8192)
    fi
fi

if [ "$DP_ATTENTION" = "true" ]; then
    # The DEP source recipe enforces 2*CONC = DP_WORLD_SIZE*MAX_NUM_SEQS.
    MAX_NUM_SEQS=$((2 * CONC / TP))
else
    # Preserve the previous TP4 scheduler headroom for agentic fan-out.
    MAX_NUM_SEQS=$((2 * CONC))
fi
# MTP: cudagraph capture sizes are in TOKENS. With num_speculative_tokens=N,
# every uniform decode batch of S seqs verifies S*(1+N) tokens, so capture the
# explicit multiples (1+N), 2*(1+N), ..., MAX_NUM_SEQS*(1+N) -- one graph per
# decode batch of 1..MAX_NUM_SEQS seqs. vLLM rounds configured sizes up to
# multiples of (1+N) and dedups (adjust_cudagraph_sizes_for_spec_decode), so a
# plain 1..MAX_NUM_SEQS list would collapse to coverage of only
# MAX_NUM_SEQS/(1+N) seqs and drop the largest decode batches to eager.
NUM_SPEC_TOKENS=3
TOKENS_PER_SEQ=$((1 + NUM_SPEC_TOKENS))
# Throughput pins synthetic MTP acceptance to the dsv4-pro golden AL (thinking_on,
# num_speculative_tokens=3, golden_al_distribution/dsv4_mtp.yaml). The EVAL_ONLY
# accuracy run uses real target verification instead -- synthetic acceptance
# bypasses verification and corrupts the SWE-bench eval (0.0000 score).
if [ "${EVAL_ONLY:-false}" = "true" ]; then
    SPEC_CONFIG="{\"method\": \"mtp\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS}"
else
    SPEC_CONFIG="{\"method\": \"mtp\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS, \"rejection_sample_method\": \"synthetic\", \"synthetic_acceptance_length\": 2.49}"
fi
CUDA_GRAPH_CAPTURE_SIZES=""
for ((num_seqs = 1; num_seqs <= MAX_NUM_SEQS; num_seqs++)); do
    if [ -n "$CUDA_GRAPH_CAPTURE_SIZES" ]; then
        CUDA_GRAPH_CAPTURE_SIZES+=","
    fi
    CUDA_GRAPH_CAPTURE_SIZES+="$((num_seqs * TOKENS_PER_SEQ))"
done
COMPILATION_CONFIG="{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[${CUDA_GRAPH_CAPTURE_SIZES}],\"mode\":0}"

echo "Starting vllm server..."
export TORCH_CUDA_ARCH_LIST="10.0"
export PYTHONNOUSERSITE=1
export VLLM_FLOAT32_MATMUL_PRECISION=high

# DEP8 leaves more headroom for its larger prefill token budget; all other
# topologies (TP4/DEP4/TP8) use 0.95.
GPU_MEM_UTIL=0.95
if [ "$IS_DEP8" = "true" ]; then
    GPU_MEM_UTIL=0.92
fi

{ set +x; } 2>/dev/null
VLLM_CMD=(
    vllm serve "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$VLLM_BACKEND_PORT"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --trust-remote-code
    --no-enable-flashinfer-autotune
    --no-disable-hybrid-kv-cache-manager
    --max-num-seqs "$MAX_NUM_SEQS"
    --kv-cache-dtype fp8
    --block-size 256
    --max-model-len 1048576
    --attention-config '{"use_fp4_indexer_cache":true,"backend":"FLASHINFER_MLA_SPARSE_DSV4","use_prefill_query_quantization":true}'
    --speculative-config "$SPEC_CONFIG"
    --disable-uvicorn-access-log
    --tokenizer-mode deepseek_v4
    --tool-call-parser deepseek_v4
    --enable-auto-tool-choice
    --reasoning-parser deepseek_v4
    --compilation-config "$COMPILATION_CONFIG"
    "${PARALLEL_ARGS[@]}"
    "${VLLM_CP_ARGS[@]}"
    "${TP_ARGS[@]}"
    "${MODE_ARGS[@]}"
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

if [ "${EVAL_ONLY}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
