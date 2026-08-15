#!/usr/bin/env bash
set -eo pipefail
set -x

# Agentic trace replay for Kimi-K3 (MXFP4) on B300: TP8 x DCP8, TokenspeedMLA,
# Mooncake as the external KV tier. Concurrency selects the arm:
#   conc <= 8   DSpark level 7, golden AL 3.84
#   conc 16     DSpark level 3, golden AL 3.00
#   conc >  16  no drafting
# Keep the arms disjoint in concurrency: exp-name carries conc and spec but not
# the arm, so a shared concurrency would collide.
#
# Required env vars:
#   MODEL, TP, CONC, KV_OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR, DURATION
# Optional:
#   DCP_SIZE (default 8), KV_OFFLOAD_BACKEND (mooncake, or empty for resident)
#
# TP8 is the only single-node layout: the MXFP4 checkpoint is ~1.5 TB, so TP4
# would need ~375 GB/GPU against B300's 288 GB.
#
# The draft's real acceptance on this corpus is 1.16-2.01 (13-41% position-1);
# its native window is 32k YaRN-stretched to 1M. Synthetic acceptance measures
# the system at a prescribed acceptance, not the draft's fitness at 100k+ ISL.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION

if [ "$TP" -ne 8 ]; then
    echo "Error: Kimi-K3 on B300 requires TP=8, got TP='$TP'" >&2
    exit 1
fi

if [[ -n "${EP_SIZE:-}" && "${EP_SIZE}" -gt 1 ]]; then
    echo "Error: this recipe ships the pure-TP8 profile; EP_SIZE='$EP_SIZE' is not wired" >&2
    exit 1
fi

# DCP shards decode KV across the TP ranks, so it must divide TP.
DCP_SIZE="${DCP_SIZE:-8}"
if [ $((TP % DCP_SIZE)) -ne 0 ]; then
    echo "Error: TP='$TP' must be divisible by DCP_SIZE='$DCP_SIZE'" >&2
    exit 1
fi
CP_ARGS=()
if [ "$DCP_SIZE" -gt 1 ]; then
    CP_ARGS+=(--decode-context-parallel-size "$DCP_SIZE" --dcp-comm-backend a2a)
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

DRAFT_MODEL="${DRAFT_MODEL:-Inferact/Kimi-K3-DSpark}"

# The draft must not land next to a pre-staged target: dirname(MODEL_PATH) is a
# read-only mount, so use the launcher's writable models dir.
if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
    DRAFT_MODEL_PATH="${WRITABLE_MODELS_DIR:-/data/models}/${DRAFT_MODEL##*/}"
    if [[ ! -d "$DRAFT_MODEL_PATH" || -z "$(ls -A "$DRAFT_MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$DRAFT_MODEL" --local-dir "$DRAFT_MODEL_PATH"
    fi
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
    hf download "$DRAFT_MODEL"
    DRAFT_MODEL_PATH="$DRAFT_MODEL"
fi
nvidia-smi

resolve_trace_source
install_agentic_deps

# ---- Serving environment ----------------------------------------------------
export VLLM_ALLREDUCE_USE_FLASHINFER=1
export VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION=1
export VLLM_USE_V2_MODEL_RUNNER=1
# These default to auto, which self-enables on B300; name them so the measured
# DCP a2a path is the one that runs.
export VLLM_USE_DIRECT_DCP_A2A=1
export VLLM_USE_DIRECT_DCP_Q_GATHER=1
export VLLM_USE_DIRECT_DCP_KV_GATHER=1
# ~1.5 TB of MXFP4 shards loads well past the default readiness window.
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_RPC_TIMEOUT=600000
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
export PYTHONNOUSERSITE=1
export TORCH_CUDA_ARCH_LIST=10.0
# Identical prefixes must hash to identical block keys run-to-run.
export PYTHONHASHSEED=42
# AIPerf reuses one pooled connection per session; outlast its idle gaps or a
# socket race aborts warmup.
export VLLM_HTTP_TIMEOUT_KEEP_ALIVE=900
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000

SERVER_LOG="$RESULT_DIR/server.log"
MOONCAKE_MASTER_LOG="$RESULT_DIR/mooncake_master.log"
mkdir -p "$RESULT_DIR"
MOONCAKE_MASTER_PID=""

cleanup() {
    if [[ -n "$MOONCAKE_MASTER_PID" ]]; then
        kill "$MOONCAKE_MASTER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ---- External KV tier -------------------------------------------------------
OFFLOAD_ARGS=()
case "${KV_OFFLOAD_BACKEND:-}" in
    "")
        require_agentic_kv_offload_none
        ;;
    mooncake)
        require_agentic_kv_offload_backend mooncake
        PER_RANK_GB=$((TOTAL_CPU_DRAM_GB / TP))
        MOONCAKE_VERSION=0.3.11.post1
        agentic_pip_install --quiet --no-cache-dir --no-deps \
            --force-reinstall "mooncake-transfer-engine-cuda13==$MOONCAKE_VERSION"
        python3 -c "from mooncake.store import MooncakeDistributedStore" >/dev/null

        MOONCAKE_MASTER_PORT=$((PORT + 12000))
        MOONCAKE_CONFIG_PATH="$RESULT_DIR/mooncake_config.json"
        # One rail for every rank. These nodes are rail-isolated, so two
        # different RNICs cannot reach each other even within a node, and the
        # embedded store's ranks are eight processes on one host.
        #
        # Chosen at runtime: mlx5_0 is down on some nodes (b300-016, b300-017),
        # and a hardcoded rail has no fallback -- topology discovery finds 0
        # HCAs and every rank dies in a 20-retry loop that reads like a store
        # problem. Order starts at mlx5_0 so a healthy node is unchanged.
        MOONCAKE_RAIL=""
        for _d in mlx5_0 mlx5_1 mlx5_2 mlx5_3 mlx5_4 mlx5_5 mlx5_8 mlx5_9 \
                  mlx5_10 mlx5_11 mlx5_16 mlx5_17 mlx5_20 mlx5_21 mlx5_22 mlx5_23; do
            if grep -q ACTIVE "/sys/class/infiniband/$_d/ports/1/state" 2>/dev/null; then
                MOONCAKE_RAIL="$_d"
                break
            fi
        done
        if [ -z "$MOONCAKE_RAIL" ]; then
            echo "Error: no active RDMA rail on $(hostname); Mooncake cannot initialise" >&2
            exit 1
        fi
        echo "Mooncake rail: $MOONCAKE_RAIL"

        cat > "$MOONCAKE_CONFIG_PATH" <<EOF
{
  "mode": "embedded",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "127.0.0.1:$MOONCAKE_MASTER_PORT",
  "global_segment_size": "${PER_RANK_GB}GB",
  "local_buffer_size": "4GB",
  "protocol": "rdma",
  "device_name": "$MOONCAKE_RAIL",
  "enable_offload": false
}
EOF
        export MOONCAKE_CONFIG_PATH
        export MC_GID_INDEX=3
        # Same-process transfers skip the transfer engine; off by default
        # whenever a non-TCP transport exists.
        export MC_STORE_MEMCPY=1
        export MC_ENABLE_DEST_DEVICE_AFFINITY=1
        export MC_SLICE_SIZE=1048576
        export MC_WORKERS_PER_CTX=4
        export WITH_NVIDIA_PEERMEM=0
        export VLLM_MOONCAKE_LOAD_RECV_THREADS=4

        echo "Starting Mooncake master on port $MOONCAKE_MASTER_PORT..."
        mooncake_master --port "$MOONCAKE_MASTER_PORT" \
            --eviction_high_watermark_ratio=0.95 \
            --eviction_ratio=0.10 \
            > "$MOONCAKE_MASTER_LOG" 2>&1 &
        MOONCAKE_MASTER_PID=$!
        sleep 2
        if ! kill -0 "$MOONCAKE_MASTER_PID" 2>/dev/null; then
            echo "Mooncake master died during startup." >&2
            cat "$MOONCAKE_MASTER_LOG" >&2
            exit 1
        fi

        OFFLOAD_ARGS=(
            --kv-transfer-config
            '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both","kv_load_failure_policy":"recompute","kv_connector_extra_config":{"load_async":true,"lookup_async":true,"enable_offload":false}}'
        )
        ;;
    *)
        echo "Error: unsupported KV_OFFLOAD_BACKEND='$KV_OFFLOAD_BACKEND' (expected empty or mooncake)" >&2
        exit 1
        ;;
esac

# ---- Speculative decoding ---------------------------------------------------
if [ "${SPEC_DECODING:-none}" != "mtp" ]; then
    echo "Error: this recipe expects spec-decoding=mtp for every arm, got '${SPEC_DECODING:-}'" >&2
    exit 1
fi
# Draft length by concurrency. The golden AL must track it, from the
# probabilistic curve in golden_al_distribution/.
if [ "$CONC" -le 8 ]; then
    NUM_SPEC_TOKENS=7
    SYNTHETIC_ACCEPT_LEN=3.84
elif [ "$CONC" -le 16 ]; then
    NUM_SPEC_TOKENS=3
    SYNTHETIC_ACCEPT_LEN=3.00
else
    NUM_SPEC_TOKENS=0
fi

SPEC_ARGS=()
if [ "$NUM_SPEC_TOKENS" -gt 0 ]; then
    # EVAL_ONLY needs real verification: synthetic acceptance commits drafts
    # regardless of target logits and would zero the eval score.
    if [ "${EVAL_ONLY:-false}" = "true" ]; then
        SPEC_CONFIG="{\"method\": \"dspark\", \"model\": \"$DRAFT_MODEL_PATH\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS, \"attention_backend\": \"TOKENSPEED_MLA\", \"draft_sample_method\": \"probabilistic\", \"rejection_sample_method\": \"block\"}"
    else
        SPEC_CONFIG="{\"method\": \"dspark\", \"model\": \"$DRAFT_MODEL_PATH\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS, \"attention_backend\": \"TOKENSPEED_MLA\", \"draft_sample_method\": \"probabilistic\", \"rejection_sample_method\": \"synthetic\", \"synthetic_acceptance_length\": $SYNTHETIC_ACCEPT_LEN}"
    fi
    SPEC_ARGS=(--speculative-config "$SPEC_CONFIG")
fi

MAX_NUM_SEQS=$((2 * CONC))

# 1 - this is the buffer for what is not sized against the budget: the cudagraph
# pool, the FlashInfer MoE workspace and fragmentation. Only c56 and c70 ran out
# of it (2.78 GiB wanted, 1.60 GiB free); the pool grows with concurrency, so
# lower points keep the default and their full KV cache.
if [ "$CONC" -ge 56 ]; then
    GPU_MEM_UTIL=0.90
else
    GPU_MEM_UTIL=0.92
fi

# Capture sizes: step * 1..min(max-num-seqs, 128), then the fixed powers of two
# above that. Sizes are tokens when drafting, sequences when not; the 128 caps
# the number of dense entries, not their value.
CAPTURE_STEP=$((1 + NUM_SPEC_TOKENS))
DENSE_COUNT=$MAX_NUM_SEQS
if [ "$DENSE_COUNT" -gt 128 ]; then
    DENSE_COUNT=128
fi
CAPTURE_SIZES=""
for ((n = 1; n <= DENSE_COUNT; n++)); do
    CAPTURE_SIZES+="${CAPTURE_SIZES:+,}$((n * CAPTURE_STEP))"
done
DENSE_MAX=$((DENSE_COUNT * CAPTURE_STEP))
for t in 64 128 256 512 1024 2048 4096 8192; do
    if [ "$t" -gt "$DENSE_MAX" ]; then
        CAPTURE_SIZES+=",$t"
    fi
done
COMPILATION_CONFIG="{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"cudagraph_capture_sizes\":[${CAPTURE_SIZES}]}"

echo "Starting vllm server..."

{ set +x; } 2>/dev/null
VLLM_CMD=(
    vllm serve "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --tensor-parallel-size "$TP"
    "${CP_ARGS[@]}"
    --max-num-seqs "$MAX_NUM_SEQS"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --max-num-batched-tokens 16384
    --trust-remote-code
    --language-model-only
    --load-format fastsafetensors
    --moe-backend auto
    --no-enable-flashinfer-autotune
    --enable-cumem-allocator
    --enable-prefix-caching
    --prefix-match-unit 128
    --kv-cache-dtype fp8
    --stream-interval 10
    --attention-backend TOKENSPEED_MLA
    --attention-config '{"mla_prefill_backend":"TRTLLM_RAGGED","use_prefill_query_quantization":true}'
    "${SPEC_ARGS[@]}"
    --compilation-config "$COMPILATION_CONFIG"
    --disable-uvicorn-access-log
    "${OFFLOAD_ARGS[@]}"
)
printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"
printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [ "${EVAL_ONLY}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
