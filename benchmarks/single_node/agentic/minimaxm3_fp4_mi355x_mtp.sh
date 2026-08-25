#!/usr/bin/env bash
set -eo pipefail
set -x

# Agentic trace replay benchmark for MiniMax-M3 FP4 on MI355X using vLLM
# and EAGLE3 speculative decoding.
#
# Required env vars:
#   MODEL, MODEL_PATH, TP, CONC, KV_OFFLOADING, KV_OFFLOAD_BACKEND,
#   TOTAL_CPU_DRAM_GB, RESULT_DIR, DURATION, EP_SIZE, DP_ATTENTION

source "$(dirname "$0")/../../benchmark_lib.sh"

export EVAL_FRAMEWORK="lm-eval"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE DP_ATTENTION

echo "MODEL=$MODEL TP=$TP CONC=$CONC KV_OFFLOADING=$KV_OFFLOADING TOTAL_CPU_DRAM_GB=$TOTAL_CPU_DRAM_GB RESULT_DIR=$RESULT_DIR DURATION=$DURATION EP_SIZE=$EP_SIZE DP_ATTENTION=$DP_ATTENTION"

DRAFT_MODEL="Inferact/MiniMax-M3-EAGLE3-GQA"
NUM_SPEC_TOKENS=3
# golden_al_distribution/minimaxm3_eagle3_gqa.yaml:
# minimax-m3.thinking_on[3]
SYNTHETIC_ACCEPT_LEN=2.78

if [[ -v SLURM_JOB_ID ]]; then
    echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

# ROCR/HIP visibility for vLLM 0.14+
if [[ -v ROCR_VISIBLE_DEVICES ]]; then
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

hf download "$DRAFT_MODEL"

rocm-smi || true
amd-smi || true

resolve_trace_source
install_agentic_deps

# Require the vLLM Prometheus stream in every official result. AIPerf
# deduplicates this endpoint against its automatic localhost discovery.
export AIPERF_SERVER_METRICS_URLS="http://localhost:${PORT}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="vllm:"

# ---- Server config ----------------------------------------------------------
SERVER_LOG="$RESULT_DIR/server.log"
LMCACHE_LOG="$RESULT_DIR/lmcache_server.log"
mkdir -p "$RESULT_DIR"

SERVER_PID=""
LMCACHE_PIDS=()
cleanup_agentic_services() {
    local exit_code=$?
    trap - EXIT INT TERM
    set +e
    stop_background_process_tree "$SERVER_PID" "vLLM server" 60
    local i
    for i in "${!LMCACHE_PIDS[@]}"; do
        stop_background_process_tree "${LMCACHE_PIDS[$i]}" "LMCache server $i"
    done
    exit "$exit_code"
}
trap cleanup_agentic_services EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

OFFLOAD_ARGS=()

case "$KV_OFFLOAD_BACKEND" in
    "")
        require_agentic_kv_offload_none
        ;;
    lmcache)
        require_agentic_kv_offload_backend lmcache

        # Keep the image's tested torch/ROCm stack and install only LMCache's
        # missing pure-Python runtime dependencies.
        LMCACHE_VERSION="0.5.3"
        LMCACHE_ROCM_INDEX="https://github.com/LMCache/LMCache/releases/expanded_assets/v${LMCACHE_VERSION}-rocm"
        agentic_pip_install --quiet --no-cache-dir --no-deps \
            "sortedcontainers==2.4.0" \
            "opentelemetry-exporter-prometheus==0.61b0" \
            "cupy-rocm-7-0==14.1.1" \
            "lmcache==${LMCACHE_VERSION}" --find-links "$LMCACHE_ROCM_INDEX"
        python3 -c \
            "import cupy; import lmcache.integration.vllm.lmcache_mp_connector; import opentelemetry.exporter.prometheus" \
            >/dev/null

        # Split the node-level DRAM limit evenly across one MP server per TP rank.
        LMCACHE_N_SERVERS="$TP"
        LMCACHE_L1_SIZE_GB="$TOTAL_CPU_DRAM_GB"
        SHM_FREE_GB=$(df -BG --output=avail /dev/shm 2>/dev/null | tail -1 | tr -dc '0-9')
        if [ -n "$SHM_FREE_GB" ] && [ "$SHM_FREE_GB" -gt 0 ]; then
            SHM_CAP_GB=$((SHM_FREE_GB * 90 / 100))
            if [ "$LMCACHE_L1_SIZE_GB" -gt "$SHM_CAP_GB" ]; then
                echo "Error: LMCache L1 ${LMCACHE_L1_SIZE_GB} GB exceeds 90% of free /dev/shm (${SHM_CAP_GB} GB)." >&2
                exit 1
            fi
        fi
        LMCACHE_L1_SHARD_GB=$((LMCACHE_L1_SIZE_GB / LMCACHE_N_SERVERS))
        if [ "$LMCACHE_L1_SHARD_GB" -lt 1 ]; then
            echo "Error: LMCache DRAM budget is less than 1 GB per TP rank." >&2
            exit 1
        fi

        LMCACHE_SERVER_URLS=()
        LMCACHE_HTTP_PORTS=()
        LMCACHE_LOGS=()
        : > "$RESULT_DIR/lmcache_command.txt"
        for shard in $(seq 0 $((LMCACHE_N_SERVERS - 1))); do
            shard_port=$((5555 + shard))
            shard_http_port=$((8080 + shard))
            shard_log="${LMCACHE_LOG%.log}_${shard}.log"
            LMCACHE_CMD=(
                lmcache server
                --host 127.0.0.1
                --port "$shard_port"
                --http-host 127.0.0.1
                --http-port "$shard_http_port"
                --l1-size-gb "$LMCACHE_L1_SHARD_GB"
                --l1-init-size-gb 10
                --l1-read-ttl-seconds 7200
                --chunk-size 256
                --max-workers 2
                --eviction-policy LRU
                --supported-transfer-mode lmcache_driven
            )
            append_command "$RESULT_DIR/lmcache_command.txt" "${LMCACHE_CMD[@]}"
            "${LMCACHE_CMD[@]}" > "$shard_log" 2>&1 &
            LMCACHE_PIDS+=($!)
            LMCACHE_HTTP_PORTS+=("$shard_http_port")
            LMCACHE_LOGS+=("$shard_log")
            LMCACHE_SERVER_URLS+=("tcp://127.0.0.1:${shard_port}")
        done
        for shard in "${!LMCACHE_PIDS[@]}"; do
            wait_for_ready \
                --endpoint "http://127.0.0.1:${LMCACHE_HTTP_PORTS[$shard]}/healthcheck" \
                --log "${LMCACHE_LOGS[$shard]}" \
                --pid "${LMCACHE_PIDS[$shard]}" \
                --sleep-interval 1 \
                --timeout 600
        done
        LMCACHE_SERVER_URLS_CSV=$(IFS=,; echo "${LMCACHE_SERVER_URLS[*]}")
        OFFLOAD_ARGS=(
            --kv-transfer-config
            "{\"kv_connector\":\"LMCacheMPConnector\",\"kv_connector_module_path\":\"lmcache.integration.vllm.lmcache_mp_connector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.server_urls\":\"$LMCACHE_SERVER_URLS_CSV\",\"lmcache.mp.mq_timeout\":6000.0}}"
        )
        ;;
    *)
        echo "Unsupported KV_OFFLOAD_BACKEND: $KV_OFFLOAD_BACKEND (expected empty or lmcache)" >&2
        exit 1
        ;;
esac

# ---- LLM server config ----------------------------------------------------------
PARALLEL_ARGS=(--tensor-parallel-size "$TP")
if [ "$EP_SIZE" -gt 1 ]; then
    PARALLEL_ARGS+=(--enable-expert-parallel)
fi

# Synthetic acceptance standardizes throughput against the committed golden
# EAGLE3-GQA curve. Accuracy evals must use real target verification.
if [ "${EVAL_ONLY}" = "true" ]; then
    SPEC_CONFIG="{\"method\": \"eagle3\", \"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS, \"attention_backend\": \"ROCM_AITER_UNIFIED_ATTN\"}"
else
    SPEC_CONFIG="{\"method\": \"eagle3\", \"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS, \"attention_backend\": \"ROCM_AITER_UNIFIED_ATTN\", \"rejection_sample_method\": \"synthetic\", \"synthetic_acceptance_length\": $SYNTHETIC_ACCEPT_LEN}"
fi

echo "Starting vllm server..."
export PYTHONNOUSERSITE=1

export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1
# The AITER page-16 sparse-attention path requires exactly one KV head per
# tensor-parallel rank. MiniMax-M3 has four KV heads, so TP4 uses that fast
# path while TP2 uses vLLM's supported Triton sparse-attention fallback.
if [ "$TP" -eq 4 ]; then
    export VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1
else
    export VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=0
fi
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16=0
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION_MIN_SIZE_KB=256

VLLM_CMD=(
    vllm serve "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    "${PARALLEL_ARGS[@]}"
    --trust-remote-code
    --block-size 128
    --gpu-memory-utilization 0.85
    --enable-chunked-prefill
    --max-num-batched-tokens 32768
    --language-model-only
    --enable-prefix-caching
    --attention-backend ROCM_AITER_UNIFIED_ATTN
    --moe-backend aiter
    --kv-cache-dtype fp8
    --tool-call-parser minimax_m3
    --enable-auto-tool-choice
    --default-chat-template-kwargs '{"thinking_mode":"enabled"}'
    --max-num-seqs "$((2 * CONC))"
    --stream-interval 20
    --hf-overrides '{"text_config": {"use_index_cache": true, "index_topk_freq": 4}}'
    --speculative-config "$SPEC_CONFIG"
    "${OFFLOAD_ARGS[@]}"
)
write_command "$RESULT_DIR/server_command.txt" "${VLLM_CMD[@]}"
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# ---- Run benchmark ----------------------------------------------------------
if [ "${EVAL_ONLY}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    REPLAY_CMD+=" --apply-chat-template"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
