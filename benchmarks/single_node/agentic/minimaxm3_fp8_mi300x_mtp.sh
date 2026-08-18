#!/usr/bin/env bash
set -eo pipefail
set -x

# MiniMax-M3 MXFP8 MI300X AgentX with EAGLE3-GQA and optional LMCache MP.

source "$(dirname "$0")/../../benchmark_lib.sh"


check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE DP_ATTENTION PORT EVAL_ONLY

DRAFT_MODEL="Inferact/MiniMax-M3-EAGLE3-GQA"
NUM_SPEC_TOKENS=3
SYNTHETIC_ACCEPT_LEN=2.78

if [[ -n "$SLURM_JOB_ID" ]]; then
    echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi
if [[ -n "$ROCR_VISIBLE_DEVICES" ]]; then
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

export AIPERF_SERVER_METRICS_URLS="http://localhost:${PORT}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="vllm:"

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

        LMCACHE_L1_SHARD_GB=$((TOTAL_CPU_DRAM_GB / TP))
        if [ "$LMCACHE_L1_SHARD_GB" -lt 1 ]; then
            echo "Error: LMCache DRAM budget is less than 1 GB per TP rank." >&2
            exit 1
        fi

        LMCACHE_SERVER_URLS=()
        LMCACHE_HTTP_PORTS=()
        LMCACHE_LOGS=()
        : > "$RESULT_DIR/lmcache_command.txt"
        for shard in $(seq 0 $((TP - 1))); do
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
        echo "Unsupported KV_OFFLOAD_BACKEND: $KV_OFFLOAD_BACKEND" >&2
        exit 1
        ;;
esac

PARALLEL_ARGS=(--tensor-parallel-size "$TP")
if [ "$EP_SIZE" -gt 1 ]; then
    PARALLEL_ARGS+=(--enable-expert-parallel)
fi

if [ "$EVAL_ONLY" = "true" ]; then
    SPEC_CONFIG="{\"method\":\"eagle3\",\"model\":\"$DRAFT_MODEL\",\"num_speculative_tokens\":$NUM_SPEC_TOKENS,\"attention_backend\":\"TRITON_ATTN\"}"
else
    SPEC_CONFIG="{\"method\":\"eagle3\",\"model\":\"$DRAFT_MODEL\",\"num_speculative_tokens\":$NUM_SPEC_TOKENS,\"attention_backend\":\"TRITON_ATTN\",\"rejection_sample_method\":\"synthetic\",\"synthetic_acceptance_length\":$SYNTHETIC_ACCEPT_LEN}"
fi

export PYTHONNOUSERSITE=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MHA=0
export TORCH_BLAS_PREFER_HIPBLASLT=1
export NCCL_MIN_NCHANNELS=112
export GPU_MAX_HW_QUEUES=2

VLLM_CMD=(
    vllm serve "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    "${PARALLEL_ARGS[@]}"
    --trust-remote-code
    --block-size 128
    --gpu-memory-utilization 0.90
    --enable-chunked-prefill
    --max-num-batched-tokens 16384
    --language-model-only
    --enable-prefix-caching
    --attention-backend TRITON_ATTN
    --kv-cache-dtype fp8
    --tool-call-parser minimax_m3
    --reasoning-parser minimax_m3
    --enable-auto-tool-choice
    --default-chat-template-kwargs '{"thinking_mode":"enabled"}'
    --max-num-seqs "$((2 * CONC))"
    --stream-interval 20
    --speculative-config "$SPEC_CONFIG"
    "${OFFLOAD_ARGS[@]}"
)
write_command "$RESULT_DIR/server_command.txt" "${VLLM_CMD[@]}"
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [ "$EVAL_ONLY" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    REPLAY_CMD+=" --apply-chat-template"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
