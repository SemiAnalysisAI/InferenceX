#!/usr/bin/env bash
set -eo pipefail
set -x

# H100 MiniMax-M3 MXFP8 AgentX with EAGLE3 and optional DRAM or NVMe KV offload.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION PORT EVAL_ONLY

DRAFT_MODEL="Inferact/MiniMax-M3-EAGLE3-GQA"

if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi

# Concurrent downloads on the shared HF cache can hit transient stale handles.
for attempt in 1 2 3 4 5; do
    hf download "$DRAFT_MODEL" && break
    if [ "$attempt" = 5 ]; then echo "hf download of $DRAFT_MODEL failed after $attempt attempts" >&2; exit 1; fi
    echo "hf download attempt $attempt failed; retrying in 60s" >&2
    sleep 60
done
nvidia-smi

export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126
resolve_trace_source
install_agentic_deps

export VLLM_ENGINE_READY_TIMEOUT_S=3600
export PYTHONNOUSERSITE=1

SERVER_LOG="$RESULT_DIR/server.log"
MOONCAKE_MASTER_LOG="$RESULT_DIR/mooncake_master.log"
mkdir -p "$RESULT_DIR"

SERVER_PID=""
MOONCAKE_MASTER_PID=""
cleanup_services() {
    local exit_code=$?
    trap - EXIT INT TERM
    set +e
    stop_background_process_tree "$SERVER_PID" "vLLM server" 60
    stop_background_process_tree "$MOONCAKE_MASTER_PID" "Mooncake master" 30
    exit "$exit_code"
}
trap cleanup_services EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

OFFLOAD_ARGS=()
MODEL_CPU_OFFLOAD_GB=26
MODEL_CHECKPOINT_PAGE_CACHE_GIB=414
MOONCAKE_LOCAL_BUFFER_GIB=4
if [ "$KV_OFFLOADING" = "none" ]; then
    require_agentic_kv_offload_none
elif [ "$KV_OFFLOADING" = "dram" ]; then
    require_agentic_kv_offload_backend mooncake
    TOTAL_CPU_DRAM_GIB=$((TOTAL_CPU_DRAM_GB * 1000000000 / 1073741824))
    PER_RANK_GIB=$(((TOTAL_CPU_DRAM_GIB - MODEL_CHECKPOINT_PAGE_CACHE_GIB) / TP - MODEL_CPU_OFFLOAD_GB - MOONCAKE_LOCAL_BUFFER_GIB))
    if (( PER_RANK_GIB <= 0 )); then
        echo "Error: CPU DRAM budget is too small for checkpoint cache, model, and KV offload" >&2
        exit 1
    fi
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
  "global_segment_size": "${PER_RANK_GIB}GB",
  "local_buffer_size": "${MOONCAKE_LOCAL_BUFFER_GIB}GB",
  "protocol": "rdma",
  "device_name": "",
  "enable_offload": false
}
EOF
    export MOONCAKE_CONFIG_PATH PYTHONHASHSEED=0 MC_SLICE_SIZE=1048576 MC_WORKERS_PER_CTX=4
    export MC_ENABLE_DEST_DEVICE_AFFINITY=1
    mooncake_master --port "$MOONCAKE_MASTER_PORT" \
        --eviction_high_watermark_ratio=0.80 \
        --eviction_ratio=0.10 > "$MOONCAKE_MASTER_LOG" 2>&1 &
    MOONCAKE_MASTER_PID=$!
    sleep 2
    kill -0 "$MOONCAKE_MASTER_PID"
    OFFLOAD_ARGS=(
        --kv-transfer-config
        '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"load_async":true}}'
    )
elif [ "$KV_OFFLOADING" = "nvme" ]; then
    require_agentic_kv_offload_backend vllm-simple
    : "${NVME_OFFLOAD_DIR:?NVME_OFFLOAD_DIR must be mounted by the H100 launcher}"
    NVME_OFFLOAD_TOTAL_BYTES=8000000000000
    NVME_OFFLOAD_PER_RANK_BYTES=$((NVME_OFFLOAD_TOTAL_BYTES / TP))
    OFFLOAD_ARGS=(
        --kv-transfer-config
        "{\"kv_connector\":\"SimpleCPUOffloadConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"kv_offload_backend\":\"disk\",\"disk_path\":\"$NVME_OFFLOAD_DIR/cache.bin\",\"disk_capacity_bytes\":$NVME_OFFLOAD_PER_RANK_BYTES,\"disk_buffer_slots\":4,\"lazy_offload\":false}}"
    )
else
    echo "Error: unsupported KV_OFFLOADING='$KV_OFFLOADING'" >&2
    exit 1
fi

export AIPERF_SERVER_METRICS_URLS="http://localhost:${PORT}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="vllm:"

NUM_SPEC_TOKENS=3
TOKENS_PER_SEQ=$((1 + NUM_SPEC_TOKENS))

# Golden AL is minimaxm3_eagle3_gqa.yaml thinking_on[3]; eval uses real verification.
SYNTHETIC_ACCEPT_LEN=2.78
if [ "$EVAL_ONLY" = "true" ]; then
    SPEC_CONFIG="{\"method\": \"eagle3\", \"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS, \"attention_backend\": \"FLASH_ATTN\"}"
else
    SPEC_CONFIG="{\"method\": \"eagle3\", \"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS, \"attention_backend\": \"FLASH_ATTN\", \"rejection_sample_method\": \"synthetic\", \"synthetic_acceptance_length\": $SYNTHETIC_ACCEPT_LEN}"
fi

MAX_NUM_SEQS=$((2 * CONC))
# MTP verifies four tokens per sequence, so CUDA graph capture is token-sized.
MAX_CUDAGRAPH_CAPTURE_SIZE=$((MAX_NUM_SEQS * TOKENS_PER_SEQ))

VLLM_CMD=(
    vllm serve "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --tensor-parallel-size "$TP"
    --data-parallel-size 1
    --gpu-memory-utilization 0.90
    --cpu-offload-gb "$MODEL_CPU_OFFLOAD_GB"
    --kv-cache-dtype fp8
    --attention-backend TRITON_ATTN
    --block-size 128
    --language-model-only
    --enable-prefix-caching
    --enable-prompt-tokens-details
    --default-chat-template-kwargs '{"thinking_mode":"enabled"}'
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-cudagraph-capture-size "$MAX_CUDAGRAPH_CAPTURE_SIZE"
    --speculative-config "$SPEC_CONFIG"
    --tool-call-parser minimax_m3
    --reasoning-parser minimax_m3
    --enable-auto-tool-choice
    --safetensors-load-strategy lazy
    --trust-remote-code
    "${OFFLOAD_ARGS[@]}"
)
write_command "$RESULT_DIR/vllm_command.txt" "${VLLM_CMD[@]}"
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [ "$EVAL_ONLY" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
