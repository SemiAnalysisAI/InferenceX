#!/usr/bin/env bash
set -euo pipefail
set -x

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION


if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

DRAFT_MODEL="lightseekorg/kimi-k2.6-eagle3.1-mla"

if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
    DRAFT_MODEL_PATH="/data/models/${DRAFT_MODEL##*/}"
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

SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

SERVER_PID=""

cleanup_agentic_services() {
    local exit_code=$?
    trap - EXIT INT TERM
    set +e
    stop_background_process_tree "$SERVER_PID" "vLLM server" 60
    exit "$exit_code"
}
trap cleanup_agentic_services EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

DCP_SIZE="${DCP_SIZE:-1}"
DCP_ARGS=()
if [[ "$DCP_SIZE" -gt 1 ]]; then
    DCP_ARGS+=(--decode-context-parallel-size "$DCP_SIZE" --dcp-comm-backend a2a)
    NUM_SPEC_TOKENS=3
    SYNTHETIC_ACCEPT_LEN=2.88
    SPEC_ARGS=(--speculative-config "{\"method\":\"eagle3\",\"model\":\"$DRAFT_MODEL_PATH\",\"num_speculative_tokens\":$NUM_SPEC_TOKENS,\"rejection_sample_method\":\"synthetic\",\"synthetic_acceptance_length\":$SYNTHETIC_ACCEPT_LEN,\"attention_backend\":\"TOKENSPEED_MLA\"}")
    ATTN_CONFIG='{"mla_prefill_backend":"TOKENSPEED_MLA"}'
    COMPILATION_CONFIG='{"pass_config":{"fuse_allreduce_rms":false}}'
else
    NUM_SPEC_TOKENS=4
    SYNTHETIC_ACCEPT_LEN=3.24
    SPEC_ARGS=(--speculative-config "{\"method\":\"eagle3\",\"model\":\"$DRAFT_MODEL_PATH\",\"num_speculative_tokens\":$NUM_SPEC_TOKENS,\"rejection_sample_method\":\"synthetic\",\"synthetic_acceptance_length\":$SYNTHETIC_ACCEPT_LEN}")
    ATTN_CONFIG='{"mla_prefill_backend":"TRTLLM_RAGGED","use_prefill_query_quantization":true}'
    COMPILATION_CONFIG='{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'
fi
ATTN_BACKEND_ARGS=(--attention-backend TOKENSPEED_MLA)

OFFLOAD_ARGS=()

if agentic_kv_offload_enabled; then
    case "$KV_OFFLOAD_BACKEND" in
    native)
        export VLLM_USE_SIMPLE_KV_OFFLOAD=1
        CPU_OFFLOAD_BYTES=$((TOTAL_CPU_DRAM_GB * 1024 * 1024 * 1024))
        OFFLOAD_ARGS=(
            --disable-hybrid-kv-cache-manager
            --kv-transfer-config
            "{\"kv_connector\":\"SimpleCPUOffloadConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":$CPU_OFFLOAD_BYTES,\"lazy_offload\":false}}"
        )
        ;;
    *) echo "Error: unsupported KV_OFFLOAD_BACKEND value '$KV_OFFLOAD_BACKEND' with EAGLE3 (expected: native)" >&2; exit 1 ;;
    esac
fi


GMU=0.90
if [[ "$DCP_SIZE" -gt 1 && "$KV_OFFLOADING" == "none" ]]; then
    GMU=0.85
fi

echo "Starting vllm server..."
export PYTHONNOUSERSITE=1

export VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm

{ set +x; } 2>/dev/null
VLLM_CMD=(
    vllm serve "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --kv-cache-dtype fp8
    --trust-remote-code
    --block-size 64
    --language-model-only
    --gpu-memory-utilization "$GMU"
    --max-num-seqs "$CONC"
    "${ATTN_BACKEND_ARGS[@]}"
    --attention-config "$ATTN_CONFIG"
    --compilation-config "$COMPILATION_CONFIG"
    --max-cudagraph-capture-size 2048
    --max-num-batched-tokens 16384
    --stream-interval 10
    --enable-prefix-caching
    --tensor-parallel-size "$TP"
    "${SPEC_ARGS[@]}"
    "${DCP_ARGS[@]}"
    "${OFFLOAD_ARGS[@]}"
)
printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"
printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

build_replay_cmd "$RESULT_DIR"

run_agentic_replay_and_write_outputs "$RESULT_DIR"
