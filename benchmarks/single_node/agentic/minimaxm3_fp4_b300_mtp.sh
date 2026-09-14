#!/usr/bin/env bash
set -euo pipefail
set -x

# MiniMax-M3 NVFP4 B300 AgentX with EAGLE3-GQA and synthetic acceptance.
# DRAM KV offload uses vLLM's SimpleCPUOffloadConnector in lazy mode.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION

DRAFT_MODEL="Inferact/MiniMax-M3-EAGLE3-GQA"
NUM_SPEC_TOKENS=3
# Golden AL for the GQA draft head: golden_al_distribution/minimaxm3_eagle3_gqa.yaml
# minimax-m3.thinking_on[3]. The non-GQA curve (minimaxm3_eagle3.yaml) reads 2.83
# at the same level -- that head is not what this script runs.
SYNTHETIC_ACCEPT_LEN=2.78

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

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

OFFLOAD_ARGS=()
if require_agentic_kv_offload_backend vllm-simple; then
    python3 "$(dirname "$0")/../../../runners/patch_vllm_simple_kv_offload.py"
    CPU_OFFLOAD_BYTES=$((TOTAL_CPU_DRAM_GB * 1024 * 1024 * 1024))
    export VLLM_USE_SIMPLE_KV_OFFLOAD=1
    OFFLOAD_CONFIG=$(printf \
        '{"kv_connector":"SimpleCPUOffloadConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":%d,"lazy_offload":true}}' \
        "$CPU_OFFLOAD_BYTES")
    OFFLOAD_ARGS=(--kv-transfer-config "$OFFLOAD_CONFIG")
fi

export PYTHONNOUSERSITE=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_FLOAT32_MATMUL_PRECISION=high
export VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm

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

if [ "${EVAL_ONLY:-}" = "true" ]; then
    SPEC_CONFIG=$(printf \
        '{"method":"eagle3","model":"%s","num_speculative_tokens":%d,"attention_backend":"FLASH_ATTN"}' \
        "$DRAFT_MODEL_PATH" "$NUM_SPEC_TOKENS")
else
    SPEC_CONFIG=$(printf \
        '{"method":"eagle3","model":"%s","num_speculative_tokens":%d,"attention_backend":"FLASH_ATTN","rejection_sample_method":"synthetic","synthetic_acceptance_length":%.2f}' \
        "$DRAFT_MODEL_PATH" "$NUM_SPEC_TOKENS" "$SYNTHETIC_ACCEPT_LEN")
fi

{ set +x; } 2>/dev/null
VLLM_CMD=(
    vllm serve "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --tensor-parallel-size "$TP"
    --gpu-memory-utilization 0.9
    --block-size 128
    --language-model-only
    --enable-prefix-caching
    --no-enable-flashinfer-autotune
    --reasoning-parser minimax_m3
    --tool-call-parser minimax_m3
    --enable-auto-tool-choice
    --default-chat-template-kwargs '{"thinking_mode":"enabled"}'
    --attention-config '{"backend":"FLASHINFER","use_trtllm_attention":true,"indexer_kv_dtype":"fp8"}'
    --kv-cache-dtype fp8
    --max-cudagraph-capture-size 512
    --max-num-batched-tokens 16384
    --stream-interval 20
    --trust-remote-code
    --speculative-config "$SPEC_CONFIG"
    "${OFFLOAD_ARGS[@]}"
)
printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"
printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"
set -x

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"
if [ "${EVAL_ONLY}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
