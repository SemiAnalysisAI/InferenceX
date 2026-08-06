#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for MiniMax-M3 FP4 on MI355X using vLLM
# EAGLE3 speculative decoding.
#
# Required env vars:
#   MODEL, MODEL_PATH, TP, CONC, KV_OFFLOADING,
#   TOTAL_CPU_DRAM_GB, RESULT_DIR, DURATION, EP_SIZE, DP_ATTENTION

source "$(dirname "$0")/../../benchmark_lib.sh"

# Force the eval framework to lm-eval for this recipe. run_eval derives its
# default as swebench for agentic scenarios (scenario_default=swebench when
# IS_AGENTIC/SCENARIO_TYPE=agentic-coding), but EVAL_FRAMEWORK takes precedence
# over that default (benchmark_lib.sh: framework=${EVAL_FRAMEWORK:-...}), so
# setting it here makes the effective framework always lm-eval, never swebench.
export EVAL_FRAMEWORK="lm-eval"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE DP_ATTENTION

echo "MODEL=$MODEL TP=$TP CONC=$CONC KV_OFFLOADING=$KV_OFFLOADING TOTAL_CPU_DRAM_GB=$TOTAL_CPU_DRAM_GB RESULT_DIR=$RESULT_DIR DURATION=$DURATION EP_SIZE=$EP_SIZE DP_ATTENTION=$DP_ATTENTION"

DRAFT_MODEL="Inferact/MiniMax-M3-EAGLE3-GQA"
NUM_SPEC_TOKENS=3
# golden_al_distribution/minimaxm3_eagle3_gqa.yaml:
# minimax-m3.thinking_on[3]
SYNTHETIC_ACCEPT_LEN=2.78

if [[ -n "${SLURM_JOB_ID+x}" ]]; then
    echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

# ROCR/HIP visibility for vLLM 0.14+
if [[ -n "${ROCR_VISIBLE_DEVICES+x}" ]]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

if [[ -n "${MODEL_PATH:-}" ]]; then
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

# ---- Server config ----------------------------------------------------------
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

# AgentX replays growing multi-turn prefixes, so keep prefix caching enabled
# for both GPU-resident and native-offload configurations.
OFFLOAD_ARGS=()

case "${KV_OFFLOAD_BACKEND:-}" in
    "")
        require_agentic_kv_offload_none
        ;;
    vllm-native)
        require_agentic_kv_offload_backend vllm-native
        unset VLLM_USE_SIMPLE_KV_OFFLOAD
        # Use vLLM's regular native KV-offload path (OffloadingConnector),
        # NOT the SimpleCPUOffloadConnector. The "vllm-native" backend resolves to
        # OffloadingConnector by default; setting VLLM_USE_SIMPLE_KV_OFFLOAD=1
        # would switch it to SimpleCPUOffloadConnector. We intentionally leave
        # that env var UNSET here so the regular OffloadingConnector path is
        # used. The shortcut --kv_offloading_backend native + --kv_offloading_size
        # form constructs the KVTransferConfig at engine startup
        # (vllm/config/vllm.py:662).

        # Remove --disable-hybrid-kv-cache-manager and enable hybrid kv cache manager (default)
        # This gives extra cache hit than disabling hybrid kv cache manager
        OFFLOAD_ARGS=(
            --kv_offloading_backend native
            --kv_offloading_size "$TOTAL_CPU_DRAM_GB"
        )
        ;;
    *)
        echo "Unsupported KV_OFFLOAD_BACKEND: ${KV_OFFLOAD_BACKEND:-}" >&2
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
    SPEC_CONFIG="{\"method\": \"eagle3\", \"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS, \"attention_backend\": \"TRITON_ATTN\"}"
else
    SPEC_CONFIG="{\"method\": \"eagle3\", \"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS, \"attention_backend\": \"TRITON_ATTN\", \"rejection_sample_method\": \"synthetic\", \"synthetic_acceptance_length\": $SYNTHETIC_ACCEPT_LEN}"
fi

echo "Starting vllm server..."
export PYTHONNOUSERSITE=1

export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1
export VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1
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
    --attention-backend TRITON_ATTN
    --moe-backend aiter
    --kv-cache-dtype fp8
    --tool-call-parser minimax_m3
    --enable-auto-tool-choice
    --default-chat-template-kwargs '{"thinking_mode":"enabled"}'
    --max-num-seqs "$CONC"
    --stream-interval 20
    --hf-overrides '{"text_config": {"use_index_cache": true, "index_topk_freq": 4}}'
    # --speculative-config "$SPEC_CONFIG"
    "${OFFLOAD_ARGS[@]}"
)
printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"
printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# ---- Run benchmark ----------------------------------------------------------
if [ "${EVAL_ONLY}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
