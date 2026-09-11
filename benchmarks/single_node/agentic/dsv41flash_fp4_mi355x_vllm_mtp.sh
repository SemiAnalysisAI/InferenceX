#!/usr/bin/env bash
set -eo pipefail

# DeepSeek-V4.1-Flash on MI355X: native DSpark and GPU-resident KV.
# Follow upstream AMD defaults for Engram; storage behavior needs verification.
# The ROCm release image is configured in amd-master.yaml; GPU validation is pending.
# https://github.com/vllm-project/recipes/blob/main/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml
source "$(dirname "$0")/../../benchmark_lib.sh"
check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION
require_agentic_kv_offload_none
export GPU_COUNT="$TP"

# Complete/resume partial downloads instead of trusting nonempty directories.
if [[ -n "${MODEL_PATH:-}" && "$MODEL_PATH" != "$MODEL" ]]; then
    hf download "$MODEL" --local-dir "$MODEL_PATH"
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi

if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
export OMP_NUM_THREADS=1
# Pin the full-context corpus for this 1M-context recipe.
export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126
resolve_trace_source
install_agentic_deps
mkdir -p "$RESULT_DIR"
SERVER_LOG="$RESULT_DIR/server.log"
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_USE_RUST_FRONTEND=1
export PYTHONUNBUFFERED=1

# Match the sibling's scheduler headroom for AgentX subagent fan-out.
MAX_NUM_SEQS=$((2 * CONC))
NUM_SPEC_TOKENS=5
CAPTURE_SIZE=1
while (( CAPTURE_SIZE < MAX_NUM_SEQS * (1 + NUM_SPEC_TOKENS) && CAPTURE_SIZE < 2048 )); do
    CAPTURE_SIZE=$((CAPTURE_SIZE * 2))
done

# Use the runner-specific port assigned by launch_mi355x-amds.sh.
export AIPERF_SERVER_URL="http://localhost:${PORT}"
export AIPERF_SERVER_METRICS_URLS="${AIPERF_SERVER_URL}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="vllm:"
echo "Using vLLM endpoint ${AIPERF_SERVER_URL}"

# Golden AL: golden_al_distribution/dsv41flash_dspark.yaml, thinking_on, five draft tokens.
# Accuracy evals keep real block rejection; throughput fixes acceptance to AL 3.51.
if [[ "${EVAL_ONLY:-false}" == true ]]; then
    SPEC_CONFIG='{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":true}'
else
    SPEC_CONFIG='{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"synthetic","synthetic_acceptance_length":3.51,"enable_adaptive_verification":false}'
fi
VLLM_CMD=(
    vllm serve "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0 --port "$PORT" --tensor-parallel-size "$TP"
    --language-model-only
    --tokenizer-mode deepseek_v41
    --tool-call-parser deepseek_v41 --enable-auto-tool-choice
    --reasoning-parser deepseek_v41
    --moe-backend aiter_triton_mxfp4_bf16
    --gpu-memory-utilization 0.9
    --speculative-config "$SPEC_CONFIG"
    --max-model-len 1048576
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-cudagraph-capture-size "$CAPTURE_SIZE"
    --max-num-batched-tokens 16384
    --disable-uvicorn-access-log
)
printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"
printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"
SERVER_PID=""
cleanup_server() {
    local rc=$?
    trap - EXIT INT TERM
    stop_background_process_tree "$SERVER_PID" "vLLM server" 60
    exit "$rc"
}
trap cleanup_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [[ "${EVAL_ONLY:-false}" == true ]]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
