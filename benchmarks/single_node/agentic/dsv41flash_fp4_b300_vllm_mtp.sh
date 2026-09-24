#!/usr/bin/env bash
set -eo pipefail

# Native DeepSeek-V4.1-Flash DSpark and Engram UVA weight offload for B300.
# https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash
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

nvidia-smi
resolve_trace_source
install_agentic_deps
mkdir -p "$RESULT_DIR"
SERVER_LOG="$RESULT_DIR/server.log"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"
export VLLM_USE_RUST_FRONTEND=1
export PYTHONUNBUFFERED=1

NUM_SPEC_TOKENS=5

# Piecewise CUDA graph capture sizes are multiples of the six-token DSpark
# verification block, with a denser set for smaller batches.
GRAPH_SIZES_2046='6,12,18,24,30,36,48,60,72,96,120,144,192,240,288,384,480,576,768,1020,1536,2046'
GRAPH_SIZES_8190="${GRAPH_SIZES_2046},3072,4092,6144,8190"

# Match each batched-token limit to the largest graph captured by that tier.
GPU_MEMORY_UTILIZATION=""
if (( CONC <= 4 )); then
    GRAPH_SIZES="$GRAPH_SIZES_2046"; MAX_BATCHED_TOKENS=2048
elif (( CONC >= 128 && TP == 2 )); then
    GRAPH_SIZES="$GRAPH_SIZES_2046"; MAX_BATCHED_TOKENS=2048
    GPU_MEMORY_UTILIZATION=0.97
else
    GRAPH_SIZES="$GRAPH_SIZES_8190"; MAX_BATCHED_TOKENS=8192
fi
MAX_NUM_SEQS=256
CAPTURE_SIZE="${GRAPH_SIZES##*,}"
COMPILATION_CONFIG="{\"mode\":\"VLLM_COMPILE\",\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"cudagraph_capture_sizes\":[${GRAPH_SIZES}]}"

# Pyxis shares the host network; port 8888 can already belong to a host service.
select_available_server_port
export AIPERF_SERVER_URL="http://localhost:${PORT}"
export AIPERF_SERVER_METRICS_URLS="${AIPERF_SERVER_URL}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="vllm:"
echo "Using vLLM endpoint ${AIPERF_SERVER_URL}"

# Golden AL: golden_al_distribution/dsv41flash_dspark.yaml, thinking_on, five draft tokens.
# Accuracy evals keep real block rejection; other runs use synthetic acceptance at AL 3.51.
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
    --engram-config '{"cpu_offload":true}'
    --speculative-config "$SPEC_CONFIG"
    --max-model-len 1048576
    --compilation-config "$COMPILATION_CONFIG"
    --max-cudagraph-capture-size "$CAPTURE_SIZE"
    --max-num-batched-tokens "$MAX_BATCHED_TOKENS"
    --max-num-seqs "$MAX_NUM_SEQS"
    --disable-uvicorn-access-log
)
if [[ -n "$GPU_MEMORY_UTILIZATION" ]]; then
    VLLM_CMD+=(--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION")
fi
printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"
printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [[ "${EVAL_ONLY:-false}" == true ]]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
