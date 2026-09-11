#!/usr/bin/env bash
set -eo pipefail

# Upstream GB300 TP4 text-only recipe; only throughput acceptance is synthetic.
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
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_USE_RUST_FRONTEND=1
export PYTHONUNBUFFERED=1

# Pyxis shares the host network; port 8888 can already belong to a host service.
select_available_server_port
export AIPERF_SERVER_URL="http://localhost:${PORT}"
export AIPERF_SERVER_METRICS_URLS="${AIPERF_SERVER_URL}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="vllm:"
echo "Using vLLM endpoint ${AIPERF_SERVER_URL}"

# Golden AL: golden_al_distribution/dsv41flash_dspark.yaml, thinking_on, five draft tokens.
# Accuracy evals keep real block rejection; throughput fixes acceptance to AL 3.51.
if [[ "${EVAL_ONLY:-false}" == true ]]; then
    SPEC_CONFIG='{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":true}'
else
    SPEC_CONFIG='{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"synthetic","synthetic_acceptance_length":3.51,"enable_adaptive_verification":true}'
fi
VLLM_CMD=(
    vllm serve "$MODEL_PATH"
    --tokenizer-mode deepseek_v41
    --tensor-parallel-size "$TP"
    --tool-call-parser deepseek_v41 --enable-auto-tool-choice
    --reasoning-parser deepseek_v41
    --speculative-config "$SPEC_CONFIG"
    --language-model-only
    # Transport-only addition for the shared host network; clients use this port.
    --port "$PORT"
)
# Record CPU availability without changing the existing launcher allocation.
python3 - <<'PYCPU'
import os
print("GB300 process CPU affinity:", sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count(), flush=True)
PYCPU
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
