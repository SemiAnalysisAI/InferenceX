#!/usr/bin/env bash
set -eo pipefail

# Native DeepSeek-V4.1-Flash DSpark and Engram UVA weight offload.
# https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash
source "$(dirname "$0")/../../benchmark_lib.sh"
check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION
if [[ "${KV_OFFLOAD_BACKEND:-}" == mooncake ]]; then
    require_agentic_kv_offload_backend mooncake
else
    require_agentic_kv_offload_none
fi
export GPU_COUNT="$TP"

# Complete/resume partial downloads instead of trusting nonempty directories.
if [[ -n "${MODEL_PATH:-}" && "$MODEL_PATH" != "$MODEL" ]]; then
    hf download "$MODEL" --local-dir "$MODEL_PATH"
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi

nvidia-smi
export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126
resolve_trace_source
install_agentic_deps
mkdir -p "$RESULT_DIR"
SERVER_LOG="$RESULT_DIR/server.log"
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export VLLM_USE_RUST_FRONTEND=1
export PYTHONUNBUFFERED=1

# Preserve scheduler concurrency; cap offload graphs to leave room for KV cache.
NUM_SPEC_TOKENS=5
CAPTURE_LIMIT=2048
if [[ "${KV_OFFLOAD_BACKEND:-}" == mooncake ]]; then
    CAPTURE_LIMIT=512
fi
CAPTURE_SIZE=1
while (( CAPTURE_SIZE < CONC * (1 + NUM_SPEC_TOKENS) && CAPTURE_SIZE < CAPTURE_LIMIT )); do
    CAPTURE_SIZE=$((CAPTURE_SIZE * 2))
done

# Pyxis shares the host network; port 8888 can already belong to a host service.
select_available_server_port
export AIPERF_SERVER_URL="http://localhost:${PORT}"
export AIPERF_SERVER_METRICS_URLS="${AIPERF_SERVER_URL}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="vllm:"
echo "Using vLLM endpoint ${AIPERF_SERVER_URL}"

SERVER_PID=""
MOONCAKE_MASTER_PID=""
cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    if (( rc != 0 )) && [[ "${KV_OFFLOAD_BACKEND:-}" == mooncake ]]; then
        # Preserve host OOM evidence that vLLM's generic "cancelled" error omits.
        for memory_file in /proc/meminfo /sys/fs/cgroup/memory.events /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory.max; do
            if [[ -r "$memory_file" ]]; then
                echo "Host memory diagnostic: $memory_file"
                cat "$memory_file" || true
            fi
        done
        [[ ! -f "$RESULT_DIR/mooncake_master.log" ]] || tail -n 80 "$RESULT_DIR/mooncake_master.log" || true
    fi
    stop_background_process_tree "$SERVER_PID" "vLLM server" 60
    stop_background_process_tree "$MOONCAKE_MASTER_PID" "Mooncake master" 10
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
# Engram uses about 189 GiB in aggregate; reserve 208 decimal GB.
export MOONCAKE_HOST_RESERVE_GB=208
OFFLOAD_ARGS=()
if [[ "${KV_OFFLOAD_BACKEND:-}" == mooncake ]]; then
    setup_agentic_mooncake
fi

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
    --engram-config '{"cpu_offload":true}'
    --speculative-config "$SPEC_CONFIG"
    "${OFFLOAD_ARGS[@]}"
    --max-model-len 1048576
    --max-cudagraph-capture-size "$CAPTURE_SIZE"
    --disable-uvicorn-access-log
)
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
