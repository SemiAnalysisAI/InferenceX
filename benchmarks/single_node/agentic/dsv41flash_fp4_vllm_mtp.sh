#!/usr/bin/env bash
set -eo pipefail

# Native DeepSeek-V4.1-Flash DSpark and Engram UVA weight offload.
# https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash
source "$(dirname "$0")/../../benchmark_lib.sh"
check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION
check_env_vars DSV41_MIN_CUDAGRAPH_CAPTURE_SIZE EVAL_ONLY VLLM_ENGINE_READY_TIMEOUT_S
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
export VLLM_USE_RUST_FRONTEND=1
export PYTHONUNBUFFERED=1

# Safetensors load strategy. vLLM auto-prefetches checkpoints only on NFS or
# Lustre; on other filesystems it memory-maps lazily. On cluster:h200-dgxc the
# HF cache is a VIRTIOFS mount, and on nightly-cd10ed6f the lazy path read this
# 475 GiB checkpoint at ~170 s/shard (19/48 shards when the 3600 s readiness
# deadline fired, run 35012494184) against ~12.5 s/shard for the same files on
# the deepseekv41-flash-0909 build (run 34504985992). Launchers whose cache is
# not a recognized network FS export VLLM_SAFETENSORS_LOAD_STRATEGY=prefetch so
# the shards are streamed into page cache by parallel readers first; the
# checkpoint fits comfortably in the ~1 TiB of host RAM those nodes expose.
LOAD_ARGS=()
if [[ -n "${VLLM_SAFETENSORS_LOAD_STRATEGY:-}" ]]; then
    LOAD_ARGS=(--safetensors-load-strategy "$VLLM_SAFETENSORS_LOAD_STRATEGY")
fi

# Preserve the upstream scheduler defaults; size graph capture for the sweep.
NUM_SPEC_TOKENS=5
CAPTURE_SIZE="${DSV41_MIN_CUDAGRAPH_CAPTURE_SIZE}"
while (( CAPTURE_SIZE < CONC * (1 + NUM_SPEC_TOKENS) && CAPTURE_SIZE < 2048 )); do
    CAPTURE_SIZE=$((CAPTURE_SIZE * 2))
done

# Pyxis shares the host network; port 8888 can already belong to a host service.
select_available_server_port
export AIPERF_SERVER_URL="http://localhost:${PORT}"
export AIPERF_SERVER_METRICS_URLS="${AIPERF_SERVER_URL}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="vllm:"
echo "Using vLLM endpoint ${AIPERF_SERVER_URL}"

# Golden AL: golden_al_distribution/dsv41flash_dspark.yaml, thinking_on, five draft tokens.
# Accuracy evals keep real block rejection; throughput fixes acceptance to AL 3.51.
if [[ "${EVAL_ONLY}" == true ]]; then
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
    --max-cudagraph-capture-size "$CAPTURE_SIZE"
    --disable-uvicorn-access-log
    "${LOAD_ARGS[@]}"
)
printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"
printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [[ "${EVAL_ONLY}" == true ]]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
