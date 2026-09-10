#!/usr/bin/env bash
set -eo pipefail

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC ISL OSL MAX_MODEL_LEN RANDOM_RANGE_RATIO RESULT_FILENAME

# Always complete/resume the download; a nonempty directory can be partial.
if [[ -n "${MODEL_PATH:-}" && "$MODEL_PATH" != "$MODEL" ]]; then
    hf download "$MODEL" --local-dir "$MODEL_PATH"
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi

export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_USE_RUST_FRONTEND=1
export PYTHONUNBUFFERED=1
SERVER_LOG=/workspace/server.log
SERVE_MAX_MODEL_LEN="$MAX_MODEL_LEN"
if [[ "${EVAL_ONLY:-false}" == true ]]; then
    EVAL_MAX_MODEL_LEN=$(compute_eval_context_length "$MODEL" "$MAX_MODEL_LEN")
    export EVAL_MAX_MODEL_LEN
    SERVE_MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
fi

# DSpark's native trained block has five draft tokens.
NUM_SPEC_TOKENS=5
CAPTURE_SIZE=1
while (( CAPTURE_SIZE < CONC * (1 + NUM_SPEC_TOKENS) && CAPTURE_SIZE < 2048 )); do
    CAPTURE_SIZE=$((CAPTURE_SIZE * 2))
done

nvidia-smi
start_gpu_monitor
set -x
vllm serve "$MODEL_PATH" --served-model-name "$MODEL" --host 0.0.0.0 --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --language-model-only \
    --tokenizer-mode deepseek_v41 \
    --tool-call-parser deepseek_v41 --enable-auto-tool-choice \
    --reasoning-parser deepseek_v41 \
    --engram-config '{"cpu_offload":true}' \
    --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":true}' \
    --no-enable-prefix-caching \
    --max-cudagraph-capture-size "$CAPTURE_SIZE" \
    --max-model-len "$SERVE_MAX_MODEL_LEN" \
    --max-num-batched-tokens 16384 > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

pip install -q datasets pandas
run_benchmark_serving \
    --model "$MODEL" --tokenizer "$MODEL_PATH" --tokenizer-mode deepseek_v41 \
    --port "$PORT" --backend vllm \
    --input-len "$ISL" --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts "$((CONC * 10))" --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" --result-dir /workspace/ \
    --use-chat-template --server-pid "$SERVER_PID"

if [[ "${RUN_EVAL:-false}" == true ]]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi
stop_gpu_monitor
set +x
