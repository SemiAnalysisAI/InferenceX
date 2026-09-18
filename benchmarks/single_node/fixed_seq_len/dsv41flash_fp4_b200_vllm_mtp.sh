#!/usr/bin/env bash
set -eo pipefail

# B200 fixed-sequence counterpart of the native DSpark AgentX recipe.
# https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash
source "$(dirname "$0")/../../benchmark_lib.sh"
check_env_vars MODEL TP CONC ISL OSL RANDOM_RANGE_RATIO RESULT_FILENAME RESULT_DIR MAX_MODEL_LEN
check_env_vars INFMAX_CONTAINER_WORKSPACE
check_env_vars DSV41_MIN_CUDAGRAPH_CAPTURE_SIZE EVAL_ONLY VLLM_ENGINE_READY_TIMEOUT_S
export GPU_COUNT="$TP"

if [[ -n "${MODEL_PATH:-}" && "$MODEL_PATH" != "$MODEL" ]]; then
    hf download "$MODEL" --local-dir "$MODEL_PATH"
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi

nvidia-smi
mkdir -p "$RESULT_DIR"
SERVER_LOG="$RESULT_DIR/server.log"
export VLLM_USE_RUST_FRONTEND=1
export PYTHONUNBUFFERED=1

LOAD_ARGS=()
if [[ -n "${VLLM_SAFETENSORS_LOAD_STRATEGY:-}" ]]; then
    LOAD_ARGS=(--safetensors-load-strategy "$VLLM_SAFETENSORS_LOAD_STRATEGY")
fi

NUM_SPEC_TOKENS=5
CAPTURE_SIZE="${DSV41_MIN_CUDAGRAPH_CAPTURE_SIZE}"
while (( CAPTURE_SIZE < CONC * (1 + NUM_SPEC_TOKENS) && CAPTURE_SIZE < 2048 )); do
    CAPTURE_SIZE=$((CAPTURE_SIZE * 2))
done

# TP2 leaves ~145 GiB of weights on each 180 GB B200 even with the Engram
# tables offloaded, and vLLM's memory profiling counts the captured graphs
# against the KV budget: c1-c32 served, but c64 (capture 512) ended with
# -2.65 GiB and c128 (capture 1024) with -10.8 GiB of KV memory in run
# 35316389982, while --max-model-len was still the 1M context. With the
# matrix-supplied context below, stop capturing above 512 tokens on TP2;
# larger DSpark verify batches decode eagerly. TP4 keeps the default.
if (( TP == 2 && CAPTURE_SIZE > 512 )); then
    CAPTURE_SIZE=512
fi
# Real DSpark verification (block rejection with adaptive verification) profiles
# roughly 10 GiB more than the synthetic-acceptance throughput points: on TP2 the
# c128 throughput point still had 8.13 GiB for KV at capture 512, while the
# c64 and c128 evals ended at -2.22 GiB and never started (run 35355746550).
# Halve the graph tier and let the eval use 0.95 of HBM; GSM8K prompts are short.
EVAL_MEM_ARGS=()
if [[ "${EVAL_ONLY}" == true ]] && (( TP == 2 )); then
    CAPTURE_SIZE=256
    EVAL_MEM_ARGS=(--gpu-memory-utilization 0.95)
fi
select_available_server_port

# Match AgentX's golden AL 3.51; accuracy evals use real target verification.
if [[ "${EVAL_ONLY}" == true ]]; then
    SPEC_CONFIG='{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":true}'
else
    SPEC_CONFIG='{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"synthetic","synthetic_acceptance_length":3.51,"enable_adaptive_verification":false}'
fi

start_gpu_monitor
# Fixed-sequence runs serve the matrix-supplied context (isl + osl + slack),
# not the checkpoint's 1M: the sparse-attention indexer allocates a
# [batched-tokens, max-model-len] fp8 buffer and the profiler reserves KV for
# one full-context request, which at 1M left 0.97 GiB of KV at TP2 c32 in run
# 35316389982. Accuracy evals use the eval context instead.
MODEL_LEN="$MAX_MODEL_LEN"
if [[ "${EVAL_ONLY}" == true ]]; then
    # benchmark_lib derives EVAL_MAX_MODEL_LEN (isl + osl + 256, capped at the
    # checkpoint's context) as the other fixed-seq arms do; the workflow does
    # not export it (run 35320655804: both eval jobs exited at check_env_vars).
    setup_eval_context
    MODEL_LEN="$EVAL_MAX_MODEL_LEN"
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
    --max-model-len "$MODEL_LEN"
    --max-cudagraph-capture-size "$CAPTURE_SIZE"
    "${EVAL_MEM_ARGS[@]}"
    --disable-uvicorn-access-log
    "${LOAD_ARGS[@]}"
)
printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"
printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [[ "${EVAL_ONLY}" == true ]]; then
    run_eval --framework lm-eval --port "$PORT"
    # run_eval leaves lm-eval output in its temp directory for non-agentic
    # scenarios; staging into the workspace root, where the workflow looks for
    # results*.json, is the recipe's job (run 35387151990: GSM8K scored 0.97 and
    # the job still failed with "no results*.json files found").
    append_lm_eval_summary
else
    run_benchmark_serving \
        --model "$MODEL" --port "$PORT" --backend vllm \
        --input-len "$ISL" --output-len "$OSL" \
        --random-range-ratio "$RANDOM_RANGE_RATIO" \
        --num-prompts "$((CONC * 10))" --max-concurrency "$CONC" \
        `# The workflow reads $RESULT_FILENAME.json from the repository root, which the` \
        `# dsv41flash launchers mount at /ix rather than /workspace (run 35314631817` \
        `# wrote it under RESULT_DIR and the canary reported the result missing).` \
        --result-filename "$RESULT_FILENAME" --result-dir "$INFMAX_CONTAINER_WORKSPACE/" \
        --use-chat-template --tokenizer-mode deepseek_v41 \
        --server-pid "$SERVER_PID"
fi
