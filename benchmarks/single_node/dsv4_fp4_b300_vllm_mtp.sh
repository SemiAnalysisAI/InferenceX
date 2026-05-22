#!/usr/bin/env bash

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    DP_ATTENTION \
    CONC \
    ISL \
    OSL \
    MAX_MODEL_LEN \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

nvidia-smi

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

export VLLM_ENGINE_READY_TIMEOUT_S=3600

PARALLEL_ARGS=(--tensor-parallel-size "$TP" --data-parallel-size 1)
if [ "${DP_ATTENTION}" = "true" ]; then
    PARALLEL_ARGS=(--tensor-parallel-size 1 --data-parallel-size "$TP")
fi

EP_ARGS=()
if [ "${EP_SIZE:-1}" -gt 1 ]; then
    EP_ARGS=(--enable-expert-parallel)
fi

MOE_ARGS=()
if [ "${DP_ATTENTION}" = "true" ]; then
    MOE_ARGS=(--moe-backend deep_gemm_mega_moe)
    MAX_NUM_BATCHED_TOKENS=2048
else
    MAX_NUM_BATCHED_TOKENS=$(( ISL * 2 ))
fi

PROFILE_ARGS=()
if [[ "${PROFILE:-}" == "1" ]]; then
    PROFILER_CONFIG="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${VLLM_TORCH_PROFILER_DIR:-/workspace/}\"}"
    if [[ "$MODEL" == "deepseek-ai/DeepSeek-V4-Flash" ]]; then
        PROFILER_CONFIG="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${VLLM_TORCH_PROFILER_DIR:-/workspace/}\",\"ignore_frontend\":true,\"delay_iterations\":1,\"max_iterations\":8,\"active_iterations\":8,\"torch_profiler_with_stack\":false}"
    fi
    PROFILE_ARGS=(
        --profiler-config
        "$PROFILER_CONFIG"
    )
fi

COMPILATION_ARGS=(
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'
    --max-cudagraph-capture-size 2048
)
if [[ "$MODEL" == "deepseek-ai/DeepSeek-V4-Flash" ]]; then
    COMPILATION_ARGS=(
        --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'
        --max-cudagraph-capture-size 2048
    )
fi

BENCHMARK_MAX_MODEL_LEN=$MAX_MODEL_LEN

if [ "${EVAL_ONLY}" = "true" ]; then
    EVAL_MAX_MODEL_LEN=$(compute_eval_context_length "$MODEL" "$BENCHMARK_MAX_MODEL_LEN")
    export EVAL_MAX_MODEL_LEN
    SERVE_MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
else
    SERVE_MAX_MODEL_LEN="$BENCHMARK_MAX_MODEL_LEN"
fi

# Keep the existing Pro MTP profile at 2 speculative tokens; Flash uses the
# requested 3-token MTP profile.
NUM_SPEC_TOKENS=2
if [[ "$MODEL" == "deepseek-ai/DeepSeek-V4-Flash" ]]; then
    NUM_SPEC_TOKENS=3
fi

start_gpu_monitor

set -x
vllm serve "$MODEL" --host 0.0.0.0 --port "$PORT" \
    "${PARALLEL_ARGS[@]}" \
    --pipeline-parallel-size 1 \
    --kv-cache-dtype fp8 \
    --trust-remote-code \
    --block-size 256 \
    --no-enable-prefix-caching \
    "${EP_ARGS[@]}" \
    "${MOE_ARGS[@]}" \
    "${PROFILE_ARGS[@]}" \
    "${COMPILATION_ARGS[@]}" \
    --attention_config.use_fp4_indexer_cache True \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS}" \
    --max-model-len "$SERVE_MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

pip install -q datasets pandas

# MTP acceptance rate degrades on raw random tokens; --dsv4 routes prompts
# through chat-formatted encoding as required for speculative decoding benchmarks.
run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts "$((CONC * 10))" \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/ \
    --trust-remote-code \
    --dsv4

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

stop_gpu_monitor
set +x
