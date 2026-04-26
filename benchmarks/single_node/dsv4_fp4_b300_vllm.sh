#!/usr/bin/env bash

# DeepSeek-V4-Pro B300 single-node aggregate recipe from the submitted B300
# pareto sweep. TP mode (dp-attn=false) runs without expert parallel; DP mode
# (dp-attn=true) enables expert parallel (EP_SIZE=TP value = DP size).

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

hf download "$MODEL"

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

# DeepSeek-V4-Pro weights are large; engine startup can exceed the default
# 600s. Give it an hour to load.
export VLLM_ENGINE_READY_TIMEOUT_S=3600

PARALLEL_ARGS=(--tensor-parallel-size "$TP" --data-parallel-size 1)
if [ "${DP_ATTENTION}" = "true" ]; then
    PARALLEL_ARGS=(--tensor-parallel-size 1 --data-parallel-size "$TP")
fi

EP_ARGS=()
if [ "${EP_SIZE:-1}" -gt 1 ]; then
    EP_ARGS=(--enable-expert-parallel)
fi

if [ "${DP_ATTENTION}" = "true" ]; then
    MAX_NUM_BATCHED_TOKENS=2048
else
    MAX_NUM_BATCHED_TOKENS=$(( ISL * 2 ))
fi

BENCHMARK_MAX_MODEL_LEN="$MAX_MODEL_LEN"
if [ "$ISL" -eq 1024 ] && [ "$OSL" -eq 1024 ]; then
    BENCHMARK_MAX_MODEL_LEN=4096
fi

if [ "${EVAL_ONLY}" = "true" ]; then
    EVAL_MAX_MODEL_LEN=$(compute_eval_context_length "$MODEL" "$BENCHMARK_MAX_MODEL_LEN")
    export EVAL_MAX_MODEL_LEN
    SERVE_MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
else
    SERVE_MAX_MODEL_LEN="$BENCHMARK_MAX_MODEL_LEN"
fi

# Start GPU monitoring (power, temperature, clocks every second)
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
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
    --attention_config.use_fp4_indexer_cache True \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --max-cudagraph-capture-size 2048 \
    --max-model-len "$SERVE_MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!

# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

pip install -q datasets pandas

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
    --trust-remote-code

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

# Stop GPU monitoring
stop_gpu_monitor
set +x
