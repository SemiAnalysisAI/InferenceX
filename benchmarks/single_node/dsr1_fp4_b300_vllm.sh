#!/usr/bin/env bash

# DeepSeek-R1 FP4 B300 vLLM run for the TokenSpeed MLA and fastokens paths
# introduced in vLLM v0.21.0. TokenSpeed MLA requires Blackwell and FP8 KV.

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    MAX_MODEL_LEN \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME \
    EP_SIZE

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

nvidia-smi

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

# Model loading can exceed the default timeout for the 394B FP4 checkpoint.
export VLLM_ENGINE_READY_TIMEOUT_S=3600

EP_ARGS=()
if [ "$EP_SIZE" -gt 1 ]; then
    EP_ARGS=(--enable-expert-parallel)
fi

BENCHMARK_MAX_MODEL_LEN="$MAX_MODEL_LEN"
if [ "${EVAL_ONLY}" = "true" ]; then
    EVAL_MAX_MODEL_LEN=$(compute_eval_context_length "$MODEL" "$BENCHMARK_MAX_MODEL_LEN")
    export EVAL_MAX_MODEL_LEN
    SERVE_MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
else
    SERVE_MAX_MODEL_LEN="$BENCHMARK_MAX_MODEL_LEN"
fi

MAX_NUM_BATCHED_TOKENS=$(( ISL * 2 ))

# vLLM v0.21.0 integrates the pre-v0.2 fastokens patching API as an optional package.
pip install -q 'fastokens>=0.1.1,<0.2.0' datasets pandas

# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor

set -x
vllm serve "$MODEL" --host 0.0.0.0 --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --pipeline-parallel-size 1 \
    --kv-cache-dtype fp8 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    "${EP_ARGS[@]}" \
    --attention-backend TOKENSPEED_MLA \
    --attention-config.mla_prefill_backend TOKENSPEED_MLA \
    --tokenizer-mode fastokens \
    --max-model-len "$SERVE_MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!

# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

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
