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

hf download "$MODEL"

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

# Sized for one in-flight prefill at ISL+OSL (no double-prefill overlap).
# 8K/256 is prefill-dominated; the prior 2*ISL budget over-reserved KV pool
# for a worst-case OSL=1024 decode stream we don't have.
MAX_NUM_BATCHED_TOKENS=$(( ISL + OSL ))
BENCHMARK_MAX_MODEL_LEN=$MAX_MODEL_LEN

if [ "${EVAL_ONLY}" = "true" ]; then
    EVAL_MAX_MODEL_LEN=$(compute_eval_context_length "$MODEL" "$BENCHMARK_MAX_MODEL_LEN")
    export EVAL_MAX_MODEL_LEN
    SERVE_MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
else
    SERVE_MAX_MODEL_LEN="$BENCHMARK_MAX_MODEL_LEN"
fi

# MTP-2 for 8K/256: with only ~85 decode steps per request, MTP-3's draft
# overhead doesn't amortize. MTP-2 is the apples-to-apples middle ground
# vs the cann-recipes-infer A3 reference (next_n=1) and A950 (next_n=3).
# Override via DSV4_MTP_SPEC_TOKENS env if you want to sweep.
NUM_SPEC_TOKENS="${DSV4_MTP_SPEC_TOKENS:-2}"

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
    --max-cudagraph-capture-size 256 \
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
    --num-prompts "$CONC" \
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
