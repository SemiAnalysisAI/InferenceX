#!/usr/bin/env bash
set -eo pipefail

# DeepSeek-V4-Pro on MI355X via vLLM — MTP variant of dsv4_fp4_mi355x_vllm.sh.
# Adds MTP speculative decoding per vllm-project/vllm#43385 (ROCm DeepSeek-V4
# MTP support, merged 2026-05-24, present in v0.22.0 tagged 2026-05-29):
# --speculative-config '{"method":"mtp","num_speculative_tokens":2}'.
#
# Benchmark prompts are routed through DeepSeek-V4 chat encoding via --dsv4
# (which auto-enables --use-chat-template). EAGLE/MTP-style spec decoding is
# trained against chat-formatted inputs; benchmarking against raw random
# prompts silently regresses the acceptance rate.
#
# All other serving flags mirror the non-MTP MI355X recipe (TP=8,
# VLLM_ROCM_USE_AITER=1, AITER MoE, FP8 KV cache, mp executor, async
# scheduling, mode=3 FULL_AND_PIECEWISE compilation). See
# dsv4_fp4_mi355x_vllm.sh for per-flag rationale.

source "$(dirname "$0")/../../benchmark_lib.sh"

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

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi

if [ -n "$ROCR_VISIBLE_DEVICES" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
fi

start_gpu_monitor

PARALLEL_ARGS=(--tensor-parallel-size "$TP" --data-parallel-size 1)
if [ "${DP_ATTENTION}" = "true" ]; then
    PARALLEL_ARGS=(--tensor-parallel-size 1 --data-parallel-size "$TP")
fi

EP_ARGS=()
if [ "${EP_SIZE:-1}" -gt 1 ]; then
    EP_ARGS=(--enable-expert-parallel)
fi

# use 2 speculative tokens for all configs for now
NUM_SPEC_TOKENS=2

set -x
vllm serve $MODEL --port $PORT \
    "${PARALLEL_ARGS[@]}" \
    "${EP_ARGS[@]}" \
    --async-scheduling \
    --no-enable-prefix-caching \
    --distributed-executor-backend mp \
    --gpu-memory-utilization 0.8 \
    --kv-cache-dtype fp8 \
    --trust-remote-code \
    --moe-backend aiter \
    --tokenizer-mode deepseek_v4 \
    --reasoning-parser deepseek_v4 \
    --speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS}" \
    --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}' > $SERVER_LOG 2>&1 &

SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# --dsv4 routes prompts through DeepSeek-V4 chat encoding (auto-enables
# --use-chat-template); required for meaningful MTP acceptance numbers.
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
