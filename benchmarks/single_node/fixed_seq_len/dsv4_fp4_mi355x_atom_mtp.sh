#!/usr/bin/env bash

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME \
    EP_SIZE \
    DP_ATTENTION

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

echo "TP: $TP, CONC: $CONC, ISL: $ISL, OSL: $OSL, EP_SIZE: $EP_SIZE, DP_ATTENTION: $DP_ATTENTION"

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

PARALLEL_ARGS=(-tp "$TP") #TP
if [ "$DP_ATTENTION" = "true" ]; then
    if [ "$EP_SIZE" -gt 1 ]; then #DP+EP
        PARALLEL_ARGS=(-tp "$TP" --enable-expert-parallel --enable-dp-attention )
    else #DP+TP
        PARALLEL_ARGS=(-tp "$TP" --enable-dp-attention )
    fi
fi 

SPEC_ARGS=(--method mtp --num-speculative-tokens 3 )

# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor

set -x
export ATOM_DISABLE_MMAP=true
export AITER_BF16_FP8_MOE_BOUND=0
export ATOM_MOE_GU_ITLV=1
python3 -m atom.entrypoints.openai_server \
    --model $MODEL \
    --server-port $PORT \
    "${PARALLEL_ARGS[@]}" \
    "${SPEC_ARGS[@]}" \
    --kv_cache_dtype fp8 \
    --trust-remote-code \
    > $SERVER_LOG 2>&1 &

SERVER_PID=$!

# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# --dsv4 routes prompts through encoding_dsv4.py (PR #1153), which emits the
# <bos><User>...<Assistant><think> framing DeepSeek-V4-Pro expects. The DSv4-Pro
# tokenizer ships without a jinja chat_template, so plain --use-chat-template
# would crash; --dsv4 sidesteps that and satisfies the AGENTS.md rule that all
# MTP scripts must benchmark against chat-formatted inputs (EAGLE acceptance
# silently regresses on raw random tokens).
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

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

# Stop GPU monitoring
stop_gpu_monitor
set +x
