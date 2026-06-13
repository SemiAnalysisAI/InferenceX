#!/usr/bin/env bash

# MiniMax-M3 MXFP8 B200 single-node vLLM MTP (EAGLE3) variant of
# minimaxm3_fp8_b200.sh (https://recipes.vllm.ai/MiniMaxAI/MiniMax-M3).
# Adds the recipe's spec_decoding feature: EAGLE3 speculative decoding with the
# Inferact/MiniMax-M3-EAGLE3 draft head (num_speculative_tokens=3,
# attention_backend=FLASH_ATTN). EAGLE acceptance collapses on raw random
# prompts, so the benchmark routes prompts through chat-formatted encoding via
# --use-chat-template (required for all *_mtp.sh per AGENTS.md).

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    EP_SIZE \
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

# launch_b200-dgxc.sh rewrites MODEL to the pre-downloaded
# /lustre/fsw/gharunners/models/MiniMax-M3-MXFP8 path; only download when
# handed a bare HF id (b200-cw / b200-nb runners).
if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi

# EAGLE3 draft head. launch_b200-dgxc.sh mounts it beside the main weights and
# exports DRAFT_MODEL_PATH; default to the sibling of MODEL_PATH, then fall
# back to the HF id (vLLM downloads it) for stand-alone runs.
DRAFT_MODEL="${DRAFT_MODEL_PATH:-}"
if [[ -z "$DRAFT_MODEL" ]]; then
    if [[ "$MODEL" == /* && -d "${MODEL%/*}/MiniMax-M3-EAGLE3" ]]; then
        DRAFT_MODEL="${MODEL%/*}/MiniMax-M3-EAGLE3"
    else
        DRAFT_MODEL="Inferact/MiniMax-M3-EAGLE3"
    fi
fi
if [[ "$DRAFT_MODEL" != /* ]]; then hf download "$DRAFT_MODEL"; fi
echo "EAGLE3 draft head: $DRAFT_MODEL"

SERVER_LOG=/workspace/server.log

# 444 GB of MXFP8 weights + EAGLE3 draft head off shared FS; engine startup
# can exceed the default 600s readiness window.
export VLLM_ENGINE_READY_TIMEOUT_S=3600

if [ "${DP_ATTENTION}" = "true" ]; then
  PARALLEL_ARGS="--tensor-parallel-size=1 --data-parallel-size=$TP --enable-expert-parallel"
elif [ "$EP_SIZE" -gt 1 ]; then
  PARALLEL_ARGS="--tensor-parallel-size=$TP --enable-expert-parallel"
else
  PARALLEL_ARGS="--tensor-parallel-size=$TP"
fi

# Fixed-seq-len runs don't need graphs past the request concurrency, but spec
# decoding verifies CONC*(1+num_spec) tokens per decode step; capture up to the
# next power of two >= that, capped at vLLM's 2048 ceiling.
NUM_SPEC_TOKENS=3
CAPTURE_SIZE=4
TARGET=$(( CONC * (NUM_SPEC_TOKENS + 1) ))
while (( CAPTURE_SIZE < TARGET )); do CAPTURE_SIZE=$((CAPTURE_SIZE * 2)); done
(( CAPTURE_SIZE > 2048 )) && CAPTURE_SIZE=2048

SPEC_CONFIG="{\"method\": \"eagle3\", \"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS, \"attention_backend\": \"FLASH_ATTN\"}"

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
fi
# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor

set -x
vllm serve $MODEL --port $PORT \
$PARALLEL_ARGS \
--gpu-memory-utilization 0.90 \
--max-model-len $MAX_MODEL_LEN \
--block-size 128 \
--language-model-only \
--speculative-config "$SPEC_CONFIG" \
--max-cudagraph-capture-size $CAPTURE_SIZE \
--max-num-batched-tokens "$((ISL * 2 ))" \
--stream-interval 20 --no-enable-prefix-caching \
--trust-remote-code > $SERVER_LOG 2>&1 &

SERVER_PID=$!

# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# --use-chat-template: EAGLE3 acceptance is trained against chat-formatted
# inputs; benchmarking raw prompts silently regresses the acceptance rate.
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
    --use-chat-template

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

# Stop GPU monitoring
stop_gpu_monitor
set +x
