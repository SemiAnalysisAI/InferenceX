#!/usr/bin/env bash

# Unofficial DeepSeek-V4-Flash 4xH200 recipe. The topology and checkpoint are
# adapted from the public SGLang Hopper cookbook recipe to the smaller Flash
# checkpoint: TP4, Marlin W4A16 MoE runner, EAGLE (3 steps, top-k 1, 4 draft
# tokens). Server flags follow the sibling dsv4_fp8_h200_sglang_mtp.sh recipe.
# Proposal status: Unofficial pending maintainer scope confirmation, a full green
# upstream H200 sweep/evals, NVIDIA CODEOWNER sign-off, and /reuse-sweep-run.
#
# The checkpoint revision is pinned so an Unofficial proposal stays reproducible
# while DeepSeek-V4-Flash is still being revised on the Hub.
MODEL_REVISION="60d8d70770c6776ff598c94bb586a859a38244f1"

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

if [[ "$MODEL" != /* ]]; then hf download "$MODEL" --revision "$MODEL_REVISION" --exclude "*.md"; fi

nvidia-smi

SERVER_LOG="$PWD/server.log"

echo "TP: $TP, CONC: $CONC, ISL: $ISL, OSL: $OSL"

EVAL_CONTEXT_ARGS=""
if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    EVAL_CONTEXT_ARGS="--context-length $EVAL_MAX_MODEL_LEN"
fi

start_gpu_monitor --output "$PWD/gpu_metrics.csv"

set -x
PYTHONNOUSERSITE=1 sglang serve \
    --model-path $MODEL \
    --revision "$MODEL_REVISION" \
    --host 0.0.0.0 \
    --port $PORT \
    --trust-remote-code \
    --tp $TP \
    --moe-runner-backend marlin \
    --chunked-prefill-size 4096 \
    --disable-flashinfer-autotune \
    --disable-radix-cache \
    --mem-fraction-static 0.88 \
    --max-running-requests "$(( CONC * 3 / 2 > 8 ? CONC * 3 / 2 : 8 ))" \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    $EVAL_CONTEXT_ARGS >> $SERVER_LOG 2>&1 &

SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

pip install -q datasets pandas

# --dsv4 routes prompts through encoding_dsv4.py (PR #1153), which emits the
# <bos><User>...<Assistant><think> framing the DeepSeek-V4 family expects. The
# DeepSeek-V4-Flash tokenizer ships without a jinja chat_template, so plain
# --use-chat-template would crash; --dsv4 sidesteps that and satisfies the
# AGENTS.md rule that all MTP scripts must benchmark against chat-formatted
# inputs (EAGLE acceptance silently regresses on raw random tokens). This
# mirrors the existing dsv4_fp8_h200_sglang_mtp.sh recipe.
run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts $((CONC * 10)) \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir "$PWD/" \
    --dsv4

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

stop_gpu_monitor
set +x
