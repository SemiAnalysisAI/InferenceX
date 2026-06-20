#!/usr/bin/env bash

# MiniMax-M3 MXFP8 MI300X (gfx942) single-node vLLM recipe with EAGLE3
# speculative decoding — the spec-decoding=mtp variant of
# minimaxm3_fp8_mi300x.sh. Adds the Inferact/MiniMax-M3-EAGLE3 draft head via
# --speculative-config with 3 speculative tokens. Everything else mirrors the
# non-MTP MI300X recipe: mandatory --block-size 128, --language-model-only for
# the text-only benchmark, --attention-backend TRITON_ATTN, and
# --no-enable-prefix-caching. Runs with CUDA graphs (no --enforce-eager);
# VLLM_USE_BREAKABLE_CUDAGRAPH=0 avoids the M3-decode breakable-cudagraph path.
# FP8 KV cache reduces memory pressure and increases concurrency headroom.
#
# Unlike the CUDA recipes, the drafter needs no attention_backend override:
# the FlashInfer "page size 128 requires GQA/MQA" limitation that forced
# FLASH_ATTN for the EAGLE3 MHA head on Blackwell is FlashInfer/CUDA-specific.
# Here the whole server runs on TRITON_ATTN (set globally below), which serves
# the MHA draft fine.

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

DRAFT_MODEL="Inferact/MiniMax-M3-EAGLE3"

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

# MODEL is a bare HF id on the mi300x single-node runner (a fast cache hit when
# pre-staged). The EAGLE3 draft is not staged; fetch it into the same cache.
if [[ "$MODEL" != /* ]]; then
  hf download "$MODEL"
  hf download "$DRAFT_MODEL"
fi

if [ -n "$ROCR_VISIBLE_DEVICES" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

SERVER_LOG=/workspace/server.log
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_USE_BREAKABLE_CUDAGRAPH=0

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
fi

PARALLEL_ARGS=(--tensor-parallel-size "$TP")
if [ "${DP_ATTENTION}" = "true" ]; then
    PARALLEL_ARGS=(
        --tensor-parallel-size 1
        --data-parallel-size "$TP"
        --enable-expert-parallel
    )
elif [ "$EP_SIZE" -gt 1 ]; then
    PARALLEL_ARGS+=(--enable-expert-parallel)
fi

# use 3 speculative tokens for all configs for now
NUM_SPEC_TOKENS=3

start_gpu_monitor

set -x
vllm serve "$MODEL" --port "$PORT" \
    "${PARALLEL_ARGS[@]}" \
    --block-size 128 \
    --kv-cache-dtype fp8 \
    --no-enable-prefix-caching \
    --language-model-only \
    --max-model-len "$MAX_MODEL_LEN" \
    --attention-backend TRITON_ATTN \
    --speculative-config "{\"method\": \"eagle3\", \"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS}" \
    --tool-call-parser minimax_m3 \
    --reasoning-parser minimax_m3 \
    --enable-auto-tool-choice > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

pip install -q datasets pandas

# Spec-decode acceptance rate degrades on raw random tokens; route prompts
# through the chat template as the other MTP recipes do.
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

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

stop_gpu_monitor
set +x
