#!/usr/bin/env bash

# MiniMax-M3 MXFP4 MI355X (gfx950) single-node vLLM recipe with EAGLE3
# speculative decoding. This is the spec-decoding=mtp variant of
# minimaxm3_fp4_mi355x_vllm.sh and uses three speculative tokens from
# Inferact/MiniMax-M3-EAGLE3. The pinned nightly includes upstream AMD
# MiniMax-M3 SupportsEagle3 support, so no runtime model patch is needed.
# MoE serving mirrors minimaxm3_fp4_mi355x_vllm.sh (AITER MoE, vllm#46419),
# except AITER MoE is gated off when expert parallelism is enabled (see below).

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

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi
hf download "$DRAFT_MODEL"

if [ -n "$ROCR_VISIBLE_DEVICES" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

SERVER_LOG=/workspace/server.log
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_USE_BREAKABLE_CUDAGRAPH=0

# AITER MoE accelerates the dense (non-EP) MoE path but is incompatible with
# expert parallelism, so disable it when EP is enabled (DP attention or EP > 1).
# https://github.com/SemiAnalysisAI/InferenceX/pull/1955#discussion_r3495386866
MOE_ARGS=()
if [ "${DP_ATTENTION}" = "true" ] || [ "$EP_SIZE" -gt 1 ]; then
    export VLLM_ROCM_USE_AITER=0
else
    export VLLM_ROCM_USE_AITER=1
    export VLLM_ROCM_USE_AITER_MOE=1
    export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1
    MOE_ARGS=(--moe-backend aiter)
fi

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

NUM_SPEC_TOKENS=3

start_gpu_monitor

set -x
vllm serve "$MODEL" --port "$PORT" \
    "${PARALLEL_ARGS[@]}" \
    --trust-remote-code \
    --block-size 128 \
    --no-enable-prefix-caching \
    --language-model-only \
    --max-model-len "$MAX_MODEL_LEN" \
    --attention-backend TRITON_ATTN \
    "${MOE_ARGS[@]}" \
    --speculative-config "{\"method\": \"eagle3\", \"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS}" \
    --tool-call-parser minimax_m3 \
    --enable-auto-tool-choice \
    --reasoning-parser minimax_m3 > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!
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
    --trust-remote-code \
    --use-chat-template

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

stop_gpu_monitor
set +x
