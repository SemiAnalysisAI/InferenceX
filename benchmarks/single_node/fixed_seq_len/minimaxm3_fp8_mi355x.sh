#!/usr/bin/env bash

# MiniMax-M3 MXFP8 MI355X (gfx950) single-node vLLM recipe.
# https://github.com/vllm-project/recipes/commit/2a3728ed9892debfd767a72a58ebc90b33f186e5
# The recipe recommends MXFP8 from TP=4 on gfx950 and requires block size 128.

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

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi

if [ -n "$ROCR_VISIBLE_DEVICES" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

SERVER_LOG=/workspace/server.log
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
# MI355X mxfp8 recipe (vllm-project/recipes#581): INT6 quick all-reduce plus
# the router-append shared-experts MoE fusion (vllm-project/vllm#46545). 
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT6

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

# Previously when EP is On, VLLM_ROCM_USE_AITER needs to be off.
# After https://github.com/vllm-project/vllm/pull/47158, 
# it can be simplified as VLLM_ROCM_USE_AITER=1.
# As the configs are TP only, remove the conditional check.
export VLLM_ROCM_USE_AITER=1

# Larger per-step prefill token budget to improve TP4 throughput at high
# concurrency. Overridable via env.
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"

start_gpu_monitor

set -x
vllm serve "$MODEL" --port "$PORT" \
    "${PARALLEL_ARGS[@]}" \
    --block-size 128 \
    --no-enable-prefix-caching \
    --language-model-only \
    --moe-backend aiter \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --kv-cache-dtype fp8 \
    --attention-backend TRITON_ATTN \
    --linear-backend emulation \
    --tool-call-parser minimax_m3 \
    --reasoning-parser minimax_m3 \
    --enable-auto-tool-choice > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!
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

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

stop_gpu_monitor
set +x
