#!/usr/bin/env bash

# MiniMax-M3 NVFP4 B300 single-node vLLM recipe with EAGLE3 speculative
# decoding — same shape as minimaxm3_fp8_b300_mtp.sh but uses the
# nvidia/MiniMax-M3-NVFP4 checkpoint. MiniMax-M3 modelopt NVFP4 support
# (vllm-project/vllm PR #46380) is baked into the perf container image, so no
# runtime patch is needed.

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

# The target weights are launched from MODEL_PATH (the b300 launcher points it
# at the pre-staged read-only /scratch/models/MiniMax-M3-NVFP4). The EAGLE3
# draft is not pre-staged and must be downloaded, so it cannot live next to the
# read-only target — fetch it into the writable models dir (/data/models)
# instead. When MODEL_PATH is unset (stand-alone runs) fall back to the HF cache.
if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
    DRAFT_MODEL_PATH="/data/models/${DRAFT_MODEL##*/}"
    if [[ ! -d "$DRAFT_MODEL_PATH" || -z "$(ls -A "$DRAFT_MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$DRAFT_MODEL" --local-dir "$DRAFT_MODEL_PATH"
    fi
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
    hf download "$DRAFT_MODEL"
    DRAFT_MODEL_PATH="$DRAFT_MODEL"
fi

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

nvidia-smi

SERVER_LOG=/workspace/server.log

export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_FLOAT32_MATMUL_PRECISION=high

if [ "${DP_ATTENTION}" = "true" ]; then
  PARALLEL_ARGS="--tensor-parallel-size=1 --data-parallel-size=$TP --enable-expert-parallel"
elif [ "$EP_SIZE" -gt 1 ]; then
  PARALLEL_ARGS="--tensor-parallel-size=$TP --enable-expert-parallel"
else
  PARALLEL_ARGS="--tensor-parallel-size=$TP"
fi

# use 3 speculative tokens for all configs for now
NUM_SPEC_TOKENS=3

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
fi
start_gpu_monitor

set -x
vllm serve "$MODEL_PATH" --served-model-name "$MODEL" --host 0.0.0.0 --port $PORT \
$PARALLEL_ARGS \
--gpu-memory-utilization 0.90 \
--max-model-len $MAX_MODEL_LEN \
--block-size 128 \
--language-model-only \
--max-cudagraph-capture-size 2048 \
--max-num-batched-tokens "$((ISL * 2 ))" \
--speculative-config "{\"method\": \"eagle3\", \"model\": \"$DRAFT_MODEL_PATH\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS, \"attention_backend\": \"FLASH_ATTN\"}" \
--stream-interval 20 --no-enable-prefix-caching \
--trust-remote-code > $SERVER_LOG 2>&1 &

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
