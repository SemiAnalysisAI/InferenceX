#!/usr/bin/env bash

# MiniMax-M3 NVFP4 RTX PRO 6000 Blackwell single-node vLLM recipe with
# EAGLE3 speculative decoding. The SM120 target and draft both use Triton
# attention; the target's NVFP4 experts use Marlin.
#
# The pinned vLLM image does not contain the MiniMax-M3 Marlin fixes tracked
# by vLLM PRs #45836 and #48929. Apply the narrow compatibility patch covered
# by docs/waiver/2306.md before importing vLLM.

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

SERVED_MODEL_NAME="$MODEL"
TARGET_MODEL_PATH="${MODEL_PATH:-$MODEL}"
if [[ "$TARGET_MODEL_PATH" != /* ]]; then
    TARGET_MODEL_PATH="$(hf download "$TARGET_MODEL_PATH")"
fi

DRAFT_MODEL="Inferact/MiniMax-M3-EAGLE3"
DRAFT_MODEL_PATH="$(hf download "$DRAFT_MODEL")"

if [[ -n "$SLURM_JOB_ID" ]]; then
    echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

nvidia-smi

SERVER_LOG=/workspace/server.log
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-3}"

if [[ ! "$NUM_SPEC_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_SPEC_TOKENS must be a positive integer, got: $NUM_SPEC_TOKENS" >&2
    exit 1
fi

# vLLM's speculative decode capture size is measured in tokens. Each active
# sequence can contribute one accepted token plus NUM_SPEC_TOKENS draft tokens.
CUDAGRAPH_CAPTURE_SIZE="$((CONC * (NUM_SPEC_TOKENS + 1)))"

export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_FLOAT32_MATMUL_PRECISION=high

if [ "${DP_ATTENTION}" = "true" ]; then
    PARALLEL_ARGS=(
        --tensor-parallel-size 1
        --data-parallel-size "$TP"
        --enable-expert-parallel
    )
    DRAFT_TP=1
elif [ "$EP_SIZE" -gt 1 ]; then
    PARALLEL_ARGS=(
        --tensor-parallel-size "$TP"
        --enable-expert-parallel
    )
    DRAFT_TP="$TP"
else
    PARALLEL_ARGS=(--tensor-parallel-size "$TP")
    DRAFT_TP="$TP"
fi

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
fi
start_gpu_monitor

VLLM_SITE_PACKAGES="$(
    python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'
)"
VLLM_MARLIN_PATCH=/workspace/benchmarks/patches/vllm/minimax_m3_nvfp4_marlin.patch
if ! patch --batch --forward --fuzz=0 -p1 -d "$VLLM_SITE_PACKAGES" \
    < "$VLLM_MARLIN_PATCH"; then
    echo "Failed to apply the pinned MiniMax-M3 Marlin compatibility patch" >&2
    exit 1
fi

SPECULATIVE_CONFIG="$(
    printf '{"method":"eagle3","model":"%s","num_speculative_tokens":%d,"draft_tensor_parallel_size":%d,"attention_backend":"TRITON_ATTN"}' \
        "$DRAFT_MODEL_PATH" "$NUM_SPEC_TOKENS" "$DRAFT_TP"
)"

set -x
vllm serve "$TARGET_MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --port "$PORT" \
    "${PARALLEL_ARGS[@]}" \
    --disable-custom-all-reduce \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --kv-cache-dtype fp8 \
    --block-size 128 \
    --language-model-only \
    --attention-backend TRITON_ATTN \
    --moe-backend marlin \
    --max-cudagraph-capture-size "$CUDAGRAPH_CAPTURE_SIZE" \
    --max-num-seqs "$CONC" \
    --max-num-batched-tokens "$((ISL * 2))" \
    --speculative-config "$SPECULATIVE_CONFIG" \
    --stream-interval 20 \
    --no-enable-prefix-caching \
    --trust-remote-code > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

run_benchmark_serving \
    --model "$SERVED_MODEL_NAME" \
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
