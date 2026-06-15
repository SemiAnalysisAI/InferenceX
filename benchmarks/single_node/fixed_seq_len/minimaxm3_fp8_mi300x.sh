#!/usr/bin/env bash

# MiniMax-M3 MXFP8 MI300X (gfx942) single-node vLLM recipe.
# Reuses the dedicated ROCm image and converts MXFP8 MoE weights to 128x128
# block FP8 at load time. Block size 128 is mandatory for MSA sparse attention.
# Keep the default BF16 KV cache on gfx942: the checkpoint has no calibrated
# q/prob scales for ROCm FP8 attention, and vLLM's fallback scale of 1.0
# corrupts model accuracy.
# Target image vLLM revision: 4a560dd8db67c270f5e2afb614558271b76f2294.

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

if ! VLLM_PACKAGE_ROOT="$(
    python3 - <<'PY'
from pathlib import Path

import vllm

print(Path(vllm.__file__).resolve().parent.parent)
PY
)"; then
    echo "Failed to locate the installed vLLM package" >&2
    exit 1
fi
if [[ -z "$VLLM_PACKAGE_ROOT" || ! -d "$VLLM_PACKAGE_ROOT/vllm" ]]; then
    echo "Invalid installed vLLM package root: $VLLM_PACKAGE_ROOT" >&2
    exit 1
fi

apply_vllm_patch() {
    local patch_label="$1"
    local patch_path="$2"
    local -a patch_check_args=(
        --batch
        --silent
        -d "$VLLM_PACKAGE_ROOT"
        -p1
        --dry-run
    )

    if [[ ! -f "$patch_path" ]]; then
        echo "$patch_label patch is missing: $patch_path" >&2
        exit 1
    fi
    if patch "${patch_check_args[@]}" --reverse --forward < "$patch_path"; then
        echo "$patch_label patch is already fully applied"
    elif patch "${patch_check_args[@]}" --forward < "$patch_path"; then
        if ! patch --batch --forward -d "$VLLM_PACKAGE_ROOT" -p1 < "$patch_path"; then
            echo "Failed to apply the $patch_label patch" >&2
            exit 1
        fi
    else
        echo "Installed vLLM cannot cleanly apply the $patch_label patch" >&2
        exit 1
    fi
    if ! patch "${patch_check_args[@]}" --reverse --forward < "$patch_path"; then
        echo "$patch_label patch verification failed" >&2
        exit 1
    fi
}

PATCH_DIR="$(dirname "$0")"
apply_vllm_patch \
    "MI300X block-FP8 conversion" \
    "$PATCH_DIR/minimaxm3_mi300x_mxfp8.patch"
apply_vllm_patch \
    "MI300X block-FP8 EP route compaction" \
    "$PATCH_DIR/minimaxm3_mi300x_ep_mxfp8.patch"

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi

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

PROFILE_ARGS=()
if [ "${PROFILE:-0}" = "1" ]; then
    profile_token_budget=8192
    profile_prefill_iterations=$(( (ISL * CONC + profile_token_budget - 1) / profile_token_budget ))
    profile_delay=$((profile_prefill_iterations + 16))
    benchmark_num_prompts="$CONC"
    export VLLM_TORCH_PROFILER_DIR="${VLLM_TORCH_PROFILER_DIR:-/tmp/inferencex-profile/${RESULT_FILENAME}}"
    rm -rf "$VLLM_TORCH_PROFILER_DIR"
    mkdir -p "$VLLM_TORCH_PROFILER_DIR"

    profiler_config="$(
        printf '{"profiler":"torch","torch_profiler_dir":"%s","torch_profiler_with_stack":false,"torch_profiler_with_flops":false,"torch_profiler_use_gzip":true,"torch_profiler_dump_cuda_time_total":false,"torch_profiler_record_shapes":false,"torch_profiler_with_memory":false,"ignore_frontend":true,"delay_iterations":%d,"max_iterations":1}' \
            "$VLLM_TORCH_PROFILER_DIR" "$profile_delay"
    )"
    PROFILE_ARGS=(
        --max-num-batched-tokens "$profile_token_budget"
        --profiler-config "$profiler_config"
        --compilation-config '{"cudagraph_mode":"NONE"}'
    )
    # ROCTracer does not expose every kernel launched inside a HIP graph.
    echo "Profiling one steady-state decode iteration after $profile_delay engine iterations."
else
    benchmark_num_prompts="$((CONC * 10))"
fi

start_gpu_monitor

set -x
vllm serve "$MODEL" --port "$PORT" \
    "${PARALLEL_ARGS[@]}" \
    "${PROFILE_ARGS[@]}" \
    --block-size 128 \
    --no-enable-prefix-caching \
    --language-model-only \
    --max-model-len "$MAX_MODEL_LEN" \
    --attention-backend TRITON_ATTN \
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
    --num-prompts "$benchmark_num_prompts" \
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
