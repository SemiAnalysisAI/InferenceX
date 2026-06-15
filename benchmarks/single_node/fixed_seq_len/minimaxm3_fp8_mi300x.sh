#!/usr/bin/env bash

# MiniMax-M3 MXFP8 MI300X (gfx942) single-node vLLM recipe.
# Reuses the dedicated ROCm image and applies the checked-in hybrid gfx94x
# MXFP8 MoE patch before starting vLLM. Block size 128 is mandatory for MSA
# sparse attention. Keep the default BF16 KV cache on gfx942: the checkpoint
# has no calibrated q/prob scales for ROCm FP8 attention, and vLLM's fallback
# scale of 1.0 corrupts model accuracy.
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

VLLM_PACKAGE_ROOT="$(
    python - <<'PY'
from pathlib import Path

import vllm

print(Path(vllm.__file__).resolve().parent.parent)
PY
)"
MXFP8_PATCH="$(dirname "$0")/minimaxm3_mi300x_mxfp8.patch"
MXFP8_ORACLE="$VLLM_PACKAGE_ROOT/vllm/model_executor/layers/fused_moe/oracle/mxfp8.py"
if ! grep -q "Using fused CDNA3 (gfx94x)" "$MXFP8_ORACLE"; then
    if ! patch --batch --forward -d "$VLLM_PACKAGE_ROOT" -p1 < "$MXFP8_PATCH"; then
        echo "Failed to apply the MI300X MXFP8 patch" >&2
        exit 1
    fi
fi
if ! grep -q "Using fused CDNA3 (gfx94x)" "$MXFP8_ORACLE"; then
    echo "MI300X MXFP8 backend marker is missing after patching" >&2
    exit 1
fi

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi

if [ -n "$ROCR_VISIBLE_DEVICES" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

SERVER_LOG=/workspace/server.log
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_USE_BREAKABLE_CUDAGRAPH=0

M3_AITER_AR_RMS_MODE="${M3_AITER_AR_RMS_MODE:-off}"
export M3_AITER_AR_RMS_MODE
case "$M3_AITER_AR_RMS_MODE" in
    off)
        ;;
    control|fused)
        # Enable AITER only to make the eager fused allreduce+RMSNorm op
        # available. M3 does not support torch.compile, so the runtime patch
        # calls this one op directly from its existing eager helper.
        # Keep every independently selectable AITER path disabled so the
        # control and fused runs differ only at that helper call.
        export VLLM_ROCM_USE_AITER=1
        export VLLM_ROCM_USE_AITER_PAGED_ATTN=0
        export VLLM_ROCM_USE_AITER_LINEAR=0
        export VLLM_ROCM_USE_AITER_LINEAR_HIPBMM=0
        export VLLM_ROCM_USE_AITER_MOE=0
        export VLLM_ROCM_USE_AITER_RMSNORM=0
        export VLLM_ROCM_USE_AITER_MLA=0
        export VLLM_ROCM_USE_AITER_MHA=0
        export VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=0
        export VLLM_ROCM_USE_AITER_TRITON_ROPE=0
        export VLLM_ROCM_USE_AITER_FP8BMM=0
        export VLLM_ROCM_USE_AITER_FP4BMM=0
        export VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=0
        export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0
        export VLLM_ROCM_USE_AITER_TRITON_GEMM=0

        python3 /workspace/utils/patch_minimaxm3_aiter_ar_rms.py

        echo "M3 AITER AR+RMS experiment mode: $M3_AITER_AR_RMS_MODE"
        ;;
    *)
        echo "Invalid M3_AITER_AR_RMS_MODE: $M3_AITER_AR_RMS_MODE" >&2
        exit 2
        ;;
esac

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
    export VLLM_TORCH_PROFILER_DIR="${VLLM_TORCH_PROFILER_DIR:-/workspace/profile_traces/${RESULT_FILENAME}}"
    rm -rf "$VLLM_TORCH_PROFILER_DIR"
    mkdir -p "$VLLM_TORCH_PROFILER_DIR"

    profiler_config="$(
        printf '{"profiler":"torch","torch_profiler_dir":"%s","torch_profiler_with_stack":false,"torch_profiler_with_flops":false,"torch_profiler_use_gzip":true,"torch_profiler_dump_cuda_time_total":false,"torch_profiler_record_shapes":false,"torch_profiler_with_memory":false,"ignore_frontend":true,"delay_iterations":%d,"max_iterations":1}' \
            "$VLLM_TORCH_PROFILER_DIR" "$profile_delay"
    )"
    PROFILE_ARGS=(
        --max-num-batched-tokens "$profile_token_budget"
        --profiler-config "$profiler_config"
    )
    # ROCTracer does not expose every kernel launched inside a HIP graph.
    PROFILE_ARGS+=(--compilation-config '{"cudagraph_mode":"NONE"}')
    echo "Profiling one steady-state decode iteration after $profile_delay engine iterations."
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
