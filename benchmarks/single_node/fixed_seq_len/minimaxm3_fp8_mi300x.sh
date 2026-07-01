#!/usr/bin/env bash

# MiniMax-M3 MXFP8 MI300X (gfx942) single-node vLLM recipe.
# Reuses the dedicated ROCm image and the MI355X serving shape. Block size 128
# is mandatory for MSA sparse attention. Keep the default BF16 KV cache on
# gfx942: the checkpoint has no calibrated q/prob scales for ROCm FP8
# attention, and vLLM's fallback scale of 1.0 corrupts model accuracy.
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

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi

if [ -n "$ROCR_VISIBLE_DEVICES" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

SERVER_LOG=/workspace/server.log
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_USE_BREAKABLE_CUDAGRAPH=0

M3_AITER_AR_RMS_MODE="${M3_AITER_AR_RMS_MODE:-off}"
if [ "$M3_AITER_AR_RMS_MODE" = "fused" ] && [ "$CONC" -eq 1 ]; then
    # The graph-safe two-stage AITER primitive regresses same-node c1 by 2.2%.
    M3_AITER_AR_RMS_MODE=off
    echo "M3 AITER AR+RMS graph policy: using off for concurrency 1"
fi
export M3_AITER_AR_RMS_MODE
case "$M3_AITER_AR_RMS_MODE" in
    off)
        ;;
    control|fused)
        VLLM_PACKAGE_ROOT="$(
            python - <<'PY'
from pathlib import Path

import vllm

print(Path(vllm.__file__).resolve().parent.parent)
PY
        )"

        # Enable only the AITER custom all-reduce dependency. M3 does not
        # support torch.compile, so the runtime patch invokes this primitive
        # directly from the existing allreduce+Gemma RMSNorm helper.
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

        if [ "$M3_AITER_AR_RMS_MODE" = "fused" ]; then
            # The image's AITER build predates the two-stage memory-ordering fix.
            python3 /workspace/utils/install_minimaxm3_aiter.py
        fi

        python3 /workspace/utils/patch_minimaxm3_aiter_ar_rms.py

        DEFERRED_FFN_AR_PATCH="$(dirname "$0")/minimaxm3_mi300x_deferred_ffn_ar.patch"
        M3_MODEL_SOURCE="$VLLM_PACKAGE_ROOT/vllm/models/minimax_m3/amd/model.py"
        M3_MODEL_SOURCE_SHA256="91d81f8613e32f7afbd65c289f7885c5371263f70503bd053f97880989bf7536"
        M3_MODEL_PATCHED_SHA256="d26aa77cfce7c6162b0d1ebe2b403b854f5abe8f656b3a8deda2db1d89318ea8"
        m3_model_sha256="$(sha256sum "$M3_MODEL_SOURCE" | awk '{print $1}')"
        if [ "$m3_model_sha256" = "$M3_MODEL_SOURCE_SHA256" ]; then
            if ! patch --batch --dry-run -d "$VLLM_PACKAGE_ROOT" -p1 \
                < "$DEFERRED_FFN_AR_PATCH"; then
                echo "Failed to validate the M3 deferred FFN allreduce patch" >&2
                exit 1
            fi
            if ! patch --batch -d "$VLLM_PACKAGE_ROOT" -p1 \
                < "$DEFERRED_FFN_AR_PATCH"; then
                echo "Failed to apply the M3 deferred FFN allreduce patch" >&2
                exit 1
            fi
        elif [ "$m3_model_sha256" != "$M3_MODEL_PATCHED_SHA256" ]; then
            echo "M3 model source fingerprint mismatch: $m3_model_sha256" >&2
            exit 1
        fi
        python3 -m py_compile "$M3_MODEL_SOURCE"
        m3_model_sha256="$(sha256sum "$M3_MODEL_SOURCE" | awk '{print $1}')"
        if [ "$m3_model_sha256" != "$M3_MODEL_PATCHED_SHA256" ]; then
            echo "M3 model patched fingerprint mismatch: $m3_model_sha256" >&2
            exit 1
        fi
        echo "M3 deferred FFN allreduce patch ready: $m3_model_sha256"
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

start_gpu_monitor

set -x
vllm serve "$MODEL" --port "$PORT" \
    "${PARALLEL_ARGS[@]}" \
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
