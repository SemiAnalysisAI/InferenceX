#!/usr/bin/env bash

# MiniMax-M3 MXFP8 MI300X (gfx942) single-node vLLM recipe.
# Reuses the dedicated ROCm image and applies the checked-in hybrid gfx94x
# MXFP8 MoE patch. Short-context EP8 uses the measured native/BF16 policy;
# long-context EP8 compacts local decode routes and uses low-padding BF16 GEMMs.
# Block size 128 is mandatory for MSA sparse attention. This experiment enables
# the corrected FNUZ FP8 main KV cache validated by vLLM PR #45563.
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
MXFP8_EP_PATCH="$(dirname "$0")/minimaxm3_mi300x_ep_mxfp8.patch"
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
if ! grep -q "profiled gfx94x MiniMax-M3 EP8" "$MXFP8_ORACLE"; then
    if ! patch --batch --forward -d "$VLLM_PACKAGE_ROOT" -p1 < "$MXFP8_EP_PATCH"; then
        echo "Failed to apply the MI300X EP8 MXFP8 optimization patch" >&2
        exit 1
    fi
fi
if ! grep -q "profiled gfx94x MiniMax-M3 EP8" "$MXFP8_ORACLE"; then
    echo "MI300X EP8 MXFP8 optimization marker is missing after patching" >&2
    exit 1
fi

# Port the validated ROCm FNUZ dtype handling from vLLM PR #45563.
FP8_KV_PATCH="$(dirname "$0")/minimaxm3_mi300x_fp8_kv.patch"
SPARSE_ATTN_SOURCE="$VLLM_PACKAGE_ROOT/vllm/models/minimax_m3/common/ops/sparse_attn.py"
SPARSE_ATTN_IMPL="$VLLM_PACKAGE_ROOT/vllm/models/minimax_m3/common/sparse_attention.py"
if ! grep -q "_FP8_DTYPES" "$SPARSE_ATTN_SOURCE"; then
    if ! patch --batch --dry-run -d "$VLLM_PACKAGE_ROOT" -p1 < "$FP8_KV_PATCH"; then
        echo "Failed to validate the MiniMax M3 FP8 KV-cache patch" >&2
        exit 1
    fi
    if ! patch --batch -d "$VLLM_PACKAGE_ROOT" -p1 < "$FP8_KV_PATCH"; then
        echo "Failed to apply the MiniMax M3 FP8 KV-cache patch" >&2
        exit 1
    fi
fi
if ! grep -q "current_platform.fp8_dtype()" "$SPARSE_ATTN_IMPL" \
    || ! grep -q "torch.float8_e4m3fnuz" "$SPARSE_ATTN_SOURCE"; then
    echo "MiniMax M3 FP8 KV-cache dtype fix is missing after patching" >&2
    exit 1
fi
python3 -m py_compile "$SPARSE_ATTN_SOURCE" "$SPARSE_ATTN_IMPL"

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

start_gpu_monitor

set -x
vllm serve "$MODEL" --port "$PORT" \
    "${PARALLEL_ARGS[@]}" \
    --kv-cache-dtype fp8 \
    --block-size 128 \
    --max-num-batched-tokens 16384 \
    --no-enable-prefix-caching \
    --language-model-only \
    --max-model-len "$MAX_MODEL_LEN" \
    --attention-backend TRITON_ATTN \
    --tool-call-parser minimax_m3 \
    --reasoning-parser minimax_m3 \
    --enable-auto-tool-choice > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

num_prompts_multiplier=10
if [ "$ISL" -ge 32768 ]; then
    # One full wave is enough for the 32k residency sanity check. Ten waves at
    # c256 would process roughly 84 million prompt tokens.
    num_prompts_multiplier=1
fi

run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts "$((CONC * num_prompts_multiplier))" \
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
