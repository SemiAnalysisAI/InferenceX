#!/usr/bin/env bash

# MiniMax-M3 MXFP8 MI300X (gfx942) single-node vLLM recipe.
# Reuses the dedicated ROCm image and the MI355X serving shape. This test branch
# applies the vLLM MiniMax-M3 FNUZ fix before enabling FP8 KV cache on gfx942.

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

VLLM_SITE_DIR=$(python3 -c \
    'import pathlib, vllm; print(pathlib.Path(vllm.__file__).parent)')
VLLM_PACKAGE_ROOT=$(dirname "$VLLM_SITE_DIR")
VLLM_FNUZ_PATCH=/workspace/benchmarks/single_node/fixed_seq_len/minimaxm3_fp8_mi300x_fnuz.patch

patch --dry-run --forward --batch -d "$VLLM_PACKAGE_ROOT" -p1 < "$VLLM_FNUZ_PATCH"
patch --forward --batch -d "$VLLM_PACKAGE_ROOT" -p1 < "$VLLM_FNUZ_PATCH"

python3 - <<'PY'
from pathlib import Path

import torch
import vllm
from vllm.platforms import current_platform

root = Path(vllm.__file__).parent
sparse_attention = (
    root / "models/minimax_m3/common/sparse_attention.py"
).read_text()
sparse_ops = (
    root / "models/minimax_m3/common/ops/sparse_attn.py"
).read_text()
amd_model = (root / "models/minimax_m3/amd/model.py").read_text()

assert "else current_platform.fp8_dtype()" in sparse_attention
assert "torch.float8_e4m3fnuz" in sparse_ops
assert "MINIMAX_M3_FP8_KV_RUNTIME_VERIFIED" in amd_model
assert current_platform.fp8_dtype() == torch.float8_e4m3fnuz
print("VLLM_MINIMAX_M3_FNUZ_PATCH_VERIFIED", current_platform.fp8_dtype())
PY

SERVER_LOG=/workspace/server.log
export VLLM_ENGINE_READY_TIMEOUT_S=3600

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
    --kv-cache-dtype fp8 \
    --attention-backend TRITON_ATTN \
    --enforce-eager \
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
