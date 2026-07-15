#!/usr/bin/env bash

# MiniMax-M3 NVFP4 B300 single-node vLLM recipe.
# Same shape as minimaxm3_fp8_b300.sh but uses the nvidia/MiniMax-M3-NVFP4
# checkpoint. MiniMax-M3 modelopt NVFP4 support (vllm-project/vllm PR #46380) is
# baked into the perf container image.
#
# At runtime the recipe swaps the image's FlashInfer for a pinned nightly with
# the upstream SM100 low-M MXFP8 split-K kernel (flashinfer-ai/flashinfer#3847),
# the distributed AutoTuner synchronization API (#3187), and the non-Tensor
# guard (#3918). It backports the still-unmerged #3912 memory fix and patches
# vLLM to opt in to synchronized distributed tuning. vLLM PR #48268 supplies
# the per-op AutoTuner skip control used below.

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

# --- FlashInfer nightly + targeted runtime patches --------------------------
FLASHINFER_VERSION=0.6.15.dev20260712
FLASHINFER_NIGHTLY_TAG=nightly-v0.6.15-20260712
FLASHINFER_RELEASE_URL="https://github.com/flashinfer-ai/flashinfer/releases/download/${FLASHINFER_NIGHTLY_TAG}"

python3 -m pip uninstall -y flashinfer-python flashinfer-cubin flashinfer-jit-cache

python3 -m pip install \
    "${FLASHINFER_RELEASE_URL}/flashinfer_python-${FLASHINFER_VERSION}-py3-none-any.whl" \
    "${FLASHINFER_RELEASE_URL}/flashinfer_cubin-${FLASHINFER_VERSION}-py3-none-any.whl" \
    "${FLASHINFER_RELEASE_URL}/flashinfer_jit_cache-${FLASHINFER_VERSION}+cu130-cp39-abi3-manylinux_2_28_$(uname -m).whl" \
    || { echo "FlashInfer nightly install failed" >&2; exit 1; }

if ! command -v patch >/dev/null 2>&1; then
    apt-get update -y && apt-get install -y --no-install-recommends patch \
        || { echo "Failed to install patch(1)" >&2; exit 1; }
fi
SITE_PACKAGES=$(dirname "$(python3 -c "import importlib.util; print(importlib.util.find_spec('flashinfer').submodule_search_locations[0])")") \
    || { echo "Could not locate the installed flashinfer package" >&2; exit 1; }

# Backport the runtime portion of flashinfer-ai/flashinfer#3912. Caching the
# packed top-k initializer preserves its identity across tuning calls and avoids
# retaining a fresh closure in the AutoTuner cache for every invocation.
AUTOTUNER_MEMORY_PATCH="$(dirname "$0")/patches/flashinfer-pr-3912.patch"
patch --dry-run -p1 -d "${SITE_PACKAGES}" < "${AUTOTUNER_MEMORY_PATCH}" >/dev/null \
    || { echo "FlashInfer PR #3912 patch does not apply" >&2; exit 1; }
patch -p1 -d "${SITE_PACKAGES}" < "${AUTOTUNER_MEMORY_PATCH}" \
    || { echo "FlashInfer PR #3912 patch failed" >&2; exit 1; }

# FlashInfer #3187 exposes distributed tactic synchronization as an opt-in API.
# Wire vLLM's multi-rank warmup to the existing gloo world group so every rank
# reduces the same profile timings before selecting a tactic.
VLLM_AUTOTUNER_GROUP_PATCH="$(dirname "$0")/patches/vllm-flashinfer-autotune-process-group.patch"
patch --dry-run -p1 -d "${SITE_PACKAGES}" < "${VLLM_AUTOTUNER_GROUP_PATCH}" >/dev/null \
    || { echo "vLLM FlashInfer AutoTuner process-group patch does not apply" >&2; exit 1; }
patch -p1 -d "${SITE_PACKAGES}" < "${VLLM_AUTOTUNER_GROUP_PATCH}" \
    || { echo "vLLM FlashInfer AutoTuner process-group patch failed" >&2; exit 1; }

# -----------------------------------------------------------------------------

if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

nvidia-smi

SERVER_LOG=/workspace/server.log

export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_FLOAT32_MATMUL_PRECISION=high
export VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm
export VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS='flashinfer::trtllm_fp4_block_scale_moe'
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=300

if [ "${DP_ATTENTION}" = "true" ]; then
  PARALLEL_ARGS="--tensor-parallel-size=1 --data-parallel-size=$TP --enable-expert-parallel"
elif [ "$EP_SIZE" -gt 1 ]; then
  PARALLEL_ARGS="--tensor-parallel-size=$TP --enable-expert-parallel"
else
  PARALLEL_ARGS="--tensor-parallel-size=$TP"
fi

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
fi
start_gpu_monitor

set -x
vllm serve "$MODEL_PATH" --served-model-name "$MODEL" --host 0.0.0.0 --port $PORT \
$PARALLEL_ARGS \
--attention_config.indexer_kv_dtype fp8 \
--gpu-memory-utilization 0.95 \
--max-model-len $MAX_MODEL_LEN \
--kv-cache-dtype fp8 \
--block-size 128 \
--language-model-only \
--max-cudagraph-capture-size 2048 \
--max-num-batched-tokens "$((ISL * 2 ))" \
--stream-interval 20 --no-enable-prefix-caching \
--enable-flashinfer-autotune \
--trust-remote-code > $SERVER_LOG 2>&1 &

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
