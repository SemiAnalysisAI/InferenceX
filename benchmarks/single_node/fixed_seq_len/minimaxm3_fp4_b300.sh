#!/usr/bin/env bash

# MiniMax-M3 NVFP4 B300 single-node vLLM recipe.
# Same shape as minimaxm3_fp8_b300.sh but uses the nvidia/MiniMax-M3-NVFP4
# checkpoint. MiniMax-M3 modelopt NVFP4 support (vllm-project/vllm PR #46380) is
# baked into the perf container image.
#
# At runtime the recipe swaps the image's FlashInfer for the pinned nightly
# release and applies the CuTeDSL split-K gemm patch on top, mirroring vLLM
# commits 82a00090a (nightly install) and 0a16bea6b (patch).

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

# --- FlashInfer nightly + CuTeDSL split-K gemm patch -------------------------
# Pinned to nightly release nightly-v0.6.14-20260702. The wheels are not yet on
# the flashinfer.ai/whl/nightly flat index, so install by direct release URL.
FLASHINFER_VERSION=0.6.14.dev20260702
FLASHINFER_NIGHTLY_TAG=nightly-v0.6.14-20260702
FLASHINFER_RELEASE_URL="https://github.com/flashinfer-ai/flashinfer/releases/download/${FLASHINFER_NIGHTLY_TAG}"

# The flashinfer-jit-cache wheel is CUDA-version-specific; derive cuXYZ from
# the torch build inside the container (e.g. 13.0 -> cu130).
CUDA_MAJOR_MINOR="cu$(python3 -c 'import torch; print(torch.version.cuda.replace(".", ""))')" \
    || { echo "Failed to determine CUDA version from torch" >&2; exit 1; }

python3 -m pip uninstall -y flashinfer-python flashinfer-cubin flashinfer-jit-cache

python3 -m pip install --pre \
    "${FLASHINFER_RELEASE_URL}/flashinfer_python-${FLASHINFER_VERSION}-py3-none-any.whl" \
    "${FLASHINFER_RELEASE_URL}/flashinfer_cubin-${FLASHINFER_VERSION}-py3-none-any.whl" \
    "${FLASHINFER_RELEASE_URL}/flashinfer_jit_cache-${FLASHINFER_VERSION}+${CUDA_MAJOR_MINOR}-cp39-abi3-manylinux_2_28_$(uname -m).whl" \
    || { echo "FlashInfer nightly install failed" >&2; exit 1; }

# # Apply CuTeDSL split-K gemm patch (jiahanc/flashinfer branch cutedsl-splitK,
# # flashinfer/gemm changes only) on top of the pinned nightly. The reinstall
# # above guarantees pristine files, so the patch always applies cleanly.
# FLASHINFER_PATCH="$(dirname "$0")/patches/flashinfer-cutedsl-splitk-gemm.patch"
# if ! command -v patch >/dev/null 2>&1; then
#     apt-get update -y && apt-get install -y --no-install-recommends patch \
#         || { echo "Failed to install patch(1)" >&2; exit 1; }
# fi
# SITE_PACKAGES=$(dirname "$(python3 -c "import importlib.util; print(importlib.util.find_spec('flashinfer').submodule_search_locations[0])")") \
#     || { echo "Could not locate the installed flashinfer package" >&2; exit 1; }
# patch -p1 -d "${SITE_PACKAGES}" < "$FLASHINFER_PATCH" \
#     || { echo "FlashInfer CuTeDSL split-K gemm patch failed" >&2; exit 1; }
# # -----------------------------------------------------------------------------

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
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800

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
