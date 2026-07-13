#!/usr/bin/env bash

# MiniMax-M3 NVFP4 B300 single-node vLLM recipe.
# Same shape as minimaxm3_fp8_b300.sh but uses the nvidia/MiniMax-M3-NVFP4
# checkpoint. MiniMax-M3 modelopt NVFP4 support (vllm-project/vllm PR #46380) is
# baked into the perf container image.
#
# At runtime the recipe swaps the image's FlashInfer for the first pinned
# nightly containing the upstream SM100 low-M MXFP8 split-K kernel
# (flashinfer-ai/flashinfer#3847), then restores the pre-#3687 AutoTuner,
# reverts the communication workspace changes from #3745, and restores the
# pre-#3582 trtllm-gen KV counter layout.

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
FLASHINFER_VERSION=0.6.15.dev20260710
FLASHINFER_NIGHTLY_TAG=nightly-v0.6.15-20260710
FLASHINFER_RELEASE_URL="https://github.com/flashinfer-ai/flashinfer/releases/download/${FLASHINFER_NIGHTLY_TAG}"

python3 -m pip uninstall -y flashinfer-python flashinfer-cubin flashinfer-jit-cache

python3 -m pip install \
    "${FLASHINFER_RELEASE_URL}/flashinfer_python-${FLASHINFER_VERSION}-py3-none-any.whl" \
    "${FLASHINFER_RELEASE_URL}/flashinfer_cubin-${FLASHINFER_VERSION}-py3-none-any.whl" \
    "${FLASHINFER_RELEASE_URL}/flashinfer_jit_cache-${FLASHINFER_VERSION}+cu130-cp39-abi3-manylinux_2_28_$(uname -m).whl" \
    || { echo "FlashInfer nightly install failed" >&2; exit 1; }

# Reverse all runtime changes from flashinfer-ai/flashinfer#3687 to restore the
# 0708 AutoTuner implementation and its original call sites. This intentionally
# does not apply the later flashinfer-ai/flashinfer#3918 guard fix.
AUTOTUNER_REVERT_PATCH="$(dirname "$0")/patches/flashinfer-revert-pr-3687.patch"
if ! command -v patch >/dev/null 2>&1; then
    apt-get update -y && apt-get install -y --no-install-recommends patch \
        || { echo "Failed to install patch(1)" >&2; exit 1; }
fi
SITE_PACKAGES=$(dirname "$(python3 -c "import importlib.util; print(importlib.util.find_spec('flashinfer').submodule_search_locations[0])")") \
    || { echo "Could not locate the installed flashinfer package" >&2; exit 1; }
patch --dry-run -p1 -d "${SITE_PACKAGES}" < "${AUTOTUNER_REVERT_PATCH}" >/dev/null \
    || { echo "FlashInfer PR #3687 revert patch does not apply" >&2; exit 1; }
patch -p1 -d "${SITE_PACKAGES}" < "${AUTOTUNER_REVERT_PATCH}" \
    || { echo "FlashInfer PR #3687 revert patch failed" >&2; exit 1; }

# Reverse the runtime communication changes from flashinfer-ai/flashinfer#3745
# while leaving the 0.6.15 GEMM, MoE, and attention code unchanged.
COMM_REVERT_PATCH="$(dirname "$0")/patches/flashinfer-revert-pr-3745.patch"
patch --dry-run -p1 -d "${SITE_PACKAGES}" < "${COMM_REVERT_PATCH}" >/dev/null \
    || { echo "FlashInfer PR #3745 revert patch does not apply" >&2; exit 1; }
patch -p1 -d "${SITE_PACKAGES}" < "${COMM_REVERT_PATCH}" \
    || { echo "FlashInfer PR #3745 revert patch failed" >&2; exit 1; }

# Reverse flashinfer-ai/flashinfer#3582 so trtllm-gen KV counters use the
# original shared workspace layout for this performance bisect.
ATTN_REVERT_PATCH="$(dirname "$0")/patches/flashinfer-revert-pr-3582.patch"
patch --dry-run -p1 -d "${SITE_PACKAGES}" < "${ATTN_REVERT_PATCH}" >/dev/null || { echo "FlashInfer PR #3582 revert patch does not apply" >&2; exit 1; }
patch -p1 -d "${SITE_PACKAGES}" < "${ATTN_REVERT_PATCH}" || { echo "FlashInfer PR #3582 revert patch failed" >&2; exit 1; }

# flashinfer-jit-cache ships an AOT fmha_gen.so built with the post-#3582 ABI.
# Remove it and any runtime JIT copy so the patched launcher is rebuilt.
FMHA_GEN_AOT_DIR="$(python3 -c "from flashinfer.jit import env as e; print(e.FLASHINFER_AOT_DIR)")/fmha_gen"
FMHA_GEN_JIT_DIR="$(python3 -c "from flashinfer.jit import env as e; print(e.FLASHINFER_JIT_DIR)")/fmha_gen"
if [[ "${FMHA_GEN_AOT_DIR##*/}" != "fmha_gen" || "${FMHA_GEN_JIT_DIR##*/}" != "fmha_gen" ]]; then
    echo "Refusing to remove unexpected FlashInfer fmha_gen cache paths" >&2
    exit 1
fi
rm -rf "${FMHA_GEN_AOT_DIR}" "${FMHA_GEN_JIT_DIR}"

# CUDA pip packages keep NVRTC outside /usr/local/cuda. Link only nvrtc.h into
# the active CUDA toolkit so nvcc does not mix the pip CUDA package's full header
# tree with /usr/local/cuda headers.
NVIDIA_CU13_ROOT=$(python3 -c 'import pathlib, site; print(next(pathlib.Path(root) / "nvidia" / "cu13" for root in site.getsitepackages() if (pathlib.Path(root) / "nvidia" / "cu13" / "include" / "nvrtc.h").is_file()))') || { echo "Could not locate the pip-installed CUDA 13 NVRTC package" >&2; exit 1; }
ln -sfn "${NVIDIA_CU13_ROOT}/include/nvrtc.h" /usr/local/cuda/include/nvrtc.h \
    || { echo "Failed to link nvrtc.h into /usr/local/cuda/include" >&2; exit 1; }
export LIBRARY_PATH="${NVIDIA_CU13_ROOT}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LD_LIBRARY_PATH="${NVIDIA_CU13_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

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
