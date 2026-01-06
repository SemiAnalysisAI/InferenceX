#!/usr/bin/env bash

# === Required Env Vars ===
# MODEL
# PORT
# TP
# CONC
# ISL
# OSL
# RANDOM_RANGE_RATIO
# RESULT_FILENAME
# EP_SIZE
# NUM_PROMPTS

nvidia-smi

# To improve CI stability, we patch this helper function to prevent a race condition that
# happens 1% of the time. ref: https://github.com/flashinfer-ai/flashinfer/pull/1779
sed -i '102,108d' /usr/local/lib/python3.12/dist-packages/flashinfer/jit/cubin_loader.py

export TORCH_CUDA_ARCH_LIST="10.0"
export PYTHONNOUSERSITE=1
export TP=8
# Enable FlashInfer MoE kernels with FP8 path on NVIDIA
export VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1

# Defaults for sweep if not provided by workflow
RESULT_PREFIX=${RESULT_PREFIX:-dsv32_b200}
DEFAULT_BATCH_LIST="4 16 64"

set -euo pipefail
set -x

# Persist HF caches inside container (works with mounted host cache)
export HF_HUB_CACHE=${HF_HUB_CACHE:-${HF_HOME:-/hf-hub-cache}}
export HF_HOME=${HF_HOME:-$HF_HUB_CACHE}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HF_HUB_CACHE}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HUB_CACHE}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HUB_CACHE}

# Force single-session mode only: reuse one LLM across entire matrix inside Python.
IFS=' ' read -ra batch_list <<< "${BATCH_LIST:-$DEFAULT_BATCH_LIST}"
export SINGLE_SESSION=1
export BATCH_LIST="${batch_list[*]}"
python3 utils/offline_benchmark_vllm.py
