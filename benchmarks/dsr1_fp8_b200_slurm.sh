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
# Enable FlashInfer MoE kernels with FP8 path on NVIDIA
export VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1

# Defaults for sweep if not provided by workflow
RESULT_PREFIX=${RESULT_PREFIX:-dsv32_b200}
DEFAULT_BATCH_LIST="4 16 64"

set -euo pipefail
set -x

# Build matrix of (ISL, OSL) pairs
declare -a MATRIX
if [[ -n "${TEST_MATRIX:-}" ]]; then
  # Comma-separated pairs like "1024:1024,1024:8192,8192:1024"
  IFS=',' read -ra pairs <<< "$TEST_MATRIX"
  for p in "${pairs[@]}"; do MATRIX+=("$p"); done
elif [[ -n "${ISL_LIST:-}" || -n "${OSL_LIST:-}" ]]; then
  IFS=' ' read -ra isl_list <<< "${ISL_LIST:-1024}"
  IFS=' ' read -ra osl_list <<< "${OSL_LIST:-1024}"
  for isl in "${isl_list[@]}"; do
    for osl in "${osl_list[@]}"; do
      MATRIX+=("${isl}:${osl}")
    done
  done
else
  MATRIX=("1024:1024" "1024:8192" "8192:1024")
fi

IFS=' ' read -ra batch_list <<< "${BATCH_LIST:-$DEFAULT_BATCH_LIST}"

for pair in "${MATRIX[@]}"; do
  isl=${pair%%:*}
  osl=${pair##*:}

  if [[ "$isl" == "1024" && "$osl" == "1024" ]]; then
    CALCULATED_MAX_MODEL_LEN=$((isl + osl + 20))
  elif [[ "$isl" == "8192" || "$osl" == "8192" ]]; then
    CALCULATED_MAX_MODEL_LEN=$((isl + osl + 200))
  else
    CALCULATED_MAX_MODEL_LEN=$((isl + osl + 128))
  fi

  for bs in "${batch_list[@]}"; do
    export ISL="$isl"
    export OSL="$osl"
    export CALCULATED_MAX_MODEL_LEN
    export BATCH_SIZE="$bs"
    export NUM_PROMPTS="$bs"        # single-shot offline run with batch=bs
    export CONC="$bs"               # recorded as max_concurrency in results
    export RESULT_FILENAME="${RESULT_PREFIX}_isl_${isl}_osl_${osl}_bs${bs}_tp${TP}"

    python3 utils/offline_benchmark_vllm.py
  done
done
