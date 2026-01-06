#!/usr/bin/env bash

# === Required Env Vars ===
# MODEL
# TP
# (Optional sweep controls) TEST_MATRIX, ISL_LIST, OSL_LIST, BATCH_LIST, RESULT_PREFIX

export VLLM_USE_AITER_UNIFIED_ATTENTION=1
export VLLM_ROCM_USE_AITER_MHA=0
export VLLM_ROCM_USE_AITER_FUSED_MOE_A16W4=1
# DeepSeek R1 FP8 on ROCm Sparse MLA requires KV block size = 1
export BLOCK_SIZE=${BLOCK_SIZE:-1}
export TP=8
# Persist HF caches inside container (works with mounted host cache)
export HF_HUB_CACHE=${HF_HUB_CACHE:-${HF_HOME:-/hf-hub-cache}}
export HF_HOME=${HF_HOME:-$HF_HUB_CACHE}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HF_HUB_CACHE}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HUB_CACHE}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HUB_CACHE}
if [[ -n "${ROCR_VISIBLE_DEVICES:-}" && -z "${HIP_VISIBLE_DEVICES:-}" ]]; then
  export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

RESULT_PREFIX=${RESULT_PREFIX:-dsv32_mi355x}
DEFAULT_BATCH_LIST="4 16 64"

set -euo pipefail
set -x

declare -a MATRIX
if [[ -n "${TEST_MATRIX:-}" ]]; then
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
# Force single-session mode: reuse one LLM across the entire matrix.
export SINGLE_SESSION=1
export BATCH_LIST="${batch_list[*]}"
python3 utils/offline_benchmark_vllm.py
