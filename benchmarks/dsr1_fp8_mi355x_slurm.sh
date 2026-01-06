#!/usr/bin/env bash

# === Required Env Vars ===
# MODEL
# TP
# (Optional sweep controls) TEST_MATRIX, ISL_LIST, OSL_LIST, BATCH_LIST, RESULT_PREFIX

export VLLM_USE_AITER_UNIFIED_ATTENTION=1
export VLLM_ROCM_USE_AITER_MHA=0
export VLLM_ROCM_USE_AITER_FUSED_MOE_A16W4=1
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
    export NUM_PROMPTS="$bs"
    export CONC="$bs"
    export RESULT_FILENAME="${RESULT_PREFIX}_isl_${isl}_osl_${osl}_bs${bs}_tp${TP}"

    python3 utils/offline_benchmark_vllm.py
  done
done
