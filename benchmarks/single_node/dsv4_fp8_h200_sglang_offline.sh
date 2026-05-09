#!/usr/bin/env bash

# DeepSeek-V4-Pro H200 single-node SGLang **offline** benchmark via sgl.Engine.
# FP8 variant of dsv4_fp4_b300_sglang_offline.sh — uses marlin MoE backend
# (Hopper FP8) instead of flashinfer_mxfp4 (Blackwell FP4).

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    MAX_MODEL_LEN \
    RESULT_FILENAME

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

if [[ "$MODEL" != /* ]]; then
    hf download "$MODEL"
fi

nvidia-smi

NUM_SPEC_TOKENS="$(dsv4_mtp_spec_tokens_for_spec_decoding)"
EP_SIZE="${EP_SIZE:-1}"
DPA_FLAG=()
[[ "${DP_ATTENTION}" == "true" ]] && DPA_FLAG=(--dp-attn)
start_gpu_monitor --output "$PWD/gpu_metrics.csv"

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$PWD"

set -x
PYTHONNOUSERSITE=1 python3 utils/bench_offline/run_offline.py \
    --engine sglang \
    --model "$MODEL" \
    --tp "$TP" \
    --ep "$EP_SIZE" \
    --num-chips "$TP" \
    --max-model-len "$MAX_MODEL_LEN" \
    --mtp "$NUM_SPEC_TOKENS" \
    --moe-runner-backend marlin \
    --temperature 1.0 \
    --infinitebench-input-len "$ISL" \
    --infinitebench-output-len 256 \
    --batch-size "$CONC" \
    --result-dir "$PWD/" \
    --result-filename "$RESULT_FILENAME" \
    --metadata "benchmark_input_len=$ISL" "benchmark_output_len=256" \
    "${DPA_FLAG[@]}"
set +x

stop_gpu_monitor
