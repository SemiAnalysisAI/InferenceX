#!/usr/bin/env bash

# DeepSeek-V4-Pro H200 single-node vLLM **offline** benchmark.
# FP8 variant of dsv4_fp4_b300_vllm_offline.sh — same lockstep offline
# harness (warmup + timed batch, no HTTP server), adapted for H200's
# FP8 quantization and Hopper-specific vLLM flags.

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    DP_ATTENTION \
    CONC \
    ISL \
    OSL \
    MAX_MODEL_LEN \
    RESULT_FILENAME

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

nvidia-smi
hf download "$MODEL"

export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0

NUM_SPEC_TOKENS="${DSV4_MTP_SPEC_TOKENS:-2}"
DPA_FLAG=()
[[ "${DP_ATTENTION}" == "true" ]] && DPA_FLAG=(--dp-attn)
EP_SIZE="${EP_SIZE:-1}"

start_gpu_monitor

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$PWD"

set -x
python3 utils/bench_offline/run_offline.py \
    --engine vllm \
    --model "$MODEL" \
    --tp "$TP" \
    --ep "$EP_SIZE" \
    --num-chips "$TP" \
    --max-model-len "$MAX_MODEL_LEN" \
    --mtp "$NUM_SPEC_TOKENS" \
    --temperature 1.0 \
    --infinitebench-input-len "$ISL" \
    --infinitebench-output-len 256 \
    --batch-size "$CONC" \
    --result-dir /workspace/ \
    --result-filename "$RESULT_FILENAME" \
    --metadata "benchmark_input_len=$ISL" "benchmark_output_len=256" \
    "${DPA_FLAG[@]}"
set +x

stop_gpu_monitor
