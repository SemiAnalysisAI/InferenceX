#!/usr/bin/env bash

# DeepSeek-V4-Pro B300 single-node SGLang **offline** benchmark via sgl.Engine.
# See dsv4_fp4_b300_vllm_offline.sh for the rationale (CANN infer.sh shape,
# no HTTP server, one warmup + one timed lockstep batch).

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    EP_SIZE \
    DP_ATTENTION \
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

# Keep SPEC_DECODING=offline for launcher routing, but run this B300 SGLang
# InfiniteBench path with MTP enabled by default.
NUM_SPEC_TOKENS="${DSV4_MTP_SPEC_TOKENS:-3}"
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
