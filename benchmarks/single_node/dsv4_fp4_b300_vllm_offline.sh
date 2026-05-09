#!/usr/bin/env bash

# DeepSeek-V4-Pro B300 single-node vLLM **offline** benchmark.
# Replicates cann-recipes-infer/models/deepseek-v4/infer.sh shape:
# in-process vllm.LLM, one warmup batch + one timed batch (lockstep),
# no HTTP server. Bypasses the serving harness entirely so the timing
# matches CANN's offline `model_generate` reference.

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

NUM_SPEC_TOKENS="$(dsv4_mtp_spec_tokens_for_spec_decoding)"
DPA_FLAG=()
[[ "${DP_ATTENTION}" == "true" ]] && DPA_FLAG=(--dp-attn)
EP_SIZE="${EP_SIZE:-1}"

start_gpu_monitor

# Engine workers spawn fresh python interpreters that don't inherit
# sys.path tweaks from run_offline.py — export PYTHONPATH so they can
# import utils.bench_offline.* during deserialization.
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
