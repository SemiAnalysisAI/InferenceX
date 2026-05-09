#!/usr/bin/env bash

# DeepSeek-V4-Pro B300 single-node TRT-LLM **offline** benchmark via
# tensorrt_llm.LLM. See dsv4_fp4_b300_vllm_offline.sh for rationale.

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    MAX_MODEL_LEN \
    RESULT_FILENAME \
    DP_ATTENTION \
    EP_SIZE

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

if [[ "$MODEL" != /* ]]; then
    hf download "$MODEL"
fi

nvidia-smi

NUM_SPEC_TOKENS="$(dsv4_mtp_spec_tokens_for_spec_decoding)"
DPA_FLAG=()
[[ "${DP_ATTENTION}" == "true" ]] && DPA_FLAG=(--dp-attn)
export TRTLLM_MHC_ENABLE_FUSED_HC="${TRTLLM_MHC_ENABLE_FUSED_HC:-1}"

start_gpu_monitor --output "$PWD/gpu_metrics.csv"

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$PWD"

# tensorrt_llm.LLM internally manages worker ranks but expects to be launched
# under mpirun (same wrapper trtllm-serve uses). Without this, MPI_Init_thread
# fails before the LLM constructor returns.
RUN_CMD=(
    python3 utils/bench_offline/run_offline.py
    --engine trt
    --model "$MODEL"
    --tp "$TP"
    --ep "$EP_SIZE"
    --num-chips "$TP"
    --max-model-len "$MAX_MODEL_LEN"
    --mtp "$NUM_SPEC_TOKENS"
    --temperature 1.0
    --infinitebench-input-len "$ISL"
    --infinitebench-output-len 256
    --batch-size "$CONC"
    --result-dir "$PWD/"
    --result-filename "$RESULT_FILENAME"
    --metadata "benchmark_input_len=$ISL" "benchmark_output_len=256"
    "${DPA_FLAG[@]}"
)

set -x
if [[ "${TRTLLM_DSV4_USE_MPIRUN:-1}" == "0" ]]; then
    "${RUN_CMD[@]}"
else
    mpirun -n 1 --oversubscribe --allow-run-as-root "${RUN_CMD[@]}"
fi
set +x

stop_gpu_monitor
