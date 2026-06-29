#!/usr/bin/env bash

# DeepSeek-V4-Pro GB200 single-node vllm **offline** benchmark - offline decode-step
# (offline decode-step shape): in-process engine, one warmup batch +
# one timed lockstep batch of InfiniteBench prompts. ISL is 8192 input
# tokens; OSL is 256 *decode steps* (main-model forward passes), matching
# the decode-step target - MTP bonus tokens are
# excluded from the headline metrics by construction.
#
# STANDARDIZED single-node offline template (see utils/bench_offline/README.md).
# Every dsv4_*_offline.sh is byte-identical to this file except the SKU label
# on the comment line above and the --engine value below. The multi-node block
# and --nnodes/--node-rank args make a multi-tray DEP run (TP > GPUs/node) work
# unchanged when the launcher allocates more than one node; on a single node
# SLURM_NNODES/SLURM_PROCID default to 1/0 and every added line is a no-op.

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
    RESULT_FILENAME

if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

nvidia-smi

NUM_SPEC_TOKENS="$(dsv4_mtp_spec_tokens_for_spec_decoding)"
DPA_FLAG=()
[[ "${DP_ATTENTION}" == "true" ]] && DPA_FLAG=(--dp-attn)

# Multi-node DEP (TP > GPUs/node): the launcher allocates TP/gpus_per_node
# trays and runs this script once per tray via srun --ntasks-per-node=1, with
# SLURM_PROCID as the DP node_rank and SLURM_NNODES as the node count.
# MASTER_ADDR is resolved + exported by the launcher (compute nodes lack
# scontrol). All trays share $PWD (shared FS), so suffix the per-node GPU
# metrics file to avoid clobber — node 0 keeps the canonical name the harness
# collects. On a single node this whole block is a no-op.
if [[ "${SLURM_NNODES:-1}" -gt 1 ]]; then
    export MASTER_PORT=${MASTER_PORT:-29501}
    echo "Multi-node: MASTER_ADDR=${MASTER_ADDR:-unset} MASTER_PORT=$MASTER_PORT node_rank=$SLURM_PROCID"
fi
GPU_METRICS_OUT="$PWD/gpu_metrics.csv"
[[ "${SLURM_PROCID:-0}" -gt 0 ]] && GPU_METRICS_OUT="$PWD/gpu_metrics_node${SLURM_PROCID}.csv"
start_gpu_monitor --output "$GPU_METRICS_OUT"

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$PWD"

set -x
PYTHONNOUSERSITE=1 python3 utils/bench_offline/run_offline.py \
    --engine vllm \
    --model "$MODEL_PATH" \
    --tp "$TP" \
    --ep "$EP_SIZE" \
    --num-chips "$TP" \
    --max-model-len "$MAX_MODEL_LEN" \
    --mtp "$NUM_SPEC_TOKENS" \
    --temperature 1.0 \
    --infinitebench-input-len "$ISL" \
    --decode-steps "$OSL" \
    --routing-sim-strategy "${DSV4_OFFLINE_ROUTING_SIM:-uniform_random}" \
    --nnodes "${SLURM_NNODES:-1}" \
    --node-rank "${SLURM_PROCID:-0}" \
    --batch-size "$CONC" \
    --result-dir "$PWD/" \
    --result-filename "$RESULT_FILENAME" \
    --metadata "benchmark_input_len=$ISL" "benchmark_output_len=$OSL" \
    "${DPA_FLAG[@]}"
set +x

stop_gpu_monitor
