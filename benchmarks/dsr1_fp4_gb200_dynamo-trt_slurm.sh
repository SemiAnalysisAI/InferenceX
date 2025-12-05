#!/usr/bin/bash

set -x

source "$(dirname "$0")/benchmark_lib.sh"

check_env_vars CONC_LIST ISL OSL IMAGE SPEC_DECODING \
    PREFILL_NUM_WORKERS PREFILL_TP PREFILL_EP PREFILL_DP_ATTN \
    DECODE_NUM_WORKERS DECODE_TP DECODE_EP DECODE_DP_ATTN \
    PREFILL_MAX_NUM_TOKENS PREFILL_MAX_BATCH_SIZE DECODE_MAX_NUM_TOKENS \
    DECODE_MAX_BATCH_SIZE DECODE_GPU_MEM_FRACTION

if [ "$SPEC_DECODING" == "mtp" ]; then
    check_env_vars DECODE_MTP_SIZE
else
    DECODE_MTP_SIZE="0"
fi

PERFORMANCE_SWEEPS_PATH="components/backends/trtllm/performance_sweeps"

echo "Cloning Dynamo repository..."
git clone https://github.com/ai-dynamo/dynamo.git
cd dynamo
git checkout release/0.5.1-rc0.20251105
git submodule update --init --recursive

cd "$PERFORMANCE_SWEEPS_PATH"

# Set up environment variables based on ISL/OSL
if [ "$ISL" = "1024" ] && [ "$OSL" = "1024" ]; then
    export CACHE_TRANSCEIVER_MAX_NUM_TOKENS=4608
elif [ "$ISL" = "8192" ] && [ "$OSL" = "1024" ]; then
    export CACHE_TRANSCEIVER_MAX_NUM_TOKENS=8448
else
    echo "Unsupported ISL/OSL combination: $ISL/$OSL"
    exit 1
fi

kind=dynamo_disagg
additional_slurm_args="--time=04:00:00"
ntasks_per_node=4

gen_nodes=$(((DECODE_TP + 3)/4 * DECODE_NUM_WORKERS))
total_nodes=$((PREFILL_NUM_WORKERS + gen_nodes))
total_tasks=$((total_nodes * ntasks_per_node))

decode_eplb_num_slots=0

sbatch --nodes=${total_nodes} \
    --ntasks=${total_tasks} \
    --ntasks-per-node=${ntasks_per_node} \
    --segment=${total_nodes} ${additional_slurm_args} \
    benchmark_disagg.slurm \
    ${PREFILL_NUM_WORKERS} ${PREFILL_TP} \
    ${PREFILL_MAX_BATCH_SIZE} ${PREFILL_MAX_NUM_TOKENS} \
    ${PREFILL_DP_ATTN} ${DECODE_NUM_WORKERS} \
    ${DECODE_TP} ${DECODE_MAX_BATCH_SIZE} \
    ${DECODE_MAX_NUM_TOKENS} ${DECODE_DP_ATTN} \
    ${DECODE_GPU_MEM_FRACTION} ${decode_eplb_num_slots} \
    ${DECODE_MTP_SIZE} "${CONC_LIST}" \
    ${gen_nodes} ${kind} \
    ${MODEL_PATH} ${SERVED_MODEL_NAME} \
    ${IMAGE} ${ISL} ${OSL}