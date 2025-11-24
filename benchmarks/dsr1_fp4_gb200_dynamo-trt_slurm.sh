#!/usr/bin/bash

set -x

source "$(dirname "$0")/benchmark_lib.sh"

check_env_vars CONC_LIST ISL OSL IMAGE SPEC_DECODING \
    PREFILL_NUM_WORKERS PREFILL_TP PREFILL_EP PREFILL_DP_ATTN \
    DECODE_NUM_WORKERS DECODE_TP DECODE_EP DECODE_DP_ATTN \
    PREFILL_MAX_NUM_TOKENS PREFILL_MAX_BATCH_SIZE DECODE_MAX_NUM_TOKENS \
    DECODE_MAX_BATCH_SIZE DECODE_GPU_MEM_FRACTION DECODE_EPLB_NUM_SLOTS \
    NTASKS_PER_NODE

if [ "$SPEC_DECODING" == "mtp" ]; then
    check_env_vars DECODE_MTP_SIZE
else
    DECODE_MTP_SIZE="0"
fi

PERFORMANCE_SWEEPS_PATH="components/backends/trtllm/performance_sweeps"

echo "Cloning Dynamo repository..."
git clone https://github.com/cquil11/dynamo.git
cd dynamo
git checkout release/0.5.1-rc0.20251105-cam
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

gen_nodes=$(((DECODE_TP + 3)/4 * DECODE_NUM_WORKERS))
total_nodes=$((PREFILL_NUM_WORKERS + gen_nodes))
total_tasks=$((total_nodes * NTASKS_PER_NODE))

sbatch --nodes=${total_nodes} \
    --ntasks=${total_tasks} \
    --ntasks-per-node=${NTASKS_PER_NODE} \
    --segment=${total_nodes} ${additional_slurm_args} \
    benchmark_disagg.slurm \
    ${PREFILL_NUM_WORKERS} ${PREFILL_TP} \
    ${PREFILL_MAX_BATCH_SIZE} ${PREFILL_MAX_NUM_TOKENS} \
    ${PREFILL_DP_ATTN} ${DECODE_NUM_WORKERS} \
    ${DECODE_TP} ${DECODE_MAX_BATCH_SIZE} \
    ${DECODE_MAX_NUM_TOKENS} ${DECODE_DP_ATTN} \
    ${DECODE_GPU_MEM_FRACTION} ${DECODE_EPLB_NUM_SLOTS} \
    ${DECODE_MTP_SIZE} ${CONC_LIST} \
    ${gen_nodes} ${kind} \
    ${MODEL_PATH} ${SERVED_MODEL_NAME} \
    ${IMAGE} ${ISL} ${OSL}

# Wait for all jobs to complete
echo "Waiting for all jobs to complete..."
while [ -n "$(squeue -u $USER --noheader --format='%i')" ]; do
    echo "Jobs still running..."
    squeue --steps -u $USER
    sleep 30
done

# Find the logs directory (should be only one for this ISL/OSL combination)
LOGS_DIR=$(find . -name "dynamo_disagg-bm-${ISL}-${OSL}" -type d | head -1)
if [ -z "$LOGS_DIR" ]; then
    echo "No logs directory found for ISL=${ISL}, OSL=${OSL}"
    exit 1
fi

echo "Found logs directory: $LOGS_DIR"

# Find all result subdirectories in this logs directory
RESULT_SUBDIRS=$(find "$LOGS_DIR" -name "ctx*_gen*_[td]ep*_batch*_eplb*_mtp*" -type d)

if [ -z "$RESULT_SUBDIRS" ]; then
    echo "No result subdirectories found in $LOGS_DIR"
    exit 1
fi

echo "Found result subdirectories:"
echo "$RESULT_SUBDIRS"

# Process results from all configurations
for result_subdir in $RESULT_SUBDIRS; do
    echo "Processing result subdirectory: $result_subdir"

    # Extract configuration info from directory name
    CONFIG_NAME=$(basename "$result_subdir")

    # Process individual concurrency result files
    RESULTS_SUBDIR="$result_subdir/results"

    if [ -d "$RESULTS_SUBDIR" ]; then
        echo "Processing results from: $RESULTS_SUBDIR"

        # Find all concurrency result files with new format
        CONCURRENCY_FILES=$(find "$RESULTS_SUBDIR" -name "results_concurrency_*_gpus_*.json")

        for result_file in $CONCURRENCY_FILES; do
            if [ -f "$result_file" ]; then
                # Extract concurrency and GPU count from filename
                filename=$(basename "$result_file")
                concurrency=$(echo "$filename" | sed 's/results_concurrency_\([0-9]*\)_gpus_.*\.json/\1/')
                gpus=$(echo "$filename" | sed 's/results_concurrency_.*_gpus_\([0-9]*\)\.json/\1/')
                echo "Processing concurrency $concurrency with $gpus GPUs: $result_file"

                # Copy the result file to workspace with a unique name
                WORKSPACE_RESULT_FILE="$GITHUB_WORKSPACE/${RESULT_FILENAME}_${CONFIG_NAME}_conc${concurrency}_gpus${gpus}.json"
                cp "$result_file" "$WORKSPACE_RESULT_FILE"

                echo "Copied result file to: $WORKSPACE_RESULT_FILE"
            fi
        done
    else
        echo "Results subdirectory not found: $RESULTS_SUBDIR"
    fi
done