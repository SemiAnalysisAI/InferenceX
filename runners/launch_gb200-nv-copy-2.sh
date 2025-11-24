#!/usr/bin/bash

# This script sets up the environment and launches multi-node benchmarks

set -x

# Set up environment variables for SLURM
export SLURM_PARTITION="batch"
export SLURM_ACCOUNT="benchmark"
export SLURM_JOB_NAME="benchmark-dynamo.job"

### FRAMEWORK_DIFF_IF_STATEMENT #1 - difference in setting up envvars
if [[ $FRAMEWORK == "dynamo-sglang" ]]; then
    export IMAGE="/mnt/lustre01/artifacts/containers/dynamo-sglang.sqsh"
    export MODEL_PATH="/mnt/lustre01/models/deepseek-r1-0528"
    export CONFIG_DIR="/mnt/lustre01/artifacts/sglang-configs/1k1k"
else
    SQUASH_FILE="/mnt/lustre01/users/sa-shared/images/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
    srun --partition=$SLURM_PARTITION --exclusive --time=180 bash -c "enroot import -o $SQUASH_FILE docker://$IMAGE"

    # Update the IMAGE variable to the squash file
    export IMAGE=$SQUASH_FILE

    export MODEL_PATH="/mnt/lustre01/models/deepseek-r1-0528-fp4-v2"
    export SERVED_MODEL_NAME="deepseek-r1-fp4"
fi


export ISL="$ISL"
export OSL="$OSL"

bash benchmarks/"${EXP_NAME%%_*}_${PRECISION}_gb200_${FRAMEWORK}_slurm.sh"

# ### FRAMEWORK_DIFF_IF_STATEMENT #2 - difference in launching jobs
# if [[ $FRAMEWORK == "dynamo-trtllm" ]]; then

#     # Set up Dynamo repository path
#     DYNAMO_PATH="/mnt/lustre01/users/sa-shared/benchmarks/dynamo"
#     PERFORMANCE_SWEEPS_PATH="$DYNAMO_PATH/components/backends/trtllm/performance_sweeps"

#     # Overview:
#     # The Dynamo repository contains the bench_serving repository as a submodule.
#     # The submit_disagg.sh script, located at $PERFORMANCE_SWEEPS_PATH, orchestrates the entire benchmarking workflow:
#     #   1. Launches the Dynamo inference service with the specified configuration.
#     #   2. Waits for the service to become healthy.
#     #   3. Initiates benchmarking using the bench_serving tools.
#     #   4. Monitors all jobs until completion.
#     #   5. Collects and processes the results.

#     # Always clone and setup Dynamo
#     echo "Cloning Dynamo repository..."
#     rm -rf "$DYNAMO_PATH"
#     git clone https://github.com/cquil11/dynamo.git "$DYNAMO_PATH"
#     cd "$DYNAMO_PATH"
#     git checkout release/0.5.1-rc0.20251105-cam
#     git submodule update --init --recursive

#     # Navigate to performance sweeps directory
#     cd "$PERFORMANCE_SWEEPS_PATH"

#     # 1. CACHE_TRANSCEIVER_MAX_NUM_TOKENS controls the max_tokens_in_buffer value
#     # in cache_transceiver_config of TensorRT-LLM context and generation workers.
#     # Specifically, it is the max number of tokens the transfer buffer can fit.
    
#     # Set up environment variables based on ISL/OSL
#     if [ "$ISL" = "1024" ] && [ "$OSL" = "1024" ]; then
#         export CACHE_TRANSCEIVER_MAX_NUM_TOKENS=4608
#     elif [ "$ISL" = "8192" ] && [ "$OSL" = "1024" ]; then
#         export CACHE_TRANSCEIVER_MAX_NUM_TOKENS=8448
#     else
#         echo "Unsupported ISL/OSL combination: $ISL/$OSL"
#         exit 1
#     fi

#     # New stuff
#     # CONC_LIST
#     # ISL
#     # OSL
#     # IMAGE

#     # PREFILL_NUM_WORKERS
#     # PREFILL_TP
#     # PREFILL_EP
#     # PREFILL_DP_ATTN

#     # DECODE_NUM_WORKERS
#     # DECODE_TP
#     # DECODE_EP
#     # DECODE_DP_ATTN

#     # Additional env vars needed
#     # PREFILL_MAX_NUM_TOKENS
#     # PREFILL_MAX_BATCH_SIZE
#     # DECODE_MAX_NUM_TOKENS
#     # DECODE_MAX_BATCH_SIZE
#     # DECODE_GPU_MEM_FRACTION
#     # DECODE_MTP_SIZE
#     # DECODE_EPLB_NUM_SLOTS

#     echo "CONC_LIST=$CONC_LIST"
#     echo "ISL=$ISL"
#     echo "OSL=$OSL"
#     echo "IMAGE=$IMAGE"

#     echo "PREFILL_NUM_WORKERS=$PREFILL_NUM_WORKERS"
#     echo "PREFILL_TP=$PREFILL_TP"
#     echo "PREFILL_EP=$PREFILL_EP"
#     echo "PREFILL_DP_ATTN=$PREFILL_DP_ATTN"

#     echo "DECODE_NUM_WORKERS=$DECODE_NUM_WORKERS"
#     echo "DECODE_TP=$DECODE_TP"
#     echo "DECODE_EP=$DECODE_EP"
#     echo "DECODE_DP_ATTN=$DECODE_DP_ATTN"

#     echo "PREFILL_MAX_NUM_TOKENS=$PREFILL_MAX_NUM_TOKENS"
#     echo "PREFILL_MAX_BATCH_SIZE=$PREFILL_MAX_BATCH_SIZE"
#     echo "DECODE_MAX_NUM_TOKENS=$DECODE_MAX_NUM_TOKENS"
#     echo "DECODE_MAX_BATCH_SIZE=$DECODE_MAX_BATCH_SIZE"
#     echo "DECODE_GPU_MEM_FRACTION=$DECODE_GPU_MEM_FRACTION"
#     echo "DECODE_MTP_SIZE=$DECODE_MTP_SIZE"
#     echo "DECODE_EPLB_NUM_SLOTS=$DECODE_EPLB_NUM_SLOTS"

#     # For GB200, we use 4 tasks per node.
#     ntasks_per_node=4
#     additional_slurm_args="--time=04:00:00"

#     kind=dynamo_disagg

#     gen_nodes=$(((DECODE_TP + 3)/4 * DECODE_NUM_WORKERS))
#     total_nodes=$((PREFILL_NUM_WORKERS + gen_nodes))
#     total_tasks=$((total_nodes * ntasks_per_node))

#     # 4608 prefill max num toks originally
#     if [ $ISL == $OSL ]; then
#         sbatch --nodes=${total_nodes} \
#             --ntasks=${total_tasks} \
#             --ntasks-per-node=${ntasks_per_node} \
#             --segment=${total_nodes} ${additional_slurm_args} \
#             benchmark_disagg.slurm \
#             ${PREFILL_NUM_WORKERS} ${PREFILL_TP} \
#             ${PREFILL_MAX_BATCH_SIZE} ${PREFILL_MAX_NUM_TOKENS} \
#             ${PREFILL_DP_ATTN} ${DECODE_NUM_WORKERS} \
#             ${DECODE_TP} ${DECODE_MAX_BATCH_SIZE} \
#             ${DECODE_MAX_NUM_TOKENS} ${DECODE_DP_ATTN} \
#             ${DECODE_GPU_MEM_FRACTION} ${DECODE_EPLB_NUM_SLOTS} \
#             ${DECODE_MTP_SIZE} ${CONC_LIST} \
#             ${gen_nodes} ${kind} \
#             ${MODEL_PATH} ${SERVED_MODEL_NAME} \
#             ${IMAGE} ${ISL} ${OSL}
#     # else
#     #     sbatch --nodes=${total_nodes} --ntasks=${total_tasks} --ntasks-per-node=${ntasks_per_node} --segment=${total_nodes} ${slurm_args} benchmark_disagg.slurm ${ctx_num} 4 1 8448 true ${gen_num} ${gen_tp_size} ${gen_batch_size} ${gen_max_num_tokens} ${gen_enable_attention_dp} ${gen_gpu_memory_fraction} ${gen_eplb_num_slots} ${gen_mtp_size} "${gen_concurrency_list}" ${gen_nodes} ${kind} ${MODEL_PATH} ${SERVED_MODEL_NAME} ${IMAGE} ${ISL} ${OSL}
#     fi
# fi

# # Wait for all jobs to complete
# # echo "Waiting for all jobs to complete..."
# while [ -n "$(squeue -u $USER --noheader --format='%i')" ]; do
#     echo "Jobs still running..."
#     squeue --steps -u $USER
#     sleep 60
# done

# ### FRAMEWORK_DIFF_IF_STATEMENT #3 - difference in log post-processing
# if [[ $FRAMEWORK == "dynamo-trtllm" ]]; then

#     # Find the logs directory (should be only one for this ISL/OSL combination)
#     LOGS_DIR=$(find . -name "dynamo_disagg-bm-${ISL}-${OSL}" -type d | head -1)
#     if [ -z "$LOGS_DIR" ]; then
#         echo "No logs directory found for ISL=${ISL}, OSL=${OSL}"
#         exit 1
#     fi

#     echo "Found logs directory: $LOGS_DIR"

#     # Find all result subdirectories in this logs directory
#     RESULT_SUBDIRS=$(find "$LOGS_DIR" -name "ctx*_gen*_[td]ep*_batch*_eplb*_mtp*" -type d)

#     if [ -z "$RESULT_SUBDIRS" ]; then
#         echo "No result subdirectories found in $LOGS_DIR"
#         exit 1
#     fi

#     echo "Found result subdirectories:"
#     echo "$RESULT_SUBDIRS"

#     # Process results from all configurations
#     for result_subdir in $RESULT_SUBDIRS; do
#         echo "Processing result subdirectory: $result_subdir"

#         # Extract configuration info from directory name
#         CONFIG_NAME=$(basename "$result_subdir")

#         # Process individual concurrency result files
#         RESULTS_SUBDIR="$result_subdir/results"

#         if [ -d "$RESULTS_SUBDIR" ]; then
#             echo "Processing results from: $RESULTS_SUBDIR"

#             # Find all concurrency result files with new format
#             CONCURRENCY_FILES=$(find "$RESULTS_SUBDIR" -name "results_concurrency_*_gpus_*.json")

#             for result_file in $CONCURRENCY_FILES; do
#                 if [ -f "$result_file" ]; then
#                     # Extract concurrency and GPU count from filename
#                     filename=$(basename "$result_file")
#                     concurrency=$(echo "$filename" | sed 's/results_concurrency_\([0-9]*\)_gpus_.*\.json/\1/')
#                     gpus=$(echo "$filename" | sed 's/results_concurrency_.*_gpus_\([0-9]*\)\.json/\1/')
#                     echo "Processing concurrency $concurrency with $gpus GPUs: $result_file"

#                     # Copy the result file to workspace with a unique name
#                     WORKSPACE_RESULT_FILE="$GITHUB_WORKSPACE/${RESULT_FILENAME}_${CONFIG_NAME}_conc${concurrency}_gpus${gpus}.json"
#                     cp "$result_file" "$WORKSPACE_RESULT_FILE"

#                     echo "Copied result file to: $WORKSPACE_RESULT_FILE"
#                 fi
#             done
#         else
#             echo "Results subdirectory not found: $RESULTS_SUBDIR"
#         fi
#     done

# else # search for "FRAMEWORK_DIFF_IF_STATEMENT #3" for this if-statement
#     # Find the latest log directory
#     # we do "tail -1" here since only the latest job will yield the result
#     LOGS_DIR=$(find logs/*/vllm_isl_${ISL}_osl_${OSL} -type d | sort -V | tail -1)
#     if [ -z "$LOGS_DIR" ]; then
#         echo "No logs directory found for ISL=${ISL}, OSL=${OSL}"
#         exit 1
#     fi

#     echo "Found logs directory: $LOGS_DIR"
#     ls $LOGS_DIR

#     # Result JSON are contained within the result directory
#     for result_file in $(find $LOGS_DIR -type f); do
#         # result_file should directly be isl_ISL_osl_OSL_concurrency_CONC_req_rate_R_gpus_N_ctx_M_gen_N.json
#         file_name=$(basename $result_file)
#         if [ -f $result_file ]; then
#             # Copy the result file to workspace with a unique name
#             WORKSPACE_RESULT_FILE="$GITHUB_WORKSPACE/${RESULT_FILENAME}_${file_name}"
#             echo "Found result file ${result_file}. Copying them to ${WORKSPACE_RESULT_FILE}"
#             cp $result_file $WORKSPACE_RESULT_FILE
#         fi
#     done
# fi

# echo "All result files processed"