#!/usr/bin/bash

# This script sets up the environment and launches multi-node benchmarks

# Set up environment variables for SLURM
export SLURM_PARTITION="batch"
export SLURM_ACCOUNT="benchmark"
export SLURM_JOB_NAME="benchmark-dynamo.job"
# The original Docker image is available at: nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:0.5.1-rc0.pre3
# This path is to a pre-built squash file from the above image.
export IMAGE="/mnt/lustre01/users/sa-shared/images/dynamo-trtllm_v5.sqsh"
export MODEL_PATH="/mnt/lustre01/models/deepseek-r1-0528-fp4-v2"
export SERVED_MODEL_NAME="deepseek-r1-fp4"

export ISL="$ISL"
export OSL="$OSL"

# Set up Dynamo repository path
DYNAMO_PATH="/mnt/lustre01/users/sa-shared/benchmarks/dynamo"
PERFORMANCE_SWEEPS_PATH="$DYNAMO_PATH/components/backends/trtllm/performance_sweeps"

# Overview:
# The Dynamo repository contains the bench_serving repository as a submodule.
# The submit_disagg.sh script, located at $PERFORMANCE_SWEEPS_PATH, orchestrates the entire benchmarking workflow:
#   1. Launches the Dynamo inference service with the specified configuration.
#   2. Waits for the service to become healthy.
#   3. Initiates benchmarking using the bench_serving tools.
#   4. Monitors all jobs until completion.
#   5. Collects and processes the results.


# Always clone and setup Dynamo
echo "Cloning Dynamo repository..."
rm -rf "$DYNAMO_PATH"
git clone https://github.com/ai-dynamo/dynamo.git "$DYNAMO_PATH"
cd "$DYNAMO_PATH"
git checkout release/0.5.1-rc0.pre1
git submodule update --init --recursive

# Navigate to performance sweeps directory
cd "$PERFORMANCE_SWEEPS_PATH"

# Set up environment variables based on ISL/OSL
#
# 1. CACHE_TRANSCEIVER_MAX_NUM_TOKENS controls the max_tokens_in_buffer value
# in cache_transceiver_config of TensorRT-LLM context and generation workers.
# Specifically, it is the max number of tokens the transfer buffer can fit.
if [ "$ISL" = "1024" ] && [ "$OSL" = "1024" ]; then
    export CACHE_TRANSCEIVER_MAX_NUM_TOKENS=4608
elif [ "$ISL" = "8192" ] && [ "$OSL" = "1024" ]; then
    export CACHE_TRANSCEIVER_MAX_NUM_TOKENS=8448
else
    echo "Unsupported ISL/OSL combination: $ISL/$OSL"
    exit 1
fi

# Generate benchmark configurations based on ISL/OSL and MTP mode
generate_benchmark_configs() {
    local isl="$1"
    local osl="$2"
    local mtp_mode="$3"

    # Usage: 
    # ./submit_disagg.sh <mtp_mode> <mode> [ctx_num] [gen_num] [gen_tp_size] [gen_batch_size] [gen_max_num_tokens] [gen_gpu_memory_fraction] [gen_eplb_num_slots] [gen_mtp_size] [gen_concurrency_list]"
    # MTP Modes:
    #   mtp=off - Run without Multi-Token Prediction (gen_mtp_size=0)
    #   mtp=on  - Run with Multi-Token Prediction (gen_mtp_size=1,2,3)
    # Execution Modes:
    #   tep - Run Tensor-Expert Parallel mode (attention_dp=false)
    #   dep - Run Data-Expert Parallel mode (attention_dp=true)
    # Parameters for tep/dep modes:
    #   ctx_num: Number of context nodes
    #   gen_num: Number of generation nodes
    #   gen_tp_size: Generation tensor parallel size
    #   gen_batch_size: Generation batch size
    #   gen_max_num_tokens: Generation max number of tokens
    #   gen_gpu_memory_fraction: GPU memory fraction (0.7-0.95)
    #   gen_mtp_size: Multi-Token Prediction size (0 for mtp=off, 1-3 for mtp=on)
    #   gen_eplb_num_slots: Expert load balancing slots (0, 256, 288)
    #   gen_concurrency_list: Concurrency values (space-separated, quoted)

    if [ "$isl" = "1024" ] && [ "$osl" = "1024" ]; then
        if [ "$mtp_mode" = "on" ]; then
            echo "Running 1k/1k MTP=ON configurations"

            ./submit_disagg.sh "mtp=on" "tep" 1 4 8 32 128 "0.9" 3 0 "1 2 4 8 16 36"

            ./submit_disagg.sh "mtp=on" "dep" 1 1 16 64 256 "0.7" 3 0 "512 1075"

            ./submit_disagg.sh "mtp=on" "dep" 2 1 16 128 256 "0.7" 1 0 "2150"

            ./submit_disagg.sh "mtp=on" "dep" 1 1 32 16 64 "0.6" 3 0 "512"

            ./submit_disagg.sh "mtp=on" "dep" 1 1 8 256 512 "0.8" 1 0 "2252"
        else
            echo "Running 1k/1k MTP=OFF configurations"

            ./submit_disagg.sh "mtp=off" "tep" 1 4 8 128 128 "0.9" 0 0 "1 2 4 8 16 32 64 141"

            ./submit_disagg.sh "mtp=off" "dep" 1 1 32 32 32 "0.7" 0 0 "1075"

            ./submit_disagg.sh "mtp=off" "dep" 1 1 16 64 64 "0.75" 0 0 "1075"

            ./submit_disagg.sh "mtp=off" "dep" 2 1 16 256 256 "0.75" 0 0 "2048 4300"

            ./submit_disagg.sh "mtp=off" "dep" 1 1 8 512 512 "0.8" 0 0 "4300"
        fi
    elif [ "$isl" = "8192" ] && [ "$osl" = "1024" ]; then
        if [ "$mtp_mode" = "on" ]; then
            echo "Running 8k/1k MTP=ON configurations"

            ./submit_disagg.sh "mtp=on" "tep" 1 3 8 16 64 "0.9" 3 0 "1 2 4 8 18"

            ./submit_disagg.sh "mtp=on" "dep" 5 1 32 8 32 "0.7" 3 0 "128 269"

            ./submit_disagg.sh "mtp=on" "dep" 8 1 32 16 64 "0.7" 3 0 "538"

            ./submit_disagg.sh "mtp=on" "dep" 8 1 16 64 256 "0.75" 2 0 "1075"

            ./submit_disagg.sh "mtp=on" "dep" 6 1 8 256 512 "0.8" 1 0 "2150"
        else
            echo "Running 8k/1k MTP=OFF configurations"

            ./submit_disagg.sh "mtp=off" "tep" 1 3 8 32 32 "0.9" 0 0 "1 2 4 8 16 34"

            ./submit_disagg.sh "mtp=off" "dep" 4 1 32 16 16 "0.7" 0 0 "256 538"

            ./submit_disagg.sh "mtp=off" "dep" 6 1 16 64 64 "0.75" 0 0 "1075"

            ./submit_disagg.sh "mtp=off" "dep" 8 1 16 128 128 "0.75" 0 0 "2150"

            ./submit_disagg.sh "mtp=off" "dep" 5 1 8 256 256 "0.8" 0 0 "2150"
        fi
    else
        echo "Unsupported ISL/OSL combination: $isl/$osl"
        exit 1
    fi
}

# Run all benchmark configurations
generate_benchmark_configs "$ISL" "$OSL" "$MTP_MODE"

# Wait for all jobs to complete
echo "Waiting for all jobs to complete..."
while [ -n "$(squeue -u $USER --noheader --format='%i')" ]; do
    echo "Jobs still running..."
    squeue -u $USER
    sleep 60
done
echo "All jobs completed"

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

echo "All result files processed"
