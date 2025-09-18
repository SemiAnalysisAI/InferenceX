#!/usr/bin/bash

# This script sets up the environment and launches multi-node benchmarks

# Set up environment variables for SLURM
export SLURM_PARTITION="batch"
export SLURM_ACCOUNT="benchmark"
export SLURM_JOB_NAME="benchmark-dynamo.job"
export IMAGE="/mnt/lustre01/users/sa-shared/images/dynamo-trtllm_v5.sqsh"
export MODEL_PATH="/mnt/lustre01/models/deepseek-r1-0528-fp4-v2"
export SERVED_MODEL_NAME="deepseek-r1-fp4"

export ISL="$ISL"
export OSL="$OSL"

# Set up Dynamo repository path
DYNAMO_PATH="/mnt/lustre01/users/sa-shared/benchmarks/dynamo"
PERFORMANCE_SWEEPS_PATH="$DYNAMO_PATH/components/backends/trtllm/performance_sweeps"

# Always clone and setup Dynamo
echo "Cloning Dynamo repository..."
rm -rf "$DYNAMO_PATH"
git clone https://github.com/csahithi/dynamo.git "$DYNAMO_PATH"
cd "$DYNAMO_PATH"
git checkout publish-result-json
git submodule update --init --recursive

# Navigate to performance sweeps directory
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

# Generate benchmark configurations based on ISL/OSL and MTP mode
generate_benchmark_configs() {
    local isl="$1"
    local osl="$2"
    local mtp_mode="$3"

    if [ "$isl" = "1024" ] && [ "$osl" = "1024" ]; then
        if [ "$mtp_mode" = "on" ]; then
            echo "Running 1k/1k MTP=ON configurations"

            echo "Running DEP 16GPU configuration..."
            ./submit_disagg.sh "mtp=on" "dep" 1 1 16 64 256 "0.7" 3 0 "512 1075"

            echo "Running DEP 2ctx-16GPU configuration..."
            ./submit_disagg.sh "mtp=on" "dep" 2 1 16 128 256 "0.7" 1 0 "2150"
        else
            echo "Running 1k/1k MTP=OFF configurations"

            echo "Running TEP configuration..."
            ./submit_disagg.sh "mtp=off" "tep" 1 4 8 128 128 "0.9" 0 0 "1 2 4 8 16 32 64 141"

            echo "Running DEP 2ctx-16GPU configuration..."
            ./submit_disagg.sh "mtp=off" "dep" 2 1 16 256 256 "0.75" 0 0 "2048 4300"
        fi
    elif [ "$isl" = "8192" ] && [ "$osl" = "1024" ]; then
        if [ "$mtp_mode" = "on" ]; then
            echo "Running 8k/1k MTP=ON configurations"

            echo "Running DEP 8ctx-16GPU configuration..."
            ./submit_disagg.sh "mtp=on" "dep" 8 1 16 64 256 "0.75" 2 0 "1075"

            echo "Running DEP 6ctx-8GPU configuration..."
            ./submit_disagg.sh "mtp=on" "dep" 6 1 8 256 512 "0.8" 1 0 "2150"      
        else
            echo "Running 8k/1k MTP=OFF configurations"

            echo "Running DEP 6ctx-16GPU configuration..."
            ./submit_disagg.sh "mtp=off" "dep" 6 1 16 64 64 "0.75" 0 0 "1075"

            echo "Running DEP 8ctx-16GPU configuration..."
            ./submit_disagg.sh "mtp=off" "dep" 8 1 16 128 128 "0.75" 0 0 "2150"
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
