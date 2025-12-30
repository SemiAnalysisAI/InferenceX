#!/usr/bin/bash

# This script sets up the environment and launches multi-node benchmarks

set -x

# Set up environment variables for SLURM
export SLURM_PARTITION="batch"
export SLURM_ACCOUNT="benchmark"
export SLURM_JOB_NAME="benchmark-dynamo.job"

# For SGLang - we are working on updating the 8k1k configs 
# For now we add conditionals to this script to use newer code for the 1k1k configs

### FRAMEWORK_DIFF_IF_STATEMENT #1 - difference in setting up envvars
SQUASH_FILE="/mnt/lustre01/users/sa-shared/images/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
srun --partition=$SLURM_PARTITION --exclusive --time=180 bash -c "enroot import -o $SQUASH_FILE docker://$IMAGE"

# Update the IMAGE variable to the squash file
export IMAGE=$SQUASH_FILE

# MODEL_PATH is set in `nvidia-master.yaml` or any other yaml files
export MODEL_PATH=$MODEL

if [[ $FRAMEWORK == "dynamo-sglang" ]]; then
    export CONFIG_DIR="/mnt/lustre01/artifacts/sglang-configs/1k1k"
    export SGL_SLURM_JOBS_PATH="dynamo/examples/backends/sglang/slurm_jobs"
else
    export SERVED_MODEL_NAME="deepseek-r1-fp4"
fi

export ISL="$ISL"
export OSL="$OSL"

bash benchmarks/"${EXP_NAME%%_*}_${PRECISION}_gb200_${FRAMEWORK}_slurm.sh"

# Wait for all jobs to complete
echo "Waiting for all jobs to complete..."
while [ -n "$(squeue -u $USER --noheader --format='%i')" ]; do
    echo "Jobs still running..."
    squeue --steps -u $USER
    sleep 30
done

# FIXME: The below is bad and is a result of the indirection of the ways in which
# Dynamo jobs are launched. In a follow-up PR, the location of the result file should not
# depend on the runner, it should always be in the same spot in the GH workspace.

# Process results from all configurations
if [[ $FRAMEWORK == "dynamo-trt" ]]; then

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
else # search for "FRAMEWORK_DIFF_IF_STATEMENT #3" for this if-statement
    # Find the latest log directory that contains the data
    cat > collect_latest_results.py <<'PY'
import os, sys
sgl_job_dir, isl, osl, nexp = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
for path in sorted([f"{sgl_job_dir}/logs/{name}/vllm_isl_{isl}_osl_{osl}" for name in os.listdir(f"{sgl_job_dir}/logs/") if os.path.isdir(f"{sgl_job_dir}/logs/{name}/vllm_isl_{isl}_osl_{osl}")], key=os.path.getmtime, reverse=True)[:nexp]:
    print(path)
PY

    LOGS_DIR=$(python3 collect_latest_results.py "$SGL_SLURM_JOBS_PATH" $ISL $OSL 1)
    if [ -z "$LOGS_DIR" ]; then
        echo "No logs directory found for ISL=${ISL}, OSL=${OSL}"
        exit 1
    fi

    echo "Found logs directory: $LOGS_DIR"
    ls -la $LOGS_DIR

    # Result JSON are contained within the result directory
    for result_file in $(find $LOGS_DIR -type f); do
        # result_file should directly be isl_ISL_osl_OSL_concurrency_CONC_req_rate_R_gpus_N_ctx_M_gen_N.json
        file_name=$(basename $result_file)
        if [ -f $result_file ]; then
            # Copy the result file to workspace with a unique name
            WORKSPACE_RESULT_FILE="$GITHUB_WORKSPACE/${RESULT_FILENAME}_${file_name}"
            echo "Found result file ${result_file}. Copying them to ${WORKSPACE_RESULT_FILE}"
            cp $result_file $WORKSPACE_RESULT_FILE
        fi
    done
fi

echo "All result files processed"
