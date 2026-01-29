#!/usr/bin/bash

# This script sets up the environment and launches multi-node benchmarks

set -x

# MODEL_PATH: Override with pre-downloaded paths on GB200 runner
# The yaml files specify HuggingFace model IDs for portability, but we use
# local paths to avoid repeated downloading on the shared GB200 cluster.
if [[ $FRAMEWORK == "dynamo-sglang" ]]; then
    export CONFIG_DIR="/mnt/lustre01/artifacts/sglang-configs/1k1k"
    if [[ $MODEL_PREFIX == "dsr1" ]]; then
        export MODEL_PATH="/mnt/lustre01/models/deepseek-r1-0528"
    else
        export MODEL_PATH=$MODEL
    fi
elif [[ $FRAMEWORK == "dynamo-trt" ]]; then
    if [[ $MODEL_PREFIX == "gptoss" ]]; then
        export MODEL_PATH="/mnt/lustre01/models/gpt-oss-120b"
        export SERVED_MODEL_NAME="gpt-oss-120b"
    elif [[ $MODEL_PREFIX == "dsr1" ]]; then
        export MODEL_PATH="/mnt/lustre01/models/deepseek-r1-0528-fp4-v2/"
        export SERVED_MODEL_NAME="deepseek-r1-fp4"
    else
        echo "Unsupported model prefix: $MODEL_PREFIX. Supported prefixes are: gptoss or dsr1"
        exit 1
    fi
else
    export MODEL_PATH=$MODEL
fi

# Set up environment variables for SLURM
export SLURM_PARTITION="batch"
export SLURM_ACCOUNT="benchmark"
export SLURM_JOB_NAME="benchmark-dynamo.job"

NGINX_IMAGE="nginx:1.27.4"

SQUASH_FILE="/mnt/lustre01/users-public/sa-shared/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
NGINX_SQUASH_FILE="/mnt/lustre01/users-public/sa-shared/$(echo "$NGINX_IMAGE" | sed 's/[\/:@#]/_/g').sqsh"

srun -N 1 -A $SLURM_ACCOUNT -p $SLURM_PARTITION bash -c "enroot import -o $SQUASH_FILE docker://$IMAGE"
srun -N 1 -A $SLURM_ACCOUNT -p $SLURM_PARTITION bash -c "enroot import -o $NGINX_SQUASH_FILE docker://$NGINX_IMAGE"



export ISL="$ISL"
export OSL="$OSL"

if [[ $FRAMEWORK == "dynamo-sglang" ]]; then
    export IMAGE=$SQUASH_FILE
    export SGL_SLURM_JOBS_PATH="dynamo/examples/backends/sglang/slurm_jobs"
    bash benchmarks/"${EXP_NAME%%_*}_${PRECISION}_gb200_${FRAMEWORK}.sh"
    # Wait for all jobs to complete
    echo "Waiting for all jobs to complete..."
    while [ -n "$(squeue -u $USER --noheader --format='%i')" ]; do
        echo "Jobs still running..."
        squeue --steps -u $USER
        sleep 30
    done

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

    exit 0
fi

echo "Cloning srt-slurm repository..."
SRT_REPO_DIR="srt-slurm"
if [ -d "$SRT_REPO_DIR" ]; then
    echo "Removing existing $SRT_REPO_DIR..."
    rm -rf "$SRT_REPO_DIR"
fi

git clone https://github.com/ishandhanani/srt-slurm.git "$SRT_REPO_DIR"
cd "$SRT_REPO_DIR"
git checkout sa-submission-q1-2026

echo "Installing srtctl..."
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

uv venv
source .venv/bin/activate
uv pip install -e .

if ! command -v srtctl &> /dev/null; then
    echo "Error: Failed to install srtctl"
    exit 1
fi

echo "Configs available at: $SRT_REPO_DIR/"

# Create srtslurm.yaml for srtctl (used by both frameworks)
SRTCTL_ROOT="${GITHUB_WORKSPACE}/srt-slurm"
echo "Creating srtslurm.yaml configuration..."
cat > srtslurm.yaml <<EOF
# SRT SLURM Configuration for GB200

# Default SLURM settings
default_account: "${SLURM_ACCOUNT}"
default_partition: "${SLURM_PARTITION}"
default_time_limit: "6:00:00"

# Resource defaults
gpus_per_node: 4
network_interface: ""

# Path to srtctl repo root (where the configs live)
srtctl_root: "${SRTCTL_ROOT}"

# Model path aliases
model_paths:
  "${MODEL_PREFIX}": "${MODEL_PATH}"
containers:
  dynamo-trtllm: ${SQUASH_FILE}
  dynamo-sglang: ${SQUASH_FILE}
  nginx-sqsh: ${NGINX_SQUASH_FILE}
EOF

echo "Generated srtslurm.yaml:"
cat srtslurm.yaml

echo "Running make setup..."
make setup ARCH=aarch64

# these 2 lines are for debugging
# TODO: remove when merge
echo "Make setup complete"
ls configs/

echo "Submitting job with srtctl..."
if [[ "$FRAMEWORK" == "dynamo-sglang" ]]; then
    SRTCTL_OUTPUT=$(srtctl apply -f "$CONFIG_FILE" --tags "gb200,${MODEL_PREFIX},${PRECISION},${ISL}x${OSL},infmax-$(date +%Y%m%d)" --setup-script install-torchao.sh 2>&1)
else
    SRTCTL_OUTPUT=$(srtctl apply -f "$CONFIG_FILE" --tags "gb200,${MODEL_PREFIX},${PRECISION},${ISL}x${OSL},infmax-$(date +%Y%m%d)" 2>&1)
fi
echo "$SRTCTL_OUTPUT"

JOB_ID=$(echo "$SRTCTL_OUTPUT" | grep -oP '✅ Job \K[0-9]+' || echo "$SRTCTL_OUTPUT" | grep -oP 'Job \K[0-9]+')

if [ -z "$JOB_ID" ]; then
    echo "Error: Failed to extract JOB_ID from srtctl output"
    exit 1
fi

echo "Extracted JOB_ID: $JOB_ID"

# Wait for this specific job to complete
echo "Waiting for job $JOB_ID to complete..."
while [ -n "$(squeue -j $JOB_ID --noheader 2>/dev/null)" ]; do
    echo "Job $JOB_ID still running..."
    squeue -j $JOB_ID
    sleep 30
done
echo "Job $JOB_ID completed!"

echo "Collecting results..."

# Use the JOB_ID to find the logs directory
# srtctl creates logs in outputs/JOB_ID/logs/
LOGS_DIR="outputs/$JOB_ID/logs"

if [ ! -d "$LOGS_DIR" ]; then
    echo "Warning: Logs directory not found at $LOGS_DIR"
    exit 1
fi

echo "Found logs directory: $LOGS_DIR"

cat $LOGS_DIR/sweep_$JOB_ID.log

for file in $LOGS_DIR/*; do
    if [ -f "$file" ]; then
        tail -n 500 $file
    fi
done

# Find all result subdirectories
RESULT_SUBDIRS=$(find "$LOGS_DIR" -maxdepth 1 -type d -name "*isl*osl*" 2>/dev/null)

if [ -z "$RESULT_SUBDIRS" ]; then
    echo "Warning: No result subdirectories found in $LOGS_DIR"
else
    # Process results from all configurations
    for result_subdir in $RESULT_SUBDIRS; do
        echo "Processing result subdirectory: $result_subdir"

        # Extract configuration info from directory name
        CONFIG_NAME=$(basename "$result_subdir")

        # Find all result JSON files
        RESULT_FILES=$(find "$result_subdir" -name "results_concurrency_*.json" 2>/dev/null)

        for result_file in $RESULT_FILES; do
            if [ -f "$result_file" ]; then
                # Extract metadata from filename
                # Files are of the format "results_concurrency_gpus_{num gpus}_ctx_{num ctx}_gen_{num gen}.json"
                filename=$(basename "$result_file")
                concurrency=$(echo "$filename" | sed -n 's/results_concurrency_\([0-9]*\)_gpus_.*/\1/p')
                gpus=$(echo "$filename" | sed -n 's/results_concurrency_[0-9]*_gpus_\([0-9]*\)_ctx_.*/\1/p')
                ctx=$(echo "$filename" | sed -n 's/.*_ctx_\([0-9]*\)_gen_.*/\1/p')
                gen=$(echo "$filename" | sed -n 's/.*_gen_\([0-9]*\)\.json/\1/p')

                echo "Processing concurrency $concurrency with $gpus GPUs (ctx: $ctx, gen: $gen): $result_file"

                WORKSPACE_RESULT_FILE="$GITHUB_WORKSPACE/${RESULT_FILENAME}_${CONFIG_NAME}_conc${concurrency}_gpus_${gpus}_ctx_${ctx}_gen_${gen}.json"
                cp "$result_file" "$WORKSPACE_RESULT_FILE"

                echo "Copied result file to: $WORKSPACE_RESULT_FILE"
            fi
        done
    done
fi

# Cleanup
echo "Cleaning up..."
deactivate 2>/dev/null || true
rm -rf .venv
echo "Cleanup complete"

echo "All result files processed"