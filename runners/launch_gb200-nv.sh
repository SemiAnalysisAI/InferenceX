#!/usr/bin/bash

# This script sets up the environment and launches multi-node benchmarks

set -x

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

# Set up environment variables for SLURM
export SLURM_PARTITION="batch"
export SLURM_ACCOUNT="benchmark"
export SLURM_JOB_NAME="benchmark-dynamo.job"

# MODEL_PATH is set in `nvidia-master.yaml` or any other yaml files
export MODEL_PATH=$MODEL

if [[ $FRAMEWORK == "dynamo-sglang" ]]; then
    export CONFIG_DIR="/mnt/lustre01/artifacts/sglang-configs/1k1k"
elif [[ $FRAMEWORK == "dynamo-trt" ]]; then
    if [[ $MODEL_PREFIX == "gptoss" ]]; then
        export MODEL_PATH="/mnt/lustre01/models/gpt-oss-120b"
        export SERVED_MODEL_NAME="gpt-oss-120b"
    elif [[ $MODEL_PREFIX == "dsr1" ]]; then
        export SERVED_MODEL_NAME="deepseek-r1-fp4"
    else
        echo "Unsupported model prefix: $MODEL_PREFIX. Supported prefixes are: gptoss or dsr1"
        exit 1
    fi
fi

export ISL="$ISL"
export OSL="$OSL"

NGINX_IMAGE="nginx:1.27.4"

SQUASH_FILE="/mnt/lustre01/artifacts/containers/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
NGINX_SQUASH_FILE="/mnt/lustre01/artifacts/containers/$(echo "$NGINX_IMAGE" | sed 's/[\/:@#]/_/g').sqsh"

srun -N 1 -A $SLURM_ACCOUNT -p $SLURM_PARTITION bash -c "enroot import -o $SQUASH_FILE docker://$IMAGE"
srun -N 1 -A $SLURM_ACCOUNT -p $SLURM_PARTITION bash -c "enroot import -o $NGINX_SQUASH_FILE docker://$NGINX_IMAGE"

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

cat "outputs/$JOB_ID/logs/sweep_${JOB_ID}.log"

# Use the JOB_ID to find the logs directory
# srtctl creates logs in outputs/JOB_ID/logs/
LOGS_DIR="outputs/$JOB_ID/logs"

if [ ! -d "$LOGS_DIR" ]; then
    echo "Warning: Logs directory not found at $LOGS_DIR"
    exit 1
fi

echo "Found logs directory: $LOGS_DIR"

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