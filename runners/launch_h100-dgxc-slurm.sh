#!/usr/bin/bash

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

export SLURM_PARTITION="hpc-gpu-1"
export SLURM_ACCOUNT="customer"

# Convert IMAGE to srt-slurm format (nvcr.io/ -> nvcr.io#)
CONTAINER_KEY=$(echo "$IMAGE" | sed 's|nvcr.io/|nvcr.io#|')

# Map container image to local squash file
SQUASH_FILE="/mnt/nfs/sa-shared/containers/$(echo "$IMAGE" | sed 's|nvcr.io/||' | sed 's/[\/:@#]/+/g').sqsh"

if [[ $MODEL_PREFIX == "dsr1" && $PRECISION == "fp8" ]]; then
    export MODEL_PATH="/mnt/numa1/shared/models/dsr1-fp8"
    export SERVED_MODEL_NAME="DeepSeek-R1-0528"
    export SRT_SLURM_MODEL_PREFIX="DeepSeek-R1-0528"
else
    echo "Unsupported model prefix: $MODEL_PREFIX. Supported prefixes are: DeepSeek-R1-0528"
    exit 1
fi

export ISL="$ISL"
export OSL="$OSL"

# Create srtslurm.yaml for srtctl
echo "Creating srtslurm.yaml configuration..."
cat > srtslurm.yaml <<EOF
# SRT SLURM Configuration for H100

# Default SLURM settings
default_account: "${SLURM_ACCOUNT}"
default_partition: "${SLURM_PARTITION}"
default_time_limit: "4:00:00"
# Resource defaults
gpus_per_node: 8
network_interface: ""
# Path to srtctl repo root (where the configs live)
srtctl_root: "${GITHUB_WORKSPACE}/${SRT_REPO_DIR}"
# Model path aliases
model_paths:
  "${SRT_SLURM_MODEL_PREFIX}": "${MODEL_PATH}"
containers:
  latest: "${SQUASH_FILE}"
  "${CONTAINER_KEY}": "${SQUASH_FILE}"
# SLURM directive compatibility
use_gpus_per_node_directive: true
use_segment_sbatch_directive: false
use_exclusive_sbatch_directive: false
EOF

echo "Generated srtslurm.yaml:"
cat srtslurm.yaml

echo "Running make setup..."
make setup ARCH=x86_64

echo "Submitting job with srtctl..."
SRTCTL_OUTPUT=$(srtctl apply -f "$CONFIG_FILE" --tags "h100,${MODEL_PREFIX},${PRECISION},${ISL}x${OSL},infmax-$(date +%Y%m%d)" 2>&1)
echo "$SRTCTL_OUTPUT"

# Extract JOB_ID from srtctl output
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

cat $LOGS_DIR/sweep_${JOB_ID}.log

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

echo "All result files processed"

# Cleanup
echo "Cleaning up..."
deactivate 2>/dev/null || true
rm -rf .venv
echo "Cleanup complete"
