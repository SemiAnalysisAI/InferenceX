#!/usr/bin/bash

set -x

echo "Cloning srt-slurm-trtllm repository..."
SRT_REPO_DIR="srt-slurm"
if [ -d "$SRT_REPO_DIR" ]; then
    echo "Removing existing $SRT_REPO_DIR..."
    rm -rf "$SRT_REPO_DIR"
fi

git clone https://github.com/ishandhanani/srt-slurm.git "$SRT_REPO_DIR"
cd "$SRT_REPO_DIR"
git checkout b4abe4643a7009f3539b36bdc508408874a4c930

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

export SLURM_PARTITION="batch_1"
export SLURM_ACCOUNT="benchmark"

SQUASH_FILE="/data/squash/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"

if [[ $MODEL_PREFIX == "dsr1" ]]; then
    export MODEL_PATH="/scratch/models/deepseek-r1-0528-nvfp4-v2"
    export SERVED_MODEL_NAME="deepseek-r1-fp4"
else
    echo "Unsupported model prefix: $MODEL_PREFIX. Supported prefixes are: dsr1"
    exit 1
fi

export ISL="$ISL"
export OSL="$OSL"

# Create srtslurm.yaml for srtctl
echo "Creating srtslurm.yaml configuration..."
cat > srtslurm.yaml <<EOF
# SRT SLURM Configuration for B300

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
  "${MODEL_PREFIX}": "${MODEL_PATH}"

# Container aliases
containers:
  latest: "${SQUASH_FILE}"

use_exclusive_sbatch_directive: true

default_mounts:
  "/opt/ucx-no-ud": "/usr/local/ucx"

EOF

echo "Generated srtslurm.yaml:"
cat srtslurm.yaml

echo "Running make setup..."
make setup ARCH=x86_64

echo "Submitting job with srtctl..."
SRTCTL_OUTPUT=$(srtctl apply -f "$CONFIG_FILE" --tags "b300,${MODEL_PREFIX},${PRECISION},${ISL}x${OSL},infmax-$(date +%Y%m%d)" 2>&1)
echo "$SRTCTL_OUTPUT"

# Extract JOB_ID from srtctl output (e.g., "✅ Job 1168 submitted!")
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

cat "outputs/$JOB_ID/logs/sweep_${JOB_ID}.log"

echo "Collecting results..."

# Use the JOB_ID to find the logs directory
# srtctl creates logs in outputs/JOB_ID/logs/
LOGS_DIR="outputs/$JOB_ID/logs"

if [ ! -d "$LOGS_DIR" ]; then
    echo "Warning: Logs directory not found at $LOGS_DIR"
    exit 1
fi

echo "Found logs directory: $LOGS_DIR"

# Find all result subdirectories (e.g., sa-bench_isl_8192_osl_1024)
RESULT_SUBDIRS=$(find "$LOGS_DIR" -maxdepth 1 -type d -name "*isl*osl*" 2>/dev/null)

if [ -z "$RESULT_SUBDIRS" ]; then
    echo "Warning: No result subdirectories found in $LOGS_DIR"
else
    # Process results from all configurations
    for result_subdir in $RESULT_SUBDIRS; do
        echo "Processing result subdirectory: $result_subdir"

        # Extract configuration info from directory name
        CONFIG_NAME=$(basename "$result_subdir")

        # Find all result JSON files (e.g., results_concurrency_128_gpus_16_ctx_8_gen_8.json)
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

echo "All result files processed"

# Cleanup
echo "Cleaning up..."
deactivate 2>/dev/null || true
rm -rf .venv
echo "Cleanup complete"

