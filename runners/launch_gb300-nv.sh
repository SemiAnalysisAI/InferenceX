#!/usr/bin/bash

# This script sets up the environment and launches multi-node benchmarks

set -x

export SLURM_PARTITION="batch"
export SLURM_ACCOUNT="benchmark"
export ENROOT_ROOTFS_WRITABLE=1

export MODEL_PATH=$MODEL

if [[ $MODEL_PREFIX == "dsr1" && $PRECISION == "fp4" ]]; then
    export SERVED_MODEL_NAME="deepseek-r1-fp4"
    export MODEL_PATH=/raid/shared/models/deepseek-r1-0528-fp4-v2
    export SRT_SLURM_MODEL_PREFIX="dsr1"
elif [[ $MODEL_PREFIX == "dsr1" && $PRECISION == "fp8" ]]; then
    export SERVED_MODEL_NAME="deepseek-r1-fp8"
    export MODEL_PATH=/raid/shared/models/deepseek-r1-0528
    export SRT_SLURM_MODEL_PREFIX="dsr1-fp8"
else
    echo "Unsupported model: $MODEL_PREFIX-$PRECISION. Supported models are: dsr1-fp4, dsr1-fp8"
    exit 1
fi

NGINX_IMAGE="nginx:1.27.4"

SQUASH_FILE="/home/sa-shared/squash/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
NGINX_SQUASH_FILE="/home/sa-shared/squash/$(echo "$NGINX_IMAGE" | sed 's/[\/:@#]/_/g').sqsh"

srun --partition=$SLURM_PARTITION --exclusive --time=180 bash -c "enroot import -o $SQUASH_FILE docker://$IMAGE"
srun --partition=$SLURM_PARTITION --exclusive --time=180 bash -c "enroot import -o $NGINX_SQUASH_FILE docker://$NGINX_IMAGE"

export ISL="$ISL"
export OSL="$OSL"

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
export UV_INSTALL_DIR="$GITHUB_WORKSPACE/.local/bin"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$UV_INSTALL_DIR:$PATH"

uv venv "$GITHUB_WORKSPACE/.venv"
source "$GITHUB_WORKSPACE/.venv/bin/activate"
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
# SRT SLURM Configuration for GB300

# Default SLURM settings
default_account: "${SLURM_ACCOUNT}"
default_partition: "${SLURM_PARTITION}"
default_time_limit: "4:00:00"

# Resource defaults
gpus_per_node: 4
network_interface: ""

# Path to srtctl repo root (where the configs live)
srtctl_root: "${SRTCTL_ROOT}"

# Model path aliases
model_paths:
  "${SRT_SLURM_MODEL_PREFIX}": "${MODEL_PATH}"
containers:
  dynamo-trtllm: ${SQUASH_FILE}
  dynamo-sglang: ${SQUASH_FILE}
  nginx-sqsh: ${NGINX_SQUASH_FILE}
use_segment_sbatch_directive: false
EOF

echo "Generated srtslurm.yaml:"
cat srtslurm.yaml

echo "Running make setup..."
make setup ARCH=aarch64

echo "Submitting job with srtctl..."

# Override the job name in the config file with the runner name
sed -i "s/^name:.*/name: \"${RUNNER_NAME}\"/" "$CONFIG_FILE"

SRTCTL_OUTPUT=$(srtctl apply -f "$CONFIG_FILE" --tags "gb300,${MODEL_PREFIX},${PRECISION},${ISL}x${OSL},infmax-$(date +%Y%m%d)" 2>&1)
echo "$SRTCTL_OUTPUT"

JOB_ID=$(echo "$SRTCTL_OUTPUT" | grep -oP '✅ Job \K[0-9]+' || echo "$SRTCTL_OUTPUT" | grep -oP 'Job \K[0-9]+')

set +x

if [ -z "$JOB_ID" ]; then
    echo "Error: Failed to extract JOB_ID from srtctl output"
    exit 1
fi

echo "Extracted JOB_ID: $JOB_ID"

# Use the JOB_ID to find the logs directory
# srtctl creates logs in outputs/JOB_ID/logs/
LOGS_DIR="outputs/$JOB_ID/logs"
LOG_FILE="$LOGS_DIR/sweep_${JOB_ID}.log"

# Wait for log file to appear (also check job is still alive)
while ! ls "$LOG_FILE" &>/dev/null; do
    if ! squeue -j "$JOB_ID" --noheader 2>/dev/null | grep -q "$JOB_ID"; then
        echo "ERROR: Job $JOB_ID failed before creating log file"
        scontrol show job "$JOB_ID"
        exit 1
    fi
    echo "Waiting for JOB_ID $JOB_ID to begin and $LOG_FILE to appear..."
    sleep 5
done

# Poll for job completion in background
(
    while squeue -j "$JOB_ID" --noheader 2>/dev/null | grep -q "$JOB_ID"; do
        sleep 10
    done
) &
POLL_PID=$!

echo "Tailing LOG_FILE: $LOG_FILE"

# Stream the log file until job completes (-F follows by name, polls instead of inotify for NFS)
tail -F -s 2 -n+1 "$LOG_FILE" --pid=$POLL_PID 2>/dev/null

wait $POLL_PID

set -x

echo "Job $JOB_ID completed!"
echo "Collecting results..."

if [ ! -d "$LOGS_DIR" ]; then
    echo "Warning: Logs directory not found at $LOGS_DIR"
    exit 1
fi

echo "Found logs directory: $LOGS_DIR"

cp -r "$LOGS_DIR" "$GITHUB_WORKSPACE/LOGS"
tar czf "$GITHUB_WORKSPACE/multinode_server_logs.tar.gz" -C "$LOGS_DIR" .

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

# Clean up srt-slurm outputs to prevent NFS silly-rename lock files
# from blocking the next job's checkout on this runner
echo "Cleaning up srt-slurm outputs..."
for i in 1 2 3 4 5; do
    rm -rf outputs 2>/dev/null && break
    echo "Retry $i/5: Waiting for NFS locks to release..."
    sleep 10
done
find . -name '.nfs*' -delete 2>/dev/null || true
