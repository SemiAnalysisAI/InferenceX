#!/usr/bin/bash

# System-specific configuration for H100 DGXC Slurm cluster
SLURM_PARTITION="hpc-gpu-1"
SLURM_ACCOUNT="customer"
SLURM_EXCLUDED_NODELIST="hpc-gpu-1-6,hpc-gpu-1-7"

set -x

if [[ "$IS_MULTINODE" == "true" ]]; then

    # MODEL_PATH: Override with pre-downloaded paths on H100 runner
    # The yaml files specify HuggingFace model IDs for portability, but we use
    # local paths to avoid repeated downloading on the shared H100 cluster.
    if [[ $FRAMEWORK == "dynamo-sglang" ]]; then
        if [[ $MODEL_PREFIX == "dsr1" && $PRECISION == "fp8" ]]; then
            export MODEL_PATH="/mnt/numa1/shared/models/dsr1-fp8"
            export SRT_SLURM_MODEL_PREFIX="dsr1-fp8"
        else
            echo "Unsupported model prefix/precision for dynamo-sglang: $MODEL_PREFIX/$PRECISION"
            exit 1
        fi
    elif [[ $FRAMEWORK == "dynamo-trt" ]]; then
        if [[ $MODEL_PREFIX == "dsr1" && $PRECISION == "fp8" ]]; then
            export MODEL_PATH="/mnt/numa1/shared/models/dsr1-fp8"
            export SERVED_MODEL_NAME="DeepSeek-R1-0528"
            export SRT_SLURM_MODEL_PREFIX="DeepSeek-R1-0528"
        else
            echo "Unsupported model prefix/precision for dynamo-trt: $MODEL_PREFIX/$PRECISION"
            exit 1
        fi
    else
        echo "Unsupported framework: $FRAMEWORK. Supported frameworks are: dynamo-trt, dynamo-sglang"
        exit 1
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

    # Map container images to local squash files based on framework
    NGINX_SQUASH_FILE="/mnt/nfs/lustre/containers/nginx_1.27.4.sqsh"

    if [[ $FRAMEWORK == "dynamo-sglang" ]]; then
        # SGLang container mapping
        SQUASH_FILE="/mnt/nfs/lustre/containers/lmsysorg_sglang_v0.5.8.post1-cu130.sqsh"
        CONTAINER_KEY="lmsysorg/sglang:v0.5.8-cu130"
    elif [[ $FRAMEWORK == "dynamo-trt" ]]; then
        # TRT-LLM container mapping - convert IMAGE to srt-slurm format (nvcr.io/ -> nvcr.io#)
        CONTAINER_KEY=$(echo "$IMAGE" | sed 's|nvcr.io/|nvcr.io#|')
        SQUASH_FILE="/mnt/nfs/sa-shared/containers/$(echo "$IMAGE" | sed 's|nvcr.io/||' | sed 's/[\/:@#]/+/g').sqsh"
    fi

    export ISL="$ISL"
    export OSL="$OSL"

    # Create srtslurm.yaml for srtctl (used by both frameworks)
    SRTCTL_ROOT="${GITHUB_WORKSPACE}/${SRT_REPO_DIR}"
    echo "Creating srtslurm.yaml configuration..."
    cat > srtslurm.yaml <<EOF
# SRT SLURM Configuration for H100

# Default SLURM settings
default_account: "${SLURM_ACCOUNT}"
default_partition: "${SLURM_PARTITION}"
default_time_limit: "6:00:00"
# Resource defaults
gpus_per_node: 8
network_interface: ""
# Path to srtctl repo root (where the configs live)
srtctl_root: "${SRTCTL_ROOT}"
# Model path aliases
model_paths:
  "${SRT_SLURM_MODEL_PREFIX}": "${MODEL_PATH}"
containers:
  dynamo-trtllm: "${SQUASH_FILE}"
  dynamo-sglang: "${SQUASH_FILE}"
  nginx-sqsh: "${NGINX_SQUASH_FILE}"
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
    # Override the job name in the config file with the runner name
    sed -i "s/^name:.*/name: \"${RUNNER_NAME}\"/" "$CONFIG_FILE"
    sed -i "/^name:.*/a sbatch_directives:\n  exclude: \"${SLURM_EXCLUDED_NODELIST}\"" "$CONFIG_FILE"
    SRTCTL_OUTPUT=$(srtctl apply -f "$CONFIG_FILE" --tags "h100,${MODEL_PREFIX},${PRECISION},${ISL}x${OSL},infmax-$(date +%Y%m%d)" 2>&1)
    echo "$SRTCTL_OUTPUT"

    # Extract JOB_ID from srtctl output
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

else

    HF_HUB_CACHE_MOUNT="/mnt/nfs/sa-shared/gharunners/hf-hub-cache/"
    SQUASH_FILE="/mnt/nfs/lustre/containers/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"

    salloc --exclude="$SLURM_EXCLUDED_NODELIST" --partition=$SLURM_PARTITION --account=$SLURM_ACCOUNT --gres=gpu:$TP --exclusive --time=180 --no-shell --job-name="$RUNNER_NAME"
    JOB_ID=$(squeue -u $USER -h -o %A | head -n1)

    srun --jobid=$JOB_ID bash -c "enroot import -o $SQUASH_FILE docker://$IMAGE"
    if ! srun --jobid=$JOB_ID bash -c "unsquashfs -l $SQUASH_FILE > /dev/null"; then
        echo "unsquashfs failed, removing $SQUASH_FILE and re-importing..."
        srun --jobid=$JOB_ID bash -c "rm -f $SQUASH_FILE"
        srun --jobid=$JOB_ID bash -c "enroot import -o $SQUASH_FILE docker://$IMAGE"
    fi

    srun --jobid=$JOB_ID \
        --container-image=$SQUASH_FILE \
        --container-mounts=$GITHUB_WORKSPACE:/workspace/,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE \
        --container-mount-home \
        --container-workdir=/workspace/ \
        --no-container-entrypoint --export=ALL,PORT=8888 \
        bash benchmarks/single_node/${EXP_NAME%%_*}_${PRECISION}_h100.sh

    scancel $JOB_ID

fi
