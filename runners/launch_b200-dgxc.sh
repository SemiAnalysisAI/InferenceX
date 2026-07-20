#!/usr/bin/bash

# System-specific configuration for B200 DGXC Slurm cluster
SLURM_PARTITION="gpu-2"
SLURM_ACCOUNT="benchmark"

set -x

# MODEL_PATH: Override with pre-downloaded paths on the shared Lustre tree.
# Bench scripts and srt-slurm yaml configs specify HuggingFace model IDs for
# portability, but we resolve to /lustre/fsw/models/* here to avoid repeated
# downloading on every dgxc node. Runs for both single-node and multinode
# launches.
# NOTE: per-node /raid/models/* would be faster but is only populated on a
# subset of dgxc nodes today, so we use Lustre for reliability.
if [[ $MODEL_PREFIX == "dsr1" && $PRECISION == "fp4" ]]; then
    export MODEL_PATH="/scratch/fsw/models/DeepSeek-R1-0528-NVFP4-v2"
    export SRT_SLURM_MODEL_PREFIX="dsr1"
elif [[ $MODEL_PREFIX == "dsr1" && $PRECISION == "fp8" ]]; then
    export MODEL_PATH="/lustre/fsw/models/dsr1-0528-fp8"
    export SRT_SLURM_MODEL_PREFIX="dsr1-fp8"
elif [[ $MODEL_PREFIX == "dsv4" && $PRECISION == "fp4" ]]; then
    SELECTED_MODEL_PATH=""
    if [[ -n "${MODEL_PATH:-}" && -d "${MODEL_PATH}" ]]; then
        SELECTED_MODEL_PATH="$MODEL_PATH"
    else
        for candidate in /lustre/fsw/models/deepseek-v4-pro /lustre/fsw/models/dsv4-pro /lustre/fsw/models/DeepSeek-V4-Pro; do
            if [[ -d "$candidate" ]]; then
                SELECTED_MODEL_PATH="$candidate"
                break
            fi
        done
    fi
    export MODEL_PATH="${SELECTED_MODEL_PATH:-/lustre/fsw/models/deepseek-v4-pro}"
    export SRT_SLURM_MODEL_PREFIX="deepseek-v4-pro"
elif [[ $MODEL_PREFIX == "qwen3.5" && $PRECISION == "bf16" ]]; then
    export MODEL_PATH="/lustre/fsw/models/Qwen3.5-397B-A17B"
    export SRT_SLURM_MODEL_PREFIX="qwen3.5"
elif [[ $MODEL_PREFIX == "qwen3.5" && $PRECISION == "fp8" ]]; then
    export MODEL_PATH="/lustre/fsw/models/Qwen3.5-397B-A17B-FP8"
    export SRT_SLURM_MODEL_PREFIX="qwen3.5-fp8"
elif [[ $MODEL_PREFIX == "qwen3.5" && $PRECISION == "fp4" ]]; then
    export MODEL_PATH="/lustre/fsw/models/Qwen3.5-397B-A17B-NVFP4"
    export SRT_SLURM_MODEL_PREFIX="qwen3.5-fp4"
elif [[ $MODEL_PREFIX == "glm5" && $PRECISION == "fp8" ]]; then
    export MODEL_PATH="/lustre/fsw/models/GLM-5-FP8"
    export SRT_SLURM_MODEL_PREFIX="glm5-fp8"
elif [[ $MODEL_PREFIX == "glm5" && $PRECISION == "fp4" ]]; then
    export MODEL_PATH="/lustre/fsw/models/GLM-5-NVFP4"
    export SRT_SLURM_MODEL_PREFIX="glm5-fp4"
elif [[ $MODEL_PREFIX == "kimik2.5" && $PRECISION == "int4" ]]; then
    export MODEL_PATH="/lustre/fsw/models/Kimi-K2.5"
    export SRT_SLURM_MODEL_PREFIX="kimik2.5"
elif [[ $MODEL_PREFIX == "kimik2.5" && $PRECISION == "fp4" ]]; then
    export MODEL_PATH="/lustre/fsw/models/Kimi-K2.5-NVFP4"
    export SRT_SLURM_MODEL_PREFIX="kimik2.5-fp4"
elif [[ $MODEL_PREFIX == "minimaxm2.5" && $PRECISION == "fp8" ]]; then
    export MODEL_PATH="/lustre/fsw/models/MiniMax-M2.5"
    export SRT_SLURM_MODEL_PREFIX="minimax-m2.5-fp8"
elif [[ $MODEL_PREFIX == "minimaxm2.5" && $PRECISION == "fp4" ]]; then
    export MODEL_PATH="/lustre/fsw/models/MiniMax-M2.5-NVFP4"
    export SRT_SLURM_MODEL_PREFIX="minimax-m2.5-nvfp4"
elif [[ $MODEL_PREFIX == "minimaxm2.7" && $PRECISION == "fp4" ]]; then
    # Public NVFP4 target and EAGLE3 draft are staged in the runner-writable
    # shared model tree by the benchmark's revision-pinned preflight.
    export MODEL_PATH="/lustre/fsw/gharunners/models/MiniMax-M2.7-NVFP4"
    export SRT_SLURM_MODEL_PREFIX="minimax-m2.7-nvfp4"
elif [[ $MODEL_PREFIX == "gptoss" && $PRECISION == "fp4" ]]; then
    export MODEL_PATH="/lustre/fsw/models/gpt-oss-120b"
    export SRT_SLURM_MODEL_PREFIX="gptoss"
elif [[ $MODEL_PREFIX == "minimaxm3" && $PRECISION == "fp8" ]]; then
    # Day-zero: MiniMax-M3-MXFP8 is not in the SRE-staged /lustre/fsw/models
    # tree (root-owned); it lives in the sa-shared-writable gharunners tree.
    export MODEL_PATH="/lustre/fsw/gharunners/models/MiniMax-M3-MXFP8"
    export SRT_SLURM_MODEL_PREFIX="minimax-m3-mxfp8"
elif [[ $MODEL_PREFIX == "minimaxm3" && $PRECISION == "fp4" ]]; then
    # NVFP4 checkpoint, pre-staged on the b200-dgxc scratch tree.
    export MODEL_PATH="/scratch/fsw/models/MiniMax-M3-NVFP4"
    export SRT_SLURM_MODEL_PREFIX="minimax-m3-nvfp4"
else
    echo "Unsupported model prefix/precision: $MODEL_PREFIX/$PRECISION"
    echo "Available models under /lustre/fsw/models:"
    ls -la /lustre/fsw/models
    exit 1
fi

export AIPERF_MMAP_CACHE_HOST_PATH="/lustre/fsw/gharunners/aiperf-cache"

if [[ "$IS_MULTINODE" == "true" ]]; then
    # Validate framework
    if [[ $FRAMEWORK != "dynamo-sglang" && $FRAMEWORK != "dynamo-trt" && $FRAMEWORK != "dynamo-vllm" ]]; then
        echo "Unsupported framework: $FRAMEWORK. Supported frameworks are: dynamo-trt, dynamo-sglang, dynamo-vllm"
        exit 1
    fi

    # Multinode dsv4 currently only ships with the dynamo-vllm recipe
    if [[ $MODEL_PREFIX == "dsv4" && $FRAMEWORK != "dynamo-vllm" ]]; then
        echo "Unsupported framework for multinode dsv4: $FRAMEWORK (only dynamo-vllm)"
        exit 1
    fi

    export SERVED_MODEL_NAME=$MODEL

    echo "Cloning srt-slurm repository..."
    SRT_REPO_DIR="srt-slurm"
    if [ -d "$SRT_REPO_DIR" ]; then
        echo "Removing existing $SRT_REPO_DIR..."
        rm -rf "$SRT_REPO_DIR"
    fi

    # TODO(CJQ): make first class upon srt-slurm upstream refactor
    if [[ "$IS_AGENTIC" == "1" ]]; then
        git clone --branch cam/sa-submission-q2-2026 --single-branch https://github.com/cquil11/srt-slurm-nv.git "$SRT_REPO_DIR"
        cd "$SRT_REPO_DIR" || exit 1
    elif [[ $FRAMEWORK == "dynamo-vllm" && $MODEL_PREFIX == "dsv4" ]]; then
        git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
        cd "$SRT_REPO_DIR" || exit 1
        git checkout aflowers/vllm-gb200-v0.20.0
        mkdir -p recipes/vllm/deepseek-v4
        cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/vllm/deepseek-v4" recipes/vllm/deepseek-v4
    elif [[ $FRAMEWORK == "dynamo-sglang" && $MODEL_PREFIX == "glm5" && $PRECISION == "fp8" ]]; then
        git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
        cd "$SRT_REPO_DIR" || exit 1
        git checkout main
    elif [[ $FRAMEWORK == "dynamo-sglang" && $MODEL_PREFIX == "dsr1" && $PRECISION == "fp4" ]]; then
        git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
        cd "$SRT_REPO_DIR" || exit 1
        # Pin srt-slurm: newer commits stopped honoring the hash-pinned dynamo
        # build and fall back to a dynamo release that is incompatible with this
        # sglang image (worker fails at import). This is the last commit before
        # that change. Do not float on main -- the srtctl + dynamo-install
        # toolchain is unpinned there.
        git checkout a98738de9b2233459b5456e9ed71af09ce893f92
        mkdir -p recipes/sglang/dsr1/b200-fp4
        cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/sglang/dsr1/b200-fp4" recipes/sglang/dsr1/b200-fp4
    else
        git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
        cd "$SRT_REPO_DIR" || exit 1
        git checkout sa-submission-q2-2026
    fi

    echo "Installing srtctl..."
    export UV_INSTALL_DIR="$GITHUB_WORKSPACE/.local/bin"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$UV_INSTALL_DIR:$PATH"

    if [[ $MODEL_PREFIX == "minimaxm2.5" && $FRAMEWORK == "dynamo-vllm" ]]; then
        uv venv --seed "$GITHUB_WORKSPACE/.venv"
    else
        uv venv "$GITHUB_WORKSPACE/.venv"
    fi
    source "$GITHUB_WORKSPACE/.venv/bin/activate"
    uv pip install -e .

    if ! command -v srtctl &> /dev/null; then
        echo "Error: Failed to install srtctl"
        exit 1
    fi

    # Map container images to local squash files
    NGINX_IMAGE="nginx:1.27.4"
    SQUASH_DIR="${B200_SQUASH_DIR:-/home/sa-shared/containers}"
    if [[ $MODEL_PREFIX == "minimaxm2.5" && $FRAMEWORK == "dynamo-vllm" ]]; then
        SQUASH_DIR="${B200_SQUASH_DIR:-/home/slurm-shared/gharunners/squash}"
    fi
    if ! mkdir -p "$SQUASH_DIR" 2>/dev/null || [[ ! -w "$SQUASH_DIR" ]]; then
        echo "Warning: $SQUASH_DIR is not writable; using workspace-local squash cache" >&2
        SQUASH_DIR="$GITHUB_WORKSPACE/.container-squash"
        mkdir -p "$SQUASH_DIR"
    fi
    chmod a+rx "$SQUASH_DIR" || true

    SQUASH_FILE="$SQUASH_DIR/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
    NGINX_SQUASH_FILE="$SQUASH_DIR/$(echo "$NGINX_IMAGE" | sed 's/[\/:@#]/_/g').sqsh"

    # Import containers via enroot
    import_squash() {
        local squash_file="$1"
        local image_ref="$2"
        local image_key
        image_key=$(echo "$image_ref" | sed 's/[\/:@#]/_/g')
        local lock_dir="${SQUASH_DIR}/.locks"
        mkdir -p "$lock_dir"
        local lock_file="${lock_dir}/${image_key}.lock"

        (
            flock -w 600 9 || { echo "Failed to acquire lock for $squash_file" >&2; exit 1; }
            if unsquashfs -l "$squash_file" > /dev/null 2>&1; then
                echo "Squash file already exists and is valid, skipping import: $squash_file"
            else
                rm -f "$squash_file"
                enroot import -o "$squash_file" "docker://$image_ref"
                if ! unsquashfs -l "$squash_file" > /dev/null 2>&1; then
                    echo "Error: enroot import did not produce a valid squash file: $squash_file" >&2
                    exit 1
                fi
                chmod a+r "$squash_file" || true
            fi
        ) 9>"$lock_file"
    }

    import_squash "$SQUASH_FILE" "$IMAGE" || exit 1
    import_squash "$NGINX_SQUASH_FILE" "$NGINX_IMAGE" || exit 1

    export ISL="$ISL"
    export OSL="$OSL"
    export EVAL_ONLY="${EVAL_ONLY:-false}"

    # Create srtslurm.yaml for srtctl (used by both frameworks)
    SRTCTL_ROOT="${GITHUB_WORKSPACE}/${SRT_REPO_DIR}"
    echo "Creating srtslurm.yaml configuration..."
    cat > srtslurm.yaml <<EOF
# SRT SLURM Configuration for B200

# Default SLURM settings
default_account: "${SLURM_ACCOUNT}"
default_partition: "${SLURM_PARTITION}"
default_time_limit: "4:00:00"
# Resource defaults
gpus_per_node: 8
network_interface: ""
# Path to srtctl repo root (where the configs live)
srtctl_root: "${SRTCTL_ROOT}"
# Model path aliases
model_paths:
  "${SRT_SLURM_MODEL_PREFIX}": "${MODEL_PATH}"
# Container aliases
containers:
  dynamo-trtllm: "${SQUASH_FILE}"
  dynamo-sglang: "${SQUASH_FILE}"
  dynamo-vllm: "${SQUASH_FILE}"
  sglang-v0.5.11-cu130: "${SQUASH_FILE}"
  "${IMAGE}": "${SQUASH_FILE}"
  nginx-sqsh: "${NGINX_SQUASH_FILE}"
use_exclusive_sbatch_directive: true
EOF

    echo "Generated srtslurm.yaml:"
    cat srtslurm.yaml

    echo "Running make setup..."
    make setup ARCH=x86_64

    # Export eval-related env vars for srt-slurm post-benchmark eval
    export INFMAX_WORKSPACE="$GITHUB_WORKSPACE"

    echo "Submitting job with srtctl..."
    echo "MODEL_PATH=$MODEL_PATH (exists=$(test -d "$MODEL_PATH" && echo yes || echo NO))"
    ls -ld "$MODEL_PATH" 2>&1 || ls /lustre/fsw/models/ 2>&1 | head -40

    if [[ -z "$CONFIG_FILE" ]]; then
        echo "Error: CONFIG_FILE is not set. The srt-slurm path requires a CONFIG_FILE in additional-settings." >&2
        echo "Config: MODEL_PREFIX=${MODEL_PREFIX} PRECISION=${PRECISION} FRAMEWORK=${FRAMEWORK}" >&2
        exit 1
    fi

    # Override the job name in the config file with the runner name
    sed -i "s/^name:.*/name: \"${RUNNER_NAME}\"/" "${CONFIG_FILE%%:*}"
    # Bump recipe health-check timeout from 360×10s=3600s to 720×10s=7200s
    # so large-model loads (e.g. DSR1-FP8 ~680GB off shared FS) finish in time.
    # Uses ${CONFIG_FILE%%:*} because CONFIG_FILE may carry an :override[N] suffix.
    sed -i 's/^  max_attempts: [0-9]*/  max_attempts: 720/' "${CONFIG_FILE%%:*}"
    SRTCTL_OUTPUT=$(srtctl apply -f "$CONFIG_FILE" --tags "b200,${MODEL_PREFIX},${PRECISION},${ISL}x${OSL},infmax-$(date +%Y%m%d)" 2>&1)
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

    if [[ "${EVAL_ONLY:-false}" != "true" ]]; then
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
                        # Files may be "results_concurrency_N_gpus_G_ctx_C_gen_D.json" (disagg) or "results_concurrency_N_gpus_G.json" (non-disagg)
                        filename=$(basename "$result_file")
                        concurrency=$(echo "$filename" | sed -n 's/results_concurrency_\([0-9]*\)_gpus_.*/\1/p')
                        gpus=$(echo "$filename" | sed -n 's/results_concurrency_[0-9]*_gpus_\([0-9][0-9]*\).*/\1/p')
                        ctx=$(echo "$filename" | sed -n 's/.*_ctx_\([0-9]*\)_gen_.*/\1/p')
                        gen=$(echo "$filename" | sed -n 's/.*_gen_\([0-9]*\)\.json/\1/p')

                        echo "Processing concurrency $concurrency with $gpus GPUs (ctx: $ctx, gen: $gen): $result_file"

                        if [ -n "$ctx" ] && [ -n "$gen" ]; then
                            WORKSPACE_RESULT_FILE="$GITHUB_WORKSPACE/${RESULT_FILENAME}_${CONFIG_NAME}_conc${concurrency}_gpus_${gpus}_ctx_${ctx}_gen_${gen}.json"
                        else
                            WORKSPACE_RESULT_FILE="$GITHUB_WORKSPACE/${RESULT_FILENAME}_${CONFIG_NAME}_conc${concurrency}_gpus_${gpus}.json"
                        fi
                        cp "$result_file" "$WORKSPACE_RESULT_FILE"

                        echo "Copied result file to: $WORKSPACE_RESULT_FILE"
                    fi
                done
            done
        fi

        echo "All result files processed"
    else
        echo "EVAL_ONLY=true: Skipping benchmark result collection"
    fi

    # Collect eval results if eval was requested
    if [[ "${RUN_EVAL:-false}" == "true" || "${EVAL_ONLY:-false}" == "true" ]]; then
        EVAL_DIR="$LOGS_DIR/eval_results"
        if [ -d "$EVAL_DIR" ]; then
            echo "Extracting eval results from $EVAL_DIR"
            shopt -s nullglob
            for eval_file in "$EVAL_DIR"/*; do
                [ -f "$eval_file" ] || continue
                cp "$eval_file" "$GITHUB_WORKSPACE/"
                echo "Copied eval artifact: $(basename "$eval_file")"
            done
            shopt -u nullglob
        else
            echo "WARNING: RUN_EVAL=true but no eval results found at $EVAL_DIR"
        fi
    fi

    # Clean up srt-slurm outputs to prevent NFS silly-rename lock files
    # from blocking the next job's checkout on this runner
    echo "Cleaning up srt-slurm outputs..."
    for i in 1 2 3 4 5; do
        rm -rf outputs 2>/dev/null && break
        echo "Retry $i/5: Waiting for NFS locks to release..."
        sleep 10
    done
    find . -name '.nfs*' -delete 2>/dev/null || true

else

    SQUASH_FILE="/home/sa-shared/containers/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
    # Point the bench script at the local MODEL_PATH resolved above instead of
    # pulling from the HF hub cache. Bench scripts skip `hf download` when
    # MODEL is a local path.
    export MODEL="$MODEL_PATH"
    MODEL_MOUNT_SPEC="$MODEL_PATH:$MODEL_PATH"
    if [[ "$MODEL_PREFIX" == "minimaxm2.7" ]]; then
        MODEL_ROOT=$(dirname "$MODEL_PATH")
        mkdir -p "$MODEL_PATH" "$MODEL_ROOT/MiniMax-M2.7-EAGLE3-draft-vocab200k"
        MODEL_MOUNT_SPEC="$MODEL_ROOT:$MODEL_ROOT"
    fi
    FRAMEWORK_SUFFIX=$([[ "$FRAMEWORK" == "trt" ]] && printf '_trt' || printf '')
    SPEC_SUFFIX=$([[ "$SPEC_DECODING" == "mtp" ]] && printf '_mtp' || printf '')
    # Prefer a framework-tagged script (e.g. dsv4_fp4_b200_vllm.sh) so models
    # with multiple inference engines can coexist; fall back to the historical
    # name without an engine suffix (`_trt` for trt, bare for everyone else).
    BENCH_BASE="benchmarks/single_node/${SCENARIO_SUBDIR}${EXP_NAME%%_*}_${PRECISION}_b200"
    BENCH_SCRIPT="${BENCH_BASE}_${FRAMEWORK}${SPEC_SUFFIX}.sh"
    if [[ ! -f "$BENCH_SCRIPT" ]]; then
        BENCH_SCRIPT="${BENCH_BASE}${FRAMEWORK_SUFFIX}${SPEC_SUFFIX}.sh"
    fi
    LOCK_FILE="${SQUASH_FILE}.lock"

    # TODO(Cam): lmsysorg/sglang:deepseek-v4-blackwell installs sglang editable at
    # /workspace/sglang/python (prior sglang tags used /sgl-workspace/sglang), so
    # the default $GITHUB_WORKSPACE:/workspace/ bind-mount masks the install and
    # breaks `import sglang`. Mount this one image at /ix instead; drop the
    # conditional once the image stops installing editable under /workspace.
    if [[ "$IMAGE" == *deepseek-v4-blackwell* ]]; then
        CONTAINER_MOUNT_DIR=/ix
    else
        CONTAINER_MOUNT_DIR=/workspace
    fi

    # b200-dgxc is partitioned into gpu-1 / gpu-2; the prior gpu-10 and gpu-15
    # names no longer exist. sa-shared's benchmark association currently has
    # only gpu-2_qos, so idle gpu-1 nodes cannot accept these jobs.
    export GPU_COUNT="${GPU_COUNT:-${TP:?TP must be set}}"

    SALLOC_TIME_LIMIT="${SALLOC_TIME_LIMIT:-480}"
    salloc --partition=$SLURM_PARTITION --account=$SLURM_ACCOUNT --gres=gpu:$GPU_COUNT --exclusive --mem=0 --time="$SALLOC_TIME_LIMIT" --no-shell --job-name="$RUNNER_NAME"
    JOB_ID=$(squeue --name="$RUNNER_NAME" -u "$USER" -h -o %A | head -n1)

    # Use flock to serialize concurrent imports to the same squash file
    # Override ENROOT_CACHE_PATH to avoid permission issues with system-wide cache on worker nodes
    srun --jobid=$JOB_ID bash -c "
        export ENROOT_CACHE_PATH=\$HOME/.cache/enroot
        mkdir -p \$ENROOT_CACHE_PATH
        exec 9>\"$LOCK_FILE\"
        flock -w 600 9 || { echo 'Failed to acquire lock for $SQUASH_FILE'; exit 1; }
        if unsquashfs -l \"$SQUASH_FILE\" > /dev/null 2>&1; then
            echo 'Squash file already exists and is valid, skipping import'
        else
            rm -f \"$SQUASH_FILE\"
            enroot import -o \"$SQUASH_FILE\" docker://$IMAGE
        fi
    "

    srun --jobid=$JOB_ID \
        --container-image=$SQUASH_FILE \
        --container-mounts=$GITHUB_WORKSPACE:$CONTAINER_MOUNT_DIR,$MODEL_MOUNT_SPEC,$AIPERF_MMAP_CACHE_HOST_PATH:/aiperf_mmap_cache \
        --no-container-mount-home \
        --container-workdir=$CONTAINER_MOUNT_DIR \
        --no-container-entrypoint --export=ALL,PORT=8888,AIPERF_DATASET_MMAP_CACHE_DIR=/aiperf_mmap_cache \
        bash "$BENCH_SCRIPT"
fi
