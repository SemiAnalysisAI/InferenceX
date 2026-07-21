#!/usr/bin/bash

# System-specific configuration for B300 NV Slurm cluster (sa-shared)
SLURM_PARTITION="batch_1"
SLURM_ACCOUNT="benchmark"
# b300-018 repeatedly times out UCX/NIXL transfers; allow an empty override to disable this.
MINIMAX_M3_SLURM_EXCLUDED_NODELIST="${MINIMAX_M3_SLURM_EXCLUDED_NODELIST-b300-018}"

set -x

if [[ "$IS_MULTINODE" == "true" ]]; then

# Validate framework
if [[ $FRAMEWORK != "dynamo-sglang" && $FRAMEWORK != "dynamo-trt" && $FRAMEWORK != "dynamo-vllm" ]]; then
    echo "Unsupported framework: $FRAMEWORK. Supported frameworks are: dynamo-trt, dynamo-sglang, dynamo-vllm"
    exit 1
fi

# MODEL_PATH: Override with pre-downloaded paths on B300 runner
# The yaml files specify HuggingFace model IDs for portability, but we use
# local paths to avoid repeated downloading on the shared B300 cluster.
if [[ $MODEL_PREFIX == "dsr1" && $PRECISION == "fp4" ]]; then
    export MODEL_PATH="/data/models/dsr1-fp4"
    export SERVED_MODEL_NAME="deepseek-r1-fp4"
    export SRT_SLURM_MODEL_PREFIX="dsr1"
elif [[ $MODEL_PREFIX == "dsr1" && $PRECISION == "fp8" ]]; then
    export MODEL_PATH="/data/models/dsr1-fp8"
    export SERVED_MODEL_NAME="deepseek-r1-fp8"
    export SRT_SLURM_MODEL_PREFIX="dsr1-fp8"
elif [[ $MODEL_PREFIX == "dsv4" && $PRECISION == "fp4" && $FRAMEWORK == "dynamo-vllm" ]]; then
    SELECTED_MODEL_PATH=""
    if [[ -n "${MODEL_PATH:-}" && -d "${MODEL_PATH}" ]]; then
        SELECTED_MODEL_PATH="$MODEL_PATH"
    else
        for candidate in /data/models/dsv4-pro /data/models/deepseek-v4-pro /data/models/DeepSeek-V4-Pro; do
            if [[ -d "$candidate" ]]; then
                SELECTED_MODEL_PATH="$candidate"
                break
            fi
        done
    fi
    export MODEL_PATH="${SELECTED_MODEL_PATH:-/data/models/dsv4-pro}"
    export SRT_SLURM_MODEL_PREFIX="deepseek-v4-pro"
elif [[ $MODEL_PREFIX == "minimaxm2.5" && $PRECISION == "fp4" && $FRAMEWORK == "dynamo-vllm" ]]; then
    export MODEL_PATH="/data/models/MiniMax-M2.5-NVFP4"
    export SRT_SLURM_MODEL_PREFIX="minimax-m2.5-nvfp4"
elif [[ $MODEL_PREFIX == "minimaxm2.5" && $PRECISION == "fp8" && $FRAMEWORK == "dynamo-vllm" ]]; then
    export MODEL_PATH="/data/models/MiniMax-M2.5"
    export SRT_SLURM_MODEL_PREFIX="minimax-m2.5-fp8"
elif [[ $MODEL_PREFIX == "minimaxm3" && $PRECISION == "fp4" && $FRAMEWORK == "dynamo-vllm" ]]; then
    export MODEL_PATH="/scratch/models/MiniMax-M3-NVFP4"
    export SRT_SLURM_MODEL_PREFIX="nvidia/MiniMax-M3-NVFP4"
elif [[ $MODEL_PREFIX == "minimaxm3" && $PRECISION == "fp8" && $FRAMEWORK == "dynamo-vllm" ]]; then
    export MODEL_PATH="/data/models/MiniMax-M3-MXFP8"
    export SRT_SLURM_MODEL_PREFIX="MiniMaxAI/MiniMax-M3-MXFP8"
else
    echo "Unsupported model: $MODEL_PREFIX-$PRECISION. Supported models are: dsr1-fp4, dsr1-fp8, dsv4-fp4 with dynamo-vllm, minimaxm2.5-fp4 with dynamo-vllm, minimaxm2.5-fp8 with dynamo-vllm, minimaxm3-fp4 with dynamo-vllm, minimaxm3-fp8 with dynamo-vllm"
    exit 1
fi

echo "Cloning srt-slurm repository..."
SRT_REPO_DIR="srt-slurm"
SRTCTL_SETUP_SCRIPT=""
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
elif [[ $FRAMEWORK == "dynamo-vllm" && $MODEL_PREFIX == "minimaxm3" && $PRECISION == "fp4" && "$CONFIG_FILE" == recipes/vllm/minimax-m3/b300-fp4/8k1k/mtp/*.yaml ]]; then
    git clone --branch main --single-branch https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR" || exit 1
    git checkout c1b6b5c97f323baefad577d70c4e8392b6f537d9
    mkdir -p recipes/vllm/minimax-m3
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/vllm/minimax-m3" recipes/vllm/minimax-m3
elif [[ $FRAMEWORK == "dynamo-vllm" && $MODEL_PREFIX == "minimaxm3" && $PRECISION == "fp4" && "$CONFIG_FILE" == recipes/vllm/minimax-m3/b300-fp4/8k1k/*-tp1-*.yaml ]]; then
    git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR" || exit 1
    git checkout c1fb6989fc5aca803b4ca0f2d17d8be85fad9732
    mkdir -p recipes/vllm/minimax-m3
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/vllm/minimax-m3" recipes/vllm/minimax-m3
elif [[ $FRAMEWORK == "dynamo-vllm" && $MODEL_PREFIX == "minimaxm3" && ( $PRECISION == "fp4" || $PRECISION == "fp8" ) ]]; then
    git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR" || exit 1
    git checkout sa-submission-q2-2026
    mkdir -p recipes/vllm/minimax-m3
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/vllm/minimax-m3" recipes/vllm/minimax-m3
    if [[ $PRECISION == "fp8" ]]; then
        SRTCTL_SETUP_SCRIPT="minimax-m3-vllm-fixes.sh"
    fi
    # NVIDIA/srt-slurm#38
    git show 22d46ba9971615016d2339c9ffbc7b4597accfad --format= -- src/srtctl/core/ip_utils/get_node_ip.sh | git apply - || exit 1
    if [[ -n "$SRTCTL_SETUP_SCRIPT" ]]; then
        cp \
            "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/configs/$SRTCTL_SETUP_SCRIPT" \
            "configs/$SRTCTL_SETUP_SCRIPT"
    fi
else
    git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR" || exit 1
    git checkout sa-submission-q2-2026
fi

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

# Map container images to local squash files
NGINX_IMAGE="nginx:1.27.4"
SQUASH_FILE="/data/squash/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
NGINX_SQUASH_FILE="/data/squash/$(echo "$NGINX_IMAGE" | sed 's/[\/:@#]/_/g').sqsh"

# Import containers via enroot
srun -N 1 -A $SLURM_ACCOUNT -p $SLURM_PARTITION bash -c "enroot import -o $SQUASH_FILE docker://$IMAGE"
srun -N 1 -A $SLURM_ACCOUNT -p $SLURM_PARTITION bash -c "enroot import -o $NGINX_SQUASH_FILE docker://$NGINX_IMAGE"

export ISL="$ISL"
export OSL="$OSL"
export EVAL_ONLY="${EVAL_ONLY:-false}"

# Create srtslurm.yaml for srtctl
SRTCTL_ROOT="${GITHUB_WORKSPACE}/${SRT_REPO_DIR}"
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
srtctl_root: "${SRTCTL_ROOT}"
# Model path aliases
model_paths:
  "${SRT_SLURM_MODEL_PREFIX}": "${MODEL_PATH}"
# Container aliases
containers:
  dynamo-trtllm: "${SQUASH_FILE}"
  dynamo-sglang: "${SQUASH_FILE}"
  dynamo-vllm: "${SQUASH_FILE}"
  "${IMAGE}": "${SQUASH_FILE}"
  nginx-sqsh: "${NGINX_SQUASH_FILE}"
use_exclusive_sbatch_directive: true
default_mounts:
  "/opt/ucx-no-ud": "/usr/local/ucx"
EOF

echo "Generated srtslurm.yaml:"
cat srtslurm.yaml

echo "Running make setup..."
make setup ARCH=x86_64

# Export eval-related env vars for srt-slurm post-benchmark eval
export INFMAX_WORKSPACE="$GITHUB_WORKSPACE"

echo "Submitting job with srtctl..."

if [[ -z "$CONFIG_FILE" ]]; then
    echo "Error: CONFIG_FILE is not set. The srt-slurm path requires a CONFIG_FILE in additional-settings." >&2
    echo "Config: MODEL_PREFIX=${MODEL_PREFIX} PRECISION=${PRECISION} FRAMEWORK=${FRAMEWORK}" >&2
    exit 1
fi

# Override the job name in the config file with the runner name
sed -i "s/^name:.*/name: \"${RUNNER_NAME}\"/" "$CONFIG_FILE"
if [[ "$MODEL_PREFIX" == "minimaxm3" && -n "$MINIMAX_M3_SLURM_EXCLUDED_NODELIST" ]]; then
    sed -i "/^name:.*/a sbatch_directives:\n  exclude: \"${MINIMAX_M3_SLURM_EXCLUDED_NODELIST}\"" "$CONFIG_FILE"
fi
SRTCTL_APPLY_ARGS=(
    -f "$CONFIG_FILE"
    --tags "b300,${MODEL_PREFIX},${PRECISION},${ISL}x${OSL},infmax-$(date +%Y%m%d)"
)
# The MTP and TP1 8k1k recipes use newer srt-slurm revisions whose preflight checks
# model.path on this GHA login host. MiniMax-M3 NVFP4 is intentionally staged
# under compute-node-local /scratch (as in the original B300 submission), so
# the login host cannot stat it even though workers can. Keep this bypass
# scoped to those recipe sets; runtime model loading still validates the path.
if [[ $FRAMEWORK == "dynamo-vllm" && $MODEL_PREFIX == "minimaxm3" && $PRECISION == "fp4" && ( "$CONFIG_FILE" == recipes/vllm/minimax-m3/b300-fp4/8k1k/mtp/*.yaml || "$CONFIG_FILE" == recipes/vllm/minimax-m3/b300-fp4/8k1k/*-tp1-*.yaml ) ]]; then
    SRTCTL_APPLY_ARGS+=(--no-preflight)
fi
if [[ -n "$SRTCTL_SETUP_SCRIPT" ]]; then
    SRTCTL_APPLY_ARGS+=(--setup-script "$SRTCTL_SETUP_SCRIPT")
fi
SRTCTL_OUTPUT=$(srtctl apply "${SRTCTL_APPLY_ARGS[@]}" 2>&1)
echo "$SRTCTL_OUTPUT"

# Extract JOB_ID from srtctl output
JOB_ID=$(echo "$SRTCTL_OUTPUT" | grep -oP '✅ Job \K[0-9]+' || echo "$SRTCTL_OUTPUT" | grep -oP 'Job \K[0-9]+')

set +x

if [ -z "$JOB_ID" ]; then
    echo "Error: Failed to extract JOB_ID from srtctl output"
    exit 1
fi

if [[ "$MODEL_PREFIX" == "minimaxm3" && -n "$MINIMAX_M3_SLURM_EXCLUDED_NODELIST" ]]; then
    SBATCH_SCRIPT="outputs/$JOB_ID/sbatch_script.sh"
    if ! grep -Fq "#SBATCH --exclude=${MINIMAX_M3_SLURM_EXCLUDED_NODELIST}" "$SBATCH_SCRIPT"; then
        echo "Error: Slurm node exclusion was not rendered in $SBATCH_SCRIPT" >&2
        scancel "$JOB_ID" || true
        exit 1
    fi
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
    # HF_HUB_CACHE is set to help with dataset download inside the container
    # for eval jobs. Can be updated to some other path on the cluster and
    # mounted just like HF_HUB_CACHE_MOUNT.
    export HF_HUB_CACHE="$HOME/.cache/huggingface"

    # HF_HUB_CACHE_MOUNT is read-only and holds the pre-staged weights below.
    # WRITABLE_MODELS_DIR is writable; the benchmark script downloads anything not
    # in the staged list there.
    HF_HUB_CACHE_MOUNT="/scratch/models/"
    WRITABLE_MODELS_DIR="/data/models/"

    # Pre-staged model 
    STAGED_MODELS=(
        DeepSeek-R1-0528
        DeepSeek-R1-0528-NVFP4-v2
        DeepSeek-V4-Flash
        DeepSeek-V4-Pro
        GLM-5-FP8
        GLM-5-NVFP4
        GLM-5.1
        Kimi-K2.5
        Kimi-K2.5-NVFP4
        Kimi-K2.6
        Kimi-K2.6-NVFP4
        MiniMax-M2.5
        MiniMax-M2.5-NVFP4
        MiniMax-M2.7
        MiniMax-M2.7-NVFP4
        MiniMax-M3
        MiniMax-M3-NVFP4
        Qwen3.5-397B-A17B
        Qwen3.5-397B-A17B-FP8
        Qwen3.5-397B-A17B-NVFP4
        gpt-oss-120b
    )

    # MODEL stays as the HF id for the client (--served-model-name, tokenizer);
    # MODEL_PATH is what the server reads weights from.
    MODEL_BASENAME="${MODEL##*/}"
    if [[ " ${STAGED_MODELS[*]} " == *" ${MODEL_BASENAME} "* ]]; then
        export MODEL_PATH="${HF_HUB_CACHE_MOUNT%/}/${MODEL_BASENAME}"
    else
        export MODEL_PATH="${WRITABLE_MODELS_DIR%/}/${MODEL_BASENAME}"
    fi

    SQUASH_FILE="/data/squash/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
    SPEC_SUFFIX=$([[ "$SPEC_DECODING" == "mtp" ]] && printf '_mtp' || printf '')
    # Prefer a framework-tagged script (e.g. dsv4_fp4_b300_sglang.sh) so models
    # with multiple inference engines can coexist; fall back to the historical
    # name without an engine suffix (`_trt` for trt, bare for everyone else)
    # for scripts that haven't been retagged yet.
    BENCH_BASE="benchmarks/single_node/${SCENARIO_SUBDIR}${EXP_NAME%%_*}_${PRECISION}_b300"
    BENCH_SCRIPT="${BENCH_BASE}_${FRAMEWORK}${SPEC_SUFFIX}.sh"
    if [[ ! -f "$BENCH_SCRIPT" ]]; then
        LEGACY_FW_SUFFIX=$([[ "$FRAMEWORK" == "trt" ]] && printf '_trt' || printf '')
        BENCH_SCRIPT="${BENCH_BASE}${LEGACY_FW_SUFFIX}${SPEC_SUFFIX}.sh"
    fi

    # Allow callers (e.g. the speedbench-al.yml AL-collection workflow) to run a
    # specific script instead of the auto-selected throughput benchmark.
    if [[ -n "${BENCH_SCRIPT_OVERRIDE:-}" ]]; then
        BENCH_SCRIPT="$BENCH_SCRIPT_OVERRIDE"
    fi

    LOCK_FILE="${SQUASH_FILE}.lock"

    # TODO(Cam): the deepseek-v4 sglang images (lmsysorg/sglang:deepseek-v4-blackwell
    # and its B300-recompiled forks like yhyang201/sglang-b300) install sglang
    # editable at /workspace/sglang/python (prior sglang tags used /sgl-workspace/sglang),
    # so the default $GITHUB_WORKSPACE:/workspace/ bind-mount masks the install
    # and breaks `import sglang`. Mount these images at /ix instead; drop the
    # conditional once the image stops installing editable under /workspace.
    if [[ "$IMAGE" == *deepseek-v4-blackwell* || "$IMAGE" == *deepseek-v4-bw-ultra* || "$IMAGE" == *deepseek-v4-b300* || "$IMAGE" == *sglang-b300* ]]; then
        CONTAINER_MOUNT_DIR=/ix
    else
        CONTAINER_MOUNT_DIR=/workspace
    fi

    # Import the squash file on the head node (outside any srun) under flock.
    # Parallel GH jobs target the same shared squash path; flock serializes
    # imports so only one job pulls and writes the file while the rest wait.
    (
        exec 9>"$LOCK_FILE"
        flock -w 600 9 || { echo "Failed to acquire lock for $SQUASH_FILE" >&2; exit 1; }
        if unsquashfs -l "$SQUASH_FILE" > /dev/null 2>&1; then
            echo "Squash file already exists and is valid, skipping import"
        else
            rm -f "$SQUASH_FILE"
            # enroot's working dirs are pinned to NFS /scratch by
            # /etc/enroot/enroot.conf, but enroot-aufs2ovlfs unpacks the image's
            # root-owned whiteout markers into a sticky /tmp and then can't unlink
            # them over NFS -- root-squash strips the CAP_FOWNER it would need, so
            # it fails with "failed to remove aufs whiteout: Operation not
            # permitted" and writes no .sqsh. Run the import on local disk, where
            # the extracted files are owned by us and removable. Scoped to this
            # subshell (and cleaned up on exit), so the salloc/srun below and the
            # compute node's own /scratch are unaffected.
            enroot_local="$(mktemp -d /tmp/enroot-import.XXXXXX)"
            trap 'rm -rf "$enroot_local"' EXIT
            export ENROOT_TEMP_PATH="$enroot_local/tmp"
            export ENROOT_CACHE_PATH="$enroot_local/cache"
            export ENROOT_DATA_PATH="$enroot_local/data"
            export ENROOT_RUNTIME_PATH="$enroot_local/run"
            mkdir -p "$ENROOT_TEMP_PATH" "$ENROOT_CACHE_PATH" \
                     "$ENROOT_DATA_PATH" "$ENROOT_RUNTIME_PATH"
            enroot import -o "$SQUASH_FILE" "docker://$IMAGE"
        fi
    )

    export GPU_COUNT="${GPU_COUNT:-${TP:?TP must be set}}"

    SALLOC_TIME_LIMIT="${SALLOC_TIME_LIMIT:-480}"
    salloc --partition=$SLURM_PARTITION --account=$SLURM_ACCOUNT -N 1 --gres=gpu:$GPU_COUNT --exclusive --mem=0 --time="$SALLOC_TIME_LIMIT" --no-shell --job-name="$RUNNER_NAME"
    JOB_ID=$(squeue --name="$RUNNER_NAME" -u "$USER" -h -o %A | head -n1)

    srun --jobid=$JOB_ID \
        --mpi=none \
        --container-image=$SQUASH_FILE \
        --container-mounts=$GITHUB_WORKSPACE:$CONTAINER_MOUNT_DIR,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE_MOUNT,$WRITABLE_MODELS_DIR:$WRITABLE_MODELS_DIR \
        --no-container-mount-home \
        --container-remap-root \
        --container-workdir=$CONTAINER_MOUNT_DIR \
        --no-container-entrypoint --export=ALL,PORT=8888 \
        bash "$BENCH_SCRIPT"

fi
