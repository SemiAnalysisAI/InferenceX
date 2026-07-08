#!/usr/bin/bash

# System-specific configuration for B200 DGXC Slurm cluster
SLURM_PARTITION="gpu-2"
SLURM_ACCOUNT="benchmark"

set -x

source "$(dirname "$0")/lib/multinode.sh"

# MODEL_PATH / SRT_SLURM_MODEL_PREFIX come from the model-paths registry in
# configs/runners.yaml (cluster:b200-dgxc). Bench scripts and srt-slurm
# recipes specify HuggingFace model IDs for portability; the registry
# resolves them to the pre-staged /lustre/fsw (or /scratch/fsw) copies.
# A pre-set MODEL_PATH pointing at an existing directory wins.
infx_resolve_model_paths cluster:b200-dgxc

export AIPERF_MMAP_CACHE_HOST_PATH="/lustre/fsw/gharunners/aiperf-cache"

if [[ "$IS_MULTINODE" == "true" ]]; then

    # Multinode cluster profile — declares cluster facts per the contract in
    # benchmarks/multi_node/srt_slurm/README.md and hands off to the
    # cluster-agnostic orchestrator.

    set -eo pipefail

    export SLURM_PARTITION SLURM_ACCOUNT

    INFX_CLUSTER="b200"
    INFX_GPUS_PER_NODE=8
    INFX_ARCH="x86_64"
    INFX_SLURM_TIME_LIMIT="4:00:00"
    # Large-model loads off the shared Lustre tree (e.g. DSR1-FP8 ~680 GB)
    # need more than the recipe-default health-check budget.
    INFX_HEALTH_CHECK_MAX_ATTEMPTS=720

    INFX_SRTSLURM_EXTRA="use_exclusive_sbatch_directive: true"

    export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL}"

    NGINX_IMAGE="nginx:1.27.4"
    SQUASH_DIR="${B200_SQUASH_DIR:-/home/sa-shared/containers}"
    if ! mkdir -p "$SQUASH_DIR" 2>/dev/null || [[ ! -w "$SQUASH_DIR" ]]; then
        echo "Warning: $SQUASH_DIR is not writable; using workspace-local squash cache" >&2
        SQUASH_DIR="$GITHUB_WORKSPACE/.container-squash"
        mkdir -p "$SQUASH_DIR"
    fi
    chmod a+rx "$SQUASH_DIR" || true

    SQUASH_FILE="$(infx_squash_path "$SQUASH_DIR" "$IMAGE")"
    NGINX_SQUASH_FILE="$(infx_squash_path "$SQUASH_DIR" "$NGINX_IMAGE")"

    if [[ "${INFX_DRY_RUN:-0}" != "1" ]]; then
        infx_import_squash "$SQUASH_FILE" "$IMAGE"
        infx_import_squash "$NGINX_SQUASH_FILE" "$NGINX_IMAGE"
        chmod a+r "$SQUASH_FILE" "$NGINX_SQUASH_FILE" 2>/dev/null || true
    fi

    source "$GITHUB_WORKSPACE/benchmarks/multi_node/srt_slurm/run.sh"

else

    # Point the bench script at the local MODEL_PATH resolved above instead of
    # pulling from the HF hub cache. Bench scripts skip `hf download` when
    # MODEL is a local path.
    export MODEL="$MODEL_PATH"
    SQUASH_FILE="/home/sa-shared/containers/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
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

    # b200-dgxc cluster was re-partitioned to gpu-1 / gpu-2; the prior gpu-10
    # and gpu-15 names no longer exist. gpu-2 currently has 10 fully-idle GPU
    # nodes (all of gpu-2-[0-9]); gpu-1 has 2 drained (gpu-1-4, gpu-1-8). We
    # land on gpu-2 to avoid drained nodes and skip the per-node excludes.
    export GPU_COUNT="${GPU_COUNT:-${TP:?TP must be set}}"

    SALLOC_TIME_LIMIT="${SALLOC_TIME_LIMIT:-480}"
    salloc --partition=$SLURM_PARTITION --account=$SLURM_ACCOUNT --gres=gpu:$GPU_COUNT --exclusive --mem=0 --time="$SALLOC_TIME_LIMIT" --no-shell --job-name="$RUNNER_NAME"
    JOB_ID=$(squeue --name="$RUNNER_NAME" -u "$USER" -h -o %A | head -n1)

    # DSv4 is also staged on the compute nodes' local RAID. Loading the 806 GB
    # checkpoint independently from Lustre on every TP rank leaves the loader
    # threads blocked in Lustre I/O for hours. Select the local copy only after
    # Slurm assigns a node, and retain the shared-Lustre path as a fallback for
    # nodes whose local staging is incomplete.
    if [[ "$MODEL_PREFIX" == "dsv4" && "$PRECISION" == "fp4" && "$FRAMEWORK" == "sglang" ]]; then
        LOCAL_MODEL_PATH=/raid/models/DeepSeek-V4-Pro-NVFP4
        if srun --jobid="$JOB_ID" bash -c \
            'test -f "$1/config.json" && test -f "$1/model.safetensors.index.json" && test "$(find "$1" -maxdepth 1 -name "model-*.safetensors" | wc -l)" -eq 64' \
            _ "$LOCAL_MODEL_PATH"; then
            export MODEL_PATH="$LOCAL_MODEL_PATH"
            export MODEL="$MODEL_PATH"
            echo "Using node-local DSv4 checkpoint: $MODEL_PATH"
        else
            echo "Node-local DSv4 checkpoint unavailable; using shared checkpoint: $MODEL_PATH"
        fi
    fi

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
        --container-mounts=$GITHUB_WORKSPACE:$CONTAINER_MOUNT_DIR,$MODEL_PATH:$MODEL_PATH,$AIPERF_MMAP_CACHE_HOST_PATH:/aiperf_mmap_cache \
        --no-container-mount-home \
        --container-workdir=$CONTAINER_MOUNT_DIR \
        --no-container-entrypoint --export=ALL,PORT=8888,AIPERF_DATASET_MMAP_CACHE_DIR=/aiperf_mmap_cache \
        bash "$BENCH_SCRIPT"
fi
