#!/usr/bin/bash
set -eo pipefail

# System-specific configuration for H200 DGXC Slurm cluster
SLURM_PARTITION="main"
SLURM_ACCOUNT="sa-shared"

set -x

if [[ "$IS_MULTINODE" == "true" ]]; then

    # Multinode cluster profile — declares cluster facts per the contract in
    # benchmarks/multi_node/srt_slurm/README.md and hands off to the
    # cluster-agnostic orchestrator.

    source "$(dirname "$0")/lib/multinode.sh"

    export SLURM_PARTITION SLURM_ACCOUNT

    INFX_CLUSTER="h200"
    INFX_GPUS_PER_NODE=8
    INFX_ARCH="x86_64"
    INFX_SLURM_TIME_LIMIT="4:00:00"
    # Large-model loads off the shared filesystem need more than the
    # recipe-default health-check budget.
    INFX_HEALTH_CHECK_MAX_ATTEMPTS=720

    INFX_SRTSLURM_EXTRA="use_gpus_per_node_directive: true
use_segment_sbatch_directive: false
use_exclusive_sbatch_directive: false"

    # MODEL_PATH / SRT_SLURM_MODEL_PREFIX / SERVED_MODEL_NAME come from the
    # model-paths registry in configs/runners.yaml.
    infx_resolve_model_paths cluster:h200-dgxc

    # Container squash files are pre-staged on this cluster (no import).
    # TRT images are referenced by recipes in pyxis format (nvcr.io#...).
    NGINX_SQUASH_FILE="/data/containers/nginx+1.27.4.sqsh"
    if [[ "$FRAMEWORK" == "dynamo-sglang" ]]; then
        SQUASH_FILE="$(infx_squash_path /data/containers "$IMAGE" +)"
        INFX_CONTAINER_KEY="$IMAGE"
    elif [[ "$FRAMEWORK" == "dynamo-trt" ]]; then
        INFX_CONTAINER_KEY=$(echo "$IMAGE" | sed 's|nvcr.io/|nvcr.io#|')
        SQUASH_FILE="$(infx_squash_path /data/containers "$(echo "$IMAGE" | sed 's|nvcr.io/||')" +)"
    else
        echo "Unsupported framework: $FRAMEWORK. Supported frameworks are: dynamo-trt, dynamo-sglang"
        exit 1
    fi

    source "$GITHUB_WORKSPACE/benchmarks/multi_node/srt_slurm/run.sh"

else

    HF_HUB_CACHE_MOUNT="/models/gharunners/hf-hub-cache"
    AIPERF_MMAP_CACHE_HOST_PATH="/home/sa-shared/gharunners/ai-perf-cache"
    SQUASH_FILE="/data/gharunners/containers/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"

    # Convert pyxis image format (nvcr.io#path) to docker format (nvcr.io/path) for enroot import
    DOCKER_IMAGE=$(echo "$IMAGE" | sed 's/#/\//g')
    LOCK_FILE="${SQUASH_FILE}.lock"

    export GPU_COUNT="${GPU_COUNT:-${TP:?TP must be set}}"

    salloc --partition=$SLURM_PARTITION --account=$SLURM_ACCOUNT --gres=gpu:$GPU_COUNT --exclusive --time=180 --no-shell --job-name="$RUNNER_NAME"
    JOB_ID=$(squeue --name="$RUNNER_NAME" -u "$USER" -h -o %A | head -n1)
    if [[ -z "$JOB_ID" ]]; then
        echo "ERROR: failed to resolve H200 Slurm allocation" >&2
        exit 1
    fi
    trap 'rc=$?; scancel "$JOB_ID" 2>/dev/null || true; exit "$rc"' EXIT

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
            enroot import -o \"$SQUASH_FILE\" docker://$DOCKER_IMAGE
        fi
    "

    SPEC_SUFFIX=$([[ "$SPEC_DECODING" == "mtp" ]] && printf '_mtp' || printf '')
    BENCH_BASE="benchmarks/single_node/${SCENARIO_SUBDIR}${EXP_NAME%%_*}_${PRECISION}_h200"
    BENCH_SCRIPT="${BENCH_BASE}_${FRAMEWORK}${SPEC_SUFFIX}.sh"
    if [[ ! -f "$BENCH_SCRIPT" ]]; then
        LEGACY_FW_SUFFIX=$([[ "$FRAMEWORK" == "trt" ]] && printf '_trt' || printf '')
        BENCH_SCRIPT="${BENCH_BASE}${LEGACY_FW_SUFFIX}${SPEC_SUFFIX}.sh"
    fi

    if [[ "$IMAGE" == *deepseek-v4-hopper* ]]; then
        CONTAINER_MOUNT_DIR=/ix
    else
        CONTAINER_MOUNT_DIR=/workspace
    fi

    srun --jobid=$JOB_ID \
        --container-image=$SQUASH_FILE \
        --container-mounts=$GITHUB_WORKSPACE:$CONTAINER_MOUNT_DIR/,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE,$AIPERF_MMAP_CACHE_HOST_PATH:/aiperf_mmap_cache \
        --no-container-mount-home \
        --container-remap-root \
        --container-workdir=$CONTAINER_MOUNT_DIR/ \
        --no-container-entrypoint --export=ALL,PORT=8888,AIPERF_DATASET_MMAP_CACHE_DIR=/aiperf_mmap_cache \
        bash $BENCH_SCRIPT

    scancel $JOB_ID

fi
