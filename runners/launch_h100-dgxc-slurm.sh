#!/usr/bin/bash
set -e

# System-specific configuration for H100 DGXC Slurm cluster
SLURM_PARTITION="hpc-gpu-1"
SLURM_ACCOUNT="customer"

# Route spec-decoding=mtp configs to the _mtp benchmark script (parity with
# the h200 launchers, which have carried SPEC_SUFFIX since #392).
SPEC_SUFFIX=$([[ "$SPEC_DECODING" == "mtp" ]] && printf '_mtp' || printf '')

set -x

if [[ "$IS_MULTINODE" == "true" ]]; then

    # Multinode cluster profile — declares cluster facts per the contract in
    # benchmarks/multi_node/srt_slurm/README.md and hands off to the
    # cluster-agnostic orchestrator.

    source "$(dirname "$0")/lib/multinode.sh"

    export SLURM_PARTITION SLURM_ACCOUNT

    INFX_CLUSTER="h100"
    INFX_GPUS_PER_NODE=8
    INFX_ARCH="x86_64"
    INFX_SLURM_TIME_LIMIT="6:00:00"

    INFX_SRTSLURM_EXTRA="use_gpus_per_node_directive: true
use_segment_sbatch_directive: false
use_exclusive_sbatch_directive: false"

    # uv lives on shared NFS so the (slow) installer runs at most once per
    # fleet rather than once per job.
    export UV_INSTALL_DIR="/mnt/nfs/sa-shared/.uv/bin"
    export UV_CACHE_DIR="/mnt/nfs/sa-shared/.uv/cache"
    export UV_PYTHON_INSTALL_DIR="/mnt/nfs/sa-shared/.uv/python"
    if [[ "${INFX_DRY_RUN:-0}" != "1" ]]; then
        mkdir -p "$UV_INSTALL_DIR" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"
    fi

    # MODEL_PATH / SRT_SLURM_MODEL_PREFIX / SERVED_MODEL_NAME come from the
    # model-paths registry in configs/runners.yaml.
    infx_resolve_model_paths cluster:h100-dgxc

    # Container squash files are pre-staged on this cluster (no import).
    # TRT images are referenced by recipes in pyxis format (nvcr.io#...).
    NGINX_SQUASH_FILE="/mnt/nfs/lustre/containers/nginx_1.27.4.sqsh"
    if [[ "$FRAMEWORK" == "dynamo-sglang" ]]; then
        SQUASH_FILE="/mnt/nfs/lustre/containers/lmsysorg_sglang_v0.5.8.post1-cu130.sqsh"
        INFX_CONTAINER_KEY="lmsysorg/sglang:v0.5.8-cu130"
    elif [[ "$FRAMEWORK" == "dynamo-trt" ]]; then
        INFX_CONTAINER_KEY=$(echo "$IMAGE" | sed 's|nvcr.io/|nvcr.io#|')
        SQUASH_FILE="$(infx_squash_path /mnt/nfs/sa-shared/containers "$(echo "$IMAGE" | sed 's|nvcr.io/||')" +)"
    else
        echo "Unsupported framework: $FRAMEWORK. Supported frameworks are: dynamo-trt, dynamo-sglang"
        exit 1
    fi

    # Raise sglang's torch-distributed TCPStore timeout from the 600s gloo
    # default; H100 checkpoint loads off NFS routinely exceed it.
    infx_hook_edit_recipe() {
        sed -i '/^      watchdog-timeout:/a\      dist-timeout: 1800' "$1"
    }

    source "$GITHUB_WORKSPACE/benchmarks/multi_node/srt_slurm/run.sh"

else

    HF_HUB_CACHE_MOUNT="/mnt/nfs/sa-shared/gharunners/hf-hub-cache/"
    AIPERF_MMAP_CACHE_HOST_PATH="/mnt/nfs/sa-shared/gharunners/ai-perf-cache"
    SQUASH_FILE="/mnt/nfs/lustre/containers/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
    LOCK_FILE="${SQUASH_FILE}.lock"

    export GPU_COUNT="${GPU_COUNT:-${TP:?TP must be set}}"

    salloc --partition=$SLURM_PARTITION --account=$SLURM_ACCOUNT --gres=gpu:$GPU_COUNT --exclusive --time=180 --no-shell --job-name="$RUNNER_NAME"
    JOB_ID=$(squeue --name="$RUNNER_NAME" -u "$USER" -h -o %A | head -n1)
    if [[ -z "$JOB_ID" ]]; then
        echo "ERROR: failed to resolve H100 Slurm allocation" >&2
        exit 1
    fi
    trap 'rc=$?; scancel "$JOB_ID" 2>/dev/null || true; exit "$rc"' EXIT

    # flock-serialize the enroot import so concurrent sweep jobs on the same
    # shared NFS path don't race each other into 'File already exists' (race
    # observed on PR #1509: 13/30 jobs failed, all on the dgxc-slurm runners
    # hitting the same /mnt/nfs/lustre/containers/<image>.sqsh path). Matches
    # the canonical pattern already used in launch_h100-cw.sh + the mi3xx
    # launchers. The skip-if-valid check avoids re-downloading when the file
    # was successfully created by an earlier job.
    srun --jobid=$JOB_ID bash -c "
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
        --container-mounts=$GITHUB_WORKSPACE:/workspace/,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE,$AIPERF_MMAP_CACHE_HOST_PATH:/aiperf_mmap_cache \
        --no-container-mount-home \
        --container-workdir=/workspace/ \
        --no-container-entrypoint --export=ALL,PORT=8888,AIPERF_DATASET_MMAP_CACHE_DIR=/aiperf_mmap_cache \
        bash benchmarks/single_node/${SCENARIO_SUBDIR}${EXP_NAME%%_*}_${PRECISION}_h100${SPEC_SUFFIX}.sh

    scancel $JOB_ID

fi
