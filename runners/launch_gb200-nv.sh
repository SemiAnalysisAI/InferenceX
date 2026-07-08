#!/usr/bin/bash

# Cluster profile for the GB200 fleets behind the gb200-nv_* runners.
#
# This file declares cluster facts and satisfies the contract documented in
# benchmarks/multi_node/srt_slurm/README.md, then hands off to the
# cluster-agnostic orchestrator. Cluster-specific hardcoding is expected
# here and nowhere else.
#
# Two physical clusters host these runners: the legacy NVIDIA Lustre
# cluster and the Oracle Cloud "watchtower" cluster. Watchtower does not
# cross-mount /home/slurm-shared (where GITHUB_WORKSPACE lives) to compute
# nodes, so watchtower-hosted sweeps stage the srt-slurm workspace and the
# InferenceX checkout onto a compute-visible shared filesystem.

set -eo pipefail
set -x

source "$(dirname "$0")/lib/multinode.sh"

export SLURM_PARTITION="batch"
export SLURM_ACCOUNT="benchmark"

INFX_CLUSTER="gb200"
INFX_GPUS_PER_NODE=4
INFX_ARCH="aarch64"
INFX_SLURM_TIME_LIMIT="6:00:00"

# MODEL_PATH / SRT_SLURM_MODEL_PREFIX / SERVED_MODEL_NAME come from the
# model-paths registry in configs/runners.yaml. Unregistered models fall
# back to the HF id so the server downloads from the Hub.
infx_resolve_model_paths cluster:gb200-nv --fallback-to-model "$MODEL"

# Sweeps for these models run on watchtower-hosted gb200 runners.
uses_watchtower_shared_fs() {
    case "$MODEL_PREFIX" in
        minimaxm2.5|minimaxm3|kimik2.5) return 0 ;;
        *) return 1 ;;
    esac
}

INFX_SRTSLURM_EXTRA="# On watchtower the whole batch partition is a single NVL72 rack, so
# segment contiguity buys nothing for MNNVL — but it DOES make jobs
# unschedulable when the partition is fragmented.
use_segment_sbatch_directive: false"

NGINX_IMAGE="nginx:1.27.4"
SQUASH_DIR="/mnt/lustre01/users-public/sa-shared"

if [[ "${INFX_DRY_RUN:-0}" != "1" ]]; then
    if uses_watchtower_shared_fs; then
        echo "=== cluster diagnostic (watchtower sweep) ==="
        echo "USER=$(id -un) UID=$(id -u) HOME=$HOME HOSTNAME=$(hostname -f 2>/dev/null || hostname)"
        mount | grep -E 'lustre|nfs|home|shared|/mnt' || true
        ls -ld /mnt/lustre01/users/* /mnt/lustre01/users-public/* /mnt/lustre01/groups/* 2>/dev/null || true
        echo "=== end diagnostic ==="

        # Pick a squash dir that is writable here AND visible to compute.
        SQUASH_DIR="$(infx_probe_writable_dir \
            /mnt/lustre01/users/slurm-shared/squash \
            /mnt/lustre01/users-public/slurm-shared/squash \
            /mnt/lustre01/groups/slurm-shared/squash \
            /mnt/lustre01/users-public/sa-shared \
            /nfs/slurm-shared/squash \
            /home/slurm-shared/gharunners/squash)" || exit 1
        echo "Selected SQUASH_DIR=$SQUASH_DIR"
    fi

    SQUASH_FILE="$(infx_squash_path "$SQUASH_DIR" "$IMAGE")"
    NGINX_SQUASH_FILE="$(infx_squash_path "$SQUASH_DIR" "$NGINX_IMAGE")"
    infx_import_squash "$SQUASH_FILE" "$IMAGE"
    infx_import_squash "$NGINX_SQUASH_FILE" "$NGINX_IMAGE"

    if uses_watchtower_shared_fs; then
        # srt-slurm clone + venv + outputs must live where compute can see
        # them.
        SHARED_BASE="$(infx_probe_writable_dir \
            /mnt/lustre01/users-public/sa-shared/gha-runs \
            /mnt/lustre01/users/slurm-shared/gha-runs \
            /mnt/lustre01/users-public/slurm-shared/gha-runs \
            /mnt/lustre01/groups/slurm-shared/gha-runs \
            /nfs/slurm-shared/gha-runs \
            /home/slurm-shared/gharunners/gha-runs)" || exit 1
        INFX_SRT_WORK_DIR="$SHARED_BASE"

        # Compute nodes inherit the activated venv through the shared-FS
        # repo dir; pin its interpreter to a path that exists at the same
        # location on both head and compute nodes.
        [[ -x /usr/bin/python3 ]] && INFX_VENV_PYTHON=/usr/bin/python3

        # Stage the InferenceX checkout (recipes mount it into containers)
        # onto shared FS as well.
        RUN_KEY="${GITHUB_RUN_ID:-manual}-${GITHUB_RUN_ATTEMPT:-0}-${RUNNER_NAME:-gb200-nv}-$$"
        SHARED_INFMAX_WORKSPACE="${SHARED_BASE}/infmax-workspace-${RUN_KEY}"
        mkdir -p "$SHARED_INFMAX_WORKSPACE"
        rsync -a --delete \
            --exclude='.git/' \
            --exclude='srt-slurm*/' \
            --exclude='outputs/' \
            --exclude='LOGS/' \
            --exclude='*.sqsh' \
            "${GITHUB_WORKSPACE}/" "${SHARED_INFMAX_WORKSPACE}/"
        export INFMAX_WORKSPACE="$SHARED_INFMAX_WORKSPACE"
        echo "Using shared-FS INFMAX_WORKSPACE=$INFMAX_WORKSPACE (compute-visible)"
    fi

    if [[ "${IS_AGENTIC:-0}" == "1" ]]; then
        # Persistent aiperf mmap + HF hub caches, bind-mounted into every
        # worker container (see the agentic recipes' benchmark.env).
        AIPERF_MMAP_CACHE_HOST_PATH="/mnt/lustre01/users-public/sa-shared/ai-perf-cache"
        HF_HUB_CACHE_HOST_PATH="/mnt/lustre01/users-public/sa-shared/hf-hub-cache"
        mkdir -p "$AIPERF_MMAP_CACHE_HOST_PATH" "$HF_HUB_CACHE_HOST_PATH"
        chmod 777 "$AIPERF_MMAP_CACHE_HOST_PATH" "$HF_HUB_CACHE_HOST_PATH" 2>/dev/null || true
        INFX_SRTSLURM_EXTRA="$INFX_SRTSLURM_EXTRA
default_mounts:
  \"${AIPERF_MMAP_CACHE_HOST_PATH}\": \"/aiperf_mmap_cache\"
  \"${HF_HUB_CACHE_HOST_PATH}\": \"/hf_hub_cache\""
    fi
else
    SQUASH_FILE="$(infx_squash_path "$SQUASH_DIR" "$IMAGE")"
    NGINX_SQUASH_FILE="$(infx_squash_path "$SQUASH_DIR" "$NGINX_IMAGE")"
fi

# All dynamo-sglang recipes on this cluster need torchao installed into the
# worker containers.
if [[ "$FRAMEWORK" == "dynamo-sglang" && -z "${SRTCTL_SETUP_SCRIPT:-}" ]]; then
    SRTCTL_SETUP_SCRIPT="install-torchao.sh"
fi

source "$GITHUB_WORKSPACE/benchmarks/multi_node/srt_slurm/run.sh"
