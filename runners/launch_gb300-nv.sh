#!/usr/bin/bash

# Cluster profile for the GB300 NV Slurm cluster (sa-shared).
#
# This file declares cluster facts and satisfies the contract documented in
# benchmarks/multi_node/srt_slurm/README.md, then hands off to the
# cluster-agnostic orchestrator. Cluster-specific hardcoding is expected
# here and nowhere else.

set -exo pipefail

source "$(dirname "$0")/lib/multinode.sh"

export SLURM_PARTITION="batch_1"
export SLURM_ACCOUNT="benchmark"
export ENROOT_ROOTFS_WRITABLE=1

INFX_CLUSTER="gb300"
INFX_GPUS_PER_NODE=4
INFX_ARCH="aarch64"
INFX_SLURM_TIME_LIMIT="4:00:00"

# Host-side persistent caches bind-mounted into every worker container:
# aiperf's content-addressed dataset mmap cache (~65 GB re-tokenized from
# scratch without it) and the HF hub cache holding trace-dataset downloads.
# Use the /data/ mount (not /home/sa-shared/) — both are the same Vast NFS
# backing storage, but the /home/sa-shared/ mount has a chronic ELOOP /
# "Too many levels of symbolic links" bug from workflow worker NFS sessions;
# /data/ has a separate client cache that isn't poisoned.
AIPERF_MMAP_CACHE_HOST_PATH="/data/home/sa-shared/gharunners/ai-perf-cache"
HF_HUB_CACHE_HOST_PATH="/data/home/sa-shared/gharunners/hf-hub-cache"

# MODEL_PATH / SRT_SLURM_MODEL_PREFIX / SERVED_MODEL_NAME come from the
# model-paths registry in configs/runners.yaml.
infx_resolve_model_paths cluster:gb300-nv

NGINX_IMAGE="nginx:1.27.4"
SQUASH_FILE="$(infx_squash_path /data/home/sa-shared/gharunners/squash "$IMAGE")"
NGINX_SQUASH_FILE="$(infx_squash_path /data/home/sa-shared/gharunners/squash "$NGINX_IMAGE")"

INFX_SRTSLURM_EXTRA="# srtctl defaults use_segment to true, which adds #SBATCH --segment=<nodes>.
# The batch partition is a single NVL72 rack, so segment contiguity buys
# nothing for MNNVL but makes jobs unschedulable when the partition is
# fragmented.
use_segment_sbatch_directive: false
default_mounts:
  \"${AIPERF_MMAP_CACHE_HOST_PATH}\": \"/aiperf_mmap_cache\"
  \"${HF_HUB_CACHE_HOST_PATH}\": \"/hf_hub_cache\""

if [[ "${INFX_DRY_RUN:-0}" != "1" ]]; then
    mkdir -p "$HF_HUB_CACHE_HOST_PATH"
    # Import on a compute node via srun, not on the login node: the login
    # node is x86_64 while the compute nodes are aarch64, so the arm64
    # squash file has to be built on a compute node.
    infx_import_squash_srun "$SQUASH_FILE" "$IMAGE" \
        --partition="$SLURM_PARTITION" --exclusive --time=180
    infx_import_squash_srun "$NGINX_SQUASH_FILE" "$NGINX_IMAGE" \
        --partition="$SLURM_PARTITION" --exclusive --time=180
fi

source "$GITHUB_WORKSPACE/benchmarks/multi_node/srt_slurm/run.sh"
