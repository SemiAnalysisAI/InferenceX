#!/usr/bin/env bash

# === Workflow-defined Env Vars ===
# IMAGE
# MODEL
# TP
# HF_HUB_CACHE
# ISL
# OSL
# MAX_MODEL_LEN
# RANDOM_RANGE_RATIO
# CONC
# GITHUB_WORKSPACE
# RESULT_FILENAME
# HF_TOKEN

MODEL_CODE="${EXP_NAME%%_*}"
HF_HUB_CACHE_MOUNT="/nfsdata/hf_hub_cache/"

set -x
srun --reservation=compute --exclusive \
--gres=gpu:$TP --cpus-per-task=256 --ntasks-per-node=1 --time=180 \
--container-image=$IMAGE \
--container-name="${MODEL_CODE}_container" \
--container-mounts=$GITHUB_WORKSPACE:/workspace/,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE \
--container-mount-home \
--container-writable \
--container-remap-root \
--container-workdir=/workspace/ \
--no-container-entrypoint --export=ALL,PORT=8888 \
bash benchmarks/${MODEL_CODE}_${PRECISION}_mi355x_slurm.sh
