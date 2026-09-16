#!/usr/bin/env bash

# Cluster settings selected by the workflow before recipe-specific overrides.
source "$(dirname "${BASH_SOURCE[0]}")/../benchmarks/benchmark_lib.sh" --validation-only
check_env_vars RUNNER_NAME MODEL_PREFIX PRECISION FRAMEWORK IS_MULTINODE IS_AGENTIC

case "${RUNNER_NAME%%_*}" in
    b200-nscale-slurm|b200-nscale-compat)
        export SLURM_PARTITION=batch_1 SLURM_ACCOUNT=benchmark
        export B200_SQUASH_DIR=/data/home/sa-shared/containers B200_SQUASH_LOCK_TIMEOUT=600
        case "$MODEL_PREFIX" in
            dsv4)
                if [[ "${RUNNER_NAME%%_*}" == b200-nscale-slurm && "$IS_MULTINODE" == true && "$FRAMEWORK" != tilert ]]; then
                    export MODEL_PATH=/scratch/models/DeepSeek-V4-Pro
                else
                    export MODEL_PATH=/scratch/models/DeepSeek-V4-Pro-0813
                fi
                ;;
            kimik2.5) export MODEL_PATH=/scratch/models/Kimi-K2.6-NVFP4 ;;
            kimik3) export MODEL_PATH=/scratch/models/Kimi-K3 ;;
            glm5.1) export MODEL_PATH=/scratch/models/GLM-5.1-FP8 ;;
            glm5.2) export MODEL_PATH=/scratch/models/GLM-5.2-NVFP4 ;;
            minimaxm2.5)
                if [[ "$FRAMEWORK" == dynamo-vllm ]]; then
                    export B200_SQUASH_DIR=/home/slurm-shared/gharunners/squash
                fi
                ;;
        esac
        if [[ "$FRAMEWORK" == tilert ]]; then
            export TILERT_WEIGHTS_DIR="/scratch/models/${MODEL_PREFIX}-${PRECISION}-tilert-8shard"
            export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1
            export UCX_MEMTYPE_CACHE=n UCX_MEMTYPE_REG_WHOLE=n
        fi
        ;;
    gb300-nv) export SLURM_PARTITION=batch_1 ;;
    h200-dgxc-slurm)
        export HF_HUB_CACHE_MOUNT=/models/gharunners/hf-hub-cache
        export AIPERF_MMAP_CACHE_HOST_PATH=/home/sa-shared/gharunners/ai-perf-cache
        export DSV4_MODEL_PATH="$HF_HUB_CACHE_MOUNT/DeepSeek-V4-Pro"
        export GLM52_FP8_MODEL_PATH=/models/GLM-5.2-FP8
        ;;
    b300-dsxe)
        check_env_vars HOME
        export B300_HF_CACHE_HOST_DIR="$HOME/.cache/huggingface"
        export B300_HF_CACHE_CONTAINER_DIR=/hf_hub_cache
        ;;
    mi355x-amds)
        check_env_vars GITHUB_WORKSPACE
        export BENCHMARK_LOGS_DIR="$GITHUB_WORKSPACE/benchmark_logs"
        ;;
    rtx6000pro-lat)
        export HF_HUB_CACHE_MOUNT=/var/lib/inferencex/hf-hub-cache
        export NCCL_IB_DISABLE=1
        ;;
esac
