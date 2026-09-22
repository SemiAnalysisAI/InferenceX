#!/usr/bin/env bash

source "$(dirname "${BASH_SOURCE[0]}")/../benchmarks/benchmark_lib.sh" --validation-only || exit 1
source "$(dirname "${BASH_SOURCE[0]}")/slurm_utils.sh" || exit 1
check_env_vars EVAL_ONLY IS_AGENTIC IS_MULTINODE KEEP_LOGS RUN_EVAL

# One cluster entry point. Migrated recipes delegate the entire allocation and
# serving lifecycle to srt-slurm; the adapter only maps CI inputs and artifacts.
if [[ -n "${CONFIG_FILE:-}" ]]; then
    source "$(dirname "${BASH_SOURCE[0]}")/srt_runtime.sh" || exit 1
    run_srt_recipe_job
    exit $?
fi

if [[ "$IS_MULTINODE" == "true" ]]; then
    echo "MI355X multi-node jobs require a CONFIG_FILE srt-slurm recipe." >&2
    exit 1
else

    export HF_HUB_CACHE_MOUNT="/var/lib/hf-hub-cache/"
    export AIPERF_MMAP_CACHE_HOST_PATH="/it-share/aiperf-cache/"
    export PORT_OFFSET=${RUNNER_NAME: -1}
    export PORT=$(( 8888 + ${PORT_OFFSET} ))
    FRAMEWORK_SUFFIX=$([[ "$FRAMEWORK" == "atom" ]] && printf '_atom' || printf '')
    SPEC_SUFFIX=$([[ "$SPEC_DECODING" == "mtp" || "$SPEC_DECODING" == "draft_model" ]] && printf '_mtp' || printf '')

    PARTITION="compute"
    SQUASH_FILE="/var/lib/squash/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
    LOCK_FILE="${SQUASH_FILE}.lock"

    check_env_vars GPU_COUNT

    set -x
    salloc --partition=$PARTITION --gres=gpu:$GPU_COUNT --exclusive --cpus-per-task=128 --time=500 --no-shell --job-name="$RUNNER_NAME"
    JOB_ID=$(squeue --name="$RUNNER_NAME" -h -o %A | head -n1)

    srun --jobid=$JOB_ID bash -c "docker stop \$(docker ps -a -q)"

    # Concurrent jobs import to the same squash file; serialize them.
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

    export VLLM_CACHE_ROOT="/it-share/gharunners/.cache/vllm"

    if [[ "$FRAMEWORK" == "atom" ]] || [[ "$FRAMEWORK" == "sglang" ]]; then
        SLRUM_HOME_MOUNT=""
    else
        SLRUM_HOME_MOUNT=" --container-mount-home "
    fi

    # Avoid a stale saved copy of this checkpoint; read the shared HF cache.
    if [[ ("$FRAMEWORK" == "vllm" || "$FRAMEWORK" == "atom") ]] && [[ "$MODEL" == "deepseek-ai/DeepSeek-V4-Pro" || "$MODEL" == "deepseek-ai/DeepSeek-V4-Pro-0813" ]]; then
        export HF_HUB_CACHE_MOUNT="/it-share/hf-hub-cache/"
    fi

    # MiniMax-M3 weights are pre-downloaded to the NFS share, not the node-local
    # /var/lib NVMe cache.
    if [[ "$MODEL" == MiniMaxAI/MiniMax-M3* || "$MODEL" == amd/MiniMax-M3* ]]; then
        export HF_HUB_CACHE_MOUNT="/it-share/hf-hub-cache/"
    fi

    # GLM-5.2-FP8 is ~756 GB (141 shards). Pull it once to the NFS share rather
    # than once per node-local NVMe cache, so every cell of the sweep (which
    # may land on different nodes) shares a single staged copy.
    if [[ "$MODEL" == "zai-org/GLM-5.2-FP8" ]]; then
        export HF_HUB_CACHE_MOUNT="/it-share/hf-hub-cache/"
    fi

    # DSv4.1 weights live on the persistent shared cache. Mount this recipe
    # outside /workspace so runtime setup does not create directories there.
    CONTAINER_REPO=/workspace
    if [[ "$MODEL" == "deepseek-ai/DeepSeek-V4.1-Flash" ]]; then
        export HF_HUB_CACHE_MOUNT="/it-share/hf-hub-cache/"
        CONTAINER_REPO=/ix
        export INFMAX_CONTAINER_WORKSPACE="$CONTAINER_REPO"
        case "${RESULT_DIR:-}" in
            /workspace/*) export RESULT_DIR="/ix/${RESULT_DIR#/workspace/}" ;;
        esac
    fi

    SCRIPT_BASE="${EXP_NAME%%_*}_${PRECISION}_mi355x"
    check_env_vars SCENARIO_SUBDIR
    SCRIPT_FW="benchmarks/single_node/${SCENARIO_SUBDIR}${SCRIPT_BASE}_${FRAMEWORK}${SPEC_SUFFIX}.sh"
    check_env_vars SCENARIO_SUBDIR
    SCRIPT_FALLBACK="benchmarks/single_node/${SCENARIO_SUBDIR}${SCRIPT_BASE}${FRAMEWORK_SUFFIX}${SPEC_SUFFIX}.sh"
    if [[ -f "$SCRIPT_FW" ]]; then
        BENCHMARK_SCRIPT="$SCRIPT_FW"
    else
        BENCHMARK_SCRIPT="$SCRIPT_FALLBACK"
    fi

    if [[ "$BENCHMARK_SCRIPT" == "benchmarks/single_node/agentic/minimaxm3_fp4_mi355x_atom_mtp.sh" ]]; then
        export MODEL_PATH="$MODEL"
        export ENABLE_PREFIX_CACHING=true
        export AITER_LOG_LEVEL=WARNING
        export EVAL_TASKS_DIR=infx/evals/gsm8k.yaml
    fi

    srun --jobid=$JOB_ID \
        --container-image=$SQUASH_FILE \
        --container-mounts=$GITHUB_WORKSPACE:$CONTAINER_REPO/,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE,$AIPERF_MMAP_CACHE_HOST_PATH:/aiperf_mmap_cache \
        $SLRUM_HOME_MOUNT \
        --container-writable \
        --container-workdir=$CONTAINER_REPO/ \
        --container-remap-root \
        --no-container-entrypoint --export=ALL,AIPERF_DATASET_MMAP_CACHE_DIR=/aiperf_mmap_cache \
        bash "$BENCHMARK_SCRIPT"
    benchmark_rc=$?

    scancel $JOB_ID

    if ls gpucore.* 1> /dev/null 2>&1; then
        echo "gpucore files exist. not good"
        rm -f gpucore.*
    fi

    exit "$benchmark_rc"
fi
