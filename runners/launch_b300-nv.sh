#!/usr/bin/bash

# System-specific configuration for B300 NV Slurm cluster (sa-shared)
SLURM_PARTITION="batch_1"
SLURM_ACCOUNT="benchmark"
# b300-018 repeatedly times out UCX/NIXL transfers; allow an empty override to disable this.
MINIMAX_M3_SLURM_EXCLUDED_NODELIST="${MINIMAX_M3_SLURM_EXCLUDED_NODELIST-b300-018}"

set -x

if [[ "$IS_MULTINODE" == "true" ]]; then

# Multinode cluster profile — declares cluster facts per the contract in
# benchmarks/multi_node/srt_slurm/README.md and hands off to the
# cluster-agnostic orchestrator.

set -eo pipefail

source "$(dirname "$0")/lib/multinode.sh"

export SLURM_PARTITION SLURM_ACCOUNT

INFX_CLUSTER="b300"
INFX_GPUS_PER_NODE=8
INFX_ARCH="x86_64"
INFX_SLURM_TIME_LIMIT="4:00:00"

INFX_SRTSLURM_EXTRA="use_exclusive_sbatch_directive: true
default_mounts:
  \"/opt/ucx-no-ud\": \"/usr/local/ucx\""

# MODEL_PATH / SRT_SLURM_MODEL_PREFIX / SERVED_MODEL_NAME come from the
# model-paths registry in configs/runners.yaml.
infx_resolve_model_paths cluster:b300-nv

NGINX_IMAGE="nginx:1.27.4"
SQUASH_FILE="$(infx_squash_path /data/squash "$IMAGE")"
NGINX_SQUASH_FILE="$(infx_squash_path /data/squash "$NGINX_IMAGE")"

if [[ "${INFX_DRY_RUN:-0}" != "1" ]]; then
    infx_import_squash_srun "$SQUASH_FILE" "$IMAGE" \
        -N 1 -A "$SLURM_ACCOUNT" -p "$SLURM_PARTITION"
    infx_import_squash_srun "$NGINX_SQUASH_FILE" "$NGINX_IMAGE" \
        -N 1 -A "$SLURM_ACCOUNT" -p "$SLURM_PARTITION"
fi

# Keep MiniMax-M3 jobs off nodes with known-bad UCX/NIXL behavior, and
# verify the exclusion actually rendered into the sbatch script.
infx_hook_edit_recipe() {
    local recipe="$1"
    if [[ "$MODEL_PREFIX" == "minimaxm3" && -n "$MINIMAX_M3_SLURM_EXCLUDED_NODELIST" ]]; then
        sed -i "/^name:.*/a sbatch_directives:\n  exclude: \"${MINIMAX_M3_SLURM_EXCLUDED_NODELIST}\"" "$recipe"
    fi
}

infx_hook_post_submit() {
    local job_id="$1"
    if [[ "$MODEL_PREFIX" == "minimaxm3" && -n "$MINIMAX_M3_SLURM_EXCLUDED_NODELIST" ]]; then
        local sbatch_script="outputs/$job_id/sbatch_script.sh"
        if ! grep -Fq "#SBATCH --exclude=${MINIMAX_M3_SLURM_EXCLUDED_NODELIST}" "$sbatch_script"; then
            echo "Error: Slurm node exclusion was not rendered in $sbatch_script" >&2
            scancel "$job_id" || true
            return 1
        fi
    fi
}

source "$GITHUB_WORKSPACE/benchmarks/multi_node/srt_slurm/run.sh"

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
