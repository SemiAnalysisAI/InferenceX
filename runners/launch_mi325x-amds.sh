#!/usr/bin/env bash

source "$(dirname "${BASH_SOURCE[0]}")/../benchmarks/benchmark_lib.sh" --validation-only || exit 1
check_env_vars IS_MULTINODE IS_AGENTIC
set -eo pipefail

# Select native fixed-sequence execution before the retained AgentX/multi-node paths.
EXECUTION_PATH=agentic
if [[ "$IS_MULTINODE" == true ]]; then
    EXECUTION_PATH=multinode
elif [[ "$IS_AGENTIC" == 0 || -n "${SRT_RECIPE:-}" ]]; then
    check_env_vars SRT_RECIPE
    EXECUTION_PATH=native-single-node
fi
if [[ "$EXECUTION_PATH" == native-single-node ]]; then
    check_env_vars GITHUB_WORKSPACE MODEL IMAGE
    source "$(dirname "${BASH_SOURCE[0]}")/slurm_utils.sh" || exit 1
    export HF_HUB_CACHE_MOUNT=/raid/hf-hub-cache/
    export SRT_MODEL_PATH="hf:$MODEL"
    export SALLOC_TIME_LIMIT=480
    export SRT_SRUN_OPTIONS='{"container-remap-root":"", "container-writable":""}'
    SRT_SQUASH_FILE="/raid/squash/$(printf '%s' "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
    launch_srt_single_node mi325x-amds
    exit $?
fi

export HF_HUB_CACHE_MOUNT="/raid/hf-hub-cache/"

PARTITION="compute"
SQUASH_FILE="/raid/squash/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
LOCK_FILE="${SQUASH_FILE}.lock"

SPEC_SUFFIX=$([[ "${SPEC_DECODING:-}" == "mtp" ]] && printf '_mtp' || printf '')

# DSv4.1 Flash AgentX creates runtime directories next to the repository, which
# must not land under /workspace; mount the checkout at /ix like the other
# dsv41flash launchers and rewrite the caller's RESULT_DIR to match.
CONTAINER_REPO=/workspace
if [[ "$MODEL" == "deepseek-ai/DeepSeek-V4.1-Flash" ]]; then
    CONTAINER_REPO=/ix
    export INFMAX_CONTAINER_WORKSPACE="$CONTAINER_REPO"
    case "${RESULT_DIR:-}" in
        /workspace/*) export RESULT_DIR="/ix/${RESULT_DIR#/workspace/}" ;;
    esac
fi

# Pyxis/enroot could not start the container on the allocated node, for example
# enroot-nsenter denied a user namespace after the node was reprovisioned with
# kernel.apparmor_restrict_unprivileged_userns=1 (seen on the MI300X fleet in
# run 35305037528). Drain that node so later jobs avoid it, then retry once on
# a fresh allocation that excludes it. Draining needs Slurm operator rights;
# without them the exclusion still protects the retry.
container_start_failed() {
    grep -qE "pyxis: (couldn't start container|container start failed)|enroot-nsenter: failed to create user namespace" "$1"
}
drain_broken_node() {
    local node="$1" reason="$2"
    if scontrol update NodeName="$node" State=DRAIN Reason="$reason"; then
        echo "Drained $node: $reason" >&2
    else
        echo "WARNING: could not drain $node (Slurm operator rights required); excluding it for the retry only" >&2
    fi
}

check_env_vars GPU_COUNT

set -x

BENCH_SCRIPT="benchmarks/single_node/${SCENARIO_SUBDIR}${EXP_NAME%%_*}_${PRECISION}_mi325x${SPEC_SUFFIX}.sh"
EXCLUDE_NODES=""
JOB_ID=""
trap 'rc=$?; scancel "$JOB_ID" 2>/dev/null || true; exit "$rc"' EXIT
for attempt in 1 2; do
    SALLOC_EXCLUDE_ARGS=()
    if [[ -n "$EXCLUDE_NODES" ]]; then
        SALLOC_EXCLUDE_ARGS=(--exclude="$EXCLUDE_NODES")
    fi
    JOB_ID=$(set +o pipefail; salloc --partition=$PARTITION --gres=gpu:$GPU_COUNT --cpus-per-task=256 --time=480 --no-shell "${SALLOC_EXCLUDE_ARGS[@]}" --job-name="$RUNNER_NAME" 2>&1 | tee /dev/stderr | grep -oP 'Granted job allocation \K[0-9]+')

    if [ -z "$JOB_ID" ]; then
        echo "ERROR: salloc failed to allocate a job" >&2
        exit 1
    fi

    export PORT=$(( 40000 + (JOB_ID % 10000) ))
    export XDG_CACHE_HOME="/tmp/xdg-cache-$JOB_ID"
    export TRITON_CACHE_DIR="/tmp/triton-cache-$JOB_ID"
    NODE=$(scontrol show job "$JOB_ID" -o | grep -oP ' NodeList=\K\S+' || true)

    # Concurrent jobs import to the same squash file; serialize them.
    srun --jobid="$JOB_ID" --job-name="$RUNNER_NAME" bash -c "
        set -eo pipefail
        exec 9>\"$LOCK_FILE\"
        flock -w 600 9 || { echo 'Failed to acquire lock for $SQUASH_FILE' >&2; exit 1; }
        if unsquashfs -l \"$SQUASH_FILE\" > /dev/null 2>&1; then
            echo 'Squash file already exists and is valid, skipping import'
        else
            rm -f \"$SQUASH_FILE\"
            enroot import -o \"$SQUASH_FILE\" docker://$IMAGE
        fi
    "
    SRUN_STDERR="$(mktemp)"
    set +e
    srun --jobid="$JOB_ID" \
    --container-image="$SQUASH_FILE" \
    --container-mounts="$GITHUB_WORKSPACE:$CONTAINER_REPO/,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE,/dev/kfd:/dev/kfd,/dev/dri:/dev/dri" \
    --container-mount-home \
    --container-writable \
    --container-remap-root \
    --container-workdir="$CONTAINER_REPO/" \
    --no-container-entrypoint --export=ALL \
    bash "$BENCH_SCRIPT" 2> >(tee "$SRUN_STDERR" >&2)
    benchmark_rc=$?
    set -e
    if (( benchmark_rc == 0 )); then
        break
    fi
    if (( attempt == 1 )) && [[ -n "$NODE" ]] && container_start_failed "$SRUN_STDERR"; then
        drain_broken_node "$NODE" "inferencex: pyxis container start failed (enroot user namespace)"
        scancel "$JOB_ID"
        JOB_ID=""
        EXCLUDE_NODES="$NODE"
        echo "Retrying on a fresh allocation that excludes $NODE" >&2
        continue
    fi
    exit "$benchmark_rc"
done

scancel $JOB_ID
