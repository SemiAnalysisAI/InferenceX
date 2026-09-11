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
    export HF_HUB_CACHE_MOUNT=/raid/inferencex/models/hub
    export SRT_MODEL_PATH="hf:$MODEL"
    export SALLOC_TIME_LIMIT=180
    export SRT_SRUN_OPTIONS='{"container-remap-root":"", "container-writable":""}'
    SRT_SQUASH_FILE="/raid/inferencex/squash/$(printf '%s' "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
    launch_srt_single_node mi300x-amd --var GITHUB_WORKSPACE "$GITHUB_WORKSPACE"
    exit $?
fi

# Native SRT owns the multi-node allocation, workers, router, and eval lifecycle.
if [[ "$EXECUTION_PATH" == multinode ]]; then
    check_env_vars CONFIG_FILE GITHUB_WORKSPACE HOME MODEL IMAGE FRAMEWORK RESULT_FILENAME \
        EVAL_ONLY RUN_EVAL
    source "$(dirname "${BASH_SOURCE[0]}")/slurm_utils.sh"
    if [[ "$RUN_EVAL" == true && "$EVAL_ONLY" != true ]]; then
        echo "ERROR: MI300X disaggregation requires a separate eval-only job" >&2
        exit 1
    fi

    # Containers may run as root. Keep their writable source and outputs outside
    # the Actions checkout, including on cancellation, and retain them for debugging.
    SRT_SHARED_BASE="$HOME/inferencex/srt-slurm"
    mkdir -p "$SRT_SHARED_BASE"
    SRT_RUN_ROOT=$(mktemp -d "$SRT_SHARED_BASE/mi300x.XXXXXX")
    export INFMAX_WORKSPACE="$SRT_RUN_ROOT/workspace"
    mkdir -p "$INFMAX_WORKSPACE"
    rsync -a --exclude=.git --exclude=.venv --exclude=utils/srt-slurm \
        "$GITHUB_WORKSPACE/" "$INFMAX_WORKSPACE/"
    SRTCTL_ROOT="$SRT_RUN_ROOT/srt-slurm"
    setup_srt_slurm "$SRTCTL_ROOT" "$FRAMEWORK" 0
    uv venv .venv
    source .venv/bin/activate
    uv pip install -e .
    export PYTHONPATH="$GITHUB_WORKSPACE${PYTHONPATH:+:$PYTHONPATH}"

    SRT_CONTAINER="$IMAGE"
    SRT_CACHED_IMAGE="$SRT_SHARED_BASE/containers/$(printf '%s' "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
    if [[ -r "$SRT_CACHED_IMAGE" ]] && unsquashfs -s "$SRT_CACHED_IMAGE" >/dev/null 2>&1; then
        SRT_CONTAINER="$SRT_CACHED_IMAGE"
    fi
    python3 -m infx.srt_slurm.cluster_config \
        "$INFERENCEX_SLURM_UTILS_DIR/srt-slurm/mi300x-amd.yaml" srtslurm.yaml \
        --var SRTCTL_ROOT "$SRTCTL_ROOT" --var SRT_DEFAULT_TIME_LIMIT 24:00:00 \
        --var GITHUB_WORKSPACE "$INFMAX_WORKSPACE" \
        --model "hf:$MODEL" "hf:$MODEL" --container "$IMAGE" "$SRT_CONTAINER" \
        --mount /dev/infiniband /dev/infiniband \
        --mount /raid/inferencex/models/hub /hf-cache/hub \
        --mount /raid/inferencex/aiperf-mmap-cache /aiperf_mmap_cache --exclusive
    run_srt_setup ARCH=x86_64

    SRT_EVAL_OVERRIDES=()
    if [[ "$EVAL_ONLY" == true ]]; then
        SRT_EVAL_OVERRIDES=(--unset roles.prefill.args.ep-dispatch-algorithm
            --unset roles.decode.args.ep-dispatch-algorithm)
    fi
    SRT_SUBMISSION_FILE="$SRT_RUN_ROOT/submission.json"
    SRT_JOB_ID=""
    SRT_JOB_OUTPUT=""
    finish_mi300x_multinode() {
        local rc=$?
        trap - EXIT INT TERM
        if [[ -z "$SRT_JOB_ID" ]] && python3 -m infx.srt_slurm.single_node submission \
            "$SRT_SUBMISSION_FILE" > "$SRT_RUN_ROOT/submission-fields" 2>/dev/null; then
            mapfile -t SRT_SUBMISSION < "$SRT_RUN_ROOT/submission-fields"
            SRT_JOB_ID="${SRT_SUBMISSION[0]}"
            SRT_JOB_OUTPUT="${SRT_SUBMISSION[1]}"
        fi
        if [[ -n "$SRT_JOB_ID" ]] && slurm_job_is_active "$SRT_JOB_ID"; then
            scancel "$SRT_JOB_ID" || true
        fi
        if [[ -n "$SRT_JOB_OUTPUT" && -d "$SRT_JOB_OUTPUT/logs" ]]; then
            tar czf "$GITHUB_WORKSPACE/multinode_server_logs.tar.gz" \
                -C "$SRT_JOB_OUTPUT" . || rc=1
            mkdir -p "$GITHUB_WORKSPACE/LOGS"
            cp -r "$SRT_JOB_OUTPUT/logs/." "$GITHUB_WORKSPACE/LOGS/" || rc=1
            if [[ "$EVAL_ONLY" != true ]]; then
                copy_agentic_results "$SRT_JOB_OUTPUT/logs" "$GITHUB_WORKSPACE" "$RESULT_FILENAME" || rc=1
            else
                test -s "$SRT_JOB_OUTPUT/logs/eval_results/meta_env.json" || rc=1
                copy_eval_artifacts "$SRT_JOB_OUTPUT/logs/eval_results" "$GITHUB_WORKSPACE" || rc=1
            fi
        elif [[ "$rc" == 0 ]]; then
            echo "ERROR: SRT completed without a job log directory" >&2
            rc=1
        fi
        echo "SRT runtime and diagnostics retained at $SRT_RUN_ROOT"
        exit "$rc"
    }
    trap finish_mi300x_multinode EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM

    apply_srt_recipe "$CONFIG_FILE" "$FRAMEWORK" "${SRTCTL_EVAL_ARGS[@]}" \
        "${SRT_EVAL_OVERRIDES[@]}" --json --yes --output "$SRT_RUN_ROOT/outputs" \
        > "$SRT_SUBMISSION_FILE"
    python3 -m infx.srt_slurm.single_node submission "$SRT_SUBMISSION_FILE" \
        > "$SRT_RUN_ROOT/submission-fields"
    mapfile -t SRT_SUBMISSION < "$SRT_RUN_ROOT/submission-fields"
    SRT_JOB_ID="${SRT_SUBMISSION[0]}"
    SRT_JOB_OUTPUT="${SRT_SUBMISSION[1]}"
    echo "SRT_SLURM_JOB_ID=$SRT_JOB_ID"
    stream_slurm_job_log "$SRT_JOB_ID" "$SRT_JOB_OUTPUT/logs/sweep_${SRT_JOB_ID}.log"
    verify_slurm_job_status "$SRT_JOB_ID"
    exit 0
fi

export HF_HUB_CACHE_MOUNT="/raid/inferencex/models/hub"
export AIPERF_MMAP_CACHE_MOUNT="/raid/inferencex/aiperf-mmap-cache"
export AIPERF_DATASET_MMAP_CACHE_DIR="/aiperf_mmap_cache"

PARTITION="MI300X-UBUNTU"
SQUASH_FILE="/raid/inferencex/squash/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
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

# A cold 511 GB checkpoint download plus a one-hour AgentX arm does not fit the
# 180-minute default allocation; give DSv4.1 Flash the MI325X launcher's 480.
SALLOC_TIME=180
[[ "$CONTAINER_REPO" == /ix ]] && SALLOC_TIME=480

# Pyxis/enroot could not start the container on the allocated node, for example
# enroot-nsenter denied a user namespace after the node was reprovisioned with
# kernel.apparmor_restrict_unprivileged_userns=1 (seen on smci300x-ccs-aus-e06-40
# and e07-22 in run 35305037528). Drain that node so later jobs avoid it, then
# retry on a fresh allocation that excludes every node that failed so far, up
# to six allocations: run 35307258006 hit the same denial on e06-40, e06-01 and
# e07-22 in turn, and run 35307840070 then drew e07-31, whose enroot import
# could not resolve the registry (curl exit 28), before e06-10 served the eval.
# A node that cannot import the image is excluded the same way. Draining needs Slurm operator rights (the runner account gets
# "Invalid user id"); without them the exclusion list still steers the retries.
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

BENCH_SCRIPT="benchmarks/single_node/${SCENARIO_SUBDIR}${EXP_NAME%%_*}_${PRECISION}_mi300x${SPEC_SUFFIX}.sh"
EXCLUDE_NODES=""
JOB_ID=""
trap 'scancel "$JOB_ID" 2>/dev/null || true' EXIT
MAX_ATTEMPTS=6
for (( attempt = 1; attempt <= MAX_ATTEMPTS; attempt++ )); do
    SALLOC_EXCLUDE_ARGS=()
    if [[ -n "$EXCLUDE_NODES" ]]; then
        SALLOC_EXCLUDE_ARGS=(--exclude="$EXCLUDE_NODES")
    fi
    JOB_ID=$(set +o pipefail; salloc \
        --partition="$PARTITION" \
        --gres="gpu:$GPU_COUNT" \
        --cpus-per-task=128 \
        --time="$SALLOC_TIME" \
        --no-shell \
        "${SALLOC_EXCLUDE_ARGS[@]}" \
        --job-name="$RUNNER_NAME" 2>&1 \
        | tee /dev/stderr \
        | grep -oP 'Granted job allocation \K[0-9]+')

    if [[ -z "$JOB_ID" ]]; then
        echo "ERROR: salloc failed to allocate a job" >&2
        exit 1
    fi

    export PORT=$((40000 + (JOB_ID % 10000)))
    NODE=$(scontrol show job "$JOB_ID" -o | grep -oP ' NodeList=\K\S+' || true)

    # Concurrent jobs import to the same node-local squash file; serialize them.
    set +e
    srun --jobid="$JOB_ID" --job-name="$RUNNER_NAME" bash -c "
        set -eo pipefail
        exec 9>\"$LOCK_FILE\"
        flock -w 600 9 || { echo 'Failed to acquire lock for $SQUASH_FILE' >&2; exit 1; }
        if unsquashfs -l \"$SQUASH_FILE\" >/dev/null 2>&1; then
            echo 'Squash file already exists and is valid, skipping import'
        else
            rm -f \"$SQUASH_FILE\"
            enroot import -o \"$SQUASH_FILE\" docker://$IMAGE
        fi
    "
    import_rc=$?
    set -e
    if (( import_rc != 0 )); then
        if (( attempt < MAX_ATTEMPTS )) && [[ -n "$NODE" ]]; then
            echo "WARNING: image import failed on $NODE (exit $import_rc); excluding it and retrying" >&2
            scancel "$JOB_ID"
            JOB_ID=""
            EXCLUDE_NODES="${EXCLUDE_NODES:+$EXCLUDE_NODES,}$NODE"
            continue
        fi
        exit "$import_rc"
    fi

    SRUN_STDERR="$(mktemp)"
    set +e
    srun --jobid="$JOB_ID" \
        --job-name="$RUNNER_NAME" \
        --container-image="$SQUASH_FILE" \
        --container-mounts="$GITHUB_WORKSPACE:$CONTAINER_REPO/,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE,$AIPERF_MMAP_CACHE_MOUNT:/aiperf_mmap_cache,/dev/kfd:/dev/kfd,/dev/dri:/dev/dri" \
        --container-writable \
        --container-remap-root \
        --container-workdir="$CONTAINER_REPO/" \
        --no-container-entrypoint \
        --export=ALL \
        bash "$BENCH_SCRIPT" 2> >(tee "$SRUN_STDERR" >&2)
    benchmark_rc=$?
    set -e
    if (( benchmark_rc == 0 )); then
        break
    fi
    if (( attempt < MAX_ATTEMPTS )) && [[ -n "$NODE" ]] && container_start_failed "$SRUN_STDERR"; then
        drain_broken_node "$NODE" "inferencex: pyxis container start failed (enroot user namespace)"
        scancel "$JOB_ID"
        JOB_ID=""
        EXCLUDE_NODES="${EXCLUDE_NODES:+$EXCLUDE_NODES,}$NODE"
        echo "Retrying on a fresh allocation that excludes $EXCLUDE_NODES" >&2
        continue
    fi
    exit "$benchmark_rc"
done

scancel "$JOB_ID"
