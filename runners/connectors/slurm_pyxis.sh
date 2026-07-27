#!/usr/bin/env bash
set -euo pipefail

# Reusable connector for a long-running multi-node service and an overlapping
# benchmark client on Slurm clusters using Pyxis/Enroot.
#
# Deployment inputs:
#   MULTINODE_GPU_COUNT
#   MULTINODE_SERVER_SCRIPT
#   MULTINODE_CLIENT_SCRIPT
#
# Runner profile inputs:
#   SLURM_PARTITION
#   SLURM_ACCOUNT (optional)
#   SLURM_PYXIS_GPUS_PER_NODE
#   SLURM_PYXIS_IMAGE_CACHE_DIR
#   SLURM_PYXIS_CONTAINER_MOUNTS
#   SLURM_PYXIS_CONTAINER_WORKDIR
#   MODEL_HOST_PATH
#   MODEL_PATH
#
# The server receives a scheduler-independent contract:
#   MULTINODE_NODE_COUNT
#   MULTINODE_GPUS_PER_NODE
#   MULTINODE_NODE_RANK
#   MULTINODE_MASTER_ADDR

_slurm_pyxis_require_vars() {
    local variable_name

    for variable_name in "$@"; do
        if [[ -z "${!variable_name:-}" ]]; then
            echo "Missing required Slurm/Pyxis connector variable: $variable_name" >&2
            return 1
        fi
    done
}

_slurm_pyxis_is_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

_slurm_pyxis_package_logs() {
    if [[ "${_SLURM_PYXIS_LOGS_PACKAGED:-0}" == "1" ]]; then
        return
    fi
    _SLURM_PYXIS_LOGS_PACKAGED=1

    if [[ -n "${_SLURM_PYXIS_SERVER_LOG_DIR:-}" && \
          -d "$_SLURM_PYXIS_SERVER_LOG_DIR" ]]; then
        tar czf "$GITHUB_WORKSPACE/multinode_server_logs.tar.gz" \
            -C "$_SLURM_PYXIS_SERVER_LOG_DIR" . || true
    fi
}

_slurm_pyxis_cleanup() {
    if [[ -n "${_SLURM_PYXIS_SERVER_STEP_PID:-}" ]]; then
        kill "$_SLURM_PYXIS_SERVER_STEP_PID" 2>/dev/null || true
        wait "$_SLURM_PYXIS_SERVER_STEP_PID" 2>/dev/null || true
        _SLURM_PYXIS_SERVER_STEP_PID=""
    fi

    _slurm_pyxis_package_logs

    if [[ -n "${_SLURM_PYXIS_JOB_ID:-}" ]]; then
        scancel "$_SLURM_PYXIS_JOB_ID" 2>/dev/null || true
        _SLURM_PYXIS_JOB_ID=""
    fi
}

_slurm_pyxis_exit_trap() {
    local exit_code="$1"

    trap - EXIT INT TERM
    _slurm_pyxis_cleanup
    exit "$exit_code"
}

run_slurm_pyxis_multinode_service() {
    local node_count
    local squash_file
    local lock_file
    local docker_image
    local allocated_nodelist
    local head_node
    local startup_timeout
    local poll_interval
    local max_attempts
    local attempt
    local ready=0
    local client_rc
    local slurm_gres
    local -a allocation_args

    _slurm_pyxis_require_vars \
        IMAGE \
        RUNNER_NAME \
        GITHUB_WORKSPACE \
        PORT \
        MULTINODE_GPU_COUNT \
        MULTINODE_SERVER_SCRIPT \
        MULTINODE_CLIENT_SCRIPT \
        SLURM_PARTITION \
        SLURM_PYXIS_GPUS_PER_NODE \
        SLURM_PYXIS_IMAGE_CACHE_DIR \
        SLURM_PYXIS_CONTAINER_MOUNTS \
        SLURM_PYXIS_CONTAINER_WORKDIR \
        MODEL_HOST_PATH \
        MODEL_PATH || return 1

    if ! _slurm_pyxis_is_positive_integer "$MULTINODE_GPU_COUNT"; then
        echo "MULTINODE_GPU_COUNT must be a positive integer" >&2
        return 1
    fi
    if ! _slurm_pyxis_is_positive_integer "$SLURM_PYXIS_GPUS_PER_NODE"; then
        echo "SLURM_PYXIS_GPUS_PER_NODE must be a positive integer" >&2
        return 1
    fi
    if (( MULTINODE_GPU_COUNT % SLURM_PYXIS_GPUS_PER_NODE != 0 )); then
        echo "MULTINODE_GPU_COUNT ($MULTINODE_GPU_COUNT) must be divisible by the runner's GPUs per node ($SLURM_PYXIS_GPUS_PER_NODE)" >&2
        return 1
    fi

    node_count=$((MULTINODE_GPU_COUNT / SLURM_PYXIS_GPUS_PER_NODE))
    startup_timeout="${MULTINODE_STARTUP_TIMEOUT_SECONDS:-3600}"
    poll_interval="${MULTINODE_HEALTH_POLL_SECONDS:-10}"
    if ! _slurm_pyxis_is_positive_integer "$startup_timeout" || \
       ! _slurm_pyxis_is_positive_integer "$poll_interval"; then
        echo "Multi-node startup timeout and poll interval must be positive integers" >&2
        return 1
    fi

    if [[ "$MULTINODE_SERVER_SCRIPT" == /* || \
          ! -f "$GITHUB_WORKSPACE/$MULTINODE_SERVER_SCRIPT" ]]; then
        echo "MULTINODE_SERVER_SCRIPT must be a repository-relative file: $MULTINODE_SERVER_SCRIPT" >&2
        return 1
    fi
    if [[ "$MULTINODE_CLIENT_SCRIPT" == /* || \
          ! -f "$GITHUB_WORKSPACE/$MULTINODE_CLIENT_SCRIPT" ]]; then
        echo "MULTINODE_CLIENT_SCRIPT must be a repository-relative file: $MULTINODE_CLIENT_SCRIPT" >&2
        return 1
    fi

    echo "Slurm/Pyxis deployment: ${MULTINODE_GPU_COUNT} GPUs as ${node_count} nodes x ${SLURM_PYXIS_GPUS_PER_NODE} GPUs"
    if [[ "${MULTINODE_CONNECTOR_DRY_RUN:-0}" == "1" ]]; then
        echo "Server entrypoint: $MULTINODE_SERVER_SCRIPT"
        echo "Client entrypoint: $MULTINODE_CLIENT_SCRIPT"
        return
    fi

    if [[ ! -d "$MODEL_HOST_PATH" ]]; then
        echo "Pre-staged model path not found: $MODEL_HOST_PATH" >&2
        return 1
    fi

    _SLURM_PYXIS_SERVER_LOG_DIR="${MULTINODE_SERVER_LOG_DIR:-$GITHUB_WORKSPACE/multinode_server_logs}"
    _SLURM_PYXIS_LOGS_PACKAGED=0
    _SLURM_PYXIS_JOB_ID=""
    _SLURM_PYXIS_SERVER_STEP_PID=""

    mkdir -p "$SLURM_PYXIS_IMAGE_CACHE_DIR" "$_SLURM_PYXIS_SERVER_LOG_DIR"
    squash_file="$SLURM_PYXIS_IMAGE_CACHE_DIR/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
    lock_file="${squash_file}.lock"
    docker_image="${IMAGE//#//}"
    slurm_gres="${SLURM_PYXIS_GRES:-gpu:$SLURM_PYXIS_GPUS_PER_NODE}"

    allocation_args=(
        "--partition=$SLURM_PARTITION"
        "--nodes=$node_count"
        "--ntasks-per-node=1"
        "--gres=$slurm_gres"
        --exclusive
        "--time=${MULTINODE_TIME_LIMIT_MINUTES:-240}"
        --no-shell
        "--job-name=$RUNNER_NAME"
    )
    if [[ -n "${SLURM_ACCOUNT:-}" ]]; then
        allocation_args+=("--account=$SLURM_ACCOUNT")
    fi

    salloc "${allocation_args[@]}"
    _SLURM_PYXIS_JOB_ID=$(squeue --name="$RUNNER_NAME" -u "$USER" -h -o %A | sed -n '1p')
    if [[ -z "$_SLURM_PYXIS_JOB_ID" ]]; then
        echo "Failed to resolve the Slurm/Pyxis allocation" >&2
        scancel --name="$RUNNER_NAME" 2>/dev/null || true
        return 1
    fi

    trap '_slurm_pyxis_exit_trap $?' EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM

    allocated_nodelist=$(squeue -j "$_SLURM_PYXIS_JOB_ID" -h -o %N)
    head_node=$(scontrol show hostnames "$allocated_nodelist" | sed -n '1p')
    if [[ -z "$head_node" ]]; then
        echo "Failed to resolve the rank-zero node" >&2
        return 1
    fi

    # Import once into the runner profile's shared cache. The image lock keeps
    # concurrent jobs from replacing the same squash file.
    # shellcheck disable=SC2016 # The worker-side shell expands positional args.
    srun \
        --jobid="$_SLURM_PYXIS_JOB_ID" \
        --nodes=1 \
        --ntasks=1 \
        --nodelist="$head_node" \
        bash -c '
            set -e
            image_cache_dir=$1
            squash_file=$2
            lock_file=$3
            docker_image=$4
            mkdir -p "$image_cache_dir" "$HOME/.cache/enroot"
            export ENROOT_CACHE_PATH="$HOME/.cache/enroot"
            exec 9>"$lock_file"
            flock -w "${SLURM_PYXIS_IMPORT_LOCK_TIMEOUT_SECONDS:-1800}" 9
            if unsquashfs -l "$squash_file" >/dev/null 2>&1; then
                echo "Using cached image: $squash_file"
            else
                rm -f "$squash_file"
                enroot import -o "$squash_file" "docker://$docker_image"
                unsquashfs -l "$squash_file" >/dev/null
                chmod a+r "$squash_file" || true
            fi
        ' bash "$SLURM_PYXIS_IMAGE_CACHE_DIR" "$squash_file" "$lock_file" "$docker_image"

    export MULTINODE_NODE_COUNT="$node_count"
    export MULTINODE_GPUS_PER_NODE="$SLURM_PYXIS_GPUS_PER_NODE"
    export MULTINODE_MASTER_ADDR="$head_node"

    # The connector translates SLURM_PROCID into the scheduler-independent
    # node-rank contract before invoking the in-container server entrypoint.
    # shellcheck disable=SC2016 # SLURM_PROCID is created inside each srun task.
    srun \
        --jobid="$_SLURM_PYXIS_JOB_ID" \
        --nodes="$node_count" \
        --ntasks="$node_count" \
        --ntasks-per-node=1 \
        --kill-on-bad-exit=1 \
        --container-image="$squash_file" \
        --container-mounts="$SLURM_PYXIS_CONTAINER_MOUNTS" \
        --no-container-mount-home \
        --container-remap-root \
        --container-workdir="$SLURM_PYXIS_CONTAINER_WORKDIR" \
        --no-container-entrypoint \
        --export=ALL \
        bash -c 'export MULTINODE_NODE_RANK="$SLURM_PROCID"; exec bash "$MULTINODE_SERVER_SCRIPT"' \
        >"$_SLURM_PYXIS_SERVER_LOG_DIR/combined.log" 2>&1 &
    _SLURM_PYXIS_SERVER_STEP_PID=$!

    max_attempts=$(((startup_timeout + poll_interval - 1) / poll_interval))
    for ((attempt = 1; attempt <= max_attempts; attempt++)); do
        if curl --output /dev/null --silent --fail \
            "http://$head_node:$PORT${MULTINODE_HEALTH_PATH:-/health}"; then
            ready=1
            break
        fi
        if ! kill -0 "$_SLURM_PYXIS_SERVER_STEP_PID" 2>/dev/null; then
            echo "Multi-node server step exited before becoming ready" >&2
            tail -n 200 "$_SLURM_PYXIS_SERVER_LOG_DIR/combined.log" >&2 || true
            return 1
        fi
        if (( attempt * poll_interval % 60 == 0 )); then
            echo "Waiting for multi-node server readiness ($((attempt * poll_interval))s elapsed)"
            tail -n 20 "$_SLURM_PYXIS_SERVER_LOG_DIR/combined.log" || true
        fi
        sleep "$poll_interval"
    done
    if [[ "$ready" != "1" ]]; then
        echo "Multi-node server did not become healthy within $startup_timeout seconds" >&2
        tail -n 200 "$_SLURM_PYXIS_SERVER_LOG_DIR/combined.log" >&2 || true
        return 1
    fi

    set +e
    srun \
        --jobid="$_SLURM_PYXIS_JOB_ID" \
        --overlap \
        --nodes=1 \
        --ntasks=1 \
        --nodelist="$head_node" \
        --container-image="$squash_file" \
        --container-mounts="$SLURM_PYXIS_CONTAINER_MOUNTS" \
        --no-container-mount-home \
        --container-remap-root \
        --container-workdir="$SLURM_PYXIS_CONTAINER_WORKDIR" \
        --no-container-entrypoint \
        --export=ALL,MULTINODE_NODE_RANK=0 \
        bash "$MULTINODE_CLIENT_SCRIPT"
    client_rc=$?
    set -e

    _slurm_pyxis_cleanup
    trap - EXIT INT TERM
    return "$client_rc"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    run_slurm_pyxis_multinode_service
fi
