#!/usr/bin/env bash
set -euo pipefail
set -x
NODES=$(( ${PREFILL_NODES:-1} + ${DECODE_NODES:-1} ))
if [[ "${PREFILL_NODES:-1}" != 1 || "${DECODE_NODES:-1}" != 1 ]]; then
    echo "TileRT native launcher requires one physical node per role" >&2
    exit 1
fi
GPUS_PER_NODE="${GPUS_PER_NODE:-${GPU_COUNT:-$(( ${PREFILL_TP:-8} > ${DECODE_TP:-8} ? ${PREFILL_TP:-8} : ${DECODE_TP:-8} ))}}"

SQUASH_DIR="${B200_SQUASH_DIR:-/home/sa-shared/containers}"
{ mkdir -p "$SQUASH_DIR" 2>/dev/null && [[ -w "$SQUASH_DIR" ]]; } || SQUASH_DIR="$GITHUB_WORKSPACE/.container-squash"
mkdir -p "$SQUASH_DIR"
chmod a+rx "$SQUASH_DIR" || true

DECODE_IMAGE="${DECODE_IMAGE:-$IMAGE}"
: "${PREFILL_IMAGE:?PREFILL_IMAGE is unset -- it must come from prefill.additional-settings in the master config, e.g. PREFILL_IMAGE=vllm/vllm-openai:v0.26.0}"

squash_path() { echo "$SQUASH_DIR/$(echo "$1" | sed 's/[\/:@#]/_/g').sqsh"; }
DECODE_SQUASH="$(squash_path "$DECODE_IMAGE")"
PREFILL_SQUASH="$(squash_path "$PREFILL_IMAGE")"

if [[ "${TILERT_IN_ALLOCATION:-0}" != 1 ]]; then
    # Run inside the allocation returned by this request. Looking up a runner
    # name can attach to an older job; salloc supplies the authoritative ID.
    exec salloc --partition="$SLURM_PARTITION" --account="$SLURM_ACCOUNT" \
        --nodes="$NODES" --gres=gpu:"$GPUS_PER_NODE" --exclusive --mem=0 \
        --time="${SALLOC_TIME_LIMIT:-480}" --job-name="$RUNNER_NAME" \
        env TILERT_IN_ALLOCATION=1 bash "$0" "$@"
fi
JOB_ID="${SLURM_JOB_ID:?salloc did not provide its allocation ID}"
mapfile -t HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST:?salloc did not provide its nodes}")
[[ "${#HOSTS[@]}" -eq 2 ]] || { echo "expected 2 nodes, got: ${HOSTS[*]}"; exit 1; }
export DECODE_HOST="${HOSTS[0]}" PREFILL_HOST="${HOSTS[1]}"
export POWERX_NATIVE_ENABLED=0
if [[ "${REQUIRE_POWER:-0}" =~ ^(1|true|TRUE|yes|YES)$ && "$ISL" == 8192 && "$OSL" == 1024 && "${IS_AGENTIC:-0}" != 1 && "${SCENARIO_TYPE:-}" != agentic-coding && "${EVAL_ONLY:-false}" != true ]]; then
    export POWERX_NATIVE_ENABLED=1
fi
export SLURM_JOB_ID="$JOB_ID"
POWERX_MOUNTS=""
POWERX_ENV="POWERX_NATIVE_ENABLED"
if [[ "$POWERX_NATIVE_ENABLED" == 1 ]]; then
    export POWERX_HOST_UID="$(id -u)" POWERX_HOST_GID="$(id -g)"
    export POWERX_COLLECTOR_REVISION="$(git -C "$GITHUB_WORKSPACE" rev-parse HEAD)"
    export POWERX_RAW_ROOT="${POWERX_RAW_ROOT:-/tmp/inferencex-native-$JOB_ID}"
    export POWERX_CONTROL_ROOT="$GITHUB_WORKSPACE/LOGS/power_control-$JOB_ID"
    mkdir -p "$POWERX_CONTROL_ROOT" "$GITHUB_WORKSPACE/LOGS/native_power"
    chmod 777 "$POWERX_CONTROL_ROOT"
    srun --jobid="$JOB_ID" --nodes="$NODES" --ntasks-per-node=1 mkdir -p "$POWERX_RAW_ROOT"
    srun --jobid="$JOB_ID" --nodes="$NODES" --ntasks-per-node=1 chmod 777 "$POWERX_RAW_ROOT"
    POWERX_MOUNTS=",$POWERX_RAW_ROOT:/powerx_native,$POWERX_CONTROL_ROOT:/powerx_control"
    POWERX_ENV+=",CUDA_VISIBLE_DEVICES,POWERX_HOST_UID,POWERX_HOST_GID,POWERX_COLLECTOR_REVISION,POWERX_NODE_NAME,POWERX_CLOCK_SYNCHRONIZED,POWERX_RANK,POWERX_GPU_COUNT"
fi
rm -f "${BENCHMARK_LOGS_DIR:-$GITHUB_WORKSPACE}/.tilert_done.$JOB_ID"
# Keep node-local receipts inside the allocation until both serving steps drain.
# A caught cancellation exits through the same staging path as normal completion.
wait_owned_step() {
    local pid="$1" deadline=$((SECONDS + ${TILERT_DECODE_DRAIN:-60})) rc=0
    while kill -0 "$pid" 2>/dev/null; do
        if (( SECONDS >= deadline )); then
            echo "[submit] step $pid did not drain before timeout" >&2
            kill -TERM "$pid" 2>/dev/null || true
            sleep 2
            kill -KILL "$pid" 2>/dev/null || true
            rc=1
            break
        fi
        sleep 1
    done
    wait "$pid" || rc=$?
    return "$rc"
}

finish_tilert_submit() {
    local rc=$? step_rc pid role_rank
    trap - EXIT
    trap '' TERM HUP INT
    if [[ "$POWERX_NATIVE_ENABLED" == 1 ]]; then
        printf 'stop\n' > "$POWERX_CONTROL_ROOT/stop" || { [[ "$rc" != 0 ]] || rc=1; }
    fi
    if [[ "$rc" != 0 ]]; then
        for pid in "${PREFILL_SRUN_PID:-}" "${DECODE_SRUN_PID:-}"; do
            [[ -z "$pid" ]] || kill -TERM "$pid" 2>/dev/null || true
        done
    fi
    for pid in "${PREFILL_SRUN_PID:-}" "${DECODE_SRUN_PID:-}"; do
        [[ -n "$pid" ]] || continue
        step_rc=0
        wait_owned_step "$pid" || step_rc=$?
        [[ "$rc" != 0 ]] || rc=$step_rc
    done
    if [[ "$POWERX_NATIVE_ENABLED" == 1 ]]; then
        for role_rank in 0 1; do
            # A failed or missing rank must not suppress the other rank's audit.
            srun --jobid="$JOB_ID" --nodelist="${HOSTS[$role_rank]}" --ntasks=1 \
                bash -c 'cp -R "$POWERX_RAW_ROOT/node-$1" "$GITHUB_WORKSPACE/LOGS/native_power/"' bash "$role_rank" &
            step_rc=0
            wait_owned_step "$!" || step_rc=$?
            [[ "$rc" != 0 ]] || rc=$step_rc
        done
    fi
    exit "$rc"
}
trap finish_tilert_submit EXIT
trap 'exit 143' TERM HUP
trap 'exit 130' INT

import_image() {
    local image_ref="$1" squash_file="$2" host="$3"
    local image_key; image_key=$(echo "$image_ref" | sed 's/[\/:@#]/_/g')
    local lock_file="$SQUASH_DIR/.locks/${image_key}.lock"
    mkdir -p "$SQUASH_DIR/.locks"
    srun --jobid="$JOB_ID" --nodelist="$host" --ntasks=1 bash -c "
        export ENROOT_CACHE_PATH=\$HOME/.cache/enroot; mkdir -p \$ENROOT_CACHE_PATH
        exec 9>\"$lock_file\"; flock -w 600 9 || exit 1
        unsquashfs -l \"$squash_file\" >/dev/null 2>&1 || enroot import -o \"$squash_file\" docker://$image_ref
    "
}
import_image "$DECODE_IMAGE"  "$DECODE_SQUASH"  "$DECODE_HOST"  || exit 1
import_image "$PREFILL_IMAGE" "$PREFILL_SQUASH" "$PREFILL_HOST" || exit 1

export TILERT_WEIGHTS_DIR="${TILERT_WEIGHTS_DIR:-$HOME/.cache/tilert/${MODEL_PREFIX:-model}-${PRECISION:-fp8}-tilert-8shard}"
mkdir -p "$TILERT_WEIGHTS_DIR"

run_role() {
    local role="$1" host="$2" squash_file="$3"
    local rank=0 gpu_count="${DECODE_TP:-8}"
    if [[ "$role" == prefill ]]; then rank=1; gpu_count="${PREFILL_TP:-8}"; fi
    if [[ "$POWERX_NATIVE_ENABLED" == 1 ]]; then
        export POWERX_RANK="$rank" POWERX_GPU_COUNT="$gpu_count" POWERX_NODE_NAME="$host"
        export CUDA_VISIBLE_DEVICES="$(seq -s, 0 "$((gpu_count - 1))")"
        export POWERX_CLOCK_SYNCHRONIZED
        POWERX_CLOCK_SYNCHRONIZED=$(srun --jobid="$JOB_ID" --nodelist="$host" --ntasks=1 \
            bash -c 'timedatectl show -p NTPSynchronized --value 2>/dev/null || echo false')
    fi
    # The tilert image bakes no NVIDIA_VISIBLE_DEVICES (unlike vllm-openai), and
    # enroot's nvidia hook only injects the driver when it is set — without it the
    # decode container has no libcuda and torch dies with "Found no NVIDIA driver".
    # docker --gpus sets this implicitly, which is why the image works elsewhere.
    # Exported here (not in --export) because the capabilities value contains a
    # comma, which srun's --export parsing would split on.
    export NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
    exec srun --jobid="$JOB_ID" --nodelist="$host" --ntasks=1 \
        --container-image="$squash_file" \
        --container-mounts="$GITHUB_WORKSPACE:/workspace,$MODEL_PATH:$MODEL_PATH,$TILERT_WEIGHTS_DIR:$TILERT_WEIGHTS_DIR$POWERX_MOUNTS" \
        --container-workdir=/workspace --no-container-entrypoint \
        --container-env="$POWERX_ENV" \
        --export=ALL,TILERT_ROLE="$role",DECODE_HOST="$DECODE_HOST",PREFILL_HOST="$PREFILL_HOST",PORT="${PORT:-8888}" \
        bash "/workspace/benchmarks/multi_node/tilert_utils/run_node.sh"
}

run_role decode "$DECODE_HOST" "$DECODE_SQUASH" &
DECODE_SRUN_PID=$!

# Both roles run as owned children so a signal interrupts the shell's wait and
# reaches EXIT cleanup immediately; run_role execs srun to preserve that PID.
run_role prefill "$PREFILL_HOST" "$PREFILL_SQUASH" &
PREFILL_SRUN_PID=$!
PREFILL_RC=0
wait "$PREFILL_SRUN_PID" || PREFILL_RC=$?
exit "$PREFILL_RC"
