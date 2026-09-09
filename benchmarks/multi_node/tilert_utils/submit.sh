#!/usr/bin/env bash
set -x
NODES=$(( ${PREFILL_NODES:-1} + ${DECODE_NODES:-1} ))
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

salloc --partition="$SLURM_PARTITION" --account="$SLURM_ACCOUNT" \
    --nodes="$NODES" --gres=gpu:"$GPUS_PER_NODE" --exclusive --mem=0 \
    --time="${SALLOC_TIME_LIMIT:-480}" --no-shell --job-name="$RUNNER_NAME"
JOB_ID=$(squeue --name="$RUNNER_NAME" -u "$USER" -h -o %A | head -n1)

mapfile -t HOSTS < <(scontrol show hostnames "$(squeue -j "$JOB_ID" -h -o %N)")
[[ "${#HOSTS[@]}" -ge 2 ]] || { echo "expected >=2 nodes, got: ${HOSTS[*]}"; exit 1; }
export DECODE_HOST="${HOSTS[0]}" PREFILL_HOST="${HOSTS[1]}"

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
    # The tilert image bakes no NVIDIA_VISIBLE_DEVICES (unlike vllm-openai), and
    # enroot's nvidia hook only injects the driver when it is set — without it the
    # decode container has no libcuda and torch dies with "Found no NVIDIA driver".
    # docker --gpus sets this implicitly, which is why the image works elsewhere.
    # Exported here (not in --export) because the capabilities value contains a
    # comma, which srun's --export parsing would split on.
    export NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
    srun --jobid="$JOB_ID" --nodelist="$host" --ntasks=1 \
        --container-image="$squash_file" \
        --container-mounts="$GITHUB_WORKSPACE:/workspace,$MODEL_PATH:$MODEL_PATH,$TILERT_WEIGHTS_DIR:$TILERT_WEIGHTS_DIR" \
        --container-workdir=/workspace --no-container-entrypoint \
        --export=ALL,TILERT_ROLE="$role",DECODE_HOST="$DECODE_HOST",PREFILL_HOST="$PREFILL_HOST",PORT="${PORT:-8888}" \
        bash "/workspace/benchmarks/multi_node/tilert_utils/run_node.sh"
}

run_role decode "$DECODE_HOST" "$DECODE_SQUASH" &
DECODE_SRUN_PID=$!

run_role prefill "$PREFILL_HOST" "$PREFILL_SQUASH"
PREFILL_RC=$?

for _ in $(seq 1 "${TILERT_DECODE_DRAIN:-60}"); do
    kill -0 "$DECODE_SRUN_PID" 2>/dev/null || break
    sleep 1
done
kill -0 "$DECODE_SRUN_PID" 2>/dev/null && { echo "[submit] decode srun did not exit on its own, killing it"; kill "$DECODE_SRUN_PID" 2>/dev/null; }
wait "$DECODE_SRUN_PID" 2>/dev/null || true

exit "$PREFILL_RC"
