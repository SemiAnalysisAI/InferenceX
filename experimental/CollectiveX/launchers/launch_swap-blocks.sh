#!/usr/bin/env bash
# One-node, one-process vLLM copy benchmark using each pool's Slurm or Docker runtime.
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
source "$REPO_ROOT/benchmarks/benchmark_lib.sh" --validation-only
check_env_vars COLLX_SHARD_SKU COLLX_NODES COLLX_GPUS_PER_NODE COLLX_SWAP_IMAGE \
  COLLX_SWAP_MAX_PAYLOAD_BYTES COLLX_SWAP_BLOCK_BYTES COLLX_SWAP_NUM_BLOCKS COLLX_SWAP_WARMUP COLLX_SWAP_ITERATIONS \
  COLLX_SWAP_SEED COLLX_SWAP_DEVICE COLLX_SWAP_TIME COLLX_JOB_ROOT \
  COLLECTIVEX_SOURCE_SHA COLLECTIVEX_EXECUTION_ID COLLECTIVEX_CANONICAL_GHA COLLX_VENDOR \
  COLLX_IMAGE_REFRESH
source "$HERE/../runtime/common.sh"

[ "$COLLX_NODES" = 1 ] && [ "$COLLX_GPUS_PER_NODE" = 1 ] \
  || collx_die "swap-blocks requires one GPU process"
[[ "$COLLX_SWAP_IMAGE" =~ ^vllm/vllm-openai(-rocm)?:[A-Za-z0-9._-]+$ ]] \
  || collx_die "swap-blocks requires a tagged official vLLM image"
for value in "$COLLX_SWAP_BLOCK_BYTES" "$COLLX_SWAP_NUM_BLOCKS"; do
  [[ "$value" =~ ^[1-9][0-9]*(\ [1-9][0-9]*)*$ ]] \
    || collx_die "block sizes and counts must be space-separated positive integers"
done
for value in "$COLLX_SWAP_ITERATIONS" "$COLLX_SWAP_TIME" "$COLLX_SWAP_MAX_PAYLOAD_BYTES"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || collx_die "iterations, time, and payload budget must be positive integers"
done
for value in "$COLLX_SWAP_WARMUP" "$COLLX_SWAP_SEED" "$COLLX_SWAP_DEVICE"; do
  [[ "$value" =~ ^[0-9]+$ ]] || collx_die "warmup, seed, and device must be non-negative integers"
done

export COLLX_RUNNER="$COLLX_SHARD_SKU"
JOB_ID=""
NODES="$COLLX_NODES"
collx_install_launcher_fail_safe
collx_load_operator_config
collx_select_image "$COLLX_SWAP_IMAGE"
read -r -a block_bytes <<< "$COLLX_SWAP_BLOCK_BYTES"
read -r -a num_blocks <<< "$COLLX_SWAP_NUM_BLOCKS"
case "$COLLX_SHARD_SKU" in
  mi300x-tw|mi325x-tw)
    docker_cmd=(docker)
    if ! docker ps >/dev/null 2>&1; then
      sudo -n docker ps >/dev/null 2>&1 || collx_die "Docker is unavailable"
      docker_cmd=(sudo -n docker)
    fi
    "${docker_cmd[@]}" image inspect "$COLLX_SWAP_IMAGE" >/dev/null 2>&1 \
      || "${docker_cmd[@]}" pull "$COLLX_SWAP_IMAGE"
    mkdir -p "$REPO_ROOT/experimental/CollectiveX/results"
    container="cxswap_${COLLECTIVEX_EXECUTION_ID}"
    trap '"${docker_cmd[@]}" rm -f "$container" >/dev/null 2>&1 || true' EXIT
    groups=()
    for group in video render; do
      gid="$(getent group "$group" | cut -d: -f3)"
      [ -z "$gid" ] || groups+=(--group-add "$gid")
    done
    for layout in contiguous random; do
      "${docker_cmd[@]}" run --rm --name "$container" \
        --user "$(id -u):$(id -g)" "${groups[@]}" \
        --device /dev/kfd --device /dev/dri --ipc host \
        --security-opt seccomp=unconfined --entrypoint python3 \
        -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 \
        -e COLLECTIVEX_IMAGE -e COLLECTIVEX_SOURCE_SHA -e COLLX_SHARD_SKU \
        -v "$REPO_ROOT:/ix" -w /ix/experimental/CollectiveX "$COLLX_SWAP_IMAGE" \
        bench/run_swap_blocks.py --directions h2d d2h d2d \
        --block-bytes "${block_bytes[@]}" --num-blocks "${num_blocks[@]}" \
        --layout "$layout" --seed "$COLLX_SWAP_SEED" --device "$COLLX_SWAP_DEVICE" \
        --max-payload-bytes "$COLLX_SWAP_MAX_PAYLOAD_BYTES" \
        --warmup "$COLLX_SWAP_WARMUP" --iterations "$COLLX_SWAP_ITERATIONS" \
        --output "results/swap-blocks-$layout.json"
    done
    exit 0
    ;;
esac
[ -z "${COLLX_ENROOT_CACHE_PATH:-}" ] || export ENROOT_CACHE_PATH="$COLLX_ENROOT_CACHE_PATH"
check_env_vars COLLX_PARTITION COLLX_SQUASH_DIR COLLX_IMAGE_PLATFORM
collx_prepare_stage_dir "$COLLX_RUNNER"
check_env_vars COLLX_STAGE_DIR
collx_select_image "$COLLX_SWAP_IMAGE"
MOUNT_SRC="$(collx_stage_path "$REPO_ROOT" "$COLLX_STAGE_DIR")"
collx_stage_repo "$REPO_ROOT" "$MOUNT_SRC"
mkdir -p "$MOUNT_SRC/experimental/CollectiveX/results"

allocation=(--partition="$COLLX_PARTITION" --nodes="$NODES" --gres=gpu:1
  --ntasks-per-node=1 --exclusive --time="$COLLX_SWAP_TIME")
# Operator profile fields are deliberately optional in the registry.
[ -z "${COLLX_ACCOUNT:-}" ] || allocation+=(--account="$COLLX_ACCOUNT")
[ -z "${COLLX_QOS:-}" ] || allocation+=(--qos="$COLLX_QOS")
[ -z "${COLLX_NODELIST:-}" ] || allocation+=(--nodelist="$COLLX_NODELIST")
if [ -n "${COLLX_EXCLUDE_NODES:-}" ]; then
  existing_exclusions="$(python3 "$HERE/../runtime/swap_nodes.py" "$COLLX_EXCLUDE_NODES")" \
    || collx_die "cannot validate Slurm node exclusions"
  [ -z "$existing_exclusions" ] || allocation+=(--exclude="$existing_exclusions")
fi
collx_salloc_jobid "${allocation[@]}"
check_env_vars JOB_ID
SQUASH_FILE=""
# The serving launcher uses this exact image-tag filename in its operator-staged cache.
if [ "$COLLX_IMAGE_REFRESH" = 0 ] && [ -n "${COLLX_SWAP_STAGED_DIR:-}" ]; then
  staged_image="$COLLX_SWAP_STAGED_DIR/$(printf '%s' "$COLLX_SWAP_IMAGE" | sed 's#[/:@#]#_#g').sqsh"
  if unsquashfs -l "$staged_image" >/dev/null 2>&1; then
    collx_log "using operator-staged image: $staged_image"
    SQUASH_FILE="$staged_image"
  else
    collx_log "requested image is not staged: $staged_image"
  fi
fi
if [ -z "$SQUASH_FILE" ]; then
  SQUASH_FILE="$(collx_ensure_squash_on_job "$JOB_ID" "$COLLX_SQUASH_DIR" "$COLLX_SWAP_IMAGE")"
fi
check_env_vars SQUASH_FILE
read -r -a block_bytes <<< "$COLLX_SWAP_BLOCK_BYTES"
read -r -a num_blocks <<< "$COLLX_SWAP_NUM_BLOCKS"

container_mounts="$MOUNT_SRC:/ix"
case "$COLLX_SHARD_SKU" in
  mi300x|mi325x) container_mounts+=",/dev/kfd:/dev/kfd,/dev/dri:/dev/dri" ;;
esac
for layout in contiguous random; do
  runtime_log="$(collx_private_log_path "swap-blocks-$layout")"
  if ! srun --jobid="$JOB_ID" --nodes="$NODES" --ntasks=1 --ntasks-per-node=1 \
      --chdir=/tmp --container-image="$SQUASH_FILE" \
      --container-name="cxep_${JOB_ID}" --container-writable --container-remap-root \
      --container-mounts="$container_mounts" --no-container-mount-home --no-container-entrypoint \
      --container-workdir=/ix/experimental/CollectiveX \
      --export="$(collx_host_exports),COLLECTIVEX_IMAGE,COLLECTIVEX_SOURCE_SHA,COLLX_SHARD_SKU" \
      python3 bench/run_swap_blocks.py --directions h2d d2h d2d \
      --block-bytes "${block_bytes[@]}" --num-blocks "${num_blocks[@]}" \
      --layout "$layout" --seed "$COLLX_SWAP_SEED" --device "$COLLX_SWAP_DEVICE" \
      --max-payload-bytes "$COLLX_SWAP_MAX_PAYLOAD_BYTES" \
      --warmup "$COLLX_SWAP_WARMUP" --iterations "$COLLX_SWAP_ITERATIONS" \
      --output "results/swap-blocks-$layout.json" </dev/null > "$runtime_log" 2>&1; then
    collx_log_tail "$runtime_log"
    collx_die "swap-blocks $layout failed"
  fi
  cat "$runtime_log"
done
collx_collect_results "$MOUNT_SRC" "$REPO_ROOT"
