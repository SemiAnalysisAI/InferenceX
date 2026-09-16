#!/usr/bin/env bash
# One-node, one-process vLLM copy benchmark using CollectiveX's H200 allocation path.
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
source "$REPO_ROOT/benchmarks/benchmark_lib.sh" --validation-only
check_env_vars COLLX_SHARD_SKU COLLX_NODES COLLX_GPUS_PER_NODE COLLX_SWAP_IMAGE \
  COLLX_SWAP_MAX_PAYLOAD_BYTES COLLX_SWAP_BLOCK_BYTES COLLX_SWAP_NUM_BLOCKS COLLX_SWAP_WARMUP COLLX_SWAP_ITERATIONS \
  COLLX_SWAP_SEED COLLX_SWAP_DEVICE COLLX_SWAP_TIME COLLX_JOB_ROOT \
  COLLECTIVEX_SOURCE_SHA COLLECTIVEX_EXECUTION_ID COLLECTIVEX_CANONICAL_GHA
source "$HERE/../runtime/common.sh"

[ "$COLLX_SHARD_SKU" = h200-dgxc ] && [ "$COLLX_NODES" = 1 ] \
  && [ "$COLLX_GPUS_PER_NODE" = 1 ] || collx_die "swap-blocks requires one H200 GPU process"
[[ "$COLLX_SWAP_IMAGE" =~ ^vllm/vllm-openai:[A-Za-z0-9._-]+$ ]] \
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
[ -z "${COLLX_EXCLUDE_NODES:-}" ] || allocation+=(--exclude="$COLLX_EXCLUDE_NODES")
collx_salloc_jobid "${allocation[@]}"
check_env_vars JOB_ID
SQUASH_FILE="$(collx_ensure_squash_on_job "$JOB_ID" "$COLLX_SQUASH_DIR" "$COLLX_SWAP_IMAGE")"
check_env_vars SQUASH_FILE
read -r -a block_bytes <<< "$COLLX_SWAP_BLOCK_BYTES"
read -r -a num_blocks <<< "$COLLX_SWAP_NUM_BLOCKS"

for layout in contiguous random; do
  runtime_log="$(collx_private_log_path "swap-blocks-$layout")"
  if ! srun --jobid="$JOB_ID" --nodes="$NODES" --ntasks=1 --ntasks-per-node=1 \
      --chdir=/tmp --container-image="$SQUASH_FILE" \
      --container-name="cxep_${JOB_ID}" --container-writable --container-remap-root \
      --container-mounts="$MOUNT_SRC:/ix" --no-container-mount-home --no-container-entrypoint \
      --container-workdir=/ix/experimental/CollectiveX \
      --export="$(collx_host_exports),COLLECTIVEX_IMAGE,COLLECTIVEX_SOURCE_SHA" \
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
