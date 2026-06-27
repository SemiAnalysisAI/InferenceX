#!/usr/bin/env bash
# CollectiveX — GB200 (NVL72, MNNVL domain) SKU adapter. aarch64, 4 GPU/tray.
#
# Two paths, selected by CX_NODES:
#   * CX_NODES=1 (default): single tray, 4 GPU, intra-tray MNNVL. Hands off to
#     run_in_container.sh (CX_BENCH = nccl | deepep | all), -g 4.
#   * CX_NODES>1: multi-node over the NVL72 NVLink fabric (MNNVL), e.g. CX_NODES=2
#     = 8 GPU. nccl only — builds nccl-tests (MPI=1), runs each op across all ranks
#     via `srun --mpi=pmix` (1 GPU/rank), parses on the login node. Same shape that
#     runs single-node B200 (NVLink island) and multi-node B200 (CX-7 IB) — here it
#     stays entirely on NVL72 NVLink. Validated 8-GPU (2 trays) on-node.
#
# Run from inside the InferenceX checkout on the GB200 login node:
#     bash experimental/CollectiveX/launchers/launch_gb200-nv.sh             # 4 GPU, nccl
#     CX_NODES=2 bash .../launch_gb200-nv.sh                                  # 8 GPU MNNVL
#     CX_BENCH=deepep bash .../launch_gb200-nv.sh                             # 4 GPU, DeepEP
#
# Env knobs: CX_PARTITION(batch) CX_ACCOUNT(benchmark) CX_NODES(1)
#   CX_GPUS_PER_NODE(4) CX_TIME(30) CX_IMAGE CX_SQUASH_DIR CX_STAGE_DIR CX_BENCH
#   CX_OPS CX_MIN_BYTES CX_MAX_BYTES CX_SRUN_MPI(pmix) CX_DRYRUN(0)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CX_DIR="$(cd "$HERE/.." && pwd)"
REPO_ROOT="$(cd "$CX_DIR/../.." && pwd)"
# shellcheck source=../runtime/common.sh
source "$HERE/../runtime/common.sh"

RUNNER_NAME="${RUNNER_NAME:-gb200-nv}"
PARTITION="${CX_PARTITION:-batch}"
ACCOUNT="${CX_ACCOUNT:-benchmark}"
GPUS_PER_NODE="${CX_GPUS_PER_NODE:-4}"          # NVL72 compute tray = 4 GPU/node
NODES="${CX_NODES:-1}"
TIME_MIN="${CX_TIME:-30}"
IMAGE="${CX_IMAGE:-$(cx_default_image gb200)}"
SQUASH_DIR="${CX_SQUASH_DIR:-/mnt/lustre01/users-public/sa-shared}"
MOUNT_DIR=/ix
TS="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
WORLD=$((NODES * GPUS_PER_NODE))

export CX_RUNNER="$RUNNER_NAME" CX_TS="$TS"
export CX_TOPO="gb200-nvl72-mnnvl" CX_TRANSPORT="mnnvl"
export CX_BENCH="${CX_BENCH:-nccl}"
export CX_NCCL_HOME="${CX_NCCL_HOME:-/usr}"
export COLLECTIVEX_IMAGE="$IMAGE" COLLECTIVEX_IMAGE_DIGEST="${CX_IMAGE_DIGEST:-}"
# Validated GB200 MNNVL transport env (from serving recipes) — set AND recorded.
export NCCL_CUMEM_ENABLE=1 NCCL_MNNVL_ENABLE=1 MC_FORCE_MNNVL=1

cx_log "runner=$RUNNER_NAME partition=$PARTITION nodes=$NODES x ${GPUS_PER_NODE}gpu world=$WORLD bench=$CX_BENCH (aarch64)"
cx_log "image=$IMAGE"
SQUASH_FILE="$(cx_ensure_squash "$SQUASH_DIR" "$IMAGE")"
MOUNT_SRC="$(cx_stage_repo "$REPO_ROOT" "${CX_STAGE_DIR:-}")"
cx_log "squash=$SQUASH_FILE  mount=$MOUNT_SRC -> $MOUNT_DIR"

if [ "${CX_DRYRUN:-0}" = "1" ]; then cx_log "CX_DRYRUN=1 — not allocating"; exit 0; fi
command -v salloc >/dev/null || cx_die "salloc not found — run on the Slurm login node"

# ----------------------------------------------------------------------------
if [ "$NODES" -le 1 ]; then
  # Single tray (4 GPU): generic dispatcher, -g N single process.
  export CX_NGPUS="$GPUS_PER_NODE"
  salloc --partition="$PARTITION" --account="$ACCOUNT" --gres=gpu:"$GPUS_PER_NODE" \
         --exclusive --time="$TIME_MIN" --no-shell --job-name="$RUNNER_NAME"
  JOB_ID="$(squeue --name="$RUNNER_NAME" -u "$USER" -h -o %A | head -n1)"
  [ -n "$JOB_ID" ] || cx_die "could not resolve allocated JOB_ID"
  cx_log "JOB_ID=$JOB_ID"
  trap 'scancel "$JOB_ID" 2>/dev/null || true' EXIT
  srun --jobid="$JOB_ID" \
    --container-image="$SQUASH_FILE" --container-mounts="$MOUNT_SRC:$MOUNT_DIR" \
    --no-container-mount-home --container-workdir="$MOUNT_DIR/experimental/CollectiveX" \
    --no-container-entrypoint --export=ALL \
    bash "$MOUNT_DIR/experimental/CollectiveX/runtime/run_in_container.sh"
  cx_collect_results "$MOUNT_SRC" "$REPO_ROOT"
  cx_log "done — JSON artifacts under $MOUNT_SRC/experimental/CollectiveX/results/"
  exit 0
fi

# ----------------------------------------------------------------------------
# Multi-node MNNVL (nccl only): mirrors launch_b200-dgxc-slurm but stays on the
# NVL72 NVLink fabric. Build nccl-tests MPI=1, run each op across WORLD ranks
# (1 GPU/rank) via srun --mpi=pmix, parse on the login node.
[ "$CX_BENCH" = "nccl" ] || cx_die "GB200 multi-node supports CX_BENCH=nccl only (got '$CX_BENCH')"
MPI_FLAG="${CX_SRUN_MPI:-pmix}"
declare -A BIN=( [all_reduce]=all_reduce_perf [all_gather]=all_gather_perf
                 [reduce_scatter]=reduce_scatter_perf [alltoall]=alltoall_perf )

salloc --partition="$PARTITION" --account="$ACCOUNT" --nodes="$NODES" \
       --gres=gpu:"$GPUS_PER_NODE" --exclusive --time="$TIME_MIN" \
       --no-shell --job-name="$RUNNER_NAME"
JOB_ID="$(squeue --name="$RUNNER_NAME" -u "$USER" -h -o %A | head -n1)"
[ -n "$JOB_ID" ] || cx_die "could not resolve allocated JOB_ID"
cx_log "JOB_ID=$JOB_ID nodes=[$(squeue -j "$JOB_ID" -h -o %N 2>/dev/null)]"
trap 'scancel "$JOB_ID" 2>/dev/null || true' EXIT

COMMON_MOUNT=(--container-image="$SQUASH_FILE" --container-mounts="$MOUNT_SRC:$MOUNT_DIR"
              --no-container-mount-home --container-workdir="$MOUNT_DIR/experimental/CollectiveX"
              --no-container-entrypoint)
ENVJSON="$MOUNT_SRC/experimental/CollectiveX/results/env_${RUNNER_NAME}_${TS}.json"

# 1) Build nccl-tests (MPI=1) + capture environment (single task, one node).
srun --jobid="$JOB_ID" --ntasks=1 --nodes=1 "${COMMON_MOUNT[@]}" \
     --export=ALL,CX_TS="$TS",CX_RUNNER="$RUNNER_NAME" </dev/null \
  bash -c '
    set -euo pipefail
    cd /ix/experimental/CollectiveX
    source runtime/common.sh
    mkdir -p results
    cx_build_nccl_tests "$PWD/.nccl-tests" 1 >/dev/null
    python3 env_capture.py --out "results/env_${CX_RUNNER}_${CX_TS}.json" --timestamp "$CX_TS"
  '

BUILD_IN_CTR="$MOUNT_DIR/experimental/CollectiveX/.nccl-tests/nccl-tests-mpi/build"
OPS="${CX_OPS:-all_reduce all_gather reduce_scatter alltoall}"

# 2) Per op: run across all ranks (1 GPU/rank), tee raw output to the shared FS.
for op in $OPS; do
  raw="$MOUNT_SRC/experimental/CollectiveX/results/raw_${RUNNER_NAME}_${op}_${TS}.txt"
  cx_log "running $op across $WORLD ranks (mpi=$MPI_FLAG, MNNVL) -> $raw"
  srun --jobid="$JOB_ID" --mpi="$MPI_FLAG" --nodes="$NODES" \
       --ntasks="$WORLD" --ntasks-per-node="$GPUS_PER_NODE" "${COMMON_MOUNT[@]}" \
       --export=ALL,NCCL_CUMEM_ENABLE=1,NCCL_MNNVL_ENABLE=1,MC_FORCE_MNNVL=1 </dev/null \
       "$BUILD_IN_CTR/${BIN[$op]}" -b "${CX_MIN_BYTES:-8}" -e "${CX_MAX_BYTES:-2G}" -f 2 -g 1 -c 1 -w 5 -n 20 \
       > "$raw" 2>"$raw.stderr" || cx_log "WARN: $op srun returned nonzero (see $raw.stderr)"

  # 3) Parse on the login node (pure stdlib; no container needed).
  python3 "$CX_DIR/run_nccl.py" --op "$op" --parse-only "$raw" \
    --world-size "$WORLD" --nodes "$NODES" \
    --runner "$RUNNER_NAME" --topology-class "$CX_TOPO" --transport "$CX_TRANSPORT" \
    --env-json "$ENVJSON" \
    --out "$CX_DIR/results/${RUNNER_NAME}_${op}_${TS}.json" \
    --timestamp "$TS" || cx_log "WARN: parse $op failed"
done

cx_collect_results "$MOUNT_SRC" "$REPO_ROOT"
cx_log "done — JSON artifacts under $CX_DIR/results/"
