#!/usr/bin/env bash
# CollectiveX — GB300 (NVL72 Grace-Blackwell, aarch64) GHA launcher. Lands on the gb300-nv
# self-hosted runner (on the im-gb300-login-02 slurm login) and runs the chosen EP config.
#
# Two paths by CX_NODES:
#   CX_NODES<=1 (EP4): single NVL72 tray, 4 GPU. Hands off to run_in_container.sh (torchrun -g 4).
#   CX_NODES==2 (EP8): 2 trays, 8 GPU over the MNNVL NVLink domain. run_in_container's single-node
#     torchrun can't span nodes, so this path runs run_ep.py DIRECTLY across 8 srun tasks (1 rank
#     each), per-rank RANK/LOCAL_RANK from SLURM_*, MASTER_ADDR=first node — the intranode NVLink
#     path works across <=8 ranks on MNNVL (no internode/NVSHMEM). One CX_* config per dispatch.
#
# Env: CX_NODES(2) CX_PARTITION(batch_1) CX_ACCOUNT(benchmark) CX_BENCH(deepep) CX_PHASE + the
#   CX_DISPATCH_DTYPE/CX_MODE/CX_MEASUREMENT_CONTRACT/CX_ROUTING/CX_EPLB/CX_TOKENS_LADDER knobs.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CX_DIR="$(cd "$HERE/.." && pwd)"; REPO_ROOT="$(cd "$CX_DIR/../.." && pwd)"
# shellcheck source=common.sh
source "$HERE/common.sh"

PARTITION="${CX_PARTITION:-batch_1}"; ACCOUNT="${CX_ACCOUNT:-benchmark}"
NODES="${CX_NODES:-2}"; GPN="${CX_GPUS_PER_NODE:-4}"
NGPUS="${CX_NGPUS:-$((NODES*GPN))}"; TIME_MIN="${CX_TIME:-90}"
# CX_IMAGE is a docker TAG, not a squash path: cx_ensure_squash mangles the tag to
# <repo>_<tag>.sqsh and finds the pre-staged squash by THAT name (the same convention
# H200/B300 use). Passing a .sqsh PATH here made it try `enroot import docker://<path>`
# -> "Invalid image reference", then pyxis "No such file or directory" on the mangled
# target. The pre-staged file is /data/sa-shared/containers/lmsysorg_sglang_v0.5.11-cu130.sqsh,
# which is exactly the mangled name of this tag, so it resolves with no re-import.
IMAGE="${CX_IMAGE:-$(cx_default_image gb300)}"
SQUASH_DIR="${CX_SQUASH_DIR:-/data/sa-shared/containers}"
export CX_STAGE_DIR="${CX_STAGE_DIR:-/data/sa-shared/cx_stage}"
export ENROOT_CACHE_PATH="${ENROOT_CACHE_PATH:-/data/sa-shared/.enroot_cache}"
TS="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
RUNNER="gb300-${NGPUS}x"
export CX_RUNNER="$RUNNER" CX_TS="$TS" CX_TOPO="gb300-nvl72-mnnvl" CX_TRANSPORT="mnnvl"
export CX_BENCH="${CX_BENCH:-deepep}" CX_NGPUS="$NGPUS"
export COLLECTIVEX_IMAGE="$IMAGE" COLLECTIVEX_IMAGE_DIGEST="${CX_IMAGE_DIGEST:-}"

cx_log "GB300 runner=$RUNNER nodes=$NODES x ${GPN}gpu world=$NGPUS bench=$CX_BENCH phase=${CX_PHASE:-decode}"
SQUASH_FILE="$(cx_ensure_squash "$SQUASH_DIR" "$IMAGE")"
MOUNT_SRC="$(cx_stage_repo "$REPO_ROOT" "$CX_STAGE_DIR")"
[ "${CX_DRYRUN:-0}" = "1" ] && { cx_log "DRYRUN"; exit 0; }
command -v salloc >/dev/null || cx_die "salloc not found"

if [ "$NODES" -le 1 ]; then   # ---- EP4: single tray, run_in_container (torchrun -g 4) ----
  salloc --partition="$PARTITION" --account="$ACCOUNT" --gres=gpu:"$GPN" --exclusive \
         --time="$TIME_MIN" --no-shell --job-name="$RUNNER"
  JOB_ID="$(squeue --name="$RUNNER" -u "$USER" -h -o %A | head -n1)"; [ -n "$JOB_ID" ] || cx_die "no JOB_ID"
  trap 'scancel "$JOB_ID" 2>/dev/null || true' EXIT
  srun --jobid="$JOB_ID" --container-image="$SQUASH_FILE" --container-mounts="$MOUNT_SRC:/ix" \
    --no-container-mount-home --container-workdir=/ix/experimental/CollectiveX --no-container-entrypoint \
    --export=ALL bash /ix/experimental/CollectiveX/launchers/run_in_container.sh
  cx_collect_results "$MOUNT_SRC" "$REPO_ROOT"; exit 0
fi

# ---- EP8: 2 trays, run_ep.py directly across 8 ranks (no torchrun; MNNVL intranode path) ----
salloc --partition="$PARTITION" --account="$ACCOUNT" --nodes="$NODES" --gres=gpu:"$GPN" \
       --ntasks-per-node="$GPN" --exclusive --time="$TIME_MIN" --no-shell --job-name="$RUNNER"
JOB_ID="$(squeue --name="$RUNNER" -u "$USER" -h -o %A | head -n1)"; [ -n "$JOB_ID" ] || cx_die "no JOB_ID"
trap 'scancel "$JOB_ID" 2>/dev/null || true' EXIT
MA="$(scontrol show hostnames "$(squeue -j "$JOB_ID" -h -o %N)" | head -1)"; MP=29551
mkdir -p "$MOUNT_SRC/experimental/CollectiveX/results"
phases="${CX_PHASE:-decode}"; [ "$phases" = both ] && phases="decode prefill"
WRAP='export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=$SLURM_LOCALID; exec python3 tests/run_ep.py "$@"'
for ph in $phases; do
  out="results/${RUNNER}_${CX_BENCH}_${ph}_${TS}_${CX_DISPATCH_DTYPE:-bf16}_${CX_MODE:-normal}.json"
  cx_log "EP8 $ph $CX_DISPATCH_DTYPE/$CX_MODE/$CX_MEASUREMENT_CONTRACT routing=$CX_ROUTING eplb=${CX_EPLB:-}"
  # shellcheck disable=SC2086
  timeout -k 30 "${CX_RUN_TIMEOUT:-900}" srun --jobid="$JOB_ID" --nodes="$NODES" --ntasks="$NGPUS" \
    --ntasks-per-node="$GPN" --container-image="$SQUASH_FILE" --container-mounts="$MOUNT_SRC:/ix" \
    --no-container-mount-home --container-workdir=/ix/experimental/CollectiveX --no-container-entrypoint \
    --export=ALL,MASTER_ADDR="$MA",MASTER_PORT="$MP",NCCL_MNNVL_ENABLE=1,NCCL_CUMEM_ENABLE=1 \
    bash -c "$WRAP" _ --backend "$CX_BENCH" --phase "$ph" --dispatch-dtype "${CX_DISPATCH_DTYPE:-bf16}" \
      --mode "${CX_MODE:-normal}" --measurement-contract "${CX_MEASUREMENT_CONTRACT:-layout-and-dispatch-v1}" \
      --routing "${CX_ROUTING:-uniform}" ${CX_EPLB:+--eplb} --resource-mode "${CX_RESOURCE_MODE:-tuned}" \
      --tokens-ladder "${CX_TOKENS_LADDER:-}" --hidden "${CX_HIDDEN:-7168}" --topk "${CX_TOPK:-8}" \
      --experts "${CX_EXPERTS:-256}" --warmup "${CX_WARMUP:-32}" --iters "${CX_ITERS:-200}" \
      --trials "${CX_TRIALS:-3}" --seed "${CX_SEED:-67}" --runner "$RUNNER" --topology-class "$CX_TOPO" \
      --transport "$CX_TRANSPORT" --out "$out" </dev/null 2>&1 | tail -8
  cx_log "EP8 $ph rc=${PIPESTATUS[0]}"
done
cx_collect_results "$MOUNT_SRC" "$REPO_ROOT"
