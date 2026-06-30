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
# shellcheck source=../runtime/common.sh
source "$HERE/../runtime/common.sh"

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
  JOB_ID="$(cx_salloc_jobid --partition="$PARTITION" --account="$ACCOUNT" --gres=gpu:"$GPN" --exclusive \
            --time="$TIME_MIN" --job-name="$RUNNER")"
  [ -n "$JOB_ID" ] || cx_die "no JOB_ID from salloc"
  trap 'scancel "$JOB_ID" 2>/dev/null || true' EXIT
  srun --jobid="$JOB_ID" --container-image="$SQUASH_FILE" --container-mounts="$MOUNT_SRC:/ix" \
    --no-container-mount-home --container-workdir=/ix/experimental/CollectiveX --no-container-entrypoint \
    --export=ALL bash /ix/experimental/CollectiveX/runtime/run_in_container.sh
  cx_collect_results "$MOUNT_SRC" "$REPO_ROOT"; exit 0
fi

# ---- EP8: 2 trays, run_ep.py directly across 8 ranks (no torchrun; MNNVL intranode path) ----
JOB_ID="$(cx_salloc_jobid --partition="$PARTITION" --account="$ACCOUNT" --nodes="$NODES" --gres=gpu:"$GPN" \
          --ntasks-per-node="$GPN" --exclusive --time="$TIME_MIN" --job-name="$RUNNER")"
[ -n "$JOB_ID" ] || cx_die "no JOB_ID from salloc"
trap 'scancel "$JOB_ID" 2>/dev/null || true' EXIT
MA="$(scontrol show hostnames "$(squeue -j "$JOB_ID" -h -o %N)" | head -1)"; MP=29551
mkdir -p "$MOUNT_SRC/experimental/CollectiveX/results"
WRAP='export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=$SLURM_LOCALID; exec python3 tests/run_ep.py "$@"'

# From-source kernels (DeepEP V2 / flashinfer quant-combine) cannot be built in the per-rank multi-srun
# (8 separate ephemeral containers). Build them ONCE PER NODE into a PERSISTENT named container, then
# every case-srun REUSES it (--container-name, no re-import) so the build is visible to all 8 ranks.
# Brings the EP8 rack path to parity with EP4 (run_in_container builds once + reuses). Mounts re-apply
# per srun-step (not persisted in the container fs), so each srun still passes "${CMOUNT[@]}".
CNAME="cxep8_${JOB_ID}"
CMOUNT=(--container-mounts="$MOUNT_SRC:/ix" --no-container-mount-home
        --container-workdir=/ix/experimental/CollectiveX --no-container-entrypoint)
cx_log "EP8 setup: build into named container $CNAME per node (deepep_v2=${CX_DEEPEP_V2:-} combine=${CX_COMBINE_DTYPE:-bf16})"
srun --jobid="$JOB_ID" --nodes="$NODES" --ntasks-per-node=1 \
  --container-name="$CNAME" --container-image="$SQUASH_FILE" "${CMOUNT[@]}" --export=ALL,CX_BUILD_ONLY=1 \
  bash /ix/experimental/CollectiveX/runtime/run_in_container.sh </dev/null 2>&1 | tail -15 \
  || cx_log "WARN: EP8 build-only step returned nonzero (see above)"

# The EP8 case list as TAB-separated arg-lines. SWEEP (CX_SHARD_FILE set): one line per shard case,
# so the rack-scale EP8 path sweeps EVERY case of its shard (parity with run_in_container's single-
# node SHARD loop) instead of the old single CX_* config. MANUAL (no shard file): one line per phase
# from the CX_* env — every field is :-defaulted so set -u never trips on an unset knob (the old bug:
# bare $CX_DISPATCH_DTYPE here was unbound under sweep, crashing the whole job on its first line).
cx_ep8_cases() {
  # CX_SHARD_FILE is workflow-relative (results/.shard_<id>.json, written by the Extract step with
  # working-directory=experimental/CollectiveX). This EP8 path runs on the SUBMIT HOST where cwd is
  # the repo root, so resolve it against $CX_DIR (=experimental/CollectiveX) when not found as-is —
  # else the SHARD branch is skipped and only ONE default case runs instead of the shard's N.
  local sf="${CX_SHARD_FILE:-}"
  [ -n "$sf" ] && [ ! -f "$sf" ] && [ -f "$CX_DIR/$sf" ] && sf="$CX_DIR/$sf"
  if [ -n "$sf" ] && [ -f "$sf" ]; then
    # '|'-separated (NOT tab: tab is IFS-whitespace, so `read` would collapse consecutive tabs and
    # swallow empty fields like a false eplb, shifting every column. No case field contains '|'.)
    python3 - "$sf" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for c in d.get("cases", []):
    g = lambda k, dv: (str(c[k]) if c.get(k) not in (None, "") else dv)
    print("|".join([g("phase","decode"), g("dtype","bf16"), g("mode","normal"),
        g("contract","layout-and-dispatch-v1"), g("routing","uniform"),
        ("1" if c.get("eplb") else ""), g("resource_mode","tuned"),
        g("activation_profile","normal"), g("placement","packed"), g("routing_step","0"),
        g("uneven_tokens","none"), g("hidden","7168"), g("topk","8"), g("experts","256"),
        g("ladder","")]))
PY
  else
    local phases="${CX_PHASE:-decode}"; [ "$phases" = both ] && phases="decode prefill"
    local ph
    for ph in $phases; do
      printf '%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\n' \
        "$ph" "${CX_DISPATCH_DTYPE:-bf16}" "${CX_MODE:-normal}" \
        "${CX_MEASUREMENT_CONTRACT:-layout-and-dispatch-v1}" "${CX_ROUTING:-uniform}" \
        "${CX_EPLB:+1}" "${CX_RESOURCE_MODE:-tuned}" "${CX_ACTIVATION_PROFILE:-normal}" \
        "${CX_PLACEMENT:-packed}" "${CX_ROUTING_STEP:-0}" "${CX_UNEVEN_TOKENS:-none}" \
        "${CX_HIDDEN:-7168}" "${CX_TOPK:-8}" "${CX_EXPERTS:-256}" "${CX_TOKENS_LADDER:-}"
    done
  fi
}

ci=0
while IFS='|' read -r ph dtype mode contract routing eplb rmode act placement rstep uneven hidden topk experts lad; do
  [ -n "$ph" ] || continue
  ci=$((ci+1))
  out="results/${RUNNER}_${CX_BENCH}_${ph}_${TS}-c$(printf '%03d' "$ci")_${dtype}_${mode}.json"
  cx_log "EP8[$ci] $ph $CX_BENCH $dtype/$mode/$contract rt=$routing eplb=${eplb:-} combine=${CX_COMBINE_DTYPE:-bf16}/${CX_COMBINE_QUANT_MODE:-none}"
  # shellcheck disable=SC2086
  timeout -k 30 "${CX_RUN_TIMEOUT:-900}" srun --jobid="$JOB_ID" --nodes="$NODES" --ntasks="$NGPUS" \
    --ntasks-per-node="$GPN" --container-name="$CNAME" "${CMOUNT[@]}" \
    --export=ALL,MASTER_ADDR="$MA",MASTER_PORT="$MP",NCCL_MNNVL_ENABLE=1,NCCL_CUMEM_ENABLE=1 \
    bash -c "$WRAP" _ --backend "$CX_BENCH" --phase "$ph" --dispatch-dtype "$dtype" \
      --mode "$mode" --measurement-contract "$contract" \
      --routing "$routing" ${eplb:+--eplb} --resource-mode "$rmode" \
      --activation-profile "$act" --placement "$placement" --routing-step "$rstep" --uneven-tokens "$uneven" \
      --tokens-ladder "$lad" --hidden "$hidden" --topk "$topk" \
      --experts "$experts" --warmup "${CX_WARMUP:-32}" --iters "${CX_ITERS:-200}" \
      --trials "${CX_TRIALS:-3}" --seed "${CX_SEED:-67}" --runner "$RUNNER" --topology-class "$CX_TOPO" \
      --transport "$CX_TRANSPORT" \
      ${CX_COMBINE_DTYPE:+--combine-dtype "$CX_COMBINE_DTYPE"} ${CX_COMBINE_QUANT_MODE:+--combine-quant-mode "$CX_COMBINE_QUANT_MODE"} \
      --out "$out" </dev/null 2>&1 | tail -8
  cx_log "EP8[$ci] $ph rc=${PIPESTATUS[0]}"
done < <(cx_ep8_cases)
cx_collect_results "$MOUNT_SRC" "$REPO_ROOT"
