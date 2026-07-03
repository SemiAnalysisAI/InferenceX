#!/usr/bin/env bash
# CollectiveX — GB300 (NVL72 Grace-Blackwell, aarch64) GHA launcher.
#
# Two paths by CX_NODES:
#   CX_NODES<=1 (EP4): single NVL72 tray, 4 GPU. Hands off to run_in_container.sh (torchrun -g 4).
#   CX_NODES==2 (EP8): 2 trays, 8 GPU over the MNNVL NVLink domain. run_in_container's single-node
#     torchrun can't span nodes, so this path runs run_ep.py DIRECTLY across 8 srun tasks (1 rank
#     each), per-rank RANK/LOCAL_RANK from SLURM_*, MASTER_ADDR=first node — the intranode NVLink
#     path works across <=8 ranks on MNNVL (no internode/NVSHMEM). One CX_* config per dispatch.
#
# Scheduling and compute-visible storage are supplied by the runner-local config.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CX_DIR="$(cd "$HERE/.." && pwd)"; REPO_ROOT="$(cd "$CX_DIR/../.." && pwd)"
# shellcheck source=../runtime/common.sh
source "$HERE/../runtime/common.sh"

cx_require_vars CX_PARTITION CX_ACCOUNT CX_SQUASH_DIR CX_STAGE_DIR CX_ENROOT_CACHE_PATH
PARTITION="$CX_PARTITION"; ACCOUNT="$CX_ACCOUNT"
NODES="${CX_NODES:-1}"; GPN="${CX_GPUS_PER_NODE:-4}"
SCALE_UP_DOMAIN="${CX_SCALE_UP_DOMAIN:-72}"
EXPECTED_WORLD=$((NODES * GPN))
NGPUS="${CX_NGPUS:-$EXPECTED_WORLD}"; TIME_MIN="${CX_TIME:-90}"
[ "$NODES" = 1 ] || [ "$NODES" = 2 ] || cx_die "GB300 supports one or two four-GPU trays"
[ "$GPN" = 4 ] || cx_die "GB300 requires four GPUs per tray"
[ "$SCALE_UP_DOMAIN" = 72 ] || cx_die "GB300 requires the NVL72 scale-up domain"
[ "$NGPUS" = "$EXPECTED_WORLD" ] || cx_die "GB300 world size must equal nodes x GPUs per tray"
cx_apply_timing_profile
# CX_IMAGE is a Docker tag; cx_ensure_squash derives the local squash filename.
IMAGE="${CX_IMAGE:-$(cx_default_image gb300)}"
SQUASH_DIR="$CX_SQUASH_DIR"
export ENROOT_CACHE_PATH="$CX_ENROOT_CACHE_PATH"
TS="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
RUNNER="gb300-${NGPUS}x"
export CX_RUNNER="$RUNNER" CX_TS="$TS" CX_TOPO="gb300-nvl72-mnnvl" CX_TRANSPORT="mnnvl"
export CX_GPUS_PER_NODE="$GPN" CX_SCALE_UP_DOMAIN="$SCALE_UP_DOMAIN"
export CX_BENCH="${CX_BENCH:-deepep}" CX_NGPUS="$NGPUS"
export COLLECTIVEX_IMAGE="$IMAGE" COLLECTIVEX_IMAGE_DIGEST="${CX_IMAGE_DIGEST:-$(cx_default_image_digest "$IMAGE")}"
export NCCL_CUMEM_ENABLE=1 NCCL_MNNVL_ENABLE=1 MC_FORCE_MNNVL=1

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
MA="$(scontrol show hostnames "$(squeue -j "$JOB_ID" -h -o %N 2>/dev/null)" 2>/dev/null | head -1)"; MP=29551
mkdir -p "$MOUNT_SRC/experimental/CollectiveX/results"
# Restore process-local loader/import paths and exact backend build identity from build-only.
WRAP='[ -f /tmp/.cx_backend_env ] && . /tmp/.cx_backend_env; export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=$SLURM_LOCALID; exec python3 tests/run_ep.py "$@"'

# From-source diagnostic kernels cannot be built in the per-rank multi-srun
# (8 separate ephemeral containers). Build them ONCE PER NODE into a PERSISTENT named container, then
# every case-srun REUSES it (--container-name, no re-import) so the build is visible to all 8 ranks.
# Brings the EP8 rack path to parity with EP4 (run_in_container builds once + reuses). Mounts re-apply
# per srun-step (not persisted in the container fs), so each srun still passes "${CMOUNT[@]}".
CNAME="cxep8_${JOB_ID}"
CMOUNT=(--container-mounts="$MOUNT_SRC:/ix" --no-container-mount-home
        --container-workdir=/ix/experimental/CollectiveX --no-container-entrypoint)
cx_log "EP backend preparation: bench=$CX_BENCH"
srun --jobid="$JOB_ID" --nodes="$NODES" --ntasks-per-node=1 \
  --container-name="$CNAME" --container-image="$SQUASH_FILE" "${CMOUNT[@]}" --export=ALL,CX_BUILD_ONLY=1 \
  bash /ix/experimental/CollectiveX/runtime/run_in_container.sh </dev/null 2>&1 | tail -15 \
  || cx_die "EP backend preparation failed"

# The EP8 case list as pipe-delimited records. SWEEP (CX_SHARD_FILE set): one line per shard case,
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
        g("ladder",""), g("suite",""), g("workload",""), g("required_publication",""),
        ("1" if c.get("canonical") else ""), g("case_id",""), g("ep",""),
        g("timing","8:64:32"), g("combine_quant_mode","none")]))
PY
  else
    local phases="${CX_PHASE:-decode}"; [ "$phases" = both ] && phases="decode prefill"
    local ph; local -a fields
    for ph in $phases; do
      fields=("$ph" "${CX_DISPATCH_DTYPE:-bf16}" "${CX_MODE:-normal}"
        "${CX_MEASUREMENT_CONTRACT:-layout-and-dispatch-v1}" "${CX_ROUTING:-uniform}"
        "${CX_EPLB:+1}" "${CX_RESOURCE_MODE:-tuned}" "${CX_ACTIVATION_PROFILE:-normal}"
        "${CX_PLACEMENT:-packed}" "${CX_ROUTING_STEP:-0}" "${CX_UNEVEN_TOKENS:-none}"
        "${CX_HIDDEN:-7168}" "${CX_TOPK:-8}" "${CX_EXPERTS:-256}" "${CX_TOKENS_LADDER:-}"
        "${CX_SUITE:-}" "${CX_WORKLOAD_NAME:-}" "${CX_REQUIRED_PUBLICATION:-}"
        "${CX_CANONICAL:+1}" "${CX_CASE_ID:-}" "$NGPUS"
        "${CX_ITERS:-8}:${CX_TRIALS:-64}:${CX_WARMUP:-32}" "${CX_COMBINE_QUANT_MODE:-none}")
      (IFS='|'; printf '%s\n' "${fields[*]}")
    done
  fi
}

# Per-rank env for the EP8 case sruns. flashinfer-combine rides NCCL's MNNVL transport (validated:
# cq=fp8/nvfp4 @ ws8). DeepEP main's Buffer gates multi-tray NVLink behind allow_mnnvl, which defaults
# False -> DeepEP then sets NVSHMEM_DISABLE_MNNVL=1 and the legacy buffer takes the intranode-only CUDA-IPC
# peer path, faulting across NVL72 trays (cudaErrorIllegalAddress at csrc/legacy/buffer.hpp). CX_ALLOW_MNNVL=1
# makes tests/ep_deepep.py pass allow_mnnvl=True so the NVL buffer spans both trays over the fabric API.
# Bundled V1's Buffer predates the param (its NVL buffer already spans MNNVL) -> the harness drops the kwarg.
EP8_EXPORTS="ALL,MASTER_ADDR=$MA,MASTER_PORT=$MP,NCCL_MNNVL_ENABLE=1,NCCL_CUMEM_ENABLE=1,MC_FORCE_MNNVL=1"
[ "$CX_BENCH" = "deepep" ] && EP8_EXPORTS="$EP8_EXPORTS,CX_ALLOW_MNNVL=1"

ci=0
failed_cases=0
while IFS='|' read -r ph dtype mode contract routing eplb rmode act placement rstep uneven \
    hidden topk experts lad suite workload required_pub canonical case_id ep timing combine_q; do
  [ -n "$ph" ] || continue
  ci=$((ci+1))
  case_stem="${RUNNER}_${CX_BENCH}_${ph}_${TS}-c$(printf '%03d' "$ci")"
  IFS=':' read -r case_iters case_trials case_warmup <<< "${timing:-8:64:32}"
  case_iters="${case_iters:-8}"; case_trials="${case_trials:-64}"; case_warmup="${case_warmup:-32}"
  ep="${ep:-$NGPUS}"
  export CX_CASE_ID="$case_id" CX_SUITE="$suite" CX_WORKLOAD_NAME="$workload"
  export CX_REQUIRED_PUBLICATION="$required_pub" CX_CANONICAL="$canonical" CX_EP="$ep"
  export CX_DISPATCH_DTYPE="$dtype" CX_MODE="$mode" CX_MEASUREMENT_CONTRACT="$contract"
  export CX_ROUTING="$routing" CX_EPLB="$eplb" CX_RESOURCE_MODE="$rmode"
  export CX_ACTIVATION_PROFILE="$act" CX_PLACEMENT="$placement" CX_ROUTING_STEP="$rstep"
  export CX_UNEVEN_TOKENS="$uneven" CX_TOKENS_LADDER="$lad" CX_COMBINE_QUANT_MODE="$combine_q"
  export CX_ITERS="$case_iters" CX_TRIALS="$case_trials" CX_WARMUP="$case_warmup"
  export CX_SAMPLES_PER_POINT="$((case_iters * case_trials))"
  export CX_WARMUP_SEMANTICS="full-roundtrip-per-trial-point-v1"
  cx_log "EP${NGPUS}[$ci] id=${case_id:-manual} $ph $CX_BENCH $dtype/$mode/$contract rt=$routing eplb=${eplb:-} combine=${CX_COMBINE_DTYPE:-bf16}/$combine_q"
  if [ "$ep" != "$NGPUS" ]; then
    cx_log "ERROR: case EP$ep does not match allocated world size $NGPUS"
    export CX_ATTEMPT_ID=1
    failure_out="$MOUNT_SRC/experimental/CollectiveX/results/failed_${case_stem}-a01.json"
    cx_emit_ep_failed_case "$failure_out" "$CX_BENCH" "$ph" 5
    failed_cases=$((failed_cases + 1))
    continue
  fi

  workload_dir=""
  if [ -n "$canonical" ]; then
    workload_dir=".cx_workloads/ep${ep}_${routing}"
    workload_ladder="$lad"
    [ -n "$workload_ladder" ] || workload_ladder="1 2 4 8 16 32 64 128 256 512 1024 2048 4096"
    workload_args=(python3 tests/make_workloads.py --out-dir "$workload_dir" --routing "$routing"
      --ep "$ep" --hidden "$hidden" --topk "$topk" --experts "$experts"
      --seed "${CX_SEED:-67}" --tokens-ladder "$workload_ladder")
    [ -n "$workload" ] && workload_args+=(--workload "$workload")
    stage_rc=0
    set +e
    srun --jobid="$JOB_ID" --nodes=1 --ntasks=1 --container-name="$CNAME" "${CMOUNT[@]}" \
      --export="$EP8_EXPORTS" "${workload_args[@]}" </dev/null 2>&1 | tail -8
    stage_status=("${PIPESTATUS[@]}")
    set -e
    stage_rc="${stage_status[0]}"
    if [ "$stage_rc" != 0 ]; then
      cx_log "ERROR: canonical workload staging failed rc=$stage_rc"
      export CX_ATTEMPT_ID=1
      failure_out="$MOUNT_SRC/experimental/CollectiveX/results/failed_${case_stem}-a01.json"
      cx_emit_ep_failed_case "$failure_out" "$CX_BENCH" "$ph" "$stage_rc"
      failed_cases=$((failed_cases + 1))
      continue
    fi
  fi

  ep_args=(--backend "$CX_BENCH" --phase "$ph" --dispatch-dtype "$dtype"
    --mode "$mode" --measurement-contract "$contract" --routing "$routing"
    --resource-mode "$rmode" --sm-fraction "${CX_SM_FRACTION:-0.18}"
    --num-sms "${CX_NUM_SMS:-24}" --activation-profile "$act" --placement "$placement"
    --gpus-per-node "$GPN" --scale-up-domain "$SCALE_UP_DOMAIN"
    --routing-step "$rstep" --uneven-tokens "$uneven" --tokens-ladder "$lad"
    --hidden "$hidden" --topk "$topk" --experts "$experts"
    --warmup "$case_warmup" --iters "$case_iters" --trials "$case_trials"
    --seed "${CX_SEED:-67}" --runner "$RUNNER" --topology-class "$CX_TOPO"
    --transport "$CX_TRANSPORT" --case-id "$case_id" --suite "$suite"
    --workload-name "$workload" --required-publication "$required_pub"
    --combine-quant-mode "$combine_q")
  [ -n "$eplb" ] && ep_args+=(--eplb)
  [ -n "$workload_dir" ] && ep_args+=(--workload-dir "$workload_dir")
  [ -n "${CX_COMBINE_DTYPE:-}" ] && ep_args+=(--combine-dtype "$CX_COMBINE_DTYPE")
  attempts=1
  [ "$CX_BENCH" = "flashinfer" ] && attempts=$(( ${CX_FLASHINFER_RETRIES:-3} + 1 ))
  attempt=1
  case_ok=0
  while [ "$attempt" -le "$attempts" ]; do
    export CX_ATTEMPT_ID="$attempt"
    attempt_tag="a$(printf '%02d' "$attempt")"
    out="results/${case_stem}_${attempt_tag}_${dtype}_${mode}.json"
    failure_out="$MOUNT_SRC/experimental/CollectiveX/results/failed_${case_stem}-${attempt_tag}.json"
    set +e
    timeout -k 30 "${CX_RUN_TIMEOUT:-900}" srun --jobid="$JOB_ID" --nodes="$NODES" --ntasks="$NGPUS" \
      --ntasks-per-node="$GPN" --container-name="$CNAME" "${CMOUNT[@]}" \
      --export="$EP8_EXPORTS" \
      bash -c "$WRAP" _ "${ep_args[@]}" --out "$out" </dev/null 2>&1 | tail -8
    run_status=("${PIPESTATUS[@]}")
    set -e
    run_rc="${run_status[0]}"
    expected_out="$MOUNT_SRC/experimental/CollectiveX/$out"
    if [ "$run_rc" = 0 ] && cx_has_result_doc "$expected_out"; then
      case_ok=1
      break
    fi
    [ "$run_rc" = 0 ] && run_rc=1
    if cx_has_result_doc "$expected_out"; then
      cx_demote_result_doc "$expected_out" "$run_rc" \
        || { rm -f "$expected_out"; cx_emit_ep_failed_case "$failure_out" "$CX_BENCH" "$ph" "$run_rc"; }
    else
      cx_emit_ep_failed_case "$failure_out" "$CX_BENCH" "$ph" "$run_rc"
    fi
    [ "$attempt" -lt "$attempts" ] && cx_log "EP${NGPUS}[$ci] attempt $attempt/$attempts failed; retrying"
    attempt=$((attempt + 1))
  done
  if [ "$case_ok" = 0 ]; then
    failed_cases=$((failed_cases + 1))
    cx_log "ERROR: EP${NGPUS}[$ci] failed after $attempts attempt(s)"
  fi
done < <(cx_ep8_cases)
cx_collect_results "$MOUNT_SRC" "$REPO_ROOT"
[ "$failed_cases" -eq 0 ] || exit 1
