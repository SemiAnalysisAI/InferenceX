#!/usr/bin/env bash
# CollectiveX — GB200 (NVL72, MNNVL domain) SKU adapter. aarch64, 4 GPU/tray.
#
# Two paths, selected by CX_NODES:
#   * CX_NODES=1 (default): single tray, 4 GPU, intra-tray MNNVL. Hands off to
#     run_in_container.sh, -g 4.
#   * CX_NODES>1: runs the EP adapter across all ranks in the MNNVL domain.
#
# Scheduling and compute-visible storage are supplied by the runner-local config.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CX_DIR="$(cd "$HERE/.." && pwd)"
REPO_ROOT="$(cd "$CX_DIR/../.." && pwd)"
# shellcheck source=../runtime/common.sh
source "$HERE/../runtime/common.sh"

RUNNER_NAME="${RUNNER_NAME:-gb200-nv}"
cx_require_vars CX_PARTITION CX_ACCOUNT CX_SQUASH_DIR CX_STAGE_DIR
PARTITION="$CX_PARTITION"
ACCOUNT="$CX_ACCOUNT"
GPUS_PER_NODE="${CX_GPUS_PER_NODE:-4}"          # NVL72 compute tray = 4 GPU/node
SCALE_UP_DOMAIN="${CX_SCALE_UP_DOMAIN:-72}"
NODES="${CX_NODES:-1}"
TIME_MIN="${CX_TIME:-30}"
IMAGE="${CX_IMAGE:-$(cx_default_image gb200)}"
SQUASH_DIR="$CX_SQUASH_DIR"
MOUNT_DIR=/ix
TS="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
WORLD=$((NODES * GPUS_PER_NODE))
[ "$NODES" = 1 ] || [ "$NODES" = 2 ] || cx_die "GB200 supports one or two four-GPU trays"
[ "$GPUS_PER_NODE" = 4 ] || cx_die "GB200 requires four GPUs per tray"
[ "$SCALE_UP_DOMAIN" = 72 ] || cx_die "GB200 requires the NVL72 scale-up domain"
cx_apply_timing_profile

export CX_RUNNER="$RUNNER_NAME" CX_TS="$TS" CX_NGPUS="$WORLD" CX_GPUS_PER_NODE="$GPUS_PER_NODE"
export CX_SCALE_UP_DOMAIN="$SCALE_UP_DOMAIN"
export CX_TOPO="gb200-nvl72-mnnvl" CX_TRANSPORT="mnnvl"
export CX_BENCH="${CX_BENCH:-deepep}"
case "$CX_BENCH" in
  deepep|deepep-hybrid|uccl|nccl-ep|flashinfer) ;;
  *) cx_die "unsupported GB200 EP backend: $CX_BENCH" ;;
esac
export CX_NCCL_HOME="${CX_NCCL_HOME:-/usr}"
export COLLECTIVEX_IMAGE="$IMAGE" COLLECTIVEX_IMAGE_DIGEST="${CX_IMAGE_DIGEST:-$(cx_default_image_digest "$IMAGE")}"
# Required MNNVL transport settings, also recorded in provenance.
export NCCL_CUMEM_ENABLE=1 NCCL_MNNVL_ENABLE=1 MC_FORCE_MNNVL=1

cx_log "runner=$RUNNER_NAME nodes=$NODES x ${GPUS_PER_NODE}gpu world=$WORLD bench=$CX_BENCH (aarch64)"
SQUASH_FILE="$(cx_ensure_squash "$SQUASH_DIR" "$IMAGE")"
MOUNT_SRC="$(cx_stage_repo "$REPO_ROOT" "${CX_STAGE_DIR:-}")"

if [ "${CX_DRYRUN:-0}" = "1" ]; then cx_log "CX_DRYRUN=1 — not allocating"; exit 0; fi
command -v salloc >/dev/null || cx_die "salloc not found on this runner"

# ----------------------------------------------------------------------------
if [ "$NODES" -le 1 ]; then
  # Single tray (4 GPU): generic dispatcher, -g N single process.
  export CX_NGPUS="$GPUS_PER_NODE"
  JOB_ID="$(cx_salloc_jobid --partition="$PARTITION" --account="$ACCOUNT" --gres=gpu:"$GPUS_PER_NODE" \
            --exclusive --time="$TIME_MIN" --job-name="$RUNNER_NAME")"
  [ -n "$JOB_ID" ] || cx_die "could not resolve allocated JOB_ID from salloc"
  cx_log "JOB_ID=$JOB_ID"
  trap 'scancel "$JOB_ID" 2>/dev/null || true' EXIT
  srun --jobid="$JOB_ID" \
    --container-image="$SQUASH_FILE" --container-mounts="$MOUNT_SRC:$MOUNT_DIR" \
    --no-container-mount-home --container-workdir="$MOUNT_DIR/experimental/CollectiveX" \
    --no-container-entrypoint --export=ALL \
    bash "$MOUNT_DIR/experimental/CollectiveX/runtime/run_in_container.sh"
  cx_collect_results "$MOUNT_SRC" "$REPO_ROOT"
  cx_log "done — result artifacts collected"
  exit 0
fi

# ----------------------------------------------------------------------------
# Multi-node MNNVL EP path: run_ep.py across WORLD srun tasks (1 GPU/rank,
# per-rank RANK/LOCAL_RANK from SLURM_*), intranode NVLink across <=8 MNNVL ranks. One config/dispatch.

JOB_ID="$(cx_salloc_jobid --partition="$PARTITION" --account="$ACCOUNT" --nodes="$NODES" \
          --gres=gpu:"$GPUS_PER_NODE" --exclusive --time="$TIME_MIN" --job-name="$RUNNER_NAME")"
[ -n "$JOB_ID" ] || cx_die "could not resolve allocated JOB_ID from salloc"
cx_log "JOB_ID=$JOB_ID"
trap 'scancel "$JOB_ID" 2>/dev/null || true' EXIT

# Run run_ep.py across WORLD srun tasks over MNNVL.
  MA="$(scontrol show hostnames "$(squeue -j "$JOB_ID" -h -o %N 2>/dev/null)" 2>/dev/null | head -1)"; MP=29553
  mkdir -p "$MOUNT_SRC/experimental/CollectiveX/results"
  # Restore process-local loader/import paths and exact backend build identity from build-only.
  WRAP='[ -f /tmp/.cx_backend_env ] && . /tmp/.cx_backend_env; export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=$SLURM_LOCALID; exec python3 tests/run_ep.py "$@"'

  # Build legacy direct-env DeepEP or FlashInfer quant diagnostics once per node into a persistent
  # named container, then every case-srun reuses it (build visible to all WORLD ranks). Mirrors the
  # proven launch_gb300-nv.sh EP8 path: without this, the multi-srun ran ephemeral per-rank containers
  # that bypassed the build hooks (legacy direct-env DeepEP and quant-combine diagnostics).
  CNAME="cxep_${JOB_ID}"
  CMOUNT=(--container-mounts="$MOUNT_SRC:$MOUNT_DIR" --no-container-mount-home
          --container-workdir="$MOUNT_DIR/experimental/CollectiveX" --no-container-entrypoint)
  cx_log "EP backend preparation: bench=$CX_BENCH"
  srun --jobid="$JOB_ID" --nodes="$NODES" --ntasks-per-node=1 --container-name="$CNAME" \
    --container-image="$SQUASH_FILE" "${CMOUNT[@]}" --export=ALL,CX_BUILD_ONLY=1 \
    bash "$MOUNT_DIR/experimental/CollectiveX/runtime/run_in_container.sh" </dev/null 2>&1 | tail -15 \
    || cx_die "EP backend preparation failed"

  # Per-rank env. DeepEP main spans NVL72 trays only with allow_mnnvl=True (else DeepEP sets
  # NVSHMEM_DISABLE_MNNVL=1 -> intranode-IPC path -> illegal address cross-tray); CX_ALLOW_MNNVL=1 makes
  # tests/ep_deepep.py pass it (gated on the param existing, so bundled V1 is unchanged). flashinfer rides
  # NCCL's MNNVL transport.
  EP_EXPORTS="ALL,MASTER_ADDR=$MA,MASTER_PORT=$MP,NCCL_MNNVL_ENABLE=1,NCCL_CUMEM_ENABLE=1,MC_FORCE_MNNVL=1"
  [ "$CX_BENCH" = "deepep" ] && EP_EXPORTS="$EP_EXPORTS,CX_ALLOW_MNNVL=1"

  # SWEEP (CX_SHARD_FILE set): one pipe-delimited record per shard case so the rack-scale EP path sweeps EVERY
  # case (parity with single-node). MANUAL: one line per phase from the :-defaulted CX_* env.
  cx_ep_cases() {
    # CX_SHARD_FILE is workflow-relative (results/.shard_<id>.json, written under
    # working-directory=experimental/CollectiveX). This path runs on the SUBMIT HOST (cwd=repo root),
    # so resolve against $CX_DIR when not found as-is — else the SHARD branch is skipped and only ONE
    # default case runs instead of the shard's N.
    local sf="${CX_SHARD_FILE:-}"
    [ -n "$sf" ] && [ ! -f "$sf" ] && [ -f "$CX_DIR/$sf" ] && sf="$CX_DIR/$sf"
    if [ -n "$sf" ] && [ -f "$sf" ]; then
      # '|'-separated (NOT tab: tab is IFS-whitespace, so `read` collapses consecutive tabs and
      # swallows empty fields like a false eplb, shifting columns. No case field contains '|'.)
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
          "${CX_CANONICAL:+1}" "${CX_CASE_ID:-}" "$WORLD"
          "${CX_ITERS:-8}:${CX_TRIALS:-64}:${CX_WARMUP:-32}" "${CX_COMBINE_QUANT_MODE:-none}")
        (IFS='|'; printf '%s\n' "${fields[*]}")
      done
    fi
  }

  ci=0
  failed_cases=0
  while IFS='|' read -r ph dtype mode contract routing eplb rmode act placement rstep uneven \
      hidden topk experts lad suite workload required_pub canonical case_id ep timing combine_q; do
    [ -n "$ph" ] || continue
    ci=$((ci+1))
    case_stem="${RUNNER_NAME}_${CX_BENCH}_${ph}_${TS}-c$(printf '%03d' "$ci")"
    IFS=':' read -r case_iters case_trials case_warmup <<< "${timing:-8:64:32}"
    case_iters="${case_iters:-8}"; case_trials="${case_trials:-64}"; case_warmup="${case_warmup:-32}"
    ep="${ep:-$WORLD}"
    export CX_CASE_ID="$case_id" CX_SUITE="$suite" CX_WORKLOAD_NAME="$workload"
    export CX_REQUIRED_PUBLICATION="$required_pub" CX_CANONICAL="$canonical" CX_EP="$ep"
    export CX_DISPATCH_DTYPE="$dtype" CX_MODE="$mode" CX_MEASUREMENT_CONTRACT="$contract"
    export CX_ROUTING="$routing" CX_EPLB="$eplb" CX_RESOURCE_MODE="$rmode"
    export CX_ACTIVATION_PROFILE="$act" CX_PLACEMENT="$placement" CX_ROUTING_STEP="$rstep"
    export CX_UNEVEN_TOKENS="$uneven" CX_TOKENS_LADDER="$lad" CX_COMBINE_QUANT_MODE="$combine_q"
    export CX_ITERS="$case_iters" CX_TRIALS="$case_trials" CX_WARMUP="$case_warmup"
    export CX_SAMPLES_PER_POINT="$((case_iters * case_trials))"
    export CX_WARMUP_SEMANTICS="full-roundtrip-per-trial-point-v1"
    cx_log "EP${WORLD}[$ci] id=${case_id:-manual} $ph $CX_BENCH $dtype/$mode/$contract routing=$routing eplb=${eplb:-} rmode=$rmode act=$act plc=$placement"
    if [ "$ep" != "$WORLD" ]; then
      cx_log "ERROR: case EP$ep does not match allocated world size $WORLD"
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
        --export="$EP_EXPORTS" "${workload_args[@]}" </dev/null 2>&1 | tail -8
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
      --gpus-per-node "$GPUS_PER_NODE" --scale-up-domain "$SCALE_UP_DOMAIN"
      --routing-step "$rstep" --uneven-tokens "$uneven" --tokens-ladder "$lad"
      --hidden "$hidden" --topk "$topk" --experts "$experts"
      --warmup "$case_warmup" --iters "$case_iters" --trials "$case_trials"
      --seed "${CX_SEED:-67}" --runner "$RUNNER_NAME" --topology-class "$CX_TOPO"
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
      timeout -k 30 "${CX_RUN_TIMEOUT:-900}" srun --jobid="$JOB_ID" --nodes="$NODES" --ntasks="$WORLD" \
        --ntasks-per-node="$GPUS_PER_NODE" --container-name="$CNAME" "${CMOUNT[@]}" \
        --export="$EP_EXPORTS" \
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
      [ "$attempt" -lt "$attempts" ] && cx_log "EP${WORLD}[$ci] attempt $attempt/$attempts failed; retrying"
      attempt=$((attempt + 1))
    done
    if [ "$case_ok" = 0 ]; then
      failed_cases=$((failed_cases + 1))
      cx_log "ERROR: EP${WORLD}[$ci] failed after $attempts attempt(s)"
    fi
  done < <(cx_ep_cases)
  cx_collect_results "$MOUNT_SRC" "$REPO_ROOT"
  cx_log "done — EP result artifacts collected"
  [ "$failed_cases" -eq 0 ] || exit 1
  exit 0
