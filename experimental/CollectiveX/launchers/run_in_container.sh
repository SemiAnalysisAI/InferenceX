#!/usr/bin/env bash
# CollectiveX — generic in-container benchmark dispatcher (single-node).
#
# Runs INSIDE the container under `srun`, invoked by every per-SKU adapter
# (launch_<sku>.sh). The SKU adapter handles allocation/container/transport-env;
# this script decides WHICH benchmark to run from CX_BENCH, so any benchmark can
# be driven through any SKU's launch script. Writes provenance-tagged JSON to
# results/.
#
# Required env (exported by the adapter): CX_RUNNER CX_NGPUS CX_TS CX_TOPO
# Selector:        CX_BENCH = nccl | deepep | mori | all    (default nccl)
#                  (mori = AMD ROCm EP; nccl/deepep = NVIDIA. `all` = nccl+deepep.)
# NCCL knobs:      CX_OPS, CX_MIN_BYTES, CX_MAX_BYTES, CX_TRANSPORT, CX_NCCL_HOME
# EP knobs (DeepEP/MoRI), all -> tests/run_ep.py:
#   CX_PHASE = decode | prefill | both (default decode)   <- picks the token sweep
#   CX_TOKENS_LADDER (space/comma sep; blank = phase default), CX_TOKENS_PER_RANK (legacy single point)
#   CX_HIDDEN CX_TOPK CX_EXPERTS CX_DISPATCH_DTYPE CX_ROUTING CX_MODE(normal|ll)
#   CX_NUM_SMS (DeepEP comm SMs) CX_SEED CX_ITERS
set -euo pipefail

cd /ix/experimental/CollectiveX
# shellcheck source=common.sh
source launchers/common.sh
mkdir -p results

: "${CX_RUNNER:?CX_RUNNER not set}"
: "${CX_NGPUS:?CX_NGPUS not set}"
: "${CX_TS:?CX_TS not set}"
: "${CX_TOPO:?CX_TOPO not set}"
CX_BENCH="${CX_BENCH:-nccl}"
CX_TRANSPORT="${CX_TRANSPORT:-}"
ENVJSON="results/env_${CX_RUNNER}_${CX_TS}.json"

cx_log "in-container: runner=$CX_RUNNER ngpus=$CX_NGPUS bench=$CX_BENCH topo=$CX_TOPO"
python3 env_capture.py --out "$ENVJSON" --timestamp "$CX_TS"

run_nccl_suite() {
  local build ops op sfail=0 impl=nccl
  # AMD/ROCm -> rccl-tests (fork; same binaries + output, parsed by run_nccl.py);
  # NVIDIA/CUDA -> nccl-tests. Both single-node: MPI=0, -g N.
  if [ -d /opt/rocm ] || command -v hipcc >/dev/null 2>&1; then
    impl=rccl
    build="$(cx_build_rccl_tests "$PWD/.nccl-tests" 0)" || return 1
  else
    build="$(cx_build_nccl_tests "$PWD/.nccl-tests" 0)" || return 1
  fi
  cx_log "collective impl=$impl build=$build"
  ops="${CX_OPS:-all_reduce all_gather reduce_scatter alltoall}"
  for op in $ops; do
    if ! python3 run_nccl.py --op "$op" --nccl-tests-dir "$build" \
        --world-size "$CX_NGPUS" --nodes 1 --gpus-per-proc "$CX_NGPUS" \
        --runner "$CX_RUNNER" --topology-class "$CX_TOPO" --transport "$CX_TRANSPORT" \
        --env-json "$ENVJSON" --out "results/${CX_RUNNER}_${op}_${CX_TS}.json" \
        --min-bytes "${CX_MIN_BYTES:-8}" --max-bytes "${CX_MAX_BYTES:-8G}" --check 1; then
      cx_log "WARN: $impl $op failed or invalid"; sfail=1
    fi
  done
  return "$sfail"
}

# Resolve the source-tokens-per-rank sweep: explicit CX_TOKENS_LADDER wins; else
# the legacy single-point CX_TOKENS_PER_RANK becomes a one-point ladder; else
# blank => tests/run_ep.py picks the phase default (decode small / prefill large).
cx_ep_ladder() {
  if [ -n "${CX_TOKENS_LADDER:-}" ]; then printf '%s' "$CX_TOKENS_LADDER"
  elif [ -n "${CX_TOKENS_PER_RANK:-}" ]; then printf '%s' "$CX_TOKENS_PER_RANK"
  else printf ''; fi
}

# run_ep_suite <backend: deepep|mori>
# One tests/run_ep.py invocation per phase (decode/prefill/both); dispatch and
# combine are timed separately inside it. One JSON per (backend, phase).
run_ep_suite() {
  local backend="$1" phase phases ladder rc=0
  ladder="$(cx_ep_ladder)"
  phases="${CX_PHASE:-decode}"
  [ "$phases" = "both" ] && phases="decode prefill"
  for phase in $phases; do
    cx_log "ep backend=$backend phase=$phase ngpus=$CX_NGPUS ladder='${ladder:-<phase-default>}'"
    # Hard wall-clock guard: a wedged collective (e.g. a backend that hangs at a shape)
    # must FAIL FAST, never burn the whole job timeout. timeout -k sends SIGKILL after
    # a grace period. Override with CX_RUN_TIMEOUT (seconds).
    if ! timeout -k 30 "${CX_RUN_TIMEOUT:-900}" \
        torchrun --nproc_per_node="$CX_NGPUS" tests/run_ep.py --backend "$backend" \
        --phase "$phase" --tokens-ladder "$ladder" --mode "${CX_MODE:-normal}" \
        --hidden "${CX_HIDDEN:-7168}" --topk "${CX_TOPK:-8}" --experts "${CX_EXPERTS:-256}" \
        --dispatch-dtype "${CX_DISPATCH_DTYPE:-bf16}" --routing "${CX_ROUTING:-uniform}" \
        --num-sms "${CX_NUM_SMS:-24}" --seed "${CX_SEED:-67}" --iters "${CX_ITERS:-200}" \
        --resource-mode "${CX_RESOURCE_MODE:-normalized}" --sm-fraction "${CX_SM_FRACTION:-0.18}" \
        --runner "$CX_RUNNER" --topology-class "$CX_TOPO" --transport "$CX_TRANSPORT" \
        --env-json "$ENVJSON" --out "results/${CX_RUNNER}_${backend}_${phase}_${CX_TS}.json"; then
      cx_log "WARN: $backend $phase run failed/timed out (CX_RUN_TIMEOUT=${CX_RUN_TIMEOUT:-900}s)"; rc=1
    fi
  done
  return "$rc"
}

run_deepep_suite() {
  # DeepEP is not bundled in the multi-arch image. Try to import; if absent,
  # attempt rebuild-deepep (srt-slurm setup script). Inability to run is a
  # failure, not a silent skip — the caller asked for deepep.
  if ! python3 -c "import deep_ep" 2>/dev/null; then
    if command -v rebuild-deepep.sh >/dev/null 2>&1; then
      cx_log "building DeepEP via rebuild-deepep.sh"
      rebuild-deepep.sh >&2 || { cx_log "WARN: rebuild-deepep.sh failed"; return 1; }
    else
      cx_log "WARN: deep_ep not importable and no rebuild-deepep.sh on PATH; cannot run deepep"
      return 1
    fi
  fi
  run_ep_suite deepep
}

run_mori_suite() {
  # MoRI (AMD ROCm EP), bundled in the AMD MoRI image. If absent this is a
  # failure (MoRI is not rebuildable here), not a silent skip. Single-node
  # 8x MI355X over XGMI; torch.cuda maps onto ROCm/HIP.
  if ! python3 -c "import mori" 2>/dev/null; then
    cx_log "WARN: mori not importable — needs the AMD MoRI image (rocm/sgl-dev:...-mori-...); cannot run mori"
    return 1
  fi
  run_ep_suite mori
}

rc=0
case "$CX_BENCH" in
  nccl)   run_nccl_suite || rc=1 ;;
  deepep) run_deepep_suite || rc=1 ;;
  mori)   run_mori_suite || rc=1 ;;
  all)    run_nccl_suite || rc=1; run_deepep_suite || rc=1 ;;
  *)      cx_die "unknown CX_BENCH=$CX_BENCH (want nccl|deepep|mori|all)" ;;
esac

# Summary table for the log; also fails the job if no valid results were produced.
python3 summarize.py --results-dir results --runner "$CX_RUNNER" --ts "$CX_TS" || rc=1
exit "$rc"
