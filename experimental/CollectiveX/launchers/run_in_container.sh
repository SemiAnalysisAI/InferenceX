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
# Selector:        CX_BENCH = nccl | deepep | all          (default nccl)
# NCCL knobs:      CX_OPS, CX_MIN_BYTES, CX_MAX_BYTES, CX_TRANSPORT, CX_NCCL_HOME
# DeepEP knobs:    CX_TOKENS_PER_RANK CX_HIDDEN CX_TOPK CX_EXPERTS CX_DISPATCH_DTYPE
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
  local build ops op sfail=0
  build="$(cx_build_nccl_tests "$PWD/.nccl-tests" 0)" || return 1   # single-node: MPI=0, -g N
  ops="${CX_OPS:-all_reduce all_gather reduce_scatter alltoall}"
  for op in $ops; do
    if ! python3 run_nccl.py --op "$op" --nccl-tests-dir "$build" \
        --world-size "$CX_NGPUS" --nodes 1 --gpus-per-proc "$CX_NGPUS" \
        --runner "$CX_RUNNER" --topology-class "$CX_TOPO" --transport "$CX_TRANSPORT" \
        --env-json "$ENVJSON" --out "results/${CX_RUNNER}_${op}_${CX_TS}.json" \
        --min-bytes "${CX_MIN_BYTES:-8}" --max-bytes "${CX_MAX_BYTES:-8G}" --check 1; then
      cx_log "WARN: nccl $op failed or invalid"; sfail=1
    fi
  done
  return "$sfail"
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
  torchrun --nproc_per_node="$CX_NGPUS" run_deepep.py \
    --runner "$CX_RUNNER" --topology-class "$CX_TOPO" --transport "$CX_TRANSPORT" \
    --tokens-per-rank "${CX_TOKENS_PER_RANK:-64}" --hidden "${CX_HIDDEN:-7168}" \
    --topk "${CX_TOPK:-8}" --experts "${CX_EXPERTS:-256}" \
    --dispatch-dtype "${CX_DISPATCH_DTYPE:-bf16}" \
    --env-json "$ENVJSON" --out "results/${CX_RUNNER}_deepep_${CX_TS}.json" \
    || { cx_log "WARN: deepep run failed"; return 1; }
}

rc=0
case "$CX_BENCH" in
  nccl)   run_nccl_suite || rc=1 ;;
  deepep) run_deepep_suite || rc=1 ;;
  all)    run_nccl_suite || rc=1; run_deepep_suite || rc=1 ;;
  *)      cx_die "unknown CX_BENCH=$CX_BENCH (want nccl|deepep|all)" ;;
esac

# Summary table for the log; also fails the job if no valid results were produced.
python3 summarize.py --results-dir results --runner "$CX_RUNNER" --ts "$CX_TS" || rc=1
exit "$rc"
