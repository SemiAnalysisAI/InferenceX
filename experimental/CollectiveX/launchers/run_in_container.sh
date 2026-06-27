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

# CX_TIMING="iters:trials:warmup" unpacks into the individual knobs (one workflow input feeds three,
# since GitHub caps workflow_dispatch at 25 inputs). Blank fields keep their defaults. Used for the
# MoRI/MI355X large-T probe (e.g. "8:1:4" — minimal sustained load to dodge the wedge).
if [ -n "${CX_TIMING:-}" ]; then
  _ti="${CX_TIMING%%:*}"; _rest="${CX_TIMING#*:}"; _tt="${_rest%%:*}"; _tw="${_rest#*:}"
  [ -n "$_ti" ] && [ "$_ti" != "$CX_TIMING" ] && export CX_ITERS="$_ti"
  [ -n "$_tt" ] && [ "$_tt" != "$_rest" ] && export CX_TRIALS="$_tt"
  [ -n "$_tw" ] && [ "$_tw" != "$_rest" ] && export CX_WARMUP="$_tw"
  cx_log "CX_TIMING=$CX_TIMING -> iters=${CX_ITERS:-200} trials=${CX_TRIALS:-3} warmup=${CX_WARMUP:-32}"
fi

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

# Canonical workload staging (goal P1 "official" cohort). make_workloads.py is DETERMINISTIC, so
# every SKU/backend generates byte-identical serialized traces in-container => identical workload_id
# + checksum => proven cross-hardware workload identity with NO shared filesystem. When CX_CANONICAL=1
# (and CX_WORKLOAD_DIR not already provided) we generate the routing's traces for the run's ladder
# into a NON-results dir (.cx_workloads/ — so the *.manifest.json never pollute the results glob) and
# point run_ep at it. A canonical-serialized run with full GHA provenance is publication 'official'.
cx_stage_canonical() {
  [ "${CX_CANONICAL:-0}" = "1" ] || return 0
  [ -n "${CX_WORKLOAD_DIR:-}" ] && return 0
  local dir="$PWD/.cx_workloads"
  local ladder; ladder="$(cx_ep_ladder)"
  # cover both phase ladders when none is given, so either phase finds its files.
  [ -z "$ladder" ] && ladder="1 2 4 8 16 32 64 128 256 512 1024 2048 4096"
  cx_log "staging canonical workloads (routing=${CX_ROUTING:-uniform} ep=$CX_NGPUS ladder='$ladder')"
  python3 tests/make_workloads.py --out-dir "$dir" --routing "${CX_ROUTING:-uniform}" \
    --ep "$CX_NGPUS" --hidden "${CX_HIDDEN:-7168}" --topk "${CX_TOPK:-8}" \
    --experts "${CX_EXPERTS:-256}" --seed "${CX_SEED:-67}" --tokens-ladder "$ladder" \
    || { cx_log "WARN: canonical workload staging failed — falling back to seeded-runtime"; return 0; }
  export CX_WORKLOAD_DIR="$dir"
  cx_log "canonical workloads staged at $dir"
}

# run_ep_suite <backend: deepep|mori>
# One tests/run_ep.py invocation per phase (decode/prefill/both); dispatch and
# combine are timed separately inside it. One JSON per (backend, phase).
# Preserve a FAILED case as a classified record (goal immediate P2 "preserve failed cases in
# aggregation") so a wedge/timeout/crash becomes a bounded artifact in results/ (uploaded + surfaced
# by the plot/validator) instead of vanishing. Uses tests/failure_taxonomy.py for the mode.
emit_failed_case() {  # backend phase rc
  python3 - "$1" "$2" "$3" "$CX_RUNNER" "$CX_TOPO" \
    "results/failed_${CX_RUNNER}_${1}_${2}_${CX_TS}.json" <<'PY' || true
import sys, json, os
sys.path.insert(0, "tests")
import failure_taxonomy as ft
backend, phase, rc, runner, topo, out = sys.argv[1:7]
rec = {"family": "moe", "record_type": "failed-case", "schema_version": 3,
       "generated_by": "run_in_container.sh", "runner": runner, "backend": backend,
       "phase": phase, "topology_class": topo, "status": "failed",
       "publication_status": "failed", "rows": [],
       "failure": ft.record(rc=int(rc), case={"backend": backend, "phase": phase,
                            "dispatch_dtype": os.environ.get("CX_DISPATCH_DTYPE", "bf16"),
                            "mode": os.environ.get("CX_MODE", "normal"),
                            "contract": os.environ.get("CX_MEASUREMENT_CONTRACT", "layout-and-dispatch-v1"),
                            "routing": os.environ.get("CX_ROUTING", "uniform")})}
json.dump(rec, open(out, "w"), indent=2)
print(f"preserved failed-case record ({rec['failure']['failure_mode']}) -> {out}")
PY
}

run_ep_suite() {
  local backend="$1" phase phases ladder rc=0 rc_run
  ladder="$(cx_ep_ladder)"
  phases="${CX_PHASE:-decode}"
  [ "$phases" = "both" ] && phases="decode prefill"
  cx_stage_canonical || true   # sets CX_WORKLOAD_DIR when CX_CANONICAL=1 (official cohort)
  for phase in $phases; do
    cx_log "ep backend=$backend phase=$phase ngpus=$CX_NGPUS ladder='${ladder:-<phase-default>}'"
    # Hard wall-clock guard: a wedged collective (e.g. a backend that hangs at a shape)
    # must FAIL FAST, never burn the whole job timeout. timeout -k sends SIGKILL after
    # a grace period. Override with CX_RUN_TIMEOUT (seconds).
    timeout -k 30 "${CX_RUN_TIMEOUT:-900}" \
        torchrun --nproc_per_node="$CX_NGPUS" tests/run_ep.py --backend "$backend" \
        --phase "$phase" --tokens-ladder "$ladder" --mode "${CX_MODE:-normal}" \
        --hidden "${CX_HIDDEN:-7168}" --topk "${CX_TOPK:-8}" --experts "${CX_EXPERTS:-256}" \
        --dispatch-dtype "${CX_DISPATCH_DTYPE:-bf16}" --routing "${CX_ROUTING:-uniform}" \
        ${CX_EPLB:+--eplb} ${CX_WORKLOAD_DIR:+--workload-dir "$CX_WORKLOAD_DIR"} \
        --num-sms "${CX_NUM_SMS:-24}" --seed "${CX_SEED:-67}" --iters "${CX_ITERS:-200}" \
        --trials "${CX_TRIALS:-3}" --warmup "${CX_WARMUP:-32}" \
        --measurement-contract "${CX_MEASUREMENT_CONTRACT:-layout-and-dispatch-v1}" \
        --resource-mode "${CX_RESOURCE_MODE:-normalized}" --sm-fraction "${CX_SM_FRACTION:-0.18}" \
        --activation-profile "${CX_ACTIVATION_PROFILE:-normal}" --placement "${CX_PLACEMENT:-packed}" \
        --routing-step "${CX_ROUTING_STEP:-0}" --uneven-tokens "${CX_UNEVEN_TOKENS:-none}" \
        --combine-dtype "${CX_COMBINE_DTYPE:-bf16}" --combine-quant-mode "${CX_COMBINE_QUANT_MODE:-none}" \
        ${CX_WAIVE_ANOMALY:+--waive-anomaly} \
        --runner "$CX_RUNNER" --topology-class "$CX_TOPO" --transport "$CX_TRANSPORT" \
        --env-json "$ENVJSON" --out "results/${CX_RUNNER}_${backend}_${phase}_${CX_TS}.json"
    rc_run=$?
    if [ "$rc_run" != 0 ]; then
      cx_log "WARN: $backend $phase run failed/timed out rc=$rc_run (CX_RUN_TIMEOUT=${CX_RUN_TIMEOUT:-900}s)"
      emit_failed_case "$backend" "$phase" "$rc_run"   # preserve the classified failed case
      rc=1
    fi
  done
  return "$rc"
}

# Build DeepEP V2 (NCCL Gin backend) from source, overriding the image's bundled V1 (1.2.1).
# V2 needs NCCL>=2.30.4 (symmetric memory) STRICTLY matching the NCCL torch loads, and builds JIT
# (no precompile). arch 9.0 for Hopper (H100/H200), 10.0 for Blackwell (B300/GB300). Best-effort:
# on failure the deepep run still fails loudly (preserved failed-case), never a silent V1 fallback.
cx_build_deepep_v2() {
  local arch="9.0"; case "$CX_RUNNER" in b300*|gb300*) arch="10.0";; esac
  cx_log "DeepEP V2: building from source (TORCH_CUDA_ARCH_LIST=$arch) — overrides bundled V1"
  # PEP 668: newer images (H200/B300) ship an externally-managed Python that refuses `pip install`.
  # PIP_BREAK_SYSTEM_PACKAGES is honored by pip>=23.0.1 and silently ignored by older pip (H100),
  # so this is safe across every image; --break-system-packages as a flag would error on old pip.
  export PIP_BREAK_SYSTEM_PACKAGES=1
  pip install -q "nvidia-nccl-cu13>=2.30.4" >&2 2>&1 || cx_log "WARN: nvidia-nccl-cu13 install warning"
  rm -rf /tmp/DeepEP_v2
  git clone --depth 1 https://github.com/deepseek-ai/DeepEP /tmp/DeepEP_v2 >&2 2>&1 \
    || { cx_log "ERROR: DeepEP V2 git clone failed (compute-node network?)"; return 1; }
  export DEEPEP_COMMIT="v2-$(git -C /tmp/DeepEP_v2 rev-parse --short HEAD 2>/dev/null || echo main)"
  ( cd /tmp/DeepEP_v2 && TORCH_CUDA_ARCH_LIST="$arch" MAX_JOBS=16 \
      pip install -q --no-build-isolation --force-reinstall . ) >&2 2>&1 \
    || { cx_log "ERROR: DeepEP V2 build/install failed (arch=$arch; NCCL/toolchain?)"; return 1; }
  python3 -c "import deep_ep; print('built deep_ep', getattr(deep_ep,'__version__','?'))" >&2 \
    || { cx_log "ERROR: DeepEP V2 import failed after build (NCCL version mismatch?)"; return 1; }
  cx_log "DeepEP V2 ready ($DEEPEP_COMMIT)"
}

# UCCL EP (uccl.ep.Buffer is a DeepEP-API clone). The prebuilt wheel is cu12; on a cu13
# image its kernels need a cu12 CUDA runtime on LD_LIBRARY_PATH (probe-confirmed). PEP-668
# images need PIP_BREAK_SYSTEM_PACKAGES. Best-effort; failure to import fails loudly.
cx_build_uccl() {
  cx_log "UCCL EP: pip install uccl + cu12 runtime shim"
  export PIP_BREAK_SYSTEM_PACKAGES=1
  pip install -q uccl >&2 2>&1 || { cx_log "ERROR: pip install uccl failed"; return 1; }
  pip install -q nvidia-cuda-runtime-cu12 >&2 2>&1 || cx_log "WARN: nvidia-cuda-runtime-cu12 warning"
  local cu12lib
  cu12lib="$(python3 -c "import nvidia.cuda_runtime as m, os; print(os.path.join(os.path.dirname(m.__file__),'lib'))" 2>/dev/null)"
  [ -n "$cu12lib" ] && export LD_LIBRARY_PATH="$cu12lib:${LD_LIBRARY_PATH:-}"
  export UCCL_COMMIT="pkg-$(python3 -c 'import importlib.metadata as m; print(m.version("uccl"))' 2>/dev/null || echo uccl)"
  # import torch FIRST: uccl.ep's C extension links libc10.so (torch), which is only on the loader
  # path once torch is imported (rpath). The adapter (ep_uccl.py) imports torch before uccl.ep too.
  python3 -c "import torch; from uccl.ep import Buffer; print('uccl.ep ready')" >&2 \
    || { cx_log "ERROR: uccl.ep import failed (cu12 runtime on LD_LIBRARY_PATH?)"; return 1; }
  cx_log "UCCL EP ready ($UCCL_COMMIT)"
}

run_deepep_suite() {
  # CX_DEEPEP_V2=1 -> build the V2 (NCCL Gin) kernels from source first (Hopper+Blackwell only).
  if [ "${CX_DEEPEP_V2:-0}" = "1" ]; then
    cx_build_deepep_v2 || { cx_log "WARN: DeepEP V2 setup failed — cannot run V2"; return 1; }
  fi
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

run_uccl_suite() {
  # UCCL EP (NVIDIA) — DeepEP-API clone; build the wheel + cu12 shim, then reuse the generic
  # EP sweep (run_ep.py --backend uccl). Inability to install/import is a failure, not a skip.
  cx_build_uccl || { cx_log "WARN: UCCL EP setup failed — cannot run uccl"; return 1; }
  run_ep_suite uccl
}

run_collective_bench() {
  # Single-process host/GPU memcpy-family collectives (NOT torchrun): CPU-GPU offload,
  # copy-engine/SDMA, KV-cache transfer. Each emits one family-tagged JSON like run_nccl.py.
  local kind="$1" script out rc=0
  case "$kind" in
    offload)     script="tests/offload_bench.py";    out="results/${CX_RUNNER}_offload_${CX_TS}.json" ;;
    copy-engine) script="tests/copy_engine_bench.py"; out="results/${CX_RUNNER}_copy_engine_${CX_TS}.json" ;;
    kv-cache)    script="tests/kv_cache_transfer.py"; out="results/${CX_RUNNER}_kvcache_${CX_TS}.json" ;;
    *) cx_die "unknown collective kind '$kind'" ;;
  esac
  cx_log "collective bench=$kind -> $out"
  local extra=""; [ "$kind" = "kv-cache" ] && extra="--direction all"
  # shellcheck disable=SC2086
  timeout -k 30 "${CX_RUN_TIMEOUT:-900}" python3 "$script" $extra \
      --runner "$CX_RUNNER" --topology-class "$CX_TOPO" --transport "${CX_TRANSPORT:-nvlink}" \
      --env-json "$ENVJSON" --out "$out" || rc=$?
  [ "$rc" = 0 ] || cx_log "WARN: collective $kind failed/timed out rc=$rc"
  return "$rc"
}

rc=0
case "$CX_BENCH" in
  nccl)        run_nccl_suite || rc=1 ;;
  deepep)      run_deepep_suite || rc=1 ;;
  mori)        run_mori_suite || rc=1 ;;
  uccl)        run_uccl_suite || rc=1 ;;
  offload)     run_collective_bench offload || rc=1 ;;
  copy-engine) run_collective_bench copy-engine || rc=1 ;;
  kv-cache)    run_collective_bench kv-cache || rc=1 ;;
  all)         run_nccl_suite || rc=1; run_deepep_suite || rc=1 ;;
  *)           cx_die "unknown CX_BENCH=$CX_BENCH (want nccl|deepep|mori|uccl|offload|copy-engine|kv-cache|all)" ;;
esac

# Summary table for the log; also fails the job if no valid results were produced.
python3 summarize.py --results-dir results --runner "$CX_RUNNER" --ts "$CX_TS" || rc=1
exit "$rc"
