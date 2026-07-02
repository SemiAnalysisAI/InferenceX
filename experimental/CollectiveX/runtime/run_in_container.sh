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
# shellcheck source=../runtime/common.sh
source runtime/common.sh
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
  # CROSS-NODE EP (goal 182): when CX_NNODES>1 (set per-node by a multi-node launcher with
  # CX_NODE_RANK + CX_RDZV_FILE) we span CX_NNODES*CX_NGPUS ranks over the inter-node fabric. We do
  # NOT use torchrun: its elastic agent runs its OWN cross-node TCPStore at --master-addr, which is
  # unreachable from a peer rank's enroot container net namespace (the management-subnet NodeAddr is
  # not in the container's net view — torchrun timed out 900s at exactly that bootstrap). Instead each
  # node spawns its NGPUS local ranks directly (global RANK = CX_NODE_RANK*NGPUS + local) and they
  # rendezvous via a FileStore on the compute-visible shared mount (CX_RDZV_FILE, consumed by
  # run_ep.py), so NCCL exchanges its unique-id through the shared file and connects peers over IB.
  local xnode=0
  if [ -n "${CX_NNODES:-}" ] && [ "${CX_NNODES}" -gt 1 ]; then
    xnode=1
    # shellcheck source=_xnode_net.sh
    source runtime/_xnode_net.sh 2>/dev/null || true
    : "${CX_RDZV_FILE:=$PWD/.rdzv_${CX_TS}}"; export CX_RDZV_FILE
    cx_log "cross-node EP: nnodes=$CX_NNODES node_rank=${CX_NODE_RANK:-0} world=$((CX_NNODES*CX_NGPUS)) rdzv=file://$CX_RDZV_FILE (no torchrun agent)"
  fi
  for phase in $phases; do
    cx_log "ep backend=$backend phase=$phase ngpus=$CX_NGPUS ladder='${ladder:-<phase-default>}'"
    local out="results/${CX_RUNNER}_${backend}_${phase}_${CX_TS}.json"
    # Common run_ep.py args (shared by single-node torchrun + cross-node local-spawn).
    local -a EPARGS=(--backend "$backend" --phase "$phase" --tokens-ladder "$ladder" --mode "${CX_MODE:-normal}"
      --hidden "${CX_HIDDEN:-7168}" --topk "${CX_TOPK:-8}" --experts "${CX_EXPERTS:-256}"
      --dispatch-dtype "${CX_DISPATCH_DTYPE:-bf16}" --routing "${CX_ROUTING:-uniform}"
      --num-sms "${CX_NUM_SMS:-24}" --seed "${CX_SEED:-67}" --iters "${CX_ITERS:-200}"
      --trials "${CX_TRIALS:-3}" --warmup "${CX_WARMUP:-32}"
      --measurement-contract "${CX_MEASUREMENT_CONTRACT:-layout-and-dispatch-v1}"
      --resource-mode "${CX_RESOURCE_MODE:-normalized}" --sm-fraction "${CX_SM_FRACTION:-0.18}"
      --activation-profile "${CX_ACTIVATION_PROFILE:-normal}" --placement "${CX_PLACEMENT:-packed}"
      --routing-step "${CX_ROUTING_STEP:-0}" --uneven-tokens "${CX_UNEVEN_TOKENS:-none}"
      --combine-dtype "${CX_COMBINE_DTYPE:-bf16}" --combine-quant-mode "${CX_COMBINE_QUANT_MODE:-none}"
      --runner "$CX_RUNNER" --topology-class "$CX_TOPO" --transport "$CX_TRANSPORT"
      --env-json "$ENVJSON" --out "$out")
    [ -n "${CX_EPLB:-}" ] && EPARGS+=(--eplb)
    [ -n "${CX_WORKLOAD_DIR:-}" ] && EPARGS+=(--workload-dir "$CX_WORKLOAD_DIR")
    [ -n "${CX_WAIVE_ANOMALY:-}" ] && EPARGS+=(--waive-anomaly)
    # Hard wall-clock guard: a wedged collective must FAIL FAST (timeout -k SIGKILLs after grace).
    if [ "$xnode" = 1 ]; then
      # Cross-node: spawn NGPUS local ranks, FileStore rendezvous (no torchrun agent). Only the global
      # rank 0 writes --out; the rest participate in the collectives. wait collects every rank's rc.
      local base=$(( ${CX_NODE_RANK:-0} * CX_NGPUS )) world=$(( CX_NNODES * CX_NGPUS )) i; local -a pids=()
      for i in $(seq 0 $((CX_NGPUS - 1))); do
        RANK=$((base + i)) LOCAL_RANK="$i" WORLD_SIZE="$world" \
          timeout -k 30 "${CX_RUN_TIMEOUT:-900}" python3 tests/run_ep.py "${EPARGS[@]}" &
        pids+=($!)
      done
      rc_run=0; for i in "${pids[@]}"; do wait "$i" || rc_run=$?; done
    else
      # shellcheck disable=SC2086
      timeout -k 30 "${CX_RUN_TIMEOUT:-900}" \
          torchrun --nproc_per_node="$CX_NGPUS" tests/run_ep.py "${EPARGS[@]}"
      rc_run=$?
    fi
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
# (no precompile). arch 9.0 for Hopper (H100/H200), 10.0 for Blackwell (B300/B200/GB300). Best-effort:
# on failure the deepep run still fails loudly (preserved failed-case), never a silent V1 fallback.
cx_build_deepep_v2() {
  # IDEMPOTENT: SHARD mode calls dispatch_bench (hence this) once PER CASE. Build once per allocation,
  # then skip — else a 60-case shard re-runs the from-source build 60x (force-reinstall) and blows the
  # slurm --time. Sentinel lives in the container fs (persists across the x86 in-container case loop).
  [ -f /tmp/.cx_built_deepep_v2 ] && { cx_log "DeepEP V2 already built this allocation — skip rebuild"; return 0; }
  local arch="9.0"; case "$CX_RUNNER" in b300*|gb300*|b200*) arch="10.0";; esac
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
  : > /tmp/.cx_built_deepep_v2   # sentinel: skip rebuild on subsequent cases in this allocation
  cx_log "DeepEP V2 ready ($DEEPEP_COMMIT)"
}

# Build the DeepEP `hybrid-ep` branch (NVIDIA's TMA-based impl: HybridEPBuffer, intranode NVLink +
# internode RDMA/NIXL). Three container-specific fixes, all probe-confirmed on the B300 sglang image:
#   1. CUDA-13 moved cccl/libcudacxx headers to <cuda>/include/cccl/ (not on nvcc's default path) —
#      its nvshmem_tensor.h #includes <cuda/std/tuple> -> add that dir via CPATH/NVCC_PREPEND_FLAGS.
#   2. The final link wants -l:libnvshmem_host.so but the bundled nvshmem ships only .so.3 -> create
#      the unversioned symlink.
#   3. NVSHMEM_DIR set to the bundled nvshmem enables build; unset => intranode-only (internode/LL off).
# Intranode HybridEPBuffer (single NVLink domain, <=8 ranks) needs no multi-node/NVSHMEM bring-up.
cx_build_deepep_hybrid() {
  [ -f /tmp/.cx_built_deepep_hybrid ] && { cx_log "hybrid-ep already built this allocation — skip rebuild"; return 0; }
  local arch="9.0"; case "$CX_RUNNER" in b300*|gb300*|b200*) arch="10.0";; esac
  cx_log "DeepEP hybrid-ep: building NVIDIA TMA branch from source (TORCH_CUDA_ARCH_LIST=$arch)"
  export PIP_BREAK_SYSTEM_PACKAGES=1
  export NVSHMEM_DIR="$(python3 -c 'import os,nvidia.nvshmem as n; print(os.path.dirname(n.__file__))' 2>/dev/null || echo /usr/local/lib/python3.12/dist-packages/nvidia/nvshmem)"
  local cccl; cccl="$(echo /usr/local/cuda*/targets/*/include/cccl | awk '{print $1}')"
  [ -d "$cccl" ] && { export CPATH="$cccl:${CPATH:-}"; export NVCC_PREPEND_FLAGS="-I$cccl ${NVCC_PREPEND_FLAGS:-}"; }
  [ -e "$NVSHMEM_DIR/lib/libnvshmem_host.so.3" ] && ln -sf libnvshmem_host.so.3 "$NVSHMEM_DIR/lib/libnvshmem_host.so" 2>/dev/null || true
  export LD_LIBRARY_PATH="$NVSHMEM_DIR/lib:${LD_LIBRARY_PATH:-}"
  rm -rf /tmp/DeepEP_hybrid
  git clone --depth 1 --branch hybrid-ep https://github.com/deepseek-ai/DeepEP /tmp/DeepEP_hybrid >&2 2>&1 \
    || { cx_log "ERROR: hybrid-ep git clone failed"; return 1; }
  export DEEPEP_COMMIT="hybrid-$(git -C /tmp/DeepEP_hybrid rev-parse --short HEAD 2>/dev/null || echo hybrid-ep)"
  # Install into SITE-PACKAGES so the build persists across srun steps in the pyxis named container. The
  # EP8 multi-srun runs the build-once and each case as SEPARATE srun steps; only the container rootfs
  # (site-packages) persists — /tmp does NOT. The old `build_ext --inplace` under /tmp/DeepEP_hybrid +
  # PYTHONPATH worked for the EP4 single-node path (build+run share one process) but was LOST at EP8,
  # giving `module deep_ep has no attribute HybridEPBuffer`. pip install mirrors deepep-v2 (which persists
  # correctly at EP8). Fall back to in-place build (EP4 single-node only) if this branch can't plain-install.
  if ( cd /tmp/DeepEP_hybrid && TORCH_CUDA_ARCH_LIST="$arch" MAX_JOBS=16 \
        pip install -q --no-build-isolation --force-reinstall . ) >&2 2>&1; then
    cx_log "hybrid-ep installed into site-packages (persists across srun steps)"
  else
    cx_log "WARN: hybrid-ep pip install failed — falling back to build_ext --inplace (EP4 single-node only)"
    ( cd /tmp/DeepEP_hybrid && TORCH_CUDA_ARCH_LIST="$arch" MAX_JOBS=16 python3 setup.py build_ext --inplace ) >&2 2>&1 \
      || { cx_log "ERROR: hybrid-ep build failed (arch=$arch; cccl/nvshmem?)"; return 1; }
    export PYTHONPATH="/tmp/DeepEP_hybrid:${PYTHONPATH:-}"
  fi
  python3 -c "import deep_ep; assert hasattr(deep_ep,'HybridEPBuffer'); print('built hybrid-ep deep_ep', getattr(deep_ep,'__version__','?'))" >&2 \
    || { cx_log "ERROR: hybrid-ep import / HybridEPBuffer missing after build"; return 1; }
  # nvshmem runtime libs are in site-packages (persistent); the env pointing at them is process-local, and
  # a PYTHONPATH is needed only if the in-place fallback ran. Persist both to a file the EP8 case-srun WRAP
  # sources (best-effort; with pip install the package itself is already on the default site-packages path).
  { printf 'export LD_LIBRARY_PATH=%s/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\n' "$NVSHMEM_DIR"
    [ -n "${PYTHONPATH:-}" ] && printf 'export PYTHONPATH=%s\n' "$PYTHONPATH"
  } > /tmp/.cx_hybrid_env 2>/dev/null || cx_log "WARN: could not write /tmp/.cx_hybrid_env"
  : > /tmp/.cx_built_deepep_hybrid   # sentinel: skip rebuild on subsequent cases in this allocation
  cx_log "DeepEP hybrid-ep ready ($DEEPEP_COMMIT)"
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
  # Vendor UCCL's DeepEP-API wrapper (ep/deep_ep_wrapper/deep_ep) under a NON-conflicting name
  # (uccl_deepep) so it doesn't shadow the container's real deep_ep. Its Buffer(group, num_nvl_bytes,
  # ...) takes a torch ProcessGroup (matching DeepEP + ep_uccl.py's calls) and runs the full
  # proxy/IPC-handle/runtime.sync bootstrap that the low-level uccl.ep.Buffer(rank,num_ranks) lacks.
  rm -rf /tmp/uccl_src /tmp/uccl_deepep_pkg
  # Pin the wrapper to the SAME tag as the installed wheel (pkg-0.1.1 -> v0.1.1): the wrapper's
  # dispatch calls into uccl.ep (get_rdma_buffer etc.), so a main-branch wrapper vs a 0.1.1 wheel
  # mismatches signatures. Match them.
  _uccl_tag="v$(python3 -c 'import importlib.metadata as m; print(m.version("uccl"))' 2>/dev/null || echo 0.1.1)"
  if { git clone --depth 1 --branch "$_uccl_tag" https://github.com/uccl-project/uccl /tmp/uccl_src >&2 2>&1 \
       || git clone --depth 1 https://github.com/uccl-project/uccl /tmp/uccl_src >&2 2>&1; } \
     && [ -d /tmp/uccl_src/ep/deep_ep_wrapper/deep_ep ]; then
    mkdir -p /tmp/uccl_deepep_pkg/uccl_deepep
    cp /tmp/uccl_src/ep/deep_ep_wrapper/deep_ep/*.py /tmp/uccl_deepep_pkg/uccl_deepep/ 2>/dev/null
    export PYTHONPATH="/tmp/uccl_deepep_pkg:${PYTHONPATH:-}"
    if python3 -c "import torch; from uccl_deepep import Buffer; print('uccl_deepep wrapper ready')" >&2; then
      export CX_UCCL_WRAPPER=1
    else
      cx_log "WARN: uccl_deepep wrapper import failed — falling back to low-level uccl.ep"
    fi
  else
    cx_log "WARN: uccl deep_ep_wrapper not vendored (clone/path) — low-level uccl.ep fallback"
  fi
  cx_log "UCCL EP ready ($UCCL_COMMIT, wrapper=${CX_UCCL_WRAPPER:-0})"
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
run_nccl_ep_suite() {
  # NCCL/RCCL all-to-all EP (tests/ep_nccl.py) — pure torch.distributed collectives, already in every
  # image (no build). The canonical token-shuffle EP + the only cross-node path that survives without
  # GPUDirect-RDMA: NCCL host-stages where UCCL's ibv_reg_mr / MoRI's RDMA registration abort. Works
  # cross-node via the FileStore rendezvous (CX_RDZV_FILE) on both NVIDIA (nccl) and AMD (rccl).
  run_ep_suite nccl-ep
}
run_deepep_hybrid_suite() {
  # DeepEP hybrid-ep branch (NVIDIA TMA HybridEPBuffer) — build from source (cccl + libnvshmem
  # fixes), then the generic EP sweep (run_ep.py --backend deepep-hybrid). Intranode NVLink path.
  cx_build_deepep_hybrid || { cx_log "WARN: hybrid-ep setup failed — cannot run deepep-hybrid"; return 1; }
  run_ep_suite deepep-hybrid
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

run_rl_mesh() {
  # RL trainer<->generator mesh transfer (multi-process: torchrun splits world into two meshes).
  cx_log "rl-mesh bench ngpus=$CX_NGPUS"
  timeout -k 30 "${CX_RUN_TIMEOUT:-900}" \
      torchrun --nproc_per_node="$CX_NGPUS" tests/rl_mesh_bench.py \
      --runner "$CX_RUNNER" --topology-class "$CX_TOPO" --transport "${CX_TRANSPORT:-nvlink}" \
      --env-json "$ENVJSON" --out "results/${CX_RUNNER}_rl_mesh_${CX_TS}.json"
  local rc=$?
  [ "$rc" = 0 ] || cx_log "WARN: rl-mesh failed/timed out rc=$rc"
  return "$rc"
}

run_allreduce_fw() {
  # Framework custom all-reduce (flashinfer one-shot/two-shot + sglang/vllm), multi-process torchrun.
  cx_log "allreduce-fw bench ngpus=$CX_NGPUS"
  timeout -k 30 "${CX_RUN_TIMEOUT:-900}" \
      torchrun --nproc_per_node="$CX_NGPUS" tests/allreduce_fw_bench.py \
      --runner "$CX_RUNNER" --topology-class "$CX_TOPO" --transport "${CX_TRANSPORT:-nvlink}" \
      --env-json "$ENVJSON" --out "results/${CX_RUNNER}_allreduce_fw_${CX_TS}.json"
  local rc=$?
  [ "$rc" = 0 ] || cx_log "WARN: allreduce-fw failed/timed out rc=$rc"
  return "$rc"
}

# Upgrade FlashInfer in-container to the latest wheel — the bundled 0.6.8.post1 lacks the
# quantized-COMBINE OUTPUT path (moe_a2a_combine output_dtype/output_scales, added in a newer
# release; confirmed in the main-branch source). A combine-quant run needs it; the dispatch path
# (bf16/fp8/mxfp8/nvfp4) is unaffected and stays on whatever is installed. Best-effort: a failed
# upgrade leaves the run on the bundled version (the combine-quant adapter then rejects loudly).
cx_build_flashinfer_latest() {
  [ -f /tmp/.cx_built_flashinfer ] && { cx_log "FlashInfer quant-combine build already done this allocation — skip"; return 0; }
  cx_log "FlashInfer: upgrading to latest wheel for quantized-combine output (moe_a2a_combine output_dtype)"
  export PIP_BREAK_SYSTEM_PACKAGES=1
  # moe_a2a_combine output_dtype is on flashinfer MAIN but NOT in the latest PyPI release (0.6.13) —
  # so `pip -U flashinfer-python` (PyPI) is insufficient. Install from the NIGHTLY wheel index
  # (built from main): flashinfer-python (--no-deps; the container already has torch etc.) + the
  # matching cubin + cu130 jit-cache. FLASHINFER_DISABLE_VERSION_CHECK=1 bypasses any residual
  # sub-package skew. Falls back to a PyPI -U (which then asserts-out cleanly if it lacks output_dtype).
  export FLASHINFER_DISABLE_VERSION_CHECK=1
  local before after NIDX="https://flashinfer.ai/whl/nightly"
  before="$(python3 -c 'import flashinfer;print(flashinfer.__version__)' 2>/dev/null || echo none)"
  { pip install -q -U --pre flashinfer-python --index-url "$NIDX/" --no-deps >&2 2>&1 \
      && pip install -q -U --pre flashinfer-cubin --index-url "$NIDX/" >&2 2>&1 \
      && pip install -q -U --pre flashinfer-jit-cache --index-url "$NIDX/cu130" >&2 2>&1; } \
    || { cx_log "WARN: flashinfer nightly index failed — falling back to PyPI -U"; \
         pip install -q -U flashinfer-python flashinfer-cubin flashinfer-jit-cache >&2 2>&1 || true; }
  # The nightly (main) flashinfer's CuTe-DSL kernels import newer cutlass.cute symbols (e.g.
  # OperandMajorMode) than the bundled nvidia-cutlass-dsl provides — upgrade it to match (PyPI).
  pip install -q -U nvidia-cutlass-dsl >&2 2>&1 || cx_log "WARN: nvidia-cutlass-dsl upgrade warning"
  # The cu130 nightly WHEEL (0.6.13.dev20260612) still predates the combine output_dtype PR — if it's
  # absent, build flashinfer MAIN from source (the container has the cu130 toolchain that built
  # deep_ep-v2 + hybrid-ep; cutlass-dsl 4.5.2 is now installed; JIT-first build, time-boxed).
  if ! python3 -c "import inspect, flashinfer.comm as c; assert 'output_dtype' in str(inspect.signature(c.MoeAlltoAll.combine))" 2>/dev/null; then
    cx_log "FlashInfer nightly wheel lacks combine output_dtype — building flashinfer main from source"
    # Uninstall the precompiled cubin + jit-cache FIRST: they ship the OLD 10-arg moe_a2a_combine
    # kernel, which the main Python wrapper (14-arg, with output_dtype) then mis-calls ("Expected 10
    # but got 14 arguments"). Removing them forces get_moe_alltoall_module() to JIT-compile the
    # kernel FRESH from main's csrc at runtime (14-arg, matching the wrapper).
    pip uninstall -y flashinfer-cubin flashinfer-jit-cache >&2 2>&1 || true
    rm -rf /tmp/fi_main ~/.cache/flashinfer 2>/dev/null || true
    if git clone --recursive --depth 1 https://github.com/flashinfer-ai/flashinfer.git /tmp/fi_main >&2 2>&1; then
      ( cd /tmp/fi_main && timeout 2400 pip install -q --no-build-isolation . >&2 2>&1 ) \
        || cx_log "WARN: flashinfer main source build failed/timed out"
    else
      cx_log "WARN: flashinfer main clone failed (compute-node network?)"
    fi
  fi
  after="$(python3 -c 'import flashinfer;print(flashinfer.__version__)' 2>/dev/null || echo none)"
  export FLASHINFER_COMMIT="pkg-$after"
  # Record the EXACT upgraded library stack for reproducibility — the upgrade happens AFTER
  # env_capture, so these versions live nowhere else. CX_FLASHINFER_STACK is read into the result's
  # backend_provenance by ep_flashinfer. Also logged to the GHA log even if the run later fails.
  export CX_FLASHINFER_STACK="$(python3 - <<'PY' 2>/dev/null || echo 'capture-failed'
import importlib.metadata as m
def v(p):
    try: return m.version(p)
    except Exception: return "absent"
pkgs=["flashinfer-python","flashinfer-cubin","flashinfer-jit-cache","nvidia-cutlass-dsl","torch"]
print(" ".join(f"{p}={v(p)}" for p in pkgs))
PY
)"
  cx_log "FlashInfer upgrade (nightly): $before -> $after"
  cx_log "FlashInfer stack: $CX_FLASHINFER_STACK"
  python3 -c "import inspect, flashinfer.comm as c; assert 'output_dtype' in str(inspect.signature(c.MoeAlltoAll.combine)), 'combine still has no output_dtype'; print('combine output_dtype: present')" >&2 \
    || { cx_log "ERROR: upgraded FlashInfer combine still lacks output_dtype — cannot quant-combine"; return 1; }
  : > /tmp/.cx_built_flashinfer   # sentinel: skip rebuild on subsequent cases in this allocation
}

# NIXL device-EP build-probe — the gated EP item (goal "NIXL EP"). The OLD sglang image blocked the
# meson build on Abseil 20220623; this runs in the dynamo tensorrtllm-runtime image (container switch)
# and reports whether THIS container clears it. Reports the build deps the meson tree needs (nixl lib,
# Abseil, meson/ninja/ucx) then attempts `meson setup` (which enumerates any missing dep) + a
# time-boxed compile. Informational: logs the precise outcome; never fails the suite (the transfer
# bench is the guaranteed datapoint). If it SUCCEEDS we wire ep_nixl.py against nixl_ep_cpp next.
cx_probe_nixl_ep() {
  cx_log "NIXL device-EP build-probe (gated EP item — does examples/device/ep build on this container?)"
  export PIP_BREAK_SYSTEM_PACKAGES=1
  python3 - >&2 2>&1 <<'PY' || true
import importlib.metadata as m, shutil, glob
def v(p):
    try: return m.version(p)
    except Exception: return "absent"
print("NIXL_EP_PROBE deps: nixl=%s meson=%s ninja=%s pybind11=%s cmake=%s" %
      (v("nixl"), shutil.which("meson"), shutil.which("ninja"), v("pybind11"), shutil.which("cmake")))
# Abseil version was the OLD container's blocker (20220623) — report what THIS container ships.
hits = glob.glob("/usr/**/libabsl_base*", recursive=True) + glob.glob("/opt/**/libabsl_base*", recursive=True)
print("NIXL_EP_PROBE abseil libs:", hits[:4] or "not found on /usr,/opt")
try:
    import nixl, os; print("NIXL_EP_PROBE nixl at", os.path.dirname(nixl.__file__))
except Exception as e:
    print("NIXL_EP_PROBE nixl import:", repr(e))
PY
  pip install -q meson ninja pybind11 >&2 2>&1 || cx_log "NIXL_EP_PROBE: meson/ninja/pybind11 pip warn"
  # The device-EP build needs UCX's GPU device API header <ucp/api/device/ucp_device_impl.h>; the
  # dynamo image's UCX lacks it (meson "UCX GPU Device API: NO"). Build a recent UCX from source WITH
  # CUDA (ships the device-API header) and point pkg-config at it — the directive's "see if a build
  # fixes it". If the header is still absent (device-comm needs GPUDirect-Async driver support), the
  # meson reports NO again and that precise wall is documented.
  if ! find /usr /opt -name 'ucp_device_impl.h' 2>/dev/null | grep -q .; then
    cx_log "NIXL_EP_PROBE: building UCX from source with CUDA device API -> /opt/ucx-dev"
    rm -rf /tmp/ucx_src
    if git clone --depth 1 https://github.com/openucx/ucx /tmp/ucx_src >&2 2>&1; then
      ( cd /tmp/ucx_src && timeout 1300 bash -c '
          ./autogen.sh >/dev/null 2>&1
          ./configure --prefix=/opt/ucx-dev --with-cuda=/usr/local/cuda --enable-mt --without-go --without-java >/dev/null 2>&1
          make -j"$(nproc)" install 2>&1 | tail -4' ) >&2 2>&1 || cx_log "NIXL_EP_PROBE: UCX build failed/timed out"
      export PKG_CONFIG_PATH="/opt/ucx-dev/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
      export LD_LIBRARY_PATH="/opt/ucx-dev/lib:${LD_LIBRARY_PATH:-}"
    fi
    find /opt/ucx-dev -name 'ucp_device_impl.h' 2>/dev/null | head -1 | sed 's/^/NIXL_EP_PROBE built-ucx device header: /' >&2 || true
  fi
  rm -rf /tmp/nixl_src
  git clone --depth 1 https://github.com/ai-dynamo/nixl /tmp/nixl_src >&2 2>&1 \
    || { cx_log "NIXL_EP_PROBE: clone failed (compute-node network?)"; return 0; }
  # meson-setup the whole project (it now sees the source-built UCX via PKG_CONFIG_PATH -> the "UCX
  # GPU Device API" line shows YES/NO), then a time-boxed compile. tail the decisive lines to the log.
  ( cd /tmp/nixl_src && timeout 1500 bash -c '
      echo "--- meson setup ---"; meson setup build 2>&1 | tail -34
      echo "--- meson compile (time-boxed) ---"; meson compile -C build 2>&1 | tail -40
    ' ) >&2 2>&1 || true
  if find /tmp/nixl_src/build -name 'nixl_ep_cpp*.so' 2>/dev/null | grep -q .; then
    cx_log "NIXL_EP_PROBE: SUCCESS — nixl_ep_cpp built on this container (wire ep_nixl.py next)"
  else
    cx_log "NIXL_EP_PROBE: nixl_ep_cpp NOT produced — see 'meson setup' output above for the blocker"
  fi
}

run_mooncake_suite() {
  # MoonCake KV transfer (the goal's kv-cache 'mooncake' backend). Mooncake is in no CollectiveX
  # container -> pip-install mooncake-transfer-engine first (the directive's "import a new one", as a
  # pip import). Then the single-process RDMA loopback bench. Needs an RDMA NIC.
  local out="results/${CX_RUNNER}_mooncake_${CX_TS}.json" rc=0
  export PIP_BREAK_SYSTEM_PACKAGES=1
  if ! python3 -c "import mooncake.engine" 2>/dev/null; then
    cx_log "mooncake: pip install mooncake-transfer-engine"
    pip install -q mooncake-transfer-engine >&2 2>&1 || cx_log "WARN: mooncake pip install failed"
  fi
  cx_log "mooncake transfer bench -> $out"
  timeout -k 30 "${CX_RUN_TIMEOUT:-900}" python3 tests/mooncake_transfer.py \
      --runner "$CX_RUNNER" --topology-class "$CX_TOPO" --transport "${CX_TRANSPORT:-rdma}" \
      --env-json "$ENVJSON" --out "$out" || { rc=$?; cx_log "WARN: mooncake failed/timed out rc=$rc"; }
  return "$rc"
}

run_nccl_kv_suite() {
  # NCCL/RCCL KV-cache transfer (the goal's kv-cache 'nccl'/'rccl' backend). torchrun 2 ranks,
  # rank0 dist.send -> rank1 dist.recv of KV-block-sized buffers. NCCL on NVIDIA, RCCL on ROCm
  # (same torch.distributed API). Needs >=2 GPUs.
  local out="results/${CX_RUNNER}_nccl_kv_${CX_TS}.json" rc=0 np=2
  [ "$CX_NGPUS" -lt 2 ] && { cx_log "WARN: nccl-kv needs >=2 GPUs (have $CX_NGPUS)"; return 1; }
  cx_log "nccl-kv transfer bench (2-rank send/recv) -> $out"
  timeout -k 30 "${CX_RUN_TIMEOUT:-900}" \
      torchrun --nproc_per_node="$np" tests/nccl_kv_transfer.py \
      --runner "$CX_RUNNER" --topology-class "$CX_TOPO" --transport "${CX_TRANSPORT:-nvlink}" \
      --env-json "$ENVJSON" --out "$out" || { rc=$?; cx_log "WARN: nccl-kv failed/timed out rc=$rc"; }
  return "$rc"
}

run_mori_io_suite() {
  # MoRI-IO (ROCm/mori mori.io) — AMD RDMA p2p transfer engine, bundled in the AMD MoRI image. The
  # WIRED kv-cache 'mori-io' backend (a guaranteed datapoint when mori.io imports + RDMA loopback
  # works on the ionic_rdma NICs). Single process, 2 IOEngines, GPU0<->GPU1 RDMA read.
  if ! python3 -c "import mori.io" 2>/dev/null; then
    cx_log "WARN: mori.io not importable — needs the AMD MoRI image; cannot run mori-io"; return 1
  fi
  local out="results/${CX_RUNNER}_mori_io_${CX_TS}.json" rc=0
  cx_log "mori-io transfer bench -> $out"
  timeout -k 30 "${CX_RUN_TIMEOUT:-900}" python3 tests/mori_io_transfer.py \
      --runner "$CX_RUNNER" --topology-class "$CX_TOPO" --transport "${CX_TRANSPORT:-rdma}" \
      --env-json "$ENVJSON" --out "$out" || { rc=$?; cx_log "WARN: mori-io failed/timed out rc=$rc"; }
  return "$rc"
}

run_nixl_suite() {
  # NIXL (ai-dynamo/nixl) — runs in the dynamo tensorrtllm-runtime image (cx_default_image switched
  # CX_IMAGE for CX_BENCH=nixl). Two parts: (1) the NIXL point-to-point TRANSFER bench (the wired
  # KV-cache 'nixl' backend — a guaranteed datapoint when nixl imports); (2) the device-EP build-probe
  # (the gated NIXL EP item). The transfer result drives the suite's pass/fail; the probe is logged.
  local out rc=0
  out="results/${CX_RUNNER}_nixl_${CX_TS}.json"
  cx_log "nixl transfer bench -> $out"
  timeout -k 30 "${CX_RUN_TIMEOUT:-900}" python3 tests/nixl_transfer.py --direction all \
      --runner "$CX_RUNNER" --topology-class "$CX_TOPO" --transport "${CX_TRANSPORT:-nvlink}" \
      --env-json "$ENVJSON" --out "$out" || { rc=$?; cx_log "WARN: nixl transfer failed/timed out rc=$rc"; }
  cx_probe_nixl_ep || true   # informational; never fails the suite
  return "$rc"
}

run_flashinfer_suite() {
  # FlashInfer EP (flashinfer.comm.MoeAlltoAll) — pre-installed in the sglang image. When a
  # combine-quant run is requested (CX_COMBINE_DTYPE != bf16), first upgrade FlashInfer to a wheel
  # that has the quantized-combine OUTPUT path; otherwise run on the bundled version (dispatch path).
  # Upgrade FlashInfer to the newer wheel when: (a) a combine-quant run needs the output_dtype path, OR
  # (b) CX_FLASHINFER_UPGRADE=1 — the bundled 0.6.8 MoeAlltoAll MNNVL barrier intermittently deadlocks on
  # h100 ('Rank N timed out waiting for completion flag' -> CUDA unspecified launch failure); newer
  # flashinfer carries MNNVL fixes (e.g. socket-collision #36674). Otherwise run on the bundled version.
  if { [ -n "${CX_COMBINE_DTYPE:-}" ] && [ "${CX_COMBINE_DTYPE}" != "bf16" ]; } || [ "${CX_FLASHINFER_UPGRADE:-}" = "1" ]; then
    cx_build_flashinfer_latest || { cx_log "WARN: flashinfer upgrade setup failed"; return 1; }
  fi
  if ! python3 -c "import flashinfer.comm" 2>/dev/null; then
    cx_log "WARN: flashinfer.comm not importable — cannot run flashinfer EP"; return 1
  fi
  run_ep_suite flashinfer
}

# dispatch_bench runs the CURRENT CX_BENCH (+ CX_* config env) once. The sweep workflow runs many
# of these per allocation (SHARD mode below), reusing this single container + its built backend.
dispatch_bench() {
  local rc=0
  case "$CX_BENCH" in
    nccl)        run_nccl_suite || rc=1 ;;
    deepep)      run_deepep_suite || rc=1 ;;
    mori)        run_mori_suite || rc=1 ;;
    uccl)        run_uccl_suite || rc=1 ;;
    nccl-ep)     run_nccl_ep_suite || rc=1 ;;
    flashinfer)  run_flashinfer_suite || rc=1 ;;
    deepep-hybrid) run_deepep_hybrid_suite || rc=1 ;;
    nixl)        run_nixl_suite || rc=1 ;;
    mori-io)     run_mori_io_suite || rc=1 ;;
    nccl-kv)     run_nccl_kv_suite || rc=1 ;;
    mooncake)    run_mooncake_suite || rc=1 ;;
    offload)     run_collective_bench offload || rc=1 ;;
    copy-engine) run_collective_bench copy-engine || rc=1 ;;
    kv-cache)    run_collective_bench kv-cache || rc=1 ;;
    rl-mesh)     run_rl_mesh || rc=1 ;;
    allreduce-fw) run_allreduce_fw || rc=1 ;;
    all)         run_nccl_suite || rc=1; run_deepep_suite || rc=1 ;;
    *)           cx_die "unknown CX_BENCH=$CX_BENCH (want nccl|deepep|mori|uccl|nccl-ep|flashinfer|deepep-hybrid|nixl|mori-io|nccl-kv|mooncake|offload|copy-engine|kv-cache|rl-mesh|allreduce-fw|all)" ;;
  esac
  return $rc
}

rc=0
# Build-only mode: the rack EP8 launcher runs this ONCE per node inside a PERSISTENT named container
# to pre-build the from-source kernels (DeepEP V2 / flashinfer quant-combine) that the per-rank
# multi-srun case loop cannot build itself (8 separate ephemeral containers). Build the requested
# kernels into this (named, persisting) container's site-packages, then exit — no benchmark run.
if [ -n "${CX_BUILD_ONLY:-}" ]; then
  [ -n "${CX_DEEPEP_V2:-}" ] && { cx_build_deepep_v2 || rc=1; }
  [ "${CX_BENCH:-}" = "deepep-hybrid" ] && { cx_build_deepep_hybrid || rc=1; }
  [ -n "${CX_COMBINE_DTYPE:-}" ] && [ "${CX_COMBINE_DTYPE}" != "bf16" ] && { cx_build_flashinfer_latest || rc=1; }
  cx_log "CX_BUILD_ONLY: build complete rc=$rc (deepep_v2=${CX_DEEPEP_V2:-} bench=${CX_BENCH:-} combine=${CX_COMBINE_DTYPE:-})"
  exit "$rc"
fi
if [ -n "${CX_SHARD_FILE:-}" ] && [ -f "${CX_SHARD_FILE:-/nonexistent}" ]; then
  # SHARD/SWEEP mode (collectivex-sweep.yml): run EVERY case of this shard in THIS one allocation.
  # All cases share (sku, backend, v2, nodes) so the backend build (cx_build_*) is paid once and cached
  # for the rest. Each case overrides its own mode/resource_mode/dtype/contract/routing/phase/eplb/
  # workload, then reuses the same per-config path (dispatch_bench). Collapses a whole build-group's
  # cases (all modes/resource_modes) into one allocation — the sweep shard key is now (sku,backend,v2,nodes).
  ncases="$(python3 -c "import json;print(len(json.load(open('$CX_SHARD_FILE')).get('cases',[])))" 2>/dev/null || echo 0)"
  cx_log "SHARD mode: $ncases case(s) in one allocation (shard=$CX_SHARD_FILE)"
  _cx_ts_base="$CX_TS"   # per-case CX_TS suffix below keeps each case's result file UNIQUE (else
                         # cases sharing backend+phase overwrite each other at the same timestamp).
  ci=0
  failed_cases=0
  while [ "$ci" -lt "$ncases" ]; do
    export CX_TS="${_cx_ts_base}-c$(printf '%03d' "$ci")"
    # Map case[ci] fields -> CX_* env (shell-quoted). The setup job pre-resolved hidden/topk/experts
    # + the token ladder into each case, so the loop is config-only (no workloads.yaml lookup here).
    _exports="$(python3 - "$CX_SHARD_FILE" "$ci" <<'PY'
import json, sys, shlex
c = json.load(open(sys.argv[1]))["cases"][int(sys.argv[2])]
def g(k, d=""):
    v = c.get(k, d); return "" if v is None else str(v)
env = {
  "CX_BENCH": g("backend"), "CX_MODE": g("mode", "normal"),
  "CX_DISPATCH_DTYPE": g("dtype", "bf16"),
  "CX_MEASUREMENT_CONTRACT": g("contract", "layout-and-dispatch-v1"),
  "CX_ROUTING": g("routing", "uniform"), "CX_PHASE": g("phase", "decode"),
  "CX_RESOURCE_MODE": g("resource_mode", "normalized"),
  "CX_ACTIVATION_PROFILE": g("activation_profile", "normal"),
  "CX_PLACEMENT": g("placement", "packed"), "CX_ROUTING_STEP": g("routing_step", "0"),
  "CX_UNEVEN_TOKENS": g("uneven_tokens", "none"),
  "CX_EPLB": "1" if c.get("eplb") else "",
  "CX_HIDDEN": g("hidden"), "CX_TOPK": g("topk"), "CX_EXPERTS": g("experts"),
  "CX_TOKENS_LADDER": g("ladder"), "CX_CANONICAL": ("1" if c.get("canonical") else ""),
}
lines = [f"export {k}={shlex.quote(v)}" for k, v in env.items()]
# per-case timing override "iters:trials:warmup" (e.g. the MoRI large-T minimal envelope 8:1:4);
# cases without one must fall back to the harness defaults, so UNSET rather than export-empty
# (an empty CX_ITERS would defeat the 200-iter default and break the run_ep argparse; NOTE no
# apostrophes in this heredoc — bash command-substitution scanning chokes on unbalanced quotes).
timing = g("timing")
if timing:
    parts = (timing.split(":") + ["", "", ""])[:3]
    for k, v in zip(("CX_ITERS", "CX_TRIALS", "CX_WARMUP"), parts):
        if v:
            lines.append(f"export {k}={shlex.quote(v)}")
else:
    lines.append("unset CX_ITERS CX_TRIALS CX_WARMUP 2>/dev/null || true")
print("\n".join(lines))
PY
)"
    eval "$_exports"
    # Each case has its OWN routing/dims -> its own canonical workload manifest. cx_stage_canonical
    # short-circuits when CX_WORKLOAD_DIR is already set, so without this unset the first case's
    # staged dir is reused for the rest and run_ep.py can't find the later cases' manifests
    # (FileNotFoundError .cx_workloads/<wid>.manifest.json). Unset so every case re-stages its own.
    unset CX_WORKLOAD_DIR 2>/dev/null || true
    cx_log "  [$((ci+1))/$ncases] $CX_BENCH $CX_PHASE $CX_DISPATCH_DTYPE/$CX_MODE/${CX_MEASUREMENT_CONTRACT/-v1/} rt=$CX_ROUTING eplb=${CX_EPLB:-0}"
    # flashinfer's MoeAlltoAll MNNVL barrier INTERMITTENTLY deadlocks on h100 ('Rank N timed out waiting
    # for completion flag' -> CUDA unspecified launch failure): ~half of cases, scattered across T/routing,
    # the SAME config both crashes AND passes (a transient, not config/pidfd). Upgrade to flashinfer 0.6.14
    # + a between-case shm-drop settle were both TESTED and did NOT fix it (the settle made it worse). Since
    # it's intermittent, RETRY: each fresh torchrun is another independent ~50% shot, so a few retries
    # recover almost all cases. On a retry success, drop this case's intermediate failed-case record so it
    # doesn't pollute the shard. Non-flashinfer backends run ONCE — their failures are deterministic
    # (h200 flashinfer pidfd, aarch64 uccl, deepep-hybrid ll) so retrying only wastes the allocation.
    attempts=1; [ "$CX_BENCH" = "flashinfer" ] && attempts=$(( ${CX_FLASHINFER_RETRIES:-3} + 1 ))
    a=1
    while :; do
      if dispatch_bench; then
        [ "$a" -gt 1 ] && rm -f results/failed_*"${CX_TS}"*.json 2>/dev/null || true
        break
      fi
      # A failed CASE does NOT fail the shard job. The failed-case record + the summary table are
      # the signal (the doctrine is judge-by-data, and the conclusion should match it): expected
      # per-case failures — the empty-rank diagnostic on HybridEP/UCCL Hopper, a flashinfer
      # intermittent that survived its retries — used to flip 200+-correct-point jobs red. The job
      # now fails only when the harness itself is unhealthy (summarize.py: NO valid results at all).
      # Known DETERMINISTIC whole-shard walls never even dispatch (capability RUNNER_WALLS/aarch64).
      [ "$a" -ge "$attempts" ] && { failed_cases=$((failed_cases+1)); cx_log "  [$((ci+1))/$ncases] $CX_BENCH case FAILED after $a attempt(s) — failed-case record preserved; shard continues"; break; }
      cx_log "  [$((ci+1))/$ncases] $CX_BENCH attempt $a/$attempts failed — retry (intermittent MNNVL barrier)"
      a=$((a+1))
    done
    ci=$((ci + 1))
  done
  [ "${failed_cases:-0}" -gt 0 ] && cx_log "SHARD done: $failed_cases/$ncases case(s) failed (records preserved — see the summary table + failed_*.json)" || true
else
  # Single-bench (workflow_dispatch) path gets the SAME flashinfer retry as SHARD mode — the
  # combine-quant runs (flashinfer-combine-* -> CX_BENCH=flashinfer) come through here and are
  # subject to the same intermittent h100 MNNVL-barrier deadlock; one attempt dies ~50% of the
  # time. Non-flashinfer benches run once (their failures are deterministic — retry wastes time).
  attempts=1; [ "$CX_BENCH" = "flashinfer" ] && attempts=$(( ${CX_FLASHINFER_RETRIES:-3} + 1 ))
  a=1
  while :; do
    if dispatch_bench; then
      [ "$a" -gt 1 ] && rm -f results/failed_*"${CX_TS}"*.json 2>/dev/null || true
      break
    fi
    [ "$a" -ge "$attempts" ] && { rc=1; break; }
    cx_log "$CX_BENCH attempt $a/$attempts failed — retry (intermittent MNNVL barrier)"
    a=$((a+1))
  done
fi

# Summary table for the log; also fails the job if no valid results were produced.
python3 summarize.py --results-dir results --runner "$CX_RUNNER" --ts "$CX_TS" || rc=1
exit "$rc"
