#!/usr/bin/env bash
# CollectiveX — generic in-container benchmark dispatcher (single-node).
#
# Runs INSIDE the container under `srun`, invoked by every per-SKU adapter
# (launch_<sku>.sh). The SKU adapter handles allocation/container/transport-env;
# this script selects one EP backend from CX_BENCH and writes result JSON under results/.
#
# Required env (exported by the adapter): CX_RUNNER CX_NGPUS CX_TS CX_TOPO
# Selector: CX_BENCH = deepep | mori | uccl | nccl-ep | flashinfer | deepep-hybrid
# EP knobs passed to tests/run_ep.py:
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
CX_BENCH="${CX_BENCH:-deepep}"
CX_TRANSPORT="${CX_TRANSPORT:-}"
ENVJSON="results/env_${CX_RUNNER}_${CX_TS}.json"

cx_apply_timing_profile

cx_log "in-container: runner=$CX_RUNNER ngpus=$CX_NGPUS bench=$CX_BENCH topo=$CX_TOPO"
python3 env_capture.py --redact --out "$ENVJSON" --timestamp "$CX_TS"

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
    || { cx_log "ERROR: canonical workload staging failed"; return 1; }
  export CX_WORKLOAD_DIR="$dir"
  cx_log "canonical workloads staged at $dir"
}

# run_ep_suite <backend: deepep|mori>
# One tests/run_ep.py invocation per phase (decode/prefill/both); dispatch and
# combine are timed separately inside it. One JSON per (backend, phase).
# Preserve a failed case with its full scheduled identity instead of letting it vanish.
emit_failed_case() {  # backend phase rc
  cx_emit_ep_failed_case \
    "results/failed_${CX_RUNNER}_${1}_${2}_${CX_TS}.json" "$1" "$2" "$3" || true
}

run_ep_suite() {
  local backend="$1" phase phases ladder rc=0 rc_run
  ladder="$(cx_ep_ladder)"
  phases="${CX_PHASE:-decode}"
  [ "$phases" = "both" ] && phases="decode prefill"
  if ! cx_stage_canonical; then
    for phase in $phases; do
      emit_failed_case "$backend" "$phase" 2
    done
    return 1
  fi
  for phase in $phases; do
    cx_log "ep backend=$backend phase=$phase ngpus=$CX_NGPUS ladder='${ladder:-<phase-default>}'"
    local out="results/${CX_RUNNER}_${backend}_${phase}_${CX_TS}.json"
    local -a EPARGS=(--backend "$backend" --phase "$phase" --tokens-ladder "$ladder" --mode "${CX_MODE:-normal}"
      --hidden "${CX_HIDDEN:-7168}" --topk "${CX_TOPK:-8}" --experts "${CX_EXPERTS:-256}"
      --dispatch-dtype "${CX_DISPATCH_DTYPE:-bf16}" --routing "${CX_ROUTING:-uniform}"
      --num-sms "${CX_NUM_SMS:-24}" --seed "${CX_SEED:-67}" --iters "${CX_ITERS:-8}"
      --trials "${CX_TRIALS:-64}" --warmup "${CX_WARMUP:-32}"
      --measurement-contract "${CX_MEASUREMENT_CONTRACT:-layout-and-dispatch-v1}"
      --resource-mode "${CX_RESOURCE_MODE:-normalized}" --sm-fraction "${CX_SM_FRACTION:-0.18}"
      --activation-profile "${CX_ACTIVATION_PROFILE:-normal}" --placement "${CX_PLACEMENT:-packed}"
      --gpus-per-node "${CX_GPUS_PER_NODE:-0}" --scale-up-domain "${CX_SCALE_UP_DOMAIN:-0}"
      --routing-step "${CX_ROUTING_STEP:-0}" --uneven-tokens "${CX_UNEVEN_TOKENS:-none}"
      --combine-dtype "${CX_COMBINE_DTYPE:-bf16}" --combine-quant-mode "${CX_COMBINE_QUANT_MODE:-none}"
      --case-id "${CX_CASE_ID:-}" --suite "${CX_SUITE:-}" --workload-name "${CX_WORKLOAD_NAME:-}"
      --required-publication "${CX_REQUIRED_PUBLICATION:-}"
      --runner "$CX_RUNNER" --topology-class "$CX_TOPO" --transport "$CX_TRANSPORT"
      --env-json "$ENVJSON" --out "$out")
    [ -n "${CX_EPLB:-}" ] && EPARGS+=(--eplb)
    [ -n "${CX_WORKLOAD_DIR:-}" ] && EPARGS+=(--workload-dir "$CX_WORKLOAD_DIR")
    [ -n "${CX_WAIVE_ANOMALY:-}" ] && EPARGS+=(--waive-anomaly)
    timeout -k 30 "${CX_RUN_TIMEOUT:-900}" \
      torchrun --nproc_per_node="$CX_NGPUS" tests/run_ep.py "${EPARGS[@]}"
    rc_run=$?
    if [ "$rc_run" != 0 ]; then
      cx_log "WARN: $backend $phase run failed/timed out rc=$rc_run (CX_RUN_TIMEOUT=${CX_RUN_TIMEOUT:-900}s)"
      if cx_has_result_doc "$out"; then
        cx_demote_result_doc "$out" "$rc_run" \
          || { rm -f "$out"; emit_failed_case "$backend" "$phase" "$rc_run"; }
        cx_log "preserved benchmark output as a failed attempt"
      else
        emit_failed_case "$backend" "$phase" "$rc_run"
      fi
      rc=1
    fi
  done
  return "$rc"
}

# Legacy direct-env diagnostic only. This installs DeepEP main and still drives `Buffer`; it is not
# PR #605 `ElasticBuffer` V2 evidence and is intentionally absent from workflows and v1 matrices.
# Keep the low-level hook while the real adapter is developed; its output must not be promoted.
cx_build_deepep_v2() {
  # IDEMPOTENT: SHARD mode calls dispatch_bench (hence this) once PER CASE. Build once per allocation,
  # then skip — else a 60-case shard re-runs the from-source build 60x (force-reinstall) and blows the
  # slurm --time. Sentinel lives in the container fs (persists across the x86 in-container case loop).
  [ -f /tmp/.cx_built_deepep_v2 ] && { cx_log "legacy DeepEP diagnostic already built — skip"; return 0; }
  local arch="9.0"; case "$CX_RUNNER" in b300*|gb300*|b200*) arch="10.0";; esac
  cx_log "legacy DeepEP main diagnostic: building source (TORCH_CUDA_ARCH_LIST=$arch)"
  # PEP 668: newer images (H200/B300) ship an externally-managed Python that refuses `pip install`.
  # PIP_BREAK_SYSTEM_PACKAGES is honored by pip>=23.0.1 and silently ignored by older pip (H100),
  # so this is safe across every image; --break-system-packages as a flag would error on old pip.
  export PIP_BREAK_SYSTEM_PACKAGES=1
  pip install -q "nvidia-nccl-cu13>=2.30.4" >&2 2>&1 || cx_log "WARN: nvidia-nccl-cu13 install warning"
  rm -rf /tmp/DeepEP_v2
  git clone --depth 1 https://github.com/deepseek-ai/DeepEP /tmp/DeepEP_v2 >&2 2>&1 \
    || { cx_log "ERROR: legacy DeepEP diagnostic clone failed"; return 1; }
  DEEPEP_COMMIT="legacy-main-$(git -C /tmp/DeepEP_v2 rev-parse --short HEAD 2>/dev/null || echo main)"
  export DEEPEP_COMMIT
  ( cd /tmp/DeepEP_v2 && TORCH_CUDA_ARCH_LIST="$arch" MAX_JOBS=16 \
      pip install -q --no-build-isolation --force-reinstall . ) >&2 2>&1 \
    || { cx_log "ERROR: legacy DeepEP diagnostic build failed (arch=$arch)"; return 1; }
  python3 -c "import deep_ep; print('built deep_ep', getattr(deep_ep,'__version__','?'))" >&2 \
    || { cx_log "ERROR: legacy DeepEP diagnostic import failed"; return 1; }
  : > /tmp/.cx_built_deepep_v2   # sentinel: skip rebuild on subsequent cases in this allocation
  cx_log "legacy DeepEP diagnostic ready ($DEEPEP_COMMIT; non-publication)"
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
  NVSHMEM_DIR="$(python3 -c 'import os,nvidia.nvshmem as n; print(os.path.dirname(n.__file__))' 2>/dev/null || echo /usr/local/lib/python3.12/dist-packages/nvidia/nvshmem)"
  export NVSHMEM_DIR
  local cccl; cccl="$(echo /usr/local/cuda*/targets/*/include/cccl | awk '{print $1}')"
  [ -d "$cccl" ] && { export CPATH="$cccl:${CPATH:-}"; export NVCC_PREPEND_FLAGS="-I$cccl ${NVCC_PREPEND_FLAGS:-}"; }
  [ -e "$NVSHMEM_DIR/lib/libnvshmem_host.so.3" ] && ln -sf libnvshmem_host.so.3 "$NVSHMEM_DIR/lib/libnvshmem_host.so" 2>/dev/null || true
  export LD_LIBRARY_PATH="$NVSHMEM_DIR/lib:${LD_LIBRARY_PATH:-}"
  rm -rf /tmp/DeepEP_hybrid
  git clone --depth 1 --branch hybrid-ep https://github.com/deepseek-ai/DeepEP /tmp/DeepEP_hybrid >&2 2>&1 \
    || { cx_log "ERROR: hybrid-ep git clone failed"; return 1; }
  DEEPEP_COMMIT="hybrid-$(git -C /tmp/DeepEP_hybrid rev-parse --short HEAD 2>/dev/null || echo hybrid-ep)"
  export DEEPEP_COMMIT
  # Install into site-packages so the package persists across separate srun shells in the named
  # container. The shared backend-env handoff below carries process-local loader/provenance values.
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
  : > /tmp/.cx_built_deepep_hybrid   # sentinel: skip rebuild on subsequent cases in this allocation
  cx_log "DeepEP hybrid-ep ready ($DEEPEP_COMMIT)"
}

# UCCL EP (uccl.ep.Buffer is a DeepEP-API clone). The prebuilt wheel is cu12; on a cu13
# image its kernels need a cu12 CUDA runtime on LD_LIBRARY_PATH (probe-confirmed). PEP-668
# images need PIP_BREAK_SYSTEM_PACKAGES. Best-effort; failure to import fails loudly.
cx_build_uccl() {
  if [ -f /tmp/.cx_built_uccl ]; then
    cx_log "UCCL EP already prepared this allocation — skip rebuild"
    python3 -c "import torch; from uccl_deepep import Buffer" 2>/dev/null || return 1
    return 0
  fi
  cx_log "UCCL EP: pip install uccl + cu12 runtime shim"
  export PIP_BREAK_SYSTEM_PACKAGES=1
  pip install -q uccl >&2 2>&1 || { cx_log "ERROR: pip install uccl failed"; return 1; }
  pip install -q nvidia-cuda-runtime-cu12 >&2 2>&1 || cx_log "WARN: nvidia-cuda-runtime-cu12 warning"
  local cu12lib
  cu12lib="$(python3 -c "import nvidia.cuda_runtime as m, os; print(os.path.join(os.path.dirname(m.__file__),'lib'))" 2>/dev/null)"
  [ -n "$cu12lib" ] && export LD_LIBRARY_PATH="$cu12lib:${LD_LIBRARY_PATH:-}"
  UCCL_COMMIT="pkg-$(python3 -c 'import importlib.metadata as m; print(m.version("uccl"))' 2>/dev/null || echo uccl)"
  export UCCL_COMMIT
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
    python3 -c "import torch; from uccl_deepep import Buffer; print('uccl_deepep wrapper ready')" >&2 \
      || { cx_log "ERROR: uccl_deepep wrapper import failed"; return 1; }
    export CX_UCCL_WRAPPER=1
  else
    cx_log "ERROR: uccl deep_ep_wrapper not available"
    return 1
  fi
  : > /tmp/.cx_built_uccl
  cx_log "UCCL EP ready ($UCCL_COMMIT, wrapper=${CX_UCCL_WRAPPER:-0})"
}

run_deepep_suite() {
  cx_prepare_backend deepep || { cx_log "WARN: DeepEP preparation failed"; return 1; }
  run_ep_suite deepep
}

run_mori_suite() {
  cx_prepare_backend mori || { cx_log "WARN: MoRI preparation failed"; return 1; }
  run_ep_suite mori
}

run_uccl_suite() {
  cx_prepare_backend uccl || { cx_log "WARN: UCCL EP preparation failed"; return 1; }
  run_ep_suite uccl
}
run_nccl_ep_suite() {
  # Portable torch.distributed all-to-all reference; no build step.
  run_ep_suite nccl-ep
}
run_deepep_hybrid_suite() {
  cx_prepare_backend deepep-hybrid || { cx_log "WARN: Hybrid DeepEP preparation failed"; return 1; }
  run_ep_suite deepep-hybrid
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
  cx_capture_flashinfer_identity
  cx_log "FlashInfer upgrade (nightly): $before -> $after"
  cx_log "FlashInfer stack: $CX_FLASHINFER_STACK"
  python3 -c "import inspect, flashinfer.comm as c; assert 'output_dtype' in str(inspect.signature(c.MoeAlltoAll.combine)), 'combine still has no output_dtype'; print('combine output_dtype: present')" >&2 \
    || { cx_log "ERROR: upgraded FlashInfer combine still lacks output_dtype — cannot quant-combine"; return 1; }
  : > /tmp/.cx_built_flashinfer   # sentinel: skip rebuild on subsequent cases in this allocation
}

cx_capture_deepep_identity() {
  local version
  version="$(python3 - <<'PY' 2>/dev/null || echo unknown
try:
    import importlib.metadata as metadata
    print(metadata.version("deep_ep"))
except Exception:
    import deep_ep
    print(getattr(deep_ep, "__version__", "unknown"))
PY
)"
  export DEEPEP_COMMIT="${DEEPEP_COMMIT:-pkg-$version}"
}

cx_capture_flashinfer_identity() {
  local version
  version="$(python3 - <<'PY' 2>/dev/null || echo unknown
try:
    import importlib.metadata as metadata
    print(metadata.version("flashinfer-python"))
except Exception:
    import flashinfer
    print(getattr(flashinfer, "__version__", "unknown"))
PY
)"
  export FLASHINFER_COMMIT="${FLASHINFER_COMMIT:-pkg-$version}"
  CX_FLASHINFER_STACK="$(python3 - <<'PY' 2>/dev/null || echo capture-failed
import importlib.metadata as metadata

packages = ("flashinfer-python", "flashinfer-cubin", "flashinfer-jit-cache",
            "nvidia-cutlass-dsl", "torch")
def version(name):
    try:
        return metadata.version(name)
    except Exception:
        return "absent"
print(" ".join(f"{name}={version(name)}" for name in packages))
PY
)"
  export CX_FLASHINFER_STACK
}

# A rack build-only step and its rank steps are separate shells. Persist every backend-created
# loader/import path and build identity in the named container, then source it from each rank.
cx_persist_backend_env() {
  local path=/tmp/.cx_backend_env name
  local -a names=(LD_LIBRARY_PATH PYTHONPATH NVSHMEM_DIR DEEPEP_COMMIT FLASHINFER_COMMIT
    CX_FLASHINFER_STACK FLASHINFER_DISABLE_VERSION_CHECK UCCL_COMMIT CX_UCCL_WRAPPER)
  : > "$path" || return 1
  for name in "${names[@]}"; do
    if declare -p "$name" >/dev/null 2>&1; then
      printf 'export %s=%q\n' "$name" "${!name}" >> "$path" || return 1
    fi
  done
}

# Prepare and probe one backend without running a benchmark. The same hook is used
# by normal in-container runs and by rack launchers' persistent build-only step.
cx_prepare_backend() {
  local backend="${1:-}"
  [ -f /tmp/.cx_backend_env ] && source /tmp/.cx_backend_env
  case "$backend" in
    deepep)
      if [ "${CX_DEEPEP_V2:-0}" = "1" ]; then
        cx_build_deepep_v2 || return 1
      fi
      if ! python3 -c "from deep_ep import Buffer" 2>/dev/null; then
        command -v rebuild-deepep.sh >/dev/null 2>&1 || {
          cx_log "WARN: DeepEP is unavailable and rebuild-deepep.sh is missing"
          return 1
        }
        cx_log "building normal DeepEP"
        rebuild-deepep.sh >&2 || return 1
      fi
      python3 -c "from deep_ep import Buffer" 2>/dev/null || return 1
      cx_capture_deepep_identity
      ;;
    deepep-hybrid)
      cx_build_deepep_hybrid || return 1
      ;;
    flashinfer)
      if { [ -n "${CX_COMBINE_DTYPE:-}" ] && [ "${CX_COMBINE_DTYPE}" != "bf16" ]; } \
          || [ "${CX_FLASHINFER_UPGRADE:-}" = "1" ]; then
        cx_build_flashinfer_latest || return 1
      fi
      python3 -c "import flashinfer.comm" 2>/dev/null || return 1
      cx_capture_flashinfer_identity
      ;;
    uccl)
      cx_build_uccl || return 1
      ;;
    mori)
      python3 -c "import mori" 2>/dev/null || return 1
      ;;
    nccl-ep)
      ;;
    *)
      cx_log "ERROR: unknown backend preparation request"
      return 1
      ;;
  esac
}

run_flashinfer_suite() {
  cx_prepare_backend flashinfer || { cx_log "WARN: FlashInfer preparation failed"; return 1; }
  run_ep_suite flashinfer
}

# dispatch_bench runs the CURRENT CX_BENCH (+ CX_* config env) once. The sweep workflow runs many
# of these per allocation (SHARD mode below), reusing this single container + its built backend.
dispatch_bench() {
  local rc=0
  case "$CX_BENCH" in
    deepep)      run_deepep_suite || rc=1 ;;
    mori)        run_mori_suite || rc=1 ;;
    uccl)        run_uccl_suite || rc=1 ;;
    nccl-ep)     run_nccl_ep_suite || rc=1 ;;
    flashinfer)  run_flashinfer_suite || rc=1 ;;
    deepep-hybrid) run_deepep_hybrid_suite || rc=1 ;;
    *)           cx_die "unknown CX_BENCH=$CX_BENCH (want deepep|mori|uccl|nccl-ep|flashinfer|deepep-hybrid)" ;;
  esac
  return $rc
}

rc=0
# Structured v1 shards never run the legacy DeepEP-main diagnostic, even if a self-hosted runner
# happens to inherit the old environment variable. Direct manual invocations without a shard remain.
[ -n "${CX_SHARD_FILE:-}" ] && unset CX_DEEPEP_V2
# Build-only mode: rack launchers run the shared backend preparation hook once per
# node inside a persistent named container, then direct rank processes reuse it.
if [ -n "${CX_BUILD_ONLY:-}" ]; then
  if cx_prepare_backend "${CX_BENCH:-}"; then
    cx_persist_backend_env || rc=1
  else
    rc=1
  fi
  cx_log "backend preparation: bench=${CX_BENCH:-unknown} rc=$rc"
  exit "$rc"
fi
if [ -n "${CX_SHARD_FILE:-}" ] && [ -f "${CX_SHARD_FILE:-/nonexistent}" ]; then
  # SHARD/SWEEP mode (collectivex-sweep.yml): run EVERY case of this shard in THIS one allocation.
  # All cases share (sku, backend, nodes) so the backend build (cx_build_*) is paid once and cached
  # for the rest. Each case overrides its own mode/resource_mode/dtype/contract/routing/phase/eplb/
  # workload, then reuses the same per-config path (dispatch_bench). Collapses a whole build-group's
  # cases (all modes/resource_modes) into one allocation; the shard key is (sku,backend,nodes).
  ncases="$(python3 -c "import json;print(len(json.load(open('$CX_SHARD_FILE')).get('cases',[])))" 2>/dev/null || echo 0)"
  cx_log "SHARD mode: $ncases case(s) in one allocation (shard=$CX_SHARD_FILE)"
  _cx_ts_base="$CX_TS"   # per-case CX_TS suffix below keeps each case's result file UNIQUE (else
                         # cases sharing backend+phase overwrite each other at the same timestamp).
  ci=0
  failed_cases=0
  while [ "$ci" -lt "$ncases" ]; do
    CX_TS="${_cx_ts_base}-c$(printf '%03d' "$ci")"
    export CX_TS
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
  "CX_EP": g("ep", "1"),
  "CX_EPLB": "1" if c.get("eplb") else "",
  "CX_COMBINE_QUANT_MODE": g("combine_quant_mode", "none"),
  "CX_CASE_ID": g("case_id"), "CX_SUITE": g("suite"), "CX_WORKLOAD_NAME": g("workload"),
  "CX_REQUIRED_PUBLICATION": g("required_publication"),
  "CX_HIDDEN": g("hidden"), "CX_TOPK": g("topk"), "CX_EXPERTS": g("experts"),
  "CX_TOKENS_LADDER": g("ladder"), "CX_CANONICAL": ("1" if c.get("canonical") else ""),
}
lines = [f"export {k}={shlex.quote(v)}" for k, v in env.items()]
# Per-case timing "iters:trials:warmup" (fixed-512-v1 requires 8:64:32 everywhere);
# cases without one must fall back to the harness defaults, so UNSET rather than export-empty
# (an empty CX_ITERS would defeat the 8-iter default and break the run_ep argparse; NOTE no
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
    # it's intermittent, RETRY: each fresh torchrun is another independent attempt. Every attempt gets
    # a unique identity and filename; a later success must not erase the earlier failure evidence.
    attempts=1; [ "$CX_BENCH" = "flashinfer" ] && attempts=$(( ${CX_FLASHINFER_RETRIES:-3} + 1 ))
    _cx_case_ts="$CX_TS"
    a=1
    while :; do
      CX_TS="${_cx_case_ts}-a$(printf '%02d' "$a")"
      export CX_ATTEMPT_ID="$a" CX_TS
      if dispatch_bench; then
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
    export CX_TS="$_cx_case_ts"
    ci=$((ci + 1))
  done
  [ "${failed_cases:-0}" -gt 0 ] && cx_log "SHARD done: $failed_cases/$ncases case(s) failed (records preserved — see the summary table + failed_*.json)" || true
  # RESTORE the base timestamp: the loop re-exported CX_TS per case (…-cNNN), so leaving the LAST
  # case's ts in place made the final summarize below filter to that ONE case — and when the last
  # case happened to be a failing diagnostic (empty-rank sorts last), summarize saw "no result
  # files" and flipped an otherwise-complete shard red (h200 run 28577792572: 39/40 good cases,
  # conclusion failure). The base ts is a substring of every per-case filename, so summarize then
  # gates on the WHOLE shard's results, as intended.
  export CX_TS="$_cx_ts_base"
else
  # Single-bench (workflow_dispatch) path gets the SAME flashinfer retry as SHARD mode — the
  # combine-quant runs (flashinfer-combine-* -> CX_BENCH=flashinfer) come through here and are
  # subject to the same intermittent h100 MNNVL-barrier deadlock; one attempt dies ~50% of the
  # time. Non-flashinfer benches run once (their failures are deterministic — retry wastes time).
  attempts=1; [ "$CX_BENCH" = "flashinfer" ] && attempts=$(( ${CX_FLASHINFER_RETRIES:-3} + 1 ))
  _cx_single_ts="$CX_TS"
  a=1
  while :; do
    CX_TS="${_cx_single_ts}-a$(printf '%02d' "$a")"
    export CX_ATTEMPT_ID="$a" CX_TS
    if dispatch_bench; then
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
