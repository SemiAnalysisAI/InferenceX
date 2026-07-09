#!/usr/bin/env bash
# CollectiveX — generic in-container benchmark dispatcher (single-node).
#
# Runs INSIDE the container under `srun` for single-node shards. The GB EP8 launcher invokes
# run_ep.py directly across nodes. The SKU adapter handles allocation/container/transport-env;
# this script selects one EP backend from CX_BENCH and writes result JSON under results/.
#
# Required env (exported by the adapter): CX_RUNNER CX_NGPUS CX_TS CX_TOPO
# Selector: CX_BENCH = deepep | deepep-v2 | mori | deepep-hybrid
# EP knobs passed to bench/run_ep.py:
#   CX_PHASE = decode | prefill | both (default decode)   <- picks the token sweep
#   CX_TOKENS_LADDER (space/comma sep; blank = phase default)
#   CX_HIDDEN CX_TOPK CX_EXPERTS CX_ROUTING CX_SEED CX_ITERS
set -euo pipefail

cd /ix/experimental/CollectiveX
# shellcheck source=../runtime/common.sh
source runtime/common.sh
mkdir -p results

: "${CX_RUNNER:?CX_RUNNER not set}"
: "${CX_NGPUS:?CX_NGPUS not set}"
: "${CX_TS:?CX_TS not set}"
: "${CX_TOPO:?CX_TOPO not set}"
CX_BENCH="${CX_BENCH:-deepep-v2}"
CX_TRANSPORT="${CX_TRANSPORT:-}"

cx_apply_timing_profile

cx_log "in-container: runner=$CX_RUNNER ngpus=$CX_NGPUS bench=$CX_BENCH topo=$CX_TOPO"

# Blank ladders use the phase default in bench/run_ep.py.
cx_ep_ladder() {
  printf '%s' "${CX_TOKENS_LADDER:-}"
}

# run_ep_suite <backend>
# One bench/run_ep.py invocation per phase (decode/prefill/both); dispatch and
# combine are timed separately inside it. One JSON per (backend, phase).
run_ep_suite() {
    local backend="$1" phase phases ladder failure_kind rc=0 rc_run
  ladder="$(cx_ep_ladder)"
  phases="${CX_PHASE:-decode}"
  [ "$phases" = "both" ] && phases="decode prefill"
  for phase in $phases; do
    cx_log "ep backend=$backend phase=$phase ngpus=$CX_NGPUS ladder='${ladder:-<phase-default>}'"
    local out="results/${CX_RUNNER}_${backend}_${phase}_${CX_TS}.json"
    local -a EPARGS=(--backend "$backend" --mode "${CX_MODE:-normal}" --phase "$phase"
      --tokens-ladder "$ladder"
      --hidden "${CX_HIDDEN:-7168}" --topk "${CX_TOPK:-8}" --experts "${CX_EXPERTS:-256}"
      --routing "${CX_ROUTING:-uniform}" --seed "${CX_SEED:-67}" --iters "${CX_ITERS:-8}"
      --trials "${CX_TRIALS:-64}" --warmup "${CX_WARMUP:-32}"
      --gpus-per-node "${CX_GPUS_PER_NODE:-0}" --scale-up-domain "${CX_SCALE_UP_DOMAIN:-0}"
      --scope "${CX_SCOPE:-scale-up}" --scale-up-transport "${CX_SCALE_UP_TRANSPORT:-unknown}"
      --scale-out-transport "${CX_SCALE_OUT_TRANSPORT:-}"
      --case-id "${CX_CASE_ID:-}" --suite "${CX_SUITE:-}" --workload-name "${CX_WORKLOAD_NAME:-}"
      --version "${CX_VERSION:-1}"
      --runner "$CX_RUNNER" --topology-class "$CX_TOPO" --transport "$CX_TRANSPORT"
      --out "$out")
    if timeout -k 30 "${CX_RUN_TIMEOUT:-900}" \
      torchrun --nproc_per_node="$CX_NGPUS" bench/run_ep.py "${EPARGS[@]}"; then
      rc_run=0
    else
      rc_run=$?
    fi
    # Result-document gating and terminal-outcome emission were removed with
    # contracts.py; success is now the run_ep.py return code alone. Any result
    # document it wrote is left in place for the summary renderer, which validates
    # nothing.
    if [ "$rc_run" != 0 ]; then
      failure_kind=failed
      [ "$rc_run" != 124 ] && [ "$rc_run" != 137 ] || failure_kind="timed out"
      if [ "$failure_kind" = "timed out" ]; then
        cx_log "WARN: $backend $phase run timed out rc=$rc_run (limit=${CX_RUN_TIMEOUT:-900}s)"
      else
        cx_log "WARN: $backend $phase run failed rc=$rc_run"
      fi
      rc=1
    fi
  done
  return "$rc"
}

# Resolve and verify the actual CUDA target before compiling source kernels.
cx_cuda_arch() {
  local expected detected
  case "$CX_RUNNER" in
    h100*|h200*) expected="9.0" ;;
    b200*|gb200*) expected="10.0" ;;
    b300*|gb300*) expected="10.3" ;;
    *) cx_log "ERROR: no CUDA target registered for $CX_RUNNER"; return 1 ;;
  esac
  detected="$(python3 - <<'PY'
import torch

major, minor = torch.cuda.get_device_capability()
print(f"{major}.{minor}")
PY
)" || return 1
  [ "$detected" = "$expected" ] || {
    cx_log "ERROR: $CX_RUNNER expected CUDA target $expected, detected $detected"
    return 1
  }
  printf '%s' "$detected"
}

cx_nvidia_package_root() {
  local package="$1" component="$2"
  python3 - "$package" "$component" <<'PY'
from importlib import metadata
from pathlib import Path, PurePosixPath
import sys

package, component = sys.argv[1:]
try:
    distribution = metadata.distribution(package)
    prefix = f"nvidia/{component}/"
    entries = [str(entry).replace("\\", "/") for entry in distribution.files or ()]
    if not any(entry.startswith(prefix) for entry in entries):
        raise ValueError
    root = Path(distribution.locate_file(PurePosixPath("nvidia") / component)).resolve()
    if not root.is_dir():
        raise ValueError
except (metadata.PackageNotFoundError, OSError, TypeError, ValueError):
    raise SystemExit(1)
print(root, end="")
PY
}

cx_prepare_cuda_cccl() {
  local cccl="" candidate cuda_home nvcc
  nvcc="$(command -v nvcc)" \
    || { cx_log "ERROR: CUDA nvcc is unavailable"; return 1; }
  nvcc="$(readlink -f -- "$nvcc")" \
    || { cx_log "ERROR: CUDA nvcc cannot be resolved"; return 1; }
  case "$nvcc" in
    */bin/nvcc) cuda_home="${nvcc%/bin/nvcc}" ;;
    *) cx_log "ERROR: CUDA nvcc has an unexpected path"; return 1 ;;
  esac
  [ -x "$cuda_home/bin/nvcc" ] && [ -d "$cuda_home/include" ] \
    && [ -d "$cuda_home/lib64" ] \
    || { cx_log "ERROR: CUDA toolkit root is incomplete"; return 1; }
  for candidate in "$cuda_home"/targets/*/include/cccl; do
    if [ -d "$candidate" ]; then
      cccl="$candidate"
      break
    fi
  done
  [ -n "$cccl" ] || { cx_log "ERROR: CUDA CCCL headers are unavailable"; return 1; }
  export CUDA_HOME="$cuda_home" CX_CUDA_CCCL="$cccl"
  export CPATH="$cccl:${CPATH:-}"
  export NVCC_PREPEND_FLAGS="-I$cccl ${NVCC_PREPEND_FLAGS:-}"
}

cx_prepare_deepep_toolchain() {
  local packaged overlay path root temporary
  packaged="$(cx_nvidia_package_root nvidia-nvshmem-cu12 nvshmem)" \
    || { cx_log "ERROR: nvidia.nvshmem is unavailable"; return 1; }
  root="$(cx_deepep_v2_root)" || return 1
  overlay="$root/nvshmem-overlay"
  if ! (
    umask 077
    exec 8>"$root/nvshmem-overlay.lock" || exit 1
    flock 8 || exit 1
    if [ ! -d "$overlay" ]; then
      temporary="$root/.nvshmem-overlay.$$"
      rm -rf "$temporary" || exit 1
      mkdir -p "$temporary/lib" || exit 1
      ln -s "$packaged/include" "$temporary/include" || exit 1
      for path in "$packaged"/lib/*; do
        ln -s "$path" "$temporary/lib/${path##*/}" || exit 1
      done
      [ ! -e "$packaged/lib/libnvshmem_host.so.3" ] \
        || ln -sf "$packaged/lib/libnvshmem_host.so.3" \
          "$temporary/lib/libnvshmem_host.so" || exit 1
      mv "$temporary" "$overlay" || exit 1
    fi
    [ ! -L "$overlay" ] \
      && [ "$(readlink -f "$overlay/include")" = "$(readlink -f "$packaged/include")" ] \
      && [ -e "$overlay/lib/libnvshmem_host.so" ] \
      && [ -e "$overlay/lib/libnvshmem_device.a" ]
  ); then
    cx_log "ERROR: DeepEP V2 NVSHMEM overlay is invalid"
    return 1
  fi
  NVSHMEM_DIR="$overlay"
  export NVSHMEM_DIR
  cx_prepare_cuda_cccl || return 1
  export LD_LIBRARY_PATH="$NVSHMEM_DIR/lib:${LD_LIBRARY_PATH:-}"
}

# DeepEP V2 is PR #605's ElasticBuffer implementation with upstream PR #630's pure scale-up
# initialization fix and PR #640's exact libnccl mapping check. Canonical launchers stage the
# pinned source and mount a private cluster-local build cache at /cx-cache.
cx_deepep_v2_root() {
  local arch cpu base image
  arch="$(cx_cuda_arch)" || return 1
  cpu="$(uname -m)"
  [[ "$cpu" =~ ^[A-Za-z0-9._-]+$ ]] || return 1
  base="${CX_BACKEND_CACHE_ROOT:-}"
  [[ "$base" = /* ]] || return 1
  image="$(printf '%s' "${COLLECTIVEX_IMAGE:-manual}" | tr -cs 'A-Za-z0-9_.-' '-')" \
    || return 1
  image="${image#-}"; image="${image%-}"
  [ -n "$image" ] || return 1
  printf '%s/deepep-v2-v3-%s-sm%s-%s-torch2.10.0-cu130-nccl2.30.4-nvshmem3.3.9-%s-%s-%s-%s' \
    "$base" "$cpu" "${arch/./}" "$image" \
    "${CX_DEEPEP_V2_COMMIT:0:12}" "${CX_DEEPEP_V2_TREE:0:12}" \
    "${CX_DEEPEP_V2_FMT_COMMIT:0:12}" "${CX_DEEPEP_V2_NCCL_CHECK_COMMIT:0:12}"
}

cx_activate_deepep_v2() {
  local root venv execution_id
  root="$(cx_deepep_v2_root)" || return 1
  venv="$root/venv"
  [ -x "$venv/bin/python" ] \
    || { cx_log "ERROR: DeepEP V2 venv interpreter is unavailable"; return 1; }
  export VIRTUAL_ENV="$venv"
  export PATH="$venv/bin:${PATH#"$venv/bin:"}"
  EP_NCCL_ROOT_DIR="$(cx_nvidia_package_root nvidia-nccl-cu13 nccl)" \
    || { cx_log "ERROR: DeepEP V2 NCCL package root is unavailable"; return 1; }
  EP_NVSHMEM_ROOT_DIR="$(cx_nvidia_package_root nvidia-nvshmem-cu12 nvshmem)" \
    || { cx_log "ERROR: DeepEP V2 NVSHMEM package root is unavailable"; return 1; }
  export EP_NCCL_ROOT_DIR EP_NVSHMEM_ROOT_DIR
  export LD_LIBRARY_PATH="$EP_NCCL_ROOT_DIR/lib:$EP_NVSHMEM_ROOT_DIR/lib:${LD_LIBRARY_PATH:-}"
  execution_id="${COLLECTIVEX_EXECUTION_ID:-manual}"
  [[ "$execution_id" =~ ^[A-Za-z0-9._-]+$ ]] \
    || { cx_log "ERROR: DeepEP V2 execution identity is invalid"; return 1; }
  # JIT CUBINs are per-execution evidence and must be node-local. A shared NFS cache lets
  # ranks on different nodes race the same compiler output and can trip compiler.hpp asserts.
  # The identical absolute path still lets ranks on one node share their cold build.
  export EP_JIT_CACHE_DIR="/tmp/collectivex-deepep-v2-jit-$execution_id"
  export EP_REUSE_NCCL_COMM=1
  export DEEPEP_V2_PR=605 DEEPEP_V2_FIX_PR=630 DEEPEP_V2_NCCL_CHECK_FIX_PR=640
  DEEPEP_V2_COMMIT="$CX_DEEPEP_V2_COMMIT"
  DEEPEP_V2_TREE="$CX_DEEPEP_V2_TREE"
  DEEPEP_V2_FMT_COMMIT="$CX_DEEPEP_V2_FMT_COMMIT"
  DEEPEP_V2_NCCL_CHECK_COMMIT="$CX_DEEPEP_V2_NCCL_CHECK_COMMIT"
  export DEEPEP_V2_COMMIT DEEPEP_V2_TREE DEEPEP_V2_FMT_COMMIT
  export DEEPEP_V2_NCCL_CHECK_COMMIT
  [ ! -L "$EP_JIT_CACHE_DIR" ] \
    || { cx_log "ERROR: DeepEP V2 JIT cache path is unsafe"; return 1; }
  if ! mkdir -p "$EP_JIT_CACHE_DIR" || ! chmod 700 "$EP_JIT_CACHE_DIR"; then
    cx_log "ERROR: DeepEP V2 JIT cache is unavailable"
    return 1
  fi
  unset EP_SUPPRESS_NCCL_CHECK
}

cx_probe_deepep_v2() {
  python3 - <<'PY'
import ctypes
import importlib.metadata as metadata
import inspect
import os

import torch

assert torch.__version__ == "2.10.0+cu130", torch.__version__
assert metadata.version("nvidia-nccl-cu13") == "2.30.4"
assert metadata.version("nvidia-nvshmem-cu12") == "3.3.9"
assert metadata.version("numpy") == "2.2.6"

import deep_ep
assert deep_ep.__version__ == "2.0.0", deep_ep.__version__
assert metadata.version("deep_ep") in {"2.0.0+fa8a9b1", "2.0.0+local"}, metadata.version("deep_ep")
assert inspect.isclass(deep_ep.ElasticBuffer)
assert deep_ep.ElasticBuffer.__name__ == "ElasticBuffer"
assert os.environ.get("EP_SUPPRESS_NCCL_CHECK") is None
with open("/proc/self/maps", encoding="utf-8") as handle:
    loaded_nccl = {
        os.path.realpath(line.rstrip().split()[-1])
        for line in handle
        if "libnccl.so" in line and os.path.isfile(line.rstrip().split()[-1])
    }
assert len(loaded_nccl) == 1
runtime_version = ctypes.c_int()
assert ctypes.CDLL(loaded_nccl.pop()).ncclGetVersion(ctypes.byref(runtime_version)) == 0
assert runtime_version.value == 23004, runtime_version.value
PY
}

cx_build_deepep_v2() {
  local root venv source ready lock_path arch cache_ready
  local revision="fa8a9b16898204afd347c663b89e65ef87dc6ce6"
  arch="$(cx_cuda_arch)" || return 1
  root="$(cx_deepep_v2_root)" || return 1
  venv="$root/venv"; source="$root/source"; ready="$root/.ready"
  lock_path="${root}.lock"
  command -v flock >/dev/null || { cx_log "ERROR: flock is required for DeepEP V2"; return 1; }
  mkdir -p "${root%/*}" || return 1
  cx_log "DeepEP V2: preparing PR #605 with upstream PR #630 and #640 fixes ($revision)"
  if ! (
    [ ! -L "$lock_path" ] \
      || { cx_log "ERROR: DeepEP V2 cache lock is unsafe"; exit 1; }
    (umask 077; : >> "$lock_path") && chmod 600 "$lock_path" \
      || { cx_log "ERROR: DeepEP V2 cache-lock-create failed"; exit 1; }
    exec 9<>"$lock_path" \
      || { cx_log "ERROR: DeepEP V2 cache-lock-open failed"; exit 1; }
    flock 9 \
      || { cx_log "ERROR: DeepEP V2 cache-lock-acquire failed"; exit 1; }
    cache_ready=0
    [ -f "$ready" ] && [ -x "$venv/bin/python" ] && [ -d "$source" ] && cache_ready=1
    if [ "$cache_ready" != 1 ]; then
      if [ -e "$root" ] || [ -L "$root" ]; then
        rm -rf "$root" \
          || { cx_log "ERROR: incomplete DeepEP V2 cache-reset failed"; exit 1; }
      fi
      mkdir -m 700 "$root" \
        || { cx_log "ERROR: DeepEP V2 cache-create failed"; exit 1; }
      python3 -m venv "$venv" \
        || { cx_log "ERROR: DeepEP V2 venv creation failed"; exit 1; }
      "$venv/bin/python" -m pip install -q --disable-pip-version-check --no-input \
        "pip==26.1.2" "setuptools==82.0.1" "wheel==0.47.0" "ninja==1.13.0" \
        "numpy==2.2.6" "nvidia-nvshmem-cu12==3.3.9" >&2 2>&1 \
        || { cx_log "ERROR: DeepEP V2 build-tool installation failed"; exit 1; }
      "$venv/bin/python" -m pip install -q --disable-pip-version-check --no-input \
        --index-url https://download.pytorch.org/whl/cu130 \
        --extra-index-url https://pypi.org/simple "torch==2.10.0" >&2 2>&1 \
        || { cx_log "ERROR: torch 2.10.0+cu130 installation failed"; exit 1; }
      # Torch pins NCCL 2.28.9; the PR #605 ElasticBuffer implementation requires 2.30.4.
      "$venv/bin/python" -m pip install -q --disable-pip-version-check --no-input \
        --force-reinstall --no-deps "nvidia-nccl-cu13==2.30.4" >&2 2>&1 \
        || { cx_log "ERROR: NCCL 2.30.4 installation failed"; exit 1; }
      cx_activate_deepep_v2 \
        || { cx_log "ERROR: DeepEP V2 environment activation failed"; exit 1; }
      cx_prepare_deepep_toolchain \
        || { cx_log "ERROR: DeepEP V2 toolchain preparation failed"; exit 1; }
      EP_NVSHMEM_ROOT_DIR="$NVSHMEM_DIR"
      export EP_NVSHMEM_ROOT_DIR
      cx_materialize_backend_source deepep-v2 "$source" \
        || { cx_log "ERROR: DeepEP V2 staged source is invalid"; exit 1; }
      (cd "$source" && TORCH_CUDA_ARCH_LIST="$arch" MAX_JOBS=16 \
        python3 -m pip install -q --no-build-isolation --no-deps --force-reinstall .) >&2 2>&1 \
        || { cx_log "ERROR: DeepEP V2 build failed"; exit 1; }
      cx_probe_deepep_v2 \
        || { cx_log "ERROR: DeepEP V2 import probe failed"; exit 1; }
      : > "$ready"
    fi
  ); then
    cx_log "ERROR: shared DeepEP V2 environment is incomplete"
    return 1
  fi
  cx_activate_deepep_v2 || return 1
  cx_prepare_deepep_toolchain || return 1
  EP_NVSHMEM_ROOT_DIR="$NVSHMEM_DIR"
  export EP_NVSHMEM_ROOT_DIR
  cx_probe_deepep_v2 || { cx_log "ERROR: DeepEP V2 shared runtime probe failed"; return 1; }
  cx_log "DeepEP V2 ready ($DEEPEP_V2_COMMIT, ElasticBuffer, NCCL Device API; LSA/Gin selected by adapter)"
}

# Build the pinned DeepEP `hybrid-ep` implementation. MNNVL remains one scale-up
# domain; true x86 scale-out uses the upstream DOCA/RDMA build explicitly.
cx_configure_deepep_hybrid_build() {
  local interface device rdma_name
  local -a interfaces devices
  unset HYBRID_EP_MULTINODE USE_NIXL RDMA_CORE_HOME DEEPEP_HYBRID_BUILD_MODE
  if [ "${CX_NODES:-1}" -le 1 ] || [ "${CX_TRANSPORT:-}" = mnnvl ]; then
    export DEEPEP_HYBRID_BUILD_MODE=intradomain
    return 0
  fi
  [ "$(uname -m)" = x86_64 ] \
    || { cx_log "ERROR: hybrid-ep scale-out is registered only on x86_64"; return 1; }
  [ -n "${GLOO_SOCKET_IFNAME:-}" ] && [ -n "${NCCL_IB_HCA:-}" ] \
    || { cx_log "ERROR: hybrid-ep scale-out network selectors are unavailable"; return 1; }
  IFS=, read -r -a interfaces <<< "$GLOO_SOCKET_IFNAME"
  for interface in "${interfaces[@]}"; do
    [ -d "/sys/class/net/$interface" ] \
      || { cx_log "ERROR: configured hybrid-ep socket interface is absent"; return 1; }
  done
  IFS=, read -r -a devices <<< "$NCCL_IB_HCA"
  for device in "${devices[@]}"; do
    rdma_name="$(cx_nccl_hca_device_name "$device")"
    [ -d "/sys/class/infiniband/$rdma_name" ] \
      || { cx_log "ERROR: configured hybrid-ep RDMA device is absent"; return 1; }
  done
  command -v make >/dev/null \
    || { cx_log "ERROR: make is required for hybrid-ep scale-out"; return 1; }
  [ -r /usr/include/infiniband/verbs.h ] && [ -r /usr/include/infiniband/mlx5dv.h ] \
    || { cx_log "ERROR: pinned hybrid-ep RDMA headers are unavailable"; return 1; }
  python3 - <<'PY' >/dev/null 2>&1 || {
import ctypes.util
import sys
sys.exit(0 if all(ctypes.util.find_library(name) for name in ("ibverbs", "mlx5")) else 1)
PY
    cx_log "ERROR: pinned hybrid-ep RDMA libraries are unavailable"
    return 1
  }
  export HYBRID_EP_MULTINODE=1 USE_NIXL=0 RDMA_CORE_HOME=/usr
  export DEEPEP_HYBRID_BUILD_MODE=multinode-doca
}

cx_build_deepep_hybrid() {
  local arch revision="$CX_DEEPEP_HYBRID_COMMIT" tree="$CX_DEEPEP_HYBRID_TREE"
  local build_root lock_path cache_ready build_mode
  export DEEPEP_COMMIT="$revision" DEEPEP_TREE="$tree"
  arch="$(cx_cuda_arch)" || return 1
  cx_configure_deepep_hybrid_build || return 1
  build_mode="$DEEPEP_HYBRID_BUILD_MODE"
  build_root="$PWD/.cx_backend/deepep-hybrid-${arch/./}-${build_mode}"
  lock_path="${build_root}.lock"
  cx_log "DeepEP hybrid-ep: building $revision for CUDA target $arch"
  unset NVSHMEM_DIR
  cx_prepare_cuda_cccl || return 1
  command -v flock >/dev/null || { cx_log "ERROR: flock is required for hybrid-ep"; return 1; }
  mkdir -p "$PWD/.cx_backend" || return 1
  if ! (
    [ ! -L "$lock_path" ] || exit 1
    (umask 077; : >> "$lock_path") && chmod 600 "$lock_path" || exit 1
    exec 9<>"$lock_path" || exit 1
    flock 9 || exit 1
    cache_ready=0
    [ -f "$build_root/deep_ep_cpp.so" ] && [ -f "$build_root/hybrid_ep_cpp.so" ] \
      && cache_ready=1
    if [ "$cache_ready" != 1 ]; then
      cx_materialize_backend_source deepep-hybrid "$build_root" \
        || { cx_log "ERROR: hybrid-ep staged source is invalid"; exit 1; }
      if [ "$build_mode" = multinode-doca ]; then
        [ "$(cx_git_in_tree "$build_root/third-party/nccl" rev-parse HEAD 2>/dev/null)" \
          = "$CX_DEEPEP_HYBRID_NCCL_COMMIT" ] \
          || { cx_log "ERROR: pinned hybrid-ep NCCL transport source is absent"; exit 1; }
      fi
      (cd "$build_root" && TORCH_CUDA_ARCH_LIST="$arch" MAX_JOBS=16 \
        python3 setup.py build_ext --inplace) >&2 2>&1 \
        || { cx_log "ERROR: hybrid-ep build failed"; exit 1; }
    fi
  ); then
    cx_log "ERROR: shared hybrid-ep build is incomplete"
    return 1
  fi
  export PYTHONPATH="$build_root:${PYTHONPATH:-}"
  python3 -c "import deep_ep; assert hasattr(deep_ep,'HybridEPBuffer'); print('built hybrid-ep deep_ep', getattr(deep_ep,'__version__','?'))" >&2 \
    || { cx_log "ERROR: hybrid-ep import / HybridEPBuffer missing after build"; return 1; }
  cx_log "DeepEP hybrid-ep ready ($DEEPEP_COMMIT, mode=$build_mode)"
}

# Rack build and rank steps may enter different container instances. Persist each node's
# loader/import path and build identity on the shared staged mount, then require it from every rank.
cx_persist_backend_env() {
  local root="$PWD/.cx_backend/env" node_id="${SLURM_NODEID:-0}" path temporary name
  local -a names=(PATH VIRTUAL_ENV LD_LIBRARY_PATH PYTHONPATH CUDA_HOME CPATH NVCC_PREPEND_FLAGS
    NVSHMEM_DIR DEEPEP_COMMIT DEEPEP_TREE
    EP_NCCL_ROOT_DIR EP_NVSHMEM_ROOT_DIR EP_JIT_CACHE_DIR EP_REUSE_NCCL_COMM
    DEEPEP_V2_PR DEEPEP_V2_FIX_PR DEEPEP_V2_NCCL_CHECK_FIX_PR DEEPEP_V2_COMMIT
    DEEPEP_V2_TREE DEEPEP_V2_FMT_COMMIT DEEPEP_V2_NCCL_CHECK_COMMIT
    HYBRID_EP_MULTINODE USE_NIXL RDMA_CORE_HOME DEEPEP_HYBRID_BUILD_MODE)
  [[ "$node_id" =~ ^[0-9]+$ ]] || return 1
  mkdir -p "$root" || return 1
  chmod 700 "$root" || return 1
  temporary="$(mktemp "$root/.node-${node_id}.XXXXXX")" || return 1
  chmod 600 "$temporary" || { rm -f "$temporary"; return 1; }
  for name in "${names[@]}"; do
    if declare -p "$name" >/dev/null 2>&1; then
      printf 'export %s=%q\n' "$name" "${!name}" >> "$temporary" \
        || { rm -f "$temporary"; return 1; }
    fi
  done
  path="$root/node-${node_id}.sh"
  mv -f -- "$temporary" "$path" || { rm -f "$temporary"; return 1; }
}

# Validate private scale-out selectors on every allocated compute node before a
# backend can initialize or build transport code.
cx_probe_scaleout_network() {
  local interface device rdma_name
  local -a interfaces devices
  if [ "${CX_NODES:-1}" -le 1 ] || [ "${CX_TRANSPORT:-}" = mnnvl ]; then
    return 0
  fi
  cx_restore_exact_hca_selector || return 1
  [ -n "${GLOO_SOCKET_IFNAME:-}" ] && [ -n "${NCCL_IB_HCA:-}" ] \
    || { cx_log "ERROR: scale-out network selectors are unavailable"; return 1; }
  IFS=, read -r -a interfaces <<< "$GLOO_SOCKET_IFNAME"
  for interface in "${interfaces[@]}"; do
    [ -d "/sys/class/net/$interface" ] \
      || { cx_log "ERROR: configured scale-out socket interface is absent"; return 1; }
  done
  IFS=, read -r -a devices <<< "$NCCL_IB_HCA"
  for device in "${devices[@]}"; do
    rdma_name="$(cx_nccl_hca_device_name "$device")"
    [ -d "/sys/class/infiniband/$rdma_name" ] \
      || { cx_log "ERROR: configured scale-out RDMA device is absent"; return 1; }
  done
}

# Prepare and probe one backend without running a benchmark. The same hook is used
# by normal in-container runs and by rack launchers' persistent build-only step.
cx_prepare_backend() {
  local backend="${1:-}"
  case "$backend" in
    deepep-v2)
      cx_build_deepep_v2 || return 1
      ;;
    deepep-hybrid)
      cx_build_deepep_hybrid || return 1
      ;;
    mori)
      python3 -c "import mori" \
        || { cx_log "ERROR: MoRI backend import failed"; return 1; }
      ;;
    *)
      cx_log "ERROR: unknown backend preparation request"
      return 1
      ;;
  esac
}

prepare_backend_or_record() {
  local backend="$1"
  if cx_prepare_backend "$backend"; then
    return 0
  fi
  cx_log "WARN: $backend preparation failed"
  return 1
}

# dispatch_bench runs the CURRENT CX_BENCH (+ CX_* config env) once. The sweep workflow runs many
# of these per allocation (SHARD mode below), reusing this single container + its built backend.
dispatch_bench() {
  case "$CX_BENCH" in
    deepep-v2|deepep-hybrid|mori)
      prepare_backend_or_record "$CX_BENCH" && run_ep_suite "$CX_BENCH"
      ;;
    *)
      cx_die "unknown CX_BENCH=$CX_BENCH (want deepep-v2|deepep-hybrid|mori)"
      ;;
  esac
}

rc=0
cx_load_network_control_mode "$PWD" || cx_die "cannot resolve network control mode"
cx_apply_network_profile "${CX_NODES:-1}" "${CX_TRANSPORT:-}"
# Build-only mode: rack launchers run the shared backend preparation hook once per
# node inside a persistent named container, then direct rank processes reuse it.
if [ -n "${CX_BUILD_ONLY:-}" ]; then
  if cx_probe_scaleout_network && cx_prepare_backend "${CX_BENCH:-}"; then
    cx_persist_backend_env || rc=1
  else
    rc=1
  fi
  cx_log "backend preparation: bench=${CX_BENCH:-unknown} rc=$rc"
  exit "$rc"
fi
if [ -n "${CX_SHARD_FILE:-}" ]; then
  # SHARD/SWEEP mode (collectivex-sweep.yml): run EVERY case of this shard in THIS one allocation.
  # All cases share (sku, backend, nodes), so backend preparation is paid once and cached.
  ncases="$(python3 -c "import json;print(len(json.load(open('$CX_SHARD_FILE'))['cases']))")"
  # The iterable benchmark version is a shard-level scalar (identical for every case);
  # export it once so run_ep copies it verbatim into each emitted result.
  CX_VERSION="$(python3 -c "import json;print(json.load(open('$CX_SHARD_FILE'))['version'])")"
  export CX_VERSION
  cx_log "SHARD mode: $ncases case(s) in one allocation (shard=$CX_SHARD_FILE, version=$CX_VERSION)"
  _cx_ts_base="$CX_TS"   # per-case CX_TS suffix below keeps each case's result file UNIQUE (else
                         # cases sharing backend+phase overwrite each other at the same timestamp).
  ci=0
  failed_cases=0
  while [ "$ci" -lt "$ncases" ]; do
    CX_TS="${_cx_ts_base}-c$(printf '%03d' "$ci")"
    export CX_TS
    # Map varying case fields plus the frozen v1 defaults into CX_* env.
    _exports="$(python3 - "$CX_SHARD_FILE" "$ci" <<'PY'
import json, sys, shlex
c = json.load(open(sys.argv[1]))["cases"][int(sys.argv[2])]
def g(k, d=""):
    v = c.get(k, d); return "" if v is None else str(v)
env = {
  "CX_BENCH": g("backend"),
  "CX_MODE": g("mode", "normal"),
  "CX_ROUTING": g("routing", "uniform"), "CX_PHASE": g("phase", "decode"),
  "CX_EP": g("ep", "1"),
  "CX_CASE_ID": g("case_id"), "CX_SUITE": g("suite"), "CX_WORKLOAD_NAME": g("workload"),
  "CX_HIDDEN": g("hidden"), "CX_TOPK": g("topk"), "CX_EXPERTS": g("experts"),
  "CX_TOKENS_LADDER": g("ladder"), "CX_CANONICAL": ("1" if c.get("canonical") else ""),
  "CX_NODES": g("nodes"), "CX_GPUS_PER_NODE": g("gpus_per_node"),
  "CX_SCALE_UP_DOMAIN": g("scale_up_domain"), "CX_SCOPE": g("scope"),
  "CX_SCALE_UP_TRANSPORT": g("scale_up_transport"),
  "CX_SCALE_OUT_TRANSPORT": g("scale_out_transport"),
  "CX_TRANSPORT": g("transport"), "CX_TOPO": g("topology_class"),
  "CX_SAMPLES_PER_POINT": g("samples_per_point"),
  "CX_WARMUP_SEMANTICS": g("warmup_semantics"),
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
    cx_apply_network_profile "$CX_NODES" "$CX_TRANSPORT"
    cx_log "  [$((ci+1))/$ncases] $CX_BENCH $CX_MODE/$CX_PHASE routing=$CX_ROUTING"
    _cx_case_ts="$CX_TS"
    CX_TS="${_cx_case_ts}-a01"
    export CX_ATTEMPT_ID=1 CX_TS
    dispatch_bench || {
      failed_cases=$((failed_cases+1))
      cx_log "  [$((ci+1))/$ncases] $CX_BENCH case FAILED; failed-case record preserved"
    }
    export CX_TS="$_cx_case_ts"
    ci=$((ci + 1))
  done
  if [ "${failed_cases:-0}" -gt 0 ]; then
    cx_log "SHARD done: $failed_cases/$ncases case(s) failed"
    rc=1
  fi
  # The base timestamp matches every per-case file, so the final summary covers the whole shard.
  export CX_TS="$_cx_ts_base"
else
  _cx_single_ts="$CX_TS"
  CX_TS="${_cx_single_ts}-a01"
  export CX_ATTEMPT_ID=1 CX_TS
  dispatch_bench || rc=1
fi

# Summary table for the log; also fails the job if no valid results were produced.
python3 summarize.py --results-dir results --runner "$CX_RUNNER" --ts "$CX_TS" || rc=1
exit "$rc"
