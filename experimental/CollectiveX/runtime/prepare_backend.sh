#!/usr/bin/env bash
# CollectiveX — in-container backend preparation, one task per node.
#
# Runs INSIDE the persistent named container once per allocated node before any
# benchmark case: builds or validates the COLLX_BENCH backend, then persists the
# loader environment to .collx_backend/env/node-N.sh for the per-case rank steps
# (collx_source_backend_env). Benchmark cases are driven from the host by
# collx_run_shard with run_ep.py argv decoded from the shard control; no per-case
# configuration enters the container as environment.
#
# Required env (exported by the adapter): COLLX_RUNNER
# Selector: COLLX_BENCH = deepep-v2 | deepep-hybrid | mori
set -euo pipefail

cd /ix/experimental/CollectiveX
# shellcheck source=../runtime/common.sh
source runtime/common.sh

: "${COLLX_RUNNER:?COLLX_RUNNER not set}"
: "${COLLX_BENCH:?COLLX_BENCH not set}"

collx_log "backend preparation: runner=$COLLX_RUNNER bench=$COLLX_BENCH nodes=${COLLX_NODES:-1}"

# Resolve and verify the actual CUDA target before compiling source kernels.
collx_cuda_arch() {
  local expected detected
  case "$COLLX_RUNNER" in
    h100*|h200*) expected="9.0" ;;
    b200*|gb200*) expected="10.0" ;;
    b300*|gb300*) expected="10.3" ;;
    *) collx_log "ERROR: no CUDA target registered for $COLLX_RUNNER"; return 1 ;;
  esac
  detected="$(python3 - <<'PY'
import torch

major, minor = torch.cuda.get_device_capability()
print(f"{major}.{minor}")
PY
)" || return 1
  [ "$detected" = "$expected" ] || {
    collx_log "ERROR: $COLLX_RUNNER expected CUDA target $expected, detected $detected"
    return 1
  }
  printf '%s' "$detected"
}

collx_nvidia_package_root() {
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

collx_prepare_cuda_cccl() {
  local cccl="" candidate cuda_home nvcc
  nvcc="$(command -v nvcc)" \
    || { collx_log "ERROR: CUDA nvcc is unavailable"; return 1; }
  nvcc="$(readlink -f -- "$nvcc")" \
    || { collx_log "ERROR: CUDA nvcc cannot be resolved"; return 1; }
  case "$nvcc" in
    */bin/nvcc) cuda_home="${nvcc%/bin/nvcc}" ;;
    *) collx_log "ERROR: CUDA nvcc has an unexpected path"; return 1 ;;
  esac
  [ -x "$cuda_home/bin/nvcc" ] && [ -d "$cuda_home/include" ] \
    && [ -d "$cuda_home/lib64" ] \
    || { collx_log "ERROR: CUDA toolkit root is incomplete"; return 1; }
  for candidate in "$cuda_home"/targets/*/include/cccl; do
    if [ -d "$candidate" ]; then
      cccl="$candidate"
      break
    fi
  done
  [ -n "$cccl" ] || { collx_log "ERROR: CUDA CCCL headers are unavailable"; return 1; }
  export CUDA_HOME="$cuda_home" COLLX_CUDA_CCCL="$cccl"
  export CPATH="$cccl:${CPATH:-}"
  export NVCC_PREPEND_FLAGS="-I$cccl ${NVCC_PREPEND_FLAGS:-}"
}

collx_prepare_deepep_toolchain() {
  local packaged overlay path root temporary
  packaged="$(collx_nvidia_package_root nvidia-nvshmem-cu12 nvshmem)" \
    || { collx_log "ERROR: nvidia.nvshmem is unavailable"; return 1; }
  root="$(collx_deepep_v2_root)" || return 1
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
    collx_log "ERROR: DeepEP V2 NVSHMEM overlay is invalid"
    return 1
  fi
  NVSHMEM_DIR="$overlay"
  export NVSHMEM_DIR
  collx_prepare_cuda_cccl || return 1
  export LD_LIBRARY_PATH="$NVSHMEM_DIR/lib:${LD_LIBRARY_PATH:-}"
}

# DeepEP V2 is PR #605's ElasticBuffer implementation with upstream PR #630's pure scale-up
# initialization fix and PR #640's exact libnccl mapping check. Canonical launchers stage the
# pinned source and mount a private cluster-local build cache at /cx-cache.
collx_deepep_v2_root() {
  local arch cpu base image
  arch="$(collx_cuda_arch)" || return 1
  cpu="$(uname -m)"
  [[ "$cpu" =~ ^[A-Za-z0-9._-]+$ ]] || return 1
  base="${COLLX_BACKEND_CACHE_ROOT:-}"
  [[ "$base" = /* ]] || return 1
  image="$(printf '%s' "${COLLECTIVEX_IMAGE:-manual}" | tr -cs 'A-Za-z0-9_.-' '-')" \
    || return 1
  image="${image#-}"; image="${image%-}"
  [ -n "$image" ] || return 1
  printf '%s/deepep-v2-v3-%s-sm%s-%s-torch2.10.0-cu130-nccl2.30.4-nvshmem3.3.9-%s-%s-%s-%s' \
    "$base" "$cpu" "${arch/./}" "$image" \
    "${COLLX_DEEPEP_V2_COMMIT:0:12}" "${COLLX_DEEPEP_V2_TREE:0:12}" \
    "${COLLX_DEEPEP_V2_FMT_COMMIT:0:12}" "${COLLX_DEEPEP_V2_NCCL_CHECK_COMMIT:0:12}"
}

collx_activate_deepep_v2() {
  local root venv venv_site execution_id
  root="$(collx_deepep_v2_root)" || return 1
  venv="$root/venv"
  [ -x "$venv/bin/python" ] \
    || { collx_log "ERROR: DeepEP V2 venv interpreter is unavailable"; return 1; }
  export VIRTUAL_ENV="$venv"
  export PATH="$venv/bin:${PATH#"$venv/bin:"}"
  # The per-case probe and rank steps re-source this env in a fresh container task where
  # PATH alone has proven insufficient to select the venv interpreter over the image's
  # bundled deep_ep (the amd64 sglang image ships a 1.2.1 wheel with no ElasticBuffer).
  # Pinning the venv site-packages on PYTHONPATH makes the from-source 2.0.0 build win
  # under either interpreter, mirroring the deepep-hybrid path that already runs green here.
  for venv_site in "$venv"/lib/python*/site-packages; do break; done
  [ -d "$venv_site" ] \
    || { collx_log "ERROR: DeepEP V2 venv site-packages is unavailable"; return 1; }
  export PYTHONPATH="$venv_site${PYTHONPATH:+:$PYTHONPATH}"
  EP_NCCL_ROOT_DIR="$(collx_nvidia_package_root nvidia-nccl-cu13 nccl)" \
    || { collx_log "ERROR: DeepEP V2 NCCL package root is unavailable"; return 1; }
  EP_NVSHMEM_ROOT_DIR="$(collx_nvidia_package_root nvidia-nvshmem-cu12 nvshmem)" \
    || { collx_log "ERROR: DeepEP V2 NVSHMEM package root is unavailable"; return 1; }
  export EP_NCCL_ROOT_DIR EP_NVSHMEM_ROOT_DIR
  export LD_LIBRARY_PATH="$EP_NCCL_ROOT_DIR/lib:$EP_NVSHMEM_ROOT_DIR/lib:${LD_LIBRARY_PATH:-}"
  execution_id="${COLLECTIVEX_EXECUTION_ID:-manual}"
  [[ "$execution_id" =~ ^[A-Za-z0-9._-]+$ ]] \
    || { collx_log "ERROR: DeepEP V2 execution identity is invalid"; return 1; }
  # JIT CUBINs are per-execution evidence and must be node-local. A shared NFS cache lets
  # ranks on different nodes race the same compiler output and can trip compiler.hpp asserts.
  # The identical absolute path still lets ranks on one node share their cold build.
  export EP_JIT_CACHE_DIR="/tmp/collectivex-deepep-v2-jit-$execution_id"
  export EP_REUSE_NCCL_COMM=1
  [ ! -L "$EP_JIT_CACHE_DIR" ] \
    || { collx_log "ERROR: DeepEP V2 JIT cache path is unsafe"; return 1; }
  if ! mkdir -p "$EP_JIT_CACHE_DIR" || ! chmod 700 "$EP_JIT_CACHE_DIR"; then
    collx_log "ERROR: DeepEP V2 JIT cache is unavailable"
    return 1
  fi
  unset EP_SUPPRESS_NCCL_CHECK
}

collx_probe_deepep_v2() {
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

collx_build_deepep_v2() {
  local root venv source ready lock_path arch cache_ready
  local revision="$COLLX_DEEPEP_V2_COMMIT"
  arch="$(collx_cuda_arch)" || return 1
  root="$(collx_deepep_v2_root)" || return 1
  venv="$root/venv"; source="$root/source"; ready="$root/.ready"
  lock_path="${root}.lock"
  command -v flock >/dev/null || { collx_log "ERROR: flock is required for DeepEP V2"; return 1; }
  mkdir -p "${root%/*}" || return 1
  collx_log "DeepEP V2: preparing PR #605 with upstream PR #630 and #640 fixes ($revision)"
  if ! (
    [ ! -L "$lock_path" ] \
      || { collx_log "ERROR: DeepEP V2 cache lock is unsafe"; exit 1; }
    (umask 077; : >> "$lock_path") && chmod 600 "$lock_path" \
      || { collx_log "ERROR: DeepEP V2 cache-lock-create failed"; exit 1; }
    exec 9<>"$lock_path" \
      || { collx_log "ERROR: DeepEP V2 cache-lock-open failed"; exit 1; }
    flock 9 \
      || { collx_log "ERROR: DeepEP V2 cache-lock-acquire failed"; exit 1; }
    cache_ready=0
    [ -f "$ready" ] && [ -x "$venv/bin/python" ] && [ -d "$source" ] && cache_ready=1
    if [ "$cache_ready" != 1 ]; then
      if [ -e "$root" ] || [ -L "$root" ]; then
        rm -rf "$root" \
          || { collx_log "ERROR: incomplete DeepEP V2 cache-reset failed"; exit 1; }
      fi
      mkdir -m 700 "$root" \
        || { collx_log "ERROR: DeepEP V2 cache-create failed"; exit 1; }
      python3 -m venv "$venv" \
        || { collx_log "ERROR: DeepEP V2 venv creation failed"; exit 1; }
      "$venv/bin/python" -m pip install -q --disable-pip-version-check --no-input \
        "pip==26.1.2" "setuptools==82.0.1" "wheel==0.47.0" "ninja==1.13.0" \
        "numpy==2.2.6" "nvidia-nvshmem-cu12==3.3.9" >&2 2>&1 \
        || { collx_log "ERROR: DeepEP V2 build-tool installation failed"; exit 1; }
      "$venv/bin/python" -m pip install -q --disable-pip-version-check --no-input \
        --index-url https://download.pytorch.org/whl/cu130 \
        --extra-index-url https://pypi.org/simple "torch==2.10.0" >&2 2>&1 \
        || { collx_log "ERROR: torch 2.10.0+cu130 installation failed"; exit 1; }
      # Torch pins NCCL 2.28.9; the PR #605 ElasticBuffer implementation requires 2.30.4.
      "$venv/bin/python" -m pip install -q --disable-pip-version-check --no-input \
        --force-reinstall --no-deps "nvidia-nccl-cu13==2.30.4" >&2 2>&1 \
        || { collx_log "ERROR: NCCL 2.30.4 installation failed"; exit 1; }
      collx_activate_deepep_v2 \
        || { collx_log "ERROR: DeepEP V2 environment activation failed"; exit 1; }
      collx_prepare_deepep_toolchain \
        || { collx_log "ERROR: DeepEP V2 toolchain preparation failed"; exit 1; }
      EP_NVSHMEM_ROOT_DIR="$NVSHMEM_DIR"
      export EP_NVSHMEM_ROOT_DIR
      collx_materialize_backend_source deepep-v2 "$source" \
        || { collx_log "ERROR: DeepEP V2 staged source is invalid"; exit 1; }
      (cd "$source" && TORCH_CUDA_ARCH_LIST="$arch" MAX_JOBS=16 \
        python3 -m pip install -q --no-build-isolation --no-deps --force-reinstall .) >&2 2>&1 \
        || { collx_log "ERROR: DeepEP V2 build failed"; exit 1; }
      collx_probe_deepep_v2 \
        || { collx_log "ERROR: DeepEP V2 import probe failed"; exit 1; }
      : > "$ready"
    fi
  ); then
    collx_log "ERROR: shared DeepEP V2 environment is incomplete"
    return 1
  fi
  collx_activate_deepep_v2 || return 1
  collx_prepare_deepep_toolchain || return 1
  EP_NVSHMEM_ROOT_DIR="$NVSHMEM_DIR"
  export EP_NVSHMEM_ROOT_DIR
  collx_probe_deepep_v2 || { collx_log "ERROR: DeepEP V2 shared runtime probe failed"; return 1; }
  collx_log "DeepEP V2 ready ($COLLX_DEEPEP_V2_COMMIT, ElasticBuffer, NCCL Device API; LSA/Gin selected by adapter)"
}

# Build the pinned DeepEP `hybrid-ep` implementation. MNNVL remains one scale-up
# domain; true x86 scale-out uses the upstream DOCA/RDMA build explicitly.
collx_configure_deepep_hybrid_build() {
  local interface device rdma_name
  local -a interfaces devices
  unset HYBRID_EP_MULTINODE USE_NIXL RDMA_CORE_HOME DEEPEP_HYBRID_BUILD_MODE
  if [ "${COLLX_NODES:-1}" -le 1 ] || [ "${COLLX_TRANSPORT:-}" = mnnvl ]; then
    export DEEPEP_HYBRID_BUILD_MODE=intradomain
    return 0
  fi
  [ "$(uname -m)" = x86_64 ] \
    || { collx_log "ERROR: hybrid-ep scale-out is registered only on x86_64"; return 1; }
  [ -n "${GLOO_SOCKET_IFNAME:-}" ] && [ -n "${NCCL_IB_HCA:-}" ] \
    || { collx_log "ERROR: hybrid-ep scale-out network selectors are unavailable"; return 1; }
  IFS=, read -r -a interfaces <<< "$GLOO_SOCKET_IFNAME"
  for interface in "${interfaces[@]}"; do
    [ -d "/sys/class/net/$interface" ] \
      || { collx_log "ERROR: configured hybrid-ep socket interface is absent"; return 1; }
  done
  IFS=, read -r -a devices <<< "$NCCL_IB_HCA"
  for device in "${devices[@]}"; do
    rdma_name="$(collx_nccl_hca_device_name "$device")"
    [ -d "/sys/class/infiniband/$rdma_name" ] \
      || { collx_log "ERROR: configured hybrid-ep RDMA device is absent"; return 1; }
  done
  command -v make >/dev/null \
    || { collx_log "ERROR: make is required for hybrid-ep scale-out"; return 1; }
  [ -r /usr/include/infiniband/verbs.h ] && [ -r /usr/include/infiniband/mlx5dv.h ] \
    || { collx_log "ERROR: pinned hybrid-ep RDMA headers are unavailable"; return 1; }
  python3 - <<'PY' >/dev/null 2>&1 || {
import ctypes.util
import sys
sys.exit(0 if all(ctypes.util.find_library(name) for name in ("ibverbs", "mlx5")) else 1)
PY
    collx_log "ERROR: pinned hybrid-ep RDMA libraries are unavailable"
    return 1
  }
  export HYBRID_EP_MULTINODE=1 USE_NIXL=0 RDMA_CORE_HOME=/usr
  export DEEPEP_HYBRID_BUILD_MODE=multinode-doca
}

collx_build_deepep_hybrid() {
  local arch revision="$COLLX_DEEPEP_HYBRID_COMMIT" tree="$COLLX_DEEPEP_HYBRID_TREE"
  local build_root lock_path cache_ready build_mode
  export DEEPEP_COMMIT="$revision" DEEPEP_TREE="$tree"
  arch="$(collx_cuda_arch)" || return 1
  collx_configure_deepep_hybrid_build || return 1
  build_mode="$DEEPEP_HYBRID_BUILD_MODE"
  build_root="$PWD/.collx_backend/deepep-hybrid-${arch/./}-${build_mode}"
  lock_path="${build_root}.lock"
  collx_log "DeepEP hybrid-ep: building $revision for CUDA target $arch"
  unset NVSHMEM_DIR
  collx_prepare_cuda_cccl || return 1
  command -v flock >/dev/null || { collx_log "ERROR: flock is required for hybrid-ep"; return 1; }
  mkdir -p "$PWD/.collx_backend" || return 1
  if ! (
    [ ! -L "$lock_path" ] || exit 1
    (umask 077; : >> "$lock_path") && chmod 600 "$lock_path" || exit 1
    exec 9<>"$lock_path" || exit 1
    flock 9 || exit 1
    cache_ready=0
    [ -f "$build_root/deep_ep_cpp.so" ] && [ -f "$build_root/hybrid_ep_cpp.so" ] \
      && cache_ready=1
    if [ "$cache_ready" != 1 ]; then
      collx_materialize_backend_source deepep-hybrid "$build_root" \
        || { collx_log "ERROR: hybrid-ep staged source is invalid"; exit 1; }
      if [ "$build_mode" = multinode-doca ]; then
        [ "$(collx_git_in_tree "$build_root/third-party/nccl" rev-parse HEAD 2>/dev/null)" \
          = "$COLLX_DEEPEP_HYBRID_NCCL_COMMIT" ] \
          || { collx_log "ERROR: pinned hybrid-ep NCCL transport source is absent"; exit 1; }
      fi
      (cd "$build_root" && TORCH_CUDA_ARCH_LIST="$arch" MAX_JOBS=16 \
        python3 setup.py build_ext --inplace) >&2 2>&1 \
        || { collx_log "ERROR: hybrid-ep build failed"; exit 1; }
    fi
  ); then
    collx_log "ERROR: shared hybrid-ep build is incomplete"
    return 1
  fi
  export PYTHONPATH="$build_root:${PYTHONPATH:-}"
  python3 -c "import deep_ep; assert hasattr(deep_ep,'HybridEPBuffer'); print('built hybrid-ep deep_ep', getattr(deep_ep,'__version__','?'))" >&2 \
    || { collx_log "ERROR: hybrid-ep import / HybridEPBuffer missing after build"; return 1; }
  collx_log "DeepEP hybrid-ep ready ($DEEPEP_COMMIT, mode=$build_mode)"
}

# Rack build and rank steps may enter different container instances. Persist each node's
# loader/import path and build identity on the shared staged mount, then require it from every rank.
collx_persist_backend_env() {
  local root="$PWD/.collx_backend/env" node_id="${SLURM_NODEID:-0}" path temporary name
  local -a names=(PATH VIRTUAL_ENV LD_LIBRARY_PATH PYTHONPATH CUDA_HOME CPATH NVCC_PREPEND_FLAGS
    NVSHMEM_DIR DEEPEP_COMMIT DEEPEP_TREE
    EP_NCCL_ROOT_DIR EP_NVSHMEM_ROOT_DIR EP_JIT_CACHE_DIR EP_REUSE_NCCL_COMM
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
collx_probe_scaleout_network() {
  local interface device rdma_name
  local -a interfaces devices
  if [ "${COLLX_NODES:-1}" -le 1 ] || [ "${COLLX_TRANSPORT:-}" = mnnvl ]; then
    return 0
  fi
  collx_restore_exact_hca_selector || return 1
  [ -n "${GLOO_SOCKET_IFNAME:-}" ] && [ -n "${NCCL_IB_HCA:-}" ] \
    || { collx_log "ERROR: scale-out network selectors are unavailable"; return 1; }
  IFS=, read -r -a interfaces <<< "$GLOO_SOCKET_IFNAME"
  for interface in "${interfaces[@]}"; do
    [ -d "/sys/class/net/$interface" ] \
      || { collx_log "ERROR: configured scale-out socket interface is absent"; return 1; }
  done
  IFS=, read -r -a devices <<< "$NCCL_IB_HCA"
  for device in "${devices[@]}"; do
    rdma_name="$(collx_nccl_hca_device_name "$device")"
    [ -d "/sys/class/infiniband/$rdma_name" ] \
      || { collx_log "ERROR: configured scale-out RDMA device is absent"; return 1; }
  done
}

# Prepare and probe one backend without running a benchmark.
collx_prepare_backend() {
  local backend="${1:-}"
  case "$backend" in
    deepep-v2)
      collx_build_deepep_v2 || return 1
      ;;
    deepep-hybrid)
      collx_build_deepep_hybrid || return 1
      ;;
    mori)
      python3 -c "import mori" \
        || { collx_log "ERROR: MoRI backend import failed"; return 1; }
      ;;
    *)
      collx_log "ERROR: unknown backend preparation request"
      return 1
      ;;
  esac
}

rc=0
collx_apply_network_profile "${COLLX_NODES:-1}" "${COLLX_TRANSPORT:-}"
if collx_probe_scaleout_network && collx_prepare_backend "$COLLX_BENCH"; then
  collx_persist_backend_env || rc=1
else
  rc=1
fi
collx_log "backend preparation: bench=$COLLX_BENCH rc=$rc"
exit "$rc"
