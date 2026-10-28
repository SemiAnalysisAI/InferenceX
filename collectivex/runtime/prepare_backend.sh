#!/usr/bin/env bash
# Prepare one backend per allocated node and persist its rank environment.
set -eo pipefail

cd /ix/collectivex
# shellcheck source=../runtime/common.sh
source runtime/common.sh

: "${COLLX_RUNNER:?COLLX_RUNNER not set}"
: "${COLLX_BENCH:?COLLX_BENCH not set}"

collx_log "backend preparation: runner=$COLLX_RUNNER bench=$COLLX_BENCH nodes=${COLLX_NODES:-1}"

# Fresh rank tasks source only these backend-created values. Network variables
# are reapplied by the rank wrapper from the platform profile.
readonly -a RANK_ENV_VARS=(
  PATH VIRTUAL_ENV LD_LIBRARY_PATH LIBRARY_PATH PYTHONPATH CUDA_HOME CPATH NVCC_PREPEND_FLAGS
  NVSHMEM_DIR EP_NCCL_ROOT_DIR EP_NVSHMEM_ROOT_DIR EP_JIT_CACHE_DIR
  EP_REUSE_NCCL_COMM NCCL_CUMEM_ENABLE UCCL_EP_ENABLE_AGGRESSIVE_ATOMIC COLLX_OFI_NCCL_LIB_DIR FLASHINFER_WORKSPACE_BASE
)
readonly -a DEEPEP_RANK_UNSETS=(EP_SUPPRESS_NCCL_CHECK)

# discovery

cuda_arch() {
  local expected detected
  expected="$(python3 - "$COLLX_RUNNER" <<'PY'
import json, sys
arch = json.load(open("configs/platform_config.json"))["platforms"][sys.argv[1]]["arch"]
digits = arch.removeprefix("sm")
print(f"{digits[:-1]}.{digits[-1]}" if arch.startswith("sm") and digits.isdigit() else "")
PY
)" || { collx_log "ERROR: no platform registry entry for $COLLX_RUNNER"; return 1; }
  [ -n "$expected" ] || {
    collx_log "ERROR: no CUDA target registered for $COLLX_RUNNER"; return 1
  }
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

nvidia_package_root() {
  local python="$1" package="$2" component="$3"
  "$python" - "$package" "$component" <<'PY'
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

cuda_toolchain_paths() {
  local cccl="" candidate cuda_home nvcc
  nvcc="$(command -v nvcc)" || { collx_log "ERROR: CUDA nvcc is unavailable"; return 1; }
  nvcc="$(readlink -f -- "$nvcc")" || { collx_log "ERROR: CUDA nvcc cannot be resolved"; return 1; }
  case "$nvcc" in
    */bin/nvcc) cuda_home="${nvcc%/bin/nvcc}" ;;
    *) collx_log "ERROR: CUDA nvcc has an unexpected path"; return 1 ;;
  esac
  [ -x "$cuda_home/bin/nvcc" ] && [ -d "$cuda_home/include" ] && [ -d "$cuda_home/lib64" ] \
    || { collx_log "ERROR: CUDA toolkit root is incomplete"; return 1; }
  for candidate in "$cuda_home"/targets/*/include/cccl; do
    if [ -d "$candidate" ]; then
      cccl="$candidate"
      break
    fi
  done
  [ -n "$cccl" ] || { collx_log "ERROR: CUDA CCCL headers are unavailable"; return 1; }
  printf '%s\t%s' "$cuda_home" "$cccl"
}

# <base>/<name>-<cpu>-<arch>-<image>-<key...>: the directory name is the cache key reused
# across runs, so every component that changes the build belongs in it. Returns non-zero when
# no shared cache is mounted (manual runs).
backend_cache_root() {
  local name="$1" arch="$2" cpu base image IFS=-
  shift 2
  cpu="$(uname -m)"
  [[ "$cpu" =~ ^[A-Za-z0-9._-]+$ ]] || return 1
  base="${COLLX_BACKEND_CACHE_ROOT:-}"
  [[ "$base" = /* ]] || return 1
  image="$(printf '%s' "${COLLECTIVEX_IMAGE:-manual}" | tr -cs 'A-Za-z0-9_.-' '-')"
  printf '%s/%s-%s-%s-%s-%s' "$base" "$name" "$cpu" "$arch" "${image#-}" "$*"
}

# Runs `<ready-check> <root> || <install> <root> <args...>` under the cache root's lock, so
# concurrent nodes and allocations sharing the cache build it once.
with_cache_lock() {
  local label="$1" root="$2" ready="$3" install="$4" lock_path="$2.lock"
  shift 4
  (
    [ ! -L "$lock_path" ] || { collx_log "ERROR: $label cache lock is unsafe"; exit 1; }
    (umask 077; : >> "$lock_path") && chmod 600 "$lock_path" \
      || { collx_log "ERROR: $label cache-lock-create failed"; exit 1; }
    exec 9<>"$lock_path" || { collx_log "ERROR: $label cache-lock-open failed"; exit 1; }
    flock 9 || { collx_log "ERROR: $label cache-lock-acquire failed"; exit 1; }
    "$ready" "$root" || "$install" "$root" "$@" || exit 1
  )
}

site_cache_ready() { [ -f "$1/.ready" ] && [ -d "$1/site" ]; }

# Sets CACHE_ROOT to the shared cache when one is mounted, else to the node-local fallback, and
# installs into it unless it is already ready.
prepare_site_cache() {
  local name="$1" label="$2" ident="$3" shared="$4" node_local="$5"
  shift 5
  if [ -n "$shared" ]; then
    CACHE_ROOT="$shared"
    command -v flock >/dev/null \
      || { collx_log "ERROR: flock is required for $name caching"; return 1; }
    mkdir -p "${shared%/*}" || return 1
    collx_log "$name: preparing $ident (shared cache $shared)"
    with_cache_lock "$label" "$shared" site_cache_ready "$@" \
      || { collx_log "ERROR: shared $name environment is incomplete"; return 1; }
  else
    CACHE_ROOT="$node_local"
    collx_log "$name: preparing $ident (node-local $node_local; no shared cache mounted)"
    site_cache_ready "$node_local" || "$1" "$node_local" "${@:2}" || return 1
  fi
}

deepep_nvshmem_overlay() {
  local root="$1" packaged="$2" overlay path temporary
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
  printf '%s' "$overlay"
}

deepep_cache_root() {
  # The NVSHMEM wheel is part of the built venv's identity (see common.sh: the cu12
  # wheel on cu130 images broke sm103), so it keys the cache and a spec change rebuilds.
  local nvshmem_key="${COLLX_DEEPEP_V2_NVSHMEM_SPEC#nvidia-}"
  nvshmem_key="${nvshmem_key//==/-}"
  local torch_key="${COLLX_DEEPEP_V2_TORCH_SPEC//==/-}"
  local build_gen="${COLLX_DEEPEP_V2_BUILD_GEN:?}"
  [[ "$nvshmem_key" =~ ^[A-Za-z0-9._-]+$ && "$torch_key" =~ ^[A-Za-z0-9._-]+$ \
     && "$build_gen" =~ ^[A-Za-z0-9._-]+$ ]] || return 1
  backend_cache_root deepep-v2 "sm${1/./}" "${COLLX_DEEPEP_V2_COMMIT:0:12}" \
    "$torch_key" "$nvshmem_key" "$build_gen"
}

deepep_ready() { [ -f "$1/.ready" ] && [ -x "$1/venv/bin/python" ] && [ -d "$1/source" ]; }

# Runtime images (vLLM's) ship a trimmed CUDA toolkit without the cuSPARSE and similar headers that
# torch's ATen/cuda headers include, which the extension build needs. The venv's torch already
# pulled the matching CUDA wheels, so link only the headers the toolkit lacks into
# <root>/cuda-include: nothing shadows the toolkit's own headers, and full-toolkit images leave
# the directory empty.
deepep_cuda_header_overlay() {
  local root="$1" cuda_home="$2" overlay="$1/cuda-include" wheel_include header
  mkdir -p "$overlay" || return 1
  for wheel_include in "$root"/venv/lib/python*/site-packages/nvidia/cu13/include; do
    [ -d "$wheel_include" ] || continue
    for header in "$wheel_include"/*.h; do
      [ -e "$cuda_home/include/${header##*/}" ] || ln -sfn "$header" "$overlay/" || return 1
    done
  done
}

deepep_activate() {
  local root="$1" venv venv_site nccl_root nvshmem_package overlay
  local toolchain cuda_home cccl execution_id
  venv="$root/venv"
  [ -x "$venv/bin/python" ] \
    || { collx_log "ERROR: DeepEP V2 venv interpreter is unavailable"; return 1; }
  for venv_site in "$venv"/lib/python*/site-packages; do break; done
  [ -d "$venv_site" ] \
    || { collx_log "ERROR: DeepEP V2 venv site-packages is unavailable"; return 1; }
  nccl_root="$(nvidia_package_root "$venv/bin/python" nvidia-nccl-cu13 nccl)" \
    || { collx_log "ERROR: DeepEP V2 NCCL package root is unavailable"; return 1; }
  nvshmem_package="$(nvidia_package_root \
    "$venv/bin/python" "${COLLX_DEEPEP_V2_NVSHMEM_SPEC%%==*}" nvshmem)" \
    || { collx_log "ERROR: DeepEP V2 NVSHMEM package root is unavailable"; return 1; }
  overlay="$(deepep_nvshmem_overlay "$root" "$nvshmem_package")" || return 1
  toolchain="$(cuda_toolchain_paths)" || return 1
  IFS=$'\t' read -r cuda_home cccl <<< "$toolchain"
  [ -n "$cuda_home" ] && [ -n "$cccl" ] || return 1
  execution_id="${COLLECTIVEX_EXECUTION_ID:-manual}"
  [[ "$execution_id" =~ ^[A-Za-z0-9._-]+$ ]] \
    || { collx_log "ERROR: DeepEP V2 execution identity is invalid"; return 1; }

  export \
    VIRTUAL_ENV="$venv" \
    PATH="$venv/bin:${PATH#"$venv/bin:"}" \
    PYTHONPATH="$venv_site${PYTHONPATH:+:$PYTHONPATH}" \
    CUDA_HOME="$cuda_home" \
    CPATH="$cccl:${CPATH:-}${CPATH:+:}$root/cuda-include" \
    NVCC_PREPEND_FLAGS="-I$cccl ${NVCC_PREPEND_FLAGS:-}" \
    NVSHMEM_DIR="$overlay" \
    EP_NCCL_ROOT_DIR="$nccl_root" \
    EP_NVSHMEM_ROOT_DIR="$overlay" \
    EP_JIT_CACHE_DIR="/tmp/collectivex-deepep-v2-jit-$execution_id" \
    EP_REUSE_NCCL_COMM=1 \
    NCCL_CUMEM_ENABLE=1 \
    LD_LIBRARY_PATH="$overlay/lib:$nccl_root/lib:$nvshmem_package/lib:${LD_LIBRARY_PATH:-}" \
    LIBRARY_PATH="$nccl_root/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
  # LIBRARY_PATH: DeepEP links -l:libnccl.so.2 with only an rpath to the wheel, so the link step
  # needs the wheel's lib directory on the search path. Images with a system NCCL (SGLang's)
  # resolved it from /usr/lib; vLLM's image has NCCL only as a pip wheel.
  unset "${DEEPEP_RANK_UNSETS[@]}"

  # Shared JIT caches race across nodes; keep this cache node-local. CUMEM is
  # persisted here too because image environment overrides launcher exports.
  [ ! -L "$EP_JIT_CACHE_DIR" ] \
    || { collx_log "ERROR: DeepEP V2 JIT cache path is unsafe"; return 1; }
  if ! mkdir -p "$EP_JIT_CACHE_DIR" || ! chmod 700 "$EP_JIT_CACHE_DIR"; then
    collx_log "ERROR: DeepEP V2 JIT cache is unavailable"
    return 1
  fi
}

deepep_probe() {
  "$VIRTUAL_ENV/bin/python" - <<'PY'
import inspect
import deep_ep
assert inspect.isclass(deep_ep.ElasticBuffer)
PY
}

deepep_install() {
  local root="$1" arch="$2" venv="$1/venv" source_dir="$1/source"
  local -a pip
  if [ -e "$root" ] || [ -L "$root" ]; then
    rm -rf "$root" \
      || { collx_log "ERROR: incomplete DeepEP V2 cache-reset failed"; return 1; }
  fi
  mkdir -m 700 "$root" \
    || { collx_log "ERROR: DeepEP V2 cache-create failed"; return 1; }
  python3 -m venv "$venv" \
    || { collx_log "ERROR: DeepEP V2 venv creation failed"; return 1; }
  pip=("$venv/bin/python" -m pip install -q --disable-pip-version-check --no-input)
  "${pip[@]}" \
    "pip==26.1.2" "setuptools==82.0.1" "wheel==0.47.0" "ninja==1.13.0" \
    "numpy==2.2.6" "$COLLX_DEEPEP_V2_NVSHMEM_SPEC" >&2 2>&1 \
    || { collx_log "ERROR: DeepEP V2 build-tool installation failed"; return 1; }
  "${pip[@]}" --index-url https://download.pytorch.org/whl/cu130 \
    --extra-index-url https://pypi.org/simple "$COLLX_DEEPEP_V2_TORCH_SPEC" >&2 2>&1 \
    || { collx_log "ERROR: $COLLX_DEEPEP_V2_TORCH_SPEC (cu130) installation failed"; return 1; }
  # Torch pins NCCL 2.28.9; ElasticBuffer requires 2.30.4.
  "${pip[@]}" --force-reinstall --no-deps "nvidia-nccl-cu13==2.30.4" >&2 2>&1 \
    || { collx_log "ERROR: NCCL 2.30.4 installation failed"; return 1; }
  local toolchain cuda_home cccl
  toolchain="$(cuda_toolchain_paths)" || return 1
  IFS=$'\t' read -r cuda_home cccl <<< "$toolchain"
  deepep_cuda_header_overlay "$root" "$cuda_home" \
    || { collx_log "ERROR: DeepEP V2 CUDA header overlay failed"; return 1; }
  deepep_activate "$root" \
    || { collx_log "ERROR: DeepEP V2 environment activation failed"; return 1; }
  collx_materialize_source "deepep-v2-$COLLX_DEEPEP_V2_COMMIT" "$source_dir" \
    || { collx_log "ERROR: DeepEP V2 staged source is invalid"; return 1; }
  # Upstream registers the GIN window NCCL_WIN_STRICT_ORDERING, which drops PCIe relaxed ordering
  # whatever NCCL_IB_PCI_RELAXED_ORDERING says; EP_WIN_RELAXED_ORDERING=1 opts a pool out. Unset,
  # the registration is upstream's. A pin that moves or duplicates the flag fails here.
  local window_cu="$source_dir/csrc/kernels/backend/nccl.cu"
  local relaxed='get_env("EP_WIN_RELAXED_ORDERING", 0) ? NCCL_WIN_DEFAULT :'
  [ "$(grep -c NCCL_WIN_STRICT_ORDERING "$window_cu")" = 1 ] \
    && sed -i "s/NCCL_WIN_STRICT_ORDERING/($relaxed &)/" "$window_cu" \
    || { collx_log "ERROR: DeepEP V2 window-ordering patch failed"; return 1; }
  # The RDC device-link step (nvcc -dlink) gets no -gencode from the extension build, so nvcc
  # falls back to its default arch (sm_75 on CUDA 13) and links kernels that cannot load on the
  # target GPU (gb300/sm103: cudaErrorUnknown). NVCC_PREPEND_FLAGS reaches the dlink too.
  local gencode="-gencode=arch=compute_${arch/./},code=sm_${arch/./}"
  (cd "$source_dir" && TORCH_CUDA_ARCH_LIST="$arch" MAX_JOBS=16 \
    NVCC_PREPEND_FLAGS="$gencode ${NVCC_PREPEND_FLAGS:-}" \
    "$venv/bin/python" -m pip install -q --no-build-isolation --no-deps \
      --force-reinstall .) >&2 2>&1 \
    || { collx_log "ERROR: DeepEP V2 build failed"; return 1; }
  deepep_probe \
    || { collx_log "ERROR: DeepEP V2 import probe failed"; return 1; }
  : > "$root/.ready"
}

# DeepEP lifecycle

deepep_prepare() {
  local arch root
  arch="$(cuda_arch)" || return 1
  root="$(deepep_cache_root "$arch")" || return 1
  command -v flock >/dev/null || { collx_log "ERROR: flock is required for DeepEP V2"; return 1; }
  mkdir -p "${root%/*}" || return 1
  collx_log "DeepEP V2: preparing upstream main ($COLLX_DEEPEP_V2_COMMIT)"
  if ! with_cache_lock "DeepEP V2" "$root" deepep_ready deepep_install "$arch"; then
    collx_log "ERROR: shared DeepEP V2 environment is incomplete"
    return 1
  fi
  deepep_activate "$root" || return 1
  deepep_probe || { collx_log "ERROR: DeepEP V2 shared runtime probe failed"; return 1; }
  collx_log "DeepEP V2 ready ($COLLX_DEEPEP_V2_COMMIT, ElasticBuffer, NCCL Device API; LSA/Gin selected by adapter)"
}

# UCCL-EP lifecycle

uccl_rocm_arch() {
  python3 - "$COLLX_RUNNER" <<'PY'
import json, sys
print(json.load(open("configs/platform_config.json"))["platforms"][sys.argv[1]]["arch"])
PY
}

# The build recipe keys the cache and is recorded beside the packages, where the adapter publishes
# it as library_version: a from-source build has no wheel version that tells two pins apart. Bump
# UCCL_BUILD_GEN when the recipe or the record changes without the pin moving, so caches that
# already carry .ready are rebuilt rather than reused. PER_EXPERT_BATCHING is off by default
# upstream but is UCCL's documented drop-in deep_ep build, and it changes the low-latency kernels.
readonly -a UCCL_BUILD_FLAGS=(USE_DMABUF=1 PER_EXPERT_BATCHING=1)
readonly UCCL_BUILD_GEN=rec2
readonly UCCL_BUILD_RECORD=collx_uccl_build.json

uccl_probe() {
  # import torch FIRST so libc10 is resident before the uccl.ep extension dlopens (it links
  # libc10/libtorch); importing deep_ep before torch fails with "libc10.so: cannot open".
  # DeepSeek's DeepEP, which the sglang images ship, has the same legacy Buffer API, so capability
  # checks cannot tell the two apart. Identity can: UCCL's wrapper re-exports uccl.ep's Config, both
  # must load from this cache, and its build record must name the pinned commit.
  python3 - "$1/site" "$COLLX_UCCL_COMMIT" "$UCCL_BUILD_RECORD" <<'PY'
import json
import sys
from pathlib import Path

import torch  # noqa: F401
import deep_ep
from deep_ep import Buffer
from uccl import ep

site, commit, record = Path(sys.argv[1]).resolve(), sys.argv[2], sys.argv[3]
assert deep_ep.Config is ep.Config, "deep_ep is not UCCL's deep_ep_wrapper"
for module in (deep_ep, ep):
    assert site in Path(module.__file__).resolve().parents, f"{module.__name__} is not from {site}"
assert json.loads((site / record).read_text())["commit"] == commit, "UCCL build record is stale"
assert hasattr(Buffer, "low_latency_dispatch") and hasattr(Buffer, "get_dispatch_layout")
PY
}

uccl_record_build() {
  python3 - "$1/site/$UCCL_BUILD_RECORD" "$COLLX_UCCL_COMMIT" "${3:-${UCCL_BUILD_FLAGS[*]}}" "$2" <<'PY'
import json
import sys

path, commit, build_flags, arch = sys.argv[1:]
with open(path, "w", encoding="utf-8") as record:
    json.dump({"commit": commit, "build_flags": build_flags, "arch": arch}, record)
PY
}

# UCCL is built in-container against the image's torch, not with upstream `build.sh` (that spins
# up its own Docker image and cannot run inside enroot/pyxis). The built deep_ep/uccl packages
# persist under a cache root and reach the ranks via PYTHONPATH, so later allocations skip the build.

# On EFA pools UCCL-EP builds its EFA transport only for multi-node shards, against the host
# rdma-core the launcher binds (collx_add_efa_host_mounts); inside one node UCCL starts no proxy, so
# single-node shards build without it. The two builds key separate caches.
uccl_efa_build() {
  [ "${COLLX_RDMA_FABRIC:-}" = efa ] || return 0
  if [ "${COLLX_NODES:-1}" -le 1 ]; then printf 'noefa'; else printf 'efa'; fi
}

uccl_cache_root() {
  local arch
  arch="$(printf '%s' "$1" | tr -cs 'A-Za-z0-9_.-' '-')"
  backend_cache_root uccl-ep "${arch#-}" "${COLLX_UCCL_COMMIT:0:12}" "$UCCL_BUILD_GEN"
}

# CDNA needs UCCL's aggressive host-atomic path.
uccl_activate() {
  local site="$1/site"
  [ -d "$site" ] || { collx_log "ERROR: UCCL cache site is unavailable"; return 1; }
  export PYTHONPATH="$site${PYTHONPATH:+:$PYTHONPATH}"
  [ "${COLLX_VENDOR:-nvidia}" != amd ] || export UCCL_EP_ENABLE_AGGRESSIVE_ATOMIC=1
}

uccl_install() {
  local root="$1" arch="$2" source_dir="/tmp/collectivex-uccl-$COLLX_UCCL_COMMIT" arch_env sp
  local -a flags=("${UCCL_BUILD_FLAGS[@]}")
  # UCCL builds its EFA transport whenever EFA_HOME exists, from EFA_HOME/include and EFA_HOME/lib.
  # The EFA pools' containers carry /opt/amazon/efa beside an image rdma-core whose efadv.h predates
  # the fields UCCL uses (h100-dgxc: efadv_qp_init_attr has no sl or flags), so EFA_HOME names the
  # host rdma-core bound for multi-node shards instead (see uccl_efa_build).
  case "$(uccl_efa_build)" in
    noefa) flags+=(EFA_HOME=/nonexistent) ;;
    efa)
      [ -r "$COLLX_EFA_HOST_ROOT/include/infiniband/efadv.h" ] && [ -e "$COLLX_EFA_HOST_ROOT/lib/libefa.so" ] \
        || { collx_log "ERROR: UCCL-EP across EFA needs the host rdma-core headers and libefa"; return 1; }
      flags+=(EFA_HOME="$COLLX_EFA_HOST_ROOT") ;;
  esac
  if [ -e "$root" ] || [ -L "$root" ]; then
    rm -rf "$root" || { collx_log "ERROR: incomplete UCCL cache-reset failed"; return 1; }
  fi
  mkdir -m 700 "$root" || { collx_log "ERROR: UCCL cache-create failed"; return 1; }
  collx_log "UCCL-EP: building $COLLX_UCCL_COMMIT from source (${flags[*]})"
  # Some sglang/rocm images mark the system env externally-managed (PEP 668).
  { python3 -m pip install -q --disable-pip-version-check --no-input nanobind \
      || python3 -m pip install -q --disable-pip-version-check --no-input \
           --break-system-packages nanobind; } >&2 2>&1 \
    || { collx_log "ERROR: UCCL nanobind install failed"; return 1; }
  collx_materialize_source "uccl-$COLLX_UCCL_COMMIT" "$source_dir" \
    || { collx_log "ERROR: UCCL staged source is invalid"; return 1; }
  if [ "${COLLX_VENDOR:-nvidia}" = amd ]; then
    arch_env="PYTORCH_ROCM_ARCH=$arch"
    # hipMallocManaged fails on our CDNA nodes (even 4 KiB, regardless of XNACK or memlock). UCCL's
    # HIP CPU-proxy path uses cudaMallocManaged for the d2h channel handles and proxy atomic buffer;
    # pinned host memory (cudaMallocHost) is coherent and device-accessible on gfx942/gfx950.
    sed -i 's/cudaMallocManaged/cudaMallocHost/g' \
      "$source_dir/ep/src/uccl_ep.cc" "$source_dir/ep/src/uccl_proxy.cpp" \
      || { collx_log "ERROR: UCCL AMD managed-memory patch failed"; return 1; }
    # Plain launch on AMD: hipify cannot map the cooperative cudaLaunchKernelEx path, which also
    # costs ~30 us per launch on gfx942. Defined ahead of the #ifndef-guarded upstream macros.
    python3 - "$source_dir/ep/include/ep_launch.cuh" <<'PY' \
      || { collx_log "ERROR: UCCL AMD launch patch failed"; return 1; }
import sys
path = sys.argv[1]
src = open(path).read()
plain = r"""
#define SETUP_LAUNCH_CONFIG(sms, threads, stream) \
  int __num_sms = (sms);                          \
  int __num_threads = (threads);                  \
  auto __stream = (stream)
#define LAUNCH_KERNEL(config, kernel, ...)                          \
  do {                                                              \
    kernel<<<__num_sms, __num_threads, 0, __stream>>>(__VA_ARGS__); \
    cudaError_t e = cudaGetLastError();                             \
    if (e != cudaSuccess) {                                         \
      EPException cuda_exception("CUDA", __FILE__, __LINE__,        \
                                 cudaGetErrorString(e));            \
      fprintf(stderr, "%s\n", cuda_exception.what());               \
      throw cuda_exception;                                         \
    }                                                               \
  } while (0)
"""
anchor = '#include "exception.cuh"\n'
assert src.count(anchor) == 1, "ep_launch.cuh changed shape"
open(path, "w").write(src.replace(anchor, anchor + plain, 1))
PY
  else
    arch_env="TORCH_CUDA_ARCH_LIST=$arch"
  fi
  ( cd "$source_dir/ep" \
      && env "${flags[@]}" "$arch_env" python3 setup.py install ) >&2 2>&1 \
    || { collx_log "ERROR: UCCL ep extension build failed"; return 1; }
  # --no-deps: the wrapper's install_requires=["uccl"] resolves to the PyPI uccl-cu12 wheel, absent
  # on ROCm and wrong on CUDA too, since the from-source ep build already provides uccl.ep.
  ( cd "$source_dir/ep/deep_ep_wrapper" \
      && { python3 -m pip install -q --disable-pip-version-check --no-input --no-deps . \
             || python3 -m pip install -q --disable-pip-version-check --no-input \
                  --no-deps --break-system-packages . ; } ) >&2 2>&1 \
    || { collx_log "ERROR: UCCL deep_ep_wrapper build failed"; return 1; }
  sp="$(python3 -c 'import site; print(site.getsitepackages()[0])')" \
    || { collx_log "ERROR: UCCL site-packages resolution failed"; return 1; }
  mkdir -p "$root/site" \
    && cp -R "$sp"/deep_ep* "$sp"/uccl* "$root/site/" \
    || { collx_log "ERROR: UCCL cache population failed"; return 1; }
  uccl_record_build "$root" "$arch" "${flags[*]}" \
    || { collx_log "ERROR: UCCL build record failed"; return 1; }
  : > "$root/.ready"
}

uccl_prepare() {
  local arch shared efa
  command -v python3 >/dev/null || { collx_log "ERROR: python3 unavailable for UCCL build"; return 1; }
  if [ "${COLLX_VENDOR:-nvidia}" = amd ]; then
    arch="$(uccl_rocm_arch)" || return 1
  else
    arch="$(cuda_arch)" || return 1
  fi
  efa="$(uccl_efa_build)"
  shared="$(uccl_cache_root "$arch${efa:+-$efa}")" || shared=""
  prepare_site_cache UCCL-EP UCCL "$COLLX_UCCL_COMMIT" "$shared" \
    "/tmp/collectivex-uccl-cache-$COLLX_UCCL_COMMIT-$UCCL_BUILD_GEN${efa:+-$efa}" uccl_install "$arch" || return 1
  uccl_activate "$CACHE_ROOT" || return 1
  uccl_probe "$CACHE_ROOT" || { collx_log "ERROR: UCCL import probe failed"; return 1; }
  collx_log "UCCL-EP ready ($COLLX_UCCL_COMMIT, deep_ep wrapper over uccl.ep CPU-proxy runtime)"
}

# NCCL EP lifecycle

nccl_ep_spec_slug() {
  printf '%s' "$COLLX_NCCL_EP_SPEC" | tr -cs 'A-Za-z0-9_.-' '-'
}

nccl_ep_cache_root() {
  local arch slug
  arch="$(printf '%s' "$1" | tr -cs 'A-Za-z0-9_.-' '-')"
  slug="$(nccl_ep_spec_slug)"
  backend_cache_root nccl-ep "${arch#-}" "${slug#-}"
}

# The wheel-bundled NCCL goes ahead of the image torch's older NCCL on the loader path: nccl.ep
# needs NCCL >= 2.29.3 (Device API + GIN).
nccl_ep_activate() {
  local root="$1" site="$1/site" nccl_lib
  [ -d "$site" ] || { collx_log "ERROR: NCCL EP cache site is unavailable"; return 1; }
  export PYTHONPATH="$site${PYTHONPATH:+:$PYTHONPATH}"
  for nccl_lib in "$site"/nvidia/nccl*/lib; do
    if [ -d "$nccl_lib" ]; then
      export LD_LIBRARY_PATH="$nccl_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
      break
    fi
  done
  # NCCL only advertises the Device API (LSA symmetric memory) with cuMem allocation enabled;
  # without it ncclEpCreateGroup returns ncclInvalidUsage. Persisted here so every rank has it.
  export NCCL_CUMEM_ENABLE=1
}

nccl_ep_probe() {
  # torch first so libc10/libnccl are resident before nccl.ep dlopens. The version line records
  # which libnccl_ep.so loaded, in case a stale cache or image-bundled copy shadows the wheel.
  python3 - <<'PY'
import sys

import torch  # noqa: F401
import nccl.core  # noqa: F401
import nccl.ep

print(
    f"nccl.ep: libnccl_ep {nccl.ep.get_lib_version()} at {nccl.ep.get_lib_path()}",
    file=sys.stderr,
)
PY
}

nccl_ep_install() {
  local root="$1" site="$1/site"
  if [ -e "$root" ] || [ -L "$root" ]; then
    rm -rf "$root" || { collx_log "ERROR: incomplete NCCL EP cache-reset failed"; return 1; }
  fi
  mkdir -m 700 "$root" || { collx_log "ERROR: NCCL EP cache-create failed"; return 1; }
  mkdir -p "$site" || { collx_log "ERROR: NCCL EP cache-site-create failed"; return 1; }
  collx_log "NCCL EP: installing $COLLX_NCCL_EP_SPEC (pip --target)"
  # --target does not touch the system env, so PEP 668 does not apply. $COLLX_NCCL_EP_SPEC is
  # unquoted on purpose: it carries whitespace-separated pip specs.
  # shellcheck disable=SC2086
  python3 -m pip install -q --disable-pip-version-check --no-input \
      --pre --extra-index-url https://pypi.nvidia.com \
      --target "$site" $COLLX_NCCL_EP_SPEC >&2 2>&1 \
    || { collx_log "ERROR: NCCL EP wheel install failed"; return 1; }
  nccl_ep_activate "$root" \
    || { collx_log "ERROR: NCCL EP environment activation failed"; return 1; }
  nccl_ep_probe || { collx_log "ERROR: NCCL EP import probe failed"; return 1; }
  : > "$root/.ready"
}

nccl_ep_prepare() {
  local arch shared
  command -v python3 >/dev/null || { collx_log "ERROR: python3 unavailable for NCCL EP"; return 1; }
  arch="$(cuda_arch)" || return 1
  shared="$(nccl_ep_cache_root "$arch")" || shared=""
  prepare_site_cache "NCCL EP" "NCCL EP" "$COLLX_NCCL_EP_SPEC" "$shared" \
    "/tmp/collectivex-nccl-ep-cache-$(nccl_ep_spec_slug)" nccl_ep_install || return 1
  nccl_ep_activate "$CACHE_ROOT" || return 1
  nccl_ep_probe || { collx_log "ERROR: NCCL EP import probe failed"; return 1; }
  collx_log "NCCL EP ready ($COLLX_NCCL_EP_SPEC; libnccl_ep.so JIT runtime, NCCL Device API LSA/GIN)"
}

# Multi-node EFA shards whose host plugin predates aws-ofi-nccl 1.21 build it against the host
# libfabric the launcher binds, from a hash-pinned release tarball; the rank wrapper's EFA profile
# then loads it in place of the host's (see COLLX_OFI_NCCL_VERSION). Proxy GIN only: configure finds
# no libfabric GDA ops on those hosts.
ofi_nccl_ready() { [ -f "$1/.ready" ] && [ -r "$1/lib/libnccl-net-ofi.so" ]; }

# Static hwloc for images without its headers (see COLLX_HWLOC_VERSION).
ofi_nccl_build_hwloc() {
  local root="$1" version="$COLLX_HWLOC_VERSION" tarball pciaccess
  pciaccess="libpciaccess-$COLLX_PCIACCESS_VERSION"
  curl -sSfL --connect-timeout 30 --max-time 300 --retry 3 -o "$root/$pciaccess.tar.gz" \
      "https://www.x.org/releases/individual/lib/$pciaccess.tar.gz" \
    && printf '%s  %s\n' "$COLLX_PCIACCESS_SHA256" "$root/$pciaccess.tar.gz" | sha256sum -c --quiet \
    && tar -xzf "$root/$pciaccess.tar.gz" -C "$root" \
    || { collx_log "ERROR: libpciaccess source fetch failed"; return 1; }
  (
    cd "$root/$pciaccess" \
      && ./configure -q --prefix="$root/pciaccess" --enable-static --disable-shared --with-pic \
      && make -s -j"$(nproc)" && make -s install
  ) >&2 2>&1 || { collx_log "ERROR: libpciaccess build failed"; return 1; }
  rm -rf "${root:?}/$pciaccess" "${root:?}/$pciaccess.tar.gz"
  tarball="hwloc-$version.tar.gz"
  curl -sSfL --connect-timeout 30 --max-time 300 --retry 3 -o "$root/$tarball" \
      "https://download.open-mpi.org/release/hwloc/v${version%.*}/$tarball" \
    && printf '%s  %s\n' "$COLLX_HWLOC_SHA256" "$root/$tarball" | sha256sum -c --quiet \
    && tar -xzf "$root/$tarball" -C "$root" \
    || { collx_log "ERROR: hwloc source fetch failed"; return 1; }
  # The plugin groups each GPU with its nearest EFA NICs from hwloc's PCI tree, so I/O discovery
  # must stay on (without it every group is empty and the RDMA protocol, the only one GIN supports,
  # fails to start). hwloc reads the PCI tree through libpciaccess, built static alongside it; the
  # GPU-vendor components would only link vendor libraries into the static archive.
  collx_log "aws-ofi-nccl: building hwloc $version (the image has no hwloc headers)"
  (
    cd "$root/hwloc-$version" \
      && HWLOC_PCIACCESS_CFLAGS="-I$root/pciaccess/include" \
        HWLOC_PCIACCESS_LIBS="$root/pciaccess/lib/libpciaccess.a" \
        ./configure -q --prefix="$root/hwloc" --enable-static --disable-shared --with-pic \
        --disable-libxml2 --disable-cairo --disable-plugins --disable-libudev \
        --disable-cuda --disable-nvml --disable-opencl --disable-rsmi --disable-levelzero \
      && make -s -j"$(nproc)" && make -s install
  ) >&2 2>&1 || { collx_log "ERROR: hwloc build failed"; return 1; }
  # libtool cannot pass a bare static archive through to a shared plugin, so libpciaccess's objects
  # join libhwloc.a and -lhwloc carries both; without hwloc's .la, libtool takes the archive as is.
  (
    mkdir "$root/pciaccess/objects" && cd "$root/pciaccess/objects" \
      && ar x ../lib/libpciaccess.a && ar rs "$root/hwloc/lib/libhwloc.a" ./*.o
  ) >&2 2>&1 || { collx_log "ERROR: hwloc archive merge failed"; return 1; }
  rm -f "$root/hwloc/lib/libhwloc.la"
  rm -rf "${root:?}/hwloc-$version" "${root:?}/$tarball" "${root:?}/pciaccess/objects"
}

ofi_nccl_install() {
  local root="$1" version="$COLLX_OFI_NCCL_VERSION" tarball cuda_home
  rm -rf "$root" && mkdir -m 700 "$root" \
    || { collx_log "ERROR: aws-ofi-nccl cache-create failed"; return 1; }
  tarball="aws-ofi-nccl-$version.tar.gz"
  # Bounded per attempt so a stalled download is retried instead of hanging backend preparation.
  curl -sSfL --connect-timeout 30 --max-time 300 --retry 3 -o "$root/$tarball" \
      "https://github.com/aws/aws-ofi-nccl/releases/download/v$version/$tarball" \
    && printf '%s  %s\n' "$COLLX_OFI_NCCL_SHA256" "$root/$tarball" | sha256sum -c --quiet \
    && tar -xzf "$root/$tarball" -C "$root" \
    || { collx_log "ERROR: aws-ofi-nccl source fetch failed"; return 1; }
  IFS=$'\t' read -r cuda_home _ <<< "$(cuda_toolchain_paths)" && [ -n "$cuda_home" ] || return 1
  local -a hwloc=()
  if [ ! -r /usr/include/hwloc.h ]; then
    ofi_nccl_build_hwloc "$root" || return 1
    hwloc=(--with-hwloc="$root/hwloc")
  fi
  collx_log "aws-ofi-nccl: building $version"
  (
    cd "$root/aws-ofi-nccl-$version" \
      && ./configure -q --prefix="$root" --with-libfabric=/opt/amazon/efa --with-cuda="$cuda_home" \
        "${hwloc[@]}" --enable-platform-aws --disable-tests \
      && make -s -j"$(nproc)" && make -s install
  ) >&2 2>&1 || { collx_log "ERROR: aws-ofi-nccl build failed"; return 1; }
  rm -rf "${root:?}/aws-ofi-nccl-$version" "${root:?}/$tarball"
  : > "$root/.ready"
}

ofi_nccl_prepare() {
  local root
  unset COLLX_OFI_NCCL_LIB_DIR
  [ "${COLLX_RDMA_FABRIC:-}" = efa ] && [ "${COLLX_NODES:-1}" -gt 1 ] || return 0
  # ncclGinPlugin_v14 (the EFA GDA op table) first shipped in 1.21.0: such a host plugin already has
  # the worker signal path (b300's 1.21.1, EFA installer 1.50.0).
  if grep -qF ncclGinPlugin_v14 /opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so; then
    collx_log "aws-ofi-nccl: using the host plugin"
    return 0
  fi
  # A plugin built on the bundled hwloc keys its own cache, so a change to that build replaces it.
  local variant="$COLLX_OFI_NCCL_VERSION"
  [ -r /usr/include/hwloc.h ] || variant+="-hwloc$COLLX_HWLOC_VERSION-pciaccess$COLLX_PCIACCESS_VERSION-merged"
  root="$(backend_cache_root ofi-nccl efa "$variant")" \
    || root="/tmp/collectivex-ofi-nccl-$variant"
  with_cache_lock "aws-ofi-nccl" "$root" ofi_nccl_ready ofi_nccl_install || return 1
  export COLLX_OFI_NCCL_LIB_DIR="$root/lib"
  collx_log "aws-ofi-nccl $COLLX_OFI_NCCL_VERSION ready"
}

# container boundary

write_rank_env() {
  local root="$PWD/.collx_backend/env" node_id="${SLURM_NODEID:-0}" path temporary name
  [[ "$node_id" =~ ^[0-9]+$ ]] || return 1
  mkdir -p "$root" || return 1
  chmod 700 "$root" || return 1
  temporary="$(mktemp "$root/.node-${node_id}.XXXXXX")" || return 1
  chmod 600 "$temporary" || { rm -f "$temporary"; return 1; }
  for name in "${RANK_ENV_VARS[@]}"; do
    if declare -p "$name" >/dev/null 2>&1; then
      printf 'export %s=%q\n' "$name" "${!name}" >> "$temporary" \
        || { rm -f "$temporary"; return 1; }
    fi
  done
  if [ "$COLLX_BENCH" = deepep-v2 ]; then
    for name in "${DEEPEP_RANK_UNSETS[@]}"; do
      printf 'unset %s\n' "$name" >> "$temporary" \
        || { rm -f "$temporary"; return 1; }
    done
  fi
  path="$root/node-${node_id}.sh"
  mv -f -- "$temporary" "$path" || { rm -f "$temporary"; return 1; }
}

validate_container_network() {
  local interface device rdma_name efa_domains
  local -a interfaces devices
  if [ "${COLLX_NODES:-1}" -le 1 ] || [ "${COLLX_TRANSPORT:-}" = mnnvl ]; then
    return 0
  fi
  collx_restore_exact_hca_selector || return 1
  local rdma_selector="${NCCL_IB_HCA:-}"
  if [ "${COLLX_RDMA_FABRIC:-}" = efa ]; then
    # No verbs selector on EFA; the probe-validated device list is the contract, and the host's
    # libfabric NCCL plugin (collx_efa_host_mounts) is what actually carries the traffic. A verbs
    # provider from another rdma-core loads nothing and says nothing, so require the host libfabric
    # to reach every listed device through the bound user space.
    rdma_selector="${COLLX_RDMA_DEVICES:-}"
    [ -r /opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so ] \
      && [ -r "$COLLX_EFA_HOST_ROOT/lib/libibverbs.so.1" ] \
      || { collx_log "ERROR: the host EFA user space is absent inside the container"; return 1; }
    efa_domains="$(/opt/amazon/efa/bin/fi_info -p efa -t FI_EP_RDM 2>&1)" \
      || { collx_log "ERROR: the host libfabric finds no EFA device"; return 1; }
    IFS=, read -r -a devices <<< "$rdma_selector"
    for device in "${devices[@]}"; do
      [[ "$efa_domains" == *"domain: ${device%%:*}-rdm"* ]] \
        || { collx_log "ERROR: a configured EFA device is unusable through the host libfabric"
             return 1; }
    done
  fi
  [ -n "${GLOO_SOCKET_IFNAME:-}" ] && [ -n "$rdma_selector" ] \
    || { collx_log "ERROR: scale-out network selectors are unavailable"; return 1; }
  IFS=, read -r -a interfaces <<< "$GLOO_SOCKET_IFNAME"
  for interface in "${interfaces[@]}"; do
    [ -d "/sys/class/net/$interface" ] \
      || { collx_log "ERROR: configured scale-out socket interface is absent"; return 1; }
  done
  IFS=, read -r -a devices <<< "$rdma_selector"
  for device in "${devices[@]}"; do
    device="${device#=}"
    rdma_name="${device%%:*}"
    [ -d "/sys/class/infiniband/$rdma_name" ] \
      || { collx_log "ERROR: configured scale-out RDMA device is absent"; return 1; }
  done
}

# FlashInfer lifecycle

# Keys both the shared and the node-local cache, so a pin change never reuses an older venv.
flashinfer_spec_digest() {
  # "aot" names the cache layout below; change it when the layout changes.
  printf 'aot %s %s' "$COLLX_FLASHINFER_SPEC" "$COLLX_FLASHINFER_JIT_CACHE_SPEC" \
    | sha256sum | cut -c1-12
}

flashinfer_cache_root() {
  local arch digest
  arch="$(printf '%s' "$1" | tr -cs 'A-Za-z0-9_.-' '-')"
  digest="$(flashinfer_spec_digest)" || return 1
  backend_cache_root flashinfer-ep "${arch#-}" "$digest"
}

flashinfer_ready() { [ -f "$1/.ready" ] && [ -x "$1/venv/bin/python" ]; }

# The venv sees the image's torch and NCCL (--system-site-packages) and shadows its FlashInfer.
flashinfer_activate() {
  local venv="$1/venv" site
  for site in "$venv"/lib/python*/site-packages; do break; done
  [ -d "$site" ] || { collx_log "ERROR: FlashInfer venv site-packages is unavailable"; return 1; }
  export \
    VIRTUAL_ENV="$venv" \
    PATH="$venv/bin:${PATH#"$venv/bin:"}" \
    PYTHONPATH="$site${PYTHONPATH:+:$PYTHONPATH}" \
    FLASHINFER_WORKSPACE_BASE="$1/workspace"
}

# FlashInfer takes a file lock and reruns ninja on every load of a JIT module, which fails on
# NFS without lock support and relinks the library while other ranks load it. An artifact in the
# package's own AOT directory loads without either, so the module is built once under the cache
# lock and copied there.
flashinfer_promote_aot() {
  python3 - <<'PY'
import os
import shutil

from flashinfer.comm.trtllm_moe_alltoall import gen_moe_alltoall_module
from flashinfer.jit import env as jit_env

spec = gen_moe_alltoall_module("legacy")
spec.build_and_load()
target = jit_env.FLASHINFER_AOT_DIR / spec.name / f"{spec.name}.so"
target.parent.mkdir(parents=True, exist_ok=True)
staged = target.with_name(target.name + ".tmp")
shutil.copyfile(spec.jit_library_path, staged)
os.chmod(staged, 0o755)
os.replace(staged, target)
PY
}

# Asserts the pinned wheel and that the all-to-all module loads from the AOT directory.
flashinfer_probe() {
  python3 - "${COLLX_FLASHINFER_SPEC%% *}" <<'PY'
import sys

import flashinfer
from flashinfer.comm import Mapping  # noqa: F401
from flashinfer.comm.mnnvl import MnnvlConfig  # noqa: F401
from flashinfer.comm.trtllm_moe_alltoall import (  # noqa: F401
    MoeAlltoAll,
    gen_moe_alltoall_module,
    get_moe_alltoall_module,
    moe_a2a_get_workspace_size_per_rank,
)

pinned = sys.argv[1].split("==")[1]
if flashinfer.__version__ != pinned:
    raise SystemExit(f"flashinfer {flashinfer.__version__} loaded, {pinned} pinned")
if not gen_moe_alltoall_module("legacy").is_aot:
    raise SystemExit("the all-to-all module is not in FlashInfer's AOT directory")
get_moe_alltoall_module()
print(f"FlashInfer {flashinfer.__version__} one-sided A2A available", file=sys.stderr)
PY
}

flashinfer_install() {
  local root="$1" venv="$1/venv" report="$1/pip-report.json"
  local -a pip jit_parts
  if [ -e "$root" ] || [ -L "$root" ]; then
    rm -rf "$root" || { collx_log "ERROR: incomplete FlashInfer cache-reset failed"; return 1; }
  fi
  mkdir -m 700 "$root" || { collx_log "ERROR: FlashInfer cache-create failed"; return 1; }
  python3 -m venv --system-site-packages "$venv" \
    || { collx_log "ERROR: FlashInfer venv creation failed"; return 1; }
  pip=("$venv/bin/python" -m pip install -q --disable-pip-version-check --no-input
       --extra-index-url https://flashinfer.ai/whl/cu130 --extra-index-url https://flashinfer.ai/whl)
  collx_log "FlashInfer: installing $COLLX_FLASHINFER_SPEC"
  # The specs are unquoted on purpose: the variable carries whitespace-separated pip specs.
  # shellcheck disable=SC2086
  # --no-deps: the spec names every package FlashInfer needs beyond the image. Resolving deps would
  # also re-resolve the image's torch, whose metadata pins an NCCL older than the one the image
  # ships (vLLM v0.32.0: torch 2.13 declares nvidia-nccl-cu13==2.29.7 over the installed 2.30.7), so
  # pip would downgrade NCCL. The check after the install covers what --no-deps skips.
  "${pip[@]}" --no-deps --dry-run --report "$report" $COLLX_FLASHINFER_SPEC >&2 2>&1 \
    || { collx_log "ERROR: FlashInfer dependency resolution failed"; return 1; }
  # Replacing the image's torch or NCCL would change what every other backend measures.
  python3 - "$report" <<'PY' \
    || { collx_log "ERROR: FlashInfer would replace the image's torch or NCCL"; return 1; }
import json, re, sys

names = [item["metadata"]["name"] for item in json.load(open(sys.argv[1]))["install"]]
clash = [name for name in names if re.match(r"(torch|triton|pytorch-triton|nvidia-nccl)", name, re.I)]
if clash:
    raise SystemExit(f"refusing to install {clash}")
PY
  # shellcheck disable=SC2086
  "${pip[@]}" --no-deps $COLLX_FLASHINFER_SPEC >&2 2>&1 \
    && "${pip[@]}" --no-deps "$COLLX_FLASHINFER_JIT_CACHE_SPEC" >&2 2>&1 \
    || { collx_log "ERROR: FlashInfer installation failed"; return 1; }
  # The jit-cache wheel is a meta-package over per-architecture cache wheels pinned to its own
  # version, and images that ship an older FlashInfer (vLLM's) carry older ones. Install exactly
  # the parts it declares, at its version, without re-resolving anything else.
  mapfile -t jit_parts < <("$venv/bin/python" - "${COLLX_FLASHINFER_JIT_CACHE_SPEC%%==*}" <<'PY'
import sys
from importlib import metadata

for requirement in metadata.requires(sys.argv[1]) or ():
    if requirement.startswith("flashinfer-jit-cache") and "extra ==" not in requirement:
        print(requirement.split(";")[0].replace(" ", ""))
PY
  ) || { collx_log "ERROR: FlashInfer jit-cache metadata is unreadable"; return 1; }
  [ "${#jit_parts[@]}" -eq 0 ] || "${pip[@]}" --no-deps "${jit_parts[@]}" >&2 2>&1 \
    || { collx_log "ERROR: FlashInfer jit-cache parts installation failed"; return 1; }
  # Every requirement of the installed set must already hold; pip check lines about the image's
  # own packages (its torch/NCCL pin disagreement among them) are not this venv's to settle.
  # shellcheck disable=SC2086
  # pip check exits non-zero for any broken requirement, the image's included, so only the
  # filter's verdict counts.
  { "$venv/bin/python" -m pip check 2>&1 || true; } | python3 -c '
import re, sys
ours = {re.split(r"[\[=<>]", spec)[0].lower().replace("_", "-") for spec in sys.argv[1:]}
def mine(name):
    return name in ours or name.startswith("flashinfer-jit-cache")
bad = [line for line in sys.stdin.read().splitlines()
       if mine(line.split(" ", 1)[0].lower().replace("_", "-"))]
print("\n".join(bad), file=sys.stderr)
sys.exit(1 if bad else 0)' $COLLX_FLASHINFER_SPEC "$COLLX_FLASHINFER_JIT_CACHE_SPEC" \
    || { collx_log "ERROR: FlashInfer environment has unmet requirements"; return 1; }
  flashinfer_activate "$root" || return 1
  flashinfer_promote_aot || { collx_log "ERROR: FlashInfer all-to-all build failed"; return 1; }
  flashinfer_probe || { collx_log "ERROR: FlashInfer import probe failed"; return 1; }
  : > "$root/.ready"
}

flashinfer_ep_prepare() {
  local arch shared
  command -v python3 >/dev/null || { collx_log "ERROR: python3 unavailable for FlashInfer EP"; return 1; }
  arch="$(cuda_arch)" || return 1
  shared="$(flashinfer_cache_root "$arch")" || shared=""
  if [ -n "$shared" ]; then
    CACHE_ROOT="$shared"
    command -v flock >/dev/null || { collx_log "ERROR: flock is required for FlashInfer caching"; return 1; }
    mkdir -p "${shared%/*}" || return 1
    collx_log "FlashInfer: preparing (shared cache $shared)"
    with_cache_lock "FlashInfer" "$shared" flashinfer_ready flashinfer_install \
      || { collx_log "ERROR: shared FlashInfer environment is incomplete"; return 1; }
  else
    CACHE_ROOT="/tmp/collectivex-flashinfer-cache-$arch-$(flashinfer_spec_digest)" || return 1
    collx_log "FlashInfer: preparing (node-local $CACHE_ROOT; no shared cache mounted)"
    flashinfer_ready "$CACHE_ROOT" || flashinfer_install "$CACHE_ROOT" || return 1
  fi
  flashinfer_activate "$CACHE_ROOT" || return 1
  flashinfer_probe || { collx_log "ERROR: FlashInfer EP one-sided A2A unavailable"; return 1; }
}

# Host verbs provider (collx_add_host_verbs_provider_mount). The vendor's build targets the host's
# distro: Broadcom's libbnxt_re 237.1.137.0 is linked on Ubuntu 24.04 and needs GLIBC_2.38 for one
# symbol, __isoc23_strtol (C23 strtol, which only adds 0b prefixes), while the ROCm images ship glibc
# 2.35. That symbol is unversioned and served by a forwarding shim, and the now-unused GLIBC_2.38
# requirement is retargeted to GLIBC_2.2.5 (elf_vernaux.py). The result must load and must list the
# registry's RDMA devices, or preparation fails. It replaces the image's provider of the same name
# in this node's writable container, which every rank of the shard then loads.
readonly VERBS_PATCHELF_SPEC="patchelf==0.17.2.1"

# The RDMA devices libibverbs lists, space-separated. ibv_devices is not in every image.
verbs_device_names() {
  python3 - <<'PY'
import ctypes
lib = ctypes.CDLL("libibverbs.so.1")
lib.ibv_get_device_list.restype = ctypes.POINTER(ctypes.c_void_p)
lib.ibv_get_device_name.restype = ctypes.c_char_p
count = ctypes.c_int(0)
devices = lib.ibv_get_device_list(ctypes.byref(count))
print(" ".join(lib.ibv_get_device_name(ctypes.c_void_p(devices[i])).decode()
               for i in range(count.value)) if devices else "")
PY
}

verbs_lists_registry_devices() {
  local listed device
  listed=" $(verbs_device_names 2>/dev/null) "
  for device in ${COLLX_RDMA_DEVICES//,/ }; do
    [[ "$listed" == *" ${device%%:*} "* ]] || return 1
  done
}

verbs_provider_prepare() {
  local provider="${COLLX_HOST_VERBS_PROVIDER:-}" name source providers target work
  [ -n "$provider" ] && [ "${COLLX_NODES:-1}" -gt 1 ] && [ "${COLLX_TRANSPORT:-}" != mnnvl ] || return 0
  # An image that ships a provider for this driver (the vLLM ROCm images carry Broadcom's) needs
  # nothing from the host.
  if verbs_lists_registry_devices; then
    collx_log "verbs provider: the image's provider lists every registry device; keeping it"
    return 0
  fi
  name="${provider##*/}"
  source="$COLLX_HOST_VERBS_ROOT/$name"
  [ -r "$source" ] || { collx_log "ERROR: host verbs provider $name is not bound"; return 1; }
  # rdma-core tries its provider directory before the loader path, so a copy there wins over one
  # an image installs elsewhere (/usr/local/lib for Broadcom's packages).
  providers="$(ls -d /usr/lib/*-linux-gnu/libibverbs 2>/dev/null | head -n 1)"
  [ -n "$providers" ] || { collx_log "ERROR: the image has no rdma-core provider directory"; return 1; }
  target="$providers/$name"
  work="$(mktemp -d /tmp/collectivex-verbs.XXXXXX)" || return 1
  cp "$source" "$work/$name" || return 1
  if ! python3 -c 'import ctypes, sys; ctypes.CDLL(sys.argv[1])' "$work/$name" 2>/dev/null; then
    collx_log "verbs provider: $name needs a newer glibc; shimming __isoc23_strtol"
    printf '%s\n' '#include <stdlib.h>' \
      'long __isoc23_strtol(const char *s, char **end, int base) { return strtol(s, end, base); }' \
      > "$work/cx_c23.c"
    cc -shared -fPIC -O2 -o "$work/libcx_c23.so" "$work/cx_c23.c" \
      && python3 -m pip install -q --disable-pip-version-check --no-input \
        --target "$work/patchelf" "$VERBS_PATCHELF_SPEC" >&2 \
      && "$work/patchelf/bin/patchelf" --clear-symbol-version __isoc23_strtol \
        --add-needed libcx_c23.so --set-rpath '$ORIGIN' "$work/$name" \
      && python3 "$COLLX_RUNTIME_DIR/elf_vernaux.py" "$work/$name" GLIBC_2.38 GLIBC_2.2.5 \
      || { collx_log "ERROR: verbs provider patch failed"; return 1; }
    python3 -c 'import ctypes, sys; ctypes.CDLL(sys.argv[1])' "$work/$name" \
      || { collx_log "ERROR: patched verbs provider does not load"; return 1; }
    cp "$work/libcx_c23.so" "$providers/libcx_c23.so" || return 1
  fi
  cp "$work/$name" "$target.collectivex" && mv -f "$target.collectivex" "$target" \
    || { collx_log "ERROR: verbs provider install failed"; return 1; }
  rm -rf "$work"
  verbs_lists_registry_devices \
    || { collx_log "ERROR: verbs provider $name does not list every registry device"; return 1; }
  collx_log "verbs provider: installed host $name"
}

main() {
  collx_apply_network_profile "${COLLX_NODES:-1}" "${COLLX_TRANSPORT:-}" || return 1
  verbs_provider_prepare || return 1
  validate_container_network || return 1
  ofi_nccl_prepare || return 1
  case "$COLLX_BENCH" in
    deepep-v2) deepep_prepare || return 1 ;;
    mori)
      python3 -c "import mori" \
        || { collx_log "ERROR: MoRI backend import failed"; return 1; }
      ;;
    uccl-ep) uccl_prepare || return 1 ;;
    nccl-ep) nccl_ep_prepare || return 1 ;;
    flashinfer-ep) flashinfer_ep_prepare || return 1 ;;
    # The official vLLM image ships the kernel; assert it before the cases start.
    swap-blocks)
      python3 -c "from vllm._custom_ops import swap_blocks" \
        || { collx_log "ERROR: vLLM swap_blocks import failed"; return 1; }
      ;;
    *)
      collx_log "ERROR: unknown backend preparation request"
      return 1
      ;;
  esac
  write_rank_env
}

rc=0; main || rc=$?
collx_log "backend preparation: bench=$COLLX_BENCH rc=$rc"
exit "$rc"
