#!/usr/bin/env bash
# Prepare one backend per allocated node and persist its rank environment.
set -euo pipefail

cd /ix/experimental/CollectiveX
# shellcheck source=../runtime/common.sh
source runtime/common.sh

: "${COLLX_RUNNER:?COLLX_RUNNER not set}"
: "${COLLX_BENCH:?COLLX_BENCH not set}"

collx_log "backend preparation: runner=$COLLX_RUNNER bench=$COLLX_BENCH nodes=${COLLX_NODES:-1}"

# Fresh rank tasks source only these backend-created values. Network variables
# are reapplied by the rank wrapper from the platform profile.
readonly -a RANK_ENV_VARS=(
  PATH VIRTUAL_ENV LD_LIBRARY_PATH PYTHONPATH CUDA_HOME CPATH NVCC_PREPEND_FLAGS
  NVSHMEM_DIR EP_NCCL_ROOT_DIR EP_NVSHMEM_ROOT_DIR EP_JIT_CACHE_DIR
  EP_REUSE_NCCL_COMM NCCL_CUMEM_ENABLE UCCL_EP_ENABLE_AGGRESSIVE_ATOMIC
)
readonly -a DEEPEP_RANK_UNSETS=(EP_SUPPRESS_NCCL_CHECK)

# ---- discovery --------------------------------------------------------------

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
  local arch="$1" cpu base image
  cpu="$(uname -m)"
  [[ "$cpu" =~ ^[A-Za-z0-9._-]+$ ]] || return 1
  base="${COLLX_BACKEND_CACHE_ROOT:-}"
  [[ "$base" = /* ]] || return 1
  image="$(printf '%s' "${COLLECTIVEX_IMAGE:-manual}" | tr -cs 'A-Za-z0-9_.-' '-')"
  printf '%s/deepep-v2-%s-sm%s-%s-%s' \
    "$base" "$cpu" "${arch/./}" "${image#-}" "${COLLX_DEEPEP_V2_COMMIT:0:12}"
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
    "$venv/bin/python" nvidia-nvshmem-cu12 nvshmem)" \
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
    CPATH="$cccl:${CPATH:-}" \
    NVCC_PREPEND_FLAGS="-I$cccl ${NVCC_PREPEND_FLAGS:-}" \
    NVSHMEM_DIR="$overlay" \
    EP_NCCL_ROOT_DIR="$nccl_root" \
    EP_NVSHMEM_ROOT_DIR="$overlay" \
    EP_JIT_CACHE_DIR="/tmp/collectivex-deepep-v2-jit-$execution_id" \
    EP_REUSE_NCCL_COMM=1 \
    NCCL_CUMEM_ENABLE=1 \
    LD_LIBRARY_PATH="$overlay/lib:$nccl_root/lib:$nvshmem_package/lib:${LD_LIBRARY_PATH:-}"
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
    "numpy==2.2.6" "nvidia-nvshmem-cu12==3.3.9" >&2 2>&1 \
    || { collx_log "ERROR: DeepEP V2 build-tool installation failed"; return 1; }
  "${pip[@]}" --index-url https://download.pytorch.org/whl/cu130 \
    --extra-index-url https://pypi.org/simple "torch==2.10.0" >&2 2>&1 \
    || { collx_log "ERROR: torch 2.10.0+cu130 installation failed"; return 1; }
  # Torch pins NCCL 2.28.9; ElasticBuffer requires 2.30.4.
  "${pip[@]}" --force-reinstall --no-deps "nvidia-nccl-cu13==2.30.4" >&2 2>&1 \
    || { collx_log "ERROR: NCCL 2.30.4 installation failed"; return 1; }
  deepep_activate "$root" \
    || { collx_log "ERROR: DeepEP V2 environment activation failed"; return 1; }
  collx_materialize_deepep_source "$source_dir" \
    || { collx_log "ERROR: DeepEP V2 staged source is invalid"; return 1; }
  (cd "$source_dir" && TORCH_CUDA_ARCH_LIST="$arch" MAX_JOBS=16 \
    "$venv/bin/python" -m pip install -q --no-build-isolation --no-deps \
      --force-reinstall .) >&2 2>&1 \
    || { collx_log "ERROR: DeepEP V2 build failed"; return 1; }
  deepep_probe \
    || { collx_log "ERROR: DeepEP V2 import probe failed"; return 1; }
  : > "$root/.ready"
}

# ---- DeepEP lifecycle -------------------------------------------------------

deepep_prepare() {
  local arch root venv source_dir ready lock_path
  arch="$(cuda_arch)" || return 1
  root="$(deepep_cache_root "$arch")" || return 1
  venv="$root/venv"; source_dir="$root/source"; ready="$root/.ready"
  lock_path="${root}.lock"
  command -v flock >/dev/null || { collx_log "ERROR: flock is required for DeepEP V2"; return 1; }
  mkdir -p "${root%/*}" || return 1
  collx_log "DeepEP V2: preparing PR #605 with upstream PR #630 and #640 fixes ($COLLX_DEEPEP_V2_COMMIT)"
  if ! (
    [ ! -L "$lock_path" ] \
      || { collx_log "ERROR: DeepEP V2 cache lock is unsafe"; exit 1; }
    (umask 077; : >> "$lock_path") && chmod 600 "$lock_path" \
      || { collx_log "ERROR: DeepEP V2 cache-lock-create failed"; exit 1; }
    exec 9<>"$lock_path" \
      || { collx_log "ERROR: DeepEP V2 cache-lock-open failed"; exit 1; }
    flock 9 \
      || { collx_log "ERROR: DeepEP V2 cache-lock-acquire failed"; exit 1; }
    if [ ! -f "$ready" ] || [ ! -x "$venv/bin/python" ] || [ ! -d "$source_dir" ]; then
      deepep_install "$root" "$arch" || exit 1
    fi
  ); then
    collx_log "ERROR: shared DeepEP V2 environment is incomplete"
    return 1
  fi
  deepep_activate "$root" || return 1
  deepep_probe || { collx_log "ERROR: DeepEP V2 shared runtime probe failed"; return 1; }
  collx_log "DeepEP V2 ready ($COLLX_DEEPEP_V2_COMMIT, ElasticBuffer, NCCL Device API; LSA/Gin selected by adapter)"
}

# ---- UCCL-EP lifecycle ------------------------------------------------------

# Registry arch string for the runner (gfx942/gfx950 on AMD) for PYTORCH_ROCM_ARCH.
uccl_rocm_arch() {
  python3 - "$COLLX_RUNNER" <<'PY'
import json, sys
print(json.load(open("configs/platform_config.json"))["platforms"][sys.argv[1]]["arch"])
PY
}

uccl_probe() {
  # import torch FIRST so libc10 is resident before the uccl.ep extension dlopens (it links
  # libc10/libtorch); importing deep_ep before torch fails with "libc10.so: cannot open".
  python3 - <<'PY'
import torch  # noqa: F401
import deep_ep
from deep_ep import Buffer
assert hasattr(Buffer, "low_latency_dispatch") and hasattr(Buffer, "get_dispatch_layout")
PY
}

# Direct in-container source build against the image's torch — validated on h200 (sglang
# cu130). NOT `build.sh` (that spins up its own Docker image to make a wheel and cannot run
# inside enroot/pyxis). single-slurm and mi-amds run the writable container as remapped root,
# so the build needs no venv. verbs/nl/numa dev headers ship in the sglang/rocm images; only
# nanobind must be added. The built deep_ep/uccl packages are persisted under a cache root and
# put on PYTHONPATH (which write_rank_env carries to the ranks), so later allocations reuse them
# without recompiling — the same copy+PYTHONPATH scheme the mi-tw Docker launcher already uses.

# Cache root keyed by cpu + build arch + image + pinned commit, under the shared /cx-cache mount
# ($COLLX_BACKEND_CACHE_ROOT). Returns non-zero when no shared cache is mounted (manual runs), so
# the caller falls back to a node-local build. Mirrors deepep_cache_root.
uccl_cache_root() {
  local arch="$1" cpu base image
  cpu="$(uname -m)"
  [[ "$cpu" =~ ^[A-Za-z0-9._-]+$ ]] || return 1
  base="${COLLX_BACKEND_CACHE_ROOT:-}"
  [[ "$base" = /* ]] || return 1
  image="$(printf '%s' "${COLLECTIVEX_IMAGE:-manual}" | tr -cs 'A-Za-z0-9_.-' '-')"
  arch="$(printf '%s' "$arch" | tr -cs 'A-Za-z0-9_.-' '-')"
  printf '%s/uccl-ep-%s-%s-%s-%s' \
    "$base" "$cpu" "${arch#-}" "${image#-}" "${COLLX_UCCL_COMMIT:0:12}"
}

# Put the persisted build ($root/site) on PYTHONPATH for the probe and the rank tasks; mirror the
# minimal runtime bits of deepep_activate. CDNA additionally needs the aggressive host-atomic path.
uccl_activate() {
  local site="$1/site"
  [ -d "$site" ] || { collx_log "ERROR: UCCL cache site is unavailable"; return 1; }
  export PYTHONPATH="$site${PYTHONPATH:+:$PYTHONPATH}"
  [ "${COLLX_VENDOR:-nvidia}" != amd ] || export UCCL_EP_ENABLE_AGGRESSIVE_ATOMIC=1
}

# Build UCCL from source into $root/site (fresh root, with a .ready marker written LAST). The
# build installs into the image's system python as a sandbox, then copies the built deep_ep/uccl
# packages into the cache; the runtime imports them via PYTHONPATH (uccl_activate), so cache-hit
# and cache-miss paths import identically. Only nanobind is added to the image.
uccl_install() {
  local root="$1" arch="$2" source_dir="/tmp/collectivex-uccl-$COLLX_UCCL_COMMIT" arch_env sp
  if [ -e "$root" ] || [ -L "$root" ]; then
    rm -rf "$root" || { collx_log "ERROR: incomplete UCCL cache-reset failed"; return 1; }
  fi
  mkdir -m 700 "$root" || { collx_log "ERROR: UCCL cache-create failed"; return 1; }
  collx_log "UCCL-EP: building $COLLX_UCCL_COMMIT from source (USE_DMABUF, PER_EXPERT_BATCHING)"
  # Plain install first; some sglang/rocm image variants mark the system env externally-managed
  # (PEP 668), so fall back to --break-system-packages (a no-op on older pip that lacks the flag).
  { python3 -m pip install -q --disable-pip-version-check --no-input nanobind \
      || python3 -m pip install -q --disable-pip-version-check --no-input \
           --break-system-packages nanobind; } >&2 2>&1 \
    || { collx_log "ERROR: UCCL nanobind install failed"; return 1; }
  collx_materialize_uccl_source "$source_dir" \
    || { collx_log "ERROR: UCCL staged source is invalid"; return 1; }
  if [ "${COLLX_VENDOR:-nvidia}" = amd ]; then
    arch_env="PYTORCH_ROCM_ARCH=$arch"
    # Managed/unified memory (cudaMallocManaged) is unavailable on our CDNA nodes (hipMallocManaged
    # fails even for 4 KiB, regardless of XNACK / --privileged / memlock). UCCL's HIP CPU-proxy path
    # uses it for the d2h channel handles + proxy atomic buffer; pinned host memory (cudaMallocHost)
    # is coherent + device-accessible on gfx942/gfx950 and is already used elsewhere in UCCL (e.g.
    # the RDMA scratch), so swap the two on the runtime path before building. Validated on mi300x-tw
    # (bf16/fp8 normal green). NB: build the WHOLE tree (materialize copies it) — the ROCm path
    # includes top-level util/gpu_rt.h.
    sed -i 's/cudaMallocManaged/cudaMallocHost/g' \
      "$source_dir/ep/src/uccl_ep.cc" "$source_dir/ep/src/uccl_proxy.cpp" \
      || { collx_log "ERROR: UCCL AMD managed-memory patch failed"; return 1; }
  else
    arch_env="TORCH_CUDA_ARCH_LIST=$arch"
  fi
  ( cd "$source_dir/ep" \
      && env USE_DMABUF=1 PER_EXPERT_BATCHING=1 "$arch_env" python3 setup.py install ) >&2 2>&1 \
    || { collx_log "ERROR: UCCL ep extension build failed"; return 1; }
  # Install the wrapper WITHOUT its deps: install_requires=["uccl"] resolves to the PyPI
  # uccl metapackage, which depends on the prebuilt uccl-cu12 wheel — absent on ROCm (hard
  # fail) and wrong even on CUDA, since our from-source ep build already provides uccl.ep in
  # site-packages/uccl. --no-deps makes the source build authoritative on both vendors.
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
  : > "$root/.ready"
}

# UCCL-EP lifecycle: build once per (arch, image, commit) into the shared /cx-cache behind an
# flock + .ready marker, reused on every later allocation (mirrors deepep_prepare); fall back to
# a node-local build when no shared cache is mounted (e.g. a manual run).
uccl_prepare() {
  local arch root ready lock_path
  command -v python3 >/dev/null || { collx_log "ERROR: python3 unavailable for UCCL build"; return 1; }
  if [ "${COLLX_VENDOR:-nvidia}" = amd ]; then
    arch="$(uccl_rocm_arch)" || return 1
  else
    arch="$(cuda_arch)" || return 1
  fi
  if root="$(uccl_cache_root "$arch")"; then
    ready="$root/.ready"; lock_path="${root}.lock"
    command -v flock >/dev/null \
      || { collx_log "ERROR: flock is required for UCCL-EP caching"; return 1; }
    mkdir -p "${root%/*}" || return 1
    collx_log "UCCL-EP: preparing $COLLX_UCCL_COMMIT (shared cache $root)"
    if ! (
      [ ! -L "$lock_path" ] || { collx_log "ERROR: UCCL cache lock is unsafe"; exit 1; }
      (umask 077; : >> "$lock_path") && chmod 600 "$lock_path" \
        || { collx_log "ERROR: UCCL cache-lock-create failed"; exit 1; }
      exec 9<>"$lock_path" || { collx_log "ERROR: UCCL cache-lock-open failed"; exit 1; }
      flock 9 || { collx_log "ERROR: UCCL cache-lock-acquire failed"; exit 1; }
      if [ ! -f "$ready" ] || [ ! -d "$root/site" ]; then
        uccl_install "$root" "$arch" || exit 1
      fi
    ); then
      collx_log "ERROR: shared UCCL-EP environment is incomplete"; return 1
    fi
  else
    root="/tmp/collectivex-uccl-cache-$COLLX_UCCL_COMMIT"
    collx_log "UCCL-EP: preparing $COLLX_UCCL_COMMIT (node-local $root; no shared cache mounted)"
    if [ ! -f "$root/.ready" ] || [ ! -d "$root/site" ]; then
      uccl_install "$root" "$arch" || return 1
    fi
  fi
  uccl_activate "$root" || return 1
  uccl_probe || { collx_log "ERROR: UCCL import probe failed"; return 1; }
  collx_log "UCCL-EP ready ($COLLX_UCCL_COMMIT, deep_ep wrapper over uccl.ep CPU-proxy runtime)"
}

# ---- container boundary ----------------------------------------------------

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
    device="${device#=}"
    rdma_name="${device%%:*}"
    [ -d "/sys/class/infiniband/$rdma_name" ] \
      || { collx_log "ERROR: configured scale-out RDMA device is absent"; return 1; }
  done
}

main() {
  collx_apply_network_profile "${COLLX_NODES:-1}" "${COLLX_TRANSPORT:-}" || return 1
  validate_container_network || return 1
  case "$COLLX_BENCH" in
    deepep-v2) deepep_prepare || return 1 ;;
    mori)
      python3 -c "import mori" \
        || { collx_log "ERROR: MoRI backend import failed"; return 1; }
      ;;
    uccl-ep) uccl_prepare || return 1 ;;
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
