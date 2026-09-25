# shellcheck shell=bash
# Sourced by prepare_backend.sh inside the allocated container.

# UCCL-EP lifecycle

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

# UCCL is built in-container against the image's torch, not with upstream `build.sh` (that spins
# up its own Docker image and cannot run inside enroot/pyxis). The built deep_ep/uccl packages
# persist under a cache root and reach the ranks via PYTHONPATH, so later allocations skip the build.

# Returns non-zero when no shared cache is mounted (manual runs); the caller then builds node-local.
uccl_cache_root() {
  local arch
  arch="$(printf '%s' "$1" | tr -cs 'A-Za-z0-9_.-' '-')"
  backend_cache_root uccl-ep "${arch#-}" "${COLLX_UCCL_COMMIT:0:12}"
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
  if [ -e "$root" ] || [ -L "$root" ]; then
    rm -rf "$root" || { collx_log "ERROR: incomplete UCCL cache-reset failed"; return 1; }
  fi
  mkdir -m 700 "$root" || { collx_log "ERROR: UCCL cache-create failed"; return 1; }
  collx_log "UCCL-EP: building $COLLX_UCCL_COMMIT from source (USE_DMABUF, PER_EXPERT_BATCHING)"
  # Some sglang/rocm images mark the system env externally-managed (PEP 668).
  { python3 -m pip install -q --disable-pip-version-check --no-input nanobind \
      || python3 -m pip install -q --disable-pip-version-check --no-input \
           --break-system-packages nanobind; } >&2 2>&1 \
    || { collx_log "ERROR: UCCL nanobind install failed"; return 1; }
  collx_materialize_uccl_source "$source_dir" \
    || { collx_log "ERROR: UCCL staged source is invalid"; return 1; }
  if [ "${COLLX_VENDOR:-nvidia}" = amd ]; then
    arch_env="PYTORCH_ROCM_ARCH=$arch"
    # hipMallocManaged fails on our CDNA nodes (even 4 KiB, regardless of XNACK or memlock). UCCL's
    # HIP CPU-proxy path uses cudaMallocManaged for the d2h channel handles and proxy atomic buffer;
    # pinned host memory (cudaMallocHost) is coherent and device-accessible on gfx942/gfx950.
    sed -i 's/cudaMallocManaged/cudaMallocHost/g' \
      "$source_dir/ep/src/uccl_ep.cc" "$source_dir/ep/src/uccl_proxy.cpp" \
      || { collx_log "ERROR: UCCL AMD managed-memory patch failed"; return 1; }
  else
    arch_env="TORCH_CUDA_ARCH_LIST=$arch"
  fi
  ( cd "$source_dir/ep" \
      && env USE_DMABUF=1 PER_EXPERT_BATCHING=1 "$arch_env" python3 setup.py install ) >&2 2>&1 \
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
  : > "$root/.ready"
}

uccl_prepare() {
  local arch root
  command -v python3 >/dev/null || { collx_log "ERROR: python3 unavailable for UCCL build"; return 1; }
  if [ "${COLLX_VENDOR:-nvidia}" = amd ]; then
    arch="$(uccl_rocm_arch)" || return 1
  else
    arch="$(cuda_arch)" || return 1
  fi
  if root="$(uccl_cache_root "$arch")"; then
    command -v flock >/dev/null \
      || { collx_log "ERROR: flock is required for UCCL-EP caching"; return 1; }
    mkdir -p "${root%/*}" || return 1
    collx_log "UCCL-EP: preparing $COLLX_UCCL_COMMIT (shared cache $root)"
    if ! install_backend_cache "UCCL" "$root" site uccl_install "$arch"; then
      collx_log "ERROR: shared UCCL-EP environment is incomplete"; return 1
    fi
  else
    root="/tmp/collectivex-uccl-cache-$COLLX_UCCL_COMMIT"
    collx_log "UCCL-EP: preparing $COLLX_UCCL_COMMIT (node-local $root; no shared cache mounted)"
    if ! backend_cache_ready "$root" site; then
      uccl_install "$root" "$arch" || return 1
    fi
  fi
  uccl_activate "$root" || return 1
  uccl_probe || { collx_log "ERROR: UCCL import probe failed"; return 1; }
  collx_log "UCCL-EP ready ($COLLX_UCCL_COMMIT, deep_ep wrapper over uccl.ep CPU-proxy runtime)"
}
