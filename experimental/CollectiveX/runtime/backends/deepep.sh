# shellcheck shell=bash
# Sourced by prepare_backend.sh inside the allocated container.

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
  # The NVSHMEM wheel is part of the built venv's identity (see common.sh: the cu12
  # wheel on cu130 images broke sm103), so it keys the cache and a spec change rebuilds.
  local nvshmem_key="${COLLX_DEEPEP_V2_NVSHMEM_SPEC#nvidia-}"
  nvshmem_key="${nvshmem_key//==/-}"
  local torch_key="${COLLX_DEEPEP_V2_TORCH_SPEC//==/-}"
  local build_gen="${COLLX_DEEPEP_V2_BUILD_GEN:?}"
  [[ "$nvshmem_key" =~ ^[A-Za-z0-9._-]+$ && "$torch_key" =~ ^[A-Za-z0-9._-]+$ \
     && "$build_gen" =~ ^[A-Za-z0-9._-]+$ ]] || return 1
  printf '%s/deepep-v2-%s-sm%s-%s-%s-%s-%s-%s' \
    "$base" "$cpu" "${arch/./}" "${image#-}" "${COLLX_DEEPEP_V2_COMMIT:0:12}" \
    "$torch_key" "$nvshmem_key" "$build_gen"
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
    "numpy==2.2.6" "$COLLX_DEEPEP_V2_NVSHMEM_SPEC" >&2 2>&1 \
    || { collx_log "ERROR: DeepEP V2 build-tool installation failed"; return 1; }
  "${pip[@]}" --index-url https://download.pytorch.org/whl/cu130 \
    --extra-index-url https://pypi.org/simple "$COLLX_DEEPEP_V2_TORCH_SPEC" >&2 2>&1 \
    || { collx_log "ERROR: torch 2.10.0+cu130 installation failed"; return 1; }
  # Torch pins NCCL 2.28.9; ElasticBuffer requires 2.30.4.
  "${pip[@]}" --force-reinstall --no-deps "nvidia-nccl-cu13==2.30.4" >&2 2>&1 \
    || { collx_log "ERROR: NCCL 2.30.4 installation failed"; return 1; }
  deepep_activate "$root" \
    || { collx_log "ERROR: DeepEP V2 environment activation failed"; return 1; }
  collx_materialize_deepep_source "$source_dir" \
    || { collx_log "ERROR: DeepEP V2 staged source is invalid"; return 1; }
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
  collx_log "DeepEP V2: preparing PR #605 with upstream PR #630 and #640 fixes ($COLLX_DEEPEP_V2_COMMIT)"
  if ! install_backend_cache "DeepEP V2" "$root" venv deepep_install "$arch"; then
    collx_log "ERROR: shared DeepEP V2 environment is incomplete"
    return 1
  fi
  deepep_activate "$root" || return 1
  deepep_probe || { collx_log "ERROR: DeepEP V2 shared runtime probe failed"; return 1; }
  collx_log "DeepEP V2 ready ($COLLX_DEEPEP_V2_COMMIT, ElasticBuffer, NCCL Device API; LSA/Gin selected by adapter)"
}
