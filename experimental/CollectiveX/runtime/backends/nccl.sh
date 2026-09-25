# shellcheck shell=bash
# Sourced by prepare_backend.sh inside the allocated container.

# NCCL EP lifecycle

nccl_ep_spec_slug() {
  printf '%s' "$COLLX_NCCL_EP_SPEC" | tr -cs 'A-Za-z0-9_.-' '-'
}

# Returns non-zero when no shared cache is mounted (manual runs); the caller then installs node-local.
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
  # unquoted on purpose: it carries two whitespace-separated pip specs.
  # shellcheck disable=SC2086
  python3 -m pip install -q --disable-pip-version-check --no-input \
      --target "$site" $COLLX_NCCL_EP_SPEC >&2 2>&1 \
    || { collx_log "ERROR: NCCL EP wheel install failed"; return 1; }
  nccl_ep_activate "$root" \
    || { collx_log "ERROR: NCCL EP environment activation failed"; return 1; }
  nccl_ep_probe || { collx_log "ERROR: NCCL EP import probe failed"; return 1; }
  : > "$root/.ready"
}

nccl_ep_prepare() {
  local arch root
  command -v python3 >/dev/null || { collx_log "ERROR: python3 unavailable for NCCL EP"; return 1; }
  arch="$(cuda_arch)" || return 1
  if root="$(nccl_ep_cache_root "$arch")"; then
    command -v flock >/dev/null \
      || { collx_log "ERROR: flock is required for NCCL EP caching"; return 1; }
    mkdir -p "${root%/*}" || return 1
    collx_log "NCCL EP: preparing $COLLX_NCCL_EP_SPEC (shared cache $root)"
    if ! install_backend_cache "NCCL EP" "$root" site nccl_ep_install; then
      collx_log "ERROR: shared NCCL EP environment is incomplete"; return 1
    fi
  else
    root="/tmp/collectivex-nccl-ep-cache-$(nccl_ep_spec_slug)"
    collx_log "NCCL EP: preparing $COLLX_NCCL_EP_SPEC (node-local $root; no shared cache mounted)"
    if ! backend_cache_ready "$root" site; then
      nccl_ep_install "$root" || return 1
    fi
  fi
  nccl_ep_activate "$root" || return 1
  nccl_ep_probe || { collx_log "ERROR: NCCL EP import probe failed"; return 1; }
  collx_log "NCCL EP ready ($COLLX_NCCL_EP_SPEC; libnccl_ep.so JIT runtime, NCCL Device API LSA/GIN)"
}
