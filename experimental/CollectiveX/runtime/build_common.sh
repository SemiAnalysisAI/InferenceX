# shellcheck shell=bash
# Sourced by prepare_backend.sh inside the allocated container.

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

# Cache identity is shared across installers; each caller supplies its exact recipe suffix.
backend_cache_root() {
  local backend="$1" arch="$2" suffix="$3" cpu base image
  cpu="$(uname -m)"
  [[ "$cpu" =~ ^[A-Za-z0-9._-]+$ ]] || return 1
  base="${COLLX_BACKEND_CACHE_ROOT:-}"
  [[ "$base" = /* ]] || return 1
  image="$(printf '%s' "${COLLECTIVEX_IMAGE:-manual}" | tr -cs 'A-Za-z0-9_.-' '-')"
  printf '%s/%s-%s-%s-%s-%s' "$base" "$backend" "$cpu" "$arch" "${image#-}" "$suffix"
}

backend_cache_ready() {
  local root="$1" kind="$2"
  [ -f "$root/.ready" ] || return 1
  case "$kind" in
    venv) [ -x "$root/venv/bin/python" ] && [ -d "$root/source" ] ;;
    site) [ -d "$root/site" ] ;;
    *) return 1 ;;
  esac
}

# Installation stays in a subshell under the same exclusive lock. The caller activates and
# probes after this returns, including when another allocation already populated the cache.
install_backend_cache() (
  local label="$1" root="$2" kind="$3" installer="$4" lock_path
  shift 4
  lock_path="${root}.lock"
  [ ! -L "$lock_path" ] \
    || { collx_log "ERROR: $label cache lock is unsafe"; exit 1; }
  (umask 077; : >> "$lock_path") && chmod 600 "$lock_path" \
    || { collx_log "ERROR: $label cache-lock-create failed"; exit 1; }
  exec 9<>"$lock_path" \
    || { collx_log "ERROR: $label cache-lock-open failed"; exit 1; }
  flock 9 \
    || { collx_log "ERROR: $label cache-lock-acquire failed"; exit 1; }
  if ! backend_cache_ready "$root" "$kind"; then
    "$installer" "$root" "$@" || exit 1
  fi
)
