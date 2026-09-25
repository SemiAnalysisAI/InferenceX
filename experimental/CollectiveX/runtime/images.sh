# shellcheck shell=bash
# Container identity, import locking, and squash-cache reuse. Sourced by common.sh.

# Enroot cannot reliably import a digest-qualified Docker Hub reference non-interactively, so the
# import uses the tag; the digest only stamps/checks the squash sidecar. An unresolved digest
# reuses what is staged (COLLX_IMAGE_REFRESH=1 is then the update hatch).
collx_select_image() {
  local image="$1" digest
  [[ "$image" =~ ^[A-Za-z0-9._/-]+:[A-Za-z0-9._-]+$ ]] \
    || collx_die "configured image reference is malformed"
  export COLLECTIVEX_IMAGE="$image"
  if [[ ! "${COLLX_IMAGE_DIGEST:-}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
    digest="$(python3 "$COLLX_RUNTIME_DIR/probe.py" image-digest "$image" \
      2>/dev/null)" || digest=""
    if [[ "$digest" =~ ^sha256:[0-9a-f]{64}$ ]]; then
      export COLLX_IMAGE_DIGEST="$digest"
      collx_log "image digest $digest"
    else
      unset COLLX_IMAGE_DIGEST
      collx_log "image digest unresolved; staged squash reused as-is"
    fi
  fi
}

# One squash per (platform, image reference): never per run (re-copies 30-65GB) and never per
# digest (a transient registry blip would miss the staged file and re-import). The digest lives
# in a `<sq>.digest` sidecar and decides staleness.
collx_squash_path() {
  local squash_dir="$1" image="$2" platform
  case "${COLLX_IMAGE_PLATFORM:-}" in
    linux/amd64) platform="" ;;
    linux/arm64) platform="_linux_arm64" ;;
    *) return 1 ;;
  esac
  printf '%s' "$squash_dir/${platform}_$(printf '%s' "$image" | sed 's#[/:@#]#_#g').sqsh"
}

# Echoes "reuse" or why to re-import (callers hold the import lock). refresh_epoch discards only
# files staged before this launch, so concurrent legs of a refreshing run still import once.
collx_squash_verdict() {
  local sq="$1" digest="$2" refresh_epoch="$3" stamp="" mtime
  [ -e "$sq" ] || { printf 'absent'; return; }
  if [ -n "$refresh_epoch" ]; then
    # GNU stat on the clusters; the BSD fallback keeps the seam testable on macOS.
    mtime="$(stat -c %Y "$sq" 2>/dev/null || stat -f %m "$sq" 2>/dev/null || echo 0)"
    [ "$mtime" -ge "$refresh_epoch" ] || { printf 'refresh-requested'; return; }
  fi
  [ ! -f "$sq.digest" ] || IFS= read -r stamp < "$sq.digest" || stamp=""
  if [ -n "$digest" ] && [ -n "$stamp" ] && [ "$stamp" != "$digest" ]; then
    printf 'digest-moved'; return
  fi
  printf 'reuse'
}

# Echoes the squash file path.
collx_ensure_squash() {
  local squash_dir="$1" image="$2" key sq locks lock_fd log verdict refresh_epoch=""
  local enroot_local="" import_rc=0 machine
  if [ "${COLLX_IMAGE_REFRESH:-0}" = 1 ]; then
    refresh_epoch="${COLLX_LAUNCH_EPOCH:-$(date +%s)}"
  fi
  log="$(collx_private_log_path container-import)"
  machine="$(uname -m)"
  case "${COLLX_IMAGE_PLATFORM:-}:$machine" in
    linux/amd64:x86_64|linux/amd64:amd64|linux/arm64:aarch64|linux/arm64:arm64) ;;
    *) collx_log_tail "$log"; return 1 ;;
  esac
  mkdir -p "$squash_dir" 2>> "$log" \
    || { collx_log_tail "$log"; return 1; }
  sq="$(collx_squash_path "$squash_dir" "$image")" \
    || { collx_log_tail "$log"; return 1; }
  key="${sq##*/}"
  key="${key%.sqsh}"
  locks="$squash_dir/.locks"
  mkdir -p "$locks" 2>> "$log" \
    || { collx_log_tail "$log"; return 1; }
  { exec {lock_fd}>"$locks/${key}.lock"; } 2>> "$log" \
    || { collx_log_tail "$log"; return 1; }
  # A concurrent leg holds the content-keyed lock for its full import (~18 minutes for the 32 GB
  # sglang squash on b300), so the wait must outlast an import, and a timeout must log: the
  # empty import log would otherwise make this a silent launcher death.
  flock -w 2700 "$lock_fd" 2>> "$log" \
    || { collx_log "ERROR: timed out waiting for the container import lock"
         collx_log_tail "$log"; return 1; }
  verdict="$(collx_squash_verdict "$sq" "${COLLX_IMAGE_DIGEST:-}" "$refresh_epoch")"
  [ "$verdict" != reuse ] || unsquashfs -l "$sq" >/dev/null 2>&1 || verdict=invalid
  if [ "$verdict" = reuse ]; then
    collx_log "container squash ready (reusing staged import)"
  else
    collx_log "importing configured container image ($verdict)"
    rm -f "$sq" "$sq.digest" 2>> "$log" \
      || { collx_log_tail "$log"; return 1; }
    # </dev/null: never block on an interactive password prompt.
    if [ "${COLLX_ENROOT_LOCAL_IMPORT:-0}" = 1 ]; then
      enroot_local="$(mktemp -d /tmp/inferencex-collectivex-enroot.XXXXXX)" \
        || { collx_log_tail "$log"; return 1; }
      (
        trap 'rm -rf -- "$enroot_local"' EXIT
        export ENROOT_TEMP_PATH="$enroot_local/tmp"
        export ENROOT_CACHE_PATH="$enroot_local/cache"
        export ENROOT_DATA_PATH="$enroot_local/data"
        export ENROOT_RUNTIME_PATH="$enroot_local/run"
        mkdir -p "$ENROOT_TEMP_PATH" "$ENROOT_CACHE_PATH" \
          "$ENROOT_DATA_PATH" "$ENROOT_RUNTIME_PATH"
        enroot import -o "$sq" "docker://$image" </dev/null
      ) >> "$log" 2>&1 || import_rc=$?
      rm -rf -- "$enroot_local" >/dev/null 2>&1 || true
      [ "$import_rc" = 0 ] \
        || { collx_log_tail "$log"; return 1; }
    else
      enroot import -o "$sq" "docker://$image" </dev/null >> "$log" 2>&1 \
        || { collx_log_tail "$log"; return 1; }
    fi
    unsquashfs -l "$sq" >> "$log" 2>&1 \
      || { collx_log_tail "$log"; return 1; }
    # World-readable so another account's launcher can reuse the squash instead of dying on
    # pyxis's "Invalid image format" against a 0600 file.
    chmod a+r "$sq" 2>/dev/null || true
    printf '%s\n' "${COLLX_IMAGE_DIGEST:-}" > "$sq.digest" 2>> "$log" || true
    # Retired per-run names of this image; the age gate spares a file a concurrent old-generation
    # run may still be reading.
    find "$squash_dir" -maxdepth 1 -type f \
      -name "*_$(printf '%s' "$image" | sed 's#[/:@#]#_#g').sqsh" ! -name "${sq##*/}" \
      -mmin +2880 -delete 2>/dev/null || true
  fi
  flock -u "$lock_fd"
  exec {lock_fd}>&-
  echo "$sq"
}

# Importing on an allocated compute node makes multiarch tags resolve for the target
# architecture. The squash directory must be shared with the submit host.
collx_ensure_squash_on_job() {
  local job_id="$1" squash_dir="$2" image="$3" lock_dir="${4:-}" sq key lock
  local log_label=container-import log attempt rc refresh_epoch=""
  if [ "${COLLX_IMAGE_REFRESH:-0}" = 1 ]; then
    refresh_epoch="${COLLX_LAUNCH_EPOCH:-$(date +%s)}"
  fi
  # Squash storage can be a soft-mounted network filesystem: gb300's /data is NFSv3 over RDMA
  # (proto=rdma, soft), and a transport gap surfaces as `mkdir: cannot create directory '/data':
  # Protocol family not supported` that clears minutes later. Retrying is safe because the remote
  # block re-takes the lock and removes a partial file before re-importing.
  local max_attempts="${COLLX_IMPORT_ATTEMPTS:-3}"
  [[ "$job_id" =~ ^[0-9]+$ ]] || return 1
  case "${COLLX_SALLOC_ATTEMPT:-1}" in
    1) ;;
    2|3) log_label+="-a${COLLX_SALLOC_ATTEMPT}" ;;
    *) return 1 ;;
  esac
  sq="$(collx_squash_path "$squash_dir" "$image")" || return 1
  key="${sq##*/}"
  key="${key%.sqsh}"
  [ -n "$lock_dir" ] || lock_dir="$squash_dir/.locks"
  lock="$lock_dir/${key}.lock"
  for attempt in $(seq 1 "$max_attempts"); do
    # collx_private_log_path truncates, so one log per attempt keeps the failure that caused the retry.
    if [ "$attempt" -eq 1 ]; then
      log="$(collx_private_log_path "$log_label")"
    else
      log="$(collx_private_log_path "${log_label}-r${attempt}")"
    fi
    rc=0
    # Run once per node because some clusters use node-local squash storage.
    srun --jobid="$job_id" --nodes="${COLLX_NODES:-1}" --ntasks="${COLLX_NODES:-1}" \
      --ntasks-per-node=1 --chdir=/tmp \
      --export="$(collx_host_exports)" \
      bash -s -- "$sq" "$lock" "$image" "$COLLX_IMAGE_PLATFORM" "$refresh_epoch" \
      "${COLLX_IMAGE_DIGEST:-}" "$(printf '%s' "$image" | sed 's#[/:@#]#_#g')" \
      "${COLLX_IMPORT_TMPDIR:-}" \
      > "$log" 2>&1 <<'BASH' || rc=$?
set -eo pipefail
sq="$1"; lock="$2"; image="$3"; platform="$4"
refresh_epoch="${5:-}"; digest="${6:-}"; sanitized="${7:-}"
machine="$(uname -m)"
case "$platform:$machine" in
  linux/amd64:x86_64|linux/amd64:amd64|linux/arm64:aarch64|linux/arm64:arm64) ;;
  *) exit 13 ;;
esac
# The caller may select the scratch filesystem used for image-layer conversion.
if [ -n "$8" ]; then
  [[ "$8" = /* ]] && [ -d "$8" ] || exit 14
  compute_home="$(mktemp -d "$8/inferencex-collectivex-home.XXXXXX")"
else
  compute_home="$(mktemp -d /tmp/inferencex-collectivex-home.XXXXXX)"
fi
trap 'rm -rf -- "$compute_home"' EXIT
export HOME="$compute_home" XDG_CACHE_HOME="$compute_home/.cache"
export ENROOT_TEMP_PATH="$compute_home/enroot-tmp"
export ENROOT_CACHE_PATH="$compute_home/enroot-cache"
export ENROOT_DATA_PATH="$compute_home/enroot-data"
export ENROOT_RUNTIME_PATH="$compute_home/enroot-run"
mkdir -p "$(dirname "$sq")" "$(dirname "$lock")" \
  "$ENROOT_TEMP_PATH" "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH" "$ENROOT_RUNTIME_PATH"
exec 9>"$lock"
# Shared storage serializes the import; node-local storage imports in parallel.
flock 9
# Same reuse rules as collx_squash_verdict, evaluated per node (storage may be
# node-local): a refresh discards only files staged before this launcher started
# (a sibling node's fresh import stays); a resolved digest that differs from the
# sidecar stamp means the tag moved; an unresolved digest reuses what is staged.
reuse=yes
if [ ! -e "$sq" ]; then
  reuse=absent
elif [ -n "$refresh_epoch" ] \
    && [ "$(stat -c %Y "$sq" 2>/dev/null || echo 0)" -lt "$refresh_epoch" ]; then
  reuse=refresh-requested
else
  stamp=""
  [ ! -f "$sq.digest" ] || IFS= read -r stamp < "$sq.digest" || stamp=""
  if [ -n "$digest" ] && [ -n "$stamp" ] && [ "$stamp" != "$digest" ]; then
    reuse=digest-moved
  fi
fi
[ "$reuse" != yes ] || unsquashfs -l "$sq" >/dev/null 2>&1 || reuse=invalid
if [ "$reuse" = yes ]; then
  echo 'container squash ready (reusing staged import)'
else
  echo "importing configured container image ($reuse)"
  enroot version || true
  df -hT "$ENROOT_TEMP_PATH" "$(dirname "$sq")" || true
  findmnt -T "$ENROOT_TEMP_PATH" -o TARGET,SOURCE,FSTYPE,OPTIONS || true
  for scratch in /tmp /var/tmp /dev/shm /scratch /local; do
    [ ! -d "$scratch" ] || df -hT "$scratch" || true
  done
  converter="$(command -v enroot-aufs2ovlfs || true)"
  [ -z "$converter" ] || getcap "$converter" || true
  rm -f -- "$sq" "$sq.digest"
  enroot import -o "$sq" "docker://$image" </dev/null
  unsquashfs -l "$sq" >/dev/null 2>&1
  chmod a+r "$sq" 2>/dev/null || true
  printf '%s\n' "$digest" > "$sq.digest" || true
  # Retired per-run names of this image never get touched again; the age gate
  # spares a file a concurrent old-generation run may still be reading.
  find "$(dirname "$sq")" -maxdepth 1 -type f -name "*_${sanitized}.sqsh" \
    ! -name "${sq##*/}" -mmin +2880 -delete 2>/dev/null || true
fi
BASH
    [ "$rc" = 0 ] && { printf '%s' "$sq"; return 0; }
    # 13 is the remote block's architecture guard; a platform mismatch never improves on retry.
    if [ "$rc" = 13 ]; then
      collx_log "ERROR: container image platform does not match the allocated architecture"
      collx_log_tail "$log"
      return 1
    fi
    if [ "$attempt" -lt "$max_attempts" ]; then
      collx_log "container import attempt $attempt/$max_attempts failed (rc=$rc); retrying"
      collx_log_tail "$log"
      sleep "$((attempt * 30))"
    fi
  done
  collx_log "ERROR: container import failed after $max_attempts attempts"
  collx_log_tail "$log"
  return 1
}
