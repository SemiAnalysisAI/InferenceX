# shellcheck shell=bash
# Isolated compute-visible staging and result collection. Sourced by common.sh.

collx_prepare_implicit_stage_base() {
  python3 "$COLLX_RUNTIME_DIR/stage.py" implicit-stage-base "${1:-}" "${2:-}"
}

collx_prepare_runner_shared_stage_base() {
  local runner_temp="${RUNNER_TEMP:-}" runner_root
  case "$runner_temp" in
    /*/_work/_temp) runner_root="${runner_temp%/_work/_temp}" ;;
    *) collx_die "canonical AMD execution requires a standard shared runner temp" ;;
  esac
  [ -n "$runner_root" ] && [ "$runner_root" != "$runner_temp" ] \
    || collx_die "canonical AMD execution requires a shared runner root"
  collx_prepare_implicit_stage_base "$runner_root"
}

collx_prepare_stage_dir() {
  local runner="$1"
  [ "${COLLECTIVEX_CANONICAL_GHA:-0}" = 1 ] || return 0
  [ -n "${COLLX_SQUASH_DIR:-}" ] \
    || collx_die "canonical CollectiveX execution requires shared container storage"
  case "$runner" in b300|gb300) COLLX_STAGE_DIR="" ;; esac
  if [ -z "${COLLX_STAGE_DIR:-}" ]; then
    case "$runner" in
      h100-dgxc)
        COLLX_STAGE_DIR="$(collx_prepare_implicit_stage_base "${COLLX_SQUASH_DIR%/*}")" \
          || collx_die "canonical CollectiveX execution cannot create an isolated shared stage directory"
        ;;
      b300|gb300)
        COLLX_STAGE_DIR="$(collx_prepare_implicit_stage_base "" \
          "${COLLECTIVEX_EXECUTION_ID:-${GITHUB_RUN_ID:-}}")" \
          || collx_die "canonical CollectiveX execution cannot create an isolated stage directory"
        ;;
      h200-dgxc)
        COLLX_STAGE_DIR="$(collx_prepare_implicit_stage_base)" \
          || collx_die "canonical CollectiveX execution cannot create an isolated stage directory"
        ;;
      b200-nscale)
        # The passwd home is not compute-visible; anchor at the squash dir's parent.
        COLLX_STAGE_DIR="$(collx_prepare_implicit_stage_base "${COLLX_SQUASH_DIR%/*}")" \
          || collx_die "canonical CollectiveX execution cannot create an isolated stage directory"
        ;;
      mi300x|mi325x|mi355x)
        COLLX_STAGE_DIR="$(collx_prepare_runner_shared_stage_base)" \
          || collx_die "canonical AMD execution cannot create an isolated shared stage directory"
        ;;
      *) collx_die "canonical CollectiveX execution requires a configured shared stage directory" ;;
    esac
  elif [ "$runner" = mi300x ]; then
    COLLX_STAGE_DIR="$(python3 "$COLLX_RUNTIME_DIR/stage.py" resolve-directory \
      "$COLLX_STAGE_DIR")" \
      || collx_die "canonical MI300X execution cannot resolve the shared stage directory"
  fi
  export COLLX_STAGE_DIR
}

# Resolved before any copy starts so the EXIT trap can remove an interrupted partial stage.
collx_stage_path() {
  local repo_root="$1" stage_base="${2:-}" tag stage_path
  tag="${COLLECTIVEX_EXECUTION_ID:-${GITHUB_RUN_ID:-manual-$$}}"
  [[ "$tag" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
    || collx_die "invalid staging execution identity"
  if [ -z "$stage_base" ] || [ "$stage_base" = "$repo_root" ]; then
    [ -n "${COLLX_SQUASH_DIR:-}" ] \
      || collx_die "CollectiveX staging requires COLLX_STAGE_DIR or COLLX_SQUASH_DIR"
    stage_base="$COLLX_SQUASH_DIR"
    stage_path="${stage_base%/}/.collectivex-stage-$tag"
  else
    stage_path="${stage_base%/}/job_$tag"
  fi
  python3 "$COLLX_RUNTIME_DIR/stage.py" validate-stage-path "$repo_root" "$stage_base" \
    "$stage_path" "${COLLX_JOB_ROOT:-}" "${GITHUB_WORKSPACE:-}"
}

collx_stage_repo() {
  local repo_root="$1" stage_dir="$2" log
  python3 "$COLLX_RUNTIME_DIR/stage.py" create-stage "$stage_dir" \
    || collx_die "cannot create the configured stage directory"
  collx_log "staging CollectiveX on compute-visible storage"
  log="$(collx_private_log_path repository-stage)"
  if ! python3 "$COLLX_RUNTIME_DIR/stage.py" copy-repository \
      "$repo_root/experimental/CollectiveX" \
      "$stage_dir/experimental/CollectiveX" > "$log" 2>&1; then
    rm -rf -- "$stage_dir" >/dev/null 2>&1 \
      || collx_log "ERROR: cannot remove the incomplete execution stage"
    collx_log "ERROR: repository staging failed"
    collx_log_tail "$log"
    return 1
  fi
}

# The workflow's upload-artifact reads the checkout, not the stage dir, so staged result JSONs
# are copied back to the checkout's results/.
collx_collect_results() {
  local mount_src="$1" repo_root="$2" dst log
  local -a files
  [ "$mount_src" = "$repo_root" ] && return 0
  log="$(collx_private_log_path "artifact-collection-$$-${RANDOM}")"
  dst="$repo_root/experimental/CollectiveX/results"
  mkdir -p "$dst" 2>> "$log" \
    || { collx_log "ERROR: cannot create checkout result directory"; return 1; }
  shopt -s nullglob
  files=("$mount_src/experimental/CollectiveX/results/"*.json)
  shopt -u nullglob
  [ "${#files[@]}" -gt 0 ] || { collx_log "ERROR: staged run produced no result JSON"; return 1; }
  cp -- "${files[@]}" "$dst/" >> "$log" 2>&1 \
    || { collx_log "ERROR: staged result collection failed"; return 1; }
  collx_log "collected staged results for artifact validation"
}

collx_cleanup_stage() {
  local mount_src="$1" repo_root="$2"
  [ "$mount_src" != "$repo_root" ] || return 0
  if ! python3 "$COLLX_RUNTIME_DIR/stage.py" validate-cleanup "$mount_src"; then
    collx_log "ERROR: refusing to remove an invalid stage directory"
    return 1
  fi
  rm -rf -- "$mount_src" >/dev/null 2>&1 || {
    collx_log "ERROR: cannot remove generated stage directory"
    return 1
  }
  collx_log "removed generated per-execution stage directory"
}
