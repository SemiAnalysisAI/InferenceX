# shellcheck shell=bash
# CollectiveX shared launcher helpers (sourced, not executed). Logging goes to stderr so
# functions can `echo` a single result on stdout.

unset COLLECTIVEX_OPERATOR_CONFIG_LOADED
COLLX_RUNTIME_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

collx_log() { printf '[collectivex] %s\n' "$*" >&2; }
collx_die() { printf '[collectivex] FATAL: %s\n' "$*" >&2; exit 1; }

collx_log_tail() {
  local log_path="$1"
  if [ -s "$log_path" ]; then
    collx_log "--- command log tail ---"
    tail -n 100 -- "$log_path" >&2 || true
    collx_log "--- end command log tail ---"
  fi
}

collx_launcher_prologue() {
  JOB_ID=""
  # One stable timestamp per launcher invocation: the refresh path discards only
  # squashes staged before this moment, across salloc retries and both nodes.
  : "${COLLX_LAUNCH_EPOCH:=$(date +%s)}"
  export COLLX_LAUNCH_EPOCH
  collx_install_launcher_fail_safe
  [ -n "${COLLX_SHARD_FILE:-}" ] || collx_die "COLLX_SHARD_FILE is required"
  collx_load_operator_config
  collx_prepare_stage_dir "$1"
}

collx_execute_and_collect() {
  local mount_src="$1" repo_root="$2" run_rc=0 collect_rc=0
  collx_run_shard || run_rc=$?
  collx_collect_results "$mount_src" "$repo_root" || collect_rc=$?
  FINAL_RC="$run_rc"
  [ "$FINAL_RC" != 0 ] || FINAL_RC="$collect_rc"
}

collx_job_root_is_safe() {
  local root="$1"
  if [[ "$root" =~ ^/tmp/inferencex-collectivex-[0-9]+-[0-9]+-[A-Za-z0-9._-]+$ ]]; then
    :
  elif [[ "$root" =~ ^/tmp/inferencex-collectivex-parent-([0-9]+)-([0-9]+)-([A-Za-z0-9._-]+)/inferencex-collectivex-([0-9]+)-([0-9]+)-([A-Za-z0-9._-]+)$ ]]; then
    [ "${BASH_REMATCH[1]}" = "${BASH_REMATCH[4]}" ] \
      && [ "${BASH_REMATCH[2]}" = "${BASH_REMATCH[5]}" ] \
      && [ "${BASH_REMATCH[3]}" = "${BASH_REMATCH[6]}" ] || return 1
  else
    return 1
  fi
  [ -d "$root" ] && [ ! -L "$root" ] \
    && [ "$(stat -c '%u:%a' "$root" 2>/dev/null)" = "$(id -u):700" ]
}

# Operator JSON values are never sourced or evaluated as shell.
collx_load_operator_config() {
  [ -n "${COLLECTIVEX_OPERATOR_CONFIG_LOADED:-}" ] \
    && [ "$COLLECTIVEX_OPERATOR_CONFIG_LOADED" = "$$" ] && return 0
  local config_path parsed_path key value
  unset COLLX_IMAGE COLLX_IMAGE_PLATFORM
  unset COLLX_PARTITION COLLX_ACCOUNT COLLX_QOS COLLX_SQUASH_DIR COLLX_STAGE_DIR COLLX_ENROOT_CACHE_PATH
  unset ENROOT_CACHE_PATH
  unset COLLX_EXCLUDE_NODES COLLX_NODELIST COLLX_LOCK_DIR COLLX_MASTER_PORT
  unset COLLX_SOCKET_IFNAME COLLX_RDMA_DEVICES COLLX_IB_GID_INDEX COLLX_RDMA_SERVICE_LEVEL
  unset COLLX_RDMA_TRAFFIC_CLASS COLLX_RAIL_ISOLATED COLLX_SINGLE_NODE_RDMA_DEVICES COLLX_RDMA_FABRIC
  unset MASTER_ADDR MASTER_PORT RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE
  config_path="${COLLECTIVEX_OPERATOR_CONFIG:-${XDG_CONFIG_HOME:-${HOME}/.config}/inferencex/collectivex.json}"
  if [ ! -e "$config_path" ]; then
    # No operator document: a host-utility step (no SKU) is a no-op; a known SKU uses the tracked
    # platform_config.json baseline ("-" sentinel).
    if [ -z "${COLLX_RUNNER:-${COLLX_SHARD_SKU:-}}" ]; then
      COLLECTIVEX_OPERATOR_CONFIG_LOADED="$$"
      return 0
    fi
    config_path="-"
  fi
  umask 077
  parsed_path="$(mktemp /tmp/inferencex-collectivex-parsed.XXXXXX)" \
    || collx_die "cannot parse runner configuration"
  if ! python3 "$COLLX_RUNTIME_DIR/config.py" operator-config "$config_path" \
      "${COLLX_RUNNER:-${COLLX_SHARD_SKU:-}}" \
      > "$parsed_path"
  then
    rm -f -- "$parsed_path"
    unset COLLECTIVEX_OPERATOR_CONFIG
    collx_die "runner-local configuration failed"
  fi
  while IFS= read -r -d '' key && IFS= read -r -d '' value; do
    printf -v "$key" '%s' "$value"
    export "${key?}"
  done < "$parsed_path"
  rm -f -- "$parsed_path"
  unset COLLECTIVEX_OPERATOR_CONFIG
  COLLECTIVEX_OPERATOR_CONFIG_LOADED="$$"
}

# Callers parse these logs for markers (salloc grant, per-node network selectors), so they are
# a data channel, not just failure display. They persist after the run for postmortem.
collx_private_log_path() {
  local path="${COLLX_JOB_ROOT:-/tmp/inferencex-collectivex-$(id -u)}/logs/$1.log"
  mkdir -p "${path%/*}" || collx_die "cannot create log directory"
  : > "$path" || collx_die "cannot create runtime log"
  printf '%s' "$path"
}

# Host-side utility steps never receive the complete Actions or runner environment.
collx_host_exports() {
  printf '%s' 'HOME,PATH,USER,XDG_CACHE_HOME,ENROOT_CACHE_PATH'
}

collx_require_vars() {
  local name
  local -a missing=()
  for name in "$@"; do
    [ -n "${!name:-}" ] || missing+=("$name")
  done
  [ "${#missing[@]}" -eq 0 ] || collx_die \
    "missing platform or runner configuration: ${missing[*]}"
}

# Public launcher entry point; modules share these helpers and COLLX_RUNTIME_DIR.
source "$COLLX_RUNTIME_DIR/network.sh" || return $?
source "$COLLX_RUNTIME_DIR/slurm.sh" || return $?
source "$COLLX_RUNTIME_DIR/images.sh" || return $?
source "$COLLX_RUNTIME_DIR/sources.sh" || return $?
source "$COLLX_RUNTIME_DIR/staging.sh" || return $?
source "$COLLX_RUNTIME_DIR/execution.sh" || return $?
