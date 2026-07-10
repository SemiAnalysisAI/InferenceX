# shellcheck shell=bash
# CollectiveX — shared launcher helpers (sourced, not executed).
#
# Cluster-generic scaffolding only (Slurm/container/build/staging); no
# model-serving. Logging goes to stderr so functions can `echo` a single
# result on stdout.

COLLX_DEEPEP_V2_COMMIT="fa8a9b16898204afd347c663b89e65ef87dc6ce6" # pragma: allowlist secret
COLLX_DEEPEP_V2_TREE="29809e75c5874e6609dac4804e7b651d5226959f" # pragma: allowlist secret
COLLX_DEEPEP_V2_FMT_COMMIT="a4c7e17133ee9cb6a2f45545f6e974dd3c393efa" # pragma: allowlist secret
# Consumed by prepare_backend.sh after this helper is sourced.
# shellcheck disable=SC2034
COLLX_DEEPEP_V2_NCCL_CHECK_COMMIT="93d0564188f7a0a6288c6e316484861b0efa042e" # pragma: allowlist secret
COLLX_DEEPEP_HYBRID_COMMIT="e0a5b1d9848ab3e7b4a67842bf06f067bfac67f8" # pragma: allowlist secret
COLLX_DEEPEP_HYBRID_TREE="d77aeab7f1bb52b615666fe178d26ced41fae08e" # pragma: allowlist secret
COLLX_DEEPEP_HYBRID_NCCL_COMMIT="1e0c869c39bb33f1034cb9920bd2a8a8406f04a3" # pragma: allowlist secret
unset COLLECTIVEX_OPERATOR_CONFIG_LOADED COLLECTIVEX_EPHEMERAL_CONFIG_PATH
COLLX_RUNTIME_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

collx_log() { printf '[collectivex] %s\n' "$*" >&2; }
collx_die() { printf '[collectivex] FATAL: %s\n' "$*" >&2; exit 1; }

# Keep a stable stage label while leaving diagnosis to the captured tool output.
collx_set_failure_stage() {
  local stage="$1"
  case "$stage" in
    setup|repository-stage|scheduler-allocation|container-import) ;;
    container-hash|container-launch|backend-setup|execution|artifact-collection) ;;
    *) collx_die "invalid launcher failure stage" ;;
  esac
  export COLLX_FAILSAFE_MODE="$stage"
}

collx_fail_stage() {
  local stage="$1" log_path="${2:-}" tail_lines="${COLLX_LOG_TAIL_LINES:-100}"
  collx_set_failure_stage "$stage"
  collx_log "ERROR: failure-stage=$stage"
  [[ "$tail_lines" =~ ^[1-9][0-9]{0,2}$ ]] || tail_lines=100
  if [ -n "$log_path" ] && [ -s "$log_path" ]; then
    collx_log "--- $stage log tail (last $tail_lines lines) ---"
    tail -n "$tail_lines" -- "$log_path" >&2 || true
    collx_log "--- end $stage log tail ---"
  fi
  return 1
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

# Runner-local deployment settings are strict JSON kept outside the checkout.
# Only the selected runner's allowlisted values are exported; the document is
# never sourced or evaluated as shell.
collx_load_operator_config() {
  [ -n "${COLLECTIVEX_OPERATOR_CONFIG_LOADED:-}" ] \
    && [ "$COLLECTIVEX_OPERATOR_CONFIG_LOADED" = "$$" ] && return 0
  local config_path generated=0 parsed_path config_log key value validation_code
  unset COLLX_PARTITION COLLX_ACCOUNT COLLX_QOS COLLX_SQUASH_DIR COLLX_STAGE_DIR COLLX_ENROOT_CACHE_PATH
  unset ENROOT_CACHE_PATH
  unset COLLX_EXCLUDE_NODES COLLX_NODELIST COLLX_LOCK_DIR COLLX_MASTER_PORT
  unset COLLX_SOCKET_IFNAME COLLX_RDMA_DEVICES COLLX_IB_GID_INDEX COLLX_RDMA_SERVICE_LEVEL
  unset COLLX_RDMA_TRAFFIC_CLASS
  unset MASTER_ADDR MASTER_PORT RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE
  config_path="${COLLECTIVEX_OPERATOR_CONFIG:-${XDG_CONFIG_HOME:-${HOME}/.config}/inferencex/collectivex.json}"
  if [ -n "${COLLECTIVEX_OPERATOR_CONFIG_CONTENT:-}" ]; then
    umask 077
    if collx_job_root_is_safe "${COLLX_JOB_ROOT:-}"; then
      config_path="$COLLX_JOB_ROOT/operator-config.json"
      (set -C; : > "$config_path") 2>/dev/null \
        || collx_die "cannot create ephemeral runner configuration"
    else
      config_path="$(mktemp /tmp/inferencex-collectivex-config.XXXXXX)" \
        || collx_die "cannot create ephemeral runner configuration"
    fi
    COLLECTIVEX_EPHEMERAL_CONFIG_PATH="$config_path"
    generated=1
    if ! printf '%s' "$COLLECTIVEX_OPERATOR_CONFIG_CONTENT" > "$config_path"; then
      unset COLLECTIVEX_OPERATOR_CONFIG_CONTENT
      rm -f -- "$config_path"
      unset COLLECTIVEX_EPHEMERAL_CONFIG_PATH
      collx_die "cannot materialize runner configuration"
    fi
  elif [ "${COLLECTIVEX_OPERATOR_CONFIG_REQUIRED:-0}" = 1 ]; then
    unset COLLECTIVEX_OPERATOR_CONFIG_CONTENT
    collx_die "runner configuration is unavailable"
  fi
  unset COLLECTIVEX_OPERATOR_CONFIG_CONTENT COLLECTIVEX_OPERATOR_CONFIG_REQUIRED
  if [ ! -e "$config_path" ]; then
    [ "${COLLECTIVEX_CANONICAL_GHA:-0}" != 1 ] \
      || collx_die "runner configuration is unavailable"
    COLLECTIVEX_OPERATOR_CONFIG_LOADED="$$"
    return 0
  fi
  umask 077
  parsed_path="$(mktemp /tmp/inferencex-collectivex-parsed.XXXXXX)" || {
    [ "$generated" = 0 ] || rm -f -- "$config_path"
    collx_die "cannot parse runner configuration"
  }
  config_log="$(collx_private_log_path operator-config)"
  if ! python3 "$COLLX_RUNTIME_DIR/config.py" operator-config "$config_path" \
      "${COLLX_RUNNER:-${COLLX_SHARD_SKU:-${COLLX_PUBLIC_RUNNER:-}}}" \
      > "$parsed_path" 2> "$config_log"
  then
    validation_code="$(head -n 1 "$config_log" 2>/dev/null || true)"
    rm -f -- "$parsed_path"
    [ "$generated" = 0 ] || rm -f -- "$config_path"
    unset COLLECTIVEX_EPHEMERAL_CONFIG_PATH
    unset COLLECTIVEX_OPERATOR_CONFIG COLLECTIVEX_OPERATOR_CONFIG_EPHEMERAL
    [[ "$validation_code" =~ ^validation-(line-[0-9]+|missing-required-[a-z0-9_-]+)$ ]] \
      || validation_code="validation-unknown"
    collx_die "runner-local configuration failed ($validation_code)"
  fi
  while IFS= read -r -d '' key && IFS= read -r -d '' value; do
    printf -v "$key" '%s' "$value"
    export "${key?}"
  done < "$parsed_path"
  rm -f -- "$parsed_path"
  if [ "$generated" = 1 ] || [ "${COLLECTIVEX_OPERATOR_CONFIG_EPHEMERAL:-0}" = 1 ]; then
    rm -f -- "$config_path" || collx_die "cannot remove ephemeral runner configuration"
  fi
  unset COLLECTIVEX_EPHEMERAL_CONFIG_PATH
  unset COLLECTIVEX_OPERATOR_CONFIG COLLECTIVEX_OPERATOR_CONFIG_EPHEMERAL
  COLLECTIVEX_OPERATOR_CONFIG_LOADED="$$"
}

collx_private_log_path() {
  local label="$1" root="${COLLX_JOB_ROOT:-/tmp/inferencex-collectivex-$(id -u)}" path
  [[ "$label" =~ ^[A-Za-z0-9._-]+$ ]] || collx_die "invalid private log label"
  path="$root/logs/$label.log"
  mkdir -m 700 -p "${path%/*}" || collx_die "cannot create private log directory"
  (umask 077; : > "$path") || collx_die "cannot create private runtime log"
  printf '%s' "$path"
}

collx_cleanup_private_logs() {
  [ "$1" != 0 ] || rm -rf -- "${COLLX_JOB_ROOT:-/tmp/inferencex-collectivex-$(id -u)}/logs"
}

# Host-side utility steps need only the basic login paths. They never receive
# the complete Actions or runner environment.
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
    "missing runner-local configuration: ${missing[*]} (set them in COLLECTIVEX_OPERATOR_CONFIG)"
}

collx_bool_enabled() {
  local normalized
  normalized="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "$normalized" in
    1|true|yes) return 0 ;;
    *) return 1 ;;
  esac
}

collx_nccl_hca_device_name() {
  local selector="${1#=}"
  printf '%s' "${selector%%:*}"
}

collx_export_gid_index_for_link_layer() {
  local link_layer="$1" scaleout="$2"
  unset NVSHMEM_IB_GID_INDEX NCCL_IB_GID_INDEX
  [ -n "${COLLX_IB_GID_INDEX:-}" ] || return 0
  case "$link_layer" in
    roce)
      export NVSHMEM_IB_GID_INDEX="$COLLX_IB_GID_INDEX"
      if [ "$scaleout" = 1 ]; then
        export NCCL_IB_GID_INDEX="$COLLX_IB_GID_INDEX"
      fi
      ;;
    infiniband) ;;
    *) collx_die "unsupported RDMA link layer" ;;
  esac
}

# Convert private, runner-local network selectors into the public library
# variables needed inside the container. Values are interface/HCA identifiers,
# never addresses; the rendezvous hostname is derived from the allocation.
collx_apply_network_profile() {
  local nodes="$1" transport="$2"
  local selector rdma_name rdma_names="" ep_nic=""
  local scaleout=0
  local -a selectors
  [[ "$nodes" =~ ^[1-9][0-9]*$ ]] || collx_die "invalid network placement"
  unset NCCL_NET NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME NCCL_IB_HCA
  unset NCCL_IB_GID_INDEX NCCL_IB_SL
  unset NVSHMEM_ENABLE_NIC_PE_MAPPING
  unset NVSHMEM_HCA_LIST NVSHMEM_IB_GID_INDEX NVSHMEM_IB_SL
  unset NVSHMEM_IB_ENABLE_IBGDA NVSHMEM_IBGDA_NIC_HANDLER
  unset EP_NIC_NAME EP_OVERRIDE_RDMA_SL
  unset MORI_RDMA_DEVICES
  unset MORI_RDMA_TC MORI_IO_TC MORI_RDMA_SL MORI_IO_SL
  if [ "$nodes" -gt 1 ] && [ "$transport" != mnnvl ]; then
    scaleout=1
  fi
  [ "$scaleout" = 1 ] || return 0
  [ -n "${COLLX_RDMA_DEVICES:-}" ] \
    || collx_die "RDMA execution requires a private device selector"
  if [ "$scaleout" = 1 ] && [ -n "${COLLX_SOCKET_IFNAME:-}" ]; then
    [[ "$COLLX_SOCKET_IFNAME" =~ ^[A-Za-z][A-Za-z0-9_.-]{0,31}(,[A-Za-z][A-Za-z0-9_.-]{0,31})*$ ]] \
      || collx_die "invalid private socket interface selector"
    export NCCL_SOCKET_IFNAME="$COLLX_SOCKET_IFNAME" GLOO_SOCKET_IFNAME="$COLLX_SOCKET_IFNAME"
  fi
  if [ -n "${COLLX_RDMA_DEVICES:-}" ]; then
    [[ "$COLLX_RDMA_DEVICES" =~ ^[A-Za-z][A-Za-z0-9_.-]{0,31}(:[1-9][0-9]*)?(,[A-Za-z][A-Za-z0-9_.-]{0,31}(:[1-9][0-9]*)?)*$ ]] \
      || collx_die "invalid private RDMA device selector"
    IFS=, read -r -a selectors <<< "$COLLX_RDMA_DEVICES"
    for selector in "${selectors[@]}"; do
      rdma_name="${selector%%:*}"
      rdma_names="${rdma_names}${rdma_names:+,}${rdma_name}"
      [ -n "$ep_nic" ] || ep_nic="$rdma_name"
    done
    export NVSHMEM_HCA_LIST="$COLLX_RDMA_DEVICES"
    export NVSHMEM_ENABLE_NIC_PE_MAPPING=1
    if [ "$scaleout" = 1 ]; then
      if [ "${COLLX_SHARD_SKU:-}" = mi300x ] || [ "${COLLX_SHARD_SKU:-}" = mi325x ] \
          || [ "${COLLX_SHARD_SKU:-}" = mi355x ]; then
        unset NCCL_NET
      else
        export NCCL_NET=IB
      fi
      export NCCL_IB_HCA="=$COLLX_RDMA_DEVICES"
      export MORI_RDMA_DEVICES="$rdma_names" EP_NIC_NAME="$ep_nic"
    fi
  fi
  if [ -n "${COLLX_IB_GID_INDEX:-}" ]; then
    [[ "$COLLX_IB_GID_INDEX" =~ ^[0-9]+$ ]] && [ "$COLLX_IB_GID_INDEX" -le 255 ] \
      || collx_die "invalid private IB GID index"
  fi
  if [ -n "${COLLX_RDMA_SERVICE_LEVEL:-}" ]; then
    [[ "$COLLX_RDMA_SERVICE_LEVEL" =~ ^[0-9]+$ ]] && [ "$COLLX_RDMA_SERVICE_LEVEL" -le 15 ] \
      || collx_die "invalid private RDMA service level"
    export NVSHMEM_IB_SL="$COLLX_RDMA_SERVICE_LEVEL"
    if [ "$scaleout" = 1 ]; then
      export NCCL_IB_SL="$COLLX_RDMA_SERVICE_LEVEL"
      export EP_OVERRIDE_RDMA_SL="$COLLX_RDMA_SERVICE_LEVEL"
      export MORI_RDMA_SL="$COLLX_RDMA_SERVICE_LEVEL" MORI_IO_SL="$COLLX_RDMA_SERVICE_LEVEL"
    fi
  fi
  if [ -n "${COLLX_RDMA_TRAFFIC_CLASS:-}" ]; then
    [[ "$COLLX_RDMA_TRAFFIC_CLASS" =~ ^[0-9]+$ ]] && [ "$COLLX_RDMA_TRAFFIC_CLASS" -le 255 ] \
      || collx_die "invalid private RDMA traffic class"
    [ "$scaleout" = 1 ] \
      && export MORI_RDMA_TC="$COLLX_RDMA_TRAFFIC_CLASS" MORI_IO_TC="$COLLX_RDMA_TRAFFIC_CLASS"
  fi
  local nic_handler=gpu
  export NVSHMEM_IB_ENABLE_IBGDA=1 NVSHMEM_IBGDA_NIC_HANDLER="$nic_handler"
  if [ -n "${COLLX_RDMA_LINK_LAYER:-}" ]; then
    case "$COLLX_RDMA_LINK_LAYER" in
      roce|infiniband) ;;
      *) collx_die "invalid validated RDMA link layer" ;;
    esac
    collx_export_gid_index_for_link_layer "$COLLX_RDMA_LINK_LAYER" "$scaleout"
  fi
}

# Slurm may remove NCCL's leading exact-match marker while propagating an
# inherited environment. Reconstruct it from the validated private selector at
# the container boundary instead of accepting a prefix-matched HCA list.
collx_restore_exact_hca_selector() {
  if [ "${COLLX_NODES:-1}" -le 1 ] || [ "${COLLX_TRANSPORT:-}" = mnnvl ]; then
    return 0
  fi
  [ -n "${COLLX_RDMA_DEVICES:-}" ] \
    || { collx_log "ERROR: scale-out RDMA selector is unavailable"; return 1; }
  [[ "$COLLX_RDMA_DEVICES" =~ ^[A-Za-z][A-Za-z0-9_.-]{0,31}(:[1-9][0-9]*)?(,[A-Za-z][A-Za-z0-9_.-]{0,31}(:[1-9][0-9]*)?)*$ ]] \
    || { collx_log "ERROR: invalid scale-out RDMA selector"; return 1; }
  export NCCL_IB_HCA="=$COLLX_RDMA_DEVICES"
}

collx_default_route_interface() {
  python3 "$COLLX_RUNTIME_DIR/probe.py" default-route-interface
}

# Prove that the operator-pinned scale-out fabric exists on every allocated
# node before image import or backend initialization. Selector values and node
# diagnostics stay in the runner-private log.
collx_validate_network_profile_on_job() {
  local job_id="$1" nodes="$2" transport="$3" report_failure="${4:-1}"
  local log_label=network-profile log rc=0 scaleout=0 marker_count link_layer
  if [ "$nodes" -gt 1 ] && [ "$transport" != mnnvl ]; then
    scaleout=1
  fi
  [ "$scaleout" = 1 ] || return 0
  [[ "$job_id" =~ ^[1-9][0-9]*$ && "$nodes" =~ ^[1-9][0-9]*$ ]] \
    || return 1
  [ -n "${COLLX_RDMA_DEVICES:-}" ] || return 1
  case "${COLLX_NETWORK_VALIDATION_ATTEMPT:-1}" in
    1) ;;
    2|3) log_label+="-a${COLLX_NETWORK_VALIDATION_ATTEMPT}" ;;
    *) return 1 ;;
  esac
  log="$(collx_private_log_path "$log_label")" || return 1
  COLLX_NETWORK_PROFILE_LOG="$log"
  srun --jobid="$job_id" --nodes="$nodes" --ntasks="$nodes" --ntasks-per-node=1 \
    --chdir=/tmp --input=all --export="$(collx_host_exports)" \
    python3 /dev/stdin network-profile "${COLLX_SOCKET_IFNAME:-}" \
      "$COLLX_RDMA_DEVICES" "${COLLX_IB_GID_INDEX:-}" \
    < "$COLLX_RUNTIME_DIR/probe.py" > "$log" 2>&1 || rc=$?
  if [ "$rc" != 0 ]; then
    marker="$(grep -aoE '(socket-interface|rdma-(device|port))-[0-9]+=(missing|down|inactive|default-route-missing|gid-missing|gid-empty|link-layer-missing|link-layer-invalid|link-layer-mixed)' "$log" \
      | tail -n 1 || true)"
    [ -z "$marker" ] || collx_log "ERROR: network-profile-$marker"
    [ "$report_failure" = 0 ] || collx_fail_stage setup "$log" || true
    return "$rc"
  fi
  socket_ifname="$(
    sed -nE 's/^\[collectivex-private\] socket-interface-selected=([A-Za-z][A-Za-z0-9_.-]{0,31})$/\1/p' "$log" \
      | sort -u
  )"
  marker_count="$(grep -Ec '^\[collectivex-private\] socket-interface-selected=' "$log")"
  socket_unique_count="$(printf '%s\n' "$socket_ifname" | sed '/^$/d' | wc -l | tr -d ' ')"
  if [ "$socket_unique_count" -lt 1 ] || [ "$marker_count" != "$nodes" ]; then
    collx_log "ERROR: network-profile-socket-markers=$marker_count/$nodes unique=$socket_unique_count"
    return 1
  fi
  if [ "$socket_unique_count" = 1 ]; then
    export COLLX_SOCKET_IFNAME="$socket_ifname"
  else
    unset COLLX_SOCKET_IFNAME
  fi
  link_layer="$(
    sed -nE 's/^\[collectivex-private\] rdma-link-layer=(roce|infiniband)$/\1/p' "$log" \
      | sort -u
  )"
  marker_count="$(grep -Ec '^\[collectivex-private\] rdma-link-layer=(roce|infiniband)$' "$log")"
  case "$marker_count:$link_layer" in
    "$nodes":roce|"$nodes":infiniband) ;;
    *) [ "$report_failure" = 0 ] || collx_fail_stage setup "$log" || true; return 1 ;;
  esac
  export COLLX_RDMA_LINK_LAYER="$link_layer"
  collx_export_gid_index_for_link_layer "$link_layer" "$scaleout"
}

collx_allocation_nodes_csv() {
  local job_id="$1" nodelist node output=""
  [[ "$job_id" =~ ^[1-9][0-9]*$ ]] || return 1
  nodelist="$(squeue -h -j "$job_id" -o %N 2>/dev/null)" || return 1
  [[ "$nodelist" =~ ^[][A-Za-z0-9._,-]+$ ]] || return 1
  while IFS= read -r node; do
    [ -n "$node" ] || continue
    [[ "$node" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || return 1
    [ -z "$output" ] || output+=,
    output+="$node"
  done < <(scontrol show hostnames "$nodelist" 2>/dev/null)
  [ -n "$output" ] || return 1
  printf '%s' "$output"
}

collx_resolve_slurm_rendezvous() {
  local job_id="$1" master_addr master_port socket_ifname="${COLLX_SOCKET_IFNAME:-}"
  [[ "$job_id" =~ ^[1-9][0-9]*$ ]] || collx_die "invalid rendezvous allocation"
  # Query relative node zero directly so MASTER_ADDR always hosts global rank 0.
  # Prefer the address on the already validated cross-node socket interface;
  # a short hostname may resolve onto a management network that ranks cannot use.
  if [[ "$socket_ifname" =~ ^[A-Za-z][A-Za-z0-9_.-]{0,31}$ ]]; then
    master_addr="$(srun --jobid="$job_id" --nodes=1 --ntasks=1 --relative=0 \
      --chdir=/tmp --export="$(collx_host_exports)" bash -s -- "$socket_ifname" \
      2>/dev/null <<'BASH' | head -n1
set -euo pipefail
ip -o -4 address show dev "$1" scope global \
  | awk 'NR == 1 {split($4, address, "/"); print address[1]}'
BASH
)"
    [[ "$master_addr" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]] \
      || collx_die "could not resolve the allocated primary interface"
  else
    master_addr="$(srun --jobid="$job_id" --nodes=1 --ntasks=1 --relative=0 \
      --chdir=/tmp --export="$(collx_host_exports)" hostname -s 2>/dev/null | head -n1)"
    [[ "$master_addr" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
      || collx_die "could not resolve the allocated primary node"
  fi
  master_port="${COLLX_MASTER_PORT:-29551}"
  [[ "$master_port" =~ ^[1-9][0-9]*$ ]] && [ "$master_port" -le 65535 ] \
    || collx_die "invalid distributed rendezvous port"
  export MASTER_ADDR="$master_addr" MASTER_PORT="$master_port"
}

# Printed into `bash -c` ahead of the rank wrapper or backend probe. Sources the
# per-node loader/import environment persisted by collx_persist_backend_env, refusing
# a file whose directory shape, ownership, or mode differs from what that step wrote.
collx_source_backend_env() {
  cat <<'BASH'
case "${SLURM_NODEID:-}" in ""|*[!0-9]*) exit 66;; esac
env_file="/ix/experimental/CollectiveX/.collx_backend/env/node-${SLURM_NODEID}.sh"
env_root="${env_file%/*}"
[ -d "$env_root" ] && [ ! -L "$env_root" ] || exit 66
case "$(stat -c "%a" "$env_root")" in 700|[1-7]700) ;; *) exit 66;; esac
[ -f "$env_file" ] && [ -r "$env_file" ] && [ ! -L "$env_file" ] \
  && [ "$(stat -c "%u:%a" "$env_file")" = "$(stat -c "%u" "$env_root"):600" ] || exit 66
. "$env_file" || exit 66
BASH
}

# Per-node backend import probe, run inside the persistent container after the
# build step. Selects on the runtime COLLX_BENCH so every launcher shares one probe.
collx_backend_probe() {
  collx_source_backend_env
  cat <<'BASH'
case "$COLLX_BENCH" in
  deepep-v2) python3 -c "import deep_ep; assert hasattr(deep_ep, 'ElasticBuffer')" ;;
  deepep-hybrid) python3 -c "import deep_ep; assert hasattr(deep_ep, 'HybridEPBuffer')" ;;
  mori) python3 -c "import mori" ;;
  *) exit 69 ;;
esac
BASH
}

# Printed into `bash -c` for one Slurm task per GPU. Every rank derives its
# identity from Slurm rather than accepting caller-supplied rank values.
collx_slurm_rank_wrapper() {
  cat <<'BASH'
case "${SLURM_PROCID:-}:${SLURM_NTASKS:-}:${SLURM_LOCALID:-}:${SLURM_NODEID:-}" in
  *[!0-9:]*|:*|*::*|*:) exit 67 ;;
esac
[ "$SLURM_NTASKS" = "$COLLX_NGPUS" ] || exit 67
[ "$SLURM_LOCALID" -lt "$COLLX_GPUS_PER_NODE" ] || exit 67
. /ix/experimental/CollectiveX/runtime/common.sh || exit 68
if [ "${COLLX_NODES:-1}" -gt 1 ] && [ "${COLLX_TRANSPORT:-}" != mnnvl ]; then
  if [ -z "${COLLX_SOCKET_IFNAME:-}" ]; then
    COLLX_SOCKET_IFNAME="$(collx_default_route_interface)" || exit 68
    [[ "$COLLX_SOCKET_IFNAME" =~ ^[A-Za-z][A-Za-z0-9_.-]{0,31}$ ]] || exit 68
    export COLLX_SOCKET_IFNAME
  fi
  collx_apply_network_profile "$COLLX_NODES" "$COLLX_TRANSPORT" || exit 68
fi
export RANK="$SLURM_PROCID" WORLD_SIZE="$SLURM_NTASKS"
export LOCAL_RANK="$SLURM_LOCALID" LOCAL_WORLD_SIZE="$COLLX_GPUS_PER_NODE"
exec python3 bench/run_ep.py "$@"
BASH
}

# Load the case mode needed to choose the allocation/network profile. Every
# case runs in normal mode; the in-container dispatcher reapplies the profile
# for each individual case.
collx_load_network_control_mode() {
  local collx_root="$1" shard="${COLLX_SHARD_FILE:-}" path mode
  [ -n "$shard" ] || return 0
  path="$shard"
  [ -f "$path" ] || path="${collx_root%/}/$shard"
  [ -f "$path" ] || return 1
  mode="$(python3 "$COLLX_RUNTIME_DIR/config.py" network-mode "$path")" || return 1
  case "$mode" in
    normal) export COLLX_MODE="$mode" ;;
    *) return 1 ;;
  esac
}

collx_apply_timing_profile() {
  [ -n "${COLLX_TIMING:-}" ] || return 0
  local iters trials warmup extra
  IFS=: read -r iters trials warmup extra <<< "$COLLX_TIMING"
  [[ "$iters" =~ ^[1-9][0-9]*$ && "$trials" =~ ^[1-9][0-9]*$ \
    && "$warmup" =~ ^[1-9][0-9]*$ && -z "$extra" ]] \
    || collx_die "COLLX_TIMING must be positive iters:trials:warmup"
  export COLLX_ITERS="$iters" COLLX_TRIALS="$trials" COLLX_WARMUP="$warmup"
}

collx_scheduler_job_name() {
  local execution_id="${COLLECTIVEX_EXECUTION_ID:-manual-$$}" safe
  safe="$(printf '%s' "$execution_id" | tr -cs 'A-Za-z0-9_.-' '-')" || return 1
  safe="${safe#-}"; safe="${safe%-}"
  [ -n "$safe" ] || return 1
  if [ "${#safe}" -gt 120 ]; then
    safe="${safe:0:48}-${safe: -71}"
  fi
  printf 'cx-%s' "$safe"
}

# Return 0 after recovering one allocation ID, 2 after three successful empty
# observations, and 1 for every ambiguous or failed lookup. Callers inspect the
# state variables rather than the status because all missing-ID paths still fail.
collx_reconcile_salloc_jobid() {
  local job_name="$1" scheduler_user queue_output line delay attempt
  local -a ids=()
  scheduler_user="$(id -un 2>/dev/null)" || return 1
  [[ "$scheduler_user" =~ ^[A-Za-z0-9_.-]+$ \
    && "$job_name" =~ ^cx-[A-Za-z0-9_.-]{1,120}$ ]] || return 1
  for attempt in 1 2 3; do
    ids=()
    if ! queue_output="$(
      squeue -h --user="$scheduler_user" --name="$job_name" -o %A 2>/dev/null
    )"; then
      return 1
    fi
    while IFS= read -r line; do
      [[ "$line" =~ ^[[:space:]]*$ ]] && continue
      if [[ "$line" =~ ^[[:space:]]*([1-9][0-9]*)[[:space:]]*$ ]]; then
        ids+=("${BASH_REMATCH[1]}")
      else
        return 1
      fi
    done <<< "$queue_output"
    if [ "${#ids[@]}" -eq 1 ]; then
      JOB_ID="${ids[0]}"
      COLLX_ALLOCATION_UNCERTAIN=0
      return 0
    fi
    [ "${#ids[@]}" -eq 0 ] || return 1
    if [ "$attempt" -eq 3 ]; then
      COLLX_ALLOCATION_UNCERTAIN=0
      return 2
    fi
    delay=$((1 << (attempt - 1)))
    sleep "$delay" || return 1
  done
  return 1
}

collx_verify_salloc_jobid() {
  local job_id="$1" queue_output line count=0
  [[ "$job_id" =~ ^[1-9][0-9]*$ ]] || return 1
  queue_output="$(squeue -h -j "$job_id" -o %A 2>/dev/null)" || return 1
  while IFS= read -r line; do
    [[ "$line" =~ ^[[:space:]]*$ ]] && continue
    [[ "$line" =~ ^[[:space:]]*${job_id}[[:space:]]*$ ]] || return 1
    count=$((count + 1))
  done <<< "$queue_output"
  [ "$count" -eq 1 ]
}

# Allocate via salloc's stable grant message and assign JOB_ID in this shell.
# Raw scheduler output remains in the bounded private execution log.
collx_salloc_jobid() {
  local log_label=scheduler-allocation log job_id job_name argument salloc_rc=0
  case "${COLLX_SALLOC_ATTEMPT:-1}" in
    1) ;;
    2|3) log_label+="-a${COLLX_SALLOC_ATTEMPT}" ;;
    *) return 1 ;;
  esac
  if ! log="$(collx_private_log_path "$log_label")"; then
    collx_log "ERROR: failure-stage=scheduler-allocation (private log unavailable)"
    return 1
  fi
  for argument in "$@"; do
    case "$argument" in
      --job-name|--job-name=*|-J|-J*)
        collx_log "ERROR: scheduler job names are managed by CollectiveX"
        return 1
        ;;
    esac
  done
  if ! job_name="$(collx_scheduler_job_name)"; then
    collx_log "ERROR: failure-stage=scheduler-allocation (invalid job name)"
    return 1
  fi
  COLLX_ALLOCATION_UNCERTAIN=1
  # salloc has no portable --parsable option. Parse the stable grant message
  # used by the production launchers, while also accepting a bare ID from
  # site wrappers. Contain shell-function wrappers that call exit so the
  # launcher can still reconcile and cancel an allocation.
  collx_log "scheduler-request=submit"
  (salloc "$@" --job-name="$job_name" --no-shell) > "$log" 2>&1 || salloc_rc=$?
  if ! job_id="$(sed -nE \
      -e 's/^([0-9]+)(;[^[:space:]]+)?$/\1/p; t found' \
      -e 's/.*Granted job allocation ([0-9]+).*/\1/p; t found' \
      -e 'b' -e ':found' -e 'q' "$log")"; then
    collx_log "ERROR: failure-stage=scheduler-allocation (cannot parse grant)"
    collx_reconcile_salloc_jobid "$job_name" || true
    [ -z "$JOB_ID" ] || collx_record_allocation_jobid "$JOB_ID" || true
    return 1
  fi
  if [ -n "$job_id" ]; then
    [[ "$job_id" =~ ^[0-9]+$ ]] || return 1
    JOB_ID="$job_id"
    COLLX_ALLOCATION_UNCERTAIN=0
  fi
  if [ "$salloc_rc" != 0 ]; then
    if [ -n "$JOB_ID" ] && collx_verify_salloc_jobid "$JOB_ID"; then
      collx_log "scheduler-request=verified-grant"
      collx_record_allocation_jobid "$JOB_ID" || return 1
      return 0
    fi
    collx_log "ERROR: scheduler-request=rejected"
    if [ "$salloc_rc" -ge 128 ] && [ -z "$JOB_ID" ]; then
      collx_fail_stage scheduler-allocation "$log"
      return 1
    fi
    [ -n "$JOB_ID" ] || collx_reconcile_salloc_jobid "$job_name" || true
    [ -z "$JOB_ID" ] || collx_record_allocation_jobid "$JOB_ID" || true
    collx_fail_stage scheduler-allocation "$log"
    return 1
  fi
  if [ -z "$JOB_ID" ]; then
    collx_log "ERROR: scheduler-request=missing-grant"
    collx_reconcile_salloc_jobid "$job_name" || true
    collx_fail_stage scheduler-allocation "$log"
    return 1
  fi
  collx_record_allocation_jobid "$JOB_ID" || return 1
}

collx_record_allocation_jobid() {
  local job_id="$1" root="${COLLX_JOB_ROOT:-}" path temporary
  [[ "$job_id" =~ ^[1-9][0-9]*$ ]] || return 1
  [ -n "$root" ] || return 0
  collx_job_root_is_safe "$root" || return 1
  path="$root/jobid"
  temporary="$(mktemp "$root/.jobid.XXXXXX")" || return 1
  chmod 600 "$temporary" || { rm -f -- "$temporary"; return 1; }
  printf '%s\n' "$job_id" > "$temporary" \
    || { rm -f -- "$temporary"; return 1; }
  mv -f -- "$temporary" "$path" || { rm -f -- "$temporary"; return 1; }
}

collx_clear_allocation_jobid() {
  local root="${COLLX_JOB_ROOT:-}" path
  [ -n "$root" ] || return 0
  collx_job_root_is_safe "$root" || return 1
  path="$root/jobid"
  [ ! -e "$path" ] || {
    [ -f "$path" ] && [ ! -L "$path" ] \
      && [ "$(stat -c '%u:%a' "$path" 2>/dev/null)" = "$(id -u):600" ] || return 1
    rm -f -- "$path"
  }
}

collx_cancel_job() {
  local job_id="$1" active delay
  [[ "$job_id" =~ ^[0-9]+$ ]] || return 1
  scancel "$job_id" >/dev/null 2>&1 || true
  for delay in 1 2 4 8 16 32 64; do
    if ! active="$(squeue -h -j "$job_id" -o %A 2>/dev/null)"; then
      sleep "$delay"
      continue
    fi
    [ -n "$active" ] || return 0
    sleep "$delay"
  done
  collx_log "ERROR: scheduled allocation did not terminate during cleanup"
  return 1
}

# A workflow cancellation may kill a foreground Slurm step before Bash can run
# the launcher trap. Reconcile the mode-0600 allocation record from an always()
# workflow step before isolated source cleanup is allowed.
collx_reconcile_recorded_allocation() {
  local root="$1" path job_id
  collx_job_root_is_safe "$root" || return 1
  export COLLX_JOB_ROOT="$root"
  path="$root/jobid"
  [ -e "$path" ] || return 0
  [ -f "$path" ] && [ ! -L "$path" ] \
    && [ "$(stat -c '%u:%a' "$path" 2>/dev/null)" = "$(id -u):600" ] \
    || return 1
  IFS= read -r job_id < "$path" || return 1
  [[ "$job_id" =~ ^[1-9][0-9]*$ ]] || return 1
  collx_cancel_job "$job_id" && collx_clear_allocation_jobid
}

# Single multi-arch container for ALL NVIDIA SKUs: tag `v0.5.11-cu130` is an OCI
# image index covering linux/amd64 (B200) + linux/arm64 (GB200); enroot import
# pulls the matching arch. (cu130 = CUDA 13, system nccl.h in /usr/include, torch 2.9.x.)
# Import uses the configured tag because Enroot cannot reliably import a
# digest-qualified Docker Hub reference non-interactively.
# (v0.5.12-cu130 was rejected: its 62 layers overflow enroot's overlay-based
# squash creation on these nodes — "failed to mount overlay ... Invalid argument".
# v0.5.11-cu130 imports cleanly.)
# Runtime setup verifies the image-bundled DeepEP build for the detected GPU target.
COLLX_IMAGE_MULTIARCH="lmsysorg/sglang:v0.5.11-cu130"

# AMD (ROCm/CDNA): single mi35x-tagged image bundles MoRI for all three CDNA
# SKUs (gfx942 mi300x/mi325x + gfx950 mi355x).
COLLX_IMAGE_AMD_MORI="rocm/sgl-dev:sglang-0.5.14-rocm720-mi35x-mori-0701"
COLLX_MORI_COMMIT_AMD="bf99bdf18fc69887a346913ca01c315c2aa9bd4c" # pragma: allowlist secret
collx_default_image() {
  case "$1" in
    mi300x*|mi325x*|mi355x*) echo "$COLLX_IMAGE_AMD_MORI" ;;
    b200*|gb200*|b300*|gb300*|h100*|h200*) echo "$COLLX_IMAGE_MULTIARCH" ;;
    *) collx_die "no default image for runner prefix: $1" ;;
  esac
}

collx_select_image() {
  local image="$1"
  [[ "$image" =~ ^[A-Za-z0-9._/-]+:[A-Za-z0-9._-]+$ ]] \
    || collx_die "configured image reference is malformed"
  export COLLECTIVEX_IMAGE="$image"
}

# Create a per-UID cache under validated cluster-local storage. Only the fixed
# /cx-cache mount enters the container; the operator host path does not.
collx_prepare_backend_cache() {
  local cache
  unset COLLX_PREPARED_BACKEND_CACHE
  cache="$(python3 "$COLLX_RUNTIME_DIR/probe.py" prepare-cache "$1")" || return 1
  [[ "$cache" = /* ]] || return 1
  export COLLX_PREPARED_BACKEND_CACHE="$cache"
}

collx_git() {
  GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null GIT_TERMINAL_PROMPT=0 \
    git -c credential.helper= "$@"
}

collx_git_in_tree() {
  local directory="$1" canonical
  shift
  [[ "$directory" = /* ]] && [ -d "$directory" ] && [ ! -L "$directory" ] \
    || return 1
  [[ "$directory" != *'*'* && "$directory" != *$'\n'* && "$directory" != *$'\r'* ]] \
    || return 1
  canonical="$(cd -P -- "$directory" && pwd -P)" || return 1
  collx_git -c "safe.directory=$canonical" -C "$canonical" "$@"
}

collx_fetch_revision() {
  local repository="$1" revision="$2" destination="$3" attempt
  for attempt in 1 2 3; do
    rm -rf -- "$destination"
    if collx_git init -q "$destination" \
        && collx_git_in_tree "$destination" remote add origin "$repository" \
        && collx_git_in_tree "$destination" fetch -q --no-tags --depth 1 origin "$revision" \
        && collx_git_in_tree "$destination" -c advice.detachedHead=false \
          checkout -q --detach FETCH_HEAD \
        && [ "$(collx_git_in_tree "$destination" rev-parse HEAD)" = "$revision" ]; then
      return 0
    fi
    [ "$attempt" = 3 ] || sleep $((attempt * 5))
  done
  return 1
}

collx_backend_source_pin() {
  case "$1" in
    deepep-v2) printf '%s||%s' "$COLLX_DEEPEP_V2_COMMIT" "$COLLX_DEEPEP_V2_FMT_COMMIT" ;;
    deepep-hybrid) printf '%s|||%s' "$COLLX_DEEPEP_HYBRID_COMMIT" "$COLLX_DEEPEP_HYBRID_NCCL_COMMIT" ;;
    *) return 1 ;;
  esac
}

collx_backend_source_path() {
  local root="$1" backend="$2" revision tree fmt nccl pin
  pin="$(collx_backend_source_pin "$backend")" || return 1
  IFS='|' read -r revision tree fmt nccl <<< "$pin"
  printf '%s/%s-%s' "$root" "$backend" "$revision"
}

collx_apply_deepep_v2_nccl_check_fix() {
  python3 "$COLLX_RUNTIME_DIR/stage.py" rewrite-deepep-v2 "$1/deep_ep/__init__.py"
}

# Acquire source before compute allocation, preferring the verified same-run GHA seed.
_collx_prepare_backend_source() {
  local mount_src="$1" backend="$2" root source temporary revision tree fmt nccl pin
  root="$mount_src/experimental/CollectiveX/.collx_sources"
  COLLX_BACKEND_SOURCE_STEP="source mount creation"
  if [ ! -e "$root" ] && [ ! -L "$root" ]; then
    mkdir -m 700 -- "$root" || return 1
  fi
  [ -d "$mount_src" ] && [ -d "$root" ] || return 1
  COLLX_BACKEND_SOURCE_STEP="git lookup"
  command -v git >/dev/null || return 1
  COLLX_BACKEND_SOURCE_STEP="source pin resolution"
  source="$(collx_backend_source_path "$root" "$backend")" || return 1
  [ ! -d "$source" ] || return 0
  COLLX_BACKEND_SOURCE_STEP="source checkout creation"
  temporary="$(mktemp -d "$root/.${backend}.XXXXXX")" || return 1
  COLLX_BACKEND_SOURCE_STEP="source pin resolution"
  pin="$(collx_backend_source_pin "$backend")" || {
    rm -rf -- "$temporary"
    return 1
  }
  IFS='|' read -r revision tree fmt nccl <<< "$pin"
  COLLX_BACKEND_SOURCE_STEP="revision fetch"
  if ! collx_fetch_revision \
      https://github.com/deepseek-ai/DeepEP "$revision" "$temporary"; then
    rm -rf -- "$temporary"
    return 1
  fi
  COLLX_BACKEND_SOURCE_STEP="submodule fetch"
  if [ -n "$fmt" ] && ! collx_git_in_tree "$temporary" \
      -c "safe.directory=$temporary/third-party/fmt" \
      submodule update -q --init --depth 1 third-party/fmt; then
    rm -rf -- "$temporary"
    return 1
  fi
  if [ -n "$nccl" ] && ! collx_git_in_tree "$temporary" \
      -c "safe.directory=$temporary/third-party/nccl" \
      submodule update -q --init --depth 1 third-party/nccl; then
    rm -rf -- "$temporary"
    return 1
  fi
  if [ "$backend" = deepep-v2 ]; then
    COLLX_BACKEND_SOURCE_STEP="upstream NCCL check fix"
    collx_apply_deepep_v2_nccl_check_fix "$temporary" || {
      rm -rf -- "$temporary"
      return 1
    }
  fi
  COLLX_BACKEND_SOURCE_STEP="source publication"
  if ! mv -- "$temporary" "$source"; then
    rm -rf -- "$temporary"
    return 1
  fi
}

collx_prepare_backend_source() {
  local log backend="$2" COLLX_BACKEND_SOURCE_STEP="initialization"
  log="$(collx_private_log_path "backend-source-$backend")" || return 1
  if _collx_prepare_backend_source "$@" > "$log" 2>&1; then
    return 0
  fi
  printf '%s failed\n' "$COLLX_BACKEND_SOURCE_STEP" >> "$log"
  collx_log "ERROR: backend-source-step=${COLLX_BACKEND_SOURCE_STEP// /-}"
  collx_fail_stage backend-setup "$log"
}

collx_materialize_backend_source() {
  local backend="$1" destination="$2" source parent temporary
  [ -n "${COLLX_BACKEND_SOURCE_ROOT:-}" ] || return 1
  source="$(collx_backend_source_path "$COLLX_BACKEND_SOURCE_ROOT" "$backend")" || return 1
  [ -d "$source" ] || return 1
  parent="${destination%/*}"
  [ "$parent" != "$destination" ] && [ -d "$parent" ] && [ ! -L "$parent" ] \
    || return 1
  temporary="$(mktemp -d "$parent/.collectivex-source.XXXXXX")" || return 1
  if ! cp -R -- "$source/." "$temporary/"; then
    rm -rf -- "$temporary"
    return 1
  fi
  if ! rm -rf -- "$destination" || ! mv -- "$temporary" "$destination"; then
    rm -rf -- "$temporary"
    return 1
  fi
  [ -d "$destination" ]
}

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

collx_lock_canonical_gha_env() {
  local runner="$1" trusted_lock_dir=""
  local trusted_stage_dir=""
  local trusted_qos=""
  local trusted_socket_ifname="" trusted_rdma_devices=""
  local trusted_ib_gid_index="" trusted_rdma_service_level=""
  local trusted_rdma_traffic_class=""
  [ "${COLLECTIVEX_CANONICAL_GHA:-0}" = 1 ] || return 0
  [ "${GITHUB_ACTIONS:-}" = true ] \
    || collx_die "canonical CollectiveX execution requires GitHub Actions"
  [ -n "${COLLX_SHARD_FILE:-}" ] && [ "${COLLX_SHARD_SKU:-}" = "$runner" ] \
    || collx_die "canonical CollectiveX execution requires a matched shard"
  [[ "${GITHUB_RUN_ID:-}" =~ ^[1-9][0-9]*$ \
    && "${GITHUB_RUN_ATTEMPT:-}" =~ ^[1-9][0-9]*$ \
    && "${COLLECTIVEX_SOURCE_SHA:-}" =~ ^[0-9a-f]{40,64}$ ]] \
    || collx_die "canonical CollectiveX workflow identity is incomplete"

  # collx_load_operator_config clears inherited values before setting this process marker.
  # Preserve only values parsed from that private strict document.
  if [ "${COLLECTIVEX_OPERATOR_CONFIG_LOADED:-}" = "$$" ]; then
    trusted_lock_dir="${COLLX_LOCK_DIR:-}"
    trusted_stage_dir="${COLLX_STAGE_DIR:-}"
    trusted_qos="${COLLX_QOS:-}"
    trusted_socket_ifname="${COLLX_SOCKET_IFNAME:-}"
    trusted_rdma_devices="${COLLX_RDMA_DEVICES:-}"
    trusted_ib_gid_index="${COLLX_IB_GID_INDEX:-}"
    trusted_rdma_service_level="${COLLX_RDMA_SERVICE_LEVEL:-}"
    trusted_rdma_traffic_class="${COLLX_RDMA_TRAFFIC_CLASS:-}"
  fi
  # The legacy B300 operator row contains a root-owned stage path. B300's
  # compute-visible account home is the canonical source for its private base.
  case "$runner" in b300|gb300) trusted_stage_dir="" ;; esac
  unset COLLX_MASTER_PORT COLLX_MORI_KERNEL_TYPE COLLX_LOCK_DIR COLLX_STAGE_DIR COLLX_QOS
  unset MASTER_ADDR MASTER_PORT RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE
  unset COLLX_SOCKET_IFNAME COLLX_RDMA_DEVICES COLLX_IB_GID_INDEX COLLX_RDMA_SERVICE_LEVEL
  unset COLLX_RDMA_TRAFFIC_CLASS
  unset NCCL_NET NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME NCCL_IB_HCA
  unset NCCL_IB_GID_INDEX NCCL_IB_SL
  unset NVSHMEM_ENABLE_NIC_PE_MAPPING
  unset NVSHMEM_HCA_LIST NVSHMEM_IB_GID_INDEX NVSHMEM_IB_SL
  unset NVSHMEM_IB_ENABLE_IBGDA NVSHMEM_IBGDA_NIC_HANDLER
  unset EP_NIC_NAME EP_OVERRIDE_RDMA_SL
  unset MORI_RDMA_DEVICES
  unset MORI_RDMA_TC MORI_IO_TC MORI_RDMA_SL MORI_IO_SL
  unset HYBRID_EP_MULTINODE USE_NIXL RDMA_CORE_HOME DEEPEP_HYBRID_BUILD_MODE
  unset MORI_COMMIT MORI_DISABLE_AUTO_XGMI MORI_ENABLE_SDMA
  unset MORI_APP_LOG_LEVEL MORI_SHMEM_LOG_LEVEL MORI_IO_LOG_LEVEL
  unset NCCL_CUMEM_ENABLE NCCL_MNNVL_ENABLE MC_FORCE_MNNVL
  unset COLLX_BACKEND_CACHE_ROOT
  unset COLLX_PREPARED_BACKEND_CACHE COLLX_BACKEND_SOURCE_ROOT

  [ -n "${COLLX_SQUASH_DIR:-}" ] \
    || collx_die "canonical CollectiveX execution requires shared container storage"
  if [ -z "$trusted_stage_dir" ]; then
    case "$runner" in
      h100-dgxc)
        trusted_stage_dir="$(collx_prepare_implicit_stage_base "${COLLX_SQUASH_DIR%/*}")" \
          || collx_die "canonical CollectiveX execution cannot create an isolated shared stage directory"
        ;;
      b300)
        trusted_stage_dir="$(collx_prepare_implicit_stage_base "" \
          "${COLLECTIVEX_EXECUTION_ID:-${GITHUB_RUN_ID:-}}")" \
          || collx_die "canonical CollectiveX execution cannot create an isolated stage directory"
        ;;
      gb300)
        trusted_stage_dir="$(collx_prepare_implicit_stage_base "" \
          "${COLLECTIVEX_EXECUTION_ID:-${GITHUB_RUN_ID:-}}")" \
          || collx_die "canonical CollectiveX execution cannot create an isolated stage directory"
        ;;
      h200-dgxc|b200-dgxc)
        trusted_stage_dir="$(collx_prepare_implicit_stage_base)" \
          || collx_die "canonical CollectiveX execution cannot create an isolated stage directory"
        ;;
      mi300x|mi325x|mi355x)
        # AMD self-hosted runners and compute nodes share the runner filesystem,
        # while the image cache may be root-owned. Derive a runner-owned base
        # outside _work instead of weakening stage ownership validation.
        trusted_stage_dir="$(collx_prepare_runner_shared_stage_base)" \
          || collx_die "canonical AMD execution cannot create an isolated shared stage directory"
        ;;
      *) collx_die "canonical CollectiveX execution requires a configured shared stage directory" ;;
    esac
  elif [ "$runner" = mi300x ]; then
    # The MI300X runner home is a shared-filesystem symlink. Resolve the
    # operator-selected base once; collx_stage_path still validates the canonical
    # directory's ownership, permissions, overlap, and per-run child path.
    trusted_stage_dir="$(python3 "$COLLX_RUNTIME_DIR/stage.py" resolve-directory \
      "$trusted_stage_dir")" \
      || collx_die "canonical MI300X execution cannot resolve the shared stage directory"
  fi

  local policy_file policy_key policy_value
  policy_file="$(mktemp /tmp/inferencex-collectivex-policy.XXXXXX)" \
    || collx_die "cannot derive canonical SKU policy"
  if ! python3 "$COLLX_RUNTIME_DIR/config.py" canonical-policy "$runner" \
      "${COLLX_NODES:-0}" "${COLLX_GPUS_PER_NODE:-0}" \
      "$COLLX_IMAGE_MULTIARCH" "$COLLX_IMAGE_AMD_MORI" "$COLLX_MORI_COMMIT_AMD" \
      > "$policy_file"; then
    rm -f -- "$policy_file"
    collx_die "canonical CollectiveX placement differs from the SKU policy"
  fi
  while IFS= read -r -d '' policy_key && IFS= read -r -d '' policy_value; do
    printf -v "$policy_key" '%s' "$policy_value"
  done < "$policy_file"
  rm -f -- "$policy_file"
  case "$runner:$trusted_lock_dir" in
    mi300x:?*|mi325x:?*|mi355x:?*) export COLLX_LOCK_DIR="$trusted_lock_dir" ;;
  esac
  COLLX_STAGE_DIR="$trusted_stage_dir"
  [ -z "$trusted_qos" ] || export COLLX_QOS="$trusted_qos"
  [ -z "$trusted_socket_ifname" ] \
    || export COLLX_SOCKET_IFNAME="$trusted_socket_ifname"
  [ -z "$trusted_rdma_devices" ] \
    || export COLLX_RDMA_DEVICES="$trusted_rdma_devices"
  [ -z "$trusted_ib_gid_index" ] \
    || export COLLX_IB_GID_INDEX="$trusted_ib_gid_index"
  [ -z "$trusted_rdma_service_level" ] \
    || export COLLX_RDMA_SERVICE_LEVEL="$trusted_rdma_service_level"
  [ -z "$trusted_rdma_traffic_class" ] \
    || export COLLX_RDMA_TRAFFIC_CLASS="$trusted_rdma_traffic_class"
  export COLLX_STAGE_DIR
  unset COLLX_PUBLIC_RUNNER COLLX_GB_PRODUCT COLLX_DRYRUN COLLX_TIMING
  unset COLLX_ENROOT_LOCAL_IMPORT COLLECTIVEX_IMAGE
  export COLLX_IMAGE COLLX_NGPUS COLLX_SEED COLLX_RUN_TIMEOUT
  case "$runner" in
    gb200|gb300) export COLLX_MASTER_PORT ;;
    mi300x|mi325x|mi355x)
      export COLLX_MORI_KERNEL_TYPE MORI_COMMIT MORI_DISABLE_AUTO_XGMI MORI_ENABLE_SDMA
      export MORI_APP_LOG_LEVEL MORI_SHMEM_LOG_LEVEL MORI_IO_LOG_LEVEL
      ;;
  esac
}

collx_squash_path() {
  local squash_dir="$1" image="$2" key platform run_scope
  case "${COLLX_IMAGE_PLATFORM:-}" in
    linux/amd64) platform="" ;;
    linux/arm64) platform="_linux_arm64" ;;
    *) return 1 ;;
  esac
  run_scope="${GITHUB_RUN_ID:-${COLLECTIVEX_EXECUTION_ID:-manual}}-${GITHUB_RUN_ATTEMPT:-1}"
  run_scope="$(printf '%s' "$run_scope" | tr -cs 'A-Za-z0-9_.-' '-')" || return 1
  run_scope="${run_scope#-}"; run_scope="${run_scope%-}"
  [ -n "$run_scope" ] || return 1
  key="${platform}_${run_scope}_$(
    printf '%s' "$image" | sed 's#[/:@#]#_#g'
  )"
  printf '%s' "$squash_dir/${key}.sqsh"
}

# collx_ensure_squash <squash_dir> <image>  ->  echoes the squash file path.
# Imports via Enroot only if a valid squash is not already present, under a lock.
collx_ensure_squash() {
  local squash_dir="$1" image="$2" key sq locks lock_fd log
  local enroot_local="" import_rc=0 machine
  log="$(collx_private_log_path container-import)"
  machine="$(uname -m)"
  case "${COLLX_IMAGE_PLATFORM:-}:$machine" in
    linux/amd64:x86_64|linux/amd64:amd64|linux/arm64:aarch64|linux/arm64:arm64) ;;
    *) collx_fail_stage container-import "$log"; return 1 ;;
  esac
  mkdir -p "$squash_dir" 2>> "$log" \
    || { collx_fail_stage container-import "$log"; return 1; }
  sq="$(collx_squash_path "$squash_dir" "$image")" \
    || { collx_fail_stage container-import "$log"; return 1; }
  key="${sq##*/}"
  key="${key%.sqsh}"
  locks="$squash_dir/.locks"
  mkdir -p "$locks" 2>> "$log" \
    || { collx_fail_stage container-import "$log"; return 1; }
  { exec {lock_fd}>"$locks/${key}.lock"; } 2>> "$log" \
    || { collx_fail_stage container-import "$log"; return 1; }
  flock -w 900 "$lock_fd" 2>> "$log" \
    || { collx_fail_stage container-import "$log"; return 1; }
  if unsquashfs -l "$sq" >/dev/null 2>&1; then
    collx_log "container squash ready"
  else
    collx_log "importing configured container image"
    rm -f "$sq" 2>> "$log" \
      || { collx_fail_stage container-import "$log"; return 1; }
    # </dev/null: never block on an interactive password prompt.
    if [ "${COLLX_ENROOT_LOCAL_IMPORT:-0}" = 1 ]; then
      enroot_local="$(mktemp -d /tmp/inferencex-collectivex-enroot.XXXXXX)" \
        || { collx_fail_stage container-import "$log"; return 1; }
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
        || { collx_fail_stage container-import "$log"; return 1; }
    else
      enroot import -o "$sq" "docker://$image" </dev/null >> "$log" 2>&1 \
        || { collx_fail_stage container-import "$log"; return 1; }
    fi
    unsquashfs -l "$sq" >> "$log" 2>&1 \
      || { collx_fail_stage container-import "$log"; return 1; }
  fi
  flock -u "$lock_fd"
  exec {lock_fd}>&-
  echo "$sq"
}

# Import on an allocated compute node so multiarch tags resolve for the target
# architecture. The squash directory must be shared with the submit host.
collx_ensure_squash_on_job() {
  local job_id="$1" squash_dir="$2" image="$3" lock_dir="${4:-}" sq key lock
  local log_label=container-import log
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
  log="$(collx_private_log_path "$log_label")"
  # Import (or verify) the squash on EVERY allocated node, not just one. On SKUs
  # whose squash_dir is node-local (e.g. mi355x/mi300x /var/lib/squash) a single-
  # node import leaves the remaining nodes without the squash, so the per-node
  # container-hash check and the benchmark itself then fail with "No such file"
  # on whichever node was missed. The per-node script below is flock-guarded and
  # idempotent: on shared-FS SKUs the first node imports and every other node
  # short-circuits on the unsquashfs check, so no redundant import occurs.
  if ! srun --jobid="$job_id" --nodes="${COLLX_NODES:-1}" --ntasks="${COLLX_NODES:-1}" \
      --ntasks-per-node=1 --chdir=/tmp \
      --export="$(collx_host_exports)" \
      bash -s -- "$sq" "$lock" "$image" "$COLLX_IMAGE_PLATFORM" \
      > "$log" 2>&1 <<'BASH'
set -euo pipefail
sq="$1"; lock="$2"; image="$3"; platform="$4"
machine="$(uname -m)"
case "$platform:$machine" in
  linux/amd64:x86_64|linux/amd64:amd64|linux/arm64:aarch64|linux/arm64:arm64) ;;
  *) exit 13 ;;
esac
compute_home="$(mktemp -d /tmp/inferencex-collectivex-home.XXXXXX)"
trap 'rm -rf -- "$compute_home"' EXIT
export HOME="$compute_home" XDG_CACHE_HOME="$compute_home/.cache"
export ENROOT_TEMP_PATH="$compute_home/enroot-tmp"
export ENROOT_CACHE_PATH="$compute_home/enroot-cache"
export ENROOT_DATA_PATH="$compute_home/enroot-data"
export ENROOT_RUNTIME_PATH="$compute_home/enroot-run"
mkdir -p "$(dirname "$sq")" "$(dirname "$lock")" \
  "$ENROOT_TEMP_PATH" "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH" "$ENROOT_RUNTIME_PATH"
exec 9>"$lock"
# Wait indefinitely: with a shared-FS squash_dir every node contends on the same
# lock, and a slow cold import must not spuriously time out the waiters. The lock
# is tied to fd 9, so a crashed importer releases it automatically. Node-local
# squash_dirs use independent per-node locks, so there is no contention there.
flock 9
if unsquashfs -l "$sq" >/dev/null 2>&1; then
  echo 'container squash ready'
else
  rm -f -- "$sq"
  enroot import -o "$sq" "docker://$image" </dev/null
  unsquashfs -l "$sq" >/dev/null 2>&1
fi
BASH
  then
    collx_fail_stage container-import "$log"
    return 1
  fi
  printf '%s' "$sq"
}

collx_preflight_allocation() {
  local job_id="$1" nodes="$2" mount_src="$3" squash="$4" shard="${5:-}"
  local log rc=0 runtime shard_path="" probe_root probe_token index
  runtime="$mount_src/experimental/CollectiveX/runtime/prepare_backend.sh"
  [ -z "$shard" ] || shard_path="$mount_src/experimental/CollectiveX/$shard"
  log="$(collx_private_log_path allocation-preflight)"
  probe_root="$mount_src/.collectivex-preflight"
  probe_token="$probe_root/source"
  if [ -e "$probe_root" ] || [ -L "$probe_root" ] \
      || ! mkdir -m 700 "$probe_root"; then
    collx_fail_stage repository-stage "$log"
    return 1
  fi
  if ! printf '%s\n' "${COLLECTIVEX_EXECUTION_ID:-manual-$$}" > "$probe_token" \
      || ! chmod 600 "$probe_token"; then
    chmod 700 "$probe_root" >/dev/null 2>&1 || true
    rm -rf -- "$probe_root" >/dev/null 2>&1 || true
    collx_fail_stage repository-stage "$log"
    return 1
  fi
  srun --jobid="$job_id" --nodes="$nodes" --ntasks="$nodes" --ntasks-per-node=1 \
    --chdir=/tmp --input=all \
    --export="$(collx_host_exports)" bash -s -- "$runtime" "$shard_path" "$squash" \
    "$COLLX_IMAGE_PLATFORM" "$probe_root" \
    > "$log" 2>&1 <<'BASH' || rc=$?
set -euo pipefail
machine="$(uname -m)"
case "$4:$machine" in
  linux/amd64:x86_64|linux/amd64:amd64|linux/arm64:aarch64|linux/arm64:arm64) ;;
  *) exit 13 ;;
esac
test -r "$1" || exit 10
[ -z "$2" ] || test -r "$2" || exit 11
test -r "$3" || exit 12
unsquashfs -s "$3" >/dev/null 2>&1 || exit 12
case "${SLURM_NODEID:-}" in ""|*[!0-9]*) exit 10 ;; esac
[ -d "$5" ] && [ ! -L "$5" ] && [ -r "$5/source" ] || exit 10
(set -C; cat "$5/source" > "$5/node-$SLURM_NODEID") || exit 10
cmp -s -- "$5/source" "$5/node-$SLURM_NODEID" || exit 10
BASH
  if [ "$rc" = 0 ]; then
    for ((index = 0; index < nodes; index++)); do
      if ! cmp -s -- "$probe_token" "$probe_root/node-$index"; then
        rc=10
        break
      fi
    done
  fi
  if [ -d "$probe_root" ] && [ ! -L "$probe_root" ]; then
    chmod 700 "$probe_root" >/dev/null 2>&1 || rc=10
  fi
  rm -rf -- "$probe_root" >/dev/null 2>&1 || rc=10
  [ "$rc" = 0 ] && return 0
  case "$rc" in
    10|11) collx_fail_stage repository-stage "$log" ;;
    12) collx_fail_stage container-hash "$log" ;;
    *) collx_fail_stage container-launch "$log" ;;
  esac
  return 1
}

# A clean nvidia-smi inventory does not prove that a prior cancelled workload
# released every CUDA context. Retaining each primary context catches poisoned
# allocations before a full shard spends time failing every case.
collx_validate_cuda_context_on_job() {
  local job_id="$1" nodes="$2" gpus_per_node="$3" log_label=cuda-context log
  case "${COLLX_SALLOC_ATTEMPT:-1}" in
    1) ;;
    2|3) log_label+="-a${COLLX_SALLOC_ATTEMPT}" ;;
    *) return 1 ;;
  esac
  log="$(collx_private_log_path "$log_label")"
  COLLX_CUDA_CONTEXT_LOG="$log"
  srun --jobid="$job_id" --nodes="$nodes" --ntasks="$nodes" --ntasks-per-node=1 \
    --gres=gpu:"$gpus_per_node" --chdir=/tmp --input=all \
    --export="$(collx_host_exports)" python3 /dev/stdin cuda-context "$gpus_per_node" \
    < "$COLLX_RUNTIME_DIR/probe.py" >"$log" 2>&1
}

# Resolve the exact per-execution child before any copy starts, so the parent
# EXIT trap can remove an interrupted partial stage. The configured base must
# already exist on compute-visible storage and must not traverse symlinks.
collx_stage_path() {
  local repo_root="$1" stage_base="${2:-}" tag safe_tag stage_path
  tag="${COLLECTIVEX_EXECUTION_ID:-${GITHUB_RUN_ID:-manual-$$}}"
  [[ "$tag" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
    || collx_die "invalid staging execution identity"
  safe_tag="$(printf '%s' "$tag" | tr -c 'A-Za-z0-9._-' '_')"
  if [ -z "$stage_base" ] || [ "$stage_base" = "$repo_root" ]; then
    [ -n "${COLLX_SQUASH_DIR:-}" ] \
      || collx_die "CollectiveX staging requires COLLX_STAGE_DIR or COLLX_SQUASH_DIR"
    stage_base="$COLLX_SQUASH_DIR"
    stage_path="${stage_base%/}/.collectivex-stage-$safe_tag"
  else
    stage_path="${stage_base%/}/job_$safe_tag"
  fi
  python3 "$COLLX_RUNTIME_DIR/stage.py" validate-stage-path "$repo_root" "$stage_base" \
    "$stage_path" "${COLLX_JOB_ROOT:-}" "${GITHUB_WORKSPACE:-}"
}

# Stage only the public benchmark tree into a pre-resolved, private execution
# child. A runner-owned marker makes recursive cleanup an explicit capability.
collx_stage_repo() {
  local repo_root="$1" stage_dir="$2" expected log tag copy_error
  expected="$(collx_stage_path "$repo_root" "${COLLX_STAGE_DIR:-}")" \
    || collx_die "configured stage base is unavailable or unsafe"
  [ "$stage_dir" = "$expected" ] \
    || collx_die "execution stage differs from the configured stage base"
  tag="${COLLECTIVEX_EXECUTION_ID:-${GITHUB_RUN_ID:-manual-$$}}"
  python3 "$COLLX_RUNTIME_DIR/stage.py" create-stage "$stage_dir" "$tag" \
    || collx_die "cannot create the configured stage directory"
  collx_log "staging CollectiveX on compute-visible storage"
  log="$(collx_private_log_path repository-stage)"
  if ! python3 "$COLLX_RUNTIME_DIR/stage.py" copy-repository \
      "$repo_root/experimental/CollectiveX" \
      "$stage_dir/experimental/CollectiveX" > "$log" 2>&1; then
    copy_error="$(grep -aoE 'collectivex-stage-copy-error=[A-Za-z]+:[0-9]+' "$log" \
      | tail -n 1 || true)"
    [ -z "$copy_error" ] || collx_log "ERROR: repository-stage-$copy_error"
    rm -rf -- "$stage_dir" >/dev/null 2>&1 \
      || collx_log "ERROR: cannot remove the incomplete execution stage"
    collx_fail_stage repository-stage "$log" || true
    return 1
  fi
}

# collx_collect_results <mount_src> <repo_root>
# When the run used a staged (compute-visible) mount, copy result JSONs back to
# the original checkout's results/ so the workflow's upload-artifact (which reads
# the checkout, not the stage dir) finds them. No-op when no staging was used.
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
  local mount_src="$1" repo_root="$2" base="${COLLX_STAGE_DIR:-}" tag safe_tag expected
  tag="${COLLECTIVEX_EXECUTION_ID:-${GITHUB_RUN_ID:-manual-$$}}"
  safe_tag="$(printf '%s' "$tag" | tr -c 'A-Za-z0-9._-' '_')"
  [ "$mount_src" != "$repo_root" ] || return 0
  if [ -n "$base" ] && [ "$base" != "$repo_root" ]; then
    expected="${base%/}/job_$safe_tag"
  else
    [ -n "${COLLX_SQUASH_DIR:-}" ] \
      || { collx_log "ERROR: cannot identify the generated stage directory"; return 1; }
    expected="${COLLX_SQUASH_DIR%/}/.collectivex-stage-$safe_tag"
  fi
  if [ "$mount_src" != "$expected" ] || [ "$mount_src" = / ] \
      || { [ -n "$base" ] && [ "$mount_src" = "$base" ]; }; then
    collx_log "ERROR: refusing to remove an unrecognized stage directory"
    return 1
  fi
  if ! python3 "$COLLX_RUNTIME_DIR/stage.py" validate-cleanup "$mount_src" "$tag"; then
    collx_log "ERROR: refusing to remove an unowned stage directory"
    return 1
  fi
  rm -rf -- "$mount_src" >/dev/null 2>&1 || {
    collx_log "ERROR: cannot remove generated stage directory"
    return 1
  }
  collx_log "removed generated per-execution stage directory"
}

# Run one validated shard with one Slurm task per GPU on one or more nodes.
# Launchers provide only allocation/container policy through globals and
# COLLX_DISTRIBUTED_CONTAINER_ARGS; per-case benchmark inputs travel as run_ep.py
# argv decoded from the shard control (config.py case-args), never as env.
# shellcheck disable=SC2153
collx_run_shard() {
  local build_log build_rc expected_cases ci=0 failed_cases=0
  local runtime_log run_rc summary_log argv_file case_label index shard wrap
  local -a container_args ep_args manual_phases
  [ "${NODES:-0}" -ge 1 ] && [ "${NGPUS:-0}" = "$((NODES * GPN))" ] \
    || collx_die "invalid shard launcher placement"
  [ -n "${JOB_ID:-}" ] && [ -n "${SQUASH_FILE:-}" ] \
    && [ -n "${CONTAINER_MOUNTS:-}" ] || collx_die "shard launcher is incomplete"
  wrap="$(collx_source_backend_env)"$'\n'"$(collx_slurm_rank_wrapper)"

  collx_resolve_slurm_rendezvous "$JOB_ID"
  collx_apply_network_profile "$NODES" "${COLLX_TRANSPORT:-}"
  mkdir -p "$MOUNT_SRC/experimental/CollectiveX/results"
  container_args=(--container-mounts="$CONTAINER_MOUNTS" --no-container-mount-home
    --container-workdir=/ix/experimental/CollectiveX --no-container-entrypoint)
  if declare -p COLLX_DISTRIBUTED_CONTAINER_ARGS >/dev/null 2>&1; then
    container_args+=("${COLLX_DISTRIBUTED_CONTAINER_ARGS[@]}")
  fi
  local container_name="cxep_${JOB_ID}"

  collx_log "shard backend preparation: bench=$COLLX_BENCH nodes=$NODES"
  collx_set_failure_stage backend-setup
  build_log="$(collx_private_log_path backend-prepare)"
  set +e
  srun --jobid="$JOB_ID" --nodes="$NODES" --ntasks-per-node=1 --chdir=/tmp \
    --container-name="$container_name" --container-image="$SQUASH_FILE" \
    "${container_args[@]}" --export=ALL \
    bash /ix/experimental/CollectiveX/runtime/prepare_backend.sh \
    </dev/null >"$build_log" 2>&1
  build_rc=$?
  if [ "$build_rc" = 0 ]; then
    srun --jobid="$JOB_ID" --nodes="$NODES" --ntasks-per-node=1 --chdir=/tmp \
      --container-name="$container_name" --container-image="$SQUASH_FILE" \
      "${container_args[@]}" \
      --export=ALL bash -c "$(collx_backend_probe)" \
      </dev/null >>"$build_log" 2>&1
    build_rc=$?
  fi
  set -e
  if [ "$build_rc" != 0 ]; then
    collx_fail_stage backend-setup "$build_log" || true
    return "$build_rc"
  fi
  collx_set_failure_stage execution

  shard="${COLLX_SHARD_FILE:-}"
  [ -z "$shard" ] || [ -f "$shard" ] || shard="$COLLX_DIR/$shard"
  if [ -n "$shard" ]; then
    [ -f "$shard" ] || collx_die "shard control is unavailable"
    expected_cases="$(python3 "$COLLX_RUNTIME_DIR/config.py" case-count "$shard")" \
      && [[ "$expected_cases" =~ ^[1-9][0-9]*$ ]] \
      || collx_die "could not enumerate validated shard cases"
  else
    # Ad-hoc runs without a shard control take one case per requested phase
    # from the operator's COLLX_* environment (config.py manual-args).
    local phase_list="${COLLX_PHASE:-decode}"
    [ "$phase_list" != both ] || phase_list="decode prefill"
    read -r -a manual_phases <<< "$phase_list"
    expected_cases="${#manual_phases[@]}"
  fi

  argv_file="$(mktemp)" || return 1
  while [ "$ci" -lt "$expected_cases" ]; do
    if [ -n "$shard" ]; then
      python3 "$COLLX_RUNTIME_DIR/config.py" case-args "$shard" "$ci" \
        "$RUNNER" "$TS" "${COLLX_SEED:-67}" \
        "$NGPUS" "$NODES" "$GPN" "$SCALE_UP_DOMAIN" > "$argv_file" \
        || { rm -f "$argv_file"; collx_die "shard case $ci does not decode against this allocation"; }
    else
      python3 "$COLLX_RUNTIME_DIR/config.py" manual-args "${manual_phases[ci]}" "$ci" \
        "$RUNNER" "$TS" "${COLLX_SEED:-67}" > "$argv_file" \
        || { rm -f "$argv_file"; collx_die "manual case $ci does not decode"; }
    fi
    mapfile -d '' -t ep_args < "$argv_file"
    [ "${#ep_args[@]}" -gt 0 ] \
      || { rm -f "$argv_file"; collx_die "case $ci produced no benchmark arguments"; }
    case_label=""
    for ((index = 0; index + 1 < ${#ep_args[@]}; index++)); do
      [ "${ep_args[index]}" != --case-id ] || case_label="${ep_args[index + 1]}"
    done
    collx_log "EP${NGPUS}[$((ci + 1))/$expected_cases] id=${case_label:-manual} $COLLX_BENCH"
    runtime_log="$(collx_private_log_path "runtime-c$(printf '%03d' "$ci")")"
    set +e
    timeout -k 30 "${COLLX_RUN_TIMEOUT:-900}" srun --jobid="$JOB_ID" --nodes="$NODES" \
      --ntasks="$NGPUS" --ntasks-per-node="$GPN" --chdir=/tmp \
      --container-name="$container_name" --container-image="$SQUASH_FILE" \
      "${container_args[@]}" \
      --export=ALL \
      bash -c "$wrap" _ "${ep_args[@]}" \
      </dev/null >"$runtime_log" 2>&1
    run_rc=$?
    set -e
    # A case counts as run purely on the distributed command's return code. The
    # rank-zero result the harness wrote (if any) is left in place for the
    # summary renderer, which validates nothing.
    if [ "$run_rc" != 0 ]; then
      collx_fail_stage execution "$runtime_log" || true
      failed_cases=$((failed_cases + 1))
    fi
    ci=$((ci + 1))
  done
  rm -f "$argv_file"
  if [ "$failed_cases" -ne 0 ]; then
    summary_log="$(collx_private_log_path shard-summary)"
    printf 'SHARD done: %s/%s case(s) failed\n' "$failed_cases" "$expected_cases" \
      > "$summary_log"
    collx_fail_stage execution "$summary_log" || true
    return 1
  fi
  return 0
}

# Remove this allocation's persistent pyxis container before the allocation is
# released. Clusters may run pyxis with container_scope=global, where the named
# --container-writable container every shard uses (cxep_<jobid>) survives job
# teardown and its unpacked rootfs — tens of GB per node — would otherwise
# accumulate on every allocated node's local image store until it fills and the
# next writable extraction fails with ENOSPC. Best-effort and bounded: teardown
# must never hang or fail on this.
collx_remove_distributed_container() {
  local job_id="$1" nodes="${2:-1}"
  [ -n "$job_id" ] || return 0
  [ "$nodes" -ge 1 ] 2>/dev/null || return 0
  timeout 120 srun --jobid="$job_id" --nodes="$nodes" --ntasks-per-node=1 \
    --chdir=/tmp enroot remove -f "pyxis_cxep_${job_id}" \
    </dev/null >/dev/null 2>&1 || true
}

collx_launcher_cleanup() {
  local rc="$1" stage_root="${MOUNT_SRC:-}" source_root allocation_stopped=1
  source_root="${stage_root:-${REPO_ROOT:-}}"
  trap - EXIT HUP INT TERM
  if [ -n "${COLLECTIVEX_EPHEMERAL_CONFIG_PATH:-}" ]; then
    rm -f -- "$COLLECTIVEX_EPHEMERAL_CONFIG_PATH" >/dev/null 2>&1 || true
    unset COLLECTIVEX_EPHEMERAL_CONFIG_PATH
  fi
  if [ -n "${JOB_ID:-}" ]; then
    collx_remove_distributed_container "$JOB_ID" "${NODES:-1}"
    if ! collx_cancel_job "$JOB_ID"; then
      allocation_stopped=0
      [ "$rc" != 0 ] || rc=1
    elif ! collx_clear_allocation_jobid; then
      allocation_stopped=0
      [ "$rc" != 0 ] || rc=1
    fi
  elif [ "${COLLX_ALLOCATION_UNCERTAIN:-0}" = 1 ]; then
    allocation_stopped=0
    [ "$rc" != 0 ] || rc=1
  fi
  [ "$allocation_stopped" = 1 ] || source_root="${REPO_ROOT:-$source_root}"
  if [ "$rc" != 0 ] \
      && [ -n "${REPO_ROOT:-}" ] && [ -n "${COLLX_BENCH:-}" ]; then
    collx_log "ERROR: terminal-failure-stage=${COLLX_FAILSAFE_MODE:-setup}"
    [ -d "$source_root/experimental/CollectiveX" ] || source_root="$REPO_ROOT"
    [ "$source_root" = "$REPO_ROOT" ] \
      || collx_collect_results "$source_root" "$REPO_ROOT" || true
  fi
  if [ "$allocation_stopped" = 1 ] && [ -n "${REPO_ROOT:-}" ] \
      && [ -n "$stage_root" ] && [ "$stage_root" != "$REPO_ROOT" ]; then
    if ! collx_cleanup_stage "$stage_root" "$REPO_ROOT"; then
      [ "$rc" != 0 ] || rc=1
    fi
  fi
  [ "${COLLECTIVEX_CANONICAL_GHA:-0}" = 1 ] || collx_cleanup_private_logs "$rc"
  exit "$rc"
}

collx_install_launcher_fail_safe() {
  COLLX_ALLOCATION_UNCERTAIN=0
  trap 'collx_launcher_cleanup "$?"' EXIT
  trap 'collx_launcher_cleanup 129' HUP
  trap 'collx_launcher_cleanup 130' INT
  trap 'collx_launcher_cleanup 143' TERM
}
