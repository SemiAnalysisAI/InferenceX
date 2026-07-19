# shellcheck shell=bash
# CollectiveX — shared launcher helpers (sourced, not executed).
#
# Cluster-generic scaffolding only (Slurm/container/build/staging); no
# model-serving. Logging goes to stderr so functions can `echo` a single
# result on stdout.

unset COLLECTIVEX_OPERATOR_CONFIG_LOADED
COLLX_RUNTIME_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

collx_log() { printf '[collectivex] %s\n' "$*" >&2; }
collx_die() { printf '[collectivex] FATAL: %s\n' "$*" >&2; exit 1; }

COLLX_DEEPEP_V2_REPO="https://github.com/deepseek-ai/DeepEP"
COLLX_DEEPEP_V2_COMMIT="fa8a9b16898204afd347c663b89e65ef87dc6ce6"

# Print bounded command output without maintaining a parallel failure taxonomy.
collx_log_tail() {
  local log_path="$1"
  if [ -s "$log_path" ]; then
    collx_log "--- command log tail ---"
    tail -n 100 -- "$log_path" >&2 || true
    collx_log "--- end command log tail ---"
  fi
}

# Shared launcher skeleton: the identity-stage boilerplate every launcher runs
# before any SKU-specific work.
collx_launcher_prologue() {
  JOB_ID=""
  collx_install_launcher_fail_safe
  [ -n "${COLLX_SHARD_FILE:-}" ] || collx_die "COLLX_SHARD_FILE is required"
  collx_load_operator_config
  collx_prepare_stage_dir "$1"
}

# Shared launcher tail: run the shard, collect artifacts, and fold both return
# codes into FINAL_RC (run failures win; collection failures surface otherwise).
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

# Load the selected SKU's public platform settings plus any allowlisted local
# operator overrides; JSON values are never sourced or evaluated as shell.
collx_load_operator_config() {
  [ -n "${COLLECTIVEX_OPERATOR_CONFIG_LOADED:-}" ] \
    && [ "$COLLECTIVEX_OPERATOR_CONFIG_LOADED" = "$$" ] && return 0
  local config_path parsed_path key value
  unset COLLX_IMAGE COLLX_IMAGE_PLATFORM
  unset COLLX_PARTITION COLLX_ACCOUNT COLLX_QOS COLLX_SQUASH_DIR COLLX_STAGE_DIR COLLX_ENROOT_CACHE_PATH
  unset ENROOT_CACHE_PATH
  unset COLLX_EXCLUDE_NODES COLLX_NODELIST COLLX_LOCK_DIR COLLX_MASTER_PORT
  unset COLLX_SOCKET_IFNAME COLLX_RDMA_DEVICES COLLX_IB_GID_INDEX COLLX_RDMA_SERVICE_LEVEL
  unset COLLX_RDMA_TRAFFIC_CLASS COLLX_RAIL_ISOLATED
  unset MASTER_ADDR MASTER_PORT RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE
  config_path="${COLLECTIVEX_OPERATOR_CONFIG:-${XDG_CONFIG_HOME:-${HOME}/.config}/inferencex/collectivex.json}"
  if [ ! -e "$config_path" ]; then
    # No operator document: a host-utility step (no SKU) is a no-op; a known SKU
    # emits the tracked platform_config.json operator baseline ("-" sentinel).
    # An optional local file at COLLECTIVEX_OPERATOR_CONFIG/XDG still overlays it.
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

# Per-step log files: several callers parse these for markers (salloc grant,
# stage copy-error, per-node network selectors), so they are a data channel,
# not just failure display. Logs persist after the run for postmortem.
collx_private_log_path() {
  local path="${COLLX_JOB_ROOT:-/tmp/inferencex-collectivex-$(id -u)}/logs/$1.log"
  mkdir -p "${path%/*}" || collx_die "cannot create log directory"
  : > "$path" || collx_die "cannot create runtime log"
  printf '%s' "$path"
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
    "missing platform or runner configuration: ${missing[*]}"
}

collx_export_gid_index_for_link_layer() {
  local link_layer="$1"
  unset NVSHMEM_IB_GID_INDEX NCCL_IB_GID_INDEX
  [ -n "${COLLX_IB_GID_INDEX:-}" ] || return 0
  case "$link_layer" in
    roce)
      export NVSHMEM_IB_GID_INDEX="$COLLX_IB_GID_INDEX"
      export NCCL_IB_GID_INDEX="$COLLX_IB_GID_INDEX"
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
  local -a selectors
  [[ "$nodes" =~ ^[1-9][0-9]*$ ]] || collx_die "invalid network placement"
  unset NCCL_NET NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME NCCL_IB_HCA
  unset NCCL_IB_GID_INDEX NCCL_IB_SL NCCL_IB_MERGE_NICS NCCL_CROSS_NIC
  unset NVSHMEM_ENABLE_NIC_PE_MAPPING
  unset NVSHMEM_HCA_LIST NVSHMEM_IB_GID_INDEX NVSHMEM_IB_SL
  unset NVSHMEM_IB_ENABLE_IBGDA NVSHMEM_IBGDA_NIC_HANDLER
  unset EP_NIC_NAME EP_OVERRIDE_RDMA_SL
  unset MORI_RDMA_DEVICES
  unset MORI_RDMA_TC MORI_IO_TC MORI_RDMA_SL MORI_IO_SL
  # Single-node and MNNVL runs need only the scrub above; everything past this
  # point is the scale-out path, so no per-branch scale-out guards remain. Single-node
  # low-latency also takes this early return: the decode kernels run over the intra-node
  # NVLink/XGMI path (DeepEP's allow_nvlink_for_low_latency_mode; MoRI's IntraNodeLL), so
  # they need no scale-out RDMA env and must NOT force IBGDA — verified on h200 EP8 with
  # /dev/gdrdrv absent (forcing IBGDA there would fail, as it did historically on b300).
  { [ "$nodes" -gt 1 ] && [ "$transport" != mnnvl ]; } || return 0
  [ -n "${COLLX_RDMA_DEVICES:-}" ] \
    || collx_die "RDMA execution requires a private device selector"
  if [ -n "${COLLX_SOCKET_IFNAME:-}" ]; then
    [[ "$COLLX_SOCKET_IFNAME" =~ ^[A-Za-z][A-Za-z0-9_.-]{0,31}$ ]] \
      || collx_die "invalid private socket interface selector"
    export NCCL_SOCKET_IFNAME="$COLLX_SOCKET_IFNAME" GLOO_SOCKET_IFNAME="$COLLX_SOCKET_IFNAME"
  fi
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
  # RCCL selects its own net plugin; NCCL_NET=IB breaks AMD SKUs.
  if [ "${COLLX_VENDOR:-nvidia}" = amd ]; then
    unset NCCL_NET
  else
    export NCCL_NET=IB
  fi
  export NCCL_IB_HCA="=$COLLX_RDMA_DEVICES"
  export MORI_RDMA_DEVICES="$rdma_names" EP_NIC_NAME="$ep_nic"
  # The selector enumerates individual ports. NCCL's default dual-port fusion
  # would collapse each card into one "fused" device, and any fused device
  # disables NCCL GIN (init.cc nicFused gate) — the deep_ep EP16 hybrid path
  # then asserts railedGinType == NCCL_GIN_TYPE_NONE. Single-port selectors are
  # unaffected, so pin unmerged operation for every scale-out run.
  export NCCL_IB_MERGE_NICS=0
  if [ -n "${COLLX_RAIL_ISOLATED:-}" ]; then
    [[ "$COLLX_RAIL_ISOLATED" =~ ^[01]$ ]] \
      || collx_die "invalid private rail isolation flag"
    # Rail-isolated multi-plane fabrics (per-port rail subnets, no cross-rail
    # routing): cross-NIC pairs black-hole at QP RTR, and NCCL's RAIL GIN
    # connection type is the one the fabric supports.
    [ "$COLLX_RAIL_ISOLATED" != 1 ] || export NCCL_CROSS_NIC=0
  fi
  if [ -n "${COLLX_IB_GID_INDEX:-}" ]; then
    [[ "$COLLX_IB_GID_INDEX" =~ ^[0-9]+$ ]] && [ "$COLLX_IB_GID_INDEX" -le 255 ] \
      || collx_die "invalid private IB GID index"
  fi
  if [ -n "${COLLX_RDMA_SERVICE_LEVEL:-}" ]; then
    [[ "$COLLX_RDMA_SERVICE_LEVEL" =~ ^[0-9]+$ ]] && [ "$COLLX_RDMA_SERVICE_LEVEL" -le 15 ] \
      || collx_die "invalid private RDMA service level"
    export NVSHMEM_IB_SL="$COLLX_RDMA_SERVICE_LEVEL"
    export NCCL_IB_SL="$COLLX_RDMA_SERVICE_LEVEL"
    export EP_OVERRIDE_RDMA_SL="$COLLX_RDMA_SERVICE_LEVEL"
    export MORI_RDMA_SL="$COLLX_RDMA_SERVICE_LEVEL" MORI_IO_SL="$COLLX_RDMA_SERVICE_LEVEL"
  fi
  if [ -n "${COLLX_RDMA_TRAFFIC_CLASS:-}" ]; then
    [[ "$COLLX_RDMA_TRAFFIC_CLASS" =~ ^[0-9]+$ ]] && [ "$COLLX_RDMA_TRAFFIC_CLASS" -le 255 ] \
      || collx_die "invalid private RDMA traffic class"
    export MORI_RDMA_TC="$COLLX_RDMA_TRAFFIC_CLASS" MORI_IO_TC="$COLLX_RDMA_TRAFFIC_CLASS"
  fi
  local nic_handler=gpu
  export NVSHMEM_IB_ENABLE_IBGDA=1 NVSHMEM_IBGDA_NIC_HANDLER="$nic_handler"
  if [ -n "${COLLX_RDMA_LINK_LAYER:-}" ]; then
    case "$COLLX_RDMA_LINK_LAYER" in
      roce|infiniband) ;;
      *) collx_die "invalid validated RDMA link layer" ;;
    esac
    collx_export_gid_index_for_link_layer "$COLLX_RDMA_LINK_LAYER"
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
  local job_id="$1" nodes="$2" transport="$3"
  local log_label=network-profile log rc=0 marker_count link_layer
  { [ "$nodes" -gt 1 ] && [ "$transport" != mnnvl ]; } || return 0
  [[ "$job_id" =~ ^[1-9][0-9]*$ && "$nodes" =~ ^[1-9][0-9]*$ ]] \
    || return 1
  [ -n "${COLLX_RDMA_DEVICES:-}" ] || return 1
  case "${COLLX_NETWORK_VALIDATION_ATTEMPT:-1}" in
    1) ;;
    2|3) log_label+="-a${COLLX_NETWORK_VALIDATION_ATTEMPT}" ;;
    *) return 1 ;;
  esac
  log="$(collx_private_log_path "$log_label")" || return 1
  export COLLX_NETWORK_PROFILE_LOG="$log"
  srun --jobid="$job_id" --nodes="$nodes" --ntasks="$nodes" --ntasks-per-node=1 \
    --chdir=/tmp --input=all --export="$(collx_host_exports)" \
    python3 /dev/stdin network-profile "${COLLX_SOCKET_IFNAME:-}" \
      "$COLLX_RDMA_DEVICES" "${COLLX_IB_GID_INDEX:-}" \
    < "$COLLX_RUNTIME_DIR/probe.py" > "$log" 2>&1 || rc=$?
  if [ "$rc" != 0 ]; then
    marker="$(grep -aoE '(socket-interface|rdma-(device|port))-[0-9]+=(missing|down|inactive|default-route-missing|gid-missing|gid-empty|link-layer-missing|link-layer-invalid|link-layer-mixed)' "$log" \
      | tail -n 1 || true)"
    [ -z "$marker" ] || collx_log "ERROR: network-profile-$marker"
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
    *) return 1 ;;
  esac
  export COLLX_RDMA_LINK_LAYER="$link_layer"
  collx_export_gid_index_for_link_layer "$link_layer"
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

# Printed into `bash -c` ahead of the rank wrapper. Sources the per-node backend
# environment written during preparation.
collx_source_backend_env() {
  cat <<'BASH'
case "${SLURM_NODEID:-}" in ""|*[!0-9]*) exit 66;; esac
. "/ix/experimental/CollectiveX/.collx_backend/env/node-${SLURM_NODEID}.sh" || exit 66
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

# Allocate via salloc's stable grant message and assign JOB_ID in this shell.
# Record it so workflow cleanup can release a launcher interrupted by Actions.
collx_salloc_jobid() {
  local log_label=scheduler-allocation log job_id root="${COLLX_JOB_ROOT:-}"
  case "${COLLX_SALLOC_ATTEMPT:-1}" in
    1) ;;
    2|3) log_label+="-a${COLLX_SALLOC_ATTEMPT}" ;;
    *) return 1 ;;
  esac
  if ! log="$(collx_private_log_path "$log_label")"; then
    collx_log "ERROR: scheduler log is unavailable"
    return 1
  fi
  collx_log "scheduler-request=submit"
  if ! (salloc "$@" --no-shell) > "$log" 2>&1; then
    collx_log "ERROR: scheduler allocation failed"
    collx_log_tail "$log"
    return 1
  fi
  job_id="$(sed -nE \
      's/.*Granted job allocation ([1-9][0-9]*).*/\1/p' "$log" | head -n1)"
  [[ "$job_id" =~ ^[1-9][0-9]*$ ]] || return 1
  JOB_ID="$job_id"
  if [ -n "$root" ]; then
    collx_job_root_is_safe "$root" || return 1
    (umask 077; printf '%s\n' "$JOB_ID" > "$root/jobid") || return 1
  fi
}

# Idempotent cleanup for launcher traps, allocation retries, and workflow recovery.
collx_cleanup_allocation() {
  local root="${1:-${COLLX_JOB_ROOT:-}}" path="" job_id="${JOB_ID:-}" active
  if [ -n "$root" ]; then
    collx_job_root_is_safe "$root" || return 1
    path="$root/jobid"
    if [ -z "$job_id" ] && [ -f "$path" ]; then
      IFS= read -r job_id < "$path" || return 1
    fi
  fi
  [ -n "$job_id" ] || return 0
  [[ "$job_id" =~ ^[1-9][0-9]*$ ]] || return 1
  scancel "$job_id" >/dev/null 2>&1 || true
  for _ in {1..30}; do
    active="$(squeue -h -j "$job_id" -o %A 2>/dev/null)" || active=unknown
    if [ -z "$active" ]; then
      [ -z "$path" ] || rm -f -- "$path"
      return
    fi
    sleep 1
  done
  collx_log "ERROR: scheduled allocation did not terminate during cleanup"
  return 1
}

# Import uses the configured tag because Enroot cannot reliably import a
# digest-qualified Docker Hub reference non-interactively.
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

# Fetch the pinned DeepEP tree before allocating GPUs.
collx_prepare_deepep_source() {
  local mount_src="$1" root source temporary log
  root="$mount_src/experimental/CollectiveX/.collx_sources"
  source="$root/deepep-v2-$COLLX_DEEPEP_V2_COMMIT"
  [ ! -d "$source" ] || return 0
  mkdir -p -- "$root" && chmod 700 "$root" || return 1
  temporary="$(mktemp -d "$root/.deepep-v2.XXXXXX")" || return 1
  log="$(collx_private_log_path backend-source-deepep-v2)" || return 1
  # On b300 the NFS export can realize a newly created stage dir as UID 0 while
  # git runs as the UID-mapped Actions user, tripping git's "dubious ownership"
  # guard on the source tree and its fmt submodule. HOME is this job's ephemeral
  # dir and the runner UID is inside the trusted cluster boundary, so scope the
  # exemption globally (also reaches the submodule child git).
  git config --global --add safe.directory '*' >> "$log" 2>&1 || true
  if GIT_TERMINAL_PROMPT=0 git init -q "$temporary" > "$log" 2>&1 \
      && git -C "$temporary" remote add origin "$COLLX_DEEPEP_V2_REPO" >> "$log" 2>&1 \
      && GIT_TERMINAL_PROMPT=0 git -C "$temporary" fetch -q --no-tags --depth 1 \
        origin "$COLLX_DEEPEP_V2_COMMIT" >> "$log" 2>&1 \
      && git -C "$temporary" -c advice.detachedHead=false checkout -q --detach FETCH_HEAD \
        >> "$log" 2>&1 \
      && [ "$(git -C "$temporary" rev-parse HEAD)" = "$COLLX_DEEPEP_V2_COMMIT" ] \
      && GIT_TERMINAL_PROMPT=0 git -C "$temporary" submodule update -q --init --depth 1 \
        third-party/fmt >> "$log" 2>&1 \
      && python3 "$COLLX_RUNTIME_DIR/stage.py" rewrite-deepep-v2 \
        "$temporary/deep_ep/__init__.py" >> "$log" 2>&1 \
      && mv -- "$temporary" "$source" >> "$log" 2>&1; then
    return 0
  fi
  rm -rf -- "$temporary"
  collx_log "ERROR: DeepEP source preparation failed"
  collx_log_tail "$log"
  return 1
}

collx_materialize_deepep_source() {
  local destination="$1" source
  [ -n "${COLLX_BACKEND_SOURCE_ROOT:-}" ] || return 1
  source="$COLLX_BACKEND_SOURCE_ROOT/deepep-v2-$COLLX_DEEPEP_V2_COMMIT"
  [ -d "$source" ] || return 1
  rm -rf -- "$destination" && cp -R -- "$source" "$destination"
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
      h200-dgxc|b200-dgxc)
        COLLX_STAGE_DIR="$(collx_prepare_implicit_stage_base)" \
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
  # A concurrent leg of the same run holds this lock for its full import
  # (measured ~18 minutes for the 32 GB sglang squash on b300), so the wait
  # must outlast an import and a timeout must say so — the empty import log
  # would otherwise make this the only silent launcher death.
  flock -w 2700 "$lock_fd" 2>> "$log" \
    || { collx_log "ERROR: timed out waiting for the container import lock"
         collx_log_tail "$log"; return 1; }
  if unsquashfs -l "$sq" >/dev/null 2>&1; then
    collx_log "container squash ready"
  else
    collx_log "importing configured container image"
    rm -f "$sq" 2>> "$log" \
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
  # Run once per node because some clusters use node-local squash storage.
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
# Shared storage serializes the import; node-local storage imports in parallel.
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
    collx_log "ERROR: container import failed"
    collx_log_tail "$log"
    return 1
  fi
  printf '%s' "$sq"
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
  export COLLX_CUDA_CONTEXT_LOG="$log"
  srun --jobid="$job_id" --nodes="$nodes" --ntasks="$nodes" --ntasks-per-node=1 \
    --gres=gpu:"$gpus_per_node" --chdir=/tmp --input=all \
    --export="$(collx_host_exports)" python3 /dev/stdin cuda-context "$gpus_per_node" \
    < "$COLLX_RUNTIME_DIR/probe.py" >"$log" 2>&1
}

# Resolve the exact per-execution child before any copy starts, so the parent
# EXIT trap can remove an interrupted partial stage. The configured base must
# already exist on compute-visible storage and must not traverse symlinks.
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

# Stage only the public benchmark tree into the private execution child.
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

# Run one shard with one Slurm task per GPU on one or more nodes.
# Launchers provide only allocation/container policy through globals and
# COLLX_DISTRIBUTED_CONTAINER_ARGS; per-case benchmark inputs travel as run_ep.py
# argv decoded from the shard control (config.py case-args), never as env.
# shellcheck disable=SC2153
collx_run_shard() {
  local build_log expected_cases ci=0 failed_cases=0
  local runtime_log argv_file shard wrap
  local -a container_args ep_args
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

  shard="${COLLX_SHARD_FILE:-}"
  [ -f "$shard" ] || shard="$COLLX_DIR/$shard"
  [ -f "$shard" ] || collx_die "shard control is unavailable"
  expected_cases="$(python3 "$COLLX_RUNTIME_DIR/config.py" case-count "$shard")" \
    && [[ "$expected_cases" =~ ^[1-9][0-9]*$ ]] \
    || collx_die "could not enumerate shard cases"

  collx_log "shard backend preparation: bench=$COLLX_BENCH nodes=$NODES"
  build_log="$(collx_private_log_path backend-prepare)"
  if ! srun --jobid="$JOB_ID" --nodes="$NODES" --ntasks-per-node=1 --chdir=/tmp \
    --container-name="$container_name" --container-image="$SQUASH_FILE" \
    "${container_args[@]}" --export=ALL \
    bash /ix/experimental/CollectiveX/runtime/prepare_backend.sh \
    </dev/null >"$build_log" 2>&1; then
    collx_log "ERROR: backend preparation failed"
    collx_log_tail "$build_log"
    return 1
  fi

  argv_file="$(mktemp)" || return 1
  while [ "$ci" -lt "$expected_cases" ]; do
    python3 "$COLLX_RUNTIME_DIR/config.py" case-args "$shard" "$ci" \
      "$RUNNER" "$TS" \
      "$NGPUS" "$NODES" "$GPN" "$SCALE_UP_DOMAIN" > "$argv_file" \
      || { rm -f "$argv_file"; collx_die "shard case $ci does not decode against this allocation"; }
    mapfile -d '' -t ep_args < "$argv_file"
    [ "${#ep_args[@]}" -gt 0 ] \
      || { rm -f "$argv_file"; collx_die "case $ci produced no benchmark arguments"; }
    collx_log "EP${NGPUS}[$((ci + 1))/$expected_cases] $COLLX_BENCH"
    runtime_log="$(collx_private_log_path "runtime-c$(printf '%03d' "$ci")")"
    if ! timeout -k 30 "${COLLX_RUN_TIMEOUT:-900}" \
      srun --jobid="$JOB_ID" --nodes="$NODES" \
      --ntasks="$NGPUS" --ntasks-per-node="$GPN" --chdir=/tmp \
      --container-name="$container_name" --container-image="$SQUASH_FILE" \
      "${container_args[@]}" \
      --export=ALL \
      bash -c "$wrap" _ "${ep_args[@]}" \
      </dev/null >"$runtime_log" 2>&1; then
      collx_log "ERROR: case $ci failed"
      collx_log_tail "$runtime_log"
      failed_cases=$((failed_cases + 1))
    fi
    ci=$((ci + 1))
  done
  rm -f "$argv_file"
  [ "$failed_cases" = 0 ] || {
    collx_log "ERROR: $failed_cases/$expected_cases case(s) failed"
    return 1
  }
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
  local rc="$1" stage_root="${MOUNT_SRC:-}"
  trap - EXIT HUP INT TERM
  if [ -n "${JOB_ID:-}" ]; then
    collx_remove_distributed_container "$JOB_ID" "${NODES:-1}"
    if ! collx_cleanup_allocation; then
      [ "$rc" != 0 ] || rc=1
      exit "$rc"
    fi
  fi
  if [ -n "${REPO_ROOT:-}" ] && [ -n "$stage_root" ] \
      && [ "$stage_root" != "$REPO_ROOT" ]; then
    if [ "$rc" != 0 ] && [ -d "$stage_root/experimental/CollectiveX" ]; then
      collx_collect_results "$stage_root" "$REPO_ROOT" || true
    fi
    if ! collx_cleanup_stage "$stage_root" "$REPO_ROOT"; then
      [ "$rc" != 0 ] || rc=1
    fi
  fi
  exit "$rc"
}

collx_install_launcher_fail_safe() {
  trap 'collx_launcher_cleanup "$?"' EXIT
  trap 'collx_launcher_cleanup 129' HUP
  trap 'collx_launcher_cleanup 130' INT
  trap 'collx_launcher_cleanup 143' TERM
}
