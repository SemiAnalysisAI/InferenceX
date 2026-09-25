# shellcheck shell=bash
# Network selectors and allocation fabric validation. Sourced by common.sh.

collx_export_gid_index_for_link_layer() {
  local link_layer="$1"
  unset NVSHMEM_IB_GID_INDEX NCCL_IB_GID_INDEX UCCL_IB_GID_INDEX
  [ -n "${COLLX_IB_GID_INDEX:-}" ] || return 0
  case "$link_layer" in
    roce)
      export NVSHMEM_IB_GID_INDEX="$COLLX_IB_GID_INDEX"
      export NCCL_IB_GID_INDEX="$COLLX_IB_GID_INDEX"
      # UCCL-EP reads only its own UCCL_IB_GID_INDEX (it does NOT consult NCCL_IB_GID_INDEX), so
      # RoCE runs must set it here or the CPU proxies fall back to GID 0 and mis-address the fabric.
      export UCCL_IB_GID_INDEX="$COLLX_IB_GID_INDEX"
      ;;
    infiniband|efa) ;;
    *) collx_die "unsupported RDMA link layer" ;;
  esac
}

# AWS EFA scale-out (b300-dsxe, p6-b300.48xlarge: 16 EFA devices per node). EFA is not a verbs
# HCA: its ports report link_layer Unspecified, it has no GID table, service level or traffic
# class, and NCCL reaches it only through the aws-ofi-nccl libfabric plugin, which the cluster's
# enroot hook bind-mounts into every container (/opt/amazon/{efa,ofi-nccl} on the ld path). GIN
# rides the plugin's ncclGinPlugin export (proxy or GDAKI; OFI_NCCL_GIN_TYPE selects). So the
# whole IB/RoCE selector family (NCCL_IB_*, NVSHMEM_HCA_LIST/IBGDA, MORI_*, UCCL_IB_*) stays
# unset: the plugin enumerates and rails the EFA devices itself, and the operator's rdma_devices
# list is consumed only by the network-profile probe as the set of ports that must be ACTIVE.
# NVSHMEM (deepep-v2) has its own libfabric transport; point it at EFA the same way.
collx_apply_efa_profile() {
  export NCCL_NET_PLUGIN=ofi
  export FI_PROVIDER=efa FI_EFA_FORK_SAFE=1
  export NVSHMEM_REMOTE_TRANSPORT=libfabric NVSHMEM_LIBFABRIC_PROVIDER=efa
  unset NVSHMEM_IB_ENABLE_IBGDA NVSHMEM_IBGDA_NIC_HANDLER NVSHMEM_HCA_LIST NVSHMEM_ENABLE_NIC_PE_MAPPING
}

# Selector values are interface/HCA identifiers, never addresses.
collx_apply_network_profile() {
  local nodes="$1" transport="$2"
  local selector rdma_name rdma_names="" ep_nic=""
  local -a selectors
  [[ "$nodes" =~ ^[1-9][0-9]*$ ]] || collx_die "invalid network placement"
  unset NCCL_NET NCCL_NET_PLUGIN NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME NCCL_IB_HCA
  unset NCCL_IB_GID_INDEX NCCL_IB_SL NCCL_IB_MERGE_NICS NCCL_CROSS_NIC
  unset NVSHMEM_ENABLE_NIC_PE_MAPPING
  unset NVSHMEM_HCA_LIST NVSHMEM_IB_GID_INDEX NVSHMEM_IB_SL
  unset NVSHMEM_IB_ENABLE_IBGDA NVSHMEM_IBGDA_NIC_HANDLER
  unset EP_NIC_NAME EP_OVERRIDE_RDMA_SL
  unset MORI_RDMA_DEVICES
  unset MORI_RDMA_TC MORI_IO_TC MORI_RDMA_SL MORI_IO_SL
  unset UCCL_SOCKET_IFNAME UCCL_IB_HCA UCCL_IB_GID_INDEX UCCL_IB_SL UCCL_IB_TC
  unset UCCL_IB_MAX_INFLIGHT_BYTES UCCL_IB_MAX_INFLIGHT_NORMAL UCCL_EP_ENABLE_AGGRESSIVE_ATOMIC
  # Single-node and MNNVL runs need only the scrub above. Single-node low-latency kernels run
  # over NVLink/XGMI (DeepEP allow_nvlink_for_low_latency_mode, MoRI IntraNodeLL) and must NOT
  # force IBGDA (/dev/gdrdrv is absent on h200). Exception: a SKU may pin a single-node HCA list.
  # DeepEP's legacy LL Buffer self-enables IBGDA even single-node, and on b300 the image's baked
  # NVSHMEM_HCA_PE_MAPPING steers that init onto the GPU-fabric RoCE rails, where ibv_create_ah
  # fails at every GID index; the storage-IB rails accept AH/DCT creation and carry init-time
  # traffic only. NVSHMEM_HCA_LIST wins over the baked mapping.
  if [ "$nodes" -le 1 ] && [ -n "${COLLX_SINGLE_NODE_RDMA_DEVICES:-}" ]; then
    [[ "$COLLX_SINGLE_NODE_RDMA_DEVICES" =~ ^[A-Za-z][A-Za-z0-9_.-]{0,31}(:[1-9][0-9]*)?(,[A-Za-z][A-Za-z0-9_.-]{0,31}(:[1-9][0-9]*)?)*$ ]] \
      || collx_die "invalid private single-node RDMA device selector"
    export NVSHMEM_HCA_LIST="$COLLX_SINGLE_NODE_RDMA_DEVICES"
  fi
  { [ "$nodes" -gt 1 ] && [ "$transport" != mnnvl ]; } || return 0
  [ -n "${COLLX_RDMA_DEVICES:-}" ] \
    || collx_die "RDMA execution requires a private device selector"
  [[ "${COLLX_RDMA_FABRIC:-}" =~ ^(efa)?$ ]] || collx_die "invalid private RDMA fabric"
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
  if [ "${COLLX_RDMA_FABRIC:-}" = efa ]; then
    collx_apply_efa_profile
    return 0
  fi
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
  # UCCL-EP (ep/src/rdma.cpp) honors NCCL's leading '=' exact-match and ':port' syntax; a bare
  # name list would prefix-match (mlx5_1 -> mlx5_10..19) and drop the port.
  export UCCL_IB_HCA="=$COLLX_RDMA_DEVICES"
  export UCCL_SOCKET_IFNAME="${COLLX_SOCKET_IFNAME:-}"
  if [ "${COLLX_VENDOR:-nvidia}" = amd ]; then
    export UCCL_IB_MAX_INFLIGHT_BYTES="${UCCL_IB_MAX_INFLIGHT_BYTES:-2097152}"
    export UCCL_IB_MAX_INFLIGHT_NORMAL="${UCCL_IB_MAX_INFLIGHT_NORMAL:-1}"
    export UCCL_EP_ENABLE_AGGRESSIVE_ATOMIC="${UCCL_EP_ENABLE_AGGRESSIVE_ATOMIC:-1}"
  fi
  # NCCL's default dual-port fusion collapses each card into one "fused" device, and any fused
  # device disables NCCL GIN (init.cc nicFused gate); the deep_ep EP16 hybrid path then asserts
  # railedGinType == NCCL_GIN_TYPE_NONE.
  export NCCL_IB_MERGE_NICS=0
  if [ -n "${COLLX_RAIL_ISOLATED:-}" ]; then
    [[ "$COLLX_RAIL_ISOLATED" =~ ^[01]$ ]] \
      || collx_die "invalid private rail isolation flag"
    # On rail-isolated fabrics (per-port subnets, no cross-rail routing) cross-NIC pairs
    # black-hole at QP RTR.
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
    export UCCL_IB_SL="$COLLX_RDMA_SERVICE_LEVEL"
  fi
  if [ -n "${COLLX_RDMA_TRAFFIC_CLASS:-}" ]; then
    [[ "$COLLX_RDMA_TRAFFIC_CLASS" =~ ^[0-9]+$ ]] && [ "$COLLX_RDMA_TRAFFIC_CLASS" -le 255 ] \
      || collx_die "invalid private RDMA traffic class"
    export MORI_RDMA_TC="$COLLX_RDMA_TRAFFIC_CLASS" MORI_IO_TC="$COLLX_RDMA_TRAFFIC_CLASS"
    export UCCL_IB_TC="$COLLX_RDMA_TRAFFIC_CLASS"
  fi
  local nic_handler=gpu
  export NVSHMEM_IB_ENABLE_IBGDA=1 NVSHMEM_IBGDA_NIC_HANDLER="$nic_handler"
  if [ -n "${COLLX_RDMA_LINK_LAYER:-}" ]; then
    case "$COLLX_RDMA_LINK_LAYER" in
      roce|infiniband|efa) ;;
      *) collx_die "invalid validated RDMA link layer" ;;
    esac
    collx_export_gid_index_for_link_layer "$COLLX_RDMA_LINK_LAYER"
  fi
}

# Slurm may strip NCCL_IB_HCA's leading '=' exact-match marker while propagating the
# environment; rebuild it at the container boundary rather than accept a prefix-matched list.
collx_restore_exact_hca_selector() {
  if [ "${COLLX_NODES:-1}" -le 1 ] || [ "${COLLX_TRANSPORT:-}" = mnnvl ]; then
    return 0
  fi
  [ -n "${COLLX_RDMA_DEVICES:-}" ] \
    || { collx_log "ERROR: scale-out RDMA selector is unavailable"; return 1; }
  [[ "$COLLX_RDMA_DEVICES" =~ ^[A-Za-z][A-Za-z0-9_.-]{0,31}(:[1-9][0-9]*)?(,[A-Za-z][A-Za-z0-9_.-]{0,31}(:[1-9][0-9]*)?)*$ ]] \
    || { collx_log "ERROR: invalid scale-out RDMA selector"; return 1; }
  # EFA devices are not verbs HCAs to NCCL: the libfabric plugin enumerates them itself and
  # NCCL_IB_HCA would only steer the (unused) builtin IB transport. Leave it unset there.
  [ "${COLLX_RDMA_FABRIC:-}" = efa ] || export NCCL_IB_HCA="=$COLLX_RDMA_DEVICES"
}

collx_default_route_interface() {
  python3 "$COLLX_RUNTIME_DIR/probe.py" default-route-interface
}

# Selector values and node diagnostics stay in the runner-private log.
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
      "$COLLX_RDMA_DEVICES" "${COLLX_IB_GID_INDEX:-}" "${COLLX_RDMA_FABRIC:-}" \
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
    sed -nE 's/^\[collectivex-private\] rdma-link-layer=(roce|infiniband|efa)$/\1/p' "$log" \
      | sort -u
  )"
  marker_count="$(grep -Ec '^\[collectivex-private\] rdma-link-layer=(roce|infiniband|efa)$' "$log")"
  case "$marker_count:$link_layer" in
    "$nodes":roce|"$nodes":infiniband|"$nodes":efa) ;;
    *) return 1 ;;
  esac
  export COLLX_RDMA_LINK_LAYER="$link_layer"
  collx_export_gid_index_for_link_layer "$link_layer"
}
