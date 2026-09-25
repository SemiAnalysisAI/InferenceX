#!/usr/bin/env bash
# Prepare one backend per allocated node and persist its rank environment.
set -eo pipefail

cd /ix/experimental/CollectiveX
# shellcheck source=../runtime/common.sh
source runtime/common.sh

: "${COLLX_RUNNER:?COLLX_RUNNER not set}"
: "${COLLX_BENCH:?COLLX_BENCH not set}"

collx_log "backend preparation: runner=$COLLX_RUNNER bench=$COLLX_BENCH nodes=${COLLX_NODES:-1}"

# Fresh rank tasks source only these backend-created values. Network variables
# are reapplied by the rank wrapper from the platform profile.
readonly -a RANK_ENV_VARS=(
  PATH VIRTUAL_ENV LD_LIBRARY_PATH PYTHONPATH CUDA_HOME CPATH NVCC_PREPEND_FLAGS
  NVSHMEM_DIR EP_NCCL_ROOT_DIR EP_NVSHMEM_ROOT_DIR EP_JIT_CACHE_DIR
  EP_REUSE_NCCL_COMM NCCL_CUMEM_ENABLE UCCL_EP_ENABLE_AGGRESSIVE_ATOMIC
)
readonly -a DEEPEP_RANK_UNSETS=(EP_SUPPRESS_NCCL_CHECK)

source "$COLLX_RUNTIME_DIR/build_common.sh"
source "$COLLX_RUNTIME_DIR/backends/deepep.sh"
source "$COLLX_RUNTIME_DIR/backends/uccl.sh"
source "$COLLX_RUNTIME_DIR/backends/nccl.sh"

# container boundary

write_rank_env() {
  local root="$PWD/.collx_backend/env" node_id="${SLURM_NODEID:-0}" path temporary name
  [[ "$node_id" =~ ^[0-9]+$ ]] || return 1
  mkdir -p "$root" || return 1
  chmod 700 "$root" || return 1
  temporary="$(mktemp "$root/.node-${node_id}.XXXXXX")" || return 1
  chmod 600 "$temporary" || { rm -f "$temporary"; return 1; }
  for name in "${RANK_ENV_VARS[@]}"; do
    if declare -p "$name" >/dev/null 2>&1; then
      printf 'export %s=%q\n' "$name" "${!name}" >> "$temporary" \
        || { rm -f "$temporary"; return 1; }
    fi
  done
  if [ "$COLLX_BENCH" = deepep-v2 ]; then
    for name in "${DEEPEP_RANK_UNSETS[@]}"; do
      printf 'unset %s\n' "$name" >> "$temporary" \
        || { rm -f "$temporary"; return 1; }
    done
  fi
  path="$root/node-${node_id}.sh"
  mv -f -- "$temporary" "$path" || { rm -f "$temporary"; return 1; }
}

validate_container_network() {
  local interface device rdma_name
  local -a interfaces devices
  if [ "${COLLX_NODES:-1}" -le 1 ] || [ "${COLLX_TRANSPORT:-}" = mnnvl ]; then
    return 0
  fi
  collx_restore_exact_hca_selector || return 1
  local rdma_selector="${NCCL_IB_HCA:-}"
  if [ "${COLLX_RDMA_FABRIC:-}" = efa ]; then
    # No verbs selector on EFA; the probe-validated device list is the contract, and the
    # libfabric NCCL plugin the enroot hook mounts is what actually carries the traffic.
    rdma_selector="${COLLX_RDMA_DEVICES:-}"
    [ -r /opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so ] \
      || { collx_log "ERROR: aws-ofi-nccl plugin is absent inside the container"; return 1; }
  fi
  [ -n "${GLOO_SOCKET_IFNAME:-}" ] && [ -n "$rdma_selector" ] \
    || { collx_log "ERROR: scale-out network selectors are unavailable"; return 1; }
  IFS=, read -r -a interfaces <<< "$GLOO_SOCKET_IFNAME"
  for interface in "${interfaces[@]}"; do
    [ -d "/sys/class/net/$interface" ] \
      || { collx_log "ERROR: configured scale-out socket interface is absent"; return 1; }
  done
  IFS=, read -r -a devices <<< "$rdma_selector"
  for device in "${devices[@]}"; do
    device="${device#=}"
    rdma_name="${device%%:*}"
    [ -d "/sys/class/infiniband/$rdma_name" ] \
      || { collx_log "ERROR: configured scale-out RDMA device is absent"; return 1; }
  done
}

# The pinned SGLang images ship flashinfer-python with the one-sided MoE all-to-all, so this is a
# capability assert, not an install: fail early rather than mid-case inside create_buffer.
flashinfer_ep_prepare() {
  command -v python3 >/dev/null \
    || { collx_log "ERROR: python3 unavailable for FlashInfer EP"; return 1; }
  python3 - <<'FICHECK'
import sys
try:
    import flashinfer
    from flashinfer.comm import Mapping  # noqa: F401
    from flashinfer.comm.mnnvl import MnnvlConfig  # noqa: F401
    from flashinfer.comm.trtllm_moe_alltoall import (  # noqa: F401
        MoeAlltoAll,
        moe_a2a_get_workspace_size_per_rank,
    )
except Exception as exc:  # noqa: BLE001 - the reason belongs in the leg log
    print(f"flashinfer one-sided a2a import failed: {exc}", file=sys.stderr)
    raise SystemExit(1)
print(f"FlashInfer {getattr(flashinfer, '__version__', 'unknown')} one-sided A2A available")
FICHECK
  local rc=$?
  [ "$rc" -eq 0 ] || { collx_log "ERROR: FlashInfer EP one-sided A2A unavailable in this image"; return 1; }
}

main() {
  collx_apply_network_profile "${COLLX_NODES:-1}" "${COLLX_TRANSPORT:-}" || return 1
  validate_container_network || return 1
  case "$COLLX_BENCH" in
    deepep-v2) deepep_prepare || return 1 ;;
    mori)
      python3 -c "import mori" \
        || { collx_log "ERROR: MoRI backend import failed"; return 1; }
      ;;
    uccl-ep) uccl_prepare || return 1 ;;
    nccl-ep) nccl_ep_prepare || return 1 ;;
    flashinfer-ep) flashinfer_ep_prepare || return 1 ;;
    *)
      collx_log "ERROR: unknown backend preparation request"
      return 1
      ;;
  esac
  write_rank_env
}

rc=0; main || rc=$?
collx_log "backend preparation: bench=$COLLX_BENCH rc=$rc"
exit "$rc"
