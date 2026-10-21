#!/usr/bin/env bash
set -eo pipefail

# Rail-isolated nodes: two RNICs cannot reach each other even within a node, so
# every rank uses one rail, the first ACTIVE InfiniBand port that mlx5_core drives.
# DSXE names its ConnectX-7 ports ibp*, and its EFA rdmap* devices have no RC queue
# pairs for Mooncake's rdma transport (segment registration fails with ENOMEM).
select_mooncake_rdma_device() {
    local device
    MOONCAKE_RAIL=""
    for device in "$1"/*; do
        [[ "$(readlink "$device/device/driver" 2>/dev/null)" == */mlx5_core ]] || continue
        [[ "$(cat "$device/ports/1/state" 2>/dev/null)" == "4: ACTIVE" ]] || continue
        [[ "$(cat "$device/ports/1/link_layer" 2>/dev/null)" == InfiniBand ]] || continue
        MOONCAKE_RAIL="${device##*/}"
        return 0
    done
    echo "Error: no ACTIVE InfiniBand mlx5_core port under $1; Mooncake cannot initialise" >&2
    for device in "$1"/*; do
        echo "  ${device##*/}: driver=$(readlink "$device/device/driver" 2>&1)" \
            "state=$(cat "$device/ports/1/state" 2>&1)" \
            "link_layer=$(cat "$device/ports/1/link_layer" 2>&1)" >&2
    done
    return 1
}

# libibverbs loads "$RDMAV_DRIVERS-rdmav<ABI>.so". rdma-core installs that file as a
# relative link to ../libmlx5.so.1.*, so it resolves only when the provider's parent
# directory is mounted too; -f follows the link.
check_mooncake_verbs_provider() {
    local providers=("$RDMAV_DRIVERS"-rdmav*.so)
    if [[ ! -f "${providers[0]}" ]]; then
        echo "Error: no verbs provider resolves for $RDMAV_DRIVERS-rdmav*.so" \
            "(${providers[0]} -> $(readlink "${providers[0]}"))" >&2
        return 1
    fi
    MOONCAKE_VERBS_PROVIDER="${providers[0]} -> $(readlink -f "${providers[0]}")"
}

# Sourcing only defines the functions.
[[ "${BASH_SOURCE[0]}" == "$0" ]] || return 0

source /infmax-workspace/benchmarks/check_env.sh
check_env_vars KV_OFFLOAD_BACKEND_VERSION MOONCAKE_CONFIG_PATH MC_GID_INDEX RDMAV_DRIVERS
# An InfiniBand port addresses peers by GID index 0.
if [[ "$MC_GID_INDEX" != 0 ]]; then
    echo "Error: MC_GID_INDEX=$MC_GID_INDEX; the InfiniBand Mooncake rail needs GID index 0" >&2
    exit 1
fi
check_mooncake_verbs_provider
select_mooncake_rdma_device /sys/class/infiniband
bash "$(dirname "${BASH_SOURCE[0]}")/vllm-mooncake.sh"

python3 - "$MOONCAKE_CONFIG_PATH" "$MOONCAKE_RAIL" <<'PY'
import json, sys
path, rail = sys.argv[1:]
with open(path) as handle:
    config = json.load(handle)
config["device_name"] = rail
with open(path, "w") as handle:
    json.dump(config, handle, indent=2)
PY
echo "Mooncake rail: $MOONCAKE_RAIL (InfiniBand, MC_GID_INDEX=$MC_GID_INDEX);" \
    "host verbs provider $MOONCAKE_VERBS_PROVIDER; image mlx5 provider" \
    /usr/lib/x86_64-linux-gnu/libibverbs/libmlx5-rdmav*.so
