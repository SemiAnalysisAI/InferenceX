#!/usr/bin/env bash
# Minimal residual after #3610 removed the shared benchmark library.
# kimik3-b300-mooncake.sh (PR #3088) still sources this file with
# --validation-only for select_mooncake_rdma_device.

# Setup scripts source this library with --validation-only, so the Mooncake
# rail helper must stay above that gate.
select_mooncake_rdma_device() {
    local sysfs_root="${1:-/sys/class/infiniband}"
    local device
    MOONCAKE_RAIL=""
    for device in "$sysfs_root"/*; do
        # DSXE has both EFA and Mellanox adapters. The latter may be renamed
        # ibp*, so identify the driver rather than assuming an mlx5_* name.
        [[ "$(readlink "$device/device/driver" 2>/dev/null)" == */mlx5_core ]] || continue
        grep -qx '4: ACTIVE' "$device/ports/1/state" 2>/dev/null || continue
        case "$(cat "$device/ports/1/link_layer" 2>/dev/null)" in
            InfiniBand) MC_GID_INDEX=0 ;;
            Ethernet) MC_GID_INDEX=3 ;;
            *) continue ;;
        esac
        MOONCAKE_RAIL="${device##*/}"
        if [[ -z "$MOONCAKE_RAIL" || "$MOONCAKE_RAIL" == "*" ]]; then
            echo "Error: resolved an empty Mooncake RDMA rail name from $device" >&2
            return 1
        fi
        export MOONCAKE_RAIL
        export MC_GID_INDEX
        return 0
    done
    echo "Error: no active Mellanox RDMA rail; Mooncake cannot initialise" >&2
    return 1
}

# Launchers may load only input validation, without benchmark initialization.
if [[ "${1-}" == "--validation-only" ]]; then
    return 0
fi
