#!/usr/bin/env bash
# Pin the worker's Mooncake client, backport load-failure recovery, and point
# the store at one active Mellanox RDMA rail (driver-selected, including ibp*).
set -eo pipefail

ws=/infmax-workspace
# Temporary upstream #55297 backport; docs/waiver/3088.md is pending review.
python3 "$ws/runners/patch_kimik3_mooncake_recovery.py"

pip_install=(python3 -m pip install)
if python3 -m pip install --help 2>/dev/null | grep -q -- --break-system-packages; then
    pip_install+=(--break-system-packages)
fi
"${pip_install[@]}" --quiet --no-cache-dir --no-deps --force-reinstall \
    mooncake-transfer-engine-cuda13==0.3.11.post1
python3 -c "from mooncake.store import MooncakeDistributedStore" >/dev/null

# Rail-isolated nodes: two RNICs cannot reach each other even within a node, so
# every rank uses one Mellanox rail. Identify by driver (mlx5_core) rather than
# assuming mlx5_* names — DSXE may rename them ibp*. EFA rails are skipped.
# shellcheck source=/dev/null
source "$ws/benchmarks/benchmark_lib.sh" --validation-only
select_mooncake_rdma_device
rail="$MOONCAKE_RAIL"
if [[ -z "$rail" ]]; then
    echo "Error: select_mooncake_rdma_device returned an empty rail" >&2
    exit 1
fi
echo "Mooncake rail: $rail (MC_GID_INDEX=$MC_GID_INDEX)"

# The enroot EFA hook binds the host libibverbs over the image's, and
# libibverbs only loads providers built against its own private ABI, so the
# image's mlx5 provider never loads and the Mellanox rail vanishes.
# runners.yaml mounts the host library directory at /host-usr-lib; libibverbs
# appends its own -rdmavNN suffix to an absolute RDMAV_DRIVERS entry.
if [[ ! -d /host-usr-lib/libibverbs ]]; then
    echo "Error: /host-usr-lib/libibverbs missing; host mlx5 provider required" >&2
    exit 1
fi
export RDMAV_DRIVERS=/host-usr-lib/libibverbs/libmlx5

config="${MOONCAKE_CONFIG_PATH:-/logs/mooncake_store_config.json}"
python3 - "$config" "$rail" <<'PY'
import json, sys
path, rail = sys.argv[1:]
if not rail.strip():
    raise SystemExit("Error: refusing to write empty Mooncake device_name")
with open(path) as handle:
    config = json.load(handle)
config["device_name"] = rail
with open(path, "w") as handle:
    json.dump(config, handle, indent=2)
written = json.load(open(path))
if not str(written.get("device_name") or "").strip():
    raise SystemExit(f"Error: mooncake store config still has empty device_name: {written!r}")
print(f"Patched mooncake_store_config device_name={written['device_name']!r}")
PY
