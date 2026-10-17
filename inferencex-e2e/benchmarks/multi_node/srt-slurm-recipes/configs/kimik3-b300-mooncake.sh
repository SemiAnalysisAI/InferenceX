#!/usr/bin/env bash
# Install the worker's Mooncake client and point its store at one active RDMA rail.
set -eo pipefail
bash "$(dirname "${BASH_SOURCE[0]}")/vllm-mooncake.sh"

# Rail-isolated nodes: two RNICs cannot reach each other even within a node, so
# every rank uses one rail. mlx5_0 is down on some nodes, and topology discovery
# then finds no HCA, so take the first active rail at runtime. DSXE nodes name
# their rails rdmap*.
rail=""
for device in mlx5_0 mlx5_1 mlx5_2 mlx5_3 mlx5_4 mlx5_5 mlx5_8 mlx5_9 \
              mlx5_10 mlx5_11 mlx5_16 mlx5_17 mlx5_20 mlx5_21 mlx5_22 mlx5_23 \
              $(ls /sys/class/infiniband 2>/dev/null | grep '^rdmap' | sort -V); do
    if grep -q ACTIVE "/sys/class/infiniband/$device/ports/1/state" 2>/dev/null; then
        rail="$device"
        break
    fi
done
if [[ -z "$rail" ]]; then
    echo "Error: no active RDMA rail on $(hostname); Mooncake cannot initialise" >&2
    for state in /sys/class/infiniband/*/ports/*/state; do
        echo "$state: $(cat "$state" 2>&1)" >&2
    done
    exit 1
fi
config="${MOONCAKE_CONFIG_PATH:-/logs/mooncake_store_config.json}"
python3 - "$config" "$rail" <<'PY'
import json, sys
path, rail = sys.argv[1:]
with open(path) as handle:
    config = json.load(handle)
config["device_name"] = rail
with open(path, "w") as handle:
    json.dump(config, handle, indent=2)
PY
echo "Mooncake rail: $rail"
