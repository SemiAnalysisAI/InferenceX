#!/usr/bin/env bash
# Install the worker's Mooncake client and point its store at one active RDMA rail.
set -eo pipefail
bash "$(dirname "${BASH_SOURCE[0]}")/vllm-mooncake.sh"

# Rail-isolated nodes: two RNICs cannot reach each other even within a node, so
# every rank uses one rail. mlx5_0 is down on some nodes, and topology discovery
# then finds no HCA, so take the first active rail at runtime. DSXE nodes name
# their rails rdmap*.
# Mooncake's rdma transport opens RC queue pairs, which EFA (rdmap*) devices reject. This row
# (+x1lr-rail-ib) runs only on an ACTIVE, LinkUp InfiniBand HCA and refuses to start without one
# instead of falling back to another rail.
rail=""
for device in $(ls /sys/class/infiniband 2>/dev/null | sort -V); do
    port="/sys/class/infiniband/$device/ports/1"
    layer=$(cat "$port/link_layer" 2>/dev/null || echo absent)
    state=$(cat "$port/state" 2>/dev/null || echo absent)
    phys=$(cat "$port/phys_state" 2>/dev/null || echo absent)
    echo "Mooncake rail candidate: $device link_layer=$layer state=$state phys_state=$phys"
    if [[ -z "$rail" && "$layer" == InfiniBand && "$state" == *ACTIVE* && "$phys" == *LinkUp* ]]; then
        rail="$device"
    fi
done
if [[ -z "$rail" ]]; then
    echo "Error: no ACTIVE LinkUp InfiniBand HCA on $(hostname); the +x1lr-rail-ib row does not fall back to EFA" >&2
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
