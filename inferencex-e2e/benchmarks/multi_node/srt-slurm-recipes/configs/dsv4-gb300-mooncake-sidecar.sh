#!/usr/bin/env bash
# On decode nodes, start standalone Mooncake store daemons that lend host DRAM
# to the store. No-op on roles that do not set MC_SIDECAR_SEGMENT_SIZE.
set -euo pipefail

# Decode ranks use a chunk cache, so their host memory never joins the store on
# its own. Standalone daemons on the decode nodes mount it instead; SGLang never
# talks to them, only the master sees their segments. Only roles that set
# MC_SIDECAR_SEGMENT_SIZE start them.
if [ -z "${MC_SIDECAR_SEGMENT_SIZE:-}" ]; then
    exit 0
fi
for var in MOONCAKE_MASTER MOONCAKE_TE_META_DATA_SERVER; do
    if [ -z "${!var:-}" ]; then
        echo "ERROR: ${var} unset; cannot start Mooncake store daemons" >&2
        exit 1
    fi
done

MY_IP=$(hostname -i | awk '{print $1}')
HOST=$(hostname)
COUNT="${MC_SIDECAR_COUNT:-1}"
echo "[mc-sidecar] host=${MY_IP} master=${MOONCAKE_MASTER} count=${COUNT} size=${MC_SIDECAR_SEGMENT_SIZE}"

# One daemon per rank-equivalent keeps segment sizes uniform across the pool; a
# single oversized segment would concentrate every put on this node's NICs.
PIDS=()
for i in $(seq 0 $((COUNT - 1))); do
    PORT=$((50052 + i))
    # Writers resolve a segment's transfer-engine descriptor through the HTTP
    # metadata server. A daemon on P2PHANDSHAKE mounts fine, but every remote
    # open of its segment 404s and the puts placed there are revoked.
    nohup mooncake_client \
        --host="${MY_IP}" \
        --port="${PORT}" \
        --master_server_address="${MOONCAKE_MASTER}" \
        --metadata_server="${MOONCAKE_TE_META_DATA_SERVER}" \
        --protocol="${MOONCAKE_PROTOCOL:-rdma}" \
        --device_names="${MOONCAKE_DEVICE:-}" \
        --global_segment_size="${MC_SIDECAR_SEGMENT_SIZE}" \
        --local_buffer_size=1GB \
        > "/logs/mc_sidecar_${HOST}_${PORT}.out" 2>&1 &
    PIDS+=($!)
done

mounted=0
for _ in $(seq 1 180); do
    mounted=$( (grep -l "Starting real client service" /logs/mc_sidecar_"${HOST}"_*.out 2>/dev/null || true) | wc -l)
    [ "${mounted}" -ge "${COUNT}" ] && break
    sleep 1
done
if [ "${mounted}" -lt "${COUNT}" ]; then
    echo "ERROR: only ${mounted}/${COUNT} Mooncake store daemons serving within 180s" >&2
    exit 1
fi

# A daemon has exited shortly after startup before; require all to survive.
sleep 20
for pid in "${PIDS[@]}"; do
    if ! kill -0 "${pid}" 2>/dev/null; then
        echo "ERROR: Mooncake store daemon pid ${pid} exited after startup" >&2
        exit 1
    fi
done

# The descriptor must resolve exactly the way remote writers look it up. The
# segment name is the transfer-engine host:port, not the RPC listen port.
for f in /logs/mc_sidecar_"${HOST}"_*.out; do
    ep=$(grep -aoE "parseHostNameWithPort\. server_name: [0-9.]+ port: [0-9]+" "${f}" | head -1 | awk '{print $3 ":" $5}' || true)
    key="mooncake%2Fram%2F${ep%:*}%3A${ep#*:}"
    code=$(curl -s -o /dev/null -w '%{http_code}' "${MOONCAKE_TE_META_DATA_SERVER}?key=${key}" || true)
    if [ -z "${ep}" ] || [ "${code}" != "200" ]; then
        echo "ERROR: descriptor for '${ep}' not on metadata server (http=${code})" >&2
        exit 1
    fi
done
echo "[mc-sidecar] ${COUNT} daemons serving and resolvable"
