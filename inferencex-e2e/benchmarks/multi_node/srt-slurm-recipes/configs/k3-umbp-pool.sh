#!/usr/bin/env bash
set -eo pipefail
if [[ -z ${UMBP_ENDPOINT-} ]]; then
    exit 0
fi
source /infmax-workspace/benchmarks/check_env.sh
check_env_vars HEAD_NODE_IP NCCL_SOCKET_IFNAME NCCL_IB_HCA UMBP_ENDPOINT UMBP_DRAM_CAPACITY

# The pool shares the engine container PID/IPC namespaces for HIP IPC.
export UMBP_MASTER_ADDRESS="${HEAD_NODE_IP}:25190"
export UMBP_NODE_ID="$(hostname -s)"
export UMBP_NODE_ADDRESS
UMBP_NODE_ADDRESS=$(python3 - "$NCCL_SOCKET_IFNAME" <<'PY'
import fcntl
import socket
import struct
import sys

interface = sys.argv[1].encode()
if not 0 < len(interface) < 16:
    raise ValueError("Expected one Linux interface name")
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
    # Linux SIOCGIFADDR returns the primary IPv4 address in struct ifreq.
    address = fcntl.ioctl(sock.fileno(), 0x8915, struct.pack("256s", interface))
print(socket.inet_ntoa(address[20:24]))
PY
)
export UMBP_IO_ENGINE_HOST="$UMBP_NODE_ADDRESS"
export UMBP_IO_ENGINE_PORT=25192
export UMBP_PEER_SERVICE_PORT=25191
export IBDEVICES="$NCCL_IB_HCA"

# Fail before startup if the reserved hugepages cannot hold the pool.
page_kb=$(awk '/^Hugepagesize:/ {print $2}' /proc/meminfo)
free_pages=$(awk '/^HugePages_Free:/ {print $2}' /proc/meminfo)
reserved_pages=$(awk '/^HugePages_Rsvd:/ {print $2}' /proc/meminfo)
(( (free_pages - reserved_pages) * page_kb * 1024 >= UMBP_DRAM_CAPACITY ))
[[ ! -e $UMBP_ENDPOINT ]]
mkdir -p "$(dirname "$UMBP_ENDPOINT")"
server=/usr/local/lib/python3.12/dist-packages/mori/umbp_standalone_server
log="/logs/umbp-${UMBP_NODE_ID}.log"
"$server" "unix://$UMBP_ENDPOINT" > "$log" 2>&1 &
server_pid=$!
printf '%s\n' "$server_pid" > "/logs/umbp-${UMBP_NODE_ID}.pid"
for ((attempt=0; attempt<900; attempt++)); do
    kill -0 "$server_pid"
    if [[ -S $UMBP_ENDPOINT ]]; then
        exit 0
    fi
    sleep 1
done
echo "UMBP server readiness timed out; inspect $log" >&2
kill -TERM "$server_pid"
exit 1
