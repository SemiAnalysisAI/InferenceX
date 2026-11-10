#!/usr/bin/env bash
set -eo pipefail
# srtctl also invokes this hook in the router container. Only engine workers
# receive HEAD_NODE_IP; the router must not start a standalone pool.
if [[ -z ${HEAD_NODE_IP:-} ]]; then
    echo "Skipping UMBP pool setup outside an engine worker"
    exit 0
fi
source /infmax-workspace/benchmarks/check_env.sh
check_env_vars MORI_RDMA_TC MORI_IO_TC MORI_IO_TC_DISABLE HEAD_NODE_IP NCCL_SOCKET_IFNAME NCCL_IB_HCA UMBP_ENDPOINT UMBP_DRAM_CAPACITY UMBP_DISTRIBUTED_DRAM_PAGE_SIZE UMBP_DRAM_USE_HUGEPAGES UMBP_DRAM_HUGEPAGE_SIZE UMBP_DRAM_NUMA_NODE UMBP_DRAM_MAX_MR_BYTES UMBP_POOL_MAX_RD_ATOMIC
[[ $UMBP_POOL_MAX_RD_ATOMIC == 1 ]]
/opt/runtime-venv/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path
import mori

receipt = json.loads(Path('/opt/k31-readcap-provenance.json').read_text())
library = Path(mori.__file__).parent / 'libmori_application.so'
expected = receipt['libraries'][str(library)]
actual = hashlib.sha256(library.read_bytes()).hexdigest()
if actual != expected or receipt['upstream_read_cap_commit'] != 'e65d1a3e59d6a700522054dc6edd443b94a3f200':
    raise RuntimeError('UMBP READ cap library does not match the pinned image provenance')
print({'umbp_read_cap_library': str(library), 'sha256': actual})
PY
printf 'RDMA traffic class: MORI_RDMA_TC=%s MORI_IO_TC=%s MORI_IO_TC_DISABLE=%s\n' "$MORI_RDMA_TC" "$MORI_IO_TC" "$MORI_IO_TC_DISABLE"
[[ $MORI_RDMA_TC == 104 && $MORI_IO_TC == 104 && $MORI_IO_TC_DISABLE == 0 ]]
export UMBP_MASTER_ADDRESS="${HEAD_NODE_IP}:25190"
export UMBP_NODE_ID="$(hostname -s)"
export UMBP_NODE_ADDRESS
UMBP_NODE_ADDRESS=$(/opt/runtime-venv/bin/python - "$NCCL_SOCKET_IFNAME" <<'PY'
import fcntl
import socket
import struct
import sys

interface = sys.argv[1].encode()
if not 0 < len(interface) < 16:
    raise ValueError('Expected one Linux interface name')
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
    address = fcntl.ioctl(sock.fileno(), 0x8915, struct.pack('256s', interface))
print(socket.inet_ntoa(address[20:24]))
PY
)
export UMBP_IO_ENGINE_HOST="$UMBP_NODE_ADDRESS"
export UMBP_IO_ENGINE_PORT=25300
export UMBP_PEER_SERVICE_PORT=25200
export UMBP_STANDALONE_ADDRESS="unix://$UMBP_ENDPOINT"
export IBDEVICES="$NCCL_IB_HCA"
[[ $UMBP_DRAM_USE_HUGEPAGES == 1 && $UMBP_DRAM_HUGEPAGE_SIZE == 2097152 ]]
(( UMBP_DRAM_MAX_MR_BYTES > 0 && UMBP_DRAM_MAX_MR_BYTES <= 549755813888 ))
page_kb=$(awk '/^Hugepagesize:/ {print $2}' /proc/meminfo)
free_pages=$(awk '/^HugePages_Free:/ {print $2}' /proc/meminfo)
reserved_pages=$(awk '/^HugePages_Rsvd:/ {print $2}' /proc/meminfo)
(( page_kb * 1024 == UMBP_DRAM_HUGEPAGE_SIZE ))
(( (free_pages - reserved_pages) * page_kb * 1024 >= UMBP_DRAM_CAPACITY ))
IFS=',' read -r -a numa_nodes <<< "$UMBP_DRAM_NUMA_NODE"
for numa_node in "${numa_nodes[@]}"; do
    [[ $numa_node =~ ^[0-9]+$ ]]
    read -r numa_free < "/sys/devices/system/node/node${numa_node}/hugepages/hugepages-2048kB/free_hugepages"
    (( numa_free * 2097152 >= (UMBP_DRAM_CAPACITY + ${#numa_nodes[@]} - 1) / ${#numa_nodes[@]} ))
done
[[ ! -e $UMBP_ENDPOINT ]]
mkdir -p "$(dirname "$UMBP_ENDPOINT")"
log_dir=$(mktemp -d /private-state/umbp-pool-XXXXXX)
(
    # Slurm can lower soft memlock even when the container hard limit is unlimited.
    ulimit -Sl "$(ulimit -Hl)"
    printf 'Pool memlock KiB: soft=%s hard=%s\n' "$(ulimit -Sl)" "$(ulimit -Hl)"
    exec env -u MORI_IO_RDMA_NIC_IDX MORI_IO_QP_MAX_RD_ATOMIC="$UMBP_POOL_MAX_RD_ATOMIC" MORI_APP_LOG_LEVEL=info /usr/local/bin/umbp_standalone_server "$UMBP_STANDALONE_ADDRESS"
) > "$log_dir/server.log" 2>&1 &
server_pid=$!
tail --pid="$server_pid" -n +1 -F "$log_dir/server.log" &
printf '%s\n' "$server_pid" > "$log_dir/server.pid"
for ((attempt=0; attempt<900; attempt++)); do
    kill -0 "$server_pid"
    if [[ -S $UMBP_ENDPOINT ]]; then
        cat "/proc/$server_pid/limits" > "$log_dir/server.limits"
        cat "$log_dir/server.limits"
        if grep -Eq 'falling back to.*4 KiB|4096-byte pages' "$log_dir/server.log"; then
            echo "UMBP fell back from HugeTLB; inspect $log_dir/server.log" >&2
            kill -TERM "$server_pid"
            exit 1
        fi
        if ! grep -q '2097152-byte pages' "$log_dir/server.log"; then
            echo "UMBP did not confirm 2 MiB backing; inspect $log_dir/server.log" >&2
            kill -TERM "$server_pid"
            exit 1
        fi
        echo "UMBP pool ready at $UMBP_ENDPOINT; log: $log_dir/server.log"
        PYTHONPATH=/infmax-workspace /opt/runtime-venv/bin/python -m infx.bench umbp-health --pool-state "$UMBP_ENDPOINT.health.json" --record-pid "$server_pid"
        exit 0
    fi
    sleep 1
done
echo "UMBP server readiness timed out; inspect $log_dir/server.log" >&2
kill -TERM "$server_pid"
exit 1
