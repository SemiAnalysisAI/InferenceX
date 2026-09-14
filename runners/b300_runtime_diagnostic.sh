#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../benchmarks/benchmark_lib.sh"
select_mooncake_rdma_device
export MOONCAKE_RAIL IBV_SHOW_WARNINGS=1
printf 'node=%s selected=%s gid=%s\n' "$(hostname)" "$MOONCAKE_RAIL" "$MC_GID_INDEX"
nvidia-smi --query-gpu=uuid,name --format=csv
printf 'CPU affinity: '; taskset -pc $$
printf 'RDMA libraries before recipe install:\n'
ldconfig -p | grep -E 'lib(ibverbs|mlx5|efa)'
for lib in /usr/lib/x86_64-linux-gnu/libibverbs/lib* /opt/amazon/efa/lib/libibverbs/lib*; do
    [[ -f "$lib" ]] || continue
    printf '%s -> %s\n' "$lib" "$(readlink -f "$lib")"
    ldd "$lib" 2>&1 | grep -E 'IBVERBS|not found|libibverbs' || true
done
agentic_pip_install --quiet --no-cache-dir --no-deps --force-reinstall mooncake-transfer-engine-cuda13==0.3.11.post1
python3 runners/b300_rdma_devices.py
