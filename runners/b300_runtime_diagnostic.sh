#!/usr/bin/env bash
# Mooncake found 0 HCAs on DSXE although the rail exists in sysfs. The enroot
# hooks bind the host libibverbs over the image's, so the image's mlx5 provider
# may no longer load; check that, then trial the host provider as the remedy.
set -uo pipefail
cd /workspace
source benchmarks/benchmark_lib.sh
export IBV_SHOW_WARNINGS=1
lib=/usr/lib/x86_64-linux-gnu
abi() { readelf -V "$1" 2>/dev/null | grep -oE 'IBVERBS_PRIVATE_[0-9]+' | sort -u | tr '\n' ' '; }

select_mooncake_rdma_device
export MOONCAKE_RAIL
echo "selected=$MOONCAKE_RAIL gid=$MC_GID_INDEX"
ls -l /dev/infiniband/ 2>&1
ulimit -l
echo "== /etc/libibverbs.d in the container"; ls -l /etc/libibverbs.d/; cat /etc/libibverbs.d/*.driver 2>/dev/null

echo "== hook mounts visible in the container"
grep -E ' /(usr/lib|lib|opt|etc)/[^ ]*(ibverbs|mlx5|efa|amazon|rdma|libfabric)' /proc/mounts || echo "no rdma-related bind mounts"
echo "== container libraries"
for f in "$lib/libibverbs.so.1" "$lib/libmlx5.so.1" "$lib/libefa.so.1"; do
    printf '%s -> %s : %s\n' "$f" "$(readlink -f "$f")" "$(abi "$f")"
done
ls -l "$lib/libibverbs/"
cat /etc/libibverbs.d/*.driver 2>/dev/null
for p in "$lib"/libibverbs/*.so; do printf '%s : %s\n' "$p" "$(abi "$p")"; done
ldd "$lib"/libibverbs/libmlx5-rdmav*.so 2>&1 | grep -E 'not found|ibverbs|mlx5'

echo "== mooncake wheel"
agentic_pip_install --no-cache-dir --no-deps --force-reinstall mooncake-transfer-engine-cuda13==0.3.11.post1 2>&1 | grep -vE '^\s*$' | tail -5
pkg=$(python3 -c 'import mooncake, os; print(os.path.dirname(mooncake.__file__))')
ls "$pkg" "$pkg/../mooncake.libs" 2>&1 | grep -iE 'ibverbs|mlx5|efa|rdma|\.so|libs' || echo "wheel bundles no rdma libraries"

echo "== enumerate before remedy"
python3 runners/b300_rdma_devices.py

echo "== host provider candidates"
for p in /host-usr-lib/libibverbs/libmlx5-rdmav*.so /host-opt-amazon/efa/lib/libibverbs/libmlx5-rdmav*.so; do
    [[ -e "$p" ]] && printf '%s : %s -> %s\n' "$p" "$(abi "$p")" "$(readlink -f "$p")"
done
ldd "$(readlink -f /host-usr-lib/libibverbs/libmlx5-rdmav*.so)" 2>&1 | grep -E 'not found|ibverbs|libnl' || true

# libibverbs appends its own -rdmavNN suffix to an absolute RDMAV_DRIVERS entry,
# so pointing at the host directory can only load the provider built with the
# libibverbs the hook mounted.
echo "== enumerate with RDMAV_DRIVERS=/host-usr-lib/libibverbs/libmlx5"
RDMAV_DRIVERS=/host-usr-lib/libibverbs/libmlx5 python3 runners/b300_rdma_devices.py
echo "== mooncake transfer engine on $MOONCAKE_RAIL with RDMAV_DRIVERS"
RDMAV_DRIVERS=/host-usr-lib/libibverbs/libmlx5 python3 runners/b300_rdma_devices.py --init "$MOONCAKE_RAIL"
echo "transfer engine rc=$?"

echo "== mooncake loopback bandwidth on $MOONCAKE_RAIL (recipe env)"
nvidia-smi -L 2>&1 | head -1
env RDMAV_DRIVERS=/host-usr-lib/libibverbs/libmlx5 MC_STORE_MEMCPY=1 MC_ENABLE_DEST_DEVICE_AFFINITY=1 \
    MC_SLICE_SIZE=1048576 MC_WORKERS_PER_CTX=4 WITH_NVIDIA_PEERMEM=0 \
    python3 runners/b300_mooncake_bandwidth.py "$MOONCAKE_RAIL" 2>&1 | grep -vE '^[IW][0-9]{4} .*(rpc_service|Metrics|InitGoogleLogging)'
echo "== mooncake loopback bandwidth on $MOONCAKE_RAIL (mooncake defaults)"
env RDMAV_DRIVERS=/host-usr-lib/libibverbs/libmlx5 \
    python3 runners/b300_mooncake_bandwidth.py "$MOONCAKE_RAIL" 2>&1 | grep -vE '^[IW][0-9]{4} .*(rpc_service|Metrics|InitGoogleLogging)'
