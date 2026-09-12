#!/usr/bin/env bash
# Diagnostic only: no weights, model server, package install, or benchmark.
set -uo pipefail
printf 'container_diagnostic_start=%s\n' "$(date -u --iso-8601=seconds)"
printf 'LD_LIBRARY_PATH=%s\n' "${LD_LIBRARY_PATH:-}"
printf 'NVIDIA_VISIBLE_DEVICES=%s\n' "${NVIDIA_VISIBLE_DEVICES:-}"
cat /etc/os-release
nvidia-smi --query-gpu=index,uuid,name,driver_version --format=csv
ls -l /sys/class/infiniband/ || true
for d in /sys/class/infiniband/*; do
    [[ -e "$d" ]] || continue
    printf '%s driver=' "$d"; readlink -f "$d/device/driver" || true
    cat "$d"/ports/*/state "$d"/ports/*/link_layer 2>/dev/null || true
done
ibv_devices || true
ibv_devinfo || true
ldconfig -p | grep -E 'lib(ibverbs|mlx5|efa|fabric)' || true
for lib in /lib/x86_64-linux-gnu/libibverbs.so* /lib/x86_64-linux-gnu/libmlx5.so* /usr/lib/x86_64-linux-gnu/libibverbs/lib*; do
    [[ -f "$lib" ]] || continue
    readlink -f "$lib"
    ldd "$lib" || true
    readelf -V "$lib" 2>/dev/null | grep -E 'IBVERBS_PRIVATE|Name:' || true
done
python3 - <<'PYTHON'
import importlib.metadata as m
import importlib.util
import traceback
for name in ['vllm', 'mooncake-transfer-engine', 'mooncake-transfer-engine-cuda13', 'torch']:
    try: print('package', name, m.version(name), flush=True)
    except m.PackageNotFoundError: print('package', name, 'MISSING', flush=True)
try:
    from mooncake.store import MooncakeDistributedStore
    print('MOONCAKE_IMPORT=PASS', flush=True)
except Exception:
    traceback.print_exc()
    print('MOONCAKE_IMPORT=FAIL', flush=True)
    raise SystemExit(1)
PYTHON
rc=$?
printf 'container_diagnostic_end=%s import_rc=%s\n' "$(date -u --iso-8601=seconds)" "$rc"
exit "$rc"
