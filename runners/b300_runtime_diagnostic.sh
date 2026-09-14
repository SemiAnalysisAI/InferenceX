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
want=$(abi "$lib/libibverbs.so.1")
prov=""
while read -r p; do
    printf '%s : %s -> %s\n' "$p" "$(abi "$p")" "$(readlink -f "$p")"
    [[ -z "$prov" && "$(abi "$p")" == "$want" ]] && prov="$p"
done < <(find /host-usr-lib/libibverbs /host-opt-amazon -name 'libmlx5-rdmav*.so' 2>/dev/null)
if [[ -z "$prov" ]]; then
    echo "no host mlx5 provider matches the container libibverbs ($want)"
    exit 2
fi
mlx5_dir=$(dirname "$(dirname "$prov")")
mlx5=$(readlink -f "$mlx5_dir/libmlx5.so.1")
printf 'remedy: %s + %s (%s)\n' "$prov" "$mlx5" "$(abi "$mlx5")"
cp -L "$prov" "$lib/libibverbs/" \
    && cp -L "$mlx5" "$lib/$(basename "$mlx5")" \
    && ln -sfn "$(basename "$mlx5")" "$lib/libmlx5.so.1" \
    || { echo "remedy install failed"; exit 3; }
ldd "$lib/libibverbs/$(basename "$prov")" 2>&1 | grep -E 'not found|ibverbs|mlx5'

echo "== enumerate after remedy"
python3 runners/b300_rdma_devices.py
echo "== mooncake transfer engine on $MOONCAKE_RAIL"
python3 runners/b300_rdma_devices.py --init "$MOONCAKE_RAIL"
