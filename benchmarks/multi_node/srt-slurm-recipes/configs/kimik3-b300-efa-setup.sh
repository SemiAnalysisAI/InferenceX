#!/usr/bin/env bash
set -eo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends ca-certificates curl libhwloc15 numactl
curl -fL --retry 3 https://efa-installer.amazonaws.com/aws-efa-installer-1.50.0.tar.gz -o /tmp/aws-efa-installer-1.50.0.tar.gz
printf '%s  %s\n' fa6dff8593d866866c13cb4640d9059835cd4efa427971f100ab40c97bef2841 /tmp/aws-efa-installer-1.50.0.tar.gz | sha256sum -c -
mkdir -p /tmp/efa
tar -xzf /tmp/aws-efa-installer-1.50.0.tar.gz -C /tmp/efa
if [ -d /etc/libibverbs.d ] && [ ! -w /etc/libibverbs.d ]; then
  test -f /etc/libibverbs.d/efa.driver
  printf '%s\n' 'path-exclude=/etc/libibverbs.d/*' >/etc/dpkg/dpkg.cfg.d/efa-ro-ibverbs
  trap 'rm -f /etc/dpkg/dpkg.cfg.d/efa-ro-ibverbs' EXIT
fi
efa_bin=/opt/amazon/efa/bin/fi_info
efa_flags=()
if [ -d /opt/amazon/efa ] && [ ! -w /opt/amazon/efa ]; then
  efa_bin=/opt/amazon/efa-1.50/bin/fi_info
  efa_flags+=(--minimal)
fi
(cd /tmp/efa/aws-efa-installer && ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify --skip-mpi "${efa_flags[@]}")
if [ ${#efa_flags[@]} -ne 0 ]; then
  packages=/tmp/efa/aws-efa-installer/DEBS/UBUNTU2404/x86_64
  mkdir -p /tmp/efa/libfabric /opt/amazon/efa-1.50
  for package in libfabric1-aws libfabric-aws-bin libfabric-aws-dev; do
    dpkg-deb -x "$packages/${package}_2.6.0amzn1.0_amd64.deb" /tmp/efa/libfabric
  done
  cp -a /tmp/efa/libfabric/opt/amazon/efa/. /opt/amazon/efa-1.50/
  printf '%s\n' /opt/amazon/efa-1.50/lib >/etc/ld.so.conf.d/00-efa-1.50.conf
fi
rm -rf /tmp/efa /tmp/aws-efa-installer-1.50.0.tar.gz
ldconfig
efa_lib="${efa_bin%/bin/fi_info}/lib"
test "$(readlink -f "$efa_lib/libfabric.so.1")" = "$efa_lib/libfabric.so.1.32.0"
LD_LIBRARY_PATH="$efa_lib:${LD_LIBRARY_PATH:-}" "$efa_bin" --version | grep -q 'libfabric: 2.6.0amzn1.0'
LD_LIBRARY_PATH="$efa_lib:${LD_LIBRARY_PATH:-}" "$efa_bin" -p efa >/tmp/fi-info
grep -q 'provider: efa' /tmp/fi-info
python3 -m pip uninstall -y mooncake-transfer-engine mooncake-transfer-engine-cuda13
python3 -m pip install --no-deps mooncake-transfer-engine-efa-cuda13==0.3.13.post1
mooncake_engine=$(python3 -c 'from importlib.util import find_spec; from pathlib import Path; print(Path(find_spec("mooncake").origin).with_name("engine.so"))')
fabric_loaded=$(ldd "$mooncake_engine" | awk '$1 == "libfabric.so.1" { print $3 }')
printf 'Mooncake libfabric: %s\n' "$fabric_loaded"
test "$(readlink -f "$fabric_loaded")" = "$efa_lib/libfabric.so.1.32.0"
