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
  findmnt -T /etc/libibverbs.d -no TARGET,FSTYPE,OPTIONS || true
  [ -f /etc/libibverbs.d/efa.driver ] || { echo 'EFA driver descriptor missing from read-only mount' >&2; exit 1; }
  mkdir -p /etc/dpkg/dpkg.cfg.d
  [ ! -e /etc/dpkg/dpkg.cfg.d/efa-ro-ibverbs ]
  printf '%s\n' 'path-exclude=/etc/libibverbs.d/*' >/etc/dpkg/dpkg.cfg.d/efa-ro-ibverbs
  trap 'rm -f /etc/dpkg/dpkg.cfg.d/efa-ro-ibverbs' EXIT
fi
efa_installer_flags=()
if [ -d /opt/amazon/efa/lib ] && [ ! -w /opt/amazon/efa/lib ]; then
  findmnt -T /opt/amazon/efa/lib -no TARGET,FSTYPE,OPTIONS || true
  ls -la /opt/amazon/efa/lib
  dpkg-deb -x /tmp/efa/aws-efa-installer/DEBS/UBUNTU2404/x86_64/libfabric1-aws_2.6.0amzn1.0_amd64.deb /tmp/efa/reference
  dpkg-deb -x /tmp/efa/aws-efa-installer/DEBS/UBUNTU2404/x86_64/libfabric-aws-bin_2.6.0amzn1.0_amd64.deb /tmp/efa/reference
  cmp /tmp/efa/reference/opt/amazon/efa/lib/libfabric.so.1.32.0 /opt/amazon/efa/lib/libfabric.so.1.32.0
  cmp /tmp/efa/reference/opt/amazon/efa/bin/fi_info /opt/amazon/efa/bin/fi_info
  efa_installer_flags+=(--minimal)
fi
(cd /tmp/efa/aws-efa-installer && ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify --skip-mpi "${efa_installer_flags[@]}")
rm -rf /tmp/efa /tmp/aws-efa-installer-1.50.0.tar.gz
ldconfig
/opt/amazon/efa/bin/fi_info -p efa >/tmp/fi-info
grep -q 'provider: efa' /tmp/fi-info
python3 -m pip uninstall -y mooncake-transfer-engine mooncake-transfer-engine-cuda13
python3 -m pip install --no-deps mooncake-transfer-engine-efa-cuda13==0.3.13.post1
