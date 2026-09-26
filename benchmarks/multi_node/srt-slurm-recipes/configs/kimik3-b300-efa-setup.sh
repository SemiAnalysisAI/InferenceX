#!/usr/bin/env bash
set -eo pipefail
for path in /etc/libibverbs.d /opt/amazon/efa; do
  if [ -e "$path" ] && [ ! -w "$path" ]; then
    echo "EFA Installer requires writable $path" >&2
    exit 1
  fi
done
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends ca-certificates curl libhwloc15 numactl
curl -fL --retry 3 https://efa-installer.amazonaws.com/aws-efa-installer-1.50.0.tar.gz -o /tmp/aws-efa-installer-1.50.0.tar.gz
printf '%s  %s\n' fa6dff8593d866866c13cb4640d9059835cd4efa427971f100ab40c97bef2841 /tmp/aws-efa-installer-1.50.0.tar.gz | sha256sum -c -
mkdir -p /tmp/efa
tar -xzf /tmp/aws-efa-installer-1.50.0.tar.gz -C /tmp/efa
(cd /tmp/efa/aws-efa-installer && ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify --skip-mpi)
rm -rf /tmp/efa /tmp/aws-efa-installer-1.50.0.tar.gz
ldconfig
test "$(readlink -f /opt/amazon/efa/lib/libfabric.so.1)" = /opt/amazon/efa/lib/libfabric.so.1.32.0
fabric_loaded=$(ldd /opt/amazon/efa/bin/fi_info | awk '$1 == "libfabric.so.1" { print $3 }')
test "$(readlink -f "$fabric_loaded")" = /opt/amazon/efa/lib/libfabric.so.1.32.0
/opt/amazon/efa/bin/fi_info -p efa >/tmp/fi-info
grep -q 'provider: efa' /tmp/fi-info
python3 -m pip uninstall -y mooncake-transfer-engine mooncake-transfer-engine-cuda13
python3 -m pip install --no-deps mooncake-transfer-engine-efa-cuda13==0.3.13.post1
