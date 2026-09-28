#!/usr/bin/env bash
set -eo pipefail

export EFA_VERSION=1.50.0
curl --retry 3 --retry-delay 2 -fsSL -o aws-efa-installer-${EFA_VERSION}.tar.gz \
    https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_VERSION}.tar.gz
tar -xf aws-efa-installer-${EFA_VERSION}.tar.gz
cd aws-efa-installer
apt-get update
./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify --skip-rdma-core
cd ..
rm -rf aws-efa-installer*
ldconfig

python3 -m pip uninstall --break-system-packages -y mooncake-transfer-engine-cuda13 mooncake-transfer-engine-efa-cuda13
python3 -m pip install --break-system-packages --no-deps mooncake-transfer-engine-efa-cuda13==0.3.13.post1
mkdir -p /tmp/sglang-prometheus-prefill /tmp/sglang-prometheus-decode
