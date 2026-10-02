#!/usr/bin/env bash
# [DNM] Install the TokenSpeed 0.1.0.post20260930 nightly into
# lightseekorg/tokenspeed-runner:cu130-torch-2.14.0-flashinfer-0.7.0, the image
# TokenSpeed's own Slurm CI runs (test/ci/run_slurm.sh), as TokenSpeed's release
# image does (docker/Dockerfile.release):
# https://github.com/lightseekorg/tokenspeed/blob/22251686ff26a2b2f495263b348db80980b8ac9e/docker/Dockerfile.release
# https://github.com/lightseekorg/tokenspeed/blob/22251686ff26a2b2f495263b348db80980b8ac9e/docs/guides/getting-started.md#nightly-wheels
# The nightly pins its same-date tokenspeed-kernel (from the nightly index) and,
# from PyPI, the gRPC engine (tokenspeed-smg-grpc-servicer) and the SMG build it
# runs behind (tokenspeed-smg, the `smg` command):
# https://github.com/lightseekorg/tokenspeed/blob/22251686ff26a2b2f495263b348db80980b8ac9e/python/pyproject.toml#L75-L77
# The constraint keeps the image's torch 2.14.0+cu130; torchvision comes from the
# same CUDA 13.0 index. The lightseek nightly index keeps only the latest three
# nightlies, so this date is the oldest one available.
set -euo pipefail
# Ranks that share one container run this concurrently; serialize the install.
exec 9>/tmp/tokenspeed-0.1.0.post20260930-install.lock
flock 9
if python3 -c "import importlib.metadata as m, sys; sys.exit(m.version('tokenspeed') != '0.1.0.post20260930')" 2>/dev/null; then
    exit 0
fi
constraints=$(mktemp)
echo "torch==2.14.0" > "$constraints"
pip_install=(python3 -m pip install --break-system-packages --quiet --progress-bar off -c "$constraints")
"${pip_install[@]}" "torchvision==0.29.0" --index-url https://download.pytorch.org/whl/cu130
"${pip_install[@]}" "tokenspeed==0.1.0.post20260930" --extra-index-url https://lightseek.org/whl/nightly
rm -f "$constraints"
# Fail here, not at engine start, if a component is missing. Importing the engine
# needs a GPU, and the router containers have none, so check the distributions.
python3 -c "import importlib.metadata as m; [m.version(d) for d in ('tokenspeed', 'tokenspeed-kernel', 'tokenspeed-smg', 'tokenspeed-smg-grpc-servicer')]"
command -v smg >/dev/null
