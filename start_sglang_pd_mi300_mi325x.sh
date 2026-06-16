#!/usr/bin/env bash
# Example env for PD disagg (MI300X / MI325X); invokes launchers next to this file in the InferenceX repo.
# Open firewall between nodes for PD traffic, e.g.: sudo ufw allow from <peer-ip>

_IX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

export PREFILL_MODEL_HOST_DIR="${PREFILL_MODEL_HOST_DIR:-/dev/shm}" # model path is under /dev/shm
export DECODE_MODEL_HOST_DIR="${DECODE_MODEL_HOST_DIR:-/dev/shm}"
export IMAGE="${IMAGE:-lmsysorg/sglang:v0.5.9-rocm700-mi30x}"

# Build the latest MoRI in the container to avoid a bug on topology searching in lmsysorg/sglang:v0.5.9-rocm700-mi30x docker
export INSTALL_MORI_IN_CONTAINER=1
export INSTALL_MORI_MODE=git
export MORI_GIT_REF=main
# Libbnxt: container sees /workspace = REMOTE_REPO on each GPU node (default: same path as this repo on launcher).
# The .tar.gz must exist on every node at REMOTE_REPO/driver/… (rsync InferenceX/driver if nodes differ from your laptop).
# The bnxt version on the host can be found in /usr/src/bnxt_re-***, we need to build the same version in the container
export REBUILD_LIBBNXT_IN_CONTAINER=1
export PATH_TO_BNXT_TAR_PACKAGE=/workspace/driver/libbnxt_re-231.0.162.0.tar.gz   # MI325X: Broadcom driver tarball
# export PATH_TO_BNXT_TAR_PACKAGE=/workspace/driver/libbnxt_re-230.2.52.0.tar.gz  # MI300X: Broadcom driver tarball

# export PREFILL_NODE="137.220.56.211"
# export DECODE_NODE="149.28.121.18"
# bash "${_IX_ROOT}/run_1p1d_sglang_mi300_mi325x.sh"

export PREFILL_NODE="137.220.56.211"
export DECODE_NODE_1="149.28.121.18"
export DECODE_NODE_2="207.148.10.255"
bash "${_IX_ROOT}/run_1p2d_sglang_mi300_mi325x.sh"
