#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Optionally rebuild **libbnxt** (userspace libibverbs provider) from Broadcom’s
# source tarball — typical fix for libibverbs / bnxt_re user-kernel ABI mismatch
# inside a container when the kernel module is already supplied by the host.
#
# Trigger:
#   export REBUILD_BNXT=1
#
# Required:
#   export PATH_TO_BNXT_TAR_PACKAGE=/path/to/libbnxt_re-*.tar.gz
#   Path must exist in the environment where this script runs (e.g. /workspace/... in Docker).
#
# Optional:
#   export REBUILD_BNXT_RESTORE_DIR=/some/dir  # cwd after build (default: directory of this script)
#
# Disagg Docker: set REBUILD_LIBBNXT_IN_CONTAINER=1 and pass PATH_TO_BNXT_TAR_PACKAGE
# (container path; tarballs are often under /workspace/driver/ when kept in InferenceX/driver/).
# Invoked from scripts/_disagg_container_entry.sh (in-container path below).
#
# If `ibv_devinfo` still warns about kernel ABI after rebuild, try the other tarball version
# (e.g. 230.2.52 vs 231.0.162) to match your host bnxt_re kernel module.
#
# Implementation: inline steps (legacy runner/helpers/rebuild_bnxt.sh).
###############################################################################

set -euo pipefail

if ! declare -F LOG_INFO_RANK0 >/dev/null 2>&1; then
  LOG_INFO_RANK0() { echo "$*"; }
fi

REBUILD_BNXT="${REBUILD_BNXT:-0}"
PATH_TO_BNXT_TAR_PACKAGE="${PATH_TO_BNXT_TAR_PACKAGE:-}"

if [[ "${REBUILD_BNXT}" != "1" ]]; then
  exit 0
fi

if [[ -z "${REBUILD_BNXT_RESTORE_DIR:-}" ]]; then
  # Directory to return to after the build (defaults to this script’s directory: amd_utils/).
  REBUILD_BNXT_RESTORE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  export REBUILD_BNXT_RESTORE_DIR
fi

if [[ -z "${PATH_TO_BNXT_TAR_PACKAGE}" ]]; then
  LOG_INFO_RANK0 "[hook system] Skip bnxt rebuild: PATH_TO_BNXT_TAR_PACKAGE is unset (REBUILD_BNXT=${REBUILD_BNXT})."
  exit 0
fi
if [[ ! -f "${PATH_TO_BNXT_TAR_PACKAGE}" ]]; then
  LOG_INFO_RANK0 "[hook system] Skip bnxt rebuild: tarball not found inside container at ${PATH_TO_BNXT_TAR_PACKAGE}"
  LOG_INFO_RANK0 "[hook system] (With -v HOST_REPO:/workspace, the host must have that file under HOST_REPO, e.g. InferenceX/driver/libbnxt_re-*.tar.gz on every node that runs Docker.)"
  exit 0
fi

LOG_INFO_RANK0 "[hook system] REBUILD_BNXT=1 → rebuilding libbnxt from ${PATH_TO_BNXT_TAR_PACKAGE}"

# Inline implementation (previously runner/helpers/rebuild_bnxt.sh)
tar xzf "${PATH_TO_BNXT_TAR_PACKAGE}" -C /tmp/
mv /tmp/libbnxt_re-* /tmp/libbnxt

_inbox="/usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so"
if [[ -f "${_inbox}" ]]; then
  mv "${_inbox}" "${_inbox}.inbox"
fi

cd /tmp/libbnxt/
if command -v apt-get >/dev/null 2>&1; then
  DEBIAN_FRONTEND=noninteractive apt-get update -qq || true
  if ! DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    autoconf automake libtool pkg-config make gcc; then
    LOG_INFO_RANK0 "[hook system] WARN: apt-get install build deps failed (offline image?); autogen may fail."
  fi
fi
if ! command -v autoconf >/dev/null 2>&1; then
  LOG_INFO_RANK0 "[hook system] ERROR: autoconf not found; install autoconf automake libtool pkg-config make gcc in the image or fix apt." >&2
  exit 1
fi

sh ./autogen.sh
./configure
make clean all install

echo '/usr/local/lib' > /etc/ld.so.conf.d/libbnxt_re.conf
ldconfig

# Register provider with libibverbs (paths vary by image).
mkdir -p /etc/libibverbs.d
cp -f /tmp/libbnxt/bnxt_re.driver /etc/libibverbs.d/ || LOG_INFO_RANK0 "[hook system] WARN: could not copy bnxt_re.driver to /etc/libibverbs.d"
for _verbs_d in /usr/local/etc/libibverbs.d /usr/lib/libibverbs.d; do
  if [[ -d "${_verbs_d}" ]]; then
    cp -f /tmp/libbnxt/bnxt_re.driver "${_verbs_d}/" || LOG_INFO_RANK0 "[hook system] WARN: could not copy bnxt_re.driver to ${_verbs_d}"
  fi
done

LOG_INFO_RANK0 "[hook system] Installed libbnxt_re libraries:"
ls -la /usr/local/lib/libbnxt_re*.so* 2>/dev/null || LOG_INFO_RANK0 "[hook system] WARN: no libbnxt_re*.so under /usr/local/lib"
if command -v ibv_devinfo >/dev/null 2>&1; then
  LOG_INFO_RANK0 "[hook system] ibv_devinfo (first bnxt device, if any):"
  ibv_devinfo -d bnxt_re0 2>&1 | head -40 || true
fi

cd "${REBUILD_BNXT_RESTORE_DIR}"
LOG_INFO_RANK0 "[hook system] Rebuilding libbnxt done."
