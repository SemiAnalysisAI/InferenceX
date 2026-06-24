#!/usr/bin/env bash
# Optional: build and wire MoRI into the active Python before server.sh (disagg images).
# SGLang often ships MoRI under /sgl-workspace/mori; prepending this tree to PYTHONPATH
# makes the freshly built C++ extension + Python sources win over the vendored copy.
#
# Modes (INSTALL_MORI_MODE):
#   git  — shallow clone from MORI_GIT_URL at ref MORI_GIT_REF into MORI_GIT_CLONE_DIR (default /tmp/mori_git_src)
#   path — use existing tree MORI_SOURCE_PATH (default /workspace/mori, e.g. bind-mount ROCm/mori next to InferenceX)
#
# Prereqs: network for git mode; libpci-dev / libibverbs-dev / cmake / ninja / pybind11 (image or apt).
# Example (host, before launcher):
#   export INSTALL_MORI_IN_CONTAINER=1
#   export INSTALL_MORI_MODE=git
#   export MORI_GIT_REF=main

set -euo pipefail

INSTALL_MORI_MODE="${INSTALL_MORI_MODE:-git}"
MORI_GIT_URL="${MORI_GIT_URL:-https://github.com/ROCm/mori.git}"
MORI_GIT_REF="${MORI_GIT_REF:-main}"
MORI_GIT_CLONE_DIR="${MORI_GIT_CLONE_DIR:-/tmp/mori_git_src}"
MORI_SOURCE_PATH="${MORI_SOURCE_PATH:-/workspace/mori}"

if [[ -x /opt/venv/bin/python3 ]]; then
  PY="${INSTALL_MORI_PYTHON_BIN:-/opt/venv/bin/python3}"
else
  PY="${INSTALL_MORI_PYTHON_BIN:-$(command -v python3)}"
fi

echo "[install_mori_in_container] using python: ${PY}"

SRC=""
if [[ "${INSTALL_MORI_MODE}" == "path" ]]; then
  if [[ ! -d "${MORI_SOURCE_PATH}" ]]; then
    echo "INSTALL_MORI_MODE=path but MORI_SOURCE_PATH=${MORI_SOURCE_PATH} is missing" >&2
    exit 1
  fi
  SRC="${MORI_SOURCE_PATH}"
elif [[ "${INSTALL_MORI_MODE}" == "git" ]]; then
  rm -rf "${MORI_GIT_CLONE_DIR}"
  git clone --depth 1 --branch "${MORI_GIT_REF}" "${MORI_GIT_URL}" "${MORI_GIT_CLONE_DIR}" || {
    echo "[install_mori_in_container] shallow clone failed (wrong ref or need full clone). Try:" >&2
    echo "  MORI_GIT_REF=<branch-or-tag>  or  INSTALL_MORI_MODE=path + mount mori at ${MORI_SOURCE_PATH}" >&2
    exit 1
  }
  SRC="${MORI_GIT_CLONE_DIR}"
else
  echo "Unknown INSTALL_MORI_MODE=${INSTALL_MORI_MODE} (use git or path)" >&2
  exit 1
fi

# Best-effort headers for CMake (Ubuntu/Debian); ignore failures (offline / read-only apt).
if command -v apt-get >/dev/null 2>&1; then
  apt-get update -qq >/dev/null 2>&1 || true
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    libpci-dev libibverbs-dev cmake ninja-build pybind11-dev git >/dev/null 2>&1 || true
fi

echo "[install_mori_in_container] pip install -e ${SRC}"
PIP_ARGS=(--no-cache-dir)
if [[ "${INSTALL_MORI_NO_BUILD_ISOLATION:-0}" == "1" ]]; then
  PIP_ARGS+=(--no-build-isolation)
fi
"${PY}" -m pip install -e "${SRC}" "${PIP_ARGS[@]}"

# Parent shell (_disagg_container_entry.sh) prepends this so imports beat /sgl-workspace/mori.
echo -n "${SRC}/python" >/tmp/mori_pythonpath_prefix

echo "[install_mori_in_container] done; wrote PYTHONPATH prefix to /tmp/mori_pythonpath_prefix"
