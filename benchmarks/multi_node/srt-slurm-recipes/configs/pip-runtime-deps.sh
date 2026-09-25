#!/usr/bin/env bash
# Install packages an image lacks before its process starts. The worker role or
# the frontend lists them in SETUP_PIP_PACKAGES; empty installs nothing.
set -euo pipefail
[[ -n "${SETUP_PIP_PACKAGES:-}" ]] || exit 0
pip_install=(python3 -m pip install --quiet)
if python3 -m pip install --help 2>/dev/null | grep -q -- --break-system-packages; then
    pip_install+=(--break-system-packages)
fi
read -r -a packages <<< "$SETUP_PIP_PACKAGES"
"${pip_install[@]}" "${packages[@]}"
