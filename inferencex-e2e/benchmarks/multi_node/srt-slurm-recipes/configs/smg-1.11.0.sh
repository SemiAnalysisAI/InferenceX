#!/usr/bin/env bash
# Install the Shepherd Model Gateway that fronts the workers (worker and SMG containers).
set -euo pipefail
pip_install=(python3 -m pip install)
if python3 -m pip install --help 2>/dev/null | grep -q -- --break-system-packages; then
    pip_install+=(--break-system-packages)
fi
"${pip_install[@]}" --quiet smg==1.11.0
