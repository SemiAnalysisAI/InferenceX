#!/usr/bin/env bash
# Install the Shepherd Model Gateway that fronts the workers (worker and SMG containers).
# --no-deps leaves the engine's grpcio/protobuf untouched; `smg launch` imports only
# the bundled Rust extension and setproctitle.
set -euo pipefail
pip_install=(python3 -m pip install)
if python3 -m pip install --help 2>/dev/null | grep -q -- --break-system-packages; then
    pip_install+=(--break-system-packages)
fi
"${pip_install[@]}" --quiet --no-deps smg==1.11.0
python3 -c 'import setproctitle' 2>/dev/null || "${pip_install[@]}" --quiet setproctitle
