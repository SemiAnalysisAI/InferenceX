#!/usr/bin/env bash
# Install the Shepherd Model Gateway (SMG) router (worker and router containers).
# The recipe setup script also runs in every engine container, so install SMG
# without its declared dependencies (grpcio, grpcio-health-checking, PyYAML,
# setproctitle) to leave the engine's grpc/protobuf untouched. `smg launch`
# imports only the Rust extension and setproctitle:
# https://github.com/smg-project/smg/blob/3be823a700fabaff3add8a390cf78f163479d686/bindings/python/src/smg/cli.py#L19-L83
# https://github.com/smg-project/smg/blob/3be823a700fabaff3add8a390cf78f163479d686/bindings/python/src/smg/launch_router.py#L7-L10
set -euo pipefail
# Ranks that share one container (e.g. one MPI srun step) run this concurrently;
# serialize the install so pip never sees a half-written package.
exec 9>/tmp/smg-1.11.0-install.lock
flock 9
pip_install=(python3 -m pip install)
if python3 -m pip install --help 2>/dev/null | grep -q -- --break-system-packages; then
    pip_install+=(--break-system-packages)
fi
"${pip_install[@]}" --quiet --no-deps smg==1.11.0
if ! python3 -c "import setproctitle" 2>/dev/null; then
    "${pip_install[@]}" --quiet --no-deps setproctitle
fi
