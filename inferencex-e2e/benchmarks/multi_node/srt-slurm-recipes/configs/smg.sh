#!/usr/bin/env bash
set -eo pipefail
source /infmax-workspace/benchmarks/check_env.sh
check_env_vars ROUTER_VERSION SMG_INSTALL_LOCK_DIR

# Worker ranks sharing a container must not race pip's package writes.
exec 9>"${SMG_INSTALL_LOCK_DIR}/inferencex-smg-install-${UID}.lock"
flock 9
pip_install=(python3 -m pip install)
if python3 -m pip install --help 2>/dev/null | grep -q -- --break-system-packages; then
    pip_install+=(--break-system-packages)
fi
# Keep the serving image's grpc/protobuf dependencies unchanged.
"${pip_install[@]}" --quiet --no-deps "smg==${ROUTER_VERSION}"
python3 -c "import smg, setproctitle"
