#!/usr/bin/env bash
set -eo pipefail

source /infmax-workspace/benchmarks/check_env.sh
# The tilert release ships the tilert-pd-router, so the master's router version is its version.
check_env_vars ROUTER_VERSION TILERT_ROLE
case "$TILERT_ROLE" in
    prefill|decode|router) ;;
    *) echo "Unknown TileRT role: $TILERT_ROLE" >&2; exit 1 ;;
esac

install_args=(--quiet --no-cache-dir)
if [[ "$TILERT_ROLE" == prefill ]]; then
    # Preserve the prefill image's vLLM/Torch dependency set.
    install_args+=(--no-deps)
fi
python3 -m pip install "${install_args[@]}" "tilert==$ROUTER_VERSION"

if ! python3 -c 'import mooncake.engine' >/dev/null 2>&1; then
    python3 -m pip install --quiet --no-cache-dir 'mooncake-transfer-engine-rocm>=0.3.13'
fi
if [[ "$TILERT_ROLE" != prefill ]]; then
    if ! python3 -c 'import uvicorn' >/dev/null 2>&1; then
        python3 -m pip install --quiet --no-cache-dir fastapi uvicorn httpx
    fi
    if ! python3 -c 'import transformers' >/dev/null 2>&1; then
        python3 -m pip install --quiet --no-cache-dir 'transformers>=4.56'
    fi
fi
