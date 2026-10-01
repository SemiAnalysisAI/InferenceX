#!/usr/bin/env bash
set -eo pipefail
source /infmax-workspace/benchmarks/benchmark_lib.sh --validation-only
check_env_vars TILERT_ROLE

case "$TILERT_ROLE" in
    prefill)
        python3 -m pip install --quiet --no-cache-dir --no-deps tilert==0.1.5.post3
        if ! python3 -c 'import nixl' 2>/dev/null; then
            python3 -m pip install --quiet --no-cache-dir nixl==1.3.1
        fi
        ;;
    decode|router)
        python3 -m pip install --quiet --no-cache-dir tilert==0.1.5.post3
        if ! python3 -c 'import uvicorn' 2>/dev/null; then
            python3 -m pip install --quiet --no-cache-dir fastapi uvicorn httpx
        fi
        if ! python3 -c 'import nixl' 2>/dev/null; then
            python3 -m pip install --quiet --no-cache-dir nixl==1.3.1
        fi
        if ! python3 -c 'from importlib.metadata import version; assert int(version("transformers").split(".")[0]) >= 5'; then
            python3 -m pip install --quiet --no-cache-dir 'transformers>=5.4.0'
        fi
        ;;
    *)
        echo "Unknown TileRT setup role: $TILERT_ROLE" >&2
        exit 1
        ;;
esac
