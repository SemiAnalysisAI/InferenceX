#!/usr/bin/env bash
set -eo pipefail

# SRT invokes setup in frontend and worker containers; the recipe selects prefill.
source /infmax-workspace/benchmarks/benchmark_lib.sh --validation-only
check_env_vars INFX_GLM52_NIXL_SYNC_PATCH
case "$INFX_GLM52_NIXL_SYNC_PATCH" in
    1) python3 /infmax-workspace/runners/patch_glm52_nixl_sync.py ;;
    0) echo "NIXL prefill candidate not selected for this role" ;;
    *) echo "INFX_GLM52_NIXL_SYNC_PATCH must be 0 or 1" >&2; exit 1 ;;
esac
