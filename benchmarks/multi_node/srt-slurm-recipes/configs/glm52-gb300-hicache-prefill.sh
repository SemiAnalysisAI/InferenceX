#!/usr/bin/env bash
set -eo pipefail

# SRT runs setup scripts in frontend and worker containers. The selected recipe
# explicitly enables this candidate only in its prefill role (waiver #3402).
source /infmax-workspace/benchmarks/benchmark_lib.sh --validation-only
check_env_vars INFX_GLM52_HICACHE_PATCH
case "$INFX_GLM52_HICACHE_PATCH" in
    1) python3 /infmax-workspace/runners/patch_glm52_hicache_sync.py ;;
    0) echo "HiCache prefill candidate not selected for this role" ;;
    *) echo "INFX_GLM52_HICACHE_PATCH must be 0 or 1" >&2; exit 1 ;;
esac
