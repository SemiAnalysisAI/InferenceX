#!/usr/bin/env bash
set -eo pipefail
source /infmax-workspace/benchmarks/check_env.sh
check_env_vars ROUTER_VERSION
pip_install=(python3 -m pip install)
if python3 -m pip install --help 2>/dev/null | grep -q -- --break-system-packages; then
    pip_install+=(--break-system-packages)
fi
"${pip_install[@]}" --quiet "vllm-router==${ROUTER_VERSION}"

# srtctl execs the worker from the parent shell after this setup child exits.
python3 - "$PPID" <<'PY'
import resource
import sys

pid = int(sys.argv[1])
soft, hard = resource.prlimit(pid, resource.RLIMIT_MEMLOCK)
if soft != hard:
    resource.prlimit(pid, resource.RLIMIT_MEMLOCK, (hard, hard))
print(f"RLIMIT_MEMLOCK of pid {pid}: soft {soft} -> {hard}")
PY
