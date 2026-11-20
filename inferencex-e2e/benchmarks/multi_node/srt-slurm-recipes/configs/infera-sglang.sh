#!/usr/bin/env bash
# Install Infera, at the master config's router version, into the stock SGLang image for the
# infera-sglang points. amd-infera's Linux x86_64 wheel on PyPI bundles the infera-router
# binary next to the infera.engine.sglang worker wrapper, so the master's router version is the
# Infera version. Only nats-py is missing from the lmsysorg ROCm image; --no-deps keeps every
# package the image ships (torch, sglang, transformers, ...) at its pinned version.
set -eo pipefail
source /infmax-workspace/benchmarks/check_env.sh
check_env_vars ROUTER_VERSION
pip_install=(python3 -m pip install)
if python3 -m pip install --help 2>/dev/null | grep -q -- --break-system-packages; then
    pip_install+=(--break-system-packages)
fi
"${pip_install[@]}" --quiet --no-cache-dir --no-deps \
    "nats-py==2.16.0" \
    "amd-infera==${ROUTER_VERSION}"
# The router the launcher will exec: the wheel's bundled binary, not one on PATH. Fails if pip
# fell back to the pure-Python wheel (a platform outside manylinux_2_28 x86_64).
python3 -c "import infera.engine.sglang; from infera.server.launch_rust import _find_binary; _find_binary()"
# Lift RLIMIT_MEMLOCK's soft limit to the hard limit in the shell that starts the worker
# ($PPID: srtctl runs this script, then the worker, from one shell). The RDMA driver charges
# every KV registration against the soft limit before falling back to peer memory, and the
# remapped container root has no CAP_IPC_LOCK, so the 8 GiB limit Slurm propagates fails
# Mooncake's concurrent 1 GiB registrations with ENOMEM. Infera does this itself from
# AMD-AGI/Infera#239; drop this once the router version carries it.
python3 - "$PPID" <<'PY'
import resource, sys
pid = int(sys.argv[1])
soft, hard = resource.prlimit(pid, resource.RLIMIT_MEMLOCK)
if soft != hard:
    resource.prlimit(pid, resource.RLIMIT_MEMLOCK, (hard, hard))
    print(f"RLIMIT_MEMLOCK of pid {pid}: soft {soft} -> {hard}")
PY
