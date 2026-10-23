#!/usr/bin/env bash
# srt-slurm AgentX client; infx/srt_slurm/single_node.py recognizes AgentX recipes by this name.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
if [[ ${K3_STARTUP_DIAGNOSTIC-} == 1 ]]; then
    exec python3 "$root/infx/bench/load_diagnostic.py" smoke
fi
exec env PYTHONSAFEPATH=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}" \
    python3 -m infx.bench agentic "$@"
