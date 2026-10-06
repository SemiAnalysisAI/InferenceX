#!/usr/bin/env bash
# srt-slurm AgentX client; infx/srt_slurm/single_node.py recognizes AgentX recipes by this name.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
exec env PYTHONSAFEPATH=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}" \
    python3 -m infx.bench agentic "$@"
