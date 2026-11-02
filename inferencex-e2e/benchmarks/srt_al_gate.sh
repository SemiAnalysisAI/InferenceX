#!/usr/bin/env bash
# srt-slurm benchmark stage of the SPEED-Bench acceptance-length gate (AL_GATE launches).
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
exec env PYTHONSAFEPATH=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}" \
    python3 -m infx.bench al-gate "$@"
