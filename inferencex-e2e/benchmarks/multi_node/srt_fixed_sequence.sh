#!/usr/bin/env bash
# srt-slurm multi-node fixed-sequence client, one point per CONC_LIST value; see infx/bench/fixed_seq.py.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || exit 1
exec env PYTHONSAFEPATH=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}" \
    python3 -m infx.bench fixed-seq srt-sweep --logs-dir /logs "$@"
