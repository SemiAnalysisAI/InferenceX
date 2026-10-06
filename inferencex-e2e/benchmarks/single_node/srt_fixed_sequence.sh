#!/usr/bin/env bash
# srt-slurm single-node fixed-sequence client; see infx/bench/fixed_seq.py (srt-single).
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || exit 1
exec env PYTHONSAFEPATH=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}" \
    python3 -m infx.bench fixed-seq srt-single "$@"
