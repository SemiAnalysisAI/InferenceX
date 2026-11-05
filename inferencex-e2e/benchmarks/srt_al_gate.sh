#!/usr/bin/env bash
# srt-slurm benchmark stage of an agentic eval job: the report-only SPEED-Bench acceptance gate.
# srt-slurm runs the post-benchmark eval only after a zero exit, so a failed gate exits 0;
# its result document records the error. The eval shares this job's time limit, so the gate
# is capped: timeout signals its whole process group, AIPerf included, and the gate records
# the stop as an error.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 0
export AL_GATE_TIMEOUT_S="${AL_GATE_TIMEOUT_S:-1800}"
timeout --kill-after=60 "$AL_GATE_TIMEOUT_S" \
    env PYTHONSAFEPATH=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}" \
    python3 -m infx.bench al-gate "$@" || echo "AL gate exited $? (report-only)" >&2
exit 0
