#!/usr/bin/env bash
# Lift RLIMIT_MEMLOCK's soft limit to the hard limit in the shell that starts the worker
# (srtctl runs this script, then the worker, from one shell, so that shell is $PPID).
set -eo pipefail
python3 - "$PPID" <<'PY'
import resource
import sys

pid = int(sys.argv[1])
soft, hard = resource.prlimit(pid, resource.RLIMIT_MEMLOCK)
if soft != hard:
    resource.prlimit(pid, resource.RLIMIT_MEMLOCK, (hard, hard))
    print(f"RLIMIT_MEMLOCK of pid {pid}: soft {soft} -> {hard}")
PY
