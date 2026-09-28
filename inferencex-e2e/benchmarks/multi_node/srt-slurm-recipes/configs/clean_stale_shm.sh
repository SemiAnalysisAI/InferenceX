#!/usr/bin/env bash

# Use only with exclusive node allocations. Anchor the cutoff to allocation
# start so a later frontend/worker setup cannot remove this job's segments.
clean_stale_shm() {
    python3 -S - "$@" <<'PY'
import os
import socket
import stat
import sys
import time
from pathlib import Path

directory = Path(sys.argv[1])
job_start = int(sys.argv[2])
if job_start <= 600 or job_start > time.time():
    raise ValueError("SLURM_JOB_START_TIME must be a valid allocation start timestamp")
cutoff = job_start - 600
prefixes = ("vader_segment.", "nccl-", "sem.", "psm3", "fe80::")
entries = list(directory.iterdir())
removed = 0
for entry in entries:
    if not entry.name.startswith(prefixes):
        continue
    try:
        info = entry.lstat()
        if (info.st_uid != os.getuid() or not stat.S_ISREG(info.st_mode)
                or info.st_mtime >= cutoff):
            continue
        entry.unlink()
        removed += 1
    except FileNotFoundError:
        # Another rank may have already removed the same stale segment.
        continue
print(f"[clean_stale_shm] {socket.gethostname()}: removed {removed} stale segments "
      f"from {len(entries)} entries; cutoff={cutoff}")
PY
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    source /infmax-workspace/benchmarks/benchmark_lib.sh --validation-only || exit 1
    check_env_vars SLURM_JOB_START_TIME || exit 1
    clean_stale_shm /dev/shm "$SLURM_JOB_START_TIME"
fi
