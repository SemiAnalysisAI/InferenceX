#!/usr/bin/env bash

# Source before the SRT installer so its lock and packages share a writable prefix.
source /infmax-workspace/benchmarks/benchmark_lib.sh --validation-only || return 1
check_env_vars SLURM_JOB_ID SLURM_STEP_ID || return 1

INFX_DYNAMO_VENV=$(python3 - <<'PY'
import fcntl
import os
import re
import venv
from pathlib import Path

job = os.environ["SLURM_JOB_ID"]
step = os.environ["SLURM_STEP_ID"]
if not re.fullmatch(r"[0-9]+", job) or not re.fullmatch(r"[0-9]+", step):
    raise ValueError("Dynamo virtual environment requires numeric Slurm job and step IDs")
root = Path("/tmp") / f"infx-dynamo-{job}.{step}"
root.mkdir(mode=0o700, exist_ok=True)
environment = root / "venv"
with (root / "create.lock").open("w") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    if not (root / "complete").exists():
        venv.EnvBuilder(system_site_packages=True, with_pip=True, symlinks=True).create(environment)
        (root / "complete").touch()
print(environment)
PY
) || return 1
export VIRTUAL_ENV="$INFX_DYNAMO_VENV"
export PATH="$VIRTUAL_ENV/bin:$PATH"
unset INFX_DYNAMO_VENV
