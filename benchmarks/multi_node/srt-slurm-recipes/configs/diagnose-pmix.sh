#!/usr/bin/env bash

# Read launch metadata without initializing MPI or changing its configuration.
python3 -S - <<'PY'
import json
import os
import socket
import stat
from pathlib import Path

names = (
    "SLURM_MPI_TYPE", "SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_JOB_UID",
    "SLURM_PROCID", "SLURM_LOCALID", "PMIX_NAMESPACE", "PMIX_RANK",
    "PMIX_SERVER_URI", "PMIX_SERVER_URI2", "PMIX_SERVER_URI21",
    "PMIX_SERVER_URI3", "PMIX_SERVER_URI4", "PMIX_SERVER_URI41",
    "PMIX_SERVER_TMPDIR", "PMIX_SYSTEM_TMPDIR", "PMIX_SECURITY_MODE",
    "PMIX_PTL_MODULE", "PMIX_GDS_MODULE", "PMIX_MCA_psec",
    "PMIX_MCA_ptl", "PMIX_MCA_gds",
)
environment = {name: os.environ[name] for name in names if name in os.environ}
paths = {environment[name] for name in ("PMIX_SERVER_TMPDIR", "PMIX_SYSTEM_TMPDIR")
         if environment.get(name, "").startswith("/")}
for name, value in environment.items():
    if name.startswith("PMIX_SERVER_URI"):
        for component in value.split(";"):
            if component.startswith("file:"):
                component = component.removeprefix("file:")
            if component.startswith("/"):
                paths.add(component)
job = environment.get("SLURM_JOB_ID", "")
step = environment.get("SLURM_STEP_ID", "")
uid = environment.get("SLURM_JOB_UID", "")
if job.isdecimal() and step.isdecimal():
    paths.update((f"/var/spool/slurmd/pmix.{job}.{step}",
                  f"/tmp/spmix_appdir_{job}.{step}"))
    if uid.isdecimal():
        paths.add(f"/tmp/spmix_appdir_{uid}_{job}.{step}")
metadata = {}
for name in sorted(paths):
    try:
        info = Path(name).stat()
        metadata[name] = {"uid": info.st_uid, "gid": info.st_gid,
                          "mode": stat.filemode(info.st_mode)}
    except OSError as error:
        metadata[name] = {"errno": error.errno}
try:
    uid_map = Path("/proc/self/uid_map").read_text().strip()
except OSError as error:
    uid_map = f"unavailable: errno={error.errno}"
print("PMIx launch diagnostics: " + json.dumps({
    "hostname": socket.gethostname(), "uid": os.getuid(), "gid": os.getgid(),
    "uid_map": uid_map, "environment": environment, "paths": metadata,
}, sort_keys=True))
PY

if command -v ompi_info >/dev/null 2>&1; then
    timeout 10s ompi_info --version || true
    timeout 10s ompi_info --param pmix all || true
fi
