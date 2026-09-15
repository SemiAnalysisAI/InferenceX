"""Inspect the allocated compute node before this diagnostic extracts a rootfs."""

import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys


def inspect(out, image):
    allocation = json.loads((out / "allocation.json").read_text())
    evidence = {
        "hostname": socket.gethostname(),
        "uid": os.getuid(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        "status": "blocked",
        "commands": [],
    }
    try:
        if (
            evidence["slurm_job_id"] != allocation["job_id"]
            or evidence["uid"] != allocation["uid"]
        ):
            raise RuntimeError("Compute step does not match the owned allocation")
        # Pyxis ignores job-supplied Enroot path overrides and uses the native
        # configuration. Inspect those same paths rather than a shell override.
        ignored = {
            "ENROOT_LIBRARY_PATH", "ENROOT_SYSCONF_PATH", "ENROOT_RUNTIME_PATH",
            "ENROOT_CACHE_PATH", "ENROOT_DATA_PATH", "ENROOT_TEMP_PATH",
        }
        environment = {key: value for key, value in os.environ.items() if key not in ignored}
        environment.update(SLURM_JOB_UID=str(os.getuid()), SLURM_JOB_GID=str(os.getgid()))
        evidence["ignored_job_path_overrides"] = sorted(ignored.intersection(os.environ))
        results = []
        for command in (["enroot", "info"], ["enroot", "list"]):
            result = subprocess.run(command, capture_output=True, text=True, timeout=10, env=environment)
            evidence["commands"].append({
                "argv": command, "rc": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr,
            })
            if result.returncode:
                raise RuntimeError("Enroot inspection failed; runtime absence is unknown")
            results.append(result.stdout)
        paths = dict(re.findall(r"^\s*(ENROOT_(?:DATA|RUNTIME)_PATH)=(.+)$", results[0], re.M))
        evidence["paths"] = paths
        if len(paths) != 2 or any(not Path(path).is_dir() for path in paths.values()):
            raise RuntimeError("Enroot data/runtime paths are unresolved or unavailable")
        evidence["visible_rootfs"] = results[1].splitlines()
        if evidence["visible_rootfs"]:
            raise RuntimeError(
                "Existing rootfs found; review task ownership and compatibility, then reuse its saved entry. "
                "The original unnamed C40 container has no saved rootfs identity."
            )
        image_stat = image.stat()
        if not image.is_file() or image_stat.st_size == 0:
            raise RuntimeError("The required cached image is unavailable; no import is allowed")
        evidence["image"] = {"path": str(image), "size": image_stat.st_size}
        evidence["status"] = "allow_cached_image"
        evidence["reason"] = (
            "No rootfs is visible in the effective Enroot data path on this allocated compute node. "
            "The old run saved no persistent name or separate data path; use the existing image archive."
        )
    except Exception as error:
        evidence["reason"] = str(error)
        raise
    finally:
        (out / "runtime-inspection.json").write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    inspect(Path(sys.argv[1]), Path(sys.argv[2]))
