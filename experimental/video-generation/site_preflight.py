"""Retain runner identity so SSH trust can be checked against exact CI provenance."""

import argparse
import json
import os
from pathlib import Path
import platform
import pwd
import shutil
import subprocess


def inspect_site() -> dict:
    user = pwd.getpwuid(os.getuid())
    public_keys = {}
    for algorithm in ("ed25519", "ecdsa", "rsa"):
        path = Path(f"/etc/ssh/ssh_host_{algorithm}_key.pub")
        if path.is_file():
            public_keys[algorithm] = path.read_text().strip()
    known_jumpbox = subprocess.run(
        ["ssh-keygen", "-F", "64.139.223.123"], capture_output=True, text=True, timeout=5,
    ) if shutil.which("ssh-keygen") else None
    associations = subprocess.run(
        ["sacctmgr", "-nP", "show", "assoc", "where", f"user={user.pw_name}",
         "format=Account,Partition,QOS,DefaultQOS"], capture_output=True, text=True, timeout=10,
    ) if shutil.which("sacctmgr") else None
    # Shared runtime/model cache candidates are metadata, not proof of compatibility.
    candidates = {}
    for name in ("/var/lib/squash", "/it-share/data", "/it-share/wenyao-minimax-h3",
                 "/data/home/sa-shared/wenyao-minimax-h3"):
        path = Path(name)
        candidates[name] = sorted(item.name for item in path.iterdir()
                                  if any(word in item.name.lower() for word in ("sglang", "rocm", "h3", "minimax"))) if path.is_dir() else None
    enroot_config = Path("/etc/enroot/enroot.conf")
    enroot_paths = [line.strip() for line in enroot_config.read_text().splitlines()
                    if line.strip().startswith(("ENROOT_DATA_PATH", "ENROOT_CACHE_PATH", "ENROOT_RUNTIME_PATH"))] if enroot_config.is_file() else None
    return {
        "schema_version": 1, "bundle_type": "h3_site_preflight_no_gpu",
        "hostname": platform.node(), "uid": os.getuid(), "username": user.pw_name,
        "user_home": user.pw_dir,
        "cluster": os.environ.get("H3_CLUSTER"), "runner": os.environ.get("RUNNER_NAME"),
        "ci": {key: os.environ.get(key) for key in ("GITHUB_REPOSITORY", "GITHUB_SHA", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT")},
        "commands": {name: shutil.which(name) for name in ("salloc", "srun", "enroot", "amd-smi", "nvidia-smi")},
        "ssh_host_public_keys": public_keys,
        "previously_known_amd_jumpbox": known_jumpbox.stdout if known_jumpbox and known_jumpbox.returncode == 0 else None,
        "scheduler_associations": associations.stdout if associations and associations.returncode == 0 else None,
        "runtime_candidates": candidates, "enroot_paths": enroot_paths,
        "gpu_execution": False, "runtime_compatibility": "not_tested",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "site-preflight.json").write_text(json.dumps(inspect_site(), indent=2) + "\n")
