"""Retain runner identity so SSH trust can be checked against exact CI provenance."""

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess


def inspect_site() -> dict:
    public_keys = {}
    for algorithm in ("ed25519", "ecdsa", "rsa"):
        path = Path(f"/etc/ssh/ssh_host_{algorithm}_key.pub")
        if path.is_file():
            public_keys[algorithm] = path.read_text().strip()
    known_jumpbox = subprocess.run(
        ["ssh-keygen", "-F", "64.139.223.123"], capture_output=True, text=True, timeout=5,
    ) if shutil.which("ssh-keygen") else None
    return {
        "schema_version": 1, "bundle_type": "h3_site_preflight_no_gpu",
        "hostname": platform.node(), "uid": os.getuid(),
        "cluster": os.environ.get("H3_CLUSTER"), "runner": os.environ.get("RUNNER_NAME"),
        "ci": {key: os.environ.get(key) for key in ("GITHUB_REPOSITORY", "GITHUB_SHA", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT")},
        "commands": {name: shutil.which(name) for name in ("salloc", "srun", "enroot", "amd-smi", "nvidia-smi")},
        "ssh_host_public_keys": public_keys,
        "previously_known_amd_jumpbox": known_jumpbox.stdout if known_jumpbox and known_jumpbox.returncode == 0 else None,
        "gpu_execution": False, "runtime_compatibility": "not_tested",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "site-preflight.json").write_text(json.dumps(inspect_site(), indent=2) + "\n")
