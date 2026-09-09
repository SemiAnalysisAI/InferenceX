"""Retain runner identity so SSH trust can be checked against exact CI provenance."""

import argparse
import json
import os
from pathlib import Path
import platform
import pwd
import shutil
import subprocess


def amd_runtime_observations() -> dict:
    """A failed extraction can leave reusable files; inspect before rebuilding."""
    root = Path("/it-share/data/wenyao-minimax-h3")
    rootfs = root / "enroot-data/wenyao-minimax-h3-rocm"
    files = {}
    for path in (rootfs.with_suffix(".image.json"), rootfs / "etc/rc",
                 rootfs / "etc/os-release", root / "work/campaigns/h3-cross-hardware/model-ready.json"):
        files[str(path)] = path.read_text()[:16000] if path.is_file() else None
    entries = {}
    for name in ("usr/bin/python3", "usr/local/bin/python3", "opt/venv/bin/python3", "bin/bash", "etc/rc", "dev", "proc", "sys"):
        path = rootfs / name
        entries[name] = {"exists": path.exists(), "symlink": str(path.readlink()) if path.is_symlink() else None}
    installed_sources = {}
    for directory in (Path("/usr/local/lib/enroot"), Path("/usr/lib/enroot")):
        for path in sorted(directory.glob("*.sh")):
            lines = path.read_text().splitlines()
            selected = set()
            for i, line in enumerate(lines):
                if any(word in line for word in ("unsquashfs", "runtime::create", "mksquashfs", "xattr")):
                    selected.update(range(max(0, i - 4), min(len(lines), i + 60)))
            if selected:
                installed_sources[str(path)] = "\n".join(f"{i + 1}: {lines[i]}" for i in sorted(selected))[:30000]
    return {"rootfs": str(rootfs), "exists": rootfs.is_dir(), "files": files,
            "entries": entries, "installed_enroot_sources": installed_sources,
            "gpu_allocation": False, "rootfs_changed": False}


def allocation_observations(root: Path) -> list[dict]:
    """Read this task's saved scheduler receipts without requesting resources."""
    records = []
    for path in sorted(root.glob("*/allocation.json"))[-8:]:
        receipt = json.loads(path.read_text())
        identity = receipt.get("identity", {})
        job = identity.get("JobId", "")
        if receipt.get("task_id") != "h3-cross-hardware" or not str(job).isdigit():
            continue
        result = subprocess.run(["scontrol", "show", "job", "-o", str(job)],
                                capture_output=True, text=True, timeout=10,
                                env={**os.environ, "TZ": "UTC", "LC_ALL": "C"})
        records.append({"receipt": str(path), "identity": identity, "exit_code": result.returncode,
                        "stdout": result.stdout, "stderr": result.stderr})
        if result.returncode:
            accounting = subprocess.run(
                ["sacct", "-X", "-j", str(job), "--noheader", "--parsable2",
                 "--format=JobID,State,Start,End,AllocCPUS,AllocTRES,ReqTRES,NodeList,Elapsed,ExitCode"],
                capture_output=True, text=True, timeout=10,
                env={**os.environ, "TZ": "UTC", "LC_ALL": "C"})
            records[-1]["accounting"] = {"exit_code": accounting.returncode, "stdout": accounting.stdout,
                                         "stderr": accounting.stderr}
    return records


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
    defaults = subprocess.run(
        ["sacctmgr", "-nP", "show", "user", "where", f"name={user.pw_name}",
         "format=User,DefaultAccount"], capture_output=True, text=True, timeout=10,
    ) if shutil.which("sacctmgr") else None
    active_accounts = subprocess.run(
        ["squeue", "--noheader", "--user=" + user.pw_name, "--format=%a"],
        capture_output=True, text=True, timeout=10,
    ) if shutil.which("squeue") else None
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
    storage = {}
    for name in ("/it-share", "/it-share/data", "/it-share/hf-hub-cache", "/it-share/gharunners2", user.pw_dir):
        path = Path(name)
        if path.is_dir():
            info = path.stat()
            mount = subprocess.run(["findmnt", "--target", name, "--noheadings", "--output", "TARGET,SOURCE,FSTYPE"],
                                   capture_output=True, text=True, timeout=5) if shutil.which("findmnt") else None
            storage[name] = {"uid": info.st_uid, "gid": info.st_gid, "mode": oct(info.st_mode & 0o777),
                             "writable": os.access(path, os.W_OK), "free_bytes": shutil.disk_usage(path).free,
                             "mount": mount.stdout.strip() if mount and mount.returncode == 0 else None}
        else:
            storage[name] = None
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
        "scheduler_default_account": defaults.stdout if defaults and defaults.returncode == 0 else None,
        "scheduler_active_accounts": sorted(set(active_accounts.stdout.split())) if active_accounts and active_accounts.returncode == 0 else None,
        "runtime_candidates": candidates, "enroot_paths": enroot_paths, "persistent_storage": storage,
        "saved_allocations": allocation_observations(Path("/it-share/data/wenyao-minimax-h3/work/results/h3-cross-hardware"))
        if os.environ.get("H3_CLUSTER") == "mi355x-amds" and shutil.which("scontrol") else [],
        "amd_runtime": amd_runtime_observations() if os.environ.get("H3_CLUSTER") == "mi355x-amds" else None,
        "gpu_execution": False, "runtime_compatibility": "not_tested",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "site-preflight.json").write_text(json.dumps(inspect_site(), indent=2) + "\n")
