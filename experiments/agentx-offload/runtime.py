"""Four-tier configuration, storage proof and task-owned cache lifecycle."""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import shutil
import signal
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
GIB = 2**30
SCRATCH_ROOT = Path("/offload-scratch")


def connector_config(
    arm: str, tp: int, dram: int, nvme: int, cache: Path
) -> dict[str, Any] | None:
    """Build the pinned engine's real connector configuration, in bytes."""
    if tp <= 0 or dram < 0 or nvme < 0:
        raise ValueError("Invalid topology or capacity")
    if arm == "none":
        if dram or nvme:
            raise ValueError("HBM-only cannot declare an external cache")
        return None
    if arm == "dram":
        if dram <= 0 or nvme:
            raise ValueError("DRAM arm requires only a positive DRAM capacity")
        extra = {
            "kv_offload_backend": "cpu",
            "cpu_bytes_to_use": dram,
            "lazy_offload": True,
        }
    elif arm == "nvme":
        if dram or nvme <= 0 or nvme % tp:
            raise ValueError(
                "NVMe arm requires a divisible NVMe budget and no DRAM cache"
            )
        # This field is logical fallback capacity in disk mode, not resident DRAM.
        extra = {
            "kv_offload_backend": "disk",
            "cpu_bytes_to_use": nvme,
            "lazy_offload": True,
            "disk_path": str(cache / "cache.bin"),
            "disk_capacity_bytes": nvme // tp,
            "disk_buffer_slots": 4,
            "use_page_cache": False,
        }
    elif arm == "dram-nvme":
        if dram <= 0 or nvme <= 0:
            raise ValueError("Tiered arm requires both budgets")
        return {
            "kv_connector": "OffloadingConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "spec_name": "TieringOffloadingSpec",
                "cpu_bytes_to_use": dram,
                "eviction_policy": "lru",
                "secondary_tiers": [
                    {
                        "type": "fs",
                        "root_dir": str(cache),
                        "n_read_threads": 32,
                        "n_write_threads": 16,
                        "locality": "LOCAL",
                    }
                ],
            },
        }
    else:
        raise ValueError(f"Unknown arm: {arm}")
    return {
        "kv_connector": "SimpleCPUOffloadConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": extra,
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def read_config() -> tuple[Path, dict[str, Any]]:
    result = Path(os.environ["RESULT_DIR"])
    return result, json.loads((result / "offload_config.json").read_text())


def storage_proof(root: Path) -> dict[str, Any]:
    mount = json.loads(
        subprocess.check_output(["findmnt", "--json", "--target", str(root)], text=True)
    )["filesystems"][0]
    if mount["fstype"] not in {"xfs", "ext4"}:
        raise RuntimeError(f"Expected local block filesystem, got {mount['fstype']}")
    devices = json.loads(
        subprocess.check_output(
            [
                "lsblk",
                "--json",
                "--inverse",
                "--output",
                "NAME,TYPE,ROTA,TRAN",
                mount["source"].split("[", 1)[0],
            ],
            text=True,
        )
    )
    leaves: list[dict[str, Any]] = []

    def visit(node: dict[str, Any]) -> None:
        if node.get("children"):
            for child in node["children"]:
                visit(child)
        else:
            leaves.append(node)

    for device in devices["blockdevices"]:
        visit(device)
    if not leaves or any(not n["name"].startswith("nvme") or n["rota"] for n in leaves):
        raise RuntimeError("Could not verify NVMe backing devices")
    return {
        "mount": mount,
        "devices": devices,
        "free_bytes": shutil.disk_usage(root).free,
    }


def prepare() -> None:
    study = json.loads((ROOT / "study.json").read_text())
    arm, tp, conc = (
        os.environ["KV_OFFLOADING"],
        int(os.environ["TP"]),
        int(os.environ["CONC"]),
    )
    if tp != study["tp"] or not 1 <= conc <= study["max_concurrency"]:
        raise ValueError("Topology or concurrency differs from the study")
    dram = int(os.environ["TOTAL_CPU_DRAM_GB"]) * 10**9
    expected_dram = study["dram_bytes_per_node"] if arm in {"dram", "dram-nvme"} else 0
    if dram != expected_dram:
        raise ValueError(
            f"Matrix DRAM budget differs from study: {dram} != {expected_dram}"
        )
    nvme = study["nvme_bytes_per_node"] if arm in {"nvme", "dram-nvme"} else 0
    job, run, attempt = (
        os.environ[k] for k in ("SLURM_JOB_ID", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT")
    )
    if not all(v.isdigit() for v in (job, run, attempt)):
        raise ValueError("Expected numeric workflow and allocation identities")
    result = Path(os.environ["RESULT_DIR"])
    result.mkdir(parents=True, exist_ok=True)
    scratch_root = SCRATCH_ROOT
    proof = storage_proof(scratch_root)
    if nvme and proof["free_bytes"] < nvme + 128 * GIB:
        raise RuntimeError("Insufficient node-local NVMe space for budget plus reserve")
    scratch = (
        scratch_root
        / f"inferencex-offload-{run}-{attempt}-{job}-{uuid.uuid4().hex[:12]}"
    )
    connector = connector_config(arm, tp, dram, nvme, scratch / "cache")
    scratch.mkdir(mode=0o700)
    try:
        (scratch / "cache").mkdir(mode=0o700)
        write_json(
            scratch / "owner.json",
            {"study": study["study"], "run": run, "attempt": attempt, "job": job},
        )
        if nvme:
            probe = scratch / "direct-io-probe"
            fd = os.open(probe, os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_DIRECT, 0o600)
            try:
                with mmap.mmap(-1, 4096) as aligned:
                    if os.write(fd, aligned) != 4096:
                        raise RuntimeError("O_DIRECT probe was incomplete")
                os.fsync(fd)
            finally:
                os.close(fd)
                probe.unlink()
        provenance = {
            str(p.relative_to(ROOT.parent.parent)): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in (
                ROOT / "study.json",
                ROOT / "configs.yaml",
                ROOT / "runtime.py",
                ROOT / "configure.sh",
            )
        }
        write_json(
            result / "offload_config.json",
            {
                "study": study,
                "arm": arm,
                "concurrency": conc,
                "tp": tp,
                "hbm_bytes_per_gpu": study["hbm_bytes_per_gpu"],
                "dram_bytes": dram,
                "nvme_bytes": nvme,
                "connector": connector,
                "scratch": str(scratch),
                "run": run,
                "attempt": attempt,
                "job": job,
                "git_sha": os.environ["GITHUB_SHA"],
                "storage": proof,
                "o_direct_verified": bool(nvme),
                "sources": provenance,
                "tiered_capacity_semantics": "Stop guard, not LRU eviction; no capacity-boundary claim"
                if arm == "dram-nvme"
                else None,
            },
        )
    except BaseException:
        shutil.rmtree(scratch)
        raise


def cache_usage(cache: Path) -> dict[str, int]:
    logical = allocated = files = 0
    for root, _, names in os.walk(cache):
        for name in names:
            try:
                st = (Path(root) / name).stat()
            except FileNotFoundError:
                continue
            logical += st.st_size
            allocated += st.st_blocks * 512
            files += 1
    return {"logical_bytes": logical, "allocated_bytes": allocated, "files": files}


def monitor(parent: int) -> None:
    result, cfg = read_config()
    cache = Path(cfg["scratch"]) / "cache"
    with (result / "offload-telemetry.jsonl").open("a", buffering=1) as output:
        while True:
            try:
                os.kill(parent, 0)
            except ProcessLookupError:
                return
            usage = cache_usage(cache)
            output.write(
                json.dumps(
                    {
                        "time": time.time(),
                        **usage,
                        "meminfo": Path("/proc/meminfo").read_text(),
                        "diskstats": Path("/proc/diskstats").read_text(),
                    }
                )
                + "\n"
            )
            # Tiered files include headers. A 1% metadata margin does not extend
            # the declared payload capacity. Raw bytes are retained for review.
            if cfg["nvme_bytes"] and usage["logical_bytes"] > cfg["nvme_bytes"] * 1.01:
                write_json(
                    result / "offload-guard.json",
                    {"reason": "NVMe capacity guard exceeded", **usage},
                )
                os.kill(parent, signal.SIGTERM)
                return
            time.sleep(30)


def finish(exit_code: int) -> None:
    result, cfg = read_config()
    scratch = Path(cfg["scratch"])
    if scratch.parent != SCRATCH_ROOT or scratch.is_symlink():
        raise RuntimeError("Refusing non-owned scratch cleanup")
    owner = json.loads((scratch / "owner.json").read_text())
    if owner != {
        "study": cfg["study"]["study"],
        **{k: cfg[k] for k in ("run", "attempt", "job")},
    }:
        raise RuntimeError("Scratch identity mismatch")
    receipt = {
        "scratch": str(scratch),
        "usage": cache_usage(scratch / "cache"),
        "exit_code": exit_code,
        "run": cfg["run"],
        "job": cfg["job"],
        "deleted": False,
    }
    write_json(result / "offload_cleanup.json", receipt)
    shutil.rmtree(scratch)
    receipt["deleted"] = True
    write_json(result / "offload_cleanup.json", receipt)
    if (result / "offload-guard.json").exists():
        raise RuntimeError("Storage guard invalidated this run; evidence retained")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "monitor", "finish"])
    parser.add_argument("--parent", type=int)
    parser.add_argument("--exit-code", type=int)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "monitor":
        if args.parent is None:
            parser.error("monitor requires --parent")
        try:
            monitor(args.parent)
        except Exception as exc:
            write_json(
                Path(os.environ["RESULT_DIR"]) / "offload-guard.json",
                {"reason": "Offload monitor failed", "error": str(exc)},
            )
            os.kill(args.parent, signal.SIGTERM)
            raise
    else:
        if args.exit_code is None:
            parser.error("finish requires --exit-code")
        finish(args.exit_code)


if __name__ == "__main__":
    main()
