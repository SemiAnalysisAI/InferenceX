#!/usr/bin/env python3
"""Stage the frozen model from an accepted H3 run, without allocating GPUs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from urllib.parse import quote
from urllib.request import urlopen

import ci
from evaluator.mvp_gpu_job import validate_gpu_job
from export_ci import REPOSITORY, source_ids, verified_execution


def matches(path: Path, entry: dict) -> bool:
    return (path.is_file() and path.stat().st_size == entry["size_bytes"]
            and ci.digest(path) == entry["sha256"])


def fetch_weight(root: Path, revision: str, entry: dict) -> None:
    target = root / entry["path"]
    if matches(target, entry):
        return
    ci.need(not target.exists(), "Existing model file differs from the frozen manifest: " + entry["path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    url = "https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/" + revision + "/" + quote(entry["path"], safe="/")
    digest, size = hashlib.sha256(), 0
    # No GitHub or implicit Hugging Face credentials are sent to the model host.
    with urlopen(url, timeout=60) as response, partial.open("wb") as stream:
        while chunk := response.read(8 * 1024 * 1024):
            size += len(chunk)
            ci.need(size <= entry["size_bytes"], "Model download exceeds frozen size")
            digest.update(chunk)
            stream.write(chunk)
    ci.need(size == entry["size_bytes"] and digest.hexdigest() == entry["sha256"],
            "Downloaded model differs from frozen size or SHA256: " + entry["path"])
    partial.replace(target)


def stage_model(spec: dict, workspace: Path, candidates: list[Path]) -> dict:
    spec = validate_gpu_job(spec)
    ci.need(spec["plan"]["model_id"] == "MiniMaxAI/MiniMax-H3", "Only the approved H3 model is supported")
    approval = spec["authorization"]
    ci.need(approval["compute_approved"] and approval["model_license_reviewed"]
            and approval["approval_reference"].strip(), "Source lacks recorded model approval")
    model = spec["model"]
    entries = model["files"]
    observations = []
    root = workspace / "models" / "MiniMax-H3" / model["revision"]
    for candidate in [*candidates, root]:
        present = candidate.is_dir()
        valid = present and all(matches(candidate / entry["path"], entry) for entry in entries)
        observations.append({"path": str(candidate), "present": present, "verified": valid})
        if valid:
            root = candidate
            break
    else:
        root.mkdir(parents=True, exist_ok=True)
        missing_bytes = sum(entry["size_bytes"] for entry in entries if not (root / entry["path"]).exists())
        ci.need(shutil.disk_usage(root).free >= missing_bytes, "Insufficient persistent space for frozen model")
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda entry: fetch_weight(root, model["revision"], entry), entries))
    return {"model_path": str(root), "model_revision": model["revision"],
            "manifest_sha256": hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "verified_files": len(entries), "total_bytes": sum(entry["size_bytes"] for entry in entries),
            "reuse_candidates": observations, "verification": "complete frozen file sizes and SHA256"}


def prepare(run_id: str, workspace: Path, output: Path) -> None:
    ci.need(source_ids(run_id) == [run_id], "One accepted source run is required")
    ci.need(workspace.is_absolute(), "Persistent workspace must be absolute")
    output.mkdir(parents=True, exist_ok=True)
    record = {"schema_version": 1, "bundle_type": "h3_model_preparation_no_gpu",
              "gpu_allocation": False, "gpu_execution": False, "status": "preparing",
              "workspace": {"host": str(workspace), "container": "/work"},
              "ci": {key: os.environ.get(key) for key in ("GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_SHA")}}
    ci.write(output / "model-preparation.json", record)
    try:
        source, artifact = verified_execution(run_id)
        record.update(source_ci=source, source_artifact=artifact)
        original = output / "source"
        subprocess.run(["gh", "run", "download", run_id, "--repo", REPOSITORY,
                        "--name", artifact["name"], "--dir", str(original)], check=True, timeout=180)
        sums = dict(line.split("  ", 1)[::-1] for line in (original / "SHA256SUMS").read_text().splitlines())
        path = "gpu/c1/spec.json"
        ci.need(sums.get(path) == ci.digest(original / path), "Frozen source specification checksum differs")
        spec = ci.read(original / path)
        workspace.mkdir(parents=True, exist_ok=True)
        with ci.task_lock(workspace / ".model-preparation.lock"):
            revision = spec["model"]["revision"]
            candidates = [base / "models--MiniMaxAI--MiniMax-H3" / "snapshots" / revision
                          for base in (Path("/it-share/hf-hub-cache"), Path.home() / ".cache/huggingface/hub")]
            record.update(stage_model(spec, workspace, candidates), status="complete")
            receipt = workspace / "campaigns/h3-cross-hardware/model-ready.json"
            receipt.parent.mkdir(parents=True, exist_ok=True)
            ci.write(receipt, record)
    except Exception as error:
        record.update(status="failed", error=str(error))
        raise
    finally:
        ci.write(output / "model-preparation.json", record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.source_run_id, args.workspace, args.output)
