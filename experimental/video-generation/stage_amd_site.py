"""Seal AMD C1 inputs after inspection, without allocating or installing."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess

import ci
from evaluator.mvp_gpu_job import source_file_manifest, validate_gpu_job
from prepare_amd_runtime import CONTAINER, REVISION
from stage_model_ci import source_spec

WORKSPACE = Path("/it-share/data/wenyao-minimax-h3/work")
INPUTS = Path(__file__).parent / "campaigns/h3-cross-hardware"


def runtime_probe(record: dict) -> dict:
    ci.need(record.get("status") in {"inspected", "recovered"} and record.get("source_revision") == REVISION,
            "AMD runtime inspection is missing or has a different source")
    probe = record["probe"]
    required = ("torch", "torchvision", "av", "numpy", "diffusers", "transformers", "sglang", "aiter", "triton", "amdsmi")
    missing = [name for name in required if not probe.get("imports", {}).get(name, {}).get("path")]
    ci.need(not missing, "AMD runtime imports require preparation: " + ", ".join(missing))
    if record["status"] == "inspected":
        ci.need(probe.get("torch_hip") and not probe.get("device_error")
                and len(probe.get("hip_devices", [])) == 8, "AMD HIP device enumeration is not verified")
        devices = probe.get("torch_devices", [])
        ci.need(len(devices) == 8 and all("MI355X" in item["name"] for item in devices),
                "Runtime is not the inspected eight-MI355X node")
    ci.need(Path(probe["python"]).is_absolute(), "Inspected Python path must be absolute")
    return probe


def timing_source(source: Path) -> tuple[Path, dict]:
    from evaluator import mvp_runtime_timing as timing
    destination = source.with_name(source.name + "-timing")
    if not destination.exists():
        subprocess.run(["git", "-C", str(source), "worktree", "add", "-b", "feat/h3-amd-serving-timing",
                        str(destination), REVISION], check=True)
        subprocess.run(["git", "-C", str(destination), "apply", "--check", str(timing.PATCH)], check=True)
        subprocess.run(["git", "-C", str(destination), "apply", str(timing.PATCH)], check=True)
        subprocess.run(["git", "-C", str(destination), "add", *timing.PATCHED_FILES], check=True)
        subprocess.run(["git", "-C", str(destination), "-c", "user.name=H3 Benchmark", "-c", "user.email=h3-benchmark@localhost",
                        "commit", "-m", "feat: record request-correlated H3 serving stages",
                        "-m", "记录 H3 请求的服务端阶段时间，保留原始运行时作为基线。"], check=True)
    timing.validate_source(destination)
    return destination, timing.identity()


def stage(spec: dict, output: Path, *, server_timing: bool = False, allocation_minutes: int = 110, destination: Path | None = None) -> dict:
    workspace = WORKSPACE
    control = workspace / "campaigns/h3-cross-hardware"
    readiness = control / "runtime-inspected.json"
    if not readiness.is_file():
        readiness = control / "rootfs-recovered.json"
    runtime = ci.read(readiness)
    probe = runtime_probe(runtime)
    model = ci.read(control / "model-ready.json")
    entries = spec["model"]["files"]
    manifest = hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    ci.need(model.get("status") == "complete" and model.get("manifest_sha256") == manifest
            and model.get("model_revision") == spec["model"]["revision"],
            "Prepared AMD weights differ from the frozen source")
    rootfs = workspace.parent / "enroot-data" / CONTAINER
    ci.need(runtime["rootfs"] == str(rootfs) and rootfs.is_dir(), "Prepared AMD rootfs is missing or changed")
    source = workspace / ("runtime-sglang-" + REVISION)
    identity = source_file_manifest(source)
    ci.need(identity["revision"] == REVISION, "AMD source revision changed after inspection")
    instrumentation = None
    if server_timing:
        source, instrumentation = timing_source(source)
        identity = source_file_manifest(source)
    destination = destination or control / ("formal-c1-timing-v1" if server_timing else "formal-c1-v1")
    ci.need(not destination.exists(), "AMD formal inputs already exist; inspect and reuse the sealed configuration")
    source_server = copy.deepcopy(spec["server"])
    spec = copy.deepcopy(spec)
    if server_timing:
        spec["server_timing"] = True
    plan = ci.read(INPUTS / "formal-8s-plan.json")
    spec.update(gpu_vendor="amd", job_id=plan["plan_id"], plan=plan,
                gpu_uuids=[f"00000000-0000-0000-0000-{i:012d}" for i in range(4)],
                lock_directory="/work/campaigns/h3-cross-hardware/control/gpu-locks", port=30317,
                serving={"mode": "closed_loop", "concurrency": 1, "delivery_deadline_seconds": None})
    spec["server"] = {"tp_size": 1, "ulysses_degree": 4, "encoder_parallel": "auto",
                      "performance_mode": "speed", "dit_cpu_offload": False, "attention_backend": "aiter"}
    ci.need(30 <= allocation_minutes <= 110, "AMD serving requires 30–110 remaining allocation minutes")
    spec["limits"].update(job_seconds=(allocation_minutes - 10) * 60, startup_seconds=900, request_seconds=900,
                          cleanup_seconds=60, command_seconds=30, telemetry_interval_seconds=1)
    spec["authorization"]["approval_reference"] = (
        "User authorized MI355X recovery, one warmup and exactly twenty measured attempts on 2026-09-09. "
        "Retain source model-license approval. AMD initial C1 uses 20 measured requests and one separate warmup, "
        "8 allocated GPUs and 4 participating GPUs, one allocation capped at 120 minutes, "
        f"with {allocation_minutes} minutes remaining for serving and ten minutes reserved for cleanup. "
        "Generation compatibility is unverified; retain failures and release the owned allocation.")
    spec["model"]["path"] = str(Path("/work") / Path(model["model_path"]).relative_to(workspace))
    for role in ("baseline", "candidate"):
        spec[role] = {"source": "/work/" + source.name, "source_sha256": identity["source_sha256"],
                      "revision": identity["revision"], "python": probe["python"]}
    spec = validate_gpu_job(spec)
    destination.mkdir()
    entry = destination / "entry-only.sh"
    command = '-c \'exec "$@"\' h3-entry "$@"' if 'exec bash "$@"' in (runtime.get("entrypoint") or "") else '"$@"'
    entry.write_text((INPUTS / "entry-only-amd.sh").read_text().replace("@ENTRY@", command)
                     .replace("runtime-inspected.json", readiness.name))
    ci.write(destination / "gpu-spec.json", spec)
    config = {"schema_version": 1, "task_id": "h3-cross-hardware", "site": ci.AMD_SITE,
              "workspace": {"host": str(workspace), "container": "/work"},
              "runtime": {"entry": str(entry), "entry_sha256": ci.digest(entry), "rootfs": str(rootfs),
                          "ready_marker": str(readiness), "python": probe["python"]},
              "spec": {"path": str(destination / "gpu-spec.json"), "sha256": ci.digest(destination / "gpu-spec.json")},
              "resources": {"gpus": 4, "allocated_gpus": 8, "cpus": 32, "memory_gb": 1024, "minutes": allocation_minutes},
              "allocation_receipts": [], "mode": "serving-smoke", "concurrencies": [1]}
    config = ci.validate_config(config)
    ci.prepared_spec(config)
    ci.write(destination / "site.json", config)
    for name in ("entry-only.sh", "gpu-spec.json", "site.json"):
        (output / name).write_bytes((destination / name).read_bytes())
    return {"site_config": str(destination / "site.json"), "source": identity,
            "runtime_inspection": runtime, "instrumentation": instrumentation,
            "server_configuration_deviation": {"source_server": source_server, "executed_server": spec["server"],
                "reason": "AMD uses TP1/Ulysses4 with AITER; NVIDIA source settings are retained for explicit comparison."},
            "model_receipt": model, "generation_executed": False,
            "status": "prepared", "compatibility": "Imports checked; allocated HIP identity and full video/audio warmup remain mandatory before measurement"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--server-timing", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    record = {"status": "preparing", "generation_executed": False,
              "ci": {key: os.environ.get(key) for key in ("H3_RUN_ID", "H3_SOURCE_SHA")}}
    try:
        spec, provenance = source_spec(args.source_run_id, args.output)
        record.update(provenance)
        with ci.task_lock(WORKSPACE / "campaigns/h3-cross-hardware/.site-preparation.lock"):
            record.update(stage(spec, args.output, server_timing=args.server_timing))
    except Exception as error:
        record.update(status="failed", error=str(error))
        raise
    finally:
        ci.write(args.output / "site-preparation.json", record)
