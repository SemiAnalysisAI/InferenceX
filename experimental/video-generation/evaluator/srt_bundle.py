"""Assemble the app-facing H3 result bundle from one srt-slurm video job.

The job leaves the client's per-concurrency run bundles under ``<logs>/h3/gpu/c<N>``
and the server's bound spec under ``<logs>/h3/server``; InferenceX validates each
cell's measured window against the native power package and leaves the aggregate and
validation sidecar in the runner workspace. This module joins them into the matrix
layout the dashboard ingests (``serving-smoke.json``, ``gpu/c<N>/{spec,gpu-job,power}.json``,
``manifest.json``, ``ci.json``, ``SHA256SUMS``) with evidence kind
``srt_slurm_h3_service``: power comes from srt-slurm's DCGM telemetry, re-validated by
InferenceX, never from the H3 supervisor's own sampler.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EVIDENCE_KIND = "srt_slurm_h3_service"
TASK_ID = "h3-video"
MODE = "serving-smoke"
BUNDLE_TYPE = "h3_serving_smoke_matrix"
RECEIPT_BUNDLE_TYPE = "srt_slurm_h3_service_receipt"
POWER_SCHEMA = "1.0.0"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _read(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _iso(unix: float) -> str:
    return datetime.fromtimestamp(unix, timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _text(path: Path) -> str | None:
    try:
        return path.read_text().strip() or None
    except OSError:
        return None


def _source_sha(workspace: Path, environ: dict[str, str]) -> str:
    try:
        head = subprocess.run(["git", "-C", str(workspace), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        head = ""
    return head or environ.get("GITHUB_SHA", "") or "unknown"


def _ci_context(environ: dict[str, str]) -> dict:
    run_id = environ.get("GITHUB_RUN_ID", "local")
    server = environ.get("GITHUB_SERVER_URL", "https://github.com")
    repository = environ.get("GITHUB_REPOSITORY", "")
    return {
        "repository": repository or None,
        "workflow_ref": environ.get("GITHUB_WORKFLOW_REF"),
        "head_sha": environ.get("GITHUB_HEAD_SHA"),
        "workflow_sha": environ.get("GITHUB_WORKFLOW_SHA") or environ.get("GITHUB_SHA"),
        "actor": environ.get("GITHUB_ACTOR"),
        "triggering_actor": environ.get("GITHUB_TRIGGERING_ACTOR"),
        "run_url": f"{server}/{repository}/actions/runs/{run_id}" if repository else None,
        "runner_name": environ.get("RUNNER_NAME"),
    }


def _samples(power_dir: Path) -> tuple[list[dict], list[str]]:
    """Per-scrape total and per-GPU watts from srt-slurm's samples.csv (any wire version)."""
    path = power_dir / "samples.csv"
    if not path.is_file():
        return [], []
    scrapes: dict[int, dict] = {}
    uuids: list[str] = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                seq, stamp, watts = int(row["scrape_seq"]), float(row["timestamp_unix"]), float(row["power_w"])
            except (KeyError, TypeError, ValueError):
                continue
            uuid = row.get("gpu_uuid") or f"{row.get('hostname')}/{row.get('gpu_index')}"
            if uuid not in uuids:
                uuids.append(uuid)
            scrape = scrapes.setdefault(seq, {"unix_seconds": stamp, "per_gpu_watts": {}})
            scrape["unix_seconds"] = min(scrape["unix_seconds"], stamp)
            scrape["per_gpu_watts"][uuid] = watts
    series = []
    for seq in sorted(scrapes):
        scrape = scrapes[seq]
        series.append({
            "at": _iso(scrape["unix_seconds"]), "unix_seconds": scrape["unix_seconds"],
            "scrape_seq": seq, "aggregate_watts": sum(scrape["per_gpu_watts"].values()),
            "per_gpu_watts": dict(sorted(scrape["per_gpu_watts"].items())),
        })  # fmt: skip
    return series, uuids


def _power(validation: dict, series: list[dict], run: dict, bench: dict, gpu_uuids: list[str]) -> dict:
    """Project InferenceX's validation sidecar onto the bundle's power document."""
    valid = bool(validation.get("power_valid"))
    metrics = validation.get("metrics") or {}
    window = validation.get("benchmark_window") or {}
    start, end = window.get("start_time_unix"), window.get("end_time_unix")
    within = [s for s in series if start is not None and end is not None and start <= s["unix_seconds"] <= end]
    per_gpu_energy = validation.get("per_gpu_energy_j") or {}
    per_gpu_role = validation.get("per_gpu_role") or {}
    duration = (end - start) if start is not None and end is not None else None
    per_gpu = {}
    for label, energy in per_gpu_energy.items():
        uuid = label.rsplit("/", 1)[-1]
        watts = [s["per_gpu_watts"][uuid] for s in within if uuid in s["per_gpu_watts"]]
        per_gpu[uuid] = {
            "energy_j": energy,
            "avg_power_w": energy / duration if duration else None,
            "observed_peak_power_w": max(watts) if watts else None,
            "role": per_gpu_role.get(label),
            "max_sample_gap_s": (validation.get("per_gpu_max_sample_gap_s") or {}).get(label),
        }
    summary = run.get("summary") or {}
    measurement = {
        "status": "valid" if valid else "invalid",
        "valid": valid,
        "invalid_reasons": list(validation.get("reasons") or []),
        "attempted": summary.get("scheduled", 0) - summary.get("not_started", 0),
        "completed": summary.get("completed", 0),
        "valid_clips": summary.get("valid", 0),
        "duration_seconds": duration,
        "window_count": 1,
        "valid_window_count": int(valid),
        "aggregate": {
            "avg_power_w": metrics.get("avg_total_gpu_power_w"),
            "energy_j": metrics.get("total_gpu_energy_j"),
            "joules_per_valid_clip": metrics.get("joules_per_successful_query"),
            "observed_peak_power_w": max((s["aggregate_watts"] for s in within), default=None),
        } if valid else None,
        "per_gpu": {uuid: {k: v for k, v in entry.items() if k in ("avg_power_w", "energy_j", "observed_peak_power_w")}
                    for uuid, entry in per_gpu.items()} if valid else None,
    }
    producer = validation.get("producer") or {}
    return {
        "schema_version": POWER_SCHEMA,
        "bundle_type": "h3_native_power",
        "evidence_kind": EVIDENCE_KIND,
        "source": "srt-slurm.dcgm-power",
        "status": "valid" if valid else "invalid",
        "valid": valid,
        "invalid_reasons": list(validation.get("reasons") or []),
        "gpu_uuids": gpu_uuids,
        "expected_gpu_count": validation.get("expected_gpu_count"),
        "observed_gpu_count": validation.get("observed_gpu_count"),
        "window": {
            "start_unix": start, "end_unix": end, "duration_seconds": duration,
            "clock_source": "head_node_unix_clock",
            "utc_minus_monotonic_seconds": (bench.get("clock") or {}).get("utc_minus_monotonic_seconds"),
        },
        "producer": {
            "producer_git_commit": producer.get("producer_git_commit"),
            "exporter_image_sha256": producer.get("exporter_image_sha256"),
            "integration_method": validation.get("integration_method"),
            "power_percentile_method": validation.get("power_percentile_method"),
        },
        "semantics": {
            "scope": "job GPUs sampled by srt-slurm's DCGM exporter; excludes host and unsampled devices",
            "sensor": "DCGM_FI_DEV_POWER_USAGE board power, head-node Unix clock",
            "integration": validation.get("integration_method"),
            "generation_window": "client measured block: first measurement submit to last downloaded media",
            "energy_per_valid_clip": "whole-window GPU energy divided by technically valid clips",
            "energy_unit": "J", "power_unit": "W", "time_unit": "s",
        },
        "metrics": dict(metrics),
        "per_gpu": per_gpu,
        "phases": {"measurement": measurement},
        "sample_series": [
            {"at": s["at"], "unix_seconds": s["unix_seconds"], "aggregate_watts": s["aggregate_watts"],
             "per_gpu_watts": s["per_gpu_watts"], "valid": True, "invalid_reasons": []}
            for s in series
        ],
    }


def _completion(run: dict) -> dict:
    summary = run.get("summary") or {}
    scheduled = summary.get("scheduled", 0)
    not_started = summary.get("not_started", 0)
    records = [r for r in run.get("records") or [] if r.get("phase") == "measurement"]
    finished = sum(1 for r in records if r.get("attempted") and r.get("timing_window", {}).get("end_monotonic_seconds") is not None)
    attempted = scheduled - not_started
    return {
        "scheduled": scheduled, "attempted": attempted, "completed": summary.get("completed", 0),
        "valid": summary.get("valid", 0), "failed": summary.get("failed", 0),
        "not_started": not_started, "unfinished": max(0, attempted - finished),
    }  # fmt: skip


def _copy_tree(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination, dirs_exist_ok=True)


def build_bundle(
    *,
    h3_dir: Path,
    power_dir: Path,
    workspace: Path,
    result_filename: str,
    concurrencies: list[int],
    output: Path,
    environ: dict[str, str] | None = None,
) -> dict:
    """Write the bundle under ``output``; ``status`` is complete only when every cell verified."""
    environ = dict(os.environ if environ is None else environ)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    started = _now()
    server = h3_dir / "server"
    spec = _read(server / "spec.json") if (server / "spec.json").is_file() else None
    launch = _read(server / "launch.json") if (server / "launch.json").is_file() else None
    series, sampled_uuids = _samples(power_dir)
    gpu_uuids = list(spec["gpu_uuids"]) if spec else sampled_uuids
    run_id = environ.get("GITHUB_RUN_ID", "local")
    attempt = environ.get("GITHUB_RUN_ATTEMPT", "1")
    source_sha = _source_sha(workspace, environ)
    job_status = _text(power_dir / "native-job-status.txt") or ""
    slurm_job_id, slurm_state, *rest = (job_status.split("|") + ["", ""])[:3]
    failures: list[str] = []
    cells = []
    declared_path = workspace / "video-identity.json"
    declared = _read(declared_path) if declared_path.is_file() else None
    plan = declared["workload_plan"] if declared else None
    identity = {key: declared[key] for key in ("plan_sha256", "engine")} if declared else None
    if declared:
        _write(output / "video-identity.json", declared)
    for concurrency in concurrencies:
        cell_dir = output / "gpu" / f"c{concurrency}"
        source_cell = h3_dir / "gpu" / f"c{concurrency}"
        cell: dict[str, Any] = {"concurrency": concurrency, "status": "failed", "verified": False}
        run_path = source_cell / "baseline" / "run.json"
        if not run_path.is_file():
            failures.append(f"c{concurrency}: run.json missing")
            cell["completion"] = {"scheduled": 0, "attempted": 0, "completed": 0, "valid": 0,
                                  "failed": 0, "not_started": 0, "unfinished": 0}
            cells.append(cell)
            continue
        _copy_tree(source_cell / "baseline", cell_dir / "baseline")
        run = _read(cell_dir / "baseline" / "run.json")
        configuration = run.get("configuration") or {}
        cell_identity = {"plan_sha256": run.get("plan_sha256"),
                         "engine": {"name": configuration.get("runtime"),
                                    "revision": configuration.get("runtime_revision")}}
        if identity is not None and (identity != cell_identity or plan != run.get("plan")):
            raise ValueError(f"c{concurrency}: workload or engine identity differs between cells")
        identity = cell_identity
        plan = run.get("plan")
        bench_path = source_cell / f"benchmark_concurrency_{concurrency}.json"
        bench = _read(bench_path) if bench_path.is_file() else {}
        validation_path = workspace / f"power_validation_{result_filename}_conc{concurrency}.json"
        validation = _read(validation_path) if validation_path.is_file() else {"power_valid": False, "reasons": ["validation_sidecar_missing"]}
        aggregate_path = workspace / f"{result_filename}_conc{concurrency}.json"
        aggregate = _read(aggregate_path) if aggregate_path.is_file() else None
        if bench:
            shutil.copyfile(bench_path, cell_dir / bench_path.name)
        _write(cell_dir / "power-validation.json", validation)
        if aggregate is not None:
            _write(cell_dir / "aggregate.json", aggregate)
        cell_spec = None
        if spec is not None:
            cell_spec = {**spec, "job_id": f"github-{run_id}-{attempt}-c{concurrency}",
                         "serving": {**(spec.get("serving") or {}), "concurrency": concurrency}}
            _write(cell_dir / "spec.json", cell_spec)
        power = _power(validation, series, run, bench, gpu_uuids)
        _write(cell_dir / "power.json", power)
        summary = run.get("summary") or {}
        complete = run.get("status") == "complete" and bool(summary.get("scheduled")) and summary.get("valid") == summary.get("scheduled")
        if not complete:
            failures.append(f"c{concurrency}: {summary.get('valid', 0)}/{summary.get('scheduled', 0)} valid clips")
        verified = complete and power["valid"]
        receipt = {
            "schema_version": "0.1.0",
            "bundle_type": RECEIPT_BUNDLE_TYPE,
            "evidence_kind": EVIDENCE_KIND,
            "job_id": f"github-{run_id}-{attempt}-c{concurrency}",
            "execution_id": f"slurm-{slurm_job_id or 'unknown'}",
            "spec_sha256": _hash(cell_dir / "spec.json") if cell_spec is not None else None,
            "plan_sha256": run.get("plan_sha256"),
            "status": "complete" if complete else "failed",
            "measurement_status": "complete" if complete else "incomplete",
            "measurement_verified": verified,
            "power_verified": power["valid"],
            "cleanup_status": "srt_slurm_job_teardown",
            "ci_accepted": False,
            "release_qualified": False,
            "regression_status": "inconclusive",
            "started_at": run.get("started_at"),
            "finished_at": run.get("finished_at"),
            "allocation": (spec or {}).get("allocation"),
            "authorization": (spec or {}).get("authorization"),
            "slurm": {"job_id": slurm_job_id or None, "state": slurm_state or None},
            "server": launch,
            "roles": {"baseline": {
                "run_path": "baseline/run.json", "run_sha256": _hash(cell_dir / "baseline" / "run.json"),
                "run_status": run.get("status"), "status": "complete" if complete else "failed",
                "started_at": run.get("started_at"), "finished_at": run.get("finished_at"),
                "run_summary": summary,
            }},
            "power": {"path": "power.json", "sha256": _hash(cell_dir / "power.json"), "valid": power["valid"],
                      "invalid_reasons": power["invalid_reasons"]},
            "aggregate_path": "aggregate.json" if aggregate is not None else None,
            "failures": [f for f in failures if f.startswith(f"c{concurrency}:")],
        }
        _write(cell_dir / "gpu-job.json", receipt)
        cell.update({
            "status": "complete" if complete else "failed",
            "verified": verified,
            "completion": _completion(run),
            "metrics": {
                "client_ready_p50_seconds": ((run.get("serving") or {}).get("client_ready_latency_seconds") or {}).get("p50"),
                "valid_clips_per_second": summary.get("valid_clips_per_second"),
                "latency_median_seconds": summary.get("latency_median_seconds"),
                "serving": run.get("serving"), "measurement": run.get("measurement"),
            },
            "run": {"path": f"gpu/c{concurrency}/baseline/run.json", "sha256": receipt["roles"]["baseline"]["run_sha256"]},
            "receipt": {"path": f"gpu/c{concurrency}/gpu-job.json", "sha256": _hash(cell_dir / "gpu-job.json")},
            "power": {"path": f"gpu/c{concurrency}/power.json", "sha256": receipt["power"]["sha256"], "phases": power["phases"]},
        })
        if aggregate is not None:
            cell["aggregate"] = {"path": f"gpu/c{concurrency}/aggregate.json", "sha256": _hash(cell_dir / "aggregate.json")}
        cells.append(cell)
    finished = _now()
    # Requests decide completion; power decides verification. Invalid power never publishes
    # as measured, but it fails the job only under REQUIRE_POWER (the launcher's call).
    status = "complete" if cells and all(c["status"] == "complete" for c in cells) else "failed"
    power_verified = bool(cells) and all(c["verified"] for c in cells)
    ci = _ci_context(environ)
    identity = identity or {"plan_sha256": None, "engine": None}
    artifact_name = environ.get("VIDEO_ARTIFACT_NAME")
    matrix = {
        **identity,
        "schema_version": "1.0.0", "bundle_type": BUNDLE_TYPE, "evidence_kind": EVIDENCE_KIND,
        "status": status, "power_verified": power_verified, "started_at": started, "finished_at": finished,
        "plan": plan, "runtime": (spec or {}).get("baseline"), "server": (spec or {}).get("server"),
        "gpu_uuids": gpu_uuids,
        "scheduled": sum(c["completion"]["scheduled"] for c in cells),
        "requests_per_configuration": (cells[0]["completion"]["scheduled"] if cells else 0),
        "warmup_per_configuration": (plan or {}).get("warmup_runs"),
        "ci_accepted": False, "release_qualified": False, "cells": cells,
        "completion": {key: sum(c["completion"][key] for c in cells)
                       for key in ("scheduled", "attempted", "completed", "valid", "failed", "not_started", "unfinished")},
        "failures": failures,
    }  # fmt: skip
    _write(output / "serving-smoke.json", matrix)
    ci_state = {
        "artifact_name": artifact_name,
        "schema_version": 1, "task_id": TASK_ID, "run_id": run_id, "run_attempt": attempt,
        "source_sha": source_sha, "started_at": started, "finished_at": finished,
        "phase": "complete", "mode": MODE, "launcher": "srt-slurm", "evidence_kind": EVIDENCE_KIND,
        "ci_accepted": False, "release_qualified": False, "ci": ci,
        "slurm_job": {"job_id": slurm_job_id or None, "state": slurm_state or None},
        "power": {"producer_git_commit": _text(power_dir / "power-producer-sha.txt"),
                  "exporter_image_sha256": _text(power_dir / "exporter-image.sha256")},
        "result_filename": result_filename, "concurrencies": list(concurrencies),
        "status": status, "power_verified": power_verified, "failures": failures,
    }  # fmt: skip
    _write(output / "ci.json", ci_state)
    evidence = {path.relative_to(output).as_posix(): _hash(path)
                for path in sorted(output.rglob("*.json"))}
    manifest = {
        **identity, "artifact_name": artifact_name, "recipe_fingerprint": environ.get("RECIPE_FINGERPRINT"),
        "schema_version": 1, "task_id": TASK_ID, "git_commit": source_sha, "ci": ci,
        "run_id": run_id, "run_attempt": attempt, "mode": MODE, "launcher": "srt-slurm",
        "evidence_kind": EVIDENCE_KIND, "workload_plan": plan, "gpu_uuids": gpu_uuids,
        "runtime": {"image": environ.get("IMAGE"), "python": (spec or {}).get("baseline", {}).get("python")},
        "exit_code": 0 if status == "complete" else 1,
        "evidence": evidence, "artifact_checksums": "SHA256SUMS", "excluded_persistent_caches": [],
    }  # fmt: skip
    _write(output / "manifest.json", manifest)
    lines = [f"{_hash(path)}  {path.relative_to(output).as_posix()}"
             for path in sorted(output.rglob("*")) if path.is_file()]
    (output / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    return {"status": status, "power_verified": power_verified, "output": str(output), "cells": [
        {"concurrency": c["concurrency"], "status": c["status"], "verified": c["verified"]} for c in cells
    ], "failures": failures}
