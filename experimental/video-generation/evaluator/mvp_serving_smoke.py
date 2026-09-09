"""Bounded C1/C2/C4 smoke using the existing supervisor and raw run contract."""

from __future__ import annotations

import copy
import html
import json
import time
from pathlib import Path
from urllib.parse import quote

from . import mvp_gpu_job as gpu
from .mvp_gpu_evidence import verify_measurement_job
from .mvp_report import _CSS, _number
from .mvp_serving import settings
from .mvp_runner import _summary

CONCURRENCIES = (1, 2, 4)


def validate_spec(spec: dict) -> dict:
    frozen = gpu.validate_gpu_job(spec)
    if not frozen.get("serving") or len(frozen["plan"]["cases"]) * frozen["plan"]["repetitions"] != 4:
        raise ValueError("serving smoke requires exactly four measured requests per configuration")
    return frozen


def _report(root: Path, matrix: dict) -> None:
    rows, media = [], []
    for cell in matrix["cells"]:
        summary = cell["completion"]
        metrics = cell.get("metrics") or {}
        rows.append("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in (
            cell["concurrency"], cell["status"], summary["scheduled"], summary["attempted"],
            summary["valid"], summary["failed"], summary["not_started"],
            _number(metrics.get("client_ready_p50_seconds")), _number(metrics.get("valid_clips_per_second")),
        )) + "</tr>")
        if cell.get("verified"):
            run = gpu._read(root / cell["run"]["path"])
            base = Path(cell["run"]["path"]).parent
            for record in run["records"]:
                if record["phase"] == "measurement" and record.get("artifact_path"):
                    url = "../" + quote((base / record["artifact_path"]).as_posix(), safe="/")
                    label = html.escape(f"C{cell['concurrency']} · {record['slot_id']} · {record['outcome']}")
                    media.append(f'<figure><video controls preload="metadata" src="{url}"></video><figcaption>{label}</figcaption></figure>')
    report = root / "report"
    report.mkdir(exist_ok=True)
    (report / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>H3 serving smoke</title><style>{_CSS}</style><main><h1>H3 serving smoke</h1>'
        '<p>One hardware configuration; four measured requests at each concurrency. Warmups are separate. '
        'Latency is submit → downloaded media for technically valid clips. Throughput is valid clips / delivery wall seconds. '
        'Four samples do not establish P90/P95 or sustainable capacity. Failed and unstarted requests remain counted.</p>'
        '<p><a href="../serving-smoke.json" download>Download summary and raw-evidence links</a></p>'
        '<div class="panel table-wrap"><table><thead><tr><th>Concurrency</th><th>Status</th><th>Scheduled</th>'
        '<th>Attempted</th><th>Valid</th><th>Failed</th><th>Not started</th><th>Delivery median (s)</th><th>Valid clips/s</th>'
        '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>'
        '<details><summary>Frozen workload and runtime</summary><pre>'
        + html.escape(json.dumps({key: matrix[key] for key in ("plan", "runtime", "gpu_uuids")}, indent=2))
        + '</pre></details><section class="media-grid">' + ''.join(media) + '</section></main>', encoding="utf-8")


def run_matrix(spec: dict, root: Path) -> dict:
    from .mvp_power import analyze_power

    spec = validate_spec(spec)
    deadline = time.monotonic() + spec["limits"]["job_seconds"]
    matrix = {"schema_version": "1.0.0", "bundle_type": "h3_serving_smoke_matrix", "status": "running",
              "started_at": gpu._now(), "plan": spec["plan"], "runtime": spec["baseline"], "gpu_uuids": spec["gpu_uuids"],
              "scheduled": 12, "warmup_per_configuration": spec["plan"]["warmup_runs"],
              "ci_accepted": False, "release_qualified": False,
              "cells": [{"concurrency": concurrency, "status": "not_started", "verified": False,
                         "completion": {"scheduled": 4, "attempted": 0, "completed": 0, "valid": 0, "failed": 4, "not_started": 4, "unfinished": 0}}
                        for concurrency in CONCURRENCIES]}
    path = root / "serving-smoke.json"
    gpu._write(path, matrix)
    try:
        for cell in matrix["cells"]:
            remaining = deadline - time.monotonic()
            if remaining <= max(spec["limits"]["startup_seconds"], spec["limits"]["request_seconds"]) + 2 * spec["limits"]["cleanup_seconds"]:
                raise TimeoutError("remaining matrix budget cannot bound another configuration")
            current = copy.deepcopy(spec)
            current["limits"]["job_seconds"] = remaining
            current["serving"] = settings(cell["concurrency"], spec["serving"]["delivery_deadline_seconds"])
            current["job_id"] = f"{spec['job_id']}-c{cell['concurrency']}"
            directory = root / "gpu" / f"c{cell['concurrency']}"
            cell["status"] = "running"
            gpu._write(path, matrix)
            receipt = gpu.run_gpu_job(current, directory, serving_smoke=True)
            cell["status"] = "failed"
            cell["receipt"] = {"path": (directory / "gpu-job.json").relative_to(root).as_posix(), "sha256": gpu._hash(directory / "gpu-job.json")}
            run_path = directory / "baseline/run.json"
            if run_path.is_file():
                raw = gpu._read(run_path)
                cell["run"] = {"path": run_path.relative_to(root).as_posix(), "sha256": gpu._hash(run_path)}
                summary = _summary(raw["records"], 4, raw["measurement"]["wall_seconds"])
                cell["completion"] = {key: summary[key] for key in ("scheduled", "completed", "valid", "failed")}
                finished = {r["slot_id"] for r in raw["records"] if r["phase"] == "measurement" and r["attempted"]}
                journal = directory / "baseline/events.jsonl"
                events = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
                started = finished | {event["slot_id"] for event in events if event["event"] == "attempt_started" and event["slot_id"].startswith("measurement-")}
                cell["completion"].update(attempted=len(started), not_started=4-len(started), unfinished=len(started-finished))
            gpu._write(path, matrix)
            verified = verify_measurement_job(directory, deadline=deadline, require_success=True, serving_smoke=True)
            run, role = verified["runs"]["baseline"], receipt["roles"]["baseline"]
            cell.update(status="complete", verified=True, metrics={
                "client_ready_p50_seconds": run["serving"]["client_ready_latency_seconds"]["p50"],
                "valid_clips_per_second": _summary(run["records"], 4, run["measurement"]["wall_seconds"])["valid_clips_per_second"],
                "serving": run["serving"], "measurement": run["measurement"],
            })
            samples = [json.loads(line) for line in (directory / role["telemetry_path"]).read_text().splitlines()]
            events = [json.loads(line) for line in (directory / "baseline/events.jsonl").read_text().splitlines()]
            power = analyze_power(role, run, samples, events, spec["gpu_uuids"], interval_seconds=spec["limits"]["telemetry_interval_seconds"])
            power_path = directory / "power.json"
            gpu._write(power_path, power)
            cell["power"] = {"path": power_path.relative_to(root).as_posix(), "sha256": gpu._hash(power_path), "phases": power["phases"]}
            gpu._write(path, matrix)
        matrix["status"] = "complete"
    except (Exception, KeyboardInterrupt) as error:
        matrix.update(status="failed", error=str(error) if isinstance(error, (ValueError, RuntimeError, TimeoutError)) else type(error).__name__)
        for cell in matrix["cells"]:
            if cell["status"] == "running":
                cell["status"] = "failed"
    finally:
        matrix["finished_at"] = gpu._now()
        matrix["completion"] = {key: sum(cell["completion"][key] for cell in matrix["cells"])
                                for key in ("scheduled", "attempted", "completed", "valid", "failed", "not_started", "unfinished")}
        gpu._write(path, matrix)
        _report(root, matrix)
    return matrix
