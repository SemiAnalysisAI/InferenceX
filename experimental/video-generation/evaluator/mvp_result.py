"""Versioned frontend export of a verified H3 CI artifact; never runs inference."""

from __future__ import annotations

import hashlib
import html
import math
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit

from .mvp_gpu_evidence import verify_measurement_job
from .mvp_gpu_report import _Tree, _no_symlink_parents, _pairs, _relative, _reject_constant
from .mvp_power import analyze_power
from .mvp_runner import _summary

VERSION = "1.0.0"
ROLES = ("baseline", "candidate")
DEFINITIONS = {
    "latency": {"unit": "s", "window": "request submission through downloaded, technically validated media", "population": "valid measured clips only; excludes warmup and startup"},
    "valid_clips_per_second": {"unit": "clip/s", "window": "serial measured block including failed attempts", "definition": "valid measured clips divided by measured-block wall seconds; not saturation throughput"},
    "completion": {"unit": "clip", "definition": "scheduled, attempted, completed, valid, failed and not-started measured slots; warmup separate"},
    "gpu_memory": {"unit": "MiB", "window": "role or client workload including warmup, as labelled", "definition": "maximum observed device-used memory per selected GPU, not exact allocator peaks"},
    "gpu_power": {"unit": "W", "window": "startup, warmup, and measured submit-to-terminal generation windows separately", "definition": "timestamped GPU-board sensor watts; aggregate sums selected GPUs; average is integrated energy / covered window duration; peak is observed samples"},
    "gpu_energy": {"unit": "J", "window": "same separately bounded power windows", "definition": "trapezoidal integral of timestamped GPU-board watts; includes GPU idle board draw, excludes CPU/node energy"},
    "gpu_energy_per_valid_clip": {"unit": "J/clip", "window": "measured submit-to-terminal generation windows", "definition": "energy across all attempted measured generation windows / technically valid measured clips; null for invalid coverage or zero valid clips"},
    "media_integrity": {"unit": "per-check units in original media record", "definition": "full-stream video/audio decode, geometry, timestamps, cadence, duration, motion and sound checks; not prompt adherence or human quality"},
    "paired_fidelity": {"unit": "PSNR dB, spectral cosine, absolute RMS ratio error", "definition": "aligned original decoded baseline/candidate outputs; exact video match has null finite PSNR and exact_match=true; not generative quality"},
    "hardware_tdp": {"unit": "W/GPU", "definition": "verified hardware specification, separate from configured/enforced limits and observed watts; generic H200 name does not identify form factor"},
}


def _json(tree: _Tree, name: str) -> dict:
    with tree.open(name, 8 * 1024 * 1024) as stream:
        value = json.load(stream, object_pairs_hook=_pairs, parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"{name}: expected a JSON object")
    return value


def _hash(tree: _Tree, name: str) -> str:
    with tree.open(name, 512 * 1024 * 1024) as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _inventory(root: Path, tree: _Tree) -> tuple[dict, str | None]:
    inventory = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Artifact contains a symlink")
        if path.is_file() and path != root / "SHA256SUMS":
            relative = path.relative_to(root).as_posix()
            inventory[relative] = _hash(tree, relative)
    if len(inventory) > 10000:
        raise ValueError("Artifact file inventory exceeds 10000 files")
    if not (root / "SHA256SUMS").exists():
        return inventory, None
    expected = {}
    with tree.open("SHA256SUMS", 2 * 1024 * 1024) as stream:
        for line in stream.read().decode().splitlines():
            match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
            if not match:
                raise ValueError("Malformed artifact checksum entry")
            digest, name = match.groups()
            _relative(name)
            if name in expected:
                raise ValueError("Duplicate artifact checksum path")
            expected[name] = digest
    if inventory != expected:
        raise ValueError("Artifact checksum mismatch or incomplete file inventory")
    return inventory, _hash(tree, "SHA256SUMS")


def _join(tree: _Tree, manifest: dict, ci: dict, binding: dict, spec: dict, inventory: dict, source_ci: dict | None, producer: dict) -> dict:
    if manifest.get("schema_version") != 1 or ci.get("schema_version") != 1:
        raise ValueError("Unsupported source manifest or CI version")
    for path, digest in manifest.get("evidence", {}).items():
        if inventory.get(path) != digest:
            raise ValueError(f"Manifest evidence hash mismatch: {path}")
    if (manifest.get("git_commit") != ci.get("source_sha") or manifest.get("run_id") != ci.get("run_id")
            or manifest.get("run_attempt") != ci.get("run_attempt") or manifest.get("ci") != ci.get("ci")):
        raise ValueError("Execution Git/CI provenance differs between receipts")
    job = str(binding.get("job_id"))
    step = f"{job}.{binding.get('step_id')}"
    allocation = _json(tree, "allocation.json")
    if (manifest.get("slurm_allocation") != allocation or ci.get("allocation") != allocation
            or allocation.get("identity", {}).get("JobId") != job
            or ci.get("slurm_job", {}).get("JobId") != job
            or ci.get("slurm_job", {}).get("NodeList") != binding.get("node")
            or ci.get("step_cleanup") != {"status": "ended", "step_id": step}
            or binding.get("gpu_uuids") != spec["gpu_uuids"]
            or spec["allocation"].get("label") != f"Slurm {step} on {binding.get('node')}"):
        raise ValueError("Slurm allocation/step/node/GPU provenance mismatch")
    if ci.get("allocation_cleanup", {}).get("status") not in {"released", "retained"}:
        raise ValueError("Allocation cleanup disposition is unavailable")
    if manifest.get("workload_plan") != spec["plan"]:
        raise ValueError("Outer workload plan differs from verified GPU plan")
    source = {"git_commit": manifest["git_commit"], "run_id": manifest["run_id"], "run_attempt": manifest["run_attempt"], **manifest["ci"]}
    if source_ci is not None:
        if (str(source_ci.get("databaseId")) != str(source["run_id"])
                or str(source_ci.get("runAttempt")) != str(source["run_attempt"])
                or source_ci.get("headSha") != source["git_commit"] or source_ci.get("url") != source["run_url"]):
            raise ValueError("Trusted GitHub metadata does not match source execution identity")
        current = producer.get("ci", {})
        completed = source_ci.get("status") == "completed" and source_ci.get("conclusion") == "success"
        same_run = (source_ci.get("status") == "in_progress" and source_ci.get("conclusion") is None
                    and producer.get("mode") == "same_run_export" and producer.get("git_commit") == source["git_commit"]
                    and all(str(current.get(field)) == str(source[field]) for field in ("run_id", "run_attempt", "repository")))
        if not (completed or same_run):
            raise ValueError("Source workflow is neither successful nor the exact currently exporting run")
        jobs = [job for job in source_ci.get("jobs", []) if "H3 video H200 smoke" in job.get("name", "")]
        if len(jobs) != 1 or jobs[0].get("status") != "completed" or jobs[0].get("conclusion") != "success":
            raise ValueError("Trusted source CI lacks one successful completed H3 workload job")
        source["workflow_status_at_export"] = source_ci["status"]
        source["workflow_conclusion_at_export"] = source_ci.get("conclusion")
    source["ci_accepted"] = ci.get("ci_accepted", False)
    source["external_ci_verification"] = "passed" if source_ci is not None else "not_supplied"
    return source


def _report_references(tree: _Tree, inventory: dict) -> None:
    class Links(HTMLParser):
        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            for key, value in attrs:
                if key not in {"src", "href"} or not value:
                    continue
                link = urlsplit(value)
                if link.scheme or link.netloc:
                    raise ValueError("Portable report contains an external media or asset reference")
                if link.path:
                    relative = "report/" + str(_relative(unquote(link.path)))
                    if relative not in inventory:
                        raise ValueError(f"Portable report reference is missing: {relative}")
    with tree.open("report/index.html", 8 * 1024 * 1024) as stream:
        Links().feed(stream.read().decode())


def _records(run: dict, role: str) -> list[dict]:
    return [{key: record.get(key) for key in ("slot_id", "case_id", "seed", "repetition", "phase", "status", "attempted", "latency_seconds", "submit_to_terminal_seconds", "submit_to_media_seconds", "media_validation_seconds", "media", "error")} | {
        "media_file": {"path": f"gpu/{role}/{record['artifact_path']}", "sha256": record["sha256"]} if record.get("artifact_path") else None,
    } for record in run["records"]]


def _metrics(run: dict, role: dict) -> dict:
    records = [record for record in run["records"] if record["phase"] == "measurement"]
    valid = [record for record in records if record["status"] == "succeeded" and (record.get("media") or {}).get("valid") is True]
    latencies = [record["latency_seconds"] for record in valid]
    summary = _summary(run["records"], len(records), run["measurement"]["wall_seconds"])
    telemetry = role["telemetry_summary"]
    return {
        "status": "valid", "latency_seconds": {"values": latencies, "sample_count": summary["latency_samples"],
            **{key: summary[f"latency_{key}_seconds"] for key in ("mean", "median", "min", "max", "sample_stddev")}},
        "valid_clips_per_second": summary["valid_clips_per_second"], "measurement": run["measurement"],
        "completion": {key: summary[key] for key in ("scheduled", "completed", "valid", "failed", "failed_attempts", "invalid_completed", "not_started")} | {
            "attempted": sum(row["attempted"] for row in records), "technical_success_fraction": summary["technical_success_rate"]},
        "startup_seconds": role.get("startup_seconds"),
        "gpu_memory": {"role_observed_peak_mib_by_gpu": telemetry["observed_memory_peak_mib_by_gpu"],
            "client_including_warmup_observed_peak_mib_by_gpu": telemetry.get("measurement_observed_memory_peak_mib_by_gpu"),
            "client_window": telemetry.get("measurement_window"), "start_monotonic_seconds": telemetry.get("measurement_window_start_monotonic_seconds"),
            "end_monotonic_seconds": telemetry.get("measurement_window_end_monotonic_seconds"), "coverage_qualified": telemetry["qualified"]},
    }


def _power_limits(roles: dict, gpu_uuids: list[str]) -> dict:
    """Retain contemporaneous endpoints without claiming continuous settings."""
    fields = ("configured_limit_w", "enforced_limit_w", "default_limit_w", "maximum_limit_w")
    result = {"status": "unavailable", "watts_by_gpu": None, "by_role": {},
              "scope": "role prelaunch and postcleanup snapshots; no continuous stability claim",
              "source": "gpu/gpu-job.json", "reason": "Not recorded in original execution; later probes cannot backfill historical settings"}
    statuses = []
    for label, role in roles.items():
        snapshots = {}
        for when in ("before", "after"):
            observation = role.get(f"power_configuration_{when}")
            value = {"status": "unavailable", "observed_at": None, "gpus": None, "reason": "not recorded"}
            if observation is not None:
                try:
                    stamp = datetime.fromisoformat(observation["observed_at"])
                    boundary = datetime.fromisoformat(role["started_at" if when == "before" else "finished_at"])
                    devices = observation["gpus"]
                    if (observation.get("status") != "recorded" or stamp.tzinfo is None or boundary.tzinfo is None
                            or (when == "before" and stamp > boundary) or (when == "after" and stamp < boundary)
                            or sorted(device.get("uuid") for device in devices) != sorted(gpu_uuids)
                            or any(field not in device or (device[field] is not None and (type(device[field]) not in (int, float)
                                or not math.isfinite(device[field]) or device[field] <= 0)) for device in devices for field in fields)):
                        raise ValueError("power-limit snapshot has unavailable fields or invalid time/device/value binding")
                    missing = any(device[field] is None for device in devices for field in fields)
                    value = {"status": "partial" if missing else "recorded", "observed_at": observation["observed_at"],
                             "gpus": [{"uuid": device["uuid"], **{field: device[field] for field in fields}} for device in devices],
                             "reason": "some limit readings unavailable" if missing else None}
                except (KeyError, TypeError, ValueError) as error:
                    value["reason"] = str(error)
            snapshots[when] = value
            statuses.append(value["status"])
        before, after = snapshots["before"], snapshots["after"]
        snapshots["same_observed_values"] = (sorted(before["gpus"], key=lambda item: item["uuid"]) == sorted(after["gpus"], key=lambda item: item["uuid"])
            if before["status"] == after["status"] == "recorded" else None)
        result["by_role"][label] = snapshots
    if any(status in {"recorded", "partial"} for status in statuses):
        result.update(status="recorded" if all(status == "recorded" for status in statuses) else "partial", reason=None)
    return result


def _hardware_profile(hardware: dict, profile: dict | None) -> None:
    if profile is None:
        return
    if (profile.get("schema_version") != 1 or profile.get("observation_kind") != "read_only_inventory"
            or sorted(profile.get("gpu_uuids", [])) != sorted(hardware["gpu_uuids"])
            or sorted(profile.get("slurm", {}).get("gpu_uuids", [])) != sorted(hardware["gpu_uuids"])):
        raise ValueError("Supplemental hardware inventory does not match the measured physical GPU UUIDs")
    hardware["later_hardware_observation"] = profile
    tdp = profile.get("tdp", {})
    watts = tdp.get("watts_per_gpu")
    if (tdp.get("status") == "verified" and type(watts) in (int, float) and math.isfinite(watts) and watts > 0
            and tdp.get("hardware_variant") and str(tdp.get("source_url", "")).startswith("https://") and tdp.get("evidence")):
        hardware["tdp"] = tdp | {"applies_to": "same physical GPU UUIDs; specification identity, not historic configured limits"}


def _power_report(result: dict) -> str:
    escape = lambda value: html.escape(str(value), quote=True)
    rows = []
    for role, record in result["roles"].items():
        for phase, values in record["power"]["phases"].items():
            aggregate = values.get("aggregate") or {}
            per_gpu = values.get("per_gpu") or {}
            average = "; ".join(f"{uuid}: {gpu['avg_power_w']:.1f}" for uuid, gpu in per_gpu.items()) or "withheld"
            fractions = [item["coverage"]["coverage_fraction"] for item in record["power"]["windows"] if item["phase"] == phase and item.get("coverage")]
            coverage = f"{min(fractions):.1%} minimum" if fractions else "unavailable"
            ratio = values.get("tdp_comparison", {}).get("aggregate_average_fraction_of_tdp")
            tdp_fraction = f"{ratio:.1%}" if ratio is not None else "unavailable"
            cells = [role, phase, values["status"], average, aggregate.get("avg_power_w"), tdp_fraction,
                     aggregate.get("observed_peak_power_w"), aggregate.get("joules_per_valid_clip"),
                     f"{values['valid_window_count']}/{values['window_count']}; {coverage}", ", ".join(values["invalid_reasons"]) or "none"]
            rows.append("<tr>" + "".join("<td>" + escape(f"{cell:.3f}" if isinstance(cell, float) else "withheld" if cell is None else cell) + "</td>" for cell in cells) + "</tr>")
    status = f"Export: {result['status']}; workload: {result['workload_status']}; regression: {result['regression_status']}"
    return ("<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
            "<title>H3 measured GPU power</title><style>body{font:16px system-ui;max-width:1500px;margin:2rem auto;padding:0 1rem;color:#17212b}"
            "table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #ccd2d8;padding:.6rem;text-align:left;overflow-wrap:anywhere}"
            "a{color:#075ea8}details{margin:1rem 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>H3 measured GPU power</h1><p>" + escape(status)
            + "</p><p><a href='report/index.html'>Original video and fidelity report</a> · <a href='result.json'>Frontend result</a> · "
            "<a href='power/baseline.json'>Baseline samples and coverage</a> · <a href='power/candidate.json'>Candidate samples and coverage</a></p>"
            "<p>GPU-board power for the selected devices. Generation covers submission to observed provider completion; download and local media validation are excluded. "
            "Startup and warmup are separate. A missing or unbracketed window withholds its power and energy. Peaks are observed sensor samples.</p>"
            "<table><thead><tr><th>Role</th><th>Phase</th><th>Validity</th><th>Per-GPU average W</th><th>Aggregate average W</th><th>Average / spec TDP</th>"
            "<th>Observed aggregate peak W</th><th>J / valid clip</th><th>Valid / total windows</th><th>Withholding reasons</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table><details><summary>Hardware identity, TDP and separately observed power configuration</summary><pre>"
            + escape(json.dumps(result["hardware"], indent=2)) + "</pre></details><p>" + escape(" ".join(result["limitations"])) + "</p></html>\n")


def write_result(root: Path, *, producer: dict, source_ci: dict | None = None, hardware_profile: dict | None = None) -> dict:
    """Add a frontend manifest to a copied bundle. Save failure status, then raise.

    Call before refreshing SHA256SUMS. ``producer`` identifies this exporter,
    while ``source_ci`` is independently obtained GitHub metadata for execution.
    Existing result/power files are rejected rather than silently overwritten.
    """
    root = Path(root).absolute()
    _no_symlink_parents(root)
    if (root / "result.json").exists() or (root / "power").exists() or (root / "power-report.html").exists():
        raise ValueError("Export requires a source bundle without result.json or power outputs")
    result = {"schema_version": VERSION, "bundle_type": "h3_benchmark_result", "created_at": datetime.now(timezone.utc).isoformat(),
        "producer": dict(producer), "status": "failed", "invalid_reasons": [], "workload_status": "unknown",
        "regression_status": "inconclusive", "release_qualified": False, "definitions": DEFINITIONS,
        "execution": None, "hardware": None, "workload": None, "policy": None, "roles": {}, "paired_fidelity": None,
        "files": [], "report": None, "report_links": {}, "checksums": {"path": "SHA256SUMS", "scope": "final artifact files excluding SHA256SUMS; refreshed by publisher after export"},
        "limitations": ["Point estimates from the frozen A/A schedule; no significance or performance improvement claim.",
            "Workload execution success is separate from regression calibration and release qualification.",
            "GPU-board energy excludes CPU, DRAM, other allocated GPUs and whole-node power.",
            "Matched LLM workload, topology, precision, warmup, phase boundaries, sampling and hardware limits are required before architectural comparisons."]}
    error = None
    try:
        if not isinstance(producer, dict) or not re.fullmatch(r"[0-9a-f]{40}", producer.get("git_commit", "")):
            raise ValueError("Exporter producer requires its exact Git commit")
        result["producer"].update(exporter_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            power_analyzer_source_sha256=hashlib.sha256(Path(analyze_power.__code__.co_filename).read_bytes()).hexdigest())
        tree = _Tree(root)
        inventory, checksums_sha = _inventory(root, tree)
        for label in ROLES:
            for name in ("runtime.stdout.log", "runtime.stderr.log", "client.stderr.log", "client.stdout.json"):
                if f"gpu/supervisor/{label}/{name}" not in inventory:
                    raise ValueError(f"Required {label} runtime/client log is missing: {name}")
        manifest, ci, binding = [_json(tree, name) for name in ("manifest.json", "ci.json", "binding.json")]
        verified = verify_measurement_job(root / "gpu", deadline=time.monotonic() + 120)
        spec, receipt, comparison = verified["spec"], verified["receipt"], verified["comparison"]
        source = _join(tree, manifest, ci, binding, spec, inventory, source_ci, producer)
        result["execution"] = {"ci": source, "slurm": binding, "source_input_checksums_sha256": checksums_sha,
            "source_manifest": {"path": "manifest.json", "sha256": inventory["manifest.json"]}, "model": receipt["model_identity"],
            "runtime": {role: {key: receipt["roles"][role]["source_identity"].get(key) for key in ("revision", "source_sha256", "python_sha256", "python_version", "packages")} for role in ROLES},
            "supervisor_source_sha256": receipt.get("supervisor_source_sha256"), "cleanup_status": receipt["cleanup_status"]}
        allocation = re.search(r"(?:^|,)gres/gpu=(\d+)(?:,|$)", ci["slurm_job"].get("AllocTRES", ""))
        result["hardware"] = {"selected_gpu_count": len(spec["gpu_uuids"]), "reserved_gpu_count": int(allocation[1]) if allocation else None,
            "gpu_uuids": spec["gpu_uuids"], "devices": receipt["roles"]["baseline"]["telemetry_summary"]["gpu_identity"],
            "tdp": {"status": "unavailable", "watts_per_gpu": None, "reason": "No verified form-factor-specific TDP evidence in source execution"},
            "configured_power_limits": _power_limits(receipt["roles"], spec["gpu_uuids"])}
        _hardware_profile(result["hardware"], hardware_profile)
        result["workload"] = {"plan": spec["plan"], "plan_sha256": receipt["plan_sha256"], "server": spec["server"], "comparison": "same_revision_A/A" if spec["baseline"]["revision"] == spec["candidate"]["revision"] else "baseline_candidate"}
        result["policy"] = spec["policy"]
        (root / "power").mkdir()
        for label in ROLES:
            role, run = receipt["roles"][label], verified["runs"][label]
            with tree.open(f"gpu/{role['telemetry_path']}", 128 * 1024 * 1024) as stream:
                samples = [json.loads(line, object_pairs_hook=_pairs, parse_constant=_reject_constant) for line in stream]
            with tree.open(f"gpu/{label}/events.jsonl", 16 * 1024 * 1024) as stream:
                events = [json.loads(line, object_pairs_hook=_pairs, parse_constant=_reject_constant) for line in stream]
            power = analyze_power(role, run, samples, events, spec["gpu_uuids"], interval_seconds=spec["limits"]["telemetry_interval_seconds"])
            tdp = result["hardware"]["tdp"]
            if tdp["status"] == "verified":
                denominator = tdp["watts_per_gpu"] * len(spec["gpu_uuids"])
                for phase in power["phases"].values():
                    aggregate = phase.get("aggregate")
                    phase["tdp_comparison"] = {"status": "valid" if aggregate else "withheld", "specification_watts_per_gpu": tdp["watts_per_gpu"],
                        "aggregate_average_fraction_of_tdp": aggregate["avg_power_w"] / denominator if aggregate else None,
                        "observed_aggregate_peak_fraction_of_tdp": aggregate["observed_peak_power_w"] / denominator if aggregate else None,
                        "interpretation": "descriptive ratios only; TDP is not a measured or configured power limit"}
            power_name = f"power/{label}.json"
            (root / power_name).write_text(json.dumps(power, indent=2, allow_nan=False) + "\n")
            inventory[power_name] = _hash(tree, power_name)
            result["roles"][label] = {"run_id": run["run_id"], "metrics": _metrics(run, role), "records": _records(run, label),
                "power": {"path": power_name, "sha256": inventory[power_name], "status": power["status"], "schema_version": power["schema_version"], "phases": power["phases"],
                    "windows": [{key: window.get(key) for key in ("phase", "slot_id", "coverage", "timing_source", "timing_uncertainty_seconds", "invalid_reasons")} for window in power["windows"]]},
                "raw_telemetry": {"path": f"gpu/{role['telemetry_path']}", "sha256": role["telemetry_sha256"]},
                "media_evaluator": run["configuration"].get("media_evaluator")}
        result["paired_fidelity"] = {"source": "gpu/comparison.json", "summary": comparison.get("summary"), "checks": comparison["checks"],
            "slots": [{key: slot.get(key) for key in ("slot_id", "case_id", "seed", "repetition", "status", "metrics", "checks")} for slot in comparison["slots"]]}
        if "report/index.html" not in inventory:
            raise ValueError("Portable report is missing")
        _report_references(tree, inventory)
        result["report"] = {"path": "report/index.html", "sha256": inventory["report/index.html"]}
        result["files"] = [{"path": name, "sha256": digest} for name, digest in sorted(inventory.items())]
        result["regression_status"] = receipt.get("regression_status", "inconclusive")
        complete = all(run["summary"]["valid"] == run["summary"]["scheduled"] > 0 for run in verified["runs"].values())
        source_exit = (manifest.get("exit_code"), ci.get("exit_code"), _json(tree, "step-result.json").get("exit_code"))
        result["workload_status"] = "passed" if complete and all(code == 0 for code in source_exit) else "failed"
        if result["workload_status"] != "passed":
            raise ValueError("Original workload or Slurm payload has unsuccessful exit status")
        result["status"] = "complete"
        (root / "power-report.html").write_text(_power_report(result))
        inventory["power-report.html"] = _hash(tree, "power-report.html")
        result["report_links"] = {"original": result["report"], "power": {"path": "power-report.html", "sha256": inventory["power-report.html"]}}
        result["files"] = [{"path": name, "sha256": digest} for name, digest in sorted(inventory.items())]
    except (OSError, ValueError, KeyError, TypeError) as caught:
        error = caught
        result["invalid_reasons"].append(str(caught))
        result["status"] = "failed"
        for role in result["roles"].values():
            role["metrics"] = {"status": "withheld", "reason": "Artifact export failed validation; consult original evidence"}
    (root / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if error is not None:
        raise ValueError(f"H3 result export failed: {error}") from error
    return result
