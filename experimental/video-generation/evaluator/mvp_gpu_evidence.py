"""Bounded, nonrecursive integrity checks for trusted-runner GPU evidence.

No models, media decoders, subprocesses, network calls, or live GPUs are used.
Media bytes are hashed and rebound to the recorded full-stream comparison. This
does not turn unsigned operator artifacts into independent hardware attestation.
"""

from __future__ import annotations

import json
import math
import os
import stat
from datetime import datetime
from pathlib import Path

from .mvp_runner import _slots, canonical_json_bytes


def _finite(value, *, positive=False):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def _equal(left, right):
    return canonical_json_bytes(left) == canonical_json_bytes(right)


def _date(value):
    if not isinstance(value, str):
        raise ValueError("evidence timestamp is missing")
    timestamp = datetime.fromisoformat(value)
    if timestamp.tzinfo is None:
        raise ValueError("evidence timestamp must include its timezone")
    return timestamp


def _file(root: Path, raw, *, required=None) -> Path:
    if not isinstance(raw, str) or not raw or Path(raw).is_absolute() or ".." in Path(raw).parts or "\\" in raw:
        raise ValueError("evidence path is not relative and contained")
    if required is not None and raw != required:
        raise ValueError("evidence path differs from supervisor-owned artifact layout")
    path = (root / raw).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("evidence file escapes its job directory")
    return path


def _timing(record):
    fields = ("submit_to_terminal_seconds", "submit_to_media_seconds", "media_validation_seconds", "latency_seconds")
    if not all(_finite(record.get(key)) for key in fields) or not _finite(record["latency_seconds"], positive=True):
        raise ValueError("valid measured media lacks real, finite recorded timing components")
    terminal, delivered, validation, end = [record[key] for key in fields]
    if terminal > delivered or delivered + validation > end + max(1e-6, end * 1e-6):
        raise ValueError("recorded timing components are not ordered within their measured boundary")


def verify_measurement_job(directory: Path, *, deadline: float, require_success: bool = False, serving_smoke: bool = False) -> dict:
    """Verify one complete controlled job; never follow its calibration references."""
    from . import mvp_gpu_job as gpu

    directory = Path(directory).resolve(strict=True)
    gpu._check_deadline(deadline)
    spec = gpu.validate_gpu_job(gpu._read(_file(directory, "spec.json")))
    receipt = gpu._read(_file(directory, "gpu-job.json"))
    bundle_type = "controlled_serving_smoke" if serving_smoke else "controlled_gpu_job"
    if serving_smoke and not spec.get("serving"):
        raise ValueError("single-runtime smoke requires an explicit serving load")
    if receipt.get("schema_version") != gpu.VERSION or receipt.get("bundle_type") != bundle_type or receipt.get("spec_sha256") != gpu._digest(spec):
        raise ValueError("GPU job/spec identity is not verified")
    if receipt.get("job_id") != spec["job_id"] or receipt.get("plan_sha256") != gpu._digest(spec["plan"]):
        raise ValueError("GPU job workload identity mismatch")
    if receipt.get("evidence_kind") != "controlled_h3_gpu" or receipt.get("status") != "complete" or receipt.get("measurement_status") != "complete" or receipt.get("failures"):
        raise ValueError("GPU job has incomplete, fixture, or failed supervision evidence")
    if receipt.get("cleanup_status") != "clean":
        raise ValueError("GPU job did not complete clean owned-resource teardown")
    started, finished = _date(receipt.get("started_at")), _date(receipt.get("finished_at"))
    if finished < started:
        raise ValueError("GPU job timestamps are reversed")
    execution_id = receipt.get("execution_id")
    if not isinstance(execution_id, str) or not execution_id:
        raise ValueError("GPU job lacks its independent execution identity")
    model = receipt.get("model_identity", {})
    if model.get("revision") != spec["model"]["revision"] or model.get("manifest_sha256") != gpu._digest(spec["model"]["files"]):
        raise ValueError("GPU job checkpoint manifest is not bound to the specification")
    labels = ("baseline",) if serving_smoke else ("baseline", "candidate")
    if set(receipt.get("roles", {})) != set(labels):
        raise ValueError("the exact supervised role receipts are required")
    runs, identities, run_ids, nonces = {}, {}, set(), set()
    expected_slots = _slots(spec["plan"])
    scheduled = sum(slot["phase"] == "measurement" for slot in expected_slots)
    for label in labels:
        gpu._check_deadline(deadline)
        role = receipt["roles"][label]
        if role.get("status") != "complete" or role.get("client_exit_code") not in {0, 1}:
            raise ValueError("role client did not finalize its entire workload")
        if role.get("cleanup", {}).get("status") != "clean" or role["cleanup"].get("idle_after") is not True or role.get("client_cleanup", {}).get("status") != "clean":
            raise ValueError("role/client resource cleanup is not verified clean")
        before, after = role.get("gpu_before", {}), role["cleanup"].get("gpu_after", {})
        for idle in (before, after):
            if sorted(item.get("uuid") for item in idle.get("gpus", [])) != sorted(spec["gpu_uuids"]) or not gpu._idle(idle, spec["limits"]["max_idle_memory_mib"]):
                raise ValueError("GPU idle observations do not match the selected devices")
        process = role.get("process_identity", {})
        if not all(isinstance(process.get(field), int) and not isinstance(process[field], bool) and process[field] > 0 for field in ("pid", "pgid", "session_id", "start_ticks")) or process["pid"] != process["pgid"] or process["pid"] != process["session_id"]:
            raise ValueError("owned runtime session identity is missing or invalid")
        nonce = process.get("launch_nonce")
        if not isinstance(nonce, str) or not nonce or nonce in nonces:
            raise ValueError("runtime boot identities are not distinct")
        nonces.add(nonce)
        identity = role.get("source_identity", {})
        if any(identity.get(field) != spec[label][field] for field in ("revision", "source_sha256")) or not identity.get("python_sha256") or not identity.get("packages", {}).get("torch"):
            raise ValueError("observed runtime source/environment is not pinned")
        identities[label] = identity
        run_path = _file(directory, role.get("run_path"), required=f"{label}/run.json")
        if gpu._hash(run_path, deadline) != role.get("run_sha256"):
            raise ValueError("role run bundle hash mismatch")
        run = gpu._read(run_path)
        if run.get("bundle_type") != "mvp_run" or run.get("bundle_version") != "0.1.0" or run.get("evidence_kind") not in {"live_h3", "operator_endpoint"}:
            raise ValueError("role contains fixture/imported or unsupported run evidence")
        if run.get("status") not in {"complete", "partial", "failed"} or _date(run.get("finished_at")) < _date(run.get("started_at")):
            raise ValueError("role run is not finalized")
        run_id = run.get("run_id")
        if not isinstance(run_id, str) or not run_id or run_id in run_ids:
            raise ValueError("run execution IDs are missing or reused")
        run_ids.add(run_id)
        if run.get("plan_sha256") != gpu._digest(spec["plan"]) or not _equal(run.get("plan"), spec["plan"]):
            raise ValueError("run workload differs from supervised frozen plan")
        config = run.get("configuration", {})
        declared = config.get("configuration_sha256") or run.get("configuration_sha256")
        unsigned = {key: value for key, value in config.items() if key != "configuration_sha256"}
        if declared != gpu._digest(unsigned):
            raise ValueError("run configuration hash mismatch")
        if config.get("runtime") != "sglang" or config.get("runtime_revision") != identity["revision"] or config.get("model_id") != spec["plan"]["model_id"] or config.get("model_revision") != spec["model"]["revision"]:
            raise ValueError("run configuration differs from observed runtime/model identity")
        records = run.get("records", [])
        if not isinstance(records, list) or len(records) != len(expected_slots):
            raise ValueError("run must retain every planned warmup and measured outcome")
        valid_measured = 0
        attempted_seconds = 0.0
        for record, slot in zip(records, expected_slots):
            gpu._check_deadline(deadline)
            if any(not _equal(record.get(field), slot[field]) for field in ("slot_id", "case_id", "prompt", "seed", "repetition", "phase")):
                raise ValueError("run slot identity/order differs from the frozen schedule")
            if record.get("status") not in {"succeeded", "failed"} or not isinstance(record.get("attempted"), bool):
                raise ValueError("run outcome or attempted flag is missing")
            if record.get("artifact_path") is not None:
                artifact = _file(run_path.parent, record["artifact_path"])
                if gpu._hash(artifact, deadline) != record.get("sha256"):
                    raise ValueError("raw model media bytes differ from their recorded hash")
            elif record.get("sha256") or record.get("status") == "succeeded":
                raise ValueError("successful/hashed record is missing its media artifact")
            valid = record.get("status") == "succeeded" and record.get("media", {}).get("valid") is True
            if record["phase"] == "warmup" and not valid:
                raise ValueError("planned warmup failed; measurement is not qualified")
            if valid:
                _timing(record)
            if record["phase"] == "measurement":
                valid_measured += valid
                if record["attempted"]:
                    if not _finite(record.get("latency_seconds")):
                        raise ValueError("attempted record lacks finite measured client time")
                    attempted_seconds += record["latency_seconds"]
                if (require_success or label == "baseline") and not valid:
                    raise ValueError("same-build calibration or baseline requires every measured slot valid")
        summary = run.get("summary", {})
        if summary.get("scheduled") != scheduled or summary.get("valid") != valid_measured or summary.get("failed") != scheduled - valid_measured:
            raise ValueError("run failure/success denominator does not match raw outcomes")
        measurement = run.get("measurement", {})
        wall = measurement.get("wall_seconds")
        if not _equal(config.get("serving"), spec.get("serving")):
            raise ValueError("client serving load differs from supervised specification")
        if config.get("serving"):
            from .mvp_serving import summarize, validate_window
            validate_window(run)
            if not _equal(run.get("serving"), summarize(run)):
                raise ValueError("serving summary differs from raw request records")
        elif measurement.get("boundary") != "submit_to_validated_media" or measurement.get("concurrency") != 1 or not _finite(wall, positive=True) or wall + max(1e-6, attempted_seconds * 1e-6) < attempted_seconds:
            raise ValueError("run timing boundary or serial wall duration is invalid")
        run["_verified_run_sha256"] = role["run_sha256"]
        runs[label] = run
        telemetry_path = _file(directory, role.get("telemetry_path"), required=f"supervisor/{label}/telemetry.jsonl")
        if telemetry_path.stat().st_size > 128 * 1024 * 1024 or gpu._hash(telemetry_path, deadline) != role.get("telemetry_sha256"):
            raise ValueError("raw telemetry is oversized or has a mismatched hash")
        samples = []
        previous_time = None
        descriptor = os.open(telemetry_path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("raw telemetry must be a regular file")
            while True:
                gpu._check_deadline(deadline)
                line = stream.readline(1024 * 1024 + 1)
                if not line:
                    break
                if len(line) > 1024 * 1024 or not line.endswith(b"\n") or len(samples) >= 100000:
                    raise ValueError("raw telemetry has a truncated or oversized sample")
                sample = json.loads(line)
                stamp = sample.get("monotonic_seconds")
                if not _finite(stamp) or (previous_time is not None and stamp <= previous_time) or sample.get("phase") not in {"startup", "measurement", "cleanup"}:
                    raise ValueError("raw telemetry sample order/phase is invalid")
                previous_time = stamp
                devices = sample.get("gpus", [])
                if sorted(item.get("uuid") for item in devices) != sorted(spec["gpu_uuids"]):
                    raise ValueError("raw telemetry device inventory differs from selected UUIDs")
                if any(not _finite(item.get("memory_used_mib")) or not _finite(item.get("memory_total_mib"), positive=True) or not _finite(item.get("utilization_percent")) for item in devices):
                    raise ValueError("raw telemetry contains invalid required measurements")
                if sample.get("unowned_compute_apps"):
                    raise ValueError("raw telemetry records unowned/invisible GPU compute")
                original = {(app.get("gpu_uuid"), app.get("pid")) for app in sample.get("compute_apps", [])}
                observed = sample.get("owned_compute_apps", [])
                if original != {(app.get("gpu_uuid"), app.get("pid")) for app in observed}:
                    raise ValueError("raw GPU compute inventory is not fully attributed")
                for app in observed:
                    observed_process = app.get("process_identity", {})
                    if app.get("gpu_uuid") not in spec["gpu_uuids"] or observed_process.get("pid") != app.get("pid") or any(observed_process.get(field) != process[field] for field in ("pgid", "session_id")) or not _finite(observed_process.get("start_ticks"), positive=True) or observed_process["start_ticks"] < process["start_ticks"]:
                        raise ValueError("raw GPU compute process is outside the observed owned session")
                samples.append(sample)
        declared_summary = role.get("telemetry_summary", {})
        window_start = declared_summary.get("measurement_window_start_monotonic_seconds")
        window_end = declared_summary.get("measurement_window_end_monotonic_seconds")
        if not _finite(window_start) or not _finite(window_end) or window_end - window_start + 1e-6 < wall:
            raise ValueError("telemetry window does not cover the measured client workload")
        recomputed = gpu.summarize_gpu_samples(samples, spec["gpu_uuids"], spec["limits"]["telemetry_interval_seconds"],
                                               window_start=window_start, window_end=window_end,
                                               command_seconds=spec["limits"]["command_seconds"])
        if not recomputed["qualified"] or not _equal(recomputed, role.get("telemetry_summary")):
            raise ValueError("telemetry summary does not match qualified raw observations")
    verified = {"directory": directory, "spec": spec, "receipt": receipt, "runs": runs, "identities": identities, "run_ids": run_ids,
                "nonces": nonces, "started": started, "finished": finished, "execution_id": execution_id}
    if serving_smoke:
        if receipt.get("comparison_path") is not None or receipt.get("ci_accepted") is not False or receipt.get("release_qualified") is not False:
            raise ValueError("single-runtime smoke cannot claim paired or calibrated acceptance")
        return {**verified, "comparison": None}
    compared_path = _file(directory, receipt.get("comparison_path"), required="comparison.json")
    if gpu._hash(compared_path, deadline) != receipt.get("comparison_sha256"):
        raise ValueError("comparison artifact hash mismatch")
    compared = gpu._read(compared_path)
    if compared.get("bundle_type") != "mvp_comparison" or compared.get("evidence_kind") not in {"operator_endpoint", "live_h3"} or compared.get("plan_sha256") != gpu._digest(spec["plan"]) or not _equal(compared.get("policy"), spec["policy"]):
        raise ValueError("comparison evidence/workload/policy is not bound to this GPU job")
    slots = compared.get("slots", [])
    if len(slots) != scheduled:
        raise ValueError("comparison does not retain every planned measurement pair")
    for label in ("baseline", "candidate"):
        bound = compared.get(label, {})
        if bound.get("run_bundle_sha256") != runs[label]["_verified_run_sha256"] or bound.get("run_id") != runs[label]["run_id"]:
            raise ValueError("comparison references different baseline/candidate executions")
        measured_records = [record for record in runs[label]["records"] if record["phase"] == "measurement"]
        for slot, record in zip(slots, measured_records):
            observation = slot.get(label, {})
            if slot.get("slot_id") != record["slot_id"] or observation.get("sha256") != record.get("sha256") or observation.get("status") != record["status"]:
                raise ValueError("comparison media/outcomes differ from the measured run bundle")
    checks = compared.get("checks", []) + [check for slot in slots for check in slot.get("checks", [])]
    if not checks:
        raise ValueError("comparison contains no checks")
    outcome = "fail" if any(check.get("status") == "fail" for check in checks) else "inconclusive" if any(check.get("status") == "inconclusive" for check in checks) else "pass"
    if compared.get("overall_status") != outcome:
        raise ValueError("comparison decision does not match its checks")
    return {**verified, "comparison": compared}


def verify_calibration(spec: dict, current: dict, *, deadline: float) -> tuple[bool, str]:
    """Verify 2–4 prior same-build jobs without following nested references."""
    from . import mvp_gpu_job as gpu

    if spec.get("serving"):
        return False, "serving load is descriptive; serial calibration cannot qualify concurrent delivery metrics"
    if spec["policy"]["calibration_status"] != "operator_calibrated":
        return False, "policy is not calibrated; measurements are useful but CI acceptance is inconclusive"
    references = spec["policy"].get("calibration_evidence")
    if not isinstance(references, list) or not 2 <= len(references) <= 4:
        return False, "calibrated CI requires 2–4 hash-pinned independent same-build GPU jobs"
    seen_jobs = {current["execution_id"]}
    seen_runs = set(current["run_ids"])
    seen_nonces = set(current["nonces"])
    intervals = []
    for reference in references:
        gpu._check_deadline(deadline)
        if not isinstance(reference, dict) or set(reference) != {"job_path", "sha256"}:
            raise ValueError("calibration reference must identify one prior full job directory receipt")
        path = Path(reference["job_path"])
        if not isinstance(reference["sha256"], str) or not gpu._SHA.fullmatch(reference["sha256"]):
            raise ValueError("calibration reference must pin a SHA256")
        portable = current["directory"] / "calibration" / reference["sha256"] / "gpu-job.json"
        if not path.is_absolute() or path.name != "gpu-job.json":
            raise ValueError("calibration reference requires an absolute prior GPU receipt path")
        if portable.exists():
            path = _file(current["directory"], portable.relative_to(current["directory"]).as_posix())
        if gpu._hash(path, deadline) != reference["sha256"]:
            raise ValueError("calibration receipt path or hash did not verify")
        prior = verify_measurement_job(path.parent, deadline=deadline, require_success=True)
        frozen = prior["spec"]
        if prior["execution_id"] in seen_jobs or prior["run_ids"] & seen_runs or prior["nonces"] & seen_nonces:
            raise ValueError("calibration repeats an execution, run, or runtime boot identity")
        seen_jobs.add(prior["execution_id"])
        seen_runs.update(prior["run_ids"])
        seen_nonces.update(prior["nonces"])
        if prior["finished"] >= current["started"]:
            raise ValueError("calibration must predate the independent candidate job")
        if any(not (prior["finished"] <= start or end <= prior["started"]) for start, end in intervals):
            raise ValueError("calibration jobs overlap on the same GPUs")
        intervals.append((prior["started"], prior["finished"]))
        for field in ("plan", "server", "gpu_uuids"):
            if not _equal(frozen[field], spec[field]):
                raise ValueError(f"calibration {field} differs from current controlled cell")
        if frozen["model"]["revision"] != spec["model"]["revision"] or not _equal(frozen["model"]["files"], spec["model"]["files"]):
            raise ValueError("calibration uses a different checkpoint manifest")
        for label in ("baseline", "candidate"):
            identity = prior["identities"][label]
            baseline = current["identities"]["baseline"]
            for field in ("revision", "source_sha256", "python_sha256", "python_version", "packages"):
                if not _equal(identity.get(field), baseline.get(field)):
                    raise ValueError("calibration is not the current baseline build and dependency environment")
            for field in ("client_source_sha256", "client_environment", "media_evaluator", "measurement_semantics", "limits"):
                if not _equal(prior["runs"][label]["configuration"].get(field), current["runs"]["baseline"]["configuration"].get(field)):
                    raise ValueError("calibration client timing/evaluator environment differs")
            observed = prior["receipt"]["roles"][label]["telemetry_summary"]["gpu_identity"]
            if not _equal(observed, current["receipt"]["roles"]["baseline"]["telemetry_summary"]["gpu_identity"]):
                raise ValueError("calibration observed hardware or driver identity differs")
    return True, "eligibility verified against independent prior same-build raw artifacts; threshold choice remains operator-declared, not statistical certification"


def preflight_calibration(spec: dict, directory: Path, *, started_at: str,
                          execution_id: str, deadline: float) -> dict:
    """Reject known-invalid calibration before allocating any current GPUs.

    This checks prior raw evidence and every cell property known from the frozen
    specification. Runtime-package/driver observations are still compared again
    after the current job; preflight cannot predict a later environment drift.
    """
    from . import mvp_gpu_job as gpu

    if spec["policy"]["calibration_status"] != "operator_calibrated":
        return {"status": "not_required", "performed_before_gpu_lease": True,
                "reason": "policy does not claim calibrated CI acceptance"}
    references = spec["policy"].get("calibration_evidence")
    if not isinstance(references, list) or not 2 <= len(references) <= 4:
        raise ValueError("calibration preflight requires 2–4 hash-pinned complete prior jobs")
    directory = Path(directory).resolve(strict=True)
    started = _date(started_at)
    jobs, runs, nonces, intervals, hashes = {execution_id}, set(), set(), [], []
    for reference in references:
        gpu._check_deadline(deadline)
        if not isinstance(reference, dict) or set(reference) != {"job_path", "sha256"}:
            raise ValueError("calibration preflight requires explicit receipt path/hash pairs")
        digest = reference["sha256"]
        if not isinstance(digest, str) or not gpu._SHA.fullmatch(digest):
            raise ValueError("calibration preflight reference lacks a valid SHA256")
        raw = reference["job_path"]
        if not isinstance(raw, str):
            raise ValueError("calibration preflight reference path must be a string")
        path = Path(raw)
        if not path.is_absolute() or path.name != "gpu-job.json":
            raise ValueError("calibration preflight needs an absolute prior GPU receipt path")
        portable = directory / "calibration" / digest / "gpu-job.json"
        if portable.exists():
            path = _file(directory, portable.relative_to(directory).as_posix())
        if gpu._hash(path, deadline) != digest:
            raise ValueError("calibration preflight receipt hash does not match")
        prior = verify_measurement_job(path.parent, deadline=deadline, require_success=True)
        if prior["execution_id"] in jobs or prior["run_ids"] & runs or prior["nonces"] & nonces:
            raise ValueError("calibration preflight found repeated job/run/boot identities")
        jobs.add(prior["execution_id"])
        runs.update(prior["run_ids"])
        nonces.update(prior["nonces"])
        if prior["finished"] >= started:
            raise ValueError("calibration preflight requires prior jobs completed before this job")
        if any(not (prior["finished"] <= start or end <= prior["started"]) for start, end in intervals):
            raise ValueError("calibration preflight jobs overlap on the same GPUs")
        intervals.append((prior["started"], prior["finished"]))
        previous = prior["spec"]
        for field in ("plan", "server", "gpu_uuids"):
            if not _equal(previous[field], spec[field]):
                raise ValueError(f"calibration preflight {field} differs from the frozen current cell")
        if previous["model"]["revision"] != spec["model"]["revision"] or not _equal(previous["model"]["files"], spec["model"]["files"]):
            raise ValueError("calibration preflight checkpoint manifest differs from the current cell")
        for label in ("baseline", "candidate"):
            if any(prior["identities"][label].get(field) != spec["baseline"][field] for field in ("revision", "source_sha256")):
                raise ValueError("calibration preflight does not match the current baseline source pins")
        hashes.append(digest)
    return {"status": "passed", "performed_before_gpu_lease": True,
            "reference_count": len(references), "verified_receipt_sha256": hashes,
            "verified_prior_execution_ids": sorted(jobs - {execution_id}),
            "reason": "prior raw evidence and known frozen cell verified before current GPU allocation; observed runtime/driver equivalence is checked after measurement"}
