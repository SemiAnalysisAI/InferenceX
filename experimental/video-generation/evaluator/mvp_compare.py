"""Auditable, paired comparison of the small H3 execution bundles.

The MVP measures implementation fidelity, not which generative model is better.
Threshold decisions are deliberately separate from release qualification.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


_POLICY_NUMBERS = {
    "max_latency_increase_fraction": (0.0, None),
    "min_video_psnr_db": (0.0, None),
    "min_audio_spectral_cosine": (-1.0, 1.0),
    "max_audio_rms_ratio_error": (0.0, None),
}


def analyze_media(path: Path, expected: dict | None = None) -> dict:
    """Load the optional media backend only when a comparison needs it."""
    from evaluator.mvp_media import analyze_media as implementation

    return implementation(path, expected=expected)


def compare_media(baseline: Path, candidate: Path) -> dict:
    from evaluator.mvp_media import compare_media as implementation

    return implementation(baseline, candidate)


def _finite(value: Any, *, minimum: float | None = None) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and (minimum is None or value >= minimum)
    )


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON numeric literal is not allowed: {value}")


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _text(mapping: dict, field: str, label: str) -> str:
    value = mapping.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}.{field} must be a nonempty string")
    return value


def _policy(policy: dict) -> dict:
    if not isinstance(policy, dict):
        raise ValueError("comparison policy must be an object")
    _text(policy, "policy_id", "policy")
    if policy.get("calibration_status") not in {
        "uncalibrated", "fixture_control", "operator_calibrated"
    }:
        raise ValueError("policy.calibration_status must explicitly describe calibration")
    for name, (minimum, maximum) in _POLICY_NUMBERS.items():
        value = policy.get(name)
        if not _finite(value, minimum=minimum) or (
            maximum is not None and value > maximum
        ):
            raise ValueError(f"policy.{name} must be an explicit, finite value in range")
    # Reject non-JSON extras as well; retain any declared calibration provenance.
    return json.loads(_canonical(policy))


def _planned_slots(plan: dict) -> dict[str, dict]:
    for field in ("plan_id", "model_id", "model_revision"):
        _text(plan, field, "plan")
    cases = plan.get("cases")
    repetitions = plan.get("repetitions")
    if not isinstance(cases, list) or not cases:
        raise ValueError("plan.cases must be a nonempty list")
    if not isinstance(repetitions, int) or isinstance(repetitions, bool) or repetitions < 1:
        raise ValueError("plan.repetitions must be a positive integer")
    warmups = plan.get("warmup_runs")
    if not isinstance(warmups, int) or isinstance(warmups, bool) or warmups < 0:
        raise ValueError("plan.warmup_runs must be an explicit nonnegative integer")
    if len(cases) * repetitions + warmups > 10000:
        raise ValueError("plan exceeds 10000 total slots")
    generation = plan.get("generation")
    if not isinstance(generation, dict):
        raise ValueError("plan.generation must declare expected media properties")
    for field, maximum in (("width", 8192), ("height", 8192), ("frame_count", 10000)):
        value = generation.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= maximum:
            raise ValueError(f"plan.generation.{field} must be a positive bounded integer")
    for field, maximum in (("fps", 240), ("duration_seconds", 3600)):
        value = generation.get(field)
        if not _finite(value, minimum=0.000001) or value > maximum:
            raise ValueError(f"plan.generation.{field} must be a positive bounded number")
    if "audio_required" in generation and not isinstance(generation["audio_required"], bool):
        raise ValueError("plan.generation.audio_required must be a boolean")
    if generation.get("audio_required") is not False:
        for field, maximum in (("audio_sample_rate_hz", 192000), ("audio_channels", 8)):
            value = generation.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= maximum:
                raise ValueError(f"plan.generation.{field} must declare a positive bounded integer")
    ids = []
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("plan cases must be objects")
        ids.append(_text(case, "case_id", "case"))
        _text(case, "prompt", "case")
        if not isinstance(case.get("seed"), int) or isinstance(case.get("seed"), bool):
            raise ValueError("each case must declare an integer seed")
        for field in ("requires_motion", "requires_sound"):
            if not isinstance(case.get(field), bool):
                raise ValueError(f"case.{field} must be an explicit boolean")
    if len(set(ids)) != len(ids):
        raise ValueError("plan contains duplicate case_id values")
    return {
        f"measurement-r{repetition:03d}-c{index:03d}": {
            **case,
            "repetition": repetition,
        }
        for repetition in range(1, repetitions + 1)
        for index, case in enumerate(cases, start=1)
    }


def _artifact(run_dir: Path, record: dict) -> Path | None:
    raw_path, digest = record.get("artifact_path"), record.get("sha256")
    if raw_path is None:
        if digest is not None:
            raise ValueError(f"{record['slot_id']}: hash without an artifact")
        if record.get("status") == "succeeded":
            raise ValueError(f"{record['slot_id']}: successful record has no artifact")
        return None
    if not isinstance(raw_path, str) or not raw_path or "\\" in raw_path:
        raise ValueError(f"{record['slot_id']}: artifact_path must be a relative path")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{record['slot_id']}: artifact path escapes the run directory")
    try:
        path = (run_dir / relative).resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{record['slot_id']}: missing artifact: {raw_path}") from error
    if not path.is_relative_to(run_dir) or not path.is_file():
        raise ValueError(f"{record['slot_id']}: artifact is not a file inside its run directory")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or _sha256(path) != digest
    ):
        raise ValueError(f"{record['slot_id']}: artifact SHA256 mismatch")
    return path


def _validate_timing_boundaries(record: dict, evidence_kind: str) -> None:
    """Validate the optional, nested client timings without inventing old data."""
    fields = ("submit_to_terminal_seconds", "submit_to_media_seconds", "media_validation_seconds")
    if not any(field in record for field in fields):
        return  # Earlier bundles did not instrument these boundaries.
    slot_id = record["slot_id"]
    values = [record.get(field) for field in fields]
    for field, value in zip(fields, values):
        if value is not None and not _finite(value, minimum=0):
            raise ValueError(f"{slot_id}: {field} must be a finite nonnegative timing or null")
    if not any(value is not None for value in values):
        if record["status"] == "succeeded" and evidence_kind != "imported_media":
            raise ValueError(f"{slot_id}: successful instrumented record is missing client timing boundaries")
        return
    if evidence_kind == "imported_media" or record.get("attempted") is False:
        raise ValueError(f"{slot_id}: imported or not-started media cannot have client timing boundaries")
    latency = record.get("latency_seconds")
    if not _finite(latency, minimum=0):
        raise ValueError(f"{slot_id}: client timing boundaries require a finite nonnegative total latency")
    terminal, media, validation = values
    if record["status"] == "succeeded" and any(value is None for value in values):
        raise ValueError(f"{slot_id}: successful instrumented record is missing client timing boundaries")
    if (media is not None and terminal is None) or (validation is not None and media is None):
        raise ValueError(f"{slot_id}: client timing boundaries are missing an earlier stage")
    tolerance = max(1e-6, latency * 1e-6)
    if any(value is not None and value > latency + tolerance for value in values):
        raise ValueError(f"{slot_id}: client timing boundary exceeds total latency")
    if terminal is not None and media is not None and terminal > media + tolerance:
        raise ValueError(f"{slot_id}: terminal status timing follows completed media download")
    if media is not None and validation is not None and media + validation > latency + tolerance:
        raise ValueError(f"{slot_id}: download plus validation time exceeds total latency")


def _load_run(directory: Path) -> tuple[dict, dict[str, dict]]:
    directory = Path(directory).resolve(strict=True)
    path = directory / "run.json"
    if not directory.is_dir() or not path.is_file():
        raise ValueError(f"not an MVP run directory: {directory}")
    with path.open(encoding="utf-8") as stream:
        run = json.load(stream, parse_constant=_reject_nonfinite, object_pairs_hook=_unique_keys)
    if not isinstance(run, dict):
        raise ValueError("run.json must be an object")
    if run.get("bundle_version") != "0.1.0" or run.get("bundle_type") != "mvp_run":
        raise ValueError("unsupported MVP run bundle type/version")
    if run.get("status") not in {"complete", "partial", "failed"} or not run.get("finished_at"):
        raise ValueError("run bundle is not finalized")
    try:
        started = datetime.fromisoformat(_text(run, "started_at", "run"))
        finished = datetime.fromisoformat(_text(run, "finished_at", "run"))
        if started.tzinfo is None or finished.tzinfo is None or finished < started:
            raise ValueError("invalid run timestamps")
    except (TypeError, ValueError) as error:
        raise ValueError("finalized run requires ordered timezone-aware started_at/finished_at timestamps") from error
    for field in ("run_id", "plan_id", "plan_sha256"):
        _text(run, field, "run")
    if run.get("evidence_kind") not in {"operator_endpoint", "live_h3", "fixture", "imported_media"}:
        raise ValueError("run must declare its actual evidence_kind")
    plan = run.get("plan")
    if not isinstance(plan, dict) or hashlib.sha256(_canonical(plan)).hexdigest() != run["plan_sha256"]:
        raise ValueError("run plan SHA256 does not match its canonical plan content")
    if plan.get("plan_id") != run["plan_id"]:
        raise ValueError("run.plan_id does not match its embedded plan")
    configuration = run.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("run.configuration must be an object")
    for field in ("model_id", "model_revision", "runtime", "runtime_revision", "hardware_label"):
        _text(configuration, field, "configuration")
    for field in ("model_id", "model_revision"):
        if configuration[field] != plan.get(field):
            raise ValueError(f"configuration.{field} does not match the frozen plan")
    unsigned_configuration = {key: value for key, value in configuration.items() if key != "configuration_sha256"}
    configuration_digest = hashlib.sha256(_canonical(unsigned_configuration)).hexdigest()
    if not (run.get("configuration_sha256") or configuration.get("configuration_sha256")):
        raise ValueError("run must declare its configuration SHA256")
    for declared in (run.get("configuration_sha256"), configuration.get("configuration_sha256")):
        if declared is not None and declared != configuration_digest:
            raise ValueError("configuration SHA256 does not match its content")
    measurement = run.get("measurement")
    if not isinstance(measurement, dict):
        raise ValueError("run.measurement must be an object")
    serving = configuration.get("serving")
    boundary = "not_measured_imported_media" if run["evidence_kind"] == "imported_media" else ("submit_to_downloaded_media" if serving else "submit_to_validated_media")
    if measurement.get("boundary") != boundary:
        raise ValueError(f"MVP {run['evidence_kind']} comparison requires {boundary} timing boundary")
    if serving:
        from .mvp_serving import validate_window
        if run["evidence_kind"] == "imported_media":
            raise ValueError("imported media cannot establish serving load")
    elif type(measurement.get("concurrency")) is not int or measurement["concurrency"] != 1:
        raise ValueError("MVP comparison requires serial concurrency=1 measurements")
    planned = _planned_slots(plan)
    planned_warmups = {
        f"warmup-{index:03d}": {**plan["cases"][(index - 1) % len(plan["cases"])], "repetition": 0}
        for index in range(1, plan["warmup_runs"] + 1)
    }
    records = run.get("records")
    if not isinstance(records, list):
        raise ValueError("run.records must be a list")
    measurements: dict[str, dict] = {}
    seen = set()
    warmups_succeeded = True
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("run records must be objects")
        slot_id = _text(record, "slot_id", "record")
        if slot_id in seen:
            raise ValueError(f"duplicate slot_id: {slot_id}")
        seen.add(slot_id)
        if record.get("phase") not in {"warmup", "measurement"}:
            raise ValueError(f"{slot_id}: unknown measurement phase")
        if record.get("status") not in {"succeeded", "failed"}:
            raise ValueError(f"{slot_id}: unknown execution status")
        if "attempted" in record and not isinstance(record["attempted"], bool):
            raise ValueError(f"{slot_id}: attempted must be a boolean")
        if record.get("attempted") is False and (
            record["status"] != "failed" or record.get("artifact_path") is not None
            or record.get("sha256") is not None or record.get("media") is not None
            or record.get("latency_seconds") != 0
        ):
            raise ValueError(f"{slot_id}: a not-started slot cannot contain a successful attempt or measured artifact")
        _validate_timing_boundaries(record, run["evidence_kind"])
        # Resolve and hash even warmups and failed partial artifacts; none are trusted.
        verified_path = _artifact(directory, record)
        if record["phase"] == "warmup":
            if slot_id not in planned_warmups:
                raise ValueError(f"unexpected warmup slot: {slot_id}")
            for field in ("case_id", "prompt", "seed", "repetition"):
                if _canonical(record.get(field)) != _canonical(planned_warmups[slot_id].get(field)):
                    raise ValueError(f"{slot_id}: {field} differs from planned warmup")
            if measurements:
                raise ValueError("warmup slots must precede measurement slots")
            warmup_valid = False
            if verified_path is not None and record["status"] == "succeeded":
                try:
                    warmup_media = analyze_media(verified_path, expected=_expected(plan, planned_warmups[slot_id]))
                    warmup_valid = warmup_media.get("valid") is True
                except Exception:
                    warmup_valid = False
            warmups_succeeded = warmups_succeeded and warmup_valid
            continue
        if slot_id not in planned:
            raise ValueError(f"unexpected measurement slot: {slot_id}")
        case = planned[slot_id]
        for field in ("case_id", "prompt", "seed", "repetition"):
            if _canonical(record.get(field)) != _canonical(case.get(field)):
                raise ValueError(f"{slot_id}: {field} differs from the frozen plan")
        measurements[slot_id] = {**record, "_verified_path": verified_path}
    missing = set(planned) - set(measurements)
    if missing:
        raise ValueError(f"missing measurement slots (failures must be retained): {', '.join(sorted(missing))}")
    missing_warmups = set(planned_warmups) - seen
    if missing_warmups:
        raise ValueError(f"missing warmup slots: {', '.join(sorted(missing_warmups))}")
    if [record["slot_id"] for record in records] != list(planned_warmups) + list(planned):
        raise ValueError("run records do not follow the frozen execution order")
    if serving:
        validate_window(run)
    if run["evidence_kind"] != "imported_media" and not serving:
        attempted_seconds = sum(
            record["latency_seconds"] for record in measurements.values()
            if record.get("attempted") is not False and _finite(record.get("latency_seconds"), minimum=0)
        )
        wall = measurement.get("wall_seconds")
        tolerance = max(1e-6, attempted_seconds * 1e-6)
        if _finite(wall, minimum=0.000001) and wall + tolerance < attempted_seconds:
            raise ValueError("measured wall time is shorter than summed serial attempted latencies")
    run["_directory"] = directory
    run["_bundle_sha256"] = _sha256(path)
    run["_warmups_succeeded"] = warmups_succeeded
    return run, measurements


def _expected(plan: dict, case: dict) -> dict:
    generation = dict(plan.get("generation", {}))
    frames, fps = generation.get("frame_count"), generation.get("fps")
    if _finite(frames, minimum=1) and _finite(fps, minimum=0.000001):
        generation["duration_seconds"] = frames / fps
    generation["audio_required"] = bool(
        generation.get("audio_required", generation.get("audio_sample_rate_hz"))
    )
    for field in ("requires_motion", "requires_sound"):
        generation[field] = bool(case.get(field, False))
    return generation


def _observation(record: dict, expected: dict, *, timing_measured: bool = True, latency_field: str = "latency_seconds") -> dict:
    path = record["_verified_path"]
    analysis, analysis_error = None, None
    if path is not None:
        try:
            analysis = analyze_media(path, expected=expected)
        except Exception as error:
            # An evaluator failure is not evidence that the model failed a metric.
            analysis_error = f"{type(error).__name__}: {error}"
    if analysis and analysis.get("sha256") not in {None, record.get("sha256")}:
        raise ValueError(f"{record['slot_id']}: artifact changed between hash verification and media analysis")
    latency = record.get(latency_field) if timing_measured and record.get("attempted") is not False else None
    return {
        "status": record["status"],
        "attempted": record.get("attempted", True),
        "artifact_path": str(path) if path else None,
        "sha256": record.get("sha256"),
        "latency_seconds": latency if _finite(latency, minimum=0) else None,
        "latency_boundary": "submit_to_downloaded_media" if latency_field == "submit_to_media_seconds" else "submit_to_validated_media",
        "media": analysis,
        "error": record.get("error"),
        "analysis_error": analysis_error,
    }


def _check(name: str, status: str, reason: str, **values: Any) -> dict:
    return {"name": name, "status": status, "reason": reason, **values}


def _outcome(checks: list[dict]) -> str:
    if any(check["status"] == "fail" for check in checks):
        return "fail"
    if any(check["status"] == "inconclusive" for check in checks):
        return "inconclusive"
    return "pass"


def _valid(observation: dict) -> bool:
    return observation["status"] == "succeeded" and (observation.get("media") or {}).get("valid") is True


def _media_check(label: str, observation: dict) -> dict:
    if observation["status"] != "succeeded":
        return _check(
            f"{label}.technical_success", "fail" if label == "candidate" else "inconclusive",
            observation.get("error") or "generation did not succeed; retained in denominator",
        )
    if observation["analysis_error"]:
        return _check(f"{label}.media_validity", "inconclusive", observation["analysis_error"])
    media = observation.get("media") or {}
    if media.get("valid") is not True:
        reasons = [
            str(item.get("detail") or item.get("name"))
            for item in media.get("checks", [])
            if item.get("status") in {"failed", "fail"}
        ]
        return _check(
            f"{label}.media_validity", "fail" if label == "candidate" else "inconclusive",
            "; ".join(reasons) or "media did not pass fresh validation",
        )
    return _check(f"{label}.media_validity", "pass", "fresh decode and requested media checks passed")


def _threshold(name: str, value: Any, threshold: float, *, minimum: bool, unit: str) -> dict:
    if not _finite(value):
        return _check(name, "inconclusive", "metric missing, undefined, or non-finite; not imputed", observed=None, threshold=threshold, unit=unit)
    passed = value >= threshold if minimum else value <= threshold
    return _check(name, "pass" if passed else "fail", "within declared threshold" if passed else "outside declared threshold", observed=value, threshold=threshold, unit=unit)


def _json_safe(value: Any) -> Any:
    """Preserve missing metric states without emitting invalid JSON NaN/Infinity."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _fidelity(baseline: dict, candidate: dict, policy: dict) -> tuple[dict, list[dict], list[str]]:
    try:
        result = compare_media(Path(baseline["artifact_path"]), Path(candidate["artifact_path"]))
    except Exception as error:
        return {}, [_check("fidelity.evaluator", "inconclusive", f"{type(error).__name__}: {error}")], []
    metrics = result.get("metrics", {})
    notes = list(result.get("notes", []))
    for diagnostic in result.get("checks", []):
        if diagnostic.get("status") in {"failed", "fail"}:
            notes.append(
                f"{diagnostic.get('name', 'media comparison')}: {diagnostic.get('detail', '')} "
                f"Observed {_json_safe(diagnostic.get('observed'))}; expected {_json_safe(diagnostic.get('expected'))}."
            )
    checks = [_check(
        "fidelity.compatible_media", "pass" if result.get("compatible") is True else "fail",
        "same frame/audio geometry and time alignment" if result.get("compatible") is True
        else "streams are not compatible for a no-resize/no-truncation fidelity comparison",
    )]
    if result.get("compatible") is not True:
        return _json_safe(metrics), checks, notes
    if metrics.get("video_identical") is True:
        checks.append(_check(
            "fidelity.video_psnr", "pass", "decoded video frames are identical; zero MSE makes finite PSNR undefined",
            observed=None, exact_match=True, threshold=policy["min_video_psnr_db"], unit="dB",
        ))
    else:
        checks.append(_threshold("fidelity.video_psnr", metrics.get("video_psnr_db"), policy["min_video_psnr_db"], minimum=True, unit="dB"))
    audio_present = any((row.get("media") or {}).get("audio", {}).get("present") is True for row in (baseline, candidate))
    if not audio_present:
        checks.extend([
            _check("fidelity.audio_spectral_cosine", "not_applicable", "neither stream contains audio"),
            _check("fidelity.audio_rms_ratio_error", "not_applicable", "neither stream contains audio"),
        ])
    else:
        # Worst-channel checks prevent a healthy channel hiding a collapsed channel.
        cosine_channels = metrics.get("audio_spectral_cosine_channels")
        if cosine_channels:
            cosine = min(cosine_channels) if all(_finite(value) for value in cosine_channels) else None
        else:
            cosine = metrics.get("audio_spectral_cosine")
        checks.append(_threshold("fidelity.audio_spectral_cosine", cosine, policy["min_audio_spectral_cosine"], minimum=True, unit="cosine"))
        ratios = metrics.get("audio_rms_ratio_channels")
        if ratios:
            rms_error = max(abs(value - 1.0) for value in ratios) if all(_finite(value, minimum=0) for value in ratios) else None
        else:
            ratio = metrics.get("audio_rms_ratio")
            rms_error = abs(ratio - 1.0) if _finite(ratio, minimum=0) else None
        metrics["audio_worst_channel_rms_ratio_error"] = rms_error
        checks.append(_threshold("fidelity.audio_rms_ratio_error", rms_error, policy["max_audio_rms_ratio_error"], minimum=False, unit="absolute_ratio_error"))
    return _json_safe(metrics), checks, notes


def _summary(run: dict, observations: list[dict]) -> dict:
    valid = [row for row in observations if _valid(row)]
    valid_latencies = [row["latency_seconds"] for row in valid if _finite(row["latency_seconds"], minimum=0.000001)]
    attempted_latencies = [row["latency_seconds"] for row in observations if row["attempted"] is not False and _finite(row["latency_seconds"], minimum=0)]
    wall = run["measurement"].get("wall_seconds") if run["evidence_kind"] != "imported_media" else None
    if not _finite(wall, minimum=0.000001):
        wall = None
    return {
        "scheduled": len(observations),
        "completed": sum(row["status"] == "succeeded" for row in observations),
        "valid": len(valid),
        "failed": len(observations) - len(valid),
        "failed_attempts": sum(row["status"] == "failed" and row["attempted"] is not False for row in observations),
        "not_started": sum(row["attempted"] is False for row in observations),
        "invalid_completed": sum(row["status"] == "succeeded" and not _valid(row) for row in observations),
        "invalid_completed_interpretation": "completed but not verified valid; includes evaluator-unavailable observations",
        "evaluator_unavailable": sum(bool(row.get("analysis_error")) for row in observations),
        "known_invalid_completed": sum(row["status"] == "succeeded" and not row.get("analysis_error") and (row.get("media") or {}).get("valid") is False for row in observations),
        "technical_success_rate": len(valid) / len(observations),
        "verified_technical_success_fraction": len(valid) / len(observations),
        "technical_success_rate_interpretation": "verified-valid fraction of scheduled slots, not an estimate of model success probability when the evaluator is unavailable",
        "latency_median_seconds": statistics.median(valid_latencies) if valid_latencies else None,
        "latency_measured_count": len(valid_latencies),
        "latency_population": "valid measurement slots only; failures are retained separately",
        "attempt_latency_median_seconds": statistics.median(attempted_latencies) if attempted_latencies else None,
        "wall_seconds": wall,
        "valid_clips_per_second": len(valid) / wall if wall is not None else None,
        "throughput_population": "all scheduled measurement slots; measured wall time includes failed attempts",
        "summary_recomputed_from_verified_artifacts": True,
    }


def _configuration(configuration: dict) -> dict:
    result = dict(configuration)
    # The report does not need credentials, endpoint query parameters, or fragments.
    endpoint = result.get("endpoint")
    if isinstance(endpoint, str):
        try:
            parts = urlsplit(endpoint)
            hostname = parts.hostname or ""
            if ":" in hostname:
                hostname = f"[{hostname}]"
            authority = f"{hostname}:{parts.port}" if parts.port else hostname
            result["endpoint"] = urlunsplit((parts.scheme, authority, parts.path, "", ""))
        except ValueError:
            result["endpoint"] = "[invalid endpoint redacted]"
    return result


def _active_media_code() -> dict:
    from evaluator import mvp_media

    return {
        "implementation_version": mvp_media.IMPLEMENTATION_VERSION,
        "source_sha256": _sha256(Path(mvp_media.__file__)),
    }


def _performance_differences(baseline: dict, candidate: dict, slots: list[dict]) -> list[str]:
    """Compare measured client work as well as operator-declared server class."""
    differences = []
    left, right = baseline["configuration"], candidate["configuration"]
    if left.get("serving") or right.get("serving"):
        differences.append("serving delivery measurements are descriptive; the existing serial regression policy is not calibrated for serving load")
        if _canonical(left.get("serving")) != _canonical(right.get("serving")):
            differences.append("serving load or delivery deadline differs")
    live = bool({"operator_endpoint", "live_h3"} & {baseline["evidence_kind"], candidate["evidence_kind"]})
    for field in ("hardware_label", "runtime"):
        if left[field] != right[field]:
            differences.append(f"{field} differs")
    for field in ("limits", "client_environment", "client_source_sha256", "measurement_semantics", "media_evaluator"):
        a, b = left.get(field), right.get(field)
        absent_left, absent_right = a in (None, "", {}), b in (None, "", {})
        if absent_left and absent_right and not live:
            continue
        if absent_left or absent_right:
            differences.append(f"client {field} is missing")
        elif _canonical(a) != _canonical(b):
            differences.append(f"client {field} differs")
    # Re-analysis versions must agree with the analyzer that was timed. Otherwise
    # a changed client evaluator could be mistaken for a server speed regression.
    declared_pins = [(label, run["configuration"].get("media_evaluator")) for label, run in (("baseline", baseline), ("candidate", candidate))]
    if any(isinstance(pin, dict) and pin for _, pin in declared_pins):
        code = _active_media_code()
        fields = ("implementation_version", "source_sha256", "pyav_version", "numpy_version", "ffmpeg_libraries")
        for label, pin in declared_pins:
            if not isinstance(pin, dict) or not all(pin.get(field) for field in fields):
                differences.append(f"{label} media evaluator pin is incomplete")
                continue
            if any(pin[field] != code[field] for field in code):
                differences.append(f"{label} media evaluator code differs from fresh analysis")
            for slot in slots:
                implementation = (slot[label].get("media") or {}).get("implementation", {})
                actual = {
                    "implementation_version": implementation.get("version"),
                    "pyav_version": implementation.get("pyav_version"),
                    "numpy_version": implementation.get("numpy_version"),
                    "ffmpeg_libraries": implementation.get("ffmpeg_libraries"),
                }
                if any(_canonical(actual[field]) != _canonical(pin[field]) for field in actual):
                    differences.append(f"{label} media evaluator versions are unverified or differ from fresh analysis")
                    break
    return list(dict.fromkeys(differences))


def compare_runs(baseline_dir: Path, candidate_dir: Path, *, policy: dict) -> dict:
    """Verify bundles and compare matched measurement slots under explicit policy.

    Malformed, incomplete, or tampered bundles raise ValueError. Valid bundles can
    produce a fail or inconclusive decision. An MVP threshold pass is never a
    release qualification or a claim about generative model quality.
    """
    policy = _policy(policy)
    baseline_run, baseline_records = _load_run(baseline_dir)
    candidate_run, candidate_records = _load_run(candidate_dir)
    if {"operator_endpoint", "live_h3"} & {baseline_run["evidence_kind"], candidate_run["evidence_kind"]} and (
        baseline_run["_directory"] == candidate_run["_directory"] or baseline_run["run_id"] == candidate_run["run_id"]
    ):
        raise ValueError("live comparison requires distinct baseline and candidate executions")
    if baseline_run["plan_sha256"] != candidate_run["plan_sha256"]:
        raise ValueError("baseline and candidate must use the identical frozen plan SHA256")
    for field in ("model_id", "model_revision"):
        if baseline_run["configuration"][field] != candidate_run["configuration"][field]:
            raise ValueError(f"implementation fidelity requires identical {field}")
    if set(baseline_records) != set(candidate_records):
        raise ValueError("baseline and candidate measurement slots must match")
    evidence_kind = baseline_run["evidence_kind"] if baseline_run["evidence_kind"] == candidate_run["evidence_kind"] else "mixed"
    imported_only = evidence_kind == "imported_media"
    plan = baseline_run["plan"]
    planned = _planned_slots(plan)
    slots = []
    for slot_id, case in planned.items():
        baseline = _observation(baseline_records[slot_id], _expected(plan, case), timing_measured=baseline_run["evidence_kind"] != "imported_media", latency_field="submit_to_media_seconds" if baseline_run["configuration"].get("serving") else "latency_seconds")
        candidate = _observation(candidate_records[slot_id], _expected(plan, case), timing_measured=candidate_run["evidence_kind"] != "imported_media", latency_field="submit_to_media_seconds" if candidate_run["configuration"].get("serving") else "latency_seconds")
        checks = [_media_check("baseline", baseline), _media_check("candidate", candidate)]
        metrics, notes = {}, []
        if _valid(baseline) and _valid(candidate):
            metrics, fidelity_checks, notes = _fidelity(baseline, candidate, policy)
            checks.extend(fidelity_checks)
        else:
            checks.append(_check("fidelity.available_pair", "inconclusive", "requires valid baseline and candidate media; failed slots are retained"))
        for label, observation in (("baseline", baseline), ("candidate", candidate)):
            if not imported_only and _valid(observation) and not _finite(observation["latency_seconds"], minimum=0.000001):
                checks.append(_check(f"{label}.latency", "inconclusive", "valid generation is missing a finite, positive measured latency"))
        left, right = baseline["latency_seconds"], candidate["latency_seconds"]
        metrics["latency_increase_fraction"] = right / left - 1 if _valid(baseline) and _valid(candidate) and _finite(left, minimum=0.000001) and _finite(right, minimum=0.000001) else None
        slots.append({
            "slot_id": slot_id, "case_id": case["case_id"], "prompt": case["prompt"],
            "seed": case["seed"], "repetition": case["repetition"],
            "status": _outcome(checks), "baseline": baseline, "candidate": candidate,
            "metrics": metrics, "checks": checks, "notes": notes,
        })
    summaries = {
        "baseline": _summary(baseline_run, [slot["baseline"] for slot in slots]),
        "candidate": _summary(candidate_run, [slot["candidate"] for slot in slots]),
    }
    checks = []
    if evidence_kind == "mixed":
        checks.append(_check("evidence.same_kind", "inconclusive", "baseline and candidate have different evidence kinds"))
    different = _performance_differences(baseline_run, candidate_run, slots)
    performance_mode = (
        "not_measured_imported_media" if imported_only
        else "descriptive_only" if different or evidence_kind == "mixed"
        else "same_configuration_class_regression"
    )
    baseline_median = summaries["baseline"]["latency_median_seconds"]
    candidate_median = summaries["candidate"]["latency_median_seconds"]
    increase = candidate_median / baseline_median - 1 if _finite(baseline_median, minimum=0.000001) and _finite(candidate_median, minimum=0.000001) else None
    complete_latency = all(summary["latency_measured_count"] == len(slots) for summary in summaries.values())
    if imported_only:
        increase = None
        checks.append(_check("performance.median_latency", "not_applicable", "imported media has no measured generation latency; this is a media-fidelity-only comparison"))
    elif performance_mode == "descriptive_only":
        checks.append(_check("performance.median_latency", "descriptive", "latency is descriptive, not a regression gate: " + ("; ".join(different) if different else "imported/mixed evidence"), observed=increase, threshold=policy["max_latency_increase_fraction"], unit="fraction"))
    elif not complete_latency:
        checks.append(_check("performance.median_latency", "inconclusive", "cannot gate a survivor-only or missing-latency population", observed=increase, threshold=policy["max_latency_increase_fraction"], unit="fraction"))
    else:
        checks.append(_threshold("performance.median_latency", increase, policy["max_latency_increase_fraction"], minimum=False, unit="fraction"))
    for label in ("baseline", "candidate"):
        if imported_only:
            checks.append(_check(f"{label}.wall_time", "not_applicable", "no inference wall time or throughput was measured for imported media"))
        elif summaries[label]["wall_seconds"] is None:
            checks.append(_check(f"{label}.wall_time", "inconclusive", "missing finite measured wall time; throughput is not inferred from summed request latency"))
    for label, run in (("baseline", baseline_run), ("candidate", candidate_run)):
        if not run["_warmups_succeeded"]:
            checks.append(_check(f"{label}.warmup", "inconclusive", "a planned warmup failed or produced invalid media; warmed performance is not established"))
    all_checks = checks + [check for slot in slots for check in slot["checks"]]
    result = {
        "bundle_version": "0.1.0", "bundle_type": "mvp_comparison",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "plan_id": baseline_run["plan_id"], "plan_sha256": baseline_run["plan_sha256"], "plan": plan,
        "evidence_kind": evidence_kind,
        "comparison_scope": "media_fidelity_only" if imported_only else "runtime_regression",
        "overall_status": _outcome(all_checks),
        "release_qualified": False,
        "release_qualification_reason": "MVP threshold decisions are not a calibrated, independently verified release qualification.",
        "policy": policy,
        "measurement": {
            "boundary": baseline_run["measurement"]["boundary"] if baseline_run["measurement"]["boundary"] == candidate_run["measurement"]["boundary"] else "mixed_incomparable_boundaries",
            "concurrency": baseline_run["measurement"]["concurrency"] if baseline_run["measurement"]["concurrency"] == candidate_run["measurement"]["concurrency"] else None,
            "performance_mode": performance_mode,
            "performance_comparability_limitations": different,
            "latency_increase_fraction": increase,
            "hardware_and_model_identity": "operator-declared; not independently attested",
            "statistical_claim": "point estimates only; no confidence or significance claim",
            "timing_evidence": {
                "baseline": baseline_run["measurement"].get("timing_evidence", "not separately declared"),
                "candidate": candidate_run["measurement"].get("timing_evidence", "not separately declared"),
            },
        },
        "checks": checks,
        "slots": slots,
        "summary": {
            "measurement_slots": len(slots),
            "passed_slots": sum(slot["status"] == "pass" for slot in slots),
            "failed_slots": sum(slot["status"] == "fail" for slot in slots),
            "inconclusive_slots": sum(slot["status"] == "inconclusive" for slot in slots),
            "matched_valid_pairs": sum(_valid(slot["baseline"]) and _valid(slot["candidate"]) for slot in slots),
            "warmups_excluded": True,
            "failed_slots_retained": True,
        },
        "limitations": [
            "Frame PSNR and audio similarity measure same-request implementation fidelity, not generative video/audio quality.",
            "Fixed seeds do not guarantee matching output across different inference implementations; a fidelity failure requires investigation.",
            "Hardware, runtime revision, and model identity are supplied by the operator, not independently verified by this harness.",
            "Timing covers client submission through downloaded, validated media; it is not GPU-only kernel latency.",
            "Recorded valid-clips throughput is descriptive; serial or closed-loop delivery measurements do not establish sustainable serving capacity.",
            "Thresholds require domain calibration; no human evaluation, statistical significance, or release certification is claimed.",
            "No memory, energy, cost, prompt-following, physics, or perceptual-quality results are inferred from these fidelity checks.",
        ],
    }
    for label, run in (("baseline", baseline_run), ("candidate", candidate_run)):
        result[label] = {
            "run_id": run["run_id"], "evidence_kind": run["evidence_kind"],
            "configuration": _configuration(run["configuration"]),
            "run_bundle_sha256": run["_bundle_sha256"], "summary": summaries[label],
            "provenance": run.get("provenance", {}),
        }
    return _json_safe(result)
