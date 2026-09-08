"""Portable, read-only dashboard for controlled GPU job artifacts.

This viewer never starts a job, invents a run, or imports demonstration media.
It rechecks file hashes and recomputes denominators from the frozen schedule,
but it does not rerun the decoder or independently attest to the supervisor.
Only a bound ``controlled_h3_gpu`` receipt permits GPU timing presentation.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
import stat
import statistics
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterator


MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_MEDIA_BYTES = 512 * 1024 * 1024
MAX_TOTAL_MEDIA_BYTES = 2 * 1024 * 1024 * 1024
MAX_TELEMETRY_BYTES = 16 * 1024 * 1024
MAX_SLOTS = 10000
MAX_DISPLAY_SLOTS = 256
_ROLES = ("baseline", "candidate")
_MEDIA_SUFFIXES = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac"}
_AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac"}


class _InvalidEvidence(ValueError):
    pass


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _text(value: Any, limit: int = 2000) -> str:
    if value is None:
        return "Not recorded"
    if isinstance(value, (dict, list)):
        return "Invalid field type"
    return str(value)[:limit]


def _escape(value: Any) -> str:
    return html.escape(_text(value, 32000), quote=True)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _issue(issues: list[dict], context: str, message: str) -> None:
    if len(issues) < 256:
        issues.append({"context": _text(context, 300), "message": _text(message)})


def _relative(value: Any) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise _InvalidEvidence("Expected a nonempty relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise _InvalidEvidence("Absolute paths and path traversal are not permitted")
    return path


def _no_symlink_parents(path: Path) -> None:
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ValueError("Job and report paths must not contain symlinks")


class _Tree:
    """Open each input component relative to an fd, never following symlinks."""

    def __init__(self, root: Path):
        self.root = root

    @contextmanager
    def open(self, relative: str, maximum: int) -> Iterator[Any]:
        parts = _relative(relative).parts
        descriptors = []
        try:
            current = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            descriptors.append(current)
            for part in parts[:-1]:
                current = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current)
                descriptors.append(current)
            descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=current)
            with os.fdopen(descriptor, "rb") as stream:
                metadata = os.fstat(stream.fileno())
                if not stat.S_ISREG(metadata.st_mode):
                    raise _InvalidEvidence("Input must be a regular file")
                if metadata.st_size > maximum:
                    raise _InvalidEvidence("Input exceeds the viewer byte limit")
                yield stream
        finally:
            for descriptor in reversed(descriptors):
                os.close(descriptor)


def _pairs(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise _InvalidEvidence("Duplicate JSON keys are not permitted")
        result[key] = value
    return result


def _reject_constant(_: str) -> None:
    raise _InvalidEvidence("Non-finite JSON values are not permitted")


def _document(tree: _Tree, relative: str, issues: list[dict], *, expected: Any = None,
              canonical_hash: bool = False) -> tuple[dict | None, str | None]:
    try:
        with tree.open(relative, MAX_JSON_BYTES) as stream:
            data = stream.read(MAX_JSON_BYTES + 1)
        if len(data) > MAX_JSON_BYTES:
            raise _InvalidEvidence("JSON exceeds the viewer byte limit")
        value = json.loads(data, object_pairs_hook=_pairs, parse_constant=_reject_constant)
        if not isinstance(value, dict):
            raise _InvalidEvidence("JSON document must be an object")
        canonical = _canonical(value)  # also rejects overflow-to-infinity numbers
        digest = hashlib.sha256(canonical if canonical_hash else data).hexdigest()
        if expected is not None and (not _is_digest(expected) or expected != digest):
            raise _InvalidEvidence("Document SHA256 does not match its receipt")
        return value, digest
    except FileNotFoundError:
        return None, None
    except (OSError, ValueError, TypeError, RecursionError, OverflowError) as error:
        message = str(error) if isinstance(error, _InvalidEvidence) else "Unreadable, unsafe, or malformed JSON document"
        _issue(issues, relative, message)
        return None, None


def _planned(plan: Any, issues: list[dict], context: str) -> list[dict]:
    if not isinstance(plan, dict):
        return []
    cases, repetitions, warmups = plan.get("cases"), plan.get("repetitions"), plan.get("warmup_runs")
    if (not isinstance(cases, list) or not cases or type(repetitions) is not int or repetitions < 1
            or type(warmups) is not int or warmups < 0 or len(cases) * repetitions + warmups > MAX_SLOTS):
        _issue(issues, context, "Frozen schedule is missing or exceeds the supported slot limit")
        return []
    if any(not isinstance(case, dict) or not isinstance(case.get("case_id"), str)
           or not isinstance(case.get("prompt"), str) or type(case.get("seed")) is not int for case in cases):
        _issue(issues, context, "Frozen cases require case_id, prompt, and integer seed")
        return []
    if len({case["case_id"] for case in cases}) != len(cases):
        _issue(issues, context, "Frozen case identifiers are not unique")
        return []
    result = []
    for index in range(1, warmups + 1):
        result.append({**cases[(index - 1) % len(cases)], "phase": "warmup", "repetition": 0, "slot_id": f"warmup-{index:03d}"})
    for repetition in range(1, repetitions + 1):
        for index, case in enumerate(cases, 1):
            result.append({**case, "phase": "measurement", "repetition": repetition, "slot_id": f"measurement-r{repetition:03d}-c{index:03d}"})
    return result


def _copy_media(tree: _Tree, relative: str, digest: Any, assets: Path, budget: list[int]) -> str:
    if not _is_digest(digest):
        raise _InvalidEvidence("Media requires a lowercase SHA256 digest")
    suffix = _relative(relative).suffix.lower()
    if suffix not in _MEDIA_SUFFIXES:
        raise _InvalidEvidence("Unsupported media extension; not embedded")
    target = assets / (digest + suffix)
    created = False
    destination = None
    try:
        with tree.open(relative, MAX_MEDIA_BYTES) as source:
            size = os.fstat(source.fileno()).st_size
            if budget[0] + size > MAX_TOTAL_MEDIA_BYTES:
                raise _InvalidEvidence("Total media exceeds the viewer byte limit")
            budget[0] += size
            if target.exists() or target.is_symlink():
                descriptor = os.open(target, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                with os.fdopen(descriptor, "rb") as existing:
                    existing_metadata = os.fstat(existing.fileno())
                    if not stat.S_ISREG(existing_metadata.st_mode) or existing_metadata.st_size > MAX_MEDIA_BYTES:
                        raise _InvalidEvidence("Existing report media is not a safe regular file")
                    existing_hash = hashlib.sha256()
                    existing_bytes = 0
                    while chunk := existing.read(1024 * 1024):
                        existing_bytes += len(chunk)
                        if existing_bytes > MAX_MEDIA_BYTES:
                            raise _InvalidEvidence("Existing report media exceeds the byte limit")
                        existing_hash.update(chunk)
                    if existing_hash.hexdigest() != digest:
                        raise _InvalidEvidence("Existing content-addressed report asset has different bytes")
            else:
                destination = target.open("xb")
                created = True
            hasher, count = hashlib.sha256(), 0
            while chunk := source.read(1024 * 1024):
                count += len(chunk)
                if count > size or count > MAX_MEDIA_BYTES:
                    raise _InvalidEvidence("Media changed size while being read")
                hasher.update(chunk)
                if destination:
                    destination.write(chunk)
            if count != size or hasher.hexdigest() != digest:
                raise _InvalidEvidence("Media SHA256 mismatch or file changed during export")
    except Exception:
        if destination:
            destination.close()
            destination = None
        if created:
            target.unlink(missing_ok=True)
        raise
    finally:
        if destination:
            destination.close()
    return "assets/" + target.name


def _selected(mapping: Any, fields: tuple[str, ...]) -> dict:
    if not isinstance(mapping, dict):
        return {}
    return {field: mapping[field] for field in fields if field in mapping and isinstance(mapping[field], (str, bool, int, float, type(None)))}


def _check_data(check: dict) -> dict:
    result = _selected(check, ("name", "status", "unit", "reason"))
    for field in ("observed", "threshold"):
        value = check.get(field)
        if isinstance(value, (dict, list)):
            result[field] = json.dumps(value, ensure_ascii=False, allow_nan=False)[:2000]
        else:
            result[field] = value
    return result


def _telemetry(value: Any) -> dict:
    if not isinstance(value, dict):
        value = {}
    peaks = value.get("observed_memory_peak_mib_by_gpu", {})
    identities = value.get("gpu_identity", [])
    return {
        "sample_count": value.get("sample_count") if type(value.get("sample_count")) is int and value["sample_count"] >= 0 else None,
        "measurement_sample_count": value.get("measurement_sample_count") if type(value.get("measurement_sample_count")) is int and value["measurement_sample_count"] >= 0 else None,
        "gpu_identity": [_selected(gpu, ("uuid", "index", "name", "memory_total_mib", "driver_version")) for gpu in identities[:64] if isinstance(gpu, dict)] if isinstance(identities, list) else [],
        "observed_memory_peak_mib_by_gpu": {str(key)[:200]: item for key, item in list(peaks.items())[:64] if _finite(item)} if isinstance(peaks, dict) else {},
        "errors": [_text(error) for error in value.get("errors", [])[:64]] if isinstance(value.get("errors"), list) else [],
        "observed_owned_compute_by_gpu": {str(key)[:200]: count for key, count in list(value.get("observed_owned_compute_by_gpu", {}).items())[:64] if type(count) is int and count >= 0} if isinstance(value.get("observed_owned_compute_by_gpu"), dict) else {},
    }


def _verify_telemetry(tree: _Tree, metadata: dict, issues: list[dict], label: str, selected: Any) -> dict:
    relative, expected = metadata.get("telemetry_path"), metadata.get("telemetry_sha256")
    result = {"file_sha256_verified": False, "samples_consistent": False, "selected_owned_compute_observed": False}
    if relative is None and expected is None:
        return result
    try:
        if not _is_digest(expected):
            raise _InvalidEvidence("Telemetry lacks a valid receipt hash")
        with tree.open(relative, MAX_TELEMETRY_BYTES) as stream:
            data = stream.read(MAX_TELEMETRY_BYTES + 1)
        if len(data) > MAX_TELEMETRY_BYTES:
            raise _InvalidEvidence("Telemetry exceeds the viewer byte limit")
        if hashlib.sha256(data).hexdigest() != expected:
            raise _InvalidEvidence("Telemetry SHA256 does not match its receipt")
        result["file_sha256_verified"] = True
        if not isinstance(selected, list) or not selected or len(selected) > 64 or any(not isinstance(device, str) for device in selected) or len(set(selected)) != len(selected):
            raise _InvalidEvidence("Selected GPU UUIDs are missing or invalid")
        peaks, owned = {device: None for device in selected}, dict.fromkeys(selected, 0)
        sample_count = measurement_count = 0
        identities = None
        previous = None
        no_foreign = True
        for line in data.splitlines():
            if not line or len(line) > 1024 * 1024:
                raise _InvalidEvidence("Empty or oversized telemetry sample")
            sample = json.loads(line, object_pairs_hook=_pairs, parse_constant=_reject_constant)
            _canonical(sample)
            if not isinstance(sample, dict) or sample.get("phase") not in {"startup", "measurement", "cleanup"}:
                raise _InvalidEvidence("Telemetry sample has an invalid phase or shape")
            monotonic = sample.get("monotonic_seconds")
            if not _finite(monotonic) or previous is not None and monotonic < previous:
                raise _InvalidEvidence("Telemetry sampling timestamps are invalid")
            previous = monotonic
            gpus = sample.get("gpus")
            if not isinstance(gpus, list) or any(not isinstance(gpu, dict) for gpu in gpus) or sorted(gpu.get("uuid", "") for gpu in gpus) != sorted(selected):
                raise _InvalidEvidence("Telemetry device inventory differs from the selected UUIDs")
            current = [_selected(gpu, ("uuid", "index", "name", "memory_total_mib", "driver_version")) for gpu in gpus]
            if identities is None:
                identities = current
            elif identities != current:
                raise _InvalidEvidence("Observed GPU identity changed during the job")
            for gpu in gpus:
                memory, total = gpu.get("memory_used_mib"), gpu.get("memory_total_mib")
                if not _finite(memory) or not _finite(total) or memory > total:
                    raise _InvalidEvidence("Telemetry memory sample is invalid")
                peaks[gpu["uuid"]] = max(memory, peaks[gpu["uuid"]] or 0)
            apps, owned_apps, foreign = sample.get("compute_apps"), sample.get("owned_compute_apps"), sample.get("unowned_compute_apps")
            if any(not isinstance(items, list) for items in (apps, owned_apps, foreign)):
                raise _InvalidEvidence("Telemetry lacks explicit compute ownership observations")
            if any(not isinstance(app, dict) or app.get("gpu_uuid") not in selected or type(app.get("pid")) is not int or app["pid"] < 2 for app in apps + owned_apps + foreign):
                raise _InvalidEvidence("Telemetry compute process observations are malformed")
            if any(app not in apps for app in owned_apps + foreign) or len(owned_apps) + len(foreign) != len(apps):
                raise _InvalidEvidence("Telemetry compute ownership partition is inconsistent")
            no_foreign = no_foreign and not foreign
            sample_count += 1
            if sample["phase"] == "measurement":
                measurement_count += 1
                for device in selected:
                    owned[device] += any(app["gpu_uuid"] == device for app in owned_apps)
        summary = _telemetry(metadata.get("telemetry_summary"))
        if (summary["sample_count"] != sample_count or summary["measurement_sample_count"] != measurement_count
                or summary["gpu_identity"] != (identities or []) or summary["observed_memory_peak_mib_by_gpu"] != {key: value for key, value in peaks.items() if value is not None}
                or summary["observed_owned_compute_by_gpu"] != owned):
            raise _InvalidEvidence("Telemetry summary does not match its hash-verified samples")
        result["samples_consistent"] = True
        result["selected_owned_compute_observed"] = bool(measurement_count >= 2 and no_foreign and not summary["errors"] and all(owned.values()))
        return result
    except (OSError, ValueError, TypeError, RecursionError, OverflowError) as error:
        message = str(error) if isinstance(error, _InvalidEvidence) else "Telemetry file is missing or unsafe"
        _issue(issues, label + ".telemetry", message)
        return result


def _provenance_reasons(spec: dict, label: str, metadata: dict, configuration: dict, telemetry: dict) -> list[str]:
    declared, observed = spec.get(label), metadata.get("source_identity")
    reasons = []
    if (not isinstance(declared, dict) or not isinstance(observed, dict)
            or any(not isinstance(declared.get(field), str) or observed.get(field) != declared[field] for field in ("source", "revision", "source_sha256", "python"))
            or not _is_digest(observed.get("source_sha256")) or not _is_digest(observed.get("python_sha256"))
            or not PurePosixPath(declared["source"]).is_absolute() or not PurePosixPath(declared["python"]).is_absolute()
            or len(declared["revision"]) != 40 or any(c not in "0123456789abcdef" for c in declared["revision"])
            or configuration.get("runtime_revision") != declared.get("revision")
            or observed.get("sglang_module") != str(PurePosixPath(declared.get("source", "")) / "python" / "sglang" / "__init__.py")
            or not isinstance(observed.get("packages"), dict) or not observed["packages"].get("torch")):
        reasons.append("Observed source/Python identity does not match the pinned runtime specification")
    process = metadata.get("process_identity")
    if (not isinstance(process, dict) or any(type(process.get(field)) is not int or process[field] < 1 for field in ("pid", "pgid", "session_id", "start_ticks"))
            or process.get("pid", 0) < 2 or process.get("pgid") != process.get("pid") or process.get("session_id") != process.get("pid")
            or not isinstance(process.get("launch_nonce"), str) or len(process["launch_nonce"]) != 32 or any(c not in "0123456789abcdef" for c in process["launch_nonce"])):
        reasons.append("A plausible owned process/session receipt is unavailable")
    if not telemetry.get("file_sha256_verified") or not telemetry.get("samples_consistent") or not telemetry.get("selected_owned_compute_observed"):
        reasons.append("Hash-verified measurement samples do not establish owned compute on every selected GPU")
    return reasons


def _finalized(run: dict) -> bool:
    if run.get("status") not in {"complete", "partial", "failed"}:
        return False
    try:
        started = datetime.fromisoformat(run.get("started_at", ""))
        finished = datetime.fromisoformat(run.get("finished_at", ""))
        return started.tzinfo is not None and finished.tzinfo is not None and finished >= started
    except (TypeError, ValueError):
        return False


def _cleanup(value: Any) -> dict:
    result = _selected(value, ("status", "reason", "idle_after"))
    if isinstance(value, dict) and isinstance(value.get("remaining_owned_pids"), list):
        result["remaining_owned_pids"] = [pid for pid in value["remaining_owned_pids"][:256] if type(pid) is int]
    return result


def _role(tree: _Tree, label: str, metadata: Any, spec: dict, controlled: bool,
          assets: Path, budget: list[int], issues: list[dict]) -> dict:
    metadata = metadata if isinstance(metadata, dict) else {}
    source = metadata.get("source_identity", {})
    identity = _selected(source, ("source", "revision", "source_sha256", "python", "python_sha256", "python_version", "sglang_module"))
    if isinstance(source, dict) and isinstance(source.get("packages"), dict):
        identity["packages"] = {str(key)[:100]: _text(value, 200) for key, value in list(source["packages"].items())[:64]}
    result = {
        "status": _text(metadata.get("status", "not_started")), "source_identity": identity,
        "process_identity": _selected(metadata.get("process_identity"), ("pid", "pgid", "start_ticks", "session_id")),
        "telemetry": _telemetry(metadata.get("telemetry_summary")), "cleanup": _cleanup(metadata.get("cleanup")),
        "configuration": {}, "observations": [], "warmups": [], "summary": {}, "run_sha256": None,
        "plan_sha256": None, "evidence_kind": "missing", "gpu_timing_presented": False,
    }
    result["telemetry"].update(_verify_telemetry(tree, metadata, issues, label, spec.get("gpu_uuids")))
    relative = metadata.get("run_path", f"{label}/run.json")
    run, digest = _document(tree, relative, issues, expected=metadata.get("run_sha256"))
    if run is None and metadata.get("run_sha256") is not None:
        _issue(issues, label, "Receipt references a run bundle that is missing or invalid")
        if result["status"] == "complete":
            result["status"] = "incomplete"
    plan = spec.get("plan") if isinstance(spec.get("plan"), dict) else (run or {}).get("plan")
    schedule = _planned(plan, issues, label + ".plan")
    bound = run is not None and _is_digest(metadata.get("run_sha256")) and digest == metadata["run_sha256"]
    eligible = False
    records = {}
    if run:
        result.update(run_sha256=digest, evidence_kind=_text(run.get("evidence_kind")), run_id=_text(run.get("run_id")),
                      run_status=_text(run.get("status")), started_at=_text(run.get("started_at")), finished_at=run.get("finished_at"))
        if run.get("bundle_type") != "mvp_run" or run.get("bundle_version") != "0.1.0":
            _issue(issues, label, "Unsupported run bundle type or version")
            run = None
        elif not isinstance(run.get("plan"), dict) or hashlib.sha256(_canonical(run["plan"])).hexdigest() != run.get("plan_sha256"):
            _issue(issues, label, "Run plan hash mismatch")
            run = None
        elif plan != run.get("plan"):
            _issue(issues, label, "Run differs from the job's frozen workload")
            run = None
    if run:
        result["plan_sha256"] = run["plan_sha256"]
        configuration = run.get("configuration", {})
        result["configuration"] = _selected(configuration, ("model_id", "model_revision", "runtime", "runtime_revision", "hardware_label", "identity_verification", "client_source_sha256"))
        if not isinstance(configuration, dict):
            configuration = {}
        unsigned = {key: value for key, value in configuration.items() if key != "configuration_sha256"}
        declared = [value for value in (run.get("configuration_sha256"), configuration.get("configuration_sha256")) if value is not None]
        config_hash = hashlib.sha256(_canonical(unsigned)).hexdigest()
        config_valid = bool(declared) and all(_is_digest(value) and value == config_hash for value in declared)
        if not config_valid:
            _issue(issues, label, "Run configuration hash is absent or mismatched")
        if not bound:
            _issue(issues, label, "Run is not hash-bound to a finalized supervisor receipt; GPU timings withheld")
        measurement = run.get("measurement", {})
        if not isinstance(measurement, dict):
            measurement = {}
        eligible = bool(controlled and bound and config_valid and run.get("evidence_kind") in {"operator_endpoint", "live_h3"}
                        and measurement.get("boundary") == "submit_to_validated_media" and type(measurement.get("concurrency")) is int and measurement["concurrency"] == 1)
        if eligible and not _finalized(run):
            _issue(issues, label, "Run is not finalized with ordered timezone-aware timestamps; GPU timing withheld")
            eligible = False
        if eligible:
            reasons = _provenance_reasons(spec, label, metadata, configuration, result["telemetry"])
            for reason in reasons:
                _issue(issues, label, reason + "; GPU timing withheld")
            eligible = not reasons
        result["gpu_timing_presented"] = eligible
        raw_records = run.get("records")
        if not isinstance(raw_records, list) or len(raw_records) > MAX_SLOTS:
            _issue(issues, label, "Run records are malformed or exceed the slot limit")
        else:
            expected_ids = {slot["slot_id"] for slot in schedule}
            for record in raw_records:
                slot_id = record.get("slot_id") if isinstance(record, dict) else None
                if not isinstance(slot_id, str) or slot_id not in expected_ids or slot_id in records:
                    _issue(issues, label, "Unexpected, duplicate, or malformed run slot; not counted as valid")
                    eligible = False
                    result["gpu_timing_presented"] = False
                    continue
                records[slot_id] = record
                try:
                    from .mvp_compare import _validate_timing_boundaries
                    _validate_timing_boundaries(record, run["evidence_kind"])
                except (ValueError, KeyError, TypeError):
                    _issue(issues, label + "." + slot_id, "Invalid nested client timing boundaries; GPU timing withheld")
                    eligible = False
                    result["gpu_timing_presented"] = False
    for slot in schedule:
        observation = {key: slot[key] for key in ("slot_id", "case_id", "prompt", "seed", "phase", "repetition")}
        observation.update(status="not_recorded", attempted=None, valid=False, artifact_path=None, sha256=None,
                           latency_seconds=None, submit_to_terminal_seconds=None, submit_to_media_seconds=None,
                           media_validation_seconds=None, error=None, media={})
        record = records.get(slot["slot_id"])
        if record is not None:
            attempted = record.get("attempted", True if result["evidence_kind"] in {"fixture", "imported_media"} else None)
            if any(record.get(field) != slot[field] for field in ("case_id", "prompt", "seed", "phase", "repetition")):
                observation.update(status="invalid_record", error="Record does not match its frozen slot")
                _issue(issues, label + "." + slot["slot_id"], observation["error"])
            elif record.get("status") not in {"succeeded", "failed"} or type(attempted) is not bool:
                observation.update(status="invalid_record", error="Record status or attempted flag is invalid")
                _issue(issues, label + "." + slot["slot_id"], observation["error"])
            else:
                observation.update(status=record["status"] if attempted else "not_started", attempted=attempted, error=_text(record["error"]) if record.get("error") else None)
                if eligible and attempted:
                    for field in ("latency_seconds", "submit_to_terminal_seconds", "submit_to_media_seconds", "media_validation_seconds"):
                        observation[field] = record.get(field) if _finite(record.get(field)) else None
                media = record.get("media") if isinstance(record.get("media"), dict) else {}
                observation["media"] = {
                    "video": _selected(media.get("video"), ("width", "height", "frame_count", "fps", "duration_seconds")),
                    "audio": _selected(media.get("audio"), ("present", "sample_rate_hz", "channels", "sample_count", "duration_seconds")),
                }
                if record.get("artifact_path") is not None:
                    try:
                        parent = _relative(relative).parent
                        path = parent / _relative(record["artifact_path"])
                        observation["artifact_path"] = _copy_media(tree, path.as_posix(), record.get("sha256"), assets, budget)
                        observation["sha256"] = record["sha256"]
                        observation["valid"] = record["status"] == "succeeded" and attempted and media.get("valid") is True
                        if media.get("sha256") is not None and media["sha256"] != record["sha256"]:
                            observation["valid"] = False
                            raise _InvalidEvidence("Saved decoder hash differs from the media hash")
                    except (OSError, ValueError) as error:
                        message = str(error) if isinstance(error, _InvalidEvidence) else "Missing or unsafe media artifact"
                        observation.update(artifact_path=None, valid=False, error=message)
                        _issue(issues, label + "." + slot["slot_id"], message)
                if observation["status"] == "succeeded" and not observation["valid"]:
                    observation["status"] = "unverified_media"
        result["warmups" if slot["phase"] == "warmup" else "observations"].append(observation)
    rows = result["observations"]
    valid = [row for row in rows if row["valid"]]
    latencies = [row["latency_seconds"] for row in valid if _finite(row["latency_seconds"])]
    measurement = run.get("measurement", {}) if run else {}
    wall = measurement.get("wall_seconds") if isinstance(measurement, dict) else None
    attempted_sum = sum(row["latency_seconds"] for row in rows if _finite(row["latency_seconds"]))
    wall_valid = bool(eligible and _finite(wall) and wall > 0 and wall + 1e-6 >= attempted_sum and run and run.get("finished_at"))
    if eligible and _finite(wall) and wall + 1e-6 < attempted_sum:
        _issue(issues, label, "Measured wall time is shorter than summed serial request times; throughput withheld")
    result["summary"] = {
        "scheduled": len(rows) if schedule else None, "recorded": sum(row["status"] != "not_recorded" for row in rows),
        "valid": len(valid), "failed_attempts": sum(row["status"] == "failed" for row in rows),
        "not_started": sum(row["status"] == "not_started" for row in rows),
        "not_recorded": sum(row["status"] == "not_recorded" for row in rows),
        "unverified_media": sum(row["status"] in {"unverified_media", "invalid_record"} for row in rows),
        "latency_median_seconds": statistics.median(latencies) if latencies else None, "latency_count": len(latencies),
        "wall_seconds": wall if wall_valid else None, "valid_clips_per_second": len(valid) / wall if wall_valid else None,
        "warmups_scheduled": len(result["warmups"]), "warmups_valid": sum(row["valid"] for row in result["warmups"]),
    }
    return result


def _number(value: Any, digits: int = 3, unit: str = "") -> str:
    return f"{value:,.{digits}f}{unit}" if _finite(value) else "Not measured"


def _badge(value: Any) -> str:
    label = _text(value).replace("_", " ")
    style = "good" if value in {"clean", "complete", "succeeded", "pass"} else "bad" if value in {"failed", "fail", "invalid", "invalid_record", "unverified_media", "aborted"} else "pending"
    return f'<span class="badge {style}">{_escape(label)}</span>'


def _kv(values: list[tuple[str, Any]]) -> str:
    return '<dl>' + ''.join(f'<dt>{_escape(name)}</dt><dd>{_escape(value)}</dd>' for name, value in values) + '</dl>'


def _player(label: str, row: dict | None) -> str:
    row = row or {}
    path = row.get("artifact_path")
    if path:
        tag = "audio" if PurePosixPath(path).suffix in _AUDIO_SUFFIXES else "video"
        player = f'<{tag} controls preload="metadata" src="{_escape(path)}"></{tag}>'
    else:
        player = f'<div class="empty-media"><strong>No generated artifact available</strong><p>{_escape(row.get("error") or "This slot has not produced a verified, exportable media file.")}</p></div>'
    return (f'<figure><div class="figure-head"><strong>{_escape(label.title())}</strong>{_badge(row.get("status", "not_recorded"))}</div>{player}'
            f'<figcaption>Submit → validated media: <strong>{_escape(_number(row.get("latency_seconds"), unit=" s"))}</strong>'
            f'<br>Terminal observed: {_escape(_number(row.get("submit_to_terminal_seconds"), unit=" s"))} · Download complete: {_escape(_number(row.get("submit_to_media_seconds"), unit=" s"))}'
            f'<br>Client validation: {_escape(_number(row.get("media_validation_seconds"), unit=" s"))}</figcaption></figure>')


_CSS = """
:root{color-scheme:light;--ink:#18323b;--muted:#58717b;--line:#d7e3e7;--teal:#08796e;--soft:#f2f7f8;--amber:#806017;--red:#a03c48}*{box-sizing:border-box}body{margin:0;background:var(--soft);color:var(--ink);font:14px/1.6 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}a{color:var(--teal)}.top{background:#132e38;color:#fff;padding:14px max(22px,calc((100vw - 1232px)/2));font-size:12px;letter-spacing:.05em}.top span{color:#a1c5ca;margin-left:18px}main{max-width:1280px;margin:auto;padding:32px 24px 64px}header{display:flex;justify-content:space-between;gap:22px;align-items:center}h1{font-size:38px;line-height:1.1;letter-spacing:-.04em;margin:10px 0 14px}h2{font-size:22px;letter-spacing:-.025em;margin:0 0 12px}h3{margin:0 0 10px;font-size:17px}p{margin:0 0 12px}.eyebrow{font-size:11px;font-weight:750;text-transform:uppercase;letter-spacing:.13em;color:var(--teal)}.muted,small{color:var(--muted)}.subtitle{max-width:760px;color:var(--muted)}.download{padding:10px 15px;border:1px solid #b5cbce;border-radius:7px;background:#fff;text-decoration:none;white-space:nowrap;font-size:12px;font-weight:700}.notice{border:1px solid #d9c485;border-left:4px solid #b08b29;background:#fffbec;border-radius:8px;padding:17px 20px;margin:20px 0}.notice p{margin:3px 0 0;font-size:12px}.notice.live{background:#edf8f4;border-color:#a0cbbd;border-left-color:var(--teal)}.notice.error{background:#fff0f1;border-color:#dbafb5;border-left-color:var(--red)}.badge{display:inline-block;font-size:10px;letter-spacing:.04em;text-transform:uppercase;font-weight:750;padding:4px 8px;border-radius:5px;background:#e9eff1;white-space:nowrap}.good{color:#136b4c;background:#e2f2e9}.pending{color:var(--amber);background:#fff0c9}.bad{color:var(--red);background:#f9e3e7}.statusbar{display:flex;gap:12px;align-items:center;margin:20px 0}.cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:13px;margin:21px 0 30px}.card,.panel{border:1px solid var(--line);border-radius:10px;padding:20px;background:#fff;min-width:0}.label{font-size:11px;font-weight:650;color:var(--muted)}.value{font-size:27px;letter-spacing:-.035em;font-weight:750;line-height:1.2;margin:10px 0}.detail{font-size:11px;color:var(--muted)}section{margin-top:30px}.section-head{display:flex;justify-content:space-between;align-items:baseline;gap:20px;margin-bottom:12px}.section-head h2{margin:0}.section-head small{font-size:11px}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;text-align:left;font-size:12px}td,th{padding:11px 10px;border-bottom:1px solid var(--line);vertical-align:top}th{font-size:10px;font-weight:650;color:var(--muted);text-transform:uppercase;letter-spacing:.035em}tr:last-child td{border-bottom:0}.mono,dd{font:11px/1.7 ui-monospace,SFMono-Regular,Consolas,monospace;overflow-wrap:anywhere}dl{display:grid;grid-template-columns:130px minmax(0,1fr);gap:9px 15px;margin:0}dt{font-size:11px;color:var(--muted)}dd{margin:0}.case{background:#fff;border:1px solid var(--line);border-radius:11px;margin-bottom:18px;overflow:hidden}.case-top{padding:18px 22px;border-bottom:1px solid var(--line)}.case-top h3{margin:0 0 4px}.case-top p{margin:9px 0 0;font-size:13px}.media-grid{display:grid;grid-template-columns:1fr 1fr;gap:20px;padding:20px 22px}figure{margin:0;min-width:0}.figure-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:9px;font-size:12px}video{display:block;background:#0c1c24;border-radius:7px;aspect-ratio:16/9;width:100%;object-fit:contain}audio{width:100%;margin:48px 0}.empty-media{background:#f3f6f8;min-height:180px;aspect-ratio:16/9;border:1px dashed #bdcfd5;border-radius:7px;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;padding:24px}.empty-media p{font-size:12px;color:var(--muted);max-width:330px;margin:8px 0 0}figcaption{font-size:11px;color:var(--muted);margin-top:10px}.checks{padding:0 22px 18px}.notes{padding-left:20px;font-size:12px;color:var(--muted)}.notes li+li{margin-top:8px}summary{cursor:pointer;font-weight:650;font-size:12px}details{margin-top:13px}footer{border-top:1px solid var(--line);padding-top:18px;margin-top:38px;color:var(--muted);font-size:11px}.tight{margin-top:14px}.empty-state{padding:40px;text-align:center;background:#fff;border:1px dashed #b7cbd0;border-radius:10px}.empty-state h3{font-size:20px}.empty-state p{max-width:650px;margin:auto;color:var(--muted)}@media(max-width:920px){.cards{grid-template-columns:1fr 1fr}.grid2{grid-template-columns:1fr}header{display:block}.download{display:inline-block;margin:6px 0}.section-head{display:block}}@media(max-width:620px){main{padding:23px 12px 40px}h1{font-size:31px}.card,.panel{padding:15px}.value{font-size:23px}.media-grid{grid-template-columns:1fr;padding:15px}.case-top{padding:16px}.statusbar{flex-wrap:wrap}dl{grid-template-columns:100px minmax(0,1fr)}.top span{display:none}.section-head small{display:block}.empty-state{padding:28px 18px}}@media print{.top{background:white;color:#18323b}body{background:white}.download{display:none}.case{break-inside:avoid}main{padding:10px}}
"""


def _checks(checks: list[dict]) -> str:
    if not checks:
        return '<p class="muted">No comparison checks are available. Missing checks are not passes.</p>'
    rows = ''.join(f'<tr><td class="mono">{_escape(check.get("name"))}</td><td>{_badge(check.get("status", "inconclusive"))}</td><td>{_escape(check.get("observed"))}</td><td>{_escape(check.get("threshold"))}</td><td>{_escape(check.get("unit"))}</td><td>{_escape(check.get("reason"))}</td></tr>' for check in checks)
    return '<div class="scroll"><table><thead><tr><th>Recorded check</th><th>Status</th><th>Observed</th><th>Threshold</th><th>Unit</th><th>Interpretation</th></tr></thead><tbody>' + rows + '</tbody></table></div>'


def _pair_metrics(pair: dict) -> str:
    if not pair:
        return ""
    metrics = pair.get("metrics", {})
    psnr = "Exact decoded match" if metrics.get("video_identical") is True else _number(metrics.get("video_psnr_db"), unit=" dB")
    rows = [("Video PSNR", psnr, "RGB fidelity; not visual quality"),
            ("Video MAE", _number(metrics.get("video_mae"), 6), "Normalized RGB, 0–1"),
            ("Audio spectral cosine", _number(metrics.get("audio_spectral_cosine"), 5), "Spectral fidelity; not perceptual quality"),
            ("Audio RMS ratio", _number(metrics.get("audio_rms_ratio"), 5), "Candidate / baseline; channels checked separately")]
    table = ''.join(f'<tr><td>{_escape(name)}</td><td>{_escape(value)}</td><td>{_escape(note)}</td></tr>' for name, value, note in rows)
    return '<div class="checks"><details><summary>Recorded paired fidelity metrics and checks</summary><div class="scroll"><table><thead><tr><th>Metric</th><th>Value</th><th>Meaning</th></tr></thead><tbody>' + table + '</tbody></table></div>' + _checks(pair.get("checks", [])) + '</details></div>'


def _render(report: dict) -> str:
    roles = report["roles"]
    left, right = roles["baseline"], roles["candidate"]
    b, c = left["summary"], right["summary"]
    gpu_timing = any(role["gpu_timing_presented"] for role in roles.values())
    fixture = any(role["evidence_kind"] in {"fixture", "imported_media"} for role in roles.values())
    notice = "Controlled GPU job artifacts" if gpu_timing else "No verified GPU timing to present"
    note = "Source, process, and GPU identities are recorded by the supervisor, not independently attested. Media hashes were checked during export." if gpu_timing else "No model inference is performed by this viewer. Empty, incomplete, or unbound artifacts do not establish an H3 benchmark result."
    if fixture:
        notice, note = "Harness-only / imported evidence — not an H3 GPU result", "At least one run is a fixture or imported-media bundle. Its timing is withheld; any displayed media is explicitly non-live evidence."
    same_gpu = report["same_gpu_uuid_set"]
    pairing = "Same GPU UUIDs" if same_gpu is True else "Different GPU UUIDs" if same_gpu is False else "GPU identity missing"
    peaks = right["telemetry"]["observed_memory_peak_mib_by_gpu"]
    peak = max(peaks.values()) if peaks and right["gpu_timing_presented"] else None
    cards = [
        ("Candidate median end-to-end", _number(c["latency_median_seconds"], unit=" s"), f'Baseline {_number(b["latency_median_seconds"], unit=" s")} · {c["latency_count"]} valid timed slots'),
        ("Candidate runner-valid + hash checked", f'{c["valid"]} / {c["scheduled"]}' if c["scheduled"] is not None else "Not scheduled", "All scheduled measurement slots stay in the denominator"),
        ("Candidate sampled device memory maximum", _number(peak, 0, " MiB"), "Highest observed device sample, NOT exact peak or model-only VRAM"),
        ("Hardware comparison", pairing, "Matching device identities; a regression claim still needs complete comparable runs" if same_gpu is True else "Cross-GPU results are descriptive, not a same-device regression gate"),
    ]
    card_html = ''.join(f'<div class="card"><div class="label">{_escape(label)}</div><div class="value">{_escape(value)}</div><div class="detail">{_escape(detail)}</div></div>' for label, value, detail in cards)
    rows = [
        ("Scheduled measurement slots", "scheduled", False), ("Runner-valid, hash-checked media", "valid", False),
        ("Failed attempts", "failed_attempts", False), ("Explicitly not started", "not_started", False),
        ("Not recorded / pending", "not_recorded", False), ("Invalid or unverified media", "unverified_media", False),
        ("Scheduled warmups (not measurements)", "warmups_scheduled", False), ("Valid warmups", "warmups_valid", False),
        ("Valid timed population", "latency_count", False), ("Median submit → validated media (s)", "latency_median_seconds", True),
        ("Measured serial-block wall time (s)", "wall_seconds", True), ("Valid clips / measured wall second", "valid_clips_per_second", True),
    ]
    summary_rows = ''.join(f'<tr><td>{_escape(label)}</td><td>{_escape(_number(b[key]) if number else b[key])}</td><td>{_escape(_number(c[key]) if number else c[key])}</td></tr>' for label, key, number in rows)
    summary_table = '<div class="panel scroll"><table><thead><tr><th>Metric and population</th><th>Baseline</th><th>Candidate</th></tr></thead><tbody>' + summary_rows + '</tbody></table><p class="detail tight">Warmups are recorded separately and excluded from latency and throughput. Medians condition on valid timed clips; missing or failed attempts are not converted to zero latency.</p></div>'
    role_panels = []
    for name, role in roles.items():
        config, source, telemetry, cleanup = role["configuration"], role["source_identity"], role["telemetry"], role["cleanup"]
        gpu_names = "; ".join(f'{gpu.get("name", "unknown")} · {gpu.get("uuid", "missing UUID")}' for gpu in telemetry["gpu_identity"]) or "Not recorded"
        fields = [("Evidence kind", role["evidence_kind"]), ("Model", config.get("model_id")), ("Model revision", config.get("model_revision")),
                  ("Runtime", config.get("runtime")), ("Runtime revision", config.get("runtime_revision")), ("Observed source revision", source.get("revision")),
                  ("Source fingerprint", source.get("source_sha256")), ("GPU identities", gpu_names), ("Telemetry samples", telemetry["sample_count"]),
                  ("Measurement samples", telemetry["measurement_sample_count"]), ("Telemetry file hash", "Verified" if telemetry["file_sha256_verified"] else "Not verified"),
                  ("Observed owned compute", "; ".join(f"{gpu}: {count} samples" for gpu, count in telemetry["observed_owned_compute_by_gpu"].items()) or "Not recorded"),
                  ("Sampled device maxima", "; ".join(f"{gpu}: {value:,.0f} MiB" for gpu, value in telemetry["observed_memory_peak_mib_by_gpu"].items()) or "Not measured"),
                  ("Run SHA256", role["run_sha256"]), ("Cleanup", cleanup.get("status")), ("Idle after cleanup", cleanup.get("idle_after")),
                  ("Remaining owned PIDs", ", ".join(map(str, cleanup.get("remaining_owned_pids", []))) if "remaining_owned_pids" in cleanup else None),
                  ("Cleanup note", cleanup.get("reason"))]
        extra = [("Source directory", source.get("source")), ("Python", source.get("python")), ("Python version", source.get("python_version")), ("Python SHA256", source.get("python_sha256")), ("SGLang module", source.get("sglang_module"))]
        extra += [("Package: " + name, version) for name, version in source.get("packages", {}).items()]
        extra += [("Process: " + key, value) for key, value in role["process_identity"].items()]
        telemetry_errors = ''.join(f'<li>{_escape(error)}</li>' for error in telemetry["errors"])
        role_panels.append(f'<div class="panel"><div class="figure-head"><h3>{name.title()}</h3>{_badge(role["status"])}</div>{_kv(fields)}<details><summary>Runtime and process provenance</summary>{_kv(extra)}</details>' + (f'<ul class="notes">{telemetry_errors}</ul>' if telemetry_errors else '') + '</div>')
    indexed = {name: {row["slot_id"]: row for row in role["observations"]} for name, role in roles.items()}
    slot_ids = list(dict.fromkeys(list(indexed["baseline"]) + list(indexed["candidate"])))
    cases = []
    for slot_id in slot_ids[:MAX_DISPLAY_SLOTS]:
        slot = indexed["baseline"].get(slot_id) or indexed["candidate"][slot_id]
        cases.append(f'<article class="case"><div class="case-top"><h3>{_escape(slot["case_id"])}</h3><small class="mono">{_escape(slot_id)} · seed {_escape(slot["seed"])} · repetition {_escape(slot["repetition"])}</small><p>{_escape(slot["prompt"])}</p></div><div class="media-grid">{_player("baseline", indexed["baseline"].get(slot_id))}{_player("candidate", indexed["candidate"].get(slot_id))}</div>{_pair_metrics(report["slot_comparisons"].get(slot_id, {}))}</article>')
    if not cases:
        cases = ['<div class="empty-state"><h3>Ready for actual run artifacts</h3><p>No frozen measurement slots are available yet. This page deliberately contains no sample video, invented latency, or placeholder benchmark result.</p></div>']
    truncated = f'<p class="detail">Showing the first {MAX_DISPLAY_SLOTS} of {len(slot_ids)} scheduled slots in frozen order. Every slot remains in the evidence JSON and summary denominators.</p>' if len(slot_ids) > MAX_DISPLAY_SLOTS else ''
    failures = report["failures"] + [f'{issue["context"]}: {issue["message"]}' for issue in report["issues"]]
    problems = '<aside class="notice error"><strong>Missing evidence or execution issues</strong><ul class="notes">' + ''.join(f'<li>{_escape(failure)}</li>' for failure in failures) + '</ul></aside>' if failures else ''
    policy = report["policy"]
    policy_fields = [("Policy", policy.get("policy_id")), ("Calibration", policy.get("calibration_status", "Not calibrated")), ("Allocation", report["allocation"].get("mode")), ("Allocation label", report["allocation"].get("label")), ("Same frozen workload", report["same_workload"]), ("Job started", report.get("started_at")), ("Job finished", report.get("finished_at")), ("Spec SHA256", report.get("spec_sha256"))]
    policy_fields += [("Max latency increase (fraction)", policy.get("max_latency_increase_fraction")), ("Min video PSNR (dB)", policy.get("min_video_psnr_db")),
                      ("Min spectral cosine", policy.get("min_audio_spectral_cosine")), ("Max RMS ratio error", policy.get("max_audio_rms_ratio_error")),
                      ("Max sampled-memory increase (fraction)", policy.get("max_memory_increase_fraction"))]
    authorization = report["authorization"]
    policy_fields += [("Compute approved (operator assertion)", authorization.get("compute_approved")), ("License reviewed (operator assertion)", authorization.get("model_license_reviewed")), ("Approval reference", authorization.get("approval_reference"))]
    gate = report["recorded_gate"]
    gate_fields = [("Recorded regression gate", gate.get("regression_status")), ("CI accepted (supervisor reported)", gate.get("ci_accepted")), ("Sampled-memory gate", gate.get("memory_gate_status")),
                   ("Sampled-memory change (fraction)", gate.get("memory_increase_fraction")), ("Sampled-memory threshold (fraction)", gate.get("memory_threshold"))]
    acceptance_notes = ''.join(f'<li>{_escape(reason)}</li>' for reason in report["acceptance_reasons"])
    warmup_rows = ''.join(f'<tr><td>{name.title()}</td><td class="mono">{_escape(row["slot_id"])}</td><td>{_badge(row["status"])}</td><td>{_escape(row["error"] or ("Runner-valid, hash checked" if row["valid"] else "Not verified valid"))}</td></tr>' for name, role in roles.items() for row in role["warmups"])
    warmup_ledger = '<details><summary>Warmup ledger (excluded from measurements)</summary><div class="scroll"><table><thead><tr><th>Arm</th><th>Slot</th><th>Status</th><th>Evidence</th></tr></thead><tbody>' + warmup_rows + '</tbody></table></div></details>' if warmup_rows else ''
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'; media-src \'self\' file:; base-uri \'none\'; form-action \'none\'">'
            f'<title>H3 controlled GPU benchmark · {_escape(report["job_id"])}</title><style>{_CSS}</style></head><body>'
            '<div class="top">VIDEO GENERATION BENCHMARK<span>Artifact-driven · local · script-free</span></div><main><header><div><div class="eyebrow">Controlled GPU execution</div><h1>H3 benchmark evidence</h1><p class="subtitle">A fixed-workload view of generated video, native audio, request timing, observed GPU memory, and cleanup. Evidence comes only from this job directory.</p></div><a class="download" href="evidence.json" download>Download evidence JSON ↗</a></header>'
            f'<div class="statusbar">{_badge(report["status"])}<span class="mono">{_escape(report["job_id"])}</span><span>Recorded CI verdict: {_badge(report["ci_status"])}</span></div>'
            f'<aside class="notice {"live" if gpu_timing and not fixture else ""}"><strong>{_escape(notice)}</strong><p>{_escape(note)}</p></aside>'
            '<p class="detail">The viewer checks artifact integrity; it does not independently qualify CI or certify a release. Uncalibrated thresholds, shared-host allocation, missing evidence, and same-build repeatability are not a demonstrated performance improvement.</p>'
            f'{problems}<div class="cards">{card_html}</div><section><div class="section-head"><h2>Generated media, side by side</h2><small>Video and native audio use the original exported bytes</small></div>{truncated}{"".join(cases)}</section>'
            f'<section><div class="section-head"><h2>All-slot accounting</h2><small>Recorded media validity is not a prompt-quality score</small></div>{summary_table}{warmup_ledger}</section>'
            '<section class="panel"><h2>What the timing means</h2><p class="muted">All timing values are seconds. Submit → validated media includes server queueing and generation, status polling, download, and client-side decoding/validation. Terminal observed is poll-observed completion, not exact GPU completion. Download complete includes transfer; client validation is CPU-side analysis. These are not GPU kernel latency or time-to-first-frame measurements.</p><p class="detail">Throughput is valid clips per measured serial-block wall second at concurrency 1, not saturation capacity. Device VRAM is periodically sampled; sampled maxima can miss a true peak and are not process-isolated allocations.</p></section>'
            f'<section><div class="section-head"><h2>Identity, resources, and cleanup</h2><small>Supervisor-recorded provenance · not independent attestation</small></div><div class="grid2">{"".join(role_panels)}</div></section>'
            f'<section class="panel"><h2>Recorded comparison checks</h2>{_checks(report["checks"])}<p class="detail tight">Check status: {_escape(report["comparison_status"])}. Checks are read from the hash-bound comparison; this viewer does not rerun the decoder, calibrate thresholds, or make statistical significance claims.</p></section>'
            f'<section class="panel"><h2>Recorded supervisor acceptance</h2>{_kv(gate_fields)}<ul class="notes">{acceptance_notes}</ul><p class="detail">Acceptance is reported, not independently re-evaluated by this viewer. External calibration references are not opened. No release qualification is claimed.</p></section>'
            f'<section class="panel"><h2>Workload and policy</h2>{_kv(policy_fields)}<p class="detail tight">Approval and license-review fields are operator assertions, not proof of legal rights or scheduler isolation. Same-GPU runtime regression requires the same observed device UUIDs and frozen request cell. Cross-chip performance is a separate descriptive comparison. Neither fidelity checks nor timing establish aesthetics, physics, prompt following, lip synchronization, or human preference.</p></section>'
            f'<footer>Exported {_escape(report["created_at"])}. Share index.html, evidence.json, and assets/ together. No remote resources, scripts, automatic playback, or background job execution.</footer></main></body></html>')


def write_gpu_report(jobdir: Path, outputdir: Path) -> dict:
    """Export a safe artifact-backed dashboard into a *new* directory.

    Missing/malformed job artifacts produce a clearly incomplete/error page.
    Unsafe root/output paths and an existing output directory are rejected.
    The return value is the sanitized JSON evidence also written to disk.
    """
    root, output = Path(jobdir).absolute(), Path(outputdir).absolute()
    _no_symlink_parents(root)
    _no_symlink_parents(output)
    if not root.is_dir():
        raise ValueError("jobdir must be an existing directory")
    if output.exists():
        raise FileExistsError("Report output directory already exists")
    tree, issues = _Tree(root), []
    receipt, _ = _document(tree, "gpu-job.json", issues)
    if receipt and (receipt.get("bundle_type") != "controlled_gpu_job" or receipt.get("schema_version") != "0.1.0"):
        _issue(issues, "gpu-job.json", "Unsupported controlled GPU job type or version")
        receipt = None
    receipt = receipt or {}
    spec, spec_digest = _document(tree, "spec.json", issues, expected=receipt.get("spec_sha256"), canonical_hash=True)
    if receipt and spec is None:
        _issue(issues, "spec.json", "Job specification is missing or invalid")
    spec = spec or {}
    output.mkdir(parents=True, exist_ok=False)
    assets = output / "assets"
    assets.mkdir()
    metadata = receipt.get("roles") if isinstance(receipt.get("roles"), dict) else {}
    controlled = bool(receipt.get("evidence_kind") == "controlled_h3_gpu" and spec and _is_digest(receipt.get("spec_sha256")) and spec_digest == receipt["spec_sha256"])
    budget = [0]
    roles = {label: _role(tree, label, metadata.get(label), spec, controlled, assets, budget, issues) for label in _ROLES}
    comparison = None
    if receipt.get("comparison_path") is not None:
        comparison, _ = _document(tree, receipt["comparison_path"], issues, expected=receipt.get("comparison_sha256"))
        if comparison is None:
            _issue(issues, "comparison", "Receipt references a comparison that is missing or invalid")
        if comparison:
            matches = (comparison.get("bundle_type") == "mvp_comparison" and _is_digest(receipt.get("comparison_sha256"))
                       and all(isinstance(comparison.get(label), dict) and roles[label]["run_sha256"] is not None
                               and comparison[label].get("run_bundle_sha256") == roles[label]["run_sha256"] for label in _ROLES))
            if not matches:
                _issue(issues, "comparison", "Comparison is not bound to both observed run bundles")
                comparison = None
    comparison = comparison or {}
    checks = comparison.get("checks", [])
    checks = [_check_data(check) for check in checks[:1000] if isinstance(check, dict)] if isinstance(checks, list) else []
    paired_slots = {}
    comparison_slots = comparison.get("slots", [])
    if isinstance(comparison_slots, list):
        observed = {label: {row["slot_id"]: row for row in roles[label]["observations"]} for label in _ROLES}
        for slot in comparison_slots[:MAX_SLOTS]:
            if not isinstance(slot, dict) or not isinstance(slot.get("slot_id"), str):
                continue
            identifier = slot["slot_id"]
            if not all(identifier in observed[label] and isinstance(slot.get(label), dict) and slot[label].get("sha256") == observed[label][identifier]["sha256"] for label in _ROLES):
                _issue(issues, "comparison", "Paired slot does not match the exported media hashes")
                continue
            slot_checks = slot.get("checks", [])
            paired_slots[identifier] = {
                "metrics": _selected(slot.get("metrics"), ("video_psnr_db", "video_mae", "video_identical", "audio_spectral_cosine", "audio_rms_ratio", "latency_increase_fraction")),
                "checks": [_check_data(check) for check in slot_checks[:256] if isinstance(check, dict)] if isinstance(slot_checks, list) else [],
            }
    gpu_sets = [{gpu["uuid"] for gpu in roles[label]["telemetry"]["gpu_identity"] if isinstance(gpu.get("uuid"), str)} if roles[label]["telemetry"]["samples_consistent"] else set() for label in _ROLES]
    plans = [roles[label]["plan_sha256"] for label in _ROLES]
    policy = comparison.get("policy", spec.get("policy", {}))
    policy = policy if isinstance(policy, dict) else {}
    allocation = spec.get("allocation") if isinstance(spec.get("allocation"), dict) else {}
    ci_status = receipt.get("regression_status", "inconclusive")
    if ci_status not in {"pass", "fail", "inconclusive"}:
        ci_status = "inconclusive"
    if ci_status == "pass" and (receipt.get("status") != "complete" or receipt.get("failures") != []
                               or receipt.get("cleanup_status") != "clean" or not _finalized(receipt)
                               or issues or receipt.get("ci_accepted") is not True or receipt.get("measurement_status") != "complete"
                               or comparison.get("overall_status") != "pass" or policy.get("calibration_status") != "operator_calibrated"
                               or allocation.get("mode") != "dedicated_ci" or not all(role["gpu_timing_presented"] for role in roles.values())
                               or any(role["cleanup"].get("status") != "clean" or role["cleanup"].get("idle_after") is not True for role in roles.values())):
        ci_status = "inconclusive"
        _issue(issues, "CI verdict", "A recorded pass is not shown as passing because completed execution, cleanup, integrity, provenance, allocation, or calibration evidence is incomplete")
    status = receipt.get("status", "not_started")
    if status not in {"complete", "failed", "aborted", "running", "preflight", "not_started"}:
        _issue(issues, "gpu-job.json", "Unknown job state")
        status = "invalid"
    if issues and status in {"complete", "not_started"}:
        status = "invalid" if not receipt else "incomplete"
    if status == "complete" and (receipt.get("measurement_status") != "complete" or any(role["summary"]["scheduled"] is None or role["summary"]["not_recorded"] for role in roles.values())):
        status = "incomplete"
    report = {
        "schema_version": "0.1.0", "bundle_type": "controlled_gpu_report", "created_at": datetime.now(timezone.utc).isoformat(),
        "job_id": _text(spec.get("job_id", receipt.get("job_id", "No job receipt"))), "status": status,
        "supervisor_evidence_kind": _text(receipt.get("evidence_kind", "no_gpu_measurement")),
        "measurement_status": _text(receipt.get("measurement_status", "incomplete")),
        "ci_status": ci_status, "ci_accepted": ci_status == "pass", "release_qualified": False,
        "ci_interpretation": "Supervisor-reported CI outcome, guarded against incomplete local artifact integrity; not independent CI qualification",
        "supervisor_claimed_ci_accepted": receipt.get("ci_accepted") if isinstance(receipt.get("ci_accepted"), bool) else None,
        "started_at": _text(receipt.get("started_at")), "finished_at": _text(receipt.get("finished_at")),
        "spec_sha256": spec_digest, "roles": roles, "same_gpu_uuid_set": gpu_sets[0] == gpu_sets[1] if all(gpu_sets) else None,
        "same_workload": plans[0] == plans[1] if all(plans) else None,
        "allocation": _selected(spec.get("allocation"), ("mode", "label")),
        "authorization": _selected(spec.get("authorization"), ("compute_approved", "model_license_reviewed", "approval_reference")),
        "model_identity": _selected(receipt.get("model_identity"), ("path", "revision", "manifest_sha256", "verified_files", "total_bytes")),
        "policy": _selected(policy, ("policy_id", "calibration_status", "max_latency_increase_fraction", "min_video_psnr_db", "min_audio_spectral_cosine", "max_audio_rms_ratio_error", "max_memory_increase_fraction")),
        "recorded_gate": _selected(receipt, ("regression_status", "ci_accepted", "memory_gate_status", "memory_increase_fraction", "memory_threshold")),
        "acceptance_reasons": [_text(reason) for reason in receipt.get("acceptance_reasons", [])[:128]] if isinstance(receipt.get("acceptance_reasons"), list) else [],
        "comparison_status": _text(comparison.get("overall_status", "inconclusive")), "checks": checks, "slot_comparisons": paired_slots,
        "failures": [_text(item) for item in receipt.get("failures", [])[:256]] if isinstance(receipt.get("failures"), list) else [],
        "issues": issues, "verification": "Media SHA256 checked; recorded decoder results displayed, not recomputed; supervisor identity is not independently attested",
        "report": {"html": "index.html", "json": "evidence.json", "assets": "assets", "scripts": False},
    }
    with (output / "evidence.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    with (output / "index.html").open("x", encoding="utf-8") as stream:
        stream.write(_render(report))
    return report
