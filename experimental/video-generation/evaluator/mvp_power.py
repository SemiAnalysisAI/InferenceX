"""Validate H3 GPU-board power without turning missing samples into energy."""

from __future__ import annotations

import csv
import math
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory

from utils.aggregate_power import integrate_power


_CLOCK_TOLERANCE_SECONDS = 0.1
_LEGACY_JOURNAL_TOLERANCE_SECONDS = 0.25


def _number(value: object) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _utc(value: object) -> float:
    if not isinstance(value, str):
        raise ValueError("missing UTC timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp has no timezone")
    return parsed.timestamp()


def _ownership(sample: dict, owner: dict, devices: list[str]) -> list[str]:
    """Require the retained raw process inventory to partition into owned PIDs."""
    raw = sample.get("compute_apps")
    owned = sample.get("owned_compute_apps")
    if not isinstance(raw, list) or not isinstance(owned, list) or sample.get("unowned_compute_apps") != []:
        return ["gpu_ownership_unverified"]
    keys = ("gpu_uuid", "pid", "memory_used_mib")
    try:
        raw_rows = [tuple(app.get(key) for key in keys) for app in raw]
        owned_rows = [tuple(app.get(key) for key in keys) for app in owned]
        if len(set(raw_rows)) != len(raw_rows) or set(raw_rows) != set(owned_rows) or len(raw_rows) != len(owned_rows):
            return ["gpu_ownership_partition_mismatch"]
        for app in owned:
            identity = app.get("process_identity", {})
            if (app.get("gpu_uuid") not in devices or type(app.get("pid")) is not int
                    or app["pid"] <= 0 or identity.get("pid") != app["pid"]
                    or not owner.get("pgid") or identity.get("pgid") != owner["pgid"]
                    or not owner.get("session_id") or identity.get("session_id") != owner["session_id"]
                    or not _number(identity.get("start_ticks")) or identity["start_ticks"] <= 0):
                return ["gpu_process_identity_mismatch"]
    except (AttributeError, TypeError):
        return ["gpu_ownership_unverified"]
    return []


def _timing(record: dict, events: list[dict], offset: float | None, spread: float | None) -> dict:
    """Prefer explicit monotonic boundaries; recover only tightly agreeing journals."""
    result = {"start_monotonic_seconds": None, "end_monotonic_seconds": None,
              "timing_source": None, "timing_uncertainty_seconds": None, "invalid_reasons": []}
    if record.get("attempted") is not True:
        result["invalid_reasons"].append("request_not_attempted")
        return result
    duration = record.get("submit_to_terminal_seconds")
    latency = record.get("latency_seconds")
    if not _number(duration) or not _number(latency) or not 0 < duration <= latency:
        result["invalid_reasons"].append("invalid_request_duration")
        return result
    explicit = record.get("timing_window")
    if explicit is not None:
        result["timing_source"] = "recorded_monotonic_submit_to_terminal"
        try:
            start, terminal, end = (explicit[key] for key in (
                "start_monotonic_seconds", "terminal_monotonic_seconds", "end_monotonic_seconds"))
            if (not all(_number(value) for value in (start, terminal, end)) or not start < terminal <= end
                    or abs((terminal - start) - duration) > _CLOCK_TOLERANCE_SECONDS
                    or abs((end - start) - latency) > _CLOCK_TOLERANCE_SECONDS
                    or offset is None or abs(_utc(explicit["start_utc"]) - offset - start) > _CLOCK_TOLERANCE_SECONDS):
                raise ValueError("inconsistent timing")
            result.update(start_monotonic_seconds=start, end_monotonic_seconds=terminal,
                          timing_uncertainty_seconds=0.0)
        except (KeyError, TypeError, ValueError):
            result["invalid_reasons"].append("invalid_recorded_timing_window")
        return result
    result["timing_source"] = "legacy_utc_journal_mapped_to_monotonic"
    starts = [event for event in events if event.get("event") == "attempt_started" and event.get("slot_id") == record.get("slot_id")]
    finishes = [event for event in events if event.get("event") == "attempt_finished"
                and isinstance(event.get("record"), dict) and event["record"].get("slot_id") == record.get("slot_id")]
    try:
        if len(starts) != 1 or len(finishes) != 1 or finishes[0].get("record") != record or offset is None:
            raise ValueError("missing or conflicting journal")
        start_utc, finish_utc = _utc(starts[0]["at"]), _utc(finishes[0]["at"])
        slack = finish_utc - start_utc - latency
        if not 0 <= slack <= _LEGACY_JOURNAL_TOLERANCE_SECONDS:
            raise ValueError("journal does not agree with monotonic duration")
        # The completed journal follows the timed request. Its residual bounds
        # the possible boundary overhead; it is not exact kernel timing.
        start = finish_utc - offset - latency
        result.update(start_monotonic_seconds=start, end_monotonic_seconds=start + duration,
                      timing_uncertainty_seconds=slack + (spread or 0.0))
    except (KeyError, TypeError, ValueError):
        result["invalid_reasons"].append("legacy_timing_unverifiable")
    return result


def _startup(role: dict, offset: float | None, spread: float | None) -> dict:
    result = {"phase": "startup", "slot_id": None, "attempted": 0, "completed": 0, "valid_clips": 0,
              "start_monotonic_seconds": None, "end_monotonic_seconds": None,
              "timing_source": None, "timing_uncertainty_seconds": None, "invalid_reasons": []}
    explicit = role.get("startup_timing_window")
    try:
        duration = role.get("startup_seconds")
        if not _number(duration) or duration <= 0 or offset is None:
            raise ValueError("missing startup time")
        if explicit is not None:
            start, end = explicit["start_monotonic_seconds"], explicit["end_monotonic_seconds"]
            result["timing_source"] = "recorded_monotonic_startup"
            uncertainty = 0.0
            if abs(_utc(explicit["start_utc"]) - offset - start) > _CLOCK_TOLERANCE_SECONDS:
                raise ValueError("startup clock disagreement")
        else:
            start = _utc(role.get("started_at")) - offset
            end = role.get("telemetry_summary", {}).get("measurement_window_start_monotonic_seconds")
            result["timing_source"] = "legacy_role_start_to_sampler_measurement_start"
            uncertainty = _CLOCK_TOLERANCE_SECONDS + (spread or 0.0)
        if not _number(start) or not _number(end) or end <= start or abs(end - start - duration) > _CLOCK_TOLERANCE_SECONDS:
            raise ValueError("startup duration disagreement")
        result.update(start_monotonic_seconds=start, end_monotonic_seconds=end,
                      timing_uncertainty_seconds=uncertainty)
    except (KeyError, TypeError, ValueError):
        result["invalid_reasons"].append("startup_timing_unverifiable")
    return result


def _summarize_windows(windows: list[dict], devices: list[str]) -> dict:
    result = {"status": "not_requested" if not windows else "invalid", "valid": False,
              "invalid_reasons": sorted({reason for window in windows for reason in window["invalid_reasons"]}),
              "window_count": len(windows), "valid_window_count": sum(window["valid"] for window in windows),
              "attempted": sum(window["attempted"] for window in windows),
              "completed": sum(window["completed"] for window in windows),
              "valid_clips": sum(window["valid_clips"] for window in windows),
              "duration_seconds": None, "per_gpu": None, "aggregate": None}
    if not windows or not all(window["valid"] for window in windows):
        return result
    duration = sum(window["duration_seconds"] for window in windows)
    per_gpu = {device: {"energy_j": sum(window["per_gpu"][device]["energy_j"] for window in windows),
                        "observed_peak_power_w": max(window["per_gpu"][device]["observed_peak_power_w"] for window in windows)}
               for device in devices}
    for value in per_gpu.values():
        value["avg_power_w"] = value["energy_j"] / duration
    energy = sum(value["energy_j"] for value in per_gpu.values())
    if (not _number(duration) or duration <= 0 or not _number(energy)
            or any(not _number(value[key]) or value[key] < 0 for value in per_gpu.values() for key in ("energy_j", "avg_power_w"))):
        result["invalid_reasons"].append("nonfinite_phase_power_integration")
        return result
    result.update(status="valid", valid=True, duration_seconds=duration, per_gpu=per_gpu,
                  aggregate={"energy_j": energy, "avg_power_w": energy / duration,
                             "observed_peak_power_w": max(window["aggregate"]["observed_peak_power_w"] for window in windows),
                             "joules_per_valid_clip": energy / result["valid_clips"] if result["valid_clips"] else None})
    return result


def analyze_power(role: dict, run: dict, samples: list[dict], events: list[dict], gpu_uuids: list[str], *, interval_seconds: float) -> dict:
    """Return phase-scoped, reproducible GPU-board power with invalid values withheld.

    Request energy covers submit through observed provider terminal state, not
    client download/decoding or exact GPU kernel execution. Failed/invalid clips
    retain their observed generation energy in the phase numerator; only valid
    clips enter the denominator. Legacy UTC recovery requires a stable sampled
    clock offset within 100 ms and journal/latency agreement within 250 ms.
    """
    global_reasons = []
    if not gpu_uuids or any(not isinstance(device, str) or not device for device in gpu_uuids) or len(set(gpu_uuids)) != len(gpu_uuids):
        global_reasons.append("invalid_gpu_inventory")
    if not _number(interval_seconds) or interval_seconds <= 0:
        global_reasons.append("invalid_sampling_interval")
    max_gap = 3 * interval_seconds if not global_reasons else 0.0
    offsets, series = [], []
    previous = None
    for sample in samples:
        stamp = sample.get("monotonic_seconds")
        reasons = []
        try:
            if not _number(stamp) or (previous is not None and stamp <= previous):
                raise ValueError("sample timestamps are not increasing")
            offsets.append(_utc(sample.get("at")) - stamp)
            previous = stamp
        except (TypeError, ValueError):
            global_reasons.append("invalid_or_nonmonotonic_sample_time")
        devices = sample.get("gpus", [])
        powers = {device: None for device in gpu_uuids}
        if not isinstance(devices, list) or any(not isinstance(device, dict) for device in devices):
            devices = []
        if sorted(str(device.get("uuid")) for device in devices) != sorted(gpu_uuids):
            reasons.append("sample_gpu_inventory_mismatch")
        for device in devices:
            if device.get("uuid") in powers:
                value = device.get("power_watts")
                powers[device["uuid"]] = value if _number(value) and value >= 0 else None
        if any(value is None for value in powers.values()):
            reasons.append("invalid_power_sample")
        aggregate = sum(powers.values()) if not reasons else None
        if aggregate is not None and not _number(aggregate):
            reasons.append("nonfinite_aggregate_power_sample")
        query = sample.get("power_query")
        if query is not None:
            try:
                begin, finish = query["start_monotonic_seconds"], query["end_monotonic_seconds"]
                if (not all(_number(value) for value in (begin, finish, stamp)) or not begin <= finish <= stamp
                        or abs((_utc(query["start_utc"]) - begin) - (_utc(sample["at"]) - stamp)) > _CLOCK_TOLERANCE_SECONDS):
                    raise ValueError("power acquisition bracket is inconsistent")
            except (KeyError, TypeError, ValueError):
                reasons.append("invalid_power_acquisition_window")
        reasons.extend(_ownership(sample, role.get("process_identity", {}), gpu_uuids))
        series.append({"at": sample.get("at"), "monotonic_seconds": stamp if _number(stamp) else None,
                       "per_gpu_watts": powers, "aggregate_watts": aggregate if not reasons else None,
                       "valid": not reasons, "invalid_reasons": reasons})
    spread = max(offsets) - min(offsets) if offsets else None
    offset = median(offsets) if offsets else None
    if spread is None or spread > _CLOCK_TOLERANCE_SECONDS:
        global_reasons.append("sample_clock_offset_unstable_or_missing")
        offset = None
    windows = [_startup(role, offset, spread)]
    seen_slots = set()
    for record in run.get("records", []):
        phase, slot = record.get("phase"), record.get("slot_id")
        if phase not in ("warmup", "measurement") or not isinstance(slot, str) or slot in seen_slots:
            global_reasons.append("invalid_or_duplicate_record_slot")
            continue
        seen_slots.add(slot)
        windows.append({"phase": phase, "slot_id": slot, "case_id": record.get("case_id"),
                        "attempted": int(record.get("attempted") is True),
                        "completed": int(record.get("status") == "succeeded"),
                        "valid_clips": int(record.get("status") == "succeeded" and isinstance(record.get("media"), dict) and record["media"].get("valid") is True),
                        **_timing(record, events, offset, spread)})
    if run.get("configuration", {}).get("serving"):
        from .mvp_serving import validate_window
        try:
            validate_window(run)
        except (KeyError, TypeError, ValueError):
            global_reasons.append("invalid_serving_measurement_window")
        measured = [window for window in windows if window["phase"] == "measurement"]
        if measured:
            starts = [w["start_monotonic_seconds"] for w in measured]
            ends = [w["end_monotonic_seconds"] for w in measured]
            # Per-request board power cannot be attributed under concurrency.
            combined = {"phase": "measurement", "slot_id": None,
                        "request_slot_ids": [w["slot_id"] for w in measured],
                        "timing_source": "serving_first_submit_to_last_observed_terminal",
                        "timing_uncertainty_seconds": max((w["timing_uncertainty_seconds"] or 0 for w in measured)),
                        "start_monotonic_seconds": min(starts) if all(_number(v) for v in starts) else None,
                        "end_monotonic_seconds": max(ends) if all(_number(v) for v in ends) else None,
                        "invalid_reasons": sorted({reason for w in measured for reason in w["invalid_reasons"]}),
                        **{key: sum(w[key] for w in measured) for key in ("attempted", "completed", "valid_clips")}}
            windows = [w for w in windows if w["phase"] != "measurement"] + [combined]
    ordered = sorted((window for window in windows if window["start_monotonic_seconds"] is not None), key=lambda window: window["start_monotonic_seconds"])
    for left, right in zip(ordered, ordered[1:]):
        if left["end_monotonic_seconds"] > right["start_monotonic_seconds"]:
            left["invalid_reasons"].append("overlapping_phase_windows")
            right["invalid_reasons"].append("overlapping_phase_windows")
    with TemporaryDirectory(prefix="h3-power-") as directory:
        path = Path(directory) / "power.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["timestamp", "gpu_id", "power.draw [W]"])
            for sample in series:
                for device, power in sample["per_gpu_watts"].items():
                    # The shared CSV parser accepts decimal notation, not
                    # exponents. Preserve float values rather than parsing 6e2 as 6.
                    stamp = sample["monotonic_seconds"]
                    writer.writerow([format(Decimal(str(stamp)), "f") if stamp is not None else "",
                                     device, format(Decimal(str(power)), "f") if power is not None else "N/A"])
        for window in windows:
            reasons = window["invalid_reasons"] + global_reasons
            start, end = window["start_monotonic_seconds"], window["end_monotonic_seconds"]
            window.update(valid=False, status="invalid", duration_seconds=None, per_gpu=None, aggregate=None, coverage=None)
            if start is not None and end is not None and end > start and not global_reasons:
                integrated = integrate_power(path, start_unix=start, end_unix=end,
                                             expected_num_gpus=len(gpu_uuids), max_sample_gap_s=max_gap)
                reasons.extend(integrated.invalid_reasons)
                if integrated.power_valid and any(not _number(value) or value < 0 for value in (
                        integrated.total_gpu_energy_j, integrated.avg_total_gpu_power_w, integrated.avg_power_w,
                        *integrated.per_gpu_energy_j.values())):
                    reasons.append("nonfinite_or_negative_power_integration")
                supporting = [sample for sample in series if start - max_gap <= sample["monotonic_seconds"] <= end + max_gap]
                for sample in supporting:
                    reasons.extend(sample["invalid_reasons"])
                within = [sample for sample in series if start <= sample["monotonic_seconds"] <= end]
                if not within:
                    reasons.append("no_in_window_power_samples")
                if window["phase"] != "startup":
                    owned = {app.get("gpu_uuid") for sample in samples if start <= sample["monotonic_seconds"] <= end
                             for app in sample.get("owned_compute_apps", [])}
                    if not set(gpu_uuids).issubset(owned):
                        reasons.append("owned_compute_not_observed_on_every_gpu")
                covered = sum(max(0.0, min(end, right["monotonic_seconds"]) - max(start, left["monotonic_seconds"]))
                              for left, right in zip(series, series[1:])
                              if left["valid"] and right["valid"] and right["monotonic_seconds"] - left["monotonic_seconds"] <= max_gap)
                window["duration_seconds"] = end - start
                window["coverage"] = {"sample_count": len(within), "covered_duration_seconds": covered,
                                      "coverage_fraction": min(1.0, covered / (end - start)),
                                      "per_gpu_sample_counts": integrated.per_gpu_sample_counts,
                                      "per_gpu_max_sample_gap_seconds": integrated.per_gpu_max_sample_gap_s,
                                      "maximum_allowed_sample_gap_seconds": max_gap,
                                      "bracketed": "benchmark_window_not_bracketed" not in integrated.invalid_reasons}
                if not reasons:
                    energy = integrated.total_gpu_energy_j
                    window.update(valid=True, status="valid",
                                  per_gpu={device: {"energy_j": integrated.per_gpu_energy_j[device],
                                                    "avg_power_w": integrated.per_gpu_energy_j[device] / (end - start),
                                                    "observed_peak_power_w": max(sample["per_gpu_watts"][device] for sample in within)} for device in gpu_uuids},
                                  aggregate={"energy_j": energy, "avg_power_w": integrated.avg_total_gpu_power_w,
                                             "observed_peak_power_w": max(sample["aggregate_watts"] for sample in within),
                                             "joules_per_valid_clip": energy / window["valid_clips"] if window["valid_clips"] else None})
            window["invalid_reasons"] = sorted(set(reasons))
    phases = {phase: _summarize_windows([window for window in windows if window["phase"] == phase], gpu_uuids)
              for phase in ("startup", "warmup", "measurement")}
    requested = [phase for phase in phases.values() if phase["window_count"]]
    valid = bool(requested) and all(phase["valid"] for phase in requested)
    return {"schema_version": "1.0.0", "valid": valid,
            "status": "valid" if valid else ("partial" if any(phase["valid"] for phase in requested) else "invalid"),
            "invalid_reasons": sorted({reason for phase in phases.values() for reason in phase["invalid_reasons"]}),
            "semantics": {"scope": "selected_gpu_boards_including_memory; excludes_host_and_unselected_gpus",
                          "power_unit": "W", "energy_unit": "J", "time_unit": "s",
                          "integration": "per_device_trapezoidal_with_linear_boundary_interpolation",
                          "generation_window": ("first_submit_to_last_observed_provider_terminal; includes_intervening_idle_download_and_validation_time; concurrent_board_energy_integrated_once"
                                                if run.get("configuration", {}).get("serving") else
                                                "submit_to_observed_provider_terminal; excludes_client_download_and_decode"),
                          "peak": "maximum_observed_sensor_sample_in_window; not_instantaneous_electrical_peak",
                          "energy_per_valid_clip": "sum_generation_energy_including_failed_or_invalid_completed_attempts_divided_by_technically_valid_clips",
                          "sensor": "nvidia-smi power.draw; H200 NVML trailing_one_second_average; phase_edges_have_sensor_averaging_uncertainty",
                          "clock_agreement_limit_seconds": _CLOCK_TOLERANCE_SECONDS,
                          "legacy_journal_agreement_limit_seconds": _LEGACY_JOURNAL_TOLERANCE_SECONDS},
            "clock_alignment": {"utc_minus_monotonic_seconds": offset, "observed_offset_spread_seconds": spread},
            "gpu_uuids": gpu_uuids, "requested_interval_seconds": interval_seconds,
            "sample_series": series, "windows": windows, "phases": phases}
