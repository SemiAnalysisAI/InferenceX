"""Hand-worked phase integration and fail-closed telemetry boundaries."""

import json
from copy import deepcopy
from datetime import datetime, timezone

import pytest

from evaluator.mvp_power import analyze_power


def utc(seconds):
    return datetime.fromtimestamp(1_000_000 + seconds, timezone.utc).isoformat()


def record(slot, phase, start, terminal, *, valid=True):
    return {"slot_id": slot, "case_id": "case", "phase": phase, "attempted": True,
            "status": "succeeded", "media": {"valid": valid},
            "submit_to_terminal_seconds": terminal - start,
            "latency_seconds": terminal - start + 0.25,
            "timing_window": {"start_monotonic_seconds": start,
                              "terminal_monotonic_seconds": terminal,
                              "end_monotonic_seconds": terminal + 0.25, "start_utc": utc(start)}}


def data():
    role = {"process_identity": {"pgid": 10, "session_id": 10}, "startup_seconds": 2,
            "startup_timing_window": {"start_monotonic_seconds": 0, "end_monotonic_seconds": 2,
                                      "start_utc": utc(0)}}
    run = {"records": [record("warmup", "warmup", 3, 5), record("measured", "measurement", 6.5, 9.5)]}
    samples = []
    for time in range(-1, 14):
        apps = [{"gpu_uuid": gpu, "pid": pid, "memory_used_mib": 20} for gpu, pid in (("a", 11), ("b", 12))]
        samples.append({"at": utc(time), "monotonic_seconds": time,
                        "gpus": [{"uuid": "a", "power_watts": 10 + 5 * time}, {"uuid": "b", "power_watts": 20 + 10 * time}],
                        "compute_apps": apps, "unowned_compute_apps": [],
                        "owned_compute_apps": [{**app, "process_identity": {"pid": app["pid"], "pgid": 10, "session_id": 10, "start_ticks": 1}} for app in apps]})
    return role, run, samples


def analyze(role, run, samples, events=None):
    return analyze_power(role, run, samples, events or [], ["a", "b"], interval_seconds=1)


def test_ramp_clipping_separates_phases_and_excludes_client_decode():
    result = analyze(*data())
    assert result["valid"] is True
    measured = result["phases"]["measurement"]
    assert measured["duration_seconds"] == 3
    assert measured["per_gpu"]["a"] == {"energy_j": 150, "avg_power_w": 50, "observed_peak_power_w": 55}
    assert measured["aggregate"] == {"energy_j": 450, "avg_power_w": 150, "observed_peak_power_w": 165, "joules_per_valid_clip": 450}
    assert result["phases"]["startup"]["aggregate"]["energy_j"] == 90
    assert result["phases"]["warmup"]["aggregate"]["energy_j"] == 180
    assert result["windows"][2]["coverage"]["coverage_fraction"] == 1
    assert result["sample_series"][8]["aggregate_watts"] == 135


@pytest.mark.parametrize("defect", ["missing_power", "nan", "negative", "missing_gpu", "duplicate_gpu", "foreign", "identity", "partition", "nonmonotonic", "clock_jump"])
def test_invalid_telemetry_withholds_measurement_metrics(defect):
    role, run, samples = data()
    sample = samples[9]
    if defect in ("missing_power", "nan", "negative"):
        sample["gpus"][0]["power_watts"] = {"missing_power": None, "nan": float("nan"), "negative": -1}[defect]
    elif defect == "missing_gpu":
        sample["gpus"].pop()
    elif defect == "duplicate_gpu":
        sample["gpus"][1]["uuid"] = "a"
    elif defect == "foreign":
        sample["unowned_compute_apps"] = [sample["compute_apps"][0]]
    elif defect == "identity":
        sample["owned_compute_apps"][0]["process_identity"]["pgid"] = 99
    elif defect == "partition":
        sample["owned_compute_apps"].pop()
    elif defect == "nonmonotonic":
        sample["monotonic_seconds"] = 6
    else:
        sample["at"] = utc(88)
    result = analyze(role, run, samples)
    assert result["phases"]["measurement"]["valid"] is False
    assert result["phases"]["measurement"]["aggregate"] is None
    assert result["phases"]["measurement"]["per_gpu"] is None
    assert result["phases"]["measurement"]["invalid_reasons"]


@pytest.mark.parametrize("defect", ["gap", "end_missing", "empty", "no_owned"])
def test_missing_coverage_and_ownership_do_not_extrapolate(defect):
    role, run, samples = data()
    if defect == "gap":
        samples = [sample for sample in samples if sample["monotonic_seconds"] not in (7, 8, 9)]
    elif defect == "end_missing":
        samples = samples[:11]  # Last sample t=9, request terminal t=9.5.
    elif defect == "empty":
        samples = []
    else:
        for sample in samples:
            sample["compute_apps"] = sample["owned_compute_apps"] = []
    measured = analyze(role, run, samples)["phases"]["measurement"]
    assert measured["valid"] is False
    assert measured["aggregate"] is None


def test_missing_startup_bracket_does_not_discard_valid_generation():
    role, run, samples = data()
    result = analyze(role, run, samples[2:])
    assert result["status"] == "partial"
    assert result["phases"]["startup"]["valid"] is False
    assert result["phases"]["measurement"]["aggregate"]["energy_j"] == 450


def test_invalid_media_energy_counts_but_not_invalid_clip_denominator():
    role, run, samples = data()
    run["records"].append(record("bad-media", "measurement", 10, 11, valid=False))
    measured = analyze(role, run, samples)["phases"]["measurement"]
    assert measured["valid"] is True
    assert (measured["attempted"], measured["completed"], measured["valid_clips"]) == (2, 2, 1)
    assert measured["aggregate"]["energy_j"] == 637.5
    assert measured["aggregate"]["joules_per_valid_clip"] == 637.5
    run["records"][1]["media"]["valid"] = False
    assert analyze(role, run, samples)["phases"]["measurement"]["aggregate"]["joules_per_valid_clip"] is None


def test_legacy_journal_mapping_is_bounded_and_must_agree_with_record():
    role, run, samples = data()
    item = run["records"][1]
    item.pop("timing_window")
    events = [{"event": "attempt_started", "at": utc(6.39), "slot_id": "measured"},
              {"event": "attempt_finished", "at": utc(9.75), "record": deepcopy(item)}]
    window = analyze(role, run, samples, events)["windows"][2]
    assert window["valid"] is True
    assert window["start_monotonic_seconds"] == pytest.approx(6.5)
    assert window["timing_uncertainty_seconds"] == pytest.approx(0.11)
    assert window["aggregate"]["energy_j"] == pytest.approx(450)
    events[0]["at"] = utc(6.0)
    assert analyze(role, run, samples, events)["phases"]["measurement"]["aggregate"] is None
    events[0]["at"] = utc(6.39)
    events[1]["record"]["latency_seconds"] += 1
    assert analyze(role, run, samples, events)["phases"]["measurement"]["aggregate"] is None


@pytest.mark.parametrize("defect", ["reversed", "duration", "overlap", "missing_terminal"])
def test_malformed_generation_windows_withhold_power(defect):
    role, run, samples = data()
    item = run["records"][1]
    if defect == "reversed":
        item["timing_window"]["end_monotonic_seconds"] = 6
    elif defect == "duration":
        item["submit_to_terminal_seconds"] = 2
    elif defect == "overlap":
        run["records"][1] = record("measured", "measurement", 4, 7)
    else:
        item["submit_to_terminal_seconds"] = None
    assert analyze(role, run, samples)["phases"]["measurement"]["aggregate"] is None


@pytest.mark.parametrize("watts", [1e308, 6e307])
def test_finite_inputs_cannot_publish_overflowed_power_or_energy(watts):
    role, run, samples = data()
    for sample in samples:
        for device in sample["gpus"]:
            device["power_watts"] = watts
    result = analyze(role, run, samples)
    assert result["phases"]["measurement"]["aggregate"] is None
    assert result["phases"]["measurement"]["valid"] is False
    json.dumps(result, allow_nan=False)


def test_phase_sum_overflow_is_withheld_even_if_each_request_integrates():
    role, run, samples = data()
    run["records"] = [record("one", "measurement", 6, 7), record("two", "measurement", 8, 9)]
    for sample in samples:
        for device in sample["gpus"]:
            device["power_watts"] = 6e307
    result = analyze(role, run, samples)
    assert all(window["valid"] for window in result["windows"] if window["phase"] == "measurement")
    assert result["phases"]["measurement"]["aggregate"] is None
    assert "nonfinite_phase_power_integration" in result["phases"]["measurement"]["invalid_reasons"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("query", [None, {"start_monotonic_seconds": 8, "end_monotonic_seconds": 7, "start_utc": utc(8)},
                                  {"start_monotonic_seconds": 7, "end_monotonic_seconds": 9, "start_utc": utc(7)}])
def test_malformed_timing_evidence_cannot_publish_power(query):
    role, run, samples = data()
    if query is None:
        run["records"][1].pop("timing_window")
        events = [{"event": "attempt_finished", "record": None}]
    else:
        samples[9]["power_query"] = query
        events = []
    result = analyze(role, run, samples, events)
    assert result["phases"]["measurement"]["aggregate"] is None


def test_serving_integrates_overlapping_gpu_work_once():
    from evaluator.mvp_serving import settings
    role, run, samples = data()
    run['records'].append(record('overlap', 'measurement', 7, 9.5))
    for row in run['records']:
        row['timing_window']['transport_end_monotonic_seconds'] = row['timing_window']['end_monotonic_seconds']
    run['configuration'] = {'serving': settings(2)}
    run['measurement'] = {'concurrency': 2, 'boundary': 'submit_to_downloaded_media',
                          'start_monotonic_seconds': 6.5, 'end_monotonic_seconds': 9.75, 'wall_seconds': 3.25}
    measured = analyze(role, run, samples)['phases']['measurement']
    assert measured['valid'] is True
    assert measured['window_count'] == 1
    assert measured['valid_clips'] == 2
    assert measured['duration_seconds'] == 3
    assert measured['aggregate']['energy_j'] == 450
    assert measured['aggregate']['joules_per_valid_clip'] == 225
    run['records'][-1]['submit_to_terminal_seconds'] = None
    assert analyze(role, run, samples)['phases']['measurement']['aggregate'] is None
