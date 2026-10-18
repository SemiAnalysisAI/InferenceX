"""CPU event/transport checks; these do not establish H3 timing overhead."""

from copy import deepcopy
import hashlib
import importlib.util
import json
from types import SimpleNamespace

import pytest

from evaluator import mvp_runtime_timing as timing
from evaluator.mvp_serving import summarize
from test_mvp_serving import sample_run
from test_mvp_runner import execute, fixture_server, mocked_media, plan  # noqa: F401


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    path = tmp_path / "server-timings.jsonl"
    path.touch()
    monkeypatch.setenv("VGBENCH_SERVER_TIMING_PATH", str(path))
    monkeypatch.setenv("VGBENCH_SERVER_TIMING_INSTANCE", "cpu-fixture-instance")
    monkeypatch.setattr(timing, "_clock_id", lambda: "linux:cpu-fixture:time-namespace:CLOCK_MONOTONIC")
    return path


def complete_request(request_id):
    timing.emit(request_id, "http_received")
    timing.emit(request_id, "http_accepted")
    with timing.forward([SimpleNamespace(request_id=request_id, num_outputs_per_prompt=1)], replica_id=0, leader=True):
        pass
    timing.emit(request_id, "media_ready")


def test_real_event_writer_correlates_request_and_derives_stage_windows(ledger, monkeypatch):
    stamps = iter([1, 2, 7, 17, 19])
    monkeypatch.setattr(timing.time, "monotonic_ns", lambda: next(stamps) * 1_000_000_000)
    complete_request("video-1")
    result = timing.collect("video-1")
    assert result["status"] == "complete"
    assert [result[name] for name in timing.DURATIONS] == [1, 5, 10, 2, 18]
    assert result["observed_batch_size"] == 1 and result["replica_id"] == 0
    assert timing.collect("different-video") is None
    assert len(timing.read_events(ledger)) == 5


def test_interleaved_requests_and_missing_stage_do_not_create_queue_time(ledger):
    timing.emit("a", "http_received")
    complete_request("b")
    timing.emit("a", "http_accepted")
    partial = timing.collect("a")
    assert partial["status"] == "partial"
    assert partial["queue_delay_seconds"] is None
    assert partial["server_ready_latency_seconds"] is None
    assert partial["observed_batch_size"] is None
    assert timing.collect("b")["status"] == "complete"


@pytest.mark.parametrize("defect", ["duplicate", "clock", "reverse", "instance", "batch"])
def test_corrupt_or_incompatible_events_fail_closed(ledger, defect):
    complete_request("a")
    rows = timing.read_events(ledger)
    if defect == "duplicate":
        rows.append(rows[0])
    elif defect == "clock":
        rows[2]["clock_id"] = "another-host"
    elif defect == "reverse":
        rows[2]["monotonic_ns"] = rows[0]["monotonic_ns"] - 1
    elif defect == "instance":
        rows[2]["instance_id"] = "another-server"
    else:
        rows[2]["observed_batch_size"] = 2
    with pytest.raises(ValueError):
        timing.derive(rows, "a", "cpu-fixture-instance")


def test_worker_followers_do_not_duplicate_events_and_failure_has_no_ready_time(ledger):
    req = SimpleNamespace(request_id="a", num_outputs_per_prompt=1)
    with timing.forward([req], replica_id=0, leader=False):
        pass
    assert timing.read_events(ledger) == []
    with pytest.raises(RuntimeError, match="generation failed"):
        with timing.forward([req], replica_id=0, leader=True):
            raise RuntimeError("generation failed")
    partial = timing.collect("a")
    assert partial["execution_seconds"] >= 0
    assert partial["server_ready_latency_seconds"] is None


def test_http_client_retains_timings_in_original_record_and_journal(plan, mocked_media, fixture_server, ledger, tmp_path):
    ordinal = 0
    def record_fixture_timing():
        nonlocal ordinal
        ordinal += 1
        complete_request(f"fixture-{ordinal}")
    endpoint, _ = fixture_server(before_submit=record_fixture_timing)
    output = tmp_path / "client"
    result = execute(plan, output, endpoint)
    assert all(record["server_timings"]["request_id"] == record["job_id"] for record in result["records"])
    records = [row["record"] for row in map(json.loads, (output / "events.jsonl").read_text().splitlines())
               if row["event"] == "attempt_finished"]
    assert [row["server_timings"] for row in records] == [row["server_timings"] for row in result["records"]]


def test_serving_summary_requires_full_valid_population_for_percentiles():
    run = sample_run()
    historical = summarize(run)
    assert historical["queue_delay_seconds"] is None
    assert "server_execution_seconds" not in historical
    for record in run["records"][:10]:
        record["server_timings"] = {"status": "complete", "queue_delay_seconds": 2,
            "execution_seconds": 5, "server_ready_latency_seconds": 8,
            "prequeue_seconds": .5, "postprocess_seconds": .5, "observed_batch_size": 1, "replica_id": 0}
    observed = summarize(run)
    assert observed["queue_delay_seconds"]["p90"] == 2
    assert observed["server_execution_seconds"]["sample_count"] == 10
    assert observed["observed_batch_sizes"] == [1] * 10
    run["records"][0]["server_timings"] = None
    incomplete = summarize(run)
    assert incomplete["queue_delay_seconds"]["missing_count"] == 1
    assert incomplete["queue_delay_seconds"]["p90"] is None
    assert incomplete["observed_batch_sizes"] is None


def test_offline_verification_binds_ledger_hash_instance_and_request(ledger, tmp_path):
    complete_request("a")
    run = {"records": [{"job_id": "a", "status": "succeeded", "server_timings": timing.collect("a")}]}
    role = {"process_identity": {"launch_nonce": "cpu-fixture-instance"}, "server_timing_evidence": {
        **timing.identity(), "path": ledger.name, "instance_id": "cpu-fixture-instance",
        "sha256": hashlib.sha256(ledger.read_bytes()).hexdigest()}}
    timing.verify_evidence(tmp_path, role, run, required=True)
    changed = deepcopy(run)
    changed["records"][0]["server_timings"]["queue_delay_seconds"] = 99
    with pytest.raises(ValueError, match="raw events"):
        timing.verify_evidence(tmp_path, role, changed, required=True)
    ledger.write_text(ledger.read_text() + "{}\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        timing.verify_evidence(tmp_path, role, run, required=True)


def test_staged_runtime_helper_can_resolve_its_pinned_patch(tmp_path):
    import ci
    destination = tmp_path / "staged"
    ci.stage_package(timing.PATCH.parents[1], destination)
    spec = importlib.util.spec_from_file_location("staged_timing", destination / "evaluator/mvp_runtime_timing.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.identity() == timing.identity()
