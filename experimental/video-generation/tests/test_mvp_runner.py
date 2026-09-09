"""Offline transport tests using a local HTTP fixture, never an H3 model.

The fixture returns deliberately non-video bytes and a mocked media analyzer.
Actual decoding/corruption tests belong to the media engine's test suite.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import threading
import time
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluator import mvp_runner


MODEL_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
FIXTURE_BYTES = b"LOCAL_HTTP_FIXTURE_NOT_H3_VIDEO"


@pytest.fixture
def plan():
    return {
        "plan_id": "offline-http-fixture-plan",
        "model_id": "MiniMaxAI/MiniMax-H3",
        "model_revision": MODEL_REVISION,
        "generation": {
            "duration_seconds": 4,
            "aspect_ratio": "16:9",
            "width": 1344,
            "height": 768,
            "frame_count": 107,
            "fps": 24,
            "audio_sample_rate_hz": 32000,
            "audio_channels": 2,
            "num_inference_steps": 50,
            "flow_shift": 12.0,
            "audio_flow_shift": 3.0,
        },
        "cases": [
            {"case_id": "fixture-a", "prompt": "LOCAL TEST: motion and sound.", "seed": 42,
             "requires_motion": True, "requires_sound": True},
            {"case_id": "fixture-b", "prompt": "LOCAL TEST: a quiet still scene.", "seed": 91,
             "requires_motion": False, "requires_sound": False},
        ],
        "repetitions": 2,
        "warmup_runs": 1,
    }


@pytest.fixture
def mocked_media(monkeypatch):
    calls = []
    state = {"valid": True}

    def analyze(path, expected):
        assert path.read_bytes() == FIXTURE_BYTES
        calls.append((path, expected))
        return {
            "decode_ok": True,
            "video": {"width": expected["width"], "height": expected["height"],
                      "frame_count": expected["frame_count"], "fps": expected["fps"]},
            "audio": {"sample_rate_hz": expected["audio_sample_rate_hz"], "channels": expected["audio_channels"]},
            "checks": {"fixture_analysis_only": state["valid"]},
            "valid": state["valid"],
            "metrics": {},
        }

    monkeypatch.setitem(sys.modules, "evaluator.mvp_media", SimpleNamespace(
        analyze_media=analyze, IMPLEMENTATION_VERSION="local-test-fixture", __file__=__file__,
    ))
    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(
        __version__="local-test-fixture", library_versions={"libavcodec": (62, 1, 2)},
    ))
    monkeypatch.setitem(sys.modules, "numpy", SimpleNamespace(__version__="local-test-fixture"))
    monkeypatch.setattr(mvp_runner, "POLL_INTERVAL_SECONDS", 0.001)
    return calls, state


@pytest.fixture
def fixture_server():
    servers = []

    def start(*, mode="success", before_submit=None):
        state = {"requests": [], "posts": [], "polls": {}, "auth": [], "mode": mode}
        submission_lock = threading.Lock()

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_):
                pass

            def send(self, status, body, *, headers=None, announced_length=None):
                self.send_response(status)
                self.send_header("Content-Length", str(len(body) if announced_length is None else announced_length))
                self.send_header("Connection", "close")
                for key, value in (headers or {}).items():
                    self.send_header(key, value)
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                self.close_connection = True

            def send_json(self, status, body):
                self.send(status, json.dumps(body).encode(), headers={"Content-Type": "application/json"})

            def do_POST(self):
                state["requests"].append(("POST", self.path))
                state["auth"].append(self.headers.get("Authorization"))
                if before_submit:
                    before_submit()
                body = self.rfile.read(int(self.headers["Content-Length"]))
                content_type = self.headers.get("Content-Type", "")
                if content_type.startswith("multipart/form-data"):
                    message = BytesParser(policy=policy.default).parsebytes(
                        b"Content-Type: " + content_type.encode() + b"\r\nMIME-Version: 1.0\r\n\r\n" + body
                    )
                    payload = {part.get_param("name", header="content-disposition"): part.get_payload(decode=True).decode()
                               for part in message.iter_parts()}
                else:
                    payload = json.loads(body)
                with submission_lock:
                    state["posts"].append(payload)
                    ordinal = len(state["posts"])
                if state["mode"] == "http_error":
                    self.send_json(500, {"error": "do-not-log-this-server-secret"})
                elif state["mode"] == "redirect":
                    self.send(307, b"", headers={"Location": "/stolen-credentials"})
                elif state["mode"] == "unsafe_id":
                    self.send_json(200, {"id": "../../external?token=secret", "status": "queued"})
                else:
                    self.send_json(200, {"id": f"fixture-{ordinal}", "status": "queued"})

            def do_GET(self):
                state["requests"].append(("GET", self.path))
                state["auth"].append(self.headers.get("Authorization"))
                if self.path.endswith("/content"):
                    if state["mode"] == "oversized":
                        self.send(200, FIXTURE_BYTES, announced_length=mvp_runner.MAX_MEDIA_BYTES + 1)
                    elif state["mode"] == "truncated":
                        self.send(200, FIXTURE_BYTES, announced_length=len(FIXTURE_BYTES) + 10)
                    else:
                        self.send(200, FIXTURE_BYTES, headers={"Content-Type": "video/mp4"})
                else:
                    state["polls"][self.path] = state["polls"].get(self.path, 0) + 1
                    status = "queued" if state["mode"] == "timeout" else (
                        "failed" if state["mode"] == "job_failed" else "completed"
                    )
                    self.send_json(200, {"id": self.path.rsplit("/", 1)[-1], "status": status,
                                         "error": "do-not-log-this-server-secret"})

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.005}, daemon=True)
        thread.start()
        servers.append((server, thread))
        return f"http://127.0.0.1:{server.server_port}", state

    yield start
    for server, thread in servers:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


def execute(plan, output_dir, endpoint, **kwargs):
    return mvp_runner.run_plan(
        plan, output_dir, endpoint=endpoint,
        runtime_revision="253020450290328e9deb307eece1e402fa17f35e",
        hardware_label="LOCAL HTTP FIXTURE; no GPU and no H3 execution",
        model_revision=MODEL_REVISION,
        **kwargs,
    )


def test_preview_is_network_free_and_reproduces_frozen_slots(plan, monkeypatch):
    monkeypatch.setattr(mvp_runner, "_open_response", lambda *args, **kwargs: pytest.fail("preview made a network request"))
    preview = mvp_runner.preview_plan(plan)
    assert preview["evidence_kind"] == "request_preview_no_generation"
    assert preview["total_requests"] == 5
    assert preview["measurement_count"] == 4
    assert [slot["slot_id"] for slot in preview["slots"]] == [
        "warmup-001", "measurement-r001-c001", "measurement-r001-c002",
        "measurement-r002-c001", "measurement-r002-c002",
    ]
    assert preview["slots"][1]["request"]["seed"] == 42
    assert preview["slots"][3]["request"]["seed"] == 42
    assert "fps" not in preview["slots"][0]["request"]
    assert "num_frames" not in preview["slots"][0]["request"]
    assert preview["plan_sha256"] == hashlib.sha256(mvp_runner.canonical_json_bytes(plan)).hexdigest()


@pytest.mark.parametrize("runtime", ["sglang", "vllm-omni"])
@pytest.mark.parametrize(("duration", "frames"), [(4, 107), (8, 192)])
def test_local_http_fixture_lifecycle_and_identity(plan, tmp_path, mocked_media, fixture_server, runtime, duration, frames):
    plan["generation"].update(duration_seconds=duration, frame_count=frames)
    output = tmp_path / runtime
    intents_seen = []

    def before_submit():
        events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
        intents_seen.append(events[-1]["event"])
        assert events[-1]["event"] == "attempt_started"

    endpoint, server = fixture_server(before_submit=before_submit)
    run = execute(plan, output, endpoint, runtime=runtime)
    assert run["status"] == "complete"
    assert run["evidence_kind"] == "operator_endpoint"
    assert len(run["records"]) == 5
    assert len(server["posts"]) == 5
    assert len(intents_seen) == 5
    assert run["summary"]["scheduled"] == run["summary"]["completed"] == run["summary"]["valid"] == 4
    assert run["summary"]["failed"] == 0
    assert run["summary"]["technical_success_rate"] == 1
    assert math.isclose(run["summary"]["valid_clips_per_second"], 4 / run["measurement"]["wall_seconds"])
    assert all(record["latency_seconds"] > 0 for record in run["records"])
    for record in run["records"]:
        assert 0 < record["submit_to_terminal_seconds"] <= record["submit_to_media_seconds"] <= record["latency_seconds"]
        assert 0 <= record["media_validation_seconds"] <= record["latency_seconds"]
    assert run["summary"]["latency_samples"] == 4
    assert run["summary"]["latency_sample_stddev_seconds"] is not None
    assert run["summary"]["submit_to_terminal_median_seconds"] > 0
    assert run["configuration"]["identity_verification"] == "operator_declared"
    assert "not remotely verified" in run["configuration"]["server_identity_caveat"]
    assert "Mock-server tests are not H3 evidence" in run["evidence_caveat"]
    assert run["configuration"]["hardware_label"].startswith("LOCAL HTTP FIXTURE")
    assert run["measurement"]["warmup_qualified"] is True
    config = copy.deepcopy(run["configuration"])
    digest = config.pop("configuration_sha256")
    assert hashlib.sha256(mvp_runner.canonical_json_bytes(config)).hexdigest() == digest
    assert (output / "plan.json").read_bytes() == mvp_runner.canonical_json_bytes(plan)
    assert json.loads((output / "run.json").read_text()) == run
    for record in run["records"]:
        assert (output / record["artifact_path"]).read_bytes() == FIXTURE_BYTES
        assert record["sha256"] == hashlib.sha256(FIXTURE_BYTES).hexdigest()
        assert math.isclose(record["expected_media"]["duration_seconds"], frames / 24)
        assert record["expected_media"]["audio_required"] is True
    calls, _ = mocked_media
    assert len(calls) == 5
    assert all(expected["timeout_seconds"] > 0 for _, expected in calls)
    first = server["posts"][0]
    if runtime == "sglang":
        assert first["target"] == {"duration_seconds": duration, "aspect_ratio": "16:9", "short_edge": 768}
        assert "fps" not in first and "num_frames" not in first
        assert first["audio_flow_shift"] == 3
    else:
        assert first["width"] == "1344"
        assert first["num_frames"] == str(frames)
        assert json.loads(first["extra_params"])["audio_flow_shift"] == 3


def test_http_failure_is_durable_never_retried_and_aborts_unknown_jobs(plan, tmp_path, mocked_media, fixture_server):
    plan["warmup_runs"] = 0
    endpoint, state = fixture_server(mode="http_error")
    output = tmp_path / "failed-http-fixture"
    run = execute(plan, output, endpoint)
    assert run["status"] == "failed"
    assert len(state["posts"]) == 1
    assert run["summary"]["failed"] == 4
    assert run["summary"]["failed_attempts"] == 1
    assert run["summary"]["not_started"] == 3
    assert run["summary"]["technical_success_rate"] == 0
    assert run["summary"]["latency_median_seconds"] is None
    assert run["summary"]["submit_to_terminal_median_seconds"] is None
    assert run["summary"]["latency_samples"] == 0
    assert run["records"][0]["submit_to_terminal_seconds"] is None
    assert run["records"][1]["submit_to_media_seconds"] is None
    assert "HTTP 500" in run["records"][0]["error"]
    assert all(record["error"] == "not_started_after_uncertain_remote_completion" for record in run["records"][1:])
    assert "do-not-log-this-server-secret" not in (output / "events.jsonl").read_text()
    assert "do-not-log-this-server-secret" not in (output / "run.json").read_text()


def test_media_evaluator_fingerprint_is_covered_by_configuration_hash(plan, tmp_path, mocked_media, fixture_server):
    plan["warmup_runs"] = 0
    plan["repetitions"] = 1
    plan["cases"] = plan["cases"][:1]
    endpoint, _ = fixture_server()
    run = execute(plan, tmp_path / "evaluator-fingerprint-fixture", endpoint)
    configuration = copy.deepcopy(run["configuration"])
    assert configuration["media_evaluator"] == {
        "implementation_version": "local-test-fixture",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "pyav_version": "local-test-fixture",
        "numpy_version": "local-test-fixture",
        "ffmpeg_libraries": {"libavcodec": "62.1.2"},
    }
    original_digest = configuration.pop("configuration_sha256")
    assert hashlib.sha256(mvp_runner.canonical_json_bytes(configuration)).hexdigest() == original_digest
    configuration["media_evaluator"]["ffmpeg_libraries"]["libavcodec"] = "62.2.0"
    assert hashlib.sha256(mvp_runner.canonical_json_bytes(configuration)).hexdigest() != original_digest


def test_known_failed_job_does_not_remove_other_scheduled_attempts(plan, tmp_path, mocked_media, fixture_server):
    plan["warmup_runs"] = 0
    endpoint, state = fixture_server(mode="job_failed")
    run = execute(plan, tmp_path / "failed-jobs-fixture", endpoint)
    assert len(state["posts"]) == 4
    assert run["summary"]["failed_attempts"] == 4
    assert run["summary"]["not_started"] == 0
    assert run["abort_reason"] is None


@pytest.mark.parametrize("mode", ["success", "job_failed", "http_error"])
def test_request_timing_windows_preserve_observed_terminal_and_end(plan, tmp_path, mocked_media, fixture_server, mode):
    plan.update(warmup_runs=0, repetitions=1, cases=plan["cases"][:1])
    endpoint, _ = fixture_server(mode=mode)
    before = time.monotonic()
    run = execute(plan, tmp_path / mode, endpoint)
    after = time.monotonic()
    record = run["records"][0]
    timing = record["timing_window"]
    start, end = timing["start_monotonic_seconds"], timing["end_monotonic_seconds"]
    assert before <= start < end <= after
    assert math.isclose(end - start, record["latency_seconds"], rel_tol=1e-6, abs_tol=1e-9)
    if mode == "http_error":
        assert timing["terminal_monotonic_seconds"] is None
        assert record["submit_to_terminal_seconds"] is None
    else:
        terminal = timing["terminal_monotonic_seconds"]
        assert start < terminal <= end
        assert math.isclose(terminal - start, record["submit_to_terminal_seconds"], rel_tol=1e-6, abs_tol=1e-9)
    events = [json.loads(line) for line in (tmp_path / mode / "events.jsonl").read_text().splitlines()]
    assert next(event["record"] for event in events if event["event"] == "attempt_finished") == record


def test_timeout_is_bounded_and_retains_all_denominators(plan, tmp_path, mocked_media, fixture_server):
    plan["warmup_runs"] = 0
    endpoint, state = fixture_server(mode="timeout")
    start = time.monotonic()
    run = execute(plan, tmp_path / "timeout-fixture", endpoint, timeout_seconds=0.05)
    assert time.monotonic() - start < 3
    assert len(state["posts"]) == 1
    assert run["summary"]["scheduled"] == run["summary"]["failed"] == 4
    assert run["records"][0]["status"] == "failed"
    assert run["abort_reason"]
    assert run["records"][1]["attempted"] is False


def test_completed_but_invalid_media_is_not_technical_success(plan, tmp_path, mocked_media, fixture_server):
    plan["warmup_runs"] = 0
    _, media = mocked_media
    media["valid"] = False
    endpoint, state = fixture_server()
    run = execute(plan, tmp_path / "invalid-media-fixture", endpoint)
    assert len(state["posts"]) == 4
    assert run["summary"]["completed"] == 4
    assert run["summary"]["valid"] == 0
    assert run["summary"]["invalid_completed"] == 4
    assert run["summary"]["failed_attempts"] == 0
    assert run["summary"]["failed"] == 4


def test_failed_warmup_cannot_qualify_a_measured_run(plan, tmp_path, mocked_media, fixture_server):
    _, media = mocked_media
    media["valid"] = False
    endpoint, state = fixture_server()
    run = execute(plan, tmp_path / "failed-warmup-fixture", endpoint)
    assert len(state["posts"]) == 1
    assert run["measurement"]["warmup_qualified"] is False
    assert run["measurement"]["warmup_status"] == "failed"
    assert run["summary"]["scheduled"] == run["summary"]["not_started"] == 4
    assert all(record["error"] == "not_started_after_failed_warmup" for record in run["records"][1:])


@pytest.mark.parametrize("mode", ["oversized", "truncated"])
def test_download_size_and_integrity_limits(plan, tmp_path, mocked_media, fixture_server, mode):
    plan["warmup_runs"] = 0
    plan["repetitions"] = 1
    plan["cases"] = plan["cases"][:1]
    endpoint, state = fixture_server(mode=mode)
    run = execute(plan, tmp_path / mode, endpoint)
    assert run["summary"]["valid"] == 0
    assert run["records"][0]["artifact_path"] is None
    assert run["records"][0]["sha256"] is None
    assert len(state["posts"]) == 1
    assert not mocked_media[0]


def test_refuses_overwrite_without_network_or_file_changes(plan, tmp_path, mocked_media, fixture_server):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "user-owned.txt"
    marker.write_text("preserve this file")
    endpoint, state = fixture_server()
    with pytest.raises(FileExistsError):
        execute(plan, output, endpoint)
    assert marker.read_text() == "preserve this file"
    assert list(output.iterdir()) == [marker]
    assert not state["requests"]


def test_credential_is_not_persisted_or_forwarded_on_redirect(plan, tmp_path, mocked_media, fixture_server, monkeypatch):
    monkeypatch.setenv("VGBENCH_TEST_ONLY_TOKEN", "fixture-credential-never-a-real-key")
    endpoint, state = fixture_server(mode="redirect")
    output = tmp_path / "redirect-fixture"
    run = execute(plan, output, endpoint, api_key_env="VGBENCH_TEST_ONLY_TOKEN")
    assert len(state["requests"]) == 1
    assert state["auth"] == ["Bearer fixture-credential-never-a-real-key"]
    assert run["records"][0]["status"] == "failed"
    for path in output.rglob("*"):
        if path.is_file():
            assert b"fixture-credential-never-a-real-key" not in path.read_bytes()


@pytest.mark.parametrize("endpoint", [
    "https://user:secret@example.com", "https://example.com?token=secret",
    "https://example.com#secret", "https://api.minimax.io", "file:///tmp/video",
])
def test_refuses_unsafe_or_hosted_endpoint_before_creating_output(plan, tmp_path, mocked_media, endpoint):
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError):
        execute(plan, output, endpoint)
    assert not output.exists()


def test_untrusted_job_id_cannot_change_download_destination(plan, tmp_path, mocked_media, fixture_server):
    endpoint, state = fixture_server(mode="unsafe_id")
    run = execute(plan, tmp_path / "unsafe-id-fixture", endpoint)
    assert len(state["requests"]) == 1
    assert run["records"][0]["error"] == "submission returned a missing or unsafe job identifier"
    assert all(record["artifact_path"] is None for record in run["records"])


def test_missing_controls_and_revision_drift_fail_before_output(plan, tmp_path, mocked_media):
    invalid = copy.deepcopy(plan)
    del invalid["generation"]["num_inference_steps"]
    with pytest.raises(ValueError, match="num_inference_steps"):
        mvp_runner.preview_plan(invalid)
    invalid = copy.deepcopy(plan)
    invalid["model_revision"] = "1" * 40
    output = tmp_path / "not-created"
    with pytest.raises(ValueError, match="does not match"):
        execute(invalid, output, "http://127.0.0.1:1")
    assert not output.exists()


@pytest.mark.parametrize("change", [
    {"num_frames": 107}, {"frame_count": 96}, {"width": 1366}, {"scheduler": "secretly-changed"},
    {"duration_seconds": 8}, {"duration_seconds": 8, "frame_count": 193},
    {"frame_count": 192}, {"duration_seconds": 6, "frame_count": 158},
])
def test_unsupported_or_inconsistent_generation_contracts_are_rejected(plan, change):
    plan["generation"].update(change)
    with pytest.raises(ValueError):
        mvp_runner.preview_plan(plan)


def test_missing_media_dependency_prevents_live_submission(plan, tmp_path, mocked_media, fixture_server, monkeypatch):
    monkeypatch.setitem(sys.modules, "av", None)
    endpoint, server = fixture_server()
    output = tmp_path / "dependency-missing"
    with pytest.raises(ImportError):
        execute(plan, output, endpoint)
    assert not output.exists()
    assert not server["requests"]


def test_dns_timeout_cannot_submit_late_request(plan, tmp_path, mocked_media, fixture_server, monkeypatch):
    plan["warmup_runs"] = 0
    endpoint, server = fixture_server()
    release_resolver = threading.Event()
    original = mvp_runner.socket.getaddrinfo

    def slow_resolution(*args, **kwargs):
        release_resolver.wait(timeout=2)
        return original(*args, **kwargs)

    monkeypatch.setattr(mvp_runner.socket, "getaddrinfo", slow_resolution)
    try:
        run = execute(plan, tmp_path / "slow-dns-fixture", endpoint, timeout_seconds=0.03)
        assert run["summary"]["failed"] == 4
        assert "timed out" in run["records"][0]["error"]
        assert not server["requests"]
    finally:
        release_resolver.set()


def test_keyboard_interrupt_retains_inflight_outcome_and_schedule(plan, tmp_path, mocked_media, fixture_server, monkeypatch):
    plan["warmup_runs"] = 0
    endpoint, server = fixture_server()

    def interrupt_analysis(*_):
        raise KeyboardInterrupt

    monkeypatch.setattr(sys.modules["evaluator.mvp_media"], "analyze_media", interrupt_analysis)
    output = tmp_path / "interrupted-fixture"
    with pytest.raises(KeyboardInterrupt):
        execute(plan, output, endpoint)
    run = json.loads((output / "run.json").read_text())
    assert len(server["posts"]) == 1
    assert run["status"] == "failed"
    assert run["summary"]["failed"] == 4
    assert run["summary"]["not_started"] == 3
    assert run["records"][0]["error"].startswith("interrupted by operator")


def test_serving_overlaps_requests_without_waiting_for_local_validation(plan, tmp_path, mocked_media, fixture_server, monkeypatch):
    plan['warmup_runs'] = 0
    pair = threading.Barrier(2)
    submitted = threading.Event()
    count = [0]
    lock = threading.Lock()

    def before_submit():
        with lock:
            count[0] += 1
            if count[0] == 4:
                submitted.set()
        pair.wait(timeout=2)

    endpoint, state = fixture_server(before_submit=before_submit)
    analyzer = sys.modules['evaluator.mvp_media'].analyze_media

    def validate_after_submissions(path, expected):
        assert submitted.wait(2), 'local validation blocked later submissions'
        return analyzer(path, expected)

    monkeypatch.setattr(sys.modules['evaluator.mvp_media'], 'analyze_media', validate_after_submissions)
    run = execute(plan, tmp_path / 'serving', endpoint, serving_concurrency=2, delivery_deadline_seconds=10)
    assert run['status'] == 'complete'
    assert run['serving']['peak_client_in_flight'] == 2
    assert run['summary']['valid'] == 4
    assert run['serving']['deadline_met_valid_clips'] == 4
    assert run['serving']['client_ready_latency_seconds']['p90'] is None
    assert run['serving']['observed_batch_sizes'] is None
    assert len({r['job_id'] for r in run['records']}) == 4
    assert [r['slot_id'] for r in run['records']] == [s['slot_id'] for s in mvp_runner._slots(plan)]
    window = run['measurement']
    assert window['boundary'] == 'submit_to_downloaded_media'
    assert window['end_monotonic_seconds'] == max(r['timing_window']['transport_end_monotonic_seconds'] for r in run['records'])
    assert run['summary']['valid_clips_per_second'] == 4 / window['wall_seconds']
    for record in run['records']:
        assert 0 <= record['submit_to_accepted_seconds'] <= record['submit_to_terminal_seconds'] <= record['submit_to_media_seconds'] <= record['latency_seconds']
        assert record['server_timings'] is None
    assert len(state['posts']) == 4


def test_serving_unknown_remote_completion_stops_queued_requests(plan, tmp_path, mocked_media, fixture_server):
    plan['warmup_runs'] = 0
    pair = threading.Barrier(2)
    endpoint, state = fixture_server(mode='timeout', before_submit=lambda: pair.wait(timeout=2))
    run = execute(plan, tmp_path / 'timeout', endpoint, serving_concurrency=2, timeout_seconds=.1)
    assert len(state['posts']) == 2
    assert run['summary']['scheduled'] == run['summary']['failed'] == 4
    assert run['summary']['not_started'] == 2
    assert run['serving']['outcomes']['timed_out'] == 2, run['records']
    assert run['serving']['outcomes']['not_started'] == 2
    assert run['serving']['peak_client_in_flight'] == 2
    assert run['serving']['client_ready_latency_seconds']['sample_count'] == 0


def test_serving_failed_warmup_never_submits_measured_requests(plan, tmp_path, mocked_media, fixture_server):
    mocked_media[1]['valid'] = False
    endpoint, state = fixture_server()
    run = execute(plan, tmp_path / 'warmup-failure', endpoint, serving_concurrency=2)
    assert len(state['posts']) == 1
    assert run['summary']['not_started'] == 4
    assert run['measurement']['warmup_status'] == 'failed'
    assert run['serving']['peak_client_in_flight'] == 0


@pytest.mark.parametrize('concurrency,deadline', [(0, None), (33, None), (True, None), (1, float('nan')), (None, 10)])
def test_invalid_serving_settings_fail_before_output(plan, tmp_path, mocked_media, concurrency, deadline):
    destination = tmp_path / 'invalid-serving'
    with pytest.raises(ValueError):
        execute(plan, destination, 'http://localhost:9', serving_concurrency=concurrency, delivery_deadline_seconds=deadline)
    assert not destination.exists()


@pytest.mark.parametrize("exception", [ConnectionError, AttributeError, ValueError])
def test_watchdog_deadline_wins_over_socket_teardown_errors(plan, tmp_path, mocked_media, monkeypatch, exception):
    plan['warmup_runs'] = 0
    def stopped_transfer(*args, deadline, **kwargs):
        time.sleep(max(0, deadline-time.monotonic()) + .001)
        raise exception('CPU fixture: watchdog closed transport')
    monkeypatch.setattr(mvp_runner, '_json_request', stopped_transfer)
    run = execute(plan, tmp_path / 'deadline', 'http://127.0.0.1:1', serving_concurrency=1, timeout_seconds=.01)
    assert run['serving']['outcomes']['timed_out'] == 1
    assert run['summary']['not_started'] == 3
