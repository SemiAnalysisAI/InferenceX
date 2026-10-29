"""Container-side video command input and service discovery boundaries."""

import pytest

from infx.bench import video
from infx.bench.__main__ import main as bench_main
from infx.bench.env import InputError


def test_server_requires_explicit_model_view_and_full_engine_command():
    command = ["/usr/bin/python3", "-m", "evaluator.srt_selected", "serve", "--flag", "value"]
    parsed = video.parse_args(["serve", "--model-view", "snapshot", "--", *command])
    assert parsed.model_view == "snapshot"
    assert parsed.command == command


@pytest.mark.parametrize("args", [
    ["serve", "--model-view", "snapshot"],
    ["serve", "--model-view", "snapshot", "--"],
    ["client", "--request-seconds", "0"],
    ["client", "--request-seconds", "inf"],
])
def test_video_command_rejects_missing_engine_or_invalid_window(args):
    with pytest.raises(InputError):
        video.parse_args(args)


def test_client_window_is_explicit():
    assert video.parse_args(["client", "--request-seconds", "73"]).request_seconds == 73


def test_client_uses_discovered_service_address(monkeypatch):
    monkeypatch.setenv("H3_SERVER_SERVICE", "my-h3-server")
    monkeypatch.setenv("H3_SERVER_PORT", "30301")
    monkeypatch.setenv("SRT_SERVICE_MY_H3_SERVER_IPS", "10.0.0.3,10.0.0.4")
    assert video.endpoint() == "http://10.0.0.3:30301"
    monkeypatch.setenv("SRT_SERVICE_MY_H3_SERVER_IPS", "2001:db8::3")
    assert video.endpoint() == "http://[2001:db8::3]:30301"


@pytest.mark.parametrize(("key", "value"), [
    ("SRT_SERVICE_MY_H3_SERVER_IPS", "/invalid/host"),
    ("H3_SERVER_PORT", "0"),
    ("H3_SERVER_PORT", "65536"),
])
def test_client_rejects_invalid_service_address(monkeypatch, key, value):
    monkeypatch.setenv("H3_SERVER_SERVICE", "my-h3-server")
    monkeypatch.setenv("H3_SERVER_PORT", "30301")
    monkeypatch.setenv("SRT_SERVICE_MY_H3_SERVER_IPS", "10.0.0.3")
    monkeypatch.setenv(key, value)
    with pytest.raises(InputError):
        video.endpoint()


def test_entry_reports_missing_bound_inputs_without_running_a_client(monkeypatch, capsys):
    monkeypatch.delenv("H3_SPEC_PATH", raising=False)
    monkeypatch.delenv("H3_RESULT_DIR", raising=False)
    assert bench_main(["video", "client", "--request-seconds", "25"]) == 1
    error = capsys.readouterr().err
    assert "H3_SPEC_PATH" in error and "H3_RESULT_DIR" in error


def test_native_server_requires_bound_environment_metadata(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("H3_SPEC_PATH", str(tmp_path / "workload.json"))
    monkeypatch.setenv("H3_RESULT_DIR", str(tmp_path / "output"))
    monkeypatch.delenv("H3_RECIPE_ENV_KEYS", raising=False)
    assert bench_main(["video", "serve", "--model-view", "snapshot", "--", "engine"]) == 1
    assert "H3_RECIPE_ENV_KEYS" in capsys.readouterr().err
