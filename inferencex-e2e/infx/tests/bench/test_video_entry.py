"""Container-side video command input and service discovery boundaries."""

import pytest

from infx.bench import video
from infx.bench.__main__ import main as bench_main
from infx.bench.env import InputError


def test_video_command_rejects_an_infinite_request_window():
    with pytest.raises(InputError):
        video.parse_args(["client", "--request-seconds", "inf"])


def test_client_uses_discovered_service_address(monkeypatch):
    monkeypatch.setenv("VIDEO_SERVER_SERVICE", "my-video-server")
    monkeypatch.setenv("VIDEO_SERVER_PORT", "30301")
    monkeypatch.setenv("SRT_SERVICE_MY_VIDEO_SERVER_IPS", "10.0.0.3,10.0.0.4")
    assert video.endpoint() == "http://10.0.0.3:30301"
    monkeypatch.setenv("SRT_SERVICE_MY_VIDEO_SERVER_IPS", "2001:db8::3")
    assert video.endpoint() == "http://[2001:db8::3]:30301"


def test_client_rejects_invalid_service_address(monkeypatch):
    monkeypatch.setenv("VIDEO_SERVER_SERVICE", "my-video-server")
    monkeypatch.setenv("VIDEO_SERVER_PORT", "30301")
    monkeypatch.setenv("SRT_SERVICE_MY_VIDEO_SERVER_IPS", "/invalid/host")
    with pytest.raises(InputError):
        video.endpoint()


def test_entry_reports_missing_bound_inputs_without_running_a_client(monkeypatch, capsys):
    monkeypatch.delenv("VIDEO_SPEC_PATH", raising=False)
    monkeypatch.delenv("VIDEO_RESULT_DIR", raising=False)
    assert bench_main(["video", "client", "--request-seconds", "25"]) == 1
    error = capsys.readouterr().err
    assert "VIDEO_SPEC_PATH" in error and "VIDEO_RESULT_DIR" in error


def test_native_server_requires_bound_environment_metadata(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("VIDEO_SPEC_PATH", str(tmp_path / "workload.json"))
    monkeypatch.setenv("VIDEO_RESULT_DIR", str(tmp_path / "output"))
    monkeypatch.delenv("VIDEO_RECIPE_ENV_KEYS", raising=False)
    assert bench_main(["video", "serve", "--model-view", "snapshot", "--", "engine"]) == 1
    assert "VIDEO_RECIPE_ENV_KEYS" in capsys.readouterr().err
