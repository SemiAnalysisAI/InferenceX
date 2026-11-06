"""Server readiness against a local HTTP server: the ``wait`` command and the OpenAI chat route."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from infx.bench import server

REPO_ROOT = Path(__file__).resolve().parents[3]


def frontend(models: list[str] | None, chat: int, health: int):
    """``models`` listed at /v1/models (None: 404); a bare chat GET answers ``chat``."""

    def respond(_method: str, path: str) -> tuple[int, object]:
        if path == "/v1/models":
            return (404, {}) if models is None else (200, {"data": [{"id": m} for m in models]})
        return {"/v1/chat/completions": chat, "/health": health}.get(path, 404), {}

    return respond


def test_wait_streams_new_server_log_until_health_answers(http_server, tmp_path, capsys):
    log = tmp_path / "server.log"
    log.write_text("loading\n")
    statuses = iter([503, 200])

    def respond(method: str, path: str) -> tuple[int, object]:
        with log.open("a") as out:
            out.write(f"{method} {path}\n")
        return next(statuses), {}

    server.wait_ready(f"{http_server(respond)}/health", pid=os.getpid(), log=log, poll_s=0.01)

    assert capsys.readouterr().out == "loading\nGET /health\n"


@pytest.mark.parametrize(
    ("before", "mode", "written", "out"),
    [(b"loading\n", "wb", b"ok\n", "loading\nok\n"), (b"caf\xc3", "ab", b"\xa9\n", "caf\u00e9\n")],
    ids=["truncated", "split-utf8"],
)
def test_wait_follows_a_truncated_log_and_split_characters(
    http_server, tmp_path, capsys, before, mode, written, out
):
    log = tmp_path / "server.log"
    log.write_bytes(before)
    statuses = iter([503, 200])

    def respond(_method: str, _path: str) -> tuple[int, object]:
        status = next(statuses)
        if status == 503:
            with log.open(mode) as stream:
                stream.write(written)
        return status, {}

    server.wait_ready(f"{http_server(respond)}/health", log=log, poll_s=0.01)

    assert capsys.readouterr().out == out


def test_wait_restarts_a_log_replaced_by_a_longer_file(http_server, tmp_path, capsys):
    log = tmp_path / "server.log"
    log.write_bytes(b"old\n")
    statuses = iter([503, 200])

    def respond(_method: str, _path: str) -> tuple[int, object]:
        status = next(statuses)
        if status == 503:
            fresh = tmp_path / "server.log.new"
            fresh.write_bytes(b"restarted server\n")
            fresh.replace(log)
        return status, {}

    server.wait_ready(f"{http_server(respond)}/health", log=log, poll_s=0.01)

    assert capsys.readouterr().out == "old\nrestarted server\n"


def test_wait_fails_once_the_server_process_is_gone():
    dead = subprocess.Popen(["true"])
    dead.wait()
    url = "http://127.0.0.1:0/health"

    result = subprocess.run(
        [sys.executable, "-m", "infx.bench", "wait", "--url", url, "--pid", str(dead.pid)],
        env={"PYTHONPATH": str(REPO_ROOT)}, capture_output=True, text=True, timeout=30, check=False,
    )  # fmt: skip

    assert result.returncode == 1
    assert result.stderr == f"ERROR: process {dead.pid} died before {url} became ready\n"


def test_wait_streams_what_a_dying_server_wrote_before_failing(http_server, tmp_path, capsys):
    log = tmp_path / "server.log"
    log.write_text("loading\n")
    dead = subprocess.Popen(["true"])
    dead.wait()

    def respond(_method: str, _path: str) -> tuple[int, object]:
        with log.open("a") as out:
            out.write("CUDA out of memory\n")
        return 503, {}

    with pytest.raises(server.NotReadyError):
        server.wait_ready(f"{http_server(respond)}/health", pid=dead.pid, log=log, poll_s=0.01)

    assert capsys.readouterr().out == "loading\nCUDA out of memory\n"


@pytest.mark.parametrize(
    ("models", "health", "stabilization_s", "waiting"),
    [(["m"], 503, 0, False), (["other"], 200, 1, True)],
    ids=["route-mounted", "health-held"],
)
def test_chat_route_is_ready_once_the_model_is_served_or_health_holds(
    http_server, capsys, models, health, stabilization_s, waiting
):
    url = http_server(frontend(models, chat=405, health=health))
    chat = f"{url}/v1/chat/completions"
    start = time.monotonic()

    server.wait_chat_route(url, "m", (5, stabilization_s), poll_s=0.01)

    assert time.monotonic() - start >= stabilization_s
    waited = f"Waiting for {chat} ('m'): 0/5s\n" if waiting else ""
    assert capsys.readouterr().out == f"{waited}OpenAI chat endpoint ready for model 'm': {chat}\n"


def test_chat_route_keeps_waiting_through_a_null_model_list(http_server, capsys):
    health = iter([503, 200])

    def respond(_method: str, path: str) -> tuple[int, object]:
        if path == "/v1/models":
            return 200, {"data": None}
        return (next(health), {}) if path == "/health" else (404, {})

    url = http_server(respond)
    chat = f"{url}/v1/chat/completions"

    server.wait_chat_route(url, "m", (5, 0), poll_s=0.01)

    assert capsys.readouterr().out == (
        f"Waiting for {chat} ('m'): 0/5s\nOpenAI chat endpoint ready for model 'm': {chat}\n"
    )


def test_chat_route_gives_up_after_its_timeout(http_server, capsys):
    url = http_server(frontend(None, chat=404, health=503))
    chat = f"{url}/v1/chat/completions"

    with pytest.raises(server.NotReadyError) as raised:
        server.wait_chat_route(url, "m", (1, 0), poll_s=0.01)

    assert str(raised.value) == f"chat endpoint for model 'm' not ready within 1s: {chat}"
    assert capsys.readouterr().out == f"Waiting for {chat} ('m'): 0/1s\n"


def test_chat_route_deadline_counts_time_spent_in_slow_probes(http_server, monkeypatch, capsys):
    clock = SimpleNamespace(now=0.0)

    def sleep(seconds: float) -> None:
        clock.now += seconds

    monkeypatch.setattr(server, "time", SimpleNamespace(monotonic=lambda: clock.now, sleep=sleep))
    respond = frontend(None, chat=404, health=503)

    def slow_health(method: str, path: str) -> tuple[int, object]:
        if path == "/health":
            clock.now += 8
        return respond(method, path)

    url = http_server(slow_health)

    with pytest.raises(server.NotReadyError):
        server.wait_chat_route(url, "m", (10, 0), poll_s=1)

    assert clock.now == 17
    chat = f"{url}/v1/chat/completions"
    assert capsys.readouterr().out == f"Waiting for {chat} ('m'): 8/10s\n"
