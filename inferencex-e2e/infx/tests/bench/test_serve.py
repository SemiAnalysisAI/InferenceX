"""``serve`` against a stub server process on a real port and stub clients."""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from infx.bench import proc, serve
from infx.tests.processes import free_port, gone

REPO_ROOT = Path(__file__).resolve().parents[3]

SERVER = r"""
import http.server, json, os, signal, subprocess, sys, threading, time
port = int(sys.argv[sys.argv.index("--port") + 1])
record = os.environ["FAKE_RECORD"]
if os.environ.get("FAKE_IGNORE_TERM"):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
if os.environ.get("FAKE_DIE_ON"):
    def die():
        while not os.path.exists(os.environ["FAKE_DIE_ON"]):
            time.sleep(0.05)
        os._exit(3)
    threading.Thread(target=die, daemon=True).start()
with open(record + ".server.tmp", "w") as out:
    json.dump({"argv": sys.argv[1:], "pid": os.getpid(), "worker": worker.pid,
               "env": {name: os.environ.get(name) for name in ("SHARED",)}}, out)
os.replace(record + ".server.tmp", record + ".server")
print("server starting", flush=True)
if not os.environ.get("FAKE_IGNORE_TERM"):
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

class Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        ready = not os.environ.get("FAKE_NEVER_READY") and self.path == "/health"
        self.send_response(200 if ready else 503)
        self.end_headers()
    def log_message(self, *_):
        pass

http.server.ThreadingHTTPServer(("127.0.0.1", port), Health).serve_forever()
"""

CLIENT = r"""
import json, os, sys, time, urllib.request
name, rc = sys.argv[1], int(sys.argv[2])
with urllib.request.urlopen(os.environ["FAKE_URL"], timeout=5) as response:
    status = response.status
with open(os.environ["FAKE_RECORD"] + ".clients", "a") as out:
    out.write(json.dumps({"name": name, "health": status,
                          "env": {n: os.environ.get(n) for n in ("SHARED",)}}) + "\n")
if os.environ.get("FAKE_DIE_ON"):
    open(os.environ["FAKE_DIE_ON"], "w").close()
    for _ in range(200):
        try:
            urllib.request.urlopen(os.environ["FAKE_URL"], timeout=1).close()
        except OSError:
            break
        time.sleep(0.05)
sys.exit(rc)
"""


@pytest.fixture
def stubs(tmp_path, monkeypatch):
    """The stub server and client scripts, an environment both read, and their record."""
    (tmp_path / "server.py").write_text(SERVER)
    (tmp_path / "client.py").write_text(CLIENT)
    port = free_port()
    record = tmp_path / "record"
    url = f"http://127.0.0.1:{port}/health"
    monkeypatch.setenv("FAKE_RECORD", str(record))
    monkeypatch.setenv("FAKE_URL", url)
    monkeypatch.setenv("SHARED", "both")
    server = [sys.executable, str(tmp_path / "server.py"), "--port", str(port)]
    return {"server": server, "url": url, "record": record, "client": str(tmp_path / "client.py")}


def client(stubs, name: str, rc: int = 0) -> list[str]:
    return [sys.executable, stubs["client"], name, str(rc)]


def recorded(stubs, suffix: str) -> list[dict]:
    path = Path(f"{stubs['record']}.{suffix}")
    text = path.read_text() if path.exists() else ""
    return [json.loads(line) for line in text.splitlines()]


def test_clients_run_in_order_against_the_ready_server_which_is_stopped_with_its_workers(
    stubs, tmp_path
):
    log = tmp_path / "server.log"

    rc = serve.serve(
        stubs["server"], stubs["url"], log, [client(stubs, "throughput"), client(stubs, "eval")]
    )

    assert rc == 0
    [server] = recorded(stubs, "server")
    assert server["env"] == {"SHARED": "both"}
    assert [(run["name"], run["health"], run["env"]) for run in recorded(stubs, "clients")] == [
        ("throughput", 200, {"SHARED": "both"}),
        ("eval", 200, {"SHARED": "both"}),
    ]
    assert log.read_text() == "server starting\n"
    assert gone(server["worker"])


def test_a_failing_client_stops_the_rest_and_its_status_is_the_commands(stubs, tmp_path):
    rc = serve.serve(
        stubs["server"], stubs["url"], tmp_path / "server.log",
        [client(stubs, "throughput", rc=3), client(stubs, "eval")],
    )  # fmt: skip

    assert rc == 3
    assert [run["name"] for run in recorded(stubs, "clients")] == ["throughput"]
    assert gone(recorded(stubs, "server")[0]["worker"])


def test_a_server_that_exits_while_a_client_runs_fails_the_run_though_the_client_passed(
    stubs, tmp_path, monkeypatch
):
    monkeypatch.setenv("FAKE_DIE_ON", str(tmp_path / "die"))

    rc = serve.serve(
        stubs["server"], stubs["url"], tmp_path / "server.log", [client(stubs, "throughput")]
    )

    assert rc == 1
    assert [run["name"] for run in recorded(stubs, "clients")] == ["throughput"]
    assert gone(recorded(stubs, "server")[0]["worker"])


def test_a_server_not_ready_within_its_budget_is_stopped_without_running_a_client(
    stubs, tmp_path, monkeypatch
):
    monkeypatch.setenv("FAKE_NEVER_READY", "1")

    rc = serve.serve(
        stubs["server"],
        stubs["url"],
        tmp_path / "server.log",
        [client(stubs, "throughput")],
        ready_timeout_s=1,
    )

    assert rc == 1
    assert recorded(stubs, "clients") == []
    assert gone(recorded(stubs, "server")[0]["worker"])


def test_a_server_ignoring_sigterm_is_killed_with_its_workers_after_the_grace_period(
    stubs, tmp_path, monkeypatch
):
    monkeypatch.setenv("FAKE_IGNORE_TERM", "1")
    monkeypatch.setattr(serve, "STOP_GRACE_S", 0.2)

    def hung(signum: int, frame: object) -> None:
        raise TimeoutError("stop() never killed a server ignoring SIGTERM")

    previous = signal.signal(signal.SIGALRM, hung)
    signal.alarm(30)
    try:
        rc = serve.serve(
            stubs["server"], stubs["url"], tmp_path / "server.log", [client(stubs, "x")]
        )
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)

    assert rc == 0
    [server] = recorded(stubs, "server")
    assert gone(server["pid"])
    assert gone(server["worker"])


def test_sigterm_while_the_server_starts_stops_it_and_exits_128_plus_the_signal(stubs, tmp_path):
    env = {**os.environ, "FAKE_NEVER_READY": "1", "PYTHONPATH": str(REPO_ROOT)}
    argv = [
        sys.executable, "-m", "infx.bench", "serve", "--ready-url", stubs["url"],
        "--log", str(tmp_path / "server.log"), "--client", json.dumps(client(stubs, "x")),
        "--", *stubs["server"],
    ]  # fmt: skip
    command = subprocess.Popen(argv, env=env, cwd=tmp_path)
    deadline = time.monotonic() + 30
    while not Path(f"{stubs['record']}.server").exists():
        assert time.monotonic() < deadline, "the server never started"
        time.sleep(0.05)
    [server] = recorded(stubs, "server")

    command.send_signal(signal.SIGTERM)

    assert command.wait(timeout=30) == 128 + signal.SIGTERM
    assert recorded(stubs, "clients") == []
    assert gone(server["worker"])


PEEK = "import os, pathlib, sys; pathlib.Path(sys.argv[1]).write_text(f'{os.getpgrp()} ' + pathlib.Path(sys.argv[2]).read_text())"


def test_the_servers_group_is_on_record_before_the_server_runs(stubs, tmp_path, monkeypatch):
    groups, seen = tmp_path / "groups", tmp_path / "seen"
    monkeypatch.setenv(proc.GROUP_RECORD_ENV, str(groups))

    rc = serve.serve(
        [sys.executable, "-c", PEEK, str(seen), str(groups)],
        stubs["url"],
        tmp_path / "server.log",
        [client(stubs, "x")],
    )

    assert rc == 1
    pgid, record = seen.read_text().split(" ", 1)
    assert record == groups.read_text()
    [(recorded_pgid, at)] = proc.recorded_groups(groups)
    assert recorded_pgid == int(pgid)
    assert abs(at - time.time()) < 60


SAMPLER = r"""
import os, signal, sys, time
with open(sys.argv[1], "w") as out:
    out.write(f"{os.getpid()} {os.getpgrp()}")
time.sleep(600)
"""


def test_the_sampler_runs_in_its_own_recorded_group_and_stops_with_the_server(
    stubs, tmp_path, monkeypatch
):
    groups, seen = tmp_path / "groups", tmp_path / "sampler"
    monkeypatch.setenv(proc.GROUP_RECORD_ENV, str(groups))

    rc = serve.serve(
        stubs["server"], stubs["url"], tmp_path / "server.log", [client(stubs, "a")],
        sampler_argv=[sys.executable, "-c", SAMPLER, str(seen)],
    )  # fmt: skip

    assert rc == 0
    pid, pgid = map(int, seen.read_text().split())
    assert pid == pgid
    assert pgid in [group for group, _ in proc.recorded_groups(groups)]
    assert gone(pid)


def test_a_sampler_that_fails_costs_the_run_nothing(stubs, tmp_path):
    rc = serve.serve(
        stubs["server"], stubs["url"], tmp_path / "server.log", [client(stubs, "a")],
        sampler_argv=[sys.executable, "-c", "raise SystemExit(3)"],
    )  # fmt: skip

    assert rc == 0
    assert [entry["name"] for entry in recorded(stubs, "clients")] == ["a"]


def test_a_sampler_that_cannot_start_costs_the_run_nothing(stubs, tmp_path, capsys):
    rc = serve.serve(
        stubs["server"], stubs["url"], tmp_path / "server.log", [client(stubs, "a")],
        sampler_argv=[str(tmp_path / "no-such-sampler")],
    )  # fmt: skip

    assert rc == 0
    assert "could not start the sampler" in capsys.readouterr().err


def test_the_gate_runs_first_and_a_failing_or_unstartable_one_costs_the_run_nothing(
    stubs, tmp_path, capsys
):
    for gate in (client(stubs, "gate", rc=3), [str(tmp_path / "no-such-gate")]):
        assert serve.serve(
            stubs["server"], stubs["url"], tmp_path / "server.log", [client(stubs, "a")],
            gate_argv=gate, gate_timeout_s=30,
        ) == 0  # fmt: skip

    assert [entry["name"] for entry in recorded(stubs, "clients")] == ["gate", "a", "a"]
    assert "could not start the gate" in capsys.readouterr().err


def test_a_gate_past_its_timeout_stops_in_its_recorded_group_and_the_clients_still_run(
    stubs, tmp_path, monkeypatch
):
    groups, seen = tmp_path / "groups", tmp_path / "gate"
    monkeypatch.setenv(proc.GROUP_RECORD_ENV, str(groups))

    def clock() -> float:
        """No time passes until the gate has written its marker, and then its timeout has."""
        return math.inf if seen.exists() and seen.read_text() else 0.0

    monkeypatch.setattr(serve, "time", SimpleNamespace(monotonic=clock, sleep=time.sleep))

    rc = serve.serve(
        stubs["server"], stubs["url"], tmp_path / "server.log", [client(stubs, "a")],
        gate_argv=[sys.executable, "-c", SAMPLER, str(seen)], gate_timeout_s=5.0,
    )  # fmt: skip

    assert rc == 0
    pid, pgid = map(int, seen.read_text().split())
    assert pgid in [group for group, _ in proc.recorded_groups(groups)]
    assert gone(pid)
    assert [entry["name"] for entry in recorded(stubs, "clients")] == ["a"]


def test_sigterm_during_the_gate_stops_it_and_runs_no_client(stubs, tmp_path):
    seen = tmp_path / "gate"
    command = subprocess.Popen(
        [
            sys.executable, "-m", "infx.bench", "serve", "--ready-url", stubs["url"],
            "--log", str(tmp_path / "server.log"), "--client", json.dumps(client(stubs, "a")),
            "--gate", json.dumps([sys.executable, "-c", SAMPLER, str(seen)]),
            "--gate-timeout", "600", "--", *stubs["server"],
        ],
        cwd=REPO_ROOT,
    )  # fmt: skip
    deadline = time.monotonic() + 30
    while not (seen.exists() and seen.read_text()) and time.monotonic() < deadline:
        time.sleep(0.05)
    pid = int(seen.read_text().split()[0])

    command.send_signal(signal.SIGTERM)

    assert command.wait(timeout=90) == 128 + signal.SIGTERM
    assert gone(pid)
    assert recorded(stubs, "clients") == []
