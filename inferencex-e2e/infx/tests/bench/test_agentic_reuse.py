"""Opt-in server reuse: group validation, the drain barrier, and the boundary between points."""

from __future__ import annotations

import contextlib
import copy
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from infx.bench.agentic import drain, reuse, run
from infx.bench.agentic.run import Plan, execute
from infx.bench.agentic.venv import Runtime
from infx.bench.env import BenchError, InputError
from infx.tests.bench.stubs import executable

REPO_ROOT = Path(__file__).resolve().parents[3]
RESET = "/reset_prefix_cache?reset_running_requests=false&reset_external=false"
PREFIX = "2026-10-31 18:02:11.123 - aiperf.timing.phase.runner - INFO - "
LIFECYCLE = [
    "Phase warmup (warmup) started",
    "Phase warmup (warmup) sending complete | sent=80, completed=72, in_flight=8"
    " | sessions: sent=16, completed=0",
    "Phase warmup (warmup) complete | completed=80, cancelled=0, errors=0 | elapsed=212.40s",
    "Phase profiling (profiling) started",
    "Phase profiling (profiling) sending complete | sent=2,607, completed=2,605, in_flight=2"
    " | sessions: sent=35, completed=23 | timeout_triggered=True",
    "Phase profiling (profiling) complete | completed=2,607, cancelled=0, errors=0"
    " | sessions: completed=23, cancelled=0 | elapsed=3620.01s",
]
IMAGE = "vllm/vllm-openai@sha256:" + "2b" * 32
SERVER = {
    "model": {"path": "hf:test/model", "container": IMAGE, "precision": "fp4"},
    "engine": {"type": "vllm", "connector": None},
    "frontend": {"type": "vllm", "enable_multiple_frontends": False},
    "roles": {
        "agg": {
            "gpus": 2, "nodes": 1, "workers": 1,
            "args": {"tensor-parallel-size": 1, "data-parallel-size": 2, "max-num-seqs": 16},
            "env": {"VLLM_SERVER_DEV_MODE": "1"},
        },
    },
}  # fmt: skip
POINT = {
    "RESULT_FILENAME": "agentx",
    "EVAL_ONLY": "false",
    "IS_MULTINODE": "false",
    "ENABLE_AGENTX_POWER": "0",
    "REQUIRE_POWER": "0",
    "IS_AGENTIC": "1",
    "KV_OFFLOADING": "none",
    "PRECISION": "fp4",
    "MODEL": "test/model",
    "MODEL_PREFIX": "test",
    "FRAMEWORK": "vllm",
    "CONC": "8",
    "CONC_LIST": "8 16",
    "DURATION": "3600",
    "AIPERF_DYNAMO_SESSION_TIMEOUT_SECONDS": "3600",
    "AIPERF_EXPERIMENTAL_FAST": "0",
    "AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID": "false",
    "AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING": "1",
    "AGENTX_SERVER_REUSE": "true",
    "AGENTX_REUSE_ENGINES": "0 1",
    "AGENTX_REUSE_DRAIN_TIMEOUT_SECONDS": "1",
    "AGENTX_REUSE_CLIENT_EXIT_TIMEOUT_SECONDS": "1",
}
RECORD = {
    "metadata": {
        "conversation_id": "trace-A", "turn_index": 0, "benchmark_phase": "profiling",
        "request_start_ns": 1_000_000_000, "request_ack_ns": 1_000_000_100,
        "request_end_ns": 2_000_000_000, "was_cancelled": False,
    },
    "metrics": {
        "input_sequence_length": {"value": 100, "unit": "tokens"},
        "output_sequence_length": {"value": 50, "unit": "tokens"},
        "time_to_first_token": {"value": 30.0, "unit": "ms"},
        "request_latency": {"value": 1000.0, "unit": "ms"},
        "inter_token_latency": {"value": 18.0, "unit": "ms"},
    },
    "error": None,
}  # fmt: skip
SUMMARY = {"request_count": {"avg": 1}, "error_request_count": {"avg": 0}}
FAKE_AGENTX = f"""#!{sys.executable}
import json, os, pathlib, subprocess, sys, time
argv = sys.argv[1:]
conc = argv[argv.index("--concurrency") + 1]
out = pathlib.Path(argv[argv.index("--output-artifact-dir") + 1])
if os.environ.get("REPLAY_URL_FILE"):
    pathlib.Path(os.environ["REPLAY_URL_FILE"]).write_text(argv[argv.index("--url") + 1])
with open(os.environ["EVENTS"], "a") as events:
    events.write(f"replay c{{conc}}\\n")
time.sleep(float(os.environ.get("REPLAY_SECONDS", "0")))
lines = {LIFECYCLE!r}
if os.environ.get("ERRORS_AT") == conc:
    lines[-1] = lines[-1].replace("errors=0", "errors=1")
(out / "logs").mkdir(parents=True)
(out / "logs" / "aiperf.log").write_text("".join({PREFIX!r} + line + "\\n" for line in lines))
(out / "profile_export.jsonl").write_text(json.dumps({RECORD!r}) + "\\n")
(out / "profile_export_aiperf.json").write_text(json.dumps({SUMMARY!r}))
if os.environ.get("ORPHAN_AT") == conc:
    orphan = subprocess.Popen(["sleep", "30"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pathlib.Path(os.environ["ORPHAN_PID"]).write_text(str(orphan.pid))
"""
FAKE_PYTHON = r"""#!/bin/bash
case "$2" in
    infx.results.agentic.process_agentic_result) echo "aggregate $RESULT_FILENAME" >> "$EVENTS" ;;
    infx.results.agentic.validate_agentic_result)
        echo validate >> "$EVENTS"; [ "$VALIDATE_FAIL_AT" = "$CONC" ] && exit 3; exit 0 ;;
    infx.results.agentic.analyze_benchmark_distributions) ;;
    *) echo "unexpected $*" >> "$EVENTS"; exit 99 ;;
esac
"""
IDLE = {"0": (0, 0, 500, 200), "1": (0, 0, 400, 100)}
SERVED = {"0": (0, 0, 9000, 5000), "1": (0, 0, 8000, 4000)}


def _metrics(engines: dict[str, tuple[int, int, int, int]]) -> str:
    lines = [
        "# TYPE vllm:num_requests_running gauge",
        'vllm:num_requests_running{model_name="m"} 9',
    ]
    for engine, values in engines.items():
        labels = f'engine="{engine}",model_name="test/model"'
        names = (reuse.RUNNING, reuse.WAITING, reuse.QUERIES, reuse.HITS)
        lines += [f"{name}{{{labels}}} {value}.0" for name, value in zip(names, values)]
    return "\n".join(lines) + "\n"


def _identities(*servers: dict, concurrencies: tuple[str, ...] = ("8", "16")) -> dict:
    servers = servers or (SERVER,) * len(concurrencies)
    return {
        "schema": reuse.SCHEMA,
        "points": {
            conc: {"image": IMAGE, "model": "test/model", "server": copy.deepcopy(server)}
            for conc, server in zip(concurrencies, servers)
        },
    }


@pytest.fixture
def requirements(tmp_path: Path) -> Path:
    path = tmp_path / "requirements.txt"
    path.write_text("agentx-harness==1.0.12\ndatasets>=4.7.0\n")
    return path


@pytest.fixture(autouse=True)
def fast_polls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reuse, "POLL_S", 0.01)


def _env(tmp_path: Path, document: object = None, **overrides: str | None) -> dict[str, str]:
    identities = tmp_path / "launch.json"
    identities.write_text(json.dumps(_identities() if document is None else document))
    env = {
        **POINT,
        "PATH": os.environ["PATH"],
        "EVENTS": str(tmp_path / "events.log"),
        "RESULT_DIR": str(tmp_path / "results"),
        "AGENTIC_OUTPUT_DIR": str(tmp_path),
        "AGENTX_REUSE_LAUNCH_IDENTITIES": str(identities),
        "PORT": "1",
        **overrides,
    }
    return {name: value for name, value in env.items() if value is not None}


def _runtime(tmp_path: Path, *, real_python: bool = False) -> Runtime:
    """A job runtime with stub clients; ``real_python`` makes its python a real bare venv."""
    runtime = Runtime(tmp_path / "runtime")
    if real_python:
        command = [sys.executable, "-m", "venv", "--without-pip", str(runtime.venv)]
        subprocess.run(command, check=True)
    else:
        runtime.python.parent.mkdir(parents=True)
        executable(runtime.python, FAKE_PYTHON)
    executable(runtime.agentx, FAKE_AGENTX)
    executable(runtime.hf, "#!/bin/sh\nexit 0\n")
    return runtime


def _events(tmp_path: Path) -> list[str]:
    events = tmp_path / "events.log"
    return events.read_text().splitlines() if events.exists() else []


def _vllm(
    tmp_path: Path,
    http_server,
    after=SERVED,
    reset=(200, {"success": True}),
    dev=(200, {"is_sleeping": False}),
    log_busy: bool = False,
) -> str:
    """A vLLM API whose engines report ``after`` once the first point has replayed."""

    def respond(method: str, path: str) -> tuple[int, object]:
        if method == "POST":
            with (tmp_path / "events.log").open("a") as events:
                events.write(f"POST {path}\n")
            return reset if path == RESET else (404, {})
        if path == reuse.DEV_PROBE_PATH:
            return dev
        if path != "/metrics":
            return 404, {}
        served = "replay c8" in _events(tmp_path)
        if served and log_busy:
            with (tmp_path / "events.log").open("a") as events:
                events.write("GET /metrics busy\n")
        return 200, _metrics(after if served else IDLE)

    return http_server(respond)


def _run_group(env: dict[str, str], tmp_path: Path, requirements: Path) -> int:
    group = reuse.Group.from_env(env, requirements)
    plans = [Plan.from_env(point) for point in group.points]
    return reuse.run(group, plans, _runtime(tmp_path), execute)


def _receipt(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "results" / reuse.RECEIPT).read_text())


def _log(path: Path, lines: list[str], *, terminated: bool = True) -> Path:
    path.write_text("\n".join(PREFIX + line for line in lines) + ("\n" if terminated else ""))
    return path


def test_drain_barrier_accepts_a_fully_returned_lifecycle(tmp_path):
    receipt = drain.check(_log(tmp_path / "aiperf.log", [LIFECYCLE[0], "noise", *LIFECYCLE[1:]]))

    assert receipt["phases"] == [
        {"name": "warmup", "kind": "warmup", "sent": 80, "completed": 80},
        {"name": "profiling", "kind": "profiling", "sent": 2607, "completed": 2607},
    ]


@pytest.mark.parametrize(
    ("lines", "terminated", "error"),
    [
        (
            [*LIFECYCLE[:5], LIFECYCLE[5].replace("errors=0", "errors=2")],
            True,
            "line 6: phase has cancelled or errored requests",
        ),
        (
            [*LIFECYCLE[:5], LIFECYCLE[5].replace("completed=2,607", "completed=2,606")],
            True,
            "line 6: issued and completed request counts differ",
        ),
        (
            [*LIFECYCLE[:4], "Request 7 finished | was_cancelled=True", *LIFECYCLE[4:]],
            True,
            "line 5: cancellation, grace expiry, or forced completion",
        ),
        ([LIFECYCLE[0], *LIFECYCLE], True, "line 2: duplicate, extra, or out-of-order"),
        (LIFECYCLE[:5], True, "lacks a complete warmup/profiling lifecycle"),
        (LIFECYCLE, False, "line 6: unterminated log line"),
    ],
)
def test_drain_barrier_rejects_evidence_of_an_unclean_drain(tmp_path, lines, terminated, error):
    with pytest.raises(drain.DrainError, match=re.escape(error)):
        drain.check(_log(tmp_path / "aiperf.log", lines, terminated=terminated))


def test_drain_barrier_treats_a_missing_log_as_failure(tmp_path):
    with pytest.raises(drain.DrainError, match="cannot read"):
        drain.check(tmp_path / "logs" / "aiperf.log")


def test_group_gives_each_point_its_own_concurrency_and_result_paths(tmp_path, requirements):
    group = reuse.Group.from_env(_env(tmp_path), requirements)

    results = tmp_path / "results"
    assert group.concurrencies == (8, 16)
    assert [(p["CONC"], p["RESULT_DIR"], p["RESULT_FILENAME"]) for p in group.points] == [
        ("8", str(results / "conc_8"), "agentx_conc8"),
        ("16", str(results / "conc_16"), "agentx_conc16"),
    ]
    assert all("CONC_LIST" not in point for point in group.points)
    plans = [Plan.from_env(point) for point in group.points]
    assert [plan.replay.artifact_dir for plan in plans] == [
        results / "conc_8" / "aiperf_artifacts",
        results / "conc_16" / "aiperf_artifacts",
    ]


def _server(**changes: object) -> dict:
    """``SERVER`` with dotted-path ``changes``; ``None`` deletes the key."""
    server = copy.deepcopy(SERVER)
    for path, value in changes.items():
        *parents, leaf = path.split("__")
        node = server
        for parent in parents:
            node = node[parent]
        if value is None:
            del node[leaf]
        else:
            node[leaf] = value
    return server


@pytest.mark.parametrize(
    ("document", "overrides", "message"),
    [
        (None, {"CONC_LIST": "8"}, "at least two unique CONC_LIST points, got '8'"),
        (None, {"CONC": "16"}, "CONC='16' must be the first CONC_LIST point (8)"),
        (None, {"IS_MULTINODE": "true"}, "unsupported: IS_MULTINODE"),
        (None, {"EVAL_ONLY": "true", "ENABLE_AGENTX_POWER": "1"},
         "unsupported: EVAL_ONLY, ENABLE_AGENTX_POWER"),
        (None, {"KV_OFFLOADING": "dram", "KV_OFFLOAD_BACKEND": "lmcache"},
         "unsupported: KV_OFFLOADING, KV_OFFLOAD_BACKEND"),
        (None, {"KV_OFFLOADING": None}, "  - KV_OFFLOADING"),
        (None, {"AGENTX_REUSE_ENGINES": "0 0"}, "AGENTX_REUSE_ENGINES must list unique"),
        (_identities(SERVER, _server(roles__agg__args__max_num_seqs=32)), {},
         "declare different server launches (server.roles.agg.args.max_num_seqs)"),
        ({"points": _identities()["points"]}, {}, "must declare schema 'infx-agentx-server-reuse/v1'"),
        (_identities(concurrencies=("8",)), {},
         "cover exactly the CONC_LIST points ['8', '16']; got ['8']"),
        ({**_identities(), "points": {"8": {"image": IMAGE, "model": "test/model"}, "16": {}}}, {},
         "point 8 must hold exactly ['image', 'model', 'server']"),
        (None, {"MODEL": "other/model"}, "point 8 model 'test/model' is not MODEL='other/model'"),
        (None, {"IMAGE": "vllm/vllm-openai:latest"}, "point 8 image"),
        (_identities(_server(model__container="other:tag"), _server(model__container="other:tag")),
         {}, "point 8 server model block does not serve test/model from image"),
        (_identities(_server(roles__agg__args=None), _server(roles__agg__args=None)), {},
         "point 8 server role 'agg' has no engine args"),
        (_identities(_server(engine="sglang"), _server(engine="sglang")), {},
         "point 8 server engine must be vllm"),
        (_identities(*[_server(engine__connector="nixl")] * 2), {},
         "point 8 server engine uses connector 'nixl'"),
        (_identities(*[_server(frontend__type="vllm-router")] * 2), {},
         "point 8 vllm-router reuse requires consistent_hash and disable-retries true"),
        (_identities(_server(benchmark={"env": {}}), _server(benchmark={"env": {}})), {},
         "without 'benchmark' and 'name'"),
        (_identities(*[_server(roles__agg__args__kv_transfer_config={"kv_connector": "x"})] * 2),
         {}, "external or offloaded KV cache (roles.agg.args.kv_transfer_config, "
             "roles.agg.args.kv_transfer_config.kv_connector)"),
        (_identities(*[_server(roles__agg__env__LMCACHE_CHUNK_SIZE="256")] * 2), {},
         "external or offloaded KV cache (roles.agg.env.LMCACHE_CHUNK_SIZE)"),
        (_identities(*[_server(roles__agg__args__kv_offloading_backend="native")] * 2), {},
         "external or offloaded KV cache (roles.agg.args.kv_offloading_backend)"),
    ],
)  # fmt: skip
def test_incompatible_groups_fail_before_any_point(
    tmp_path, requirements, document, overrides, message
):
    with pytest.raises(InputError, match=re.escape(message)):
        reuse.Group.from_env(_env(tmp_path, document, **overrides), requirements)


def test_reuse_refuses_a_client_release_whose_log_barrier_is_unvalidated(tmp_path, requirements):
    requirements.write_text("aiperf @ https://example.invalid/aiperf.tar.gz\n")

    with pytest.raises(InputError, match="validates the drain barrier only for agentx-harness"):
        reuse.Group.from_env(_env(tmp_path), requirements)


def test_reuse_refuses_result_paths_left_by_an_earlier_run(tmp_path, requirements):
    (tmp_path / "results" / "conc_16").mkdir(parents=True)

    with pytest.raises(InputError, match="fresh result paths.*conc_16"):
        reuse.Group.from_env(_env(tmp_path), requirements)


@pytest.mark.parametrize("switch", [None, "false"])
def test_without_opt_in_a_multi_point_list_is_still_rejected(tmp_path, monkeypatch, switch):
    monkeypatch.setattr(os, "environ", _env(tmp_path, AGENTX_SERVER_REUSE=switch))

    with pytest.raises(InputError, match=re.escape("CONC_LIST='8 16' must equal CONC='8'")):
        run.main([])


def test_clean_points_are_separated_by_one_reset(tmp_path, http_server, requirements):
    env = _env(tmp_path, AIPERF_SERVER_URL=_vllm(tmp_path, http_server))

    assert _run_group(env, tmp_path, requirements) == 0

    assert _events(tmp_path) == [
        "replay c8", "aggregate agentx_conc8", "validate",
        f"POST {RESET}",
        "replay c16", "aggregate agentx_conc16", "validate",
    ]  # fmt: skip
    receipt = _receipt(tmp_path)
    assert receipt["status"] == "passed"
    assert [point["concurrency"] for point in receipt["points"]] == [8, 16]
    assert receipt["points"][0]["reset_response"] == '{"success": true}'
    assert receipt["points"][0]["engines_after_reset"]["1"][reuse.QUERIES] == 8000
    assert "reset_response" not in receipt["points"][1]
    for conc in (8, 16):
        log = tmp_path / "results" / f"conc_{conc}" / "aiperf_artifacts" / drain.LOG
        assert log.is_file()


def test_router_traffic_and_direct_engine_control_are_separate(tmp_path, http_server, requirements):
    backend = _vllm(tmp_path, http_server)
    router_calls = []

    def router_response(method, path):
        router_calls.append((method, path))
        return 404, {}

    router = http_server(router_response)
    topology = _server(
        frontend__type="vllm-router",
        frontend__args={"policy": "consistent_hash", "disable-retries": True},
    )
    replay_url = tmp_path / "replay-url"
    env = _env(
        tmp_path,
        _identities(topology, topology),
        AIPERF_SERVER_URL=router,
        SRT_AGG_ENDPOINTS=backend.removeprefix("http://"),
        REPLAY_URL_FILE=str(replay_url),
    )
    group = reuse.Group.from_env(env, requirements)
    plans = [Plan.from_env(point) for point in group.points]
    assert all(plan.replay.server_metrics_urls == (f"{backend}/metrics",) for plan in plans)
    assert reuse.run(group, plans, _runtime(tmp_path), execute) == 0
    assert replay_url.read_text() == router
    assert router_calls == []
    assert _receipt(tmp_path)["control_url"] == backend
    assert _events(tmp_path).count(f"POST {RESET}") == 1


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({}, "needs one SRT_AGG_ENDPOINTS backend"),
        ({"SRT_AGG_ENDPOINTS": "localhost:1,localhost:2"}, "exactly one"),
        ({"SRT_AGG_ENDPOINTS": "localhost:1/v1"}, "HTTP API root"),
        ({"SRT_AGG_ENDPOINTS": "localhost:1?x=y"}, "HTTP API root"),
        ({"SRT_AGG_ENDPOINTS": "localhost:0"}, "HTTP API root"),
        (
            {
                "SRT_AGG_ENDPOINTS": "localhost:1",
                "AIPERF_SERVER_METRICS_URLS": "http://router/metrics",
            },
            "single metrics endpoint",
        ),
    ],
)
def test_router_reuse_rejects_missing_or_ambiguous_control(
    tmp_path, requirements, overrides, message
):
    topology = _server(
        frontend__type="vllm-router",
        frontend__args={"policy": "consistent_hash", "disable-retries": True},
    )
    env = _env(tmp_path, _identities(topology, topology), **overrides)
    with pytest.raises(InputError, match=message):
        reuse.Group.from_env(env, requirements)
    assert _events(tmp_path) == []


@pytest.mark.parametrize(
    ("server", "overrides", "error", "events"),
    [
        pytest.param(
            {"reset": (200, {"success": "true"})}, {},
            "reset did not return success: true", [f"POST {RESET}"], id="non-boolean-success",
        ),
        pytest.param(
            {"reset": (503, {"success": True})}, {},
            "reset answered HTTP 503", [f"POST {RESET}"], id="reset-http-error",
        ),
        pytest.param(
            {"after": {"0": SERVED["0"]}}, {},
            "server reports engines ['0'], expected ['0', '1']", [], id="engine-missing",
        ),
        pytest.param(
            {"after": {"0": SERVED["0"], "2": SERVED["1"]}}, {},
            "server reports engines ['0', '2']", [], id="engine-changed",
        ),
        pytest.param(
            {"after": {**SERVED, "1": (0, 0, 30, 0)}}, {},
            "engine 1 vllm:prefix_cache_queries_total fell from 400 to 30; the engine restarted",
            [], id="engine-restarted-under-the-same-index",
        ),
        pytest.param(
            {"after": {**SERVED, "1": (0, 1, 8000, 4000)}}, {},
            "engines did not stay idle within 1s", [], id="engine-never-drains",
        ),
        pytest.param(
            {}, {"ERRORS_AT": "8"},
            "drain barrier: line 6: phase has cancelled or errored requests", [], id="errored",
        ),
    ],
)  # fmt: skip
def test_a_failed_boundary_never_admits_the_next_point(
    tmp_path, http_server, requirements, server, overrides, error, events
):
    env = _env(tmp_path, AIPERF_SERVER_URL=_vllm(tmp_path, http_server, **server), **overrides)

    with pytest.raises(BenchError, match=re.escape(error)):
        _run_group(env, tmp_path, requirements)

    assert _events(tmp_path) == ["replay c8", "aggregate agentx_conc8", "validate", *events]
    receipt = _receipt(tmp_path)
    assert (receipt["status"], len(receipt["points"])) == ("failed", 1)
    assert error in receipt["error"]
    assert not (tmp_path / "results" / "conc_16").exists()


@pytest.mark.parametrize(
    ("dev", "error"),
    [
        ((404, {"detail": "Not Found"}), "/is_sleeping is not mounted, so neither is the "
         "prefix-cache reset; launch the server with VLLM_SERVER_DEV_MODE=1"),
        ((200, {"is_sleeping": True}), 'expected an awake server: {"is_sleeping": false}'),
    ],
)  # fmt: skip
def test_a_server_without_the_reset_route_fails_before_the_first_point(
    tmp_path, http_server, requirements, dev, error
):
    env = _env(tmp_path, AIPERF_SERVER_URL=_vllm(tmp_path, http_server, dev=dev))

    with pytest.raises(BenchError, match=re.escape(error)):
        _run_group(env, tmp_path, requirements)

    assert _events(tmp_path) == []
    assert _receipt(tmp_path)["points"] == []


def test_a_point_the_ordinary_scorer_rejects_stops_the_group_with_its_status(
    tmp_path, http_server, requirements
):
    env = _env(tmp_path, AIPERF_SERVER_URL=_vllm(tmp_path, http_server), VALIDATE_FAIL_AT="8")

    assert _run_group(env, tmp_path, requirements) == 3

    assert _events(tmp_path) == ["replay c8", "aggregate agentx_conc8", "validate"]
    receipt = _receipt(tmp_path)
    assert (receipt["status"], receipt["error"]) == (
        "failed",
        "point c8 exited 3; later points were not run",
    )


def test_an_unreachable_control_api_fails_before_the_first_point(tmp_path, requirements):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        closed = f"http://127.0.0.1:{sock.getsockname()[1]}"

    with pytest.raises(BenchError, match=re.escape(f"GET {closed}/is_sleeping failed")):
        _run_group(_env(tmp_path, AIPERF_SERVER_URL=closed), tmp_path, requirements)

    assert _events(tmp_path) == []
    assert _receipt(tmp_path)["points"] == []


def test_a_client_process_left_running_is_killed_and_stops_the_group(
    tmp_path, http_server, requirements
):
    pid_file = tmp_path / "orphan.pid"
    env = _env(
        tmp_path,
        AIPERF_SERVER_URL=_vllm(tmp_path, http_server),
        ORPHAN_AT="8",
        ORPHAN_PID=str(pid_file),
    )

    with pytest.raises(BenchError, match="still had live processes 1s after the client exited"):
        _run_group(env, tmp_path, requirements)

    assert _events(tmp_path) == ["replay c8"]
    orphan = int(pid_file.read_text())
    deadline = time.monotonic() + 10
    while _alive(orphan):
        assert time.monotonic() < deadline, "the orphaned client process survived"
        time.sleep(0.05)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    return not out.stdout.strip().startswith("Z")


def _signal_entry_point(tmp_path: Path, env: dict[str, str], ready: str, signum: int):
    """Run ``python3 -m infx.bench agentic`` in the job's runtime; send ``signum`` at ``ready``."""
    runtime = _runtime(tmp_path, real_python=True)
    env = {**env, "AIPERF_RUNTIME_DIR": str(runtime.root), "PYTHONPATH": str(REPO_ROOT)}
    command = [str(runtime.python), "-m", "infx.bench", "agentic"]
    driver = subprocess.Popen(
        command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=True,
    )  # fmt: skip
    try:
        deadline = time.monotonic() + 30
        while ready not in _events(tmp_path):
            if driver.poll() is not None:
                pytest.fail(f"the command exited before {ready!r}: {driver.communicate()}")
            assert time.monotonic() < deadline, f"never reached {ready!r}"
            time.sleep(0.01)
        os.killpg(driver.pid, signum)
        stdout, stderr = driver.communicate(timeout=30)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(driver.pid, signal.SIGKILL)
        driver.communicate()
    return driver.returncode, stdout + stderr


def test_a_signal_during_a_point_reaches_only_its_client_and_stops_the_group(tmp_path, http_server):
    env = _env(tmp_path, AIPERF_SERVER_URL=_vllm(tmp_path, http_server), REPLAY_SECONDS="60")

    rc, output = _signal_entry_point(tmp_path, env, "replay c8", signal.SIGINT)

    assert rc == 130, output
    assert _events(tmp_path) == ["replay c8"]
    receipt = _receipt(tmp_path)
    assert (receipt["status"], receipt["error"]) == (
        "terminated",
        "point c8 exited 130; later points were not run",
    )


def test_a_signal_between_points_stops_admission_with_a_terminated_receipt(tmp_path, http_server):
    busy = {**SERVED, "1": (1, 0, 8000, 4000)}
    url = _vllm(tmp_path, http_server, after=busy, log_busy=True)
    env = _env(tmp_path, AIPERF_SERVER_URL=url, AGENTX_REUSE_DRAIN_TIMEOUT_SECONDS="60")

    rc, output = _signal_entry_point(tmp_path, env, "GET /metrics busy", signal.SIGTERM)

    assert rc == 143, output
    assert "replay c16" not in _events(tmp_path)
    assert f"POST {RESET}" not in _events(tmp_path)
    receipt = _receipt(tmp_path)
    assert (receipt["status"], receipt["error"]) == (
        "terminated",
        "signal 15 received after c8; no later point admitted",
    )
    assert (tmp_path / "agentx_conc8.json").is_file()
