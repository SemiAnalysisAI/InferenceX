"""The ``agentic`` command: point validation, each power mode's steps and status, and the shim."""

from __future__ import annotations

import contextlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from infx.bench.agentic import traces
from infx.bench.agentic.run import Plan, execute
from infx.bench.agentic.venv import Runtime
from infx.bench.env import BenchError, InputError
from infx.tests.bench.stubs import executable

REPO_ROOT = Path(__file__).resolve().parents[3]
SHIM = REPO_ROOT / "benchmarks" / "srt_agentic.sh"

POINT = {
    "RESULT_FILENAME": "agentx",
    "EVAL_ONLY": "false",
    "IS_MULTINODE": "false",
    "KV_OFFLOADING": "none",
    "PRECISION": "fp4",
    "MODEL": "test/model",
    "MODEL_PREFIX": "test",
    "FRAMEWORK": "vllm",
    "CONC": "8",
    "DURATION": "3600",
    "PORT": "8000",
    "AIPERF_DYNAMO_SESSION_TIMEOUT_SECONDS": "3600",
    "AIPERF_EXPERIMENTAL_FAST": "0",
    "AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID": "false",
    "AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING": "1",
}
# The runtime's python: each result step logs itself and exits with its *_RC.
FAKE_PYTHON = r"""#!/bin/bash
case "$2" in
    infx.results.agentic.process_agentic_result)
        echo "aggregate $RESULT_FILENAME" >> "$EVENTS"; exit "${AGGREGATE_RC:-0}" ;;
    infx.results.agentic.power_adapter)
        echo "adapter ${*:3}" >> "$EVENTS"; exit "${POWER_RC:-0}" ;;
    infx.results.agentic.validate_agentic_result)
        echo validate >> "$EVENTS"; exit "${VALIDATE_RC:-0}" ;;
    infx.results.agentic.analyze_benchmark_distributions) echo analyze >> "$EVENTS" ;;
    *) echo "unexpected $*" >> "$EVENTS"; exit 99 ;;
esac
"""
FAKE_AIPERF = r"""#!/bin/sh
echo replay >> "$EVENTS"
sleep "${REPLAY_SECONDS:-0}"
exit "${REPLAY_RC:-0}"
"""
FAKE_HF = '#!/bin/sh\nexit "${HF_RC:-0}"\n'
DRIVER = """
import os, sys
from pathlib import Path
from infx.bench.agentic.run import Plan, execute
from infx.bench.agentic.venv import Runtime
sys.exit(execute(Plan.from_env(os.environ), Runtime(Path(sys.argv[1])), os.environ))
"""


def _point(tmp_path: Path, **overrides: str | None) -> dict[str, str]:
    env = {
        **POINT,
        "PATH": os.environ["PATH"],
        "EVENTS": str(tmp_path / "events.log"),
        "REAL_PYTHON": sys.executable,
        "RESULT_DIR": str(tmp_path / "results"),
        "AGENTIC_OUTPUT_DIR": str(tmp_path),
        **overrides,
    }
    return {name: value for name, value in env.items() if value is not None}


def _runtime(tmp_path: Path) -> Runtime:
    runtime = Runtime(tmp_path / "runtime")
    runtime.python.parent.mkdir(parents=True)
    stubs = {runtime.python: FAKE_PYTHON, runtime.aiperf: FAKE_AIPERF, runtime.hf: FAKE_HF}
    for path, text in stubs.items():
        executable(path, text)
    return runtime


def _run(tmp_path: Path, **overrides: str | None) -> int:
    env = _point(tmp_path, **overrides)
    return execute(Plan.from_env(env), _runtime(tmp_path), env)


def _events(tmp_path: Path) -> list[str]:
    events = tmp_path / "events.log"
    return events.read_text().splitlines() if events.exists() else []


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"RESULT_FILENAME": None, "AIPERF_EXPERIMENTAL_FAST": None},
            "  - RESULT_FILENAME\n  - AIPERF_EXPERIMENTAL_FAST",
        ),
        ({"KV_OFFLOAD_BACKEND": "lmcache"}, "KV_OFFLOAD_BACKEND must be empty"),
        ({"KV_OFFLOADING": "dram", "TOTAL_CPU_DRAM_GB": "2400"}, "KV_OFFLOAD_BACKEND is required"),
        (
            {"KV_OFFLOADING": "dram", "KV_OFFLOAD_BACKEND": "none", "TOTAL_CPU_DRAM_GB": "2400"},
            "KV_OFFLOAD_BACKEND is required",
        ),
        (
            {"KV_OFFLOADING": "dram", "KV_OFFLOAD_BACKEND": "lmcache", "TOTAL_CPU_DRAM_GB": "0"},
            "TOTAL_CPU_DRAM_GB must be a positive integer",
        ),
        ({"KV_OFFLOADING": "cpu"}, "unsupported KV_OFFLOADING value 'cpu'"),
        ({"CONC_LIST": "4 8"}, "CONC_LIST='4 8' must equal CONC='8'"),
        ({"CONC_LIST": ""}, "CONC_LIST='' must equal CONC='8'"),
        ({"CONC": "4 8", "CONC_LIST": "4 8"}, "CONC must be a positive integer"),
        ({"IS_MULTINODE": "1"}, "IS_MULTINODE must be true or false"),
        ({"EVAL_ONLY": "true"}, "  - EVAL_ENDPOINT_READY_TIMEOUT_SECONDS"),
        ({"IS_MULTINODE": "true"}, "  - ENABLE_AGENTX_POWER\n  - REQUIRE_POWER"),
    ],
)
def test_points_that_cannot_be_measured_fail_before_setup(tmp_path, overrides, message):
    with pytest.raises(InputError, match=re.escape(message)):
        Plan.from_env(_point(tmp_path, **overrides))


WINDOW = {
    "IS_MULTINODE": "true", "ENABLE_AGENTX_POWER": "1", "REQUIRE_POWER": "0",
    "SRT_MEASUREMENT_WINDOW_DIR": "/w",
}
MISSING = {"IS_MULTINODE": "true", "ENABLE_AGENTX_POWER": "1", "REQUIRE_POWER": "1"}
MARK = "adapter --result-dir {results}/conc_8 --concurrency 8 --write-multinode-window"
OFFSET = "agentic_power_timezone_offset.txt"
REPLAYED = {"benchmark.log", "benchmark_command.txt"}


@pytest.mark.parametrize(
    ("overrides", "rc", "events", "files"),
    [
        pytest.param(
            # Only srt-slurm's multi-node telemetry measures power, whatever these say.
            {"ENABLE_AGENTX_POWER": "1", "REQUIRE_POWER": "1"},
            0,
            ["replay", "aggregate agentx", "analyze", "validate"],
            REPLAYED,
            id="single-node-publishes-no-power",
        ),
        pytest.param(
            # Multi-node recipes that pin IS_MULTINODE=false still run under the multi-node
            # workflow, which collects ${RESULT_FILENAME}_conc*.json.
            {"CONC_LIST": "8", "KV_OFFLOADING": "dram", "KV_OFFLOAD_BACKEND": "lmcache",
             "TOTAL_CPU_DRAM_GB": "2400"},
            0,
            ["replay", "aggregate agentx_conc8", "analyze", "validate"],
            {f"conc_8/{name}" for name in REPLAYED},
            id="conc-list-point",
        ),
        pytest.param(
            {**WINDOW, "ENABLE_AGENTX_POWER": "0"},
            0,
            ["replay", "aggregate agentx_conc8", "analyze", "validate"],
            {f"conc_8/{name}" for name in REPLAYED},
            id="multi-node-opt-out",
        ),
        pytest.param(
            WINDOW,
            0,
            [f"{MARK} running", "replay", "aggregate agentx_conc8", f"{MARK} completed",
             "analyze", "validate"],
            {f"conc_8/{name}" for name in (OFFSET, *REPLAYED)},
            id="multi-node-window",
        ),
        pytest.param(
            {**WINDOW, "REPLAY_RC": "143"},
            143,
            [f"{MARK} running", "replay", "aggregate agentx_conc8", "analyze", "validate"],
            {f"conc_8/{name}" for name in (OFFSET, *REPLAYED)},
            id="failed-replay-leaves-the-window-running",
        ),
        pytest.param(
            {**WINDOW, "POWER_RC": "4"},
            4,
            [f"{MARK} running"],
            {f"conc_8/{OFFSET}"},
            id="unpublished-window-skips-the-replay",
        ),
        pytest.param(
            MISSING,
            0,
            [
                "replay", "aggregate agentx_conc8", "adapter --result-dir {results}/conc_8 --agg-result {out}/agentx_conc8.json"
                " --multinode-contract-missing --require-power",
                "analyze", "validate",
            ],
            {f"conc_8/{name}" for name in REPLAYED},
            id="multi-node-without-window",
        ),
        pytest.param({"HF_RC": "6"}, 6, [], set(), id="failed-trace-download"),
    ],
)  # fmt: skip
def test_point_shape_decides_the_steps_status_and_artifacts(tmp_path, overrides, rc, events, files):
    assert _run(tmp_path, **overrides) == rc

    results = tmp_path / "results"
    assert _events(tmp_path) == [event.format(results=results, out=tmp_path) for event in events]
    written = {str(path.relative_to(results)) for path in results.rglob("*") if path.is_file()}
    assert written == files
    for offset in results.rglob(OFFSET):
        assert re.fullmatch(r"[+-]\d{4}\n", offset.read_text())


@pytest.mark.parametrize(
    ("replay", "aggregate", "validate", "power", "expected"),
    [
        ("7", "1", "3", "5", 7),
        ("0", "2", "3", "5", 2),
        ("0", "0", "3", "5", 3),
        ("0", "0", "0", "5", 5),
    ],
)
def test_every_step_runs_and_the_first_failure_in_precedence_wins(
    tmp_path, replay, aggregate, validate, power, expected
):
    rc = _run(
        tmp_path,
        **MISSING,
        REPLAY_RC=replay,
        AGGREGATE_RC=aggregate,
        VALIDATE_RC=validate,
        POWER_RC=power,
    )

    assert rc == expected
    assert [event.split()[0] for event in _events(tmp_path)] == [
        "replay", "aggregate", "adapter", "analyze", "validate",
    ]  # fmt: skip


@pytest.mark.parametrize(
    ("csv", "prefix", "error"),
    [
        (True, "vllm:", None),
        (True, "sglang:", "has no metric with required prefix 'sglang:'"),
        (False, "vllm:", "required AIPerf server metrics artifacts are missing or empty"),
    ],
)
def test_required_server_metrics_gate_an_otherwise_clean_point(tmp_path, csv, prefix, error):
    artifacts = tmp_path / "results" / "aiperf_artifacts"
    artifacts.mkdir(parents=True)
    # The quoted prefix straddles the scanner's first 1 MiB chunk.
    exported = b" " * ((1 << 20) - 3) + b'{"vllm:num_requests_running": 1}'
    (artifacts / "server_metrics_export.json").write_bytes(exported)
    if csv:
        (artifacts / "server_metrics_export.csv").write_text("metric,value\n")

    with pytest.raises(BenchError, match=re.escape(error)) if error else contextlib.nullcontext():
        assert _run(tmp_path, AIPERF_REQUIRED_SERVER_METRIC_PREFIX=prefix) == 0


def test_signal_during_the_replay_skips_scoring_and_exits_128_plus_n(tmp_path):
    runtime = _runtime(tmp_path)
    env = _point(tmp_path, REPLAY_SECONDS="60", PYTHONPATH=str(REPO_ROOT))
    driver = subprocess.Popen(
        [sys.executable, "-c", DRIVER, str(runtime.root)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 10
        while "replay" not in _events(tmp_path):
            assert time.monotonic() < deadline, "the replay did not start"
            time.sleep(0.01)
        # The whole job gets the signal, like a terminal interrupt. SIGTERM and SIGHUP share the
        # handler; test_fixed_seq's relay cases send those two.
        os.killpg(driver.pid, signal.SIGINT)
        _, stderr = driver.communicate(timeout=10)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(driver.pid, signal.SIGKILL)
        driver.communicate()

    assert driver.returncode == 130, stderr
    assert _events(tmp_path) == ["replay"]


# The replay records its argv, writes one profiled request, and prints progress.
FAKE_AIPERF_SCRIPT = r"""
import json, os, pathlib, shlex, sys
argv = sys.argv[1:]
out = pathlib.Path(argv[argv.index("--output-artifact-dir") + 1])
out.mkdir(parents=True, exist_ok=True)
pathlib.Path(os.environ["REPLAY_ARGV"]).write_text(shlex.join([sys.argv[0], *argv]))
with open(os.environ["EVENTS"], "a") as events:
    events.write("replay\n")
record = {
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
}
(out / "profile_export.jsonl").write_text(json.dumps(record) + "\n")
(out / "profile_export_aiperf.json").write_text(json.dumps({
    "request_count": {"avg": 1}, "error_request_count": {"avg": 0},
    "completed_request_count": {"avg": 1},
}))
print("fake aiperf: profiled 1 request", flush=True)
"""
# uv stand-in: `venv` makes a real (pip-less) venv; `pip install` adds AIPerf and hf stubs.
FAKE_UV = r"""#!/bin/sh
case "$1" in
    venv) for target; do :; done; exec "$REAL_PYTHON" -m venv --without-pip "$target" ;;
    pip)
        bin="$(dirname "$4")"
        printf '#!%s\n' "$bin/python" > "$bin/aiperf"
        cat "$FAKE_AIPERF_SCRIPT" >> "$bin/aiperf"
        printf '#!/bin/sh\necho "hf $*" >> "$EVENTS"\n' > "$bin/hf"
        chmod +x "$bin/aiperf" "$bin/hf" ;;
esac
"""


def test_srt_shim_builds_the_runtime_waits_for_the_server_and_writes_the_point(
    tmp_path, http_server
):
    tools = tmp_path / "tools"
    tools.mkdir()
    executable(tools / "uv", FAKE_UV)
    # The serving image's python3; it only needs the standard library.
    (tools / "python3").symlink_to(sys.executable)
    script = tmp_path / "fake_aiperf.py"
    script.write_text(FAKE_AIPERF_SCRIPT)

    def frontend(method: str, path: str) -> tuple[int, object]:
        with (tmp_path / "events.log").open("a") as events:
            events.write(f"{method} {path}\n")
        if path == "/v1/models":
            return 200, {"data": [{"id": "served/model"}]}
        return (405 if path == "/v1/chat/completions" else 200), {}

    env = _point(
        tmp_path,
        PATH=f"{tools}:/usr/bin:/bin",
        HOME=str(tmp_path),
        FAKE_AIPERF_SCRIPT=str(script),
        REPLAY_ARGV=str(tmp_path / "replay-argv"),
        HF_HUB_CACHE=str(tmp_path / "hf"),
        AIPERF_RUNTIME_DIR=str(tmp_path / "runtime"),
        EVAL_ONLY="true",
        AIPERF_SERVER_URL=http_server(frontend),
        SERVED_MODEL_NAME="served/model",
        EVAL_ENDPOINT_READY_TIMEOUT_SECONDS="30",
        EVAL_MODEL_STABILIZATION_SECONDS="0",
    )

    result = subprocess.run(
        ["bash", str(SHIM)], env=env, capture_output=True, text=True, timeout=120, check=False
    )

    assert result.returncode == 0, result.stderr
    assert _events(tmp_path) == [
        # MODEL_PREFIX=test is no 1M-context family, so the point replays the capped corpus.
        f"hf download --repo-type dataset {traces.LOADERS[traces.CAPPED]}",
        "GET /v1/models",
        "GET /v1/chat/completions",
        "replay",
    ]
    results = tmp_path / "results"
    command = (results / "benchmark_command.txt").read_text()
    assert command == (tmp_path / "replay-argv").read_text() + "\n"
    assert shlex.split(command)[0] == str(tmp_path / "runtime/venv/bin/aiperf")
    assert "fake aiperf: profiled 1 request" in (results / "benchmark.log").read_text()
    assert "fake aiperf: profiled 1 request" in result.stdout
    aggregate = json.loads((tmp_path / "agentx.json").read_text())
    assert (aggregate["conc"], aggregate["num_requests_total"]) == (8, 1)
