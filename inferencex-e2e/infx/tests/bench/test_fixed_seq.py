"""Fixed-sequence lanes against a stub client and a local frontend."""

from __future__ import annotations

import json
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from infx.bench.__main__ import main as bench
from infx.tests.bench.stubs import executable

REPO_ROOT = Path(__file__).resolve().parents[3]
# Client flags every lane passes, whatever the point.
POLICY = {
    "--dataset-name": "random",
    "--request-rate": "inf",
    "--ignore-eos": True,
    "--save-result": True,
    "--percentile-metrics": "ttft,tpot,itl,e2el",
}

# Records each client run and writes the result file the real client would. With
# FAKE_CLIENT_TRAP set, it waits for SIGTERM or SIGHUP, records the signal, and exits 0.
FAKE_CLIENT = """
import json, os, signal, sys, time
from pathlib import Path

def finish(signum, _frame):
    Path(os.environ["FAKE_CLIENT_TRAP"]).write_text(signal.Signals(signum).name)
    sys.exit(0)

trap = os.environ.get("FAKE_CLIENT_TRAP")
if trap:
    signal.signal(signal.SIGTERM, finish)
    signal.signal(signal.SIGHUP, finish)
argv = sys.argv[1:]
value = lambda flag: argv[argv.index(flag) + 1]
result = Path(value("--result-dir")) / value("--result-filename")
record = {
    "argv": argv,
    "monitored": (result.parent / "gpu_metrics.csv").exists(),
    "safe_path": os.environ.get("PYTHONSAFEPATH"),
}
with open(os.environ["FAKE_CLIENT_LOG"], "a") as log:
    log.write(json.dumps(record) + "\\n")
while trap:
    time.sleep(0.05)
if value("--max-concurrency") == os.environ.get("FAKE_CLIENT_FAIL_CONC"):
    sys.exit(3)
result.write_text(json.dumps(
    {"benchmark_start_time_unix": 100.0, "benchmark_end_time_unix": 160.0, "duration": 60.0}
))
"""


@pytest.fixture
def tools(tmp_path: Path) -> Path:
    """PATH with a stub benchmark client behind python3."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("sh", "sleep", "dirname", "env"):
        (bin_dir / tool).symlink_to(shutil.which(tool))
    (tmp_path / "fake_client.py").write_text(FAKE_CLIENT)
    python, client = shlex.quote(sys.executable), shlex.quote(str(tmp_path / "fake_client.py"))
    stubs = {
        "python3": f"""
if [ "$1 $2" = "-m infx.bench_serving.benchmark_serving" ]; then
    shift 2
    exec {python} {client} "$@"
fi
exec {python} "$@"
""",
    }
    for name, body in stubs.items():
        executable(bin_dir / name, f"#!/bin/sh\n{body}")
    return bin_dir


def client_runs(tmp_path: Path) -> list[dict]:
    log = tmp_path / "client.log"
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def options(argv: list[str]) -> dict[str, str | bool]:
    """Client flags as a mapping; a flag without a value maps to True."""
    parsed: dict[str, str | bool] = {}
    for index, token in enumerate(argv):
        if token.startswith("--"):
            following = argv[index + 1] if index + 1 < len(argv) else "--"
            parsed[token] = True if following.startswith("--") else following
    return parsed


def use_env(monkeypatch: pytest.MonkeyPatch, env: dict[str, str | None]) -> None:
    for name, value in env.items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)


def run_shim(shim: str, env: dict[str, str | None], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [shutil.which("bash"), str(REPO_ROOT / "benchmarks" / shim), *args],
        env={name: value for name, value in env.items() if value is not None},
        capture_output=True, text=True, timeout=60, check=False,
    )  # fmt: skip


def single_node_env(tmp_path: Path, tools: Path, **overrides: str | None) -> dict[str, str | None]:
    (tmp_path / "logs").mkdir(exist_ok=True)
    return {
        "PATH": str(tools),
        "PYTHONDONTWRITEBYTECODE": "1",
        "FAKE_CLIENT_LOG": str(tmp_path / "client.log"),
        "MODEL": "org/Model-FP8",
        "CONC": "4",
        "ISL": "1024",
        "OSL": "128",
        "RANDOM_RANGE_RATIO": "0.8",
        "RESULT_FILENAME": "point_conc4",
        "RESULT_DIR": str(tmp_path / "logs"),
        "SRT_FRONTEND_HOST": "10.0.0.7",
        "SRT_FRONTEND_PORT": "8000",
        "RUN_EVAL": "false",
        "EVAL_ONLY": "false",
        "SRT_MEASUREMENT_WINDOW_DIR": None,
        "USE_CHAT_TEMPLATE": "false",
        "FRAMEWORK": "sglang",
        **overrides,
    }


def sweep_env(
    tmp_path: Path, tools: Path, url: str, **overrides: str | None
) -> dict[str, str | None]:
    return {
        "PATH": str(tools),
        "PYTHONDONTWRITEBYTECODE": "1",
        "FAKE_CLIENT_LOG": str(tmp_path / "client.log"),
        "ISL": "1024",
        "OSL": "128",
        "RANDOM_RANGE_RATIO": "0.8",
        "SRT_FRONTEND_HOST": "127.0.0.1",
        "SRT_FRONTEND_PORT": str(urlsplit(url).port),
        "CONC_LIST": "4 16",
        "PREFILL_NUM_WORKERS": "1",
        "PREFILL_TP": "2",
        "DECODE_NUM_WORKERS": "2",
        "DECODE_TP": "4",
        "TOKENIZER": None,
        "SRT_MEASUREMENT_WINDOW_DIR": None,
        **overrides,
    }


def frontend(*models: str):
    """An OpenAI frontend listing ``models``."""
    return lambda _method, path: (
        (200, {"data": [{"id": model, "object": "model"} for model in models]})
        if path == "/v1/models"
        else (404, {})
    )


# What benchmarks/multi_node/llm-d/server.sh passes for one concurrency.
POINT_FLAGS = {
    "--base-url": "http://0.0.0.0:8080",
    "--model": "deepseek-ai/DeepSeek-V4-Pro",
    "--backend": "openai",
    "--tokenizer": "/models",
    "--isl": "8192",
    "--osl": "1024",
    "--random-range-ratio": "0.8",
    "--conc": "1",
    "--num-prompts": "16",
}


def point_argv(result: Path) -> list[str]:
    flags = {**POINT_FLAGS, "--result": str(result)}
    return ["fixed-seq", "point", *(token for pair in flags.items() for token in pair)]


@pytest.mark.parametrize("client_failed", [True, False])
def test_single_node_does_not_report_success_without_a_completed_window(
    tmp_path, tools, client_failed,
):
    windows = tmp_path / "logs" / "power" / "windows"
    if client_failed:
        windows.mkdir(parents=True)
    env = single_node_env(
        tmp_path, tools, SRT_MEASUREMENT_WINDOW_DIR=str(windows),
        FAKE_CLIENT_FAIL_CONC="4" if client_failed else "",
    )

    result = run_shim("single_node/srt_fixed_sequence.sh", env)

    assert result.returncode == (3 if client_failed else 1)
    assert len(client_runs(tmp_path)) == 1
    assert not (windows / "point_conc4.json").exists()


@pytest.mark.parametrize(
    ("framework", "chat_template", "args", "flags"),
    [
        ("sglang", "true", [], {"--backend": "vllm", "--use-chat-template": True}),
        ("trt", "false", ["--trust-remote-code"], {"--backend": "openai", "--trust-remote-code": True}),
    ],
)
def test_single_node_shim_runs_one_point_with_native_power(
    tmp_path, tools, framework, chat_template, args, flags
):
    windows = tmp_path / "logs" / "power" / "windows"
    windows.mkdir(parents=True)
    env = single_node_env(
        tmp_path, tools, FRAMEWORK=framework, USE_CHAT_TEMPLATE=chat_template,
        SRT_MEASUREMENT_WINDOW_DIR=str(windows),
    )

    result = run_shim("single_node/srt_fixed_sequence.sh", env, *args)

    assert result.returncode == 0, result.stderr
    [run] = client_runs(tmp_path)
    logs = tmp_path / "logs"
    assert options(run["argv"]) == {
        **POLICY,
        "--model": "org/Model-FP8",
        "--base-url": "http://10.0.0.7:8000",
        "--random-input-len": "1024",
        "--random-output-len": "128",
        "--random-range-ratio": "0.8",
        "--num-prompts": "40",
        "--max-concurrency": "4",
        "--num-warmups": "8",
        "--result-dir": str(logs),
        "--result-filename": "point_conc4.json",
        **flags,
    }
    assert (logs / "point_conc4.json").is_file()
    assert not run["monitored"]
    # The shim sets PYTHONSAFEPATH for infx.bench only; Python-script tools break under it.
    assert run["safe_path"] is None
    assert not (logs / "gpu_metrics.csv").exists()
    window = json.loads((windows / "point_conc4.json").read_text())
    assert window["result_path"] == "point_conc4.json"
    assert window["concurrency"] == 4
    assert (window["benchmark_start_time_unix"], window["benchmark_end_time_unix"]) == (100, 160)
    assert window["status"] == "completed"


@pytest.mark.parametrize(
    ("overrides", "returncode", "reported"),
    [
        ({"EVAL_ONLY": "true"}, 0, "EVAL_ONLY mode: skipping throughput benchmark\n"),
        ({"MODEL": None, "CONC": ""}, 1, "not set:\n  - MODEL\n  - CONC\n"),
        ({"FRAMEWORK": "no-such-framework"}, 1, "ERROR: unsupported fixed-sequence FRAMEWORK: no-such-framework\n"),
        ({"USE_CHAT_TEMPLATE": "yes"}, 1, "ERROR: USE_CHAT_TEMPLATE must be true or false, got 'yes'\n"),
        ({"RESULT_DIR": "/nonexistent/logs"}, 1, "ERROR: RESULT_DIR must be an existing"),
        ({}, 1, "SRT_MEASUREMENT_WINDOW_DIR"),
    ],
    ids=["eval-only", "missing", "unsupported-framework", "malformed-flag", "no-result-dir",
         "missing-native-power"],
)
def test_single_node_point_runs_nothing_when_eval_only_or_misconfigured(
    tmp_path, tools, monkeypatch, capsys, overrides, returncode, reported
):
    use_env(monkeypatch, single_node_env(tmp_path, tools, **overrides))

    assert bench(["fixed-seq", "srt-single"]) == returncode
    assert reported in "".join(capsys.readouterr())
    assert client_runs(tmp_path) == []
    assert list((tmp_path / "logs").iterdir()) == []


def test_multi_node_shim_writes_one_result_and_power_window_per_concurrency(
    tmp_path, tools, http_server
):
    url = http_server(frontend("served/model-id"))
    windows = tmp_path / "logs" / "power" / "windows"
    windows.mkdir(parents=True)
    env = sweep_env(tmp_path, tools, url, TOKENIZER="/model", SRT_MEASUREMENT_WINDOW_DIR=str(windows))

    # A later --logs-dir overrides the shim's /logs.
    result = run_shim("multi_node/srt_fixed_sequence.sh", env, "--logs-dir", str(tmp_path / "logs"))

    assert result.returncode == 0, result.stderr
    point_dir = tmp_path / "logs" / "sa-bench_isl_1024_osl_128"
    names = [
        "results_concurrency_4_gpus_10_ctx_2_gen_8.json",
        "results_concurrency_16_gpus_10_ctx_2_gen_8.json",
    ]
    first, second = (options(run["argv"]) for run in client_runs(tmp_path))
    assert first == {
        **POLICY,
        "--model": "served/model-id",
        "--backend": "openai",
        "--base-url": url,
        "--tokenizer": "/model",
        "--random-input-len": "1024",
        "--random-output-len": "128",
        "--random-range-ratio": "0.8",
        "--num-prompts": "40",
        "--max-concurrency": "4",
        "--num-warmups": "8",
        "--use-chat-template": True,
        "--trust-remote-code": True,
        "--result-dir": str(point_dir),
        "--result-filename": names[0],
    }
    assert {key: second[key] for key in ("--num-prompts", "--max-concurrency", "--num-warmups")} == {
        "--num-prompts": "160",
        "--max-concurrency": "16",
        "--num-warmups": "32",
    }
    assert sorted(path.name for path in point_dir.iterdir()) == sorted(names)
    written = {path.name: json.loads(path.read_text()) for path in windows.iterdir()}
    assert {name: (window["result_path"], window["concurrency"]) for name, window in written.items()} == {
        names[0]: (f"sa-bench_isl_1024_osl_128/{names[0]}", 4),
        names[1]: (f"sa-bench_isl_1024_osl_128/{names[1]}", 16),
    }


def test_sweep_stops_at_the_first_failing_point(tmp_path, tools, http_server, monkeypatch):
    url = http_server(frontend("served/model-id"))
    # An aggregated row: no decode workers, and no TOKENIZER.
    use_env(monkeypatch, sweep_env(
        tmp_path, tools, url, CONC_LIST="4 16 32", PREFILL_NUM_WORKERS="2", DECODE_NUM_WORKERS="0",
        FAKE_CLIENT_FAIL_CONC="16",
    ))  # fmt: skip

    assert bench(["fixed-seq", "srt-sweep", "--logs-dir", str(tmp_path / "logs")]) == 3
    runs = [options(run["argv"]) for run in client_runs(tmp_path)]
    assert [(run["--max-concurrency"], run["--tokenizer"]) for run in runs] == [
        ("4", "served/model-id"),
        ("16", "served/model-id"),
    ]
    point_dir = tmp_path / "logs" / "sa-bench_isl_1024_osl_128"
    assert [path.name for path in point_dir.iterdir()] == ["results_concurrency_4_gpus_4_ctx_4_gen_0.json"]


def test_sweep_without_a_served_model_runs_nothing(
    tmp_path, tools, http_server, monkeypatch, capsys
):
    url = http_server(frontend())
    use_env(monkeypatch, sweep_env(tmp_path, tools, url))

    assert bench(["fixed-seq", "srt-sweep", "--logs-dir", str(tmp_path / "logs")]) == 1
    assert capsys.readouterr().err == f"ERROR: {url}/v1/models lists no served model\n"
    assert client_runs(tmp_path) == []
    assert not (tmp_path / "logs").exists()


def test_point_runs_the_callers_flags_under_the_shared_policy(tmp_path, tools, monkeypatch):
    use_env(monkeypatch, {"PATH": str(tools), "FAKE_CLIENT_LOG": str(tmp_path / "client.log")})
    result = tmp_path / "results" / "stem_c1_gpus_16_ctx_8_gen_8.json"
    result.parent.mkdir()

    assert bench(point_argv(result)) == 0
    [run] = client_runs(tmp_path)
    assert options(run["argv"]) == {
        **POLICY,
        "--model": "deepseek-ai/DeepSeek-V4-Pro",
        "--backend": "openai",
        "--base-url": "http://0.0.0.0:8080",
        "--tokenizer": "/models",
        "--random-input-len": "8192",
        "--random-output-len": "1024",
        "--random-range-ratio": "0.8",
        "--num-prompts": "16",
        "--max-concurrency": "1",
        "--num-warmups": "2",
        "--result-dir": str(result.parent),
        "--result-filename": result.name,
    }
    assert result.is_file()


@pytest.mark.parametrize(
    ("sent", "rc"), [(signal.SIGTERM, 143), (signal.SIGHUP, 129)], ids=["TERM", "HUP"]
)
def test_point_relays_a_signal_and_an_interrupted_run_never_passes(tmp_path, tools, sent, rc):
    trapped = tmp_path / "trapped"
    process = subprocess.Popen(
        [sys.executable, "-m", "infx.bench", *point_argv(tmp_path / "result.json")],
        env={
            "PATH": str(tools), "PYTHONPATH": str(REPO_ROOT), "PYTHONDONTWRITEBYTECODE": "1",
            "FAKE_CLIENT_LOG": str(tmp_path / "client.log"), "FAKE_CLIENT_TRAP": str(trapped),
        },
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )  # fmt: skip
    try:
        deadline = time.monotonic() + 10
        while not client_runs(tmp_path):
            assert time.monotonic() < deadline, "the client did not start"
            time.sleep(0.02)
        process.send_signal(sent)
        _, stderr = process.communicate(timeout=30)
    finally:
        process.kill()

    # The client exited 0 on the relayed signal, but an interrupted run never passes.
    assert process.returncode == rc, stderr
    assert trapped.read_text() == sent.name
