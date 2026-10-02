"""Fixed-sequence lanes against a stub client, pip3, nvidia-smi, and a local frontend."""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from infx.bench.__main__ import main as bench
from infx.tests.bench.stubs import executable

REPO_ROOT = Path(__file__).resolve().parents[3]
FINAL_SAMPLE = "2026/07/23 12:00:11.000, 0, 500.00 W, 65, 1000, 1000, 90 %, 10 %"
# Client flags every lane passes, whatever the point.
POLICY = {
    "--dataset-name": "random",
    "--request-rate": "inf",
    "--ignore-eos": True,
    "--save-result": True,
    "--percentile-metrics": "ttft,tpot,itl,e2el",
}

# Records each client run and writes the result file the real client would.
FAKE_CLIENT = """
import json, os, sys
from pathlib import Path

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
if value("--max-concurrency") == os.environ.get("FAKE_CLIENT_FAIL_CONC"):
    sys.exit(3)
result.write_text(json.dumps(
    {"benchmark_start_time_unix": 100.0, "benchmark_end_time_unix": 160.0, "duration": 60.0}
))
"""


@pytest.fixture
def tools(tmp_path: Path) -> Path:
    """PATH with a stub benchmark client behind python3, a recording pip3, and nvidia-smi."""
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
        "pip3": f"printf '%s\\n' \"$*\" >> {shlex.quote(str(tmp_path / 'pip3.log'))}\n",
        "nvidia-smi": f"""
case "$*" in
    *" -l 1") printf 'timestamp, index, power.draw [W]\\n'; exec sleep 30 ;;
    *noheader*) printf '%s\\n' {shlex.quote(FINAL_SAMPLE)} ;;
esac
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
        "GPU_MONITOR_INTERVAL": "1",
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


@pytest.mark.parametrize(
    ("framework", "chat_template", "args", "flags"),
    [
        ("sglang", "true", [], {"--backend": "vllm", "--use-chat-template": True}),
        ("trt", "false", ["--trust-remote-code"], {"--backend": "openai", "--trust-remote-code": True}),
    ],
)
def test_single_node_shim_runs_one_point_under_the_monitor(
    tmp_path, tools, framework, chat_template, args, flags
):
    env = single_node_env(tmp_path, tools, FRAMEWORK=framework, USE_CHAT_TEMPLATE=chat_template)

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
    # The sampler was already writing when the client started and stopped after it.
    assert run["monitored"]
    # The shim sets PYTHONSAFEPATH for infx.bench only; Python-script tools break under it.
    assert run["safe_path"] is None
    assert (logs / "gpu_metrics.csv").read_text().endswith(FINAL_SAMPLE + "\n")
    assert (tmp_path / "pip3.log").read_text() == (
        "install --break-system-packages sentencepiece datasets pandas\n"
    )


@pytest.mark.parametrize(
    ("overrides", "returncode", "reported"),
    [
        ({"EVAL_ONLY": "true"}, 0, "EVAL_ONLY mode: skipping throughput benchmark\n"),
        ({"MODEL": None, "CONC": ""}, 1, "not set:\n  - MODEL\n  - CONC\n"),
        ({"FRAMEWORK": "vllm"}, 1, "ERROR: unsupported fixed-sequence FRAMEWORK: vllm\n"),
        ({"USE_CHAT_TEMPLATE": "yes"}, 1, "ERROR: USE_CHAT_TEMPLATE must be true or false, got 'yes'\n"),
        ({"RESULT_DIR": "/nonexistent/logs"}, 1, "ERROR: RESULT_DIR must be an existing"),
    ],
    ids=["eval-only", "missing", "unsupported-framework", "malformed-flag", "no-result-dir"],
)
def test_single_node_point_runs_nothing_when_eval_only_or_misconfigured(
    tmp_path, tools, monkeypatch, capsys, overrides, returncode, reported
):
    use_env(monkeypatch, single_node_env(tmp_path, tools, **overrides))

    assert bench(["fixed-seq", "srt-single"]) == returncode
    assert reported in "".join(capsys.readouterr())
    assert client_runs(tmp_path) == []
    assert not (tmp_path / "pip3.log").exists()
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
        "--endpoint": "/v1/completions",
        "--base-url": url,
        "--tokenizer": "/model",
        "--random-input-len": "1024",
        "--random-output-len": "128",
        "--random-range-ratio": "0.8",
        "--random-num-workers": "1",
        "--num-prompts": "40",
        "--max-concurrency": "4",
        "--num-warmups": "8",
        "--disable-tqdm": True,
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
