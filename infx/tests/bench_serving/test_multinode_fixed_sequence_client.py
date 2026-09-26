"""Exercise the multi-node launcher across checkout and process boundaries."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = ROOT / "benchmarks/multi_node/srt_fixed_sequence.sh"


def launcher_env(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    # The model service and container-owned /logs directory are external.
    for name, body in {
        "curl": """printf '%s\\n' '{"data":[{"id":"served model"}]}'""",
        "mkdir": "exit 0",
    }.items():
        path = binaries / name
        path.write_text(f"#!/usr/bin/env bash\n{body}\n")
        path.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{binaries}:{Path(sys.executable).parent}:{os.environ['PATH']}",
        "ISL": "128", "OSL": "32",
        "SRT_FRONTEND_HOST": "127.0.0.1", "SRT_FRONTEND_PORT": "8765",
        "CONC_LIST": "2 4",
        "PREFILL_NUM_WORKERS": "2", "PREFILL_TP": "4",
        "DECODE_NUM_WORKERS": "1", "DECODE_TP": "8",
        "CLIENT_BACKEND": "openai-chat", "USE_CHAT_TEMPLATE": "true",
        "TOKENIZER": "tokenizer with spaces", "RANDOM_RANGE_RATIO": "0.5",
    }
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONSAFEPATH", None)
    env.pop("SRT_MEASUREMENT_WINDOW_DIR", None)
    return binaries, env


@pytest.mark.parametrize("inherited_path", [False, True])
def test_multinode_launcher_rejects_bad_lengths_despite_cwd_shadow(tmp_path, inherited_path):
    _, env = launcher_env(tmp_path)
    shadow = tmp_path / "infx"
    shadow.mkdir()
    (shadow / "__init__.py").write_text("raise RuntimeError('wrong checkout loaded')\n")
    if inherited_path:
        env["PYTHONPATH"] = str(tmp_path)
    env["ISL"] = "not-an-integer"

    result = subprocess.run(
        ["bash", str(LAUNCHER)], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=30,
    )

    assert result.returncode == 2, result.stderr
    assert "--random-input-len: invalid int value: 'not-an-integer'" in result.stderr
    assert "wrong checkout loaded" not in result.stderr


@pytest.mark.parametrize("client_exit", [0, 7])
def test_multinode_launcher_forwards_workload_and_stops_on_client_failure(tmp_path, client_exit):
    binaries, env = launcher_env(tmp_path)
    capture = tmp_path / "calls.jsonl"
    python = binaries / "python3"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        f"if sys.argv[1] == '-c': os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])\n"
        "with open(os.environ['CAPTURE'], 'a') as out:\n"
        "    out.write(json.dumps({'args': sys.argv[1:], 'path': os.environ['PYTHONPATH']}) + '\\n')\n"
        "sys.exit(int(os.environ['CLIENT_EXIT']))\n"
    )
    python.chmod(0o755)
    env.update(CAPTURE=str(capture), CLIENT_EXIT=str(client_exit), PYTHONPATH="/retained path")

    result = subprocess.run(
        ["bash", str(LAUNCHER)], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=30,
    )

    assert result.returncode == client_exit, result.stderr
    calls = [json.loads(line) for line in capture.read_text().splitlines()]
    assert len(calls) == (2 if client_exit == 0 else 1)
    for call, concurrency, warmups, prompts in zip(calls, ["2", "4"], ["4", "8"], ["20", "40"]):
        args = call["args"]
        assert args[:3] == ["-P", "-m", "infx.bench_serving.benchmark_serving"]
        expected = {
            "--model": "served model", "--tokenizer": "tokenizer with spaces",
            "--backend": "openai-chat", "--endpoint": "/v1/chat/completions",
            "--random-input-len": "128", "--random-output-len": "32",
            "--random-range-ratio": "0.5", "--max-concurrency": concurrency,
            "--num-warmups": warmups, "--num-prompts": prompts,
            "--result-dir": "/logs/sa-bench_isl_128_osl_32",
            "--result-filename": f"results_concurrency_{concurrency}_gpus_16_ctx_8_gen_8.json",
        }
        for option, value in expected.items():
            assert args[args.index(option) + 1] == value
        assert "--use-chat-template" in args
        paths = call["path"].split(":")
        assert Path(paths[0]).resolve() == ROOT
        assert paths[1:] == ["/retained path"]
