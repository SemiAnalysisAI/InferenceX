"""Run the shared client against stubbed external benchmark/GPU processes."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CLIENT = ROOT / "benchmarks/single_node/srt_fixed_sequence.sh"


@pytest.fixture
def client_environment(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    benchmark = binaries / "python3"
    benchmark.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "pathlib.Path(os.environ['CAPTURE']).write_text(json.dumps(sys.argv[1:]))\n"
        "sys.exit(int(os.environ['CLIENT_EXIT']))\n"
    )
    benchmark.chmod(0o755)
    for name, body in {
        "pip3": "exit 0\n",
        "nvidia-smi": "printf 'timestamp,index,power.draw\\n'\n",
    }.items():
        binary = binaries / name
        binary.write_text(f"#!/bin/bash\n{body}")
        binary.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{binaries}:{os.environ['PATH']}",
        "MODEL": "test/model",
        "CONC": "3",
        "ISL": "128",
        "OSL": "64",
        "RANDOM_RANGE_RATIO": "0.5",
        "RESULT_FILENAME": "test-result",
        "RESULT_DIR": str(tmp_path),
        "SRT_FRONTEND_HOST": "10.2.3.4",
        "SRT_FRONTEND_PORT": "9444",
        "RUN_EVAL": "false",
        "EVAL_ONLY": "false",
        "GPU_MONITOR_INTERVAL": "2",
        "USE_CHAT_TEMPLATE": "false",
        "IS_AGENTIC": "0",
        "SCENARIO_TYPE": "fixed-seq-len",
        "CLIENT_EXIT": "0",
        "CAPTURE": str(tmp_path / "argv.json"),
    }
    for key in ("PROFILE", "INFERENCEX_SERVER_PID", "INFERENCEX_SERVER_STATE"):
        env.pop(key, None)
    return env


@pytest.mark.parametrize("exit_code,chat_template", [(0, "false"), (7, "false"), (0, "true")])
def test_native_endpoint_preserves_client_settings_and_failure(
    client_environment, exit_code, chat_template
):
    env = {**client_environment, "CLIENT_EXIT": str(exit_code), "USE_CHAT_TEMPLATE": chat_template}
    result = subprocess.run(
        ["bash", str(CLIENT)], env=env, capture_output=True, text=True
    )
    assert result.returncode == exit_code, result.stderr
    argv = json.loads(Path(env["CAPTURE"]).read_text())
    assert argv == [
        "-m",
        "infx.bench_serving.benchmark_serving",
        "--model",
        "test/model",
        "--backend",
        "vllm",
        "--base-url",
        "http://10.2.3.4:9444",
        "--dataset-name",
        "random",
        "--random-input-len",
        "128",
        "--random-output-len",
        "64",
        "--random-range-ratio",
        "0.5",
        "--num-prompts",
        "30",
        "--max-concurrency",
        "3",
        "--request-rate",
        "inf",
        "--ignore-eos",
        "--save-result",
        "--num-warmups",
        "6",
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--result-dir",
        env["RESULT_DIR"],
        "--result-filename",
        "test-result.json",
    ] + (["--use-chat-template"] if chat_template == "true" else [])
    assert (
        (Path(env["RESULT_DIR"]) / "gpu_metrics.csv")
        .read_text()
        .startswith("timestamp")
    )


@pytest.mark.parametrize(
    ("key", "value", "error"),
    [
        ("MODEL", None, "MODEL"),
        ("GPU_MONITOR_INTERVAL", None, "GPU_MONITOR_INTERVAL"),
        ("USE_CHAT_TEMPLATE", "yes", "USE_CHAT_TEMPLATE must be true or false"),
        ("CONC", "0", "CONC must be a positive integer"),
        ("RUN_EVAL", "yes", "RUN_EVAL must be true or false"),
        ("EVAL_ONLY", "yes", "EVAL_ONLY must be true or false"),
    ],
)
def test_invalid_runtime_inputs_fail_before_the_client(
    client_environment, key, value, error
):
    env = dict(client_environment)
    if value is None:
        env.pop(key)
    else:
        env[key] = value
    result = subprocess.run(
        ["bash", str(CLIENT)], env=env, capture_output=True, text=True
    )
    assert result.returncode != 0
    assert error in result.stdout + result.stderr
    assert not Path(env["CAPTURE"]).exists()


def test_legacy_client_keeps_its_local_endpoint(client_environment):
    env = client_environment
    result = subprocess.run(
        [
            "bash",
            "-c",
            """source "$1/benchmarks/benchmark_lib.sh"
run_benchmark_serving --model test/model --port 8888 --backend vllm \\
  --input-len 128 --output-len 64 --random-range-ratio 0.5 \\
  --num-prompts 30 --max-concurrency 3 --result-filename old --result-dir "$RESULT_DIR"
""",
            "bash",
            str(ROOT),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    argv = json.loads(Path(env["CAPTURE"]).read_text())
    assert argv[argv.index("--base-url") + 1] == "http://0.0.0.0:8888"


@pytest.mark.parametrize("eval_exit", [0, 7])
def test_native_post_eval_preserves_results_topology_and_failure(client_environment, tmp_path, eval_exit):
    workspace = tmp_path / "repo"
    scripts = workspace / "benchmarks/single_node"
    scripts.mkdir(parents=True)
    shutil.copyfile(ROOT / "benchmarks/benchmark_lib.sh", scripts.parent / "benchmark_lib.sh")
    shutil.copyfile(ROOT / "benchmarks/single_node/srt_eval.sh", scripts / "srt_eval.sh")
    python = tmp_path / "bin/python3"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "assert args[:2] == ['-m', 'lm_eval']\n"
        "pathlib.Path(os.environ['CAPTURE']).write_text(json.dumps(args))\n"
        "output = pathlib.Path(args[args.index('--output_path') + 1])\n"
        "output.mkdir(parents=True, exist_ok=True)\n"
        "(output / 'results_fixture.json').write_text('{\"score\":0.75}')\n"
        "sys.exit(int(os.environ['CLIENT_EXIT']))\n"
    )
    env = {
        **client_environment, "CLIENT_EXIT": str(eval_exit), "MODEL_NAME": "served-model",
        "TP": "4", "EP_SIZE": "4", "DP_ATTENTION": "true", "IS_MULTINODE": "false",
        "MAX_MODEL_LEN": "8192", "EVAL_MAX_MODEL_LEN": "8192", "OPENAI_API_KEY": "EMPTY",
        "INFERENCEX_LM_EVAL_RUNTIME_READY": "true", "EVAL_ONLY": "true", "RUN_EVAL": "true",
        "EVAL_RESULT_DIR": str(tmp_path / "eval-output"), "FRAMEWORK": "sglang", "PRECISION": "fp8",
    }
    status = tmp_path / "eval-status"
    result = subprocess.run(
        ["bash", str(scripts / "srt_eval.sh"), "http://localhost:9444", str(status)],
        env=env, capture_output=True, text=True,
    )
    assert result.returncode == eval_exit, result.stderr
    assert status.read_text() == f"{eval_exit}\n"
    assert json.loads((workspace / "results_fixture.json").read_text()) == {"score": 0.75}
    metadata = json.loads((workspace / "meta_env.json").read_text())
    assert (metadata["tp"], metadata["ep"], metadata["dp_attention"], metadata["conc"]) == (4, 4, True, 3)
    argv = json.loads(Path(env["CAPTURE"]).read_text())
    model_args = argv[argv.index("--model_args") + 1]
    assert "model=served-model,base_url=http://0.0.0.0:9444/v1/chat/completions" in model_args
    assert "num_concurrent=3" in model_args
