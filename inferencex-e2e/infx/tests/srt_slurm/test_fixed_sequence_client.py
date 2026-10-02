"""Exercise the custom benchmark shell entrypoint without launching a client."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _run_native_fixed_client(tmp_path, script):
    """Run the real wrapper and window writer with only external clients stubbed."""
    root = Path(__file__).resolve().parents[3]
    binaries = tmp_path / "bin"
    binaries.mkdir()
    logs = tmp_path / "logs"
    windows = logs / "power/windows"
    windows.mkdir(parents=True)
    scripts = {
        "python3": (
            f"#!{sys.executable}\n"
            "import json, os, sys\nfrom pathlib import Path\n"
            "if sys.argv[1:3] == ['-m', 'infx.bench_serving.benchmark_serving']:\n"
            "    result = Path(sys.argv[sys.argv.index('--result-dir') + 1])\n"
            "    result /= sys.argv[sys.argv.index('--result-filename') + 1]\n"
            "    result.write_text(json.dumps({'benchmark_start_time_unix': 1000.0, "
            "'benchmark_end_time_unix': 1060.0, 'duration': 60.0}))\n"
            "else:\n"
            f"    os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])\n"
        ),
        "pip3": "#!/bin/sh\nexit 0\n",
        "nvidia-smi": '#!/bin/sh\nprintf called > "$MONITOR_CALLED_FILE"\n',
    }
    for name, source in scripts.items():
        binary = binaries / name
        binary.write_text(source)
        binary.chmod(0o755)
    env = {
        "PATH": f"{binaries}{os.pathsep}/usr/bin:/bin",
        "PYTHONPATH": str(root),
        "PYTHONPYCACHEPREFIX": str(tmp_path / "pycache"),
        "MONITOR_CALLED_FILE": str(tmp_path / "monitor-called"),
        "MODEL": "test/model", "CONC": "4", "ISL": "128", "OSL": "64",
        "RANDOM_RANGE_RATIO": "0.5", "RESULT_FILENAME": "native-point",
        "RESULT_DIR": str(logs), "SRT_MEASUREMENT_WINDOW_DIR": str(windows),
        "SRT_FRONTEND_HOST": "router", "SRT_FRONTEND_PORT": "8123",
        "RUN_EVAL": "false", "EVAL_ONLY": "false", "GPU_MONITOR_INTERVAL": "3",
        "USE_CHAT_TEMPLATE": "false", "FRAMEWORK": "sglang", "IS_AGENTIC": "0",
    }
    return subprocess.run(
        ["bash", str(script)], env=env, cwd=tmp_path, capture_output=True, text=True
    )


def test_native_fixed_client_publishes_formal_window_without_local_sampler(tmp_path):
    script = Path(__file__).resolve().parents[3] / "benchmarks/single_node/srt_fixed_sequence.sh"
    completed = _run_native_fixed_client(tmp_path, script)
    assert completed.returncode == 0, completed.stderr
    window = json.loads((tmp_path / "logs/power/windows/native-point.json").read_text())
    assert window == {
        "schema_version": 1, "benchmark_type": "custom", "clock_source": "head_node_unix_clock",
        "status": "completed", "reason": None, "result_path": "native-point.json",
        "concurrency": 4, "benchmark_start_time_unix": 1000.0,
        "benchmark_end_time_unix": 1060.0, "duration": 60.0,
    }
    assert not (tmp_path / "monitor-called").exists()
    assert not (tmp_path / "logs/gpu_metrics.csv").exists()


@pytest.mark.parametrize("explicit", [True, False])
def test_client_model_discovery_and_explicit_request_count(tmp_path, explicit):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    scripts = {
        "python3": (
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "if sys.argv[1:2] == ['-c']:\n"
            f"    os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])\n"
            "Path(os.environ['CLIENT_ARGS_FILE']).write_text(json.dumps(sys.argv[1:]))\n"
        ),
        "curl": (
            "#!/bin/sh\n"
            'printf called > "$CURL_CALLED_FILE"\n'
            "printf '%s\\n' '{\"data\":[{\"id\":\"discovered-model\"}]}'\n"
        ),
        # No real client writes results; avoid creating its container-only /logs directory.
        "mkdir": "#!/bin/sh\nexit 0\n",
    }
    for name, source in scripts.items():
        binary = binaries / name
        binary.write_text(source)
        binary.chmod(0o755)
    args_file = tmp_path / "client-args.json"
    curl_called = tmp_path / "curl-called"
    env = {
        "PATH": f"{binaries}{os.pathsep}{os.environ['PATH']}",
        "CLIENT_ARGS_FILE": str(args_file),
        "CURL_CALLED_FILE": str(curl_called),
        "ISL": "1024",
        "OSL": "1024",
        "SRT_FRONTEND_HOST": "router",
        "SRT_FRONTEND_PORT": "8123",
        "CONC_LIST": "1",
        "PREFILL_NUM_WORKERS": "1",
        "PREFILL_TP": "8",
        "DECODE_NUM_WORKERS": "1",
        "DECODE_TP": "8",
        "CLIENT_BACKEND": "openai-chat",
        "USE_CHAT_TEMPLATE": "true",
        "SERVED_MODEL_NAME": "launcher-model-not-client-override",
    }
    if explicit:
        env.update(BENCHMARK_SERVED_MODEL_NAME="glm5", NUM_PROMPTS="16")
    script = Path(__file__).resolve().parents[3] / "benchmarks/multi_node/srt_fixed_sequence.sh"
    subprocess.run(["bash", str(script)], env=env, cwd=tmp_path, check=True, capture_output=True)
    args = json.loads(args_file.read_text())
    assert args[:2] == ["-m", "infx.bench_serving.benchmark_serving"]
    assert args[args.index("--model") + 1] == ("glm5" if explicit else "discovered-model")
    assert args[args.index("--num-prompts") + 1] == ("16" if explicit else "10")
    assert args[args.index("--endpoint") + 1] == "/v1/chat/completions"
    assert "--use-chat-template" in args
    assert curl_called.exists() is not explicit
