"""Exercise the custom benchmark shell entrypoint without launching a client."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


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
