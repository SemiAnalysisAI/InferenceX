"""Execute the fixed-sequence clients with external services replaced by local stubs."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]

LIBRARY = """check_env_vars() { :; }
start_gpu_monitor() { :; }
stop_gpu_monitor() { :; }
run_benchmark_serving() { python3 -m captured_client "$@"; }
"""

PYTHON_STUB = """#!{python}
import json
import os
import sys

if sys.argv[1] == "-c":
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
with open(os.environ["CLIENT_CALLS"], "a") as log:
    log.write(json.dumps(sys.argv[3:]) + "\\n")
"""


@pytest.mark.parametrize("lane,concurrencies", [("single_node", [3]), ("multi_node", [3, 7])])
@pytest.mark.parametrize("stale_settings", [False, True], ids=["omitted", "stale"])
def test_fixed_sequence_policy(tmp_path, lane, concurrencies, stale_settings):
    script_dir = tmp_path / "benchmarks" / lane
    script_dir.mkdir(parents=True)
    script = script_dir / "srt_fixed_sequence.sh"
    shutil.copyfile(ROOT / "benchmarks" / lane / script.name, script)
    (script_dir.parent / "benchmark_lib.sh").write_text(LIBRARY)

    binaries = tmp_path / "bin"
    binaries.mkdir()
    stubs = {
        "python3": PYTHON_STUB.format(python=sys.executable),
        "pip3": "#!/bin/sh\nexit 0\n",
        "mkdir": "#!/bin/sh\nexit 0\n",
        "curl": '#!/bin/sh\nprintf \'%s\\n\' \'{"data":[{"id":"served/model"}]}\'\n',
    }
    for name, content in stubs.items():
        binary = binaries / name
        binary.write_text(content)
        binary.chmod(0o755)

    calls_path = tmp_path / "calls.jsonl"
    environment = {
        "PATH": f"{binaries}:/usr/bin:/bin",
        "CLIENT_CALLS": str(calls_path),
        "INFERENCEX_REPO_ROOT": str(tmp_path),
        "MODEL": "checkpoint/model",
        "CONC": "3",
        "CONC_LIST": "3 7",
        "ISL": "8192",
        "OSL": "1024",
        "RESULT_FILENAME": "result.json",
        "RESULT_DIR": str(tmp_path),
        "SRT_FRONTEND_HOST": "fixture-host",
        "SRT_FRONTEND_PORT": "8123",
        "RUN_EVAL": "false",
        "EVAL_ONLY": "false",
        "GPU_MONITOR_INTERVAL": "2",
        "FRAMEWORK": "sglang",
        "CLIENT_BACKEND": "openai",
        "PREFILL_NUM_WORKERS": "1",
        "PREFILL_TP": "4",
        "DECODE_NUM_WORKERS": "2",
        "DECODE_TP": "4",
    }
    if stale_settings:
        environment.update(USE_CHAT_TEMPLATE="false", RANDOM_RANGE_RATIO="0.2")

    result = subprocess.run(
        ["bash", str(script)], env=environment, capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in calls_path.read_text().splitlines()]
    assert len(calls) == len(concurrencies)
    for arguments, concurrency in zip(calls, concurrencies, strict=True):
        assert arguments.count("--use-chat-template") == 1
        assert arguments[arguments.index("--random-range-ratio") + 1] == "0.8"
        assert arguments[arguments.index("--max-concurrency") + 1] == str(concurrency)
        assert arguments[arguments.index("--num-prompts") + 1] == str(concurrency * 10)
