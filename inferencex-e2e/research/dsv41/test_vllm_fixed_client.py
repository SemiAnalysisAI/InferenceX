import json
import os
import subprocess
import sys
from pathlib import Path


def test_vllm_fixed_client_uses_completion_backend_and_exact_workload(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    python = binaries / "python3"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['ARGV_FILE']).write_text(json.dumps(sys.argv[1:]))\n"
    )
    pip = binaries / "pip3"
    pip.write_text("#!/bin/sh\nexit 0\n")
    python.chmod(0o755)
    pip.chmod(0o755)
    argv_file = tmp_path / "argv.json"
    environment = {
        **os.environ,
        "PATH": f"{binaries}:/usr/bin:/bin",
        "ARGV_FILE": str(argv_file),
        "MODEL": "test-model",
        "CONC": "2",
        "ISL": "8192",
        "OSL": "256",
        "RANDOM_RANGE_RATIO": "1.0",
        "RESULT_FILENAME": "test-result",
        "RESULT_DIR": str(tmp_path),
        "SRT_FRONTEND_HOST": "127.0.0.1",
        "SRT_FRONTEND_PORT": "12345",
        "RUN_EVAL": "false",
        "EVAL_ONLY": "false",
        "GPU_MONITOR_INTERVAL": "1",
        "USE_CHAT_TEMPLATE": "true",
        "FRAMEWORK": "vllm",
        "GPU_METRICS_CSV": str(tmp_path / "metrics.csv"),
        "PROFILE": "0",
    }
    root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [
            "bash",
            str(root / "benchmarks/single_node/srt_fixed_sequence.sh"),
            "--trust-remote-code",
            "--dsv4",
        ],
        env=environment,
        check=True,
        capture_output=True,
    )
    args = json.loads(argv_file.read_text())
    for flag, value in [
        ("--backend", "vllm"),
        ("--random-input-len", "8192"),
        ("--random-output-len", "256"),
        ("--num-prompts", "20"),
        ("--num-warmups", "4"),
        ("--base-url", "http://127.0.0.1:12345"),
    ]:
        assert args[args.index(flag) + 1] == value
    assert "--use-chat-template" in args and "--dsv4" in args
