"""Exercise the fixed-sequence client's DeepSeek-V4 encoder forwarding."""

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_dsv4_client_forwards_encoder_and_workload(tmp_path: Path) -> None:
    harness = tmp_path / "harness.sh"
    harness.write_text(
        """
source() {
    if [[ "$1" == */benchmark_lib.sh && "$2" != --validation-only ]]; then
        start_gpu_monitor() { :; }
        stop_gpu_monitor() { :; }
        run_benchmark_serving() { printf '%s\\n' "$@" > "$CAPTURE"; }
    else
        builtin source "$@"
    fi
}
pip3() { :; }
"""
    )
    capture = tmp_path / "arguments"
    result = subprocess.run(
        ["bash", str(ROOT / "benchmarks/single_node/srt_fixed_sequence.sh"), "--dsv4"],
        env={
            **os.environ,
            "BASH_ENV": str(harness),
            "CAPTURE": str(capture),
            "INFERENCEX_REPO_ROOT": str(ROOT),
            "MODEL": "test/model",
            "FRAMEWORK": "sglang",
            "CONC": "2",
            "ISL": "256",
            "OSL": "64",
            "RANDOM_RANGE_RATIO": "0.8",
            "RESULT_FILENAME": "point",
            "RESULT_DIR": str(tmp_path),
            "SRT_FRONTEND_HOST": "127.0.0.1",
            "SRT_FRONTEND_PORT": "8000",
            "RUN_EVAL": "false",
            "EVAL_ONLY": "false",
            "GPU_MONITOR_INTERVAL": "3",
            "USE_CHAT_TEMPLATE": "true",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert capture.read_text().splitlines() == [
        "--model",
        "test/model",
        "--port",
        "8000",
        "--base-url",
        "http://127.0.0.1:8000",
        "--backend",
        "vllm",
        "--input-len",
        "256",
        "--output-len",
        "64",
        "--random-range-ratio",
        "0.8",
        "--num-prompts",
        "20",
        "--max-concurrency",
        "2",
        "--result-filename",
        "point",
        "--result-dir",
        str(tmp_path),
        "--dsv4",
        "--use-chat-template",
    ]
