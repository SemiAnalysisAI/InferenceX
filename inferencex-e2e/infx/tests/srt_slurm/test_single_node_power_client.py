"""Run the fixed-sequence entrypoint with its external client and sampler stubbed."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("native", [False, True])
def test_client_uses_only_the_selected_collector_and_writes_native_window(tmp_path, native):
    root = Path(__file__).resolve().parents[3]
    scripts = tmp_path / "benchmarks"
    (scripts / "single_node").mkdir(parents=True)
    script = scripts / "single_node/srt_fixed_sequence.sh"
    shutil.copyfile(root / "benchmarks/single_node/srt_fixed_sequence.sh", script)
    (scripts / "benchmark_lib.sh").write_text('''
check_env_vars() { :; }
start_gpu_monitor() { printf 'direct sampler' > "$RESULT_DIR/gpu_metrics.csv"; }
stop_gpu_monitor() { :; }
run_benchmark_serving() {
  printf '%s' '{"benchmark_start_time_unix": 10, "benchmark_end_time_unix": 12, "duration": 2}' > "$RESULT_DIR/$RESULT_FILENAME.json"
}
''')
    binaries = tmp_path / "bin"
    binaries.mkdir()
    pip = binaries / "pip3"
    pip.write_text("#!/bin/sh\nexit 0\n")
    pip.chmod(0o755)
    logs = tmp_path / "logs"
    windows = logs / "power/windows"
    windows.mkdir(parents=True)
    env = {
        **os.environ, "PATH": f"{binaries}:{os.environ['PATH']}",
        "INFERENCEX_REPO_ROOT": str(root), "MODEL": "test/model", "CONC": "4",
        "ISL": "8192", "OSL": "1024", "RANDOM_RANGE_RATIO": "0.8",
        "RESULT_FILENAME": "point", "RESULT_DIR": str(logs), "SRT_FRONTEND_HOST": "host",
        "SRT_FRONTEND_PORT": "8000", "RUN_EVAL": "false", "EVAL_ONLY": "false",
        "GPU_MONITOR_INTERVAL": "1", "USE_CHAT_TEMPLATE": "false", "FRAMEWORK": "sglang",
    }
    env.pop("SRT_MEASUREMENT_WINDOW_DIR", None)
    if native:
        env["SRT_MEASUREMENT_WINDOW_DIR"] = str(windows)
    subprocess.run(["bash", str(script)], env=env, check=True, capture_output=True)
    assert (logs / "gpu_metrics.csv").exists() is not native
    if native:
        window = json.loads((windows / "point.json").read_text())
        assert (window["result_path"], window["concurrency"], window["duration"]) == (
            "point.json", 4, 2,
        )
        assert (window["benchmark_start_time_unix"], window["benchmark_end_time_unix"]) == (10, 12)
    else:
        assert not (windows / "point.json").exists()
