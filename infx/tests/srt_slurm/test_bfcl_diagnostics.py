"""BFCL callback diagnostics retain failures and their evidence."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("mode,eval_exit,diagnostic_exit,expected", [
    ("kimi-metrics", 7, 0, 7), ("kimi-metrics", 0, 22, 22),
    ("kimi-metrics", 0, 0, 0), ("native-cpu-restore", 0, 9, 9),
    ("native-cpu-restore", 0, 0, 0), ("native-cpu-restore", 7, 0, 7),
])
def test_native_diagnostics_preserve_eval_failure_and_archive_evidence(
    tmp_path, mode, eval_exit, diagnostic_exit, expected
):
    workspace = tmp_path / "repo"
    scripts = workspace / "benchmarks/single_node"
    scripts.mkdir(parents=True)
    shutil.copyfile(ROOT / "benchmarks/single_node/srt_eval.sh", scripts / "srt_eval.sh")
    # The callback owns diagnostics; the evaluator is an independent process boundary.
    (scripts.parent / "benchmark_lib.sh").write_text(
        'check_env_vars() { :; }\nrun_eval() { printf "evaluated" > eval-evidence; return "$EVAL_EXIT"; }\n'
    )
    binaries = tmp_path / "bin"
    binaries.mkdir()
    (binaries / "curl").write_text(
        '#!/bin/bash\nprintf "vllm:num_requests_running 0\\n" > bfcl_diagnostic_metrics.txt\nexit "$DIAGNOSTIC_EXIT"\n'
    )
    (binaries / "timeout").write_text(
        '#!/bin/bash\nprintf "%s\\n" "$@" > restore-arguments\n'
        'printf "{\\"verified\\":true}" > native_cpu_restore_report.json\nexit "$DIAGNOSTIC_EXIT"\n'
    )
    for p in binaries.iterdir():
        p.chmod(0o755)
    status = tmp_path / "status"
    env = {
        **os.environ, "PATH": f"{binaries}:{os.environ['PATH']}",
        "INFERENCEX_REPO_ROOT": str(workspace), "IS_AGENTIC": "1", "EVAL_ONLY": "true",
        "EVAL_FRAMEWORK": "bfcl", "IS_MULTINODE": "false", "MODEL": "test/model",
        "EVAL_EXIT": str(eval_exit), "DIAGNOSTIC_EXIT": str(diagnostic_exit),
    }
    result = subprocess.run(
        ["bash", str(scripts / "srt_eval.sh"), "http://localhost:9444", str(status), mode],
        env=env, capture_output=True, text=True,
    )
    assert result.returncode == expected, result.stderr
    assert status.read_text() == f"{expected}\n"
    assert (workspace / "eval-evidence").read_text() == "evaluated"
    if mode == "kimi-metrics":
        import tarfile
        with tarfile.open(workspace / "bfcl_diagnostic_metrics_artifacts.tar.gz") as archive:
            assert archive.extractfile("bfcl_diagnostic_metrics.txt").read() == b"vllm:num_requests_running 0\n"
    elif eval_exit == 0:
        args = (workspace / "restore-arguments").read_text().splitlines()
        assert args == [
            "600", "python3", str(workspace / "experimental/bfcl/verify_native_cpu_restore.py"),
            "--base-url", "http://127.0.0.1:9444", "--model", "test/model",
            "--output", str(workspace / "native_cpu_restore_report.json"),
        ]
    else:
        assert not (workspace / "restore-arguments").exists()
