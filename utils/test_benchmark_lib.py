import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_LIB = REPO_ROOT / "benchmarks" / "benchmark_lib.sh"


def build_agentic_replay_cmd(
    *,
    max_model_len: str = "",
    aiperf_max_context_length: str = "",
) -> subprocess.CompletedProcess[str]:
    env = {
        "PATH": os.environ["PATH"],
        "SCENARIO_TYPE": "agentic-coding",
        "KV_OFFLOADING": "none",
        "MAX_MODEL_LEN": max_model_len,
        "AIPERF_MAX_CONTEXT_LENGTH": aiperf_max_context_length,
    }
    script = r'''
source "$1"
DURATION=1200
AIPERF_CLI=aiperf
PORT=8000
MODEL=test-model
CONC=4
AIPERF_LIVE_FAILED_REQUEST_THRESHOLD=0.01
AIPERF_TRACE_IDLE_GAP_CAP_SECONDS=10
FRAMEWORK=vllm
TRACE_SOURCE_FLAG="--public-dataset test-dataset"
build_replay_cmd /tmp/result || exit $?
printf '%s\n' "$REPLAY_CMD"
'''
    return subprocess.run(
        ["bash", "-c", script, "bash", str(BENCHMARK_LIB)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_agentic_replay_ignores_generated_server_context_limit_by_default() -> None:
    result = build_agentic_replay_cmd(max_model_len="65536")

    assert result.returncode == 0, result.stderr
    assert "--max-context-length" not in result.stdout


def test_agentic_replay_accepts_explicit_aiperf_context_limit() -> None:
    result = build_agentic_replay_cmd(
        max_model_len="65536",
        aiperf_max_context_length="163840",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("--max-context-length 163840") == 1


def test_agentic_replay_rejects_invalid_aiperf_context_limit() -> None:
    result = build_agentic_replay_cmd(aiperf_max_context_length="invalid")

    assert result.returncode != 0
    assert "max context length must be a positive integer" in result.stderr
