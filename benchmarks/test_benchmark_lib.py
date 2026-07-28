import shlex
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_LIB = REPO_ROOT / "benchmarks" / "benchmark_lib.sh"


def build_replay_command(burst_phase_starts: str | None) -> subprocess.CompletedProcess[str]:
    burst_setup = (
        "unset AIPERF_BURST_PHASE_STARTS"
        if burst_phase_starts is None
        else f"export AIPERF_BURST_PHASE_STARTS={shlex.quote(burst_phase_starts)}"
    )
    script = f"""
set -euo pipefail
export IS_AGENTIC=1
export KV_OFFLOADING=none
source {shlex.quote(str(BENCHMARK_LIB))}
export AIPERF_CLI=aiperf
export PORT=8000
export MODEL=example/model
export CONC=32
export DURATION=3600
export AIPERF_FAILED_REQUEST_THRESHOLD=0.10
export FRAMEWORK=dynamo-vllm
export AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING=0
export TRACE_SOURCE_FLAG="--public-dataset example"
{burst_setup}
build_replay_cmd /tmp/aiperf-test
printf '%s\\n' "$REPLAY_CMD"
"""
    return subprocess.run(
        ["bash", "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )


def test_build_replay_command_defaults_to_burst_phase_starts() -> None:
    result = build_replay_command(None)

    assert result.returncode == 0, result.stderr
    assert "--burst-phase-starts" in result.stdout


def test_build_replay_command_can_restore_spread_phase_starts() -> None:
    result = build_replay_command("0")

    assert result.returncode == 0, result.stderr
    assert "--burst-phase-starts" not in result.stdout


def test_build_replay_command_rejects_invalid_burst_value() -> None:
    result = build_replay_command("yes")

    assert result.returncode != 0
    assert "AIPERF_BURST_PHASE_STARTS must be 0 or 1" in result.stderr
