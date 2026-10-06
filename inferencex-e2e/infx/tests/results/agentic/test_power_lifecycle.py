"""Shell-contract tests for the shared AgentX power lifecycle."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
BENCHMARK_LIB = REPO_ROOT / "benchmarks" / "benchmark_lib.sh"


def _run_lifecycle(
    tmp_path: Path,
    *,
    replay_rc: int = 0,
    is_multinode: bool = False,
    enable_power: bool = True,
    require_power: bool = False,
    formal_multinode_power: bool = False,
    real_power_adapter: bool = False,
    missing_power_env: str | None = None,
) -> subprocess.CompletedProcess[str]:
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    event_log = tmp_path / "events.log"
    formal_window_dir = str(tmp_path / "power/windows") if formal_multinode_power else ""
    script = f"""
source {str(BENCHMARK_LIB)!r}
fake_replay() {{
    printf 'replay\n' >> {str(event_log)!r}
    return {replay_rc}
}}
write_agentic_result_json() {{
    printf 'aggregate\n' >> {str(event_log)!r}
    printf '{{}}\n' > "$AGENTIC_OUTPUT_DIR/$RESULT_FILENAME.json"
}}
fake_python() {{
    case "$*" in
        *infx.results.agentic.power_adapter*)
            printf 'adapter:%s\n' "$*" >> {str(event_log)!r}
            if [ {'1' if real_power_adapter else '0'} = 1 ]; then
                PYTHONPATH={str(REPO_ROOT)!r} {sys.executable!r} "$@"
                return $?
            fi
            ;;
        *validate_agentic_result*)
            printf 'validate\n' >> {str(event_log)!r}
            ;;
        *)
            printf 'analyze\n' >> {str(event_log)!r}
            ;;
    esac
    return 0
}}
validate_required_agentic_server_metrics() {{
    printf 'server-metrics\n' >> {str(event_log)!r}
}}
trap 'printf "parent-exit\\n" >> {str(event_log)!r}' EXIT
REPLAY_CMD=fake_replay
AIPERF_PYTHON=fake_python
INFMAX_CONTAINER_WORKSPACE={str(tmp_path)!r}
AGENTIC_OUTPUT_DIR={str(tmp_path)!r}
RESULT_FILENAME=agg_agentx
AIPERF_FAILED_REQUEST_THRESHOLD=0
IS_MULTINODE={'true' if is_multinode else 'false'}
ENABLE_AGENTX_POWER={'1' if enable_power else '0'}
REQUIRE_POWER={'1' if require_power else '0'}
CONC=8
SRT_MEASUREMENT_WINDOW_DIR={formal_window_dir!r}
{f'unset {missing_power_env}' if missing_power_env else ''}
set +e
run_agentic_replay_and_write_outputs {str(result_dir)!r}
rc=$?
exit "$rc"
"""
    return subprocess.run(
        ["bash", "-c", script],
        env={
            **os.environ,
            "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True,
        text=True,
        check=False,
    )


def _events(tmp_path: Path) -> list[str]:
    return (tmp_path / "events.log").read_text().splitlines()


@pytest.mark.parametrize("missing_power_env", [None, "REQUIRE_POWER"])
def test_single_node_replay_publishes_no_power_even_when_enabled(
    tmp_path: Path, missing_power_env: str | None
):
    result = _run_lifecycle(tmp_path, require_power=True, missing_power_env=missing_power_env)

    assert result.returncode == 0, result.stderr
    events = _events(tmp_path)
    assert "replay" in events
    assert not any(event.startswith("adapter:") for event in events)
    assert json.loads((tmp_path / "agg_agentx.json").read_text()) == {}
    assert not (tmp_path / "results/agentic_power_timezone_offset.txt").exists()
    assert not (tmp_path / "results/power_validation.json").exists()


def test_multinode_explicit_opt_out_skips_power(tmp_path: Path):
    result = _run_lifecycle(
        tmp_path,
        is_multinode=True,
        enable_power=False,
        formal_multinode_power=True,
    )

    assert result.returncode == 0, result.stderr
    events = _events(tmp_path)
    assert "replay" in events
    assert not any(event.startswith("adapter:") for event in events)


@pytest.mark.parametrize("require_power", [False, True])
def test_multinode_missing_contract_records_invalid_power_and_enforces_strict_mode(
    tmp_path: Path, require_power: bool
):
    result = _run_lifecycle(
        tmp_path,
        is_multinode=True,
        require_power=require_power,
        real_power_adapter=True,
    )

    assert result.returncode == int(require_power), result.stderr
    events = _events(tmp_path)
    adapter_event = next(event for event in events if event.startswith("adapter:"))
    assert events.index("aggregate") < events.index(adapter_event)
    aggregate = json.loads((tmp_path / "agg_agentx.json").read_text())
    validation = json.loads((tmp_path / "results/power_validation.json").read_text())
    assert aggregate["power_valid"] == 0
    assert "total_gpu_energy_j" not in aggregate
    assert validation["power_valid"] is False
    assert validation["reasons"] == ["multinode_power_contract_missing"]


def test_multinode_formal_window_wraps_replay(tmp_path: Path):
    result = _run_lifecycle(
        tmp_path,
        is_multinode=True,
        formal_multinode_power=True,
        require_power=True,
    )

    assert result.returncode == 0, result.stderr
    events = _events(tmp_path)
    adapters = [event for event in events if event.startswith("adapter:")]
    assert len(adapters) == 2
    assert "--write-multinode-window running" in adapters[0]
    assert "--write-multinode-window completed" in adapters[1]
    assert "--concurrency 8" in adapters[0]
    assert "--require-power" in adapters[0]
    assert events.index(adapters[0]) < events.index("replay")
    assert events.index("aggregate") < events.index(adapters[1])
    captured_offset = (tmp_path / "results/agentic_power_timezone_offset.txt").read_text().strip()
    assert re.fullmatch(r"[+-]\d{4}", captured_offset)


def test_multinode_formal_window_is_left_running_when_replay_is_interrupted(tmp_path: Path):
    result = _run_lifecycle(
        tmp_path,
        replay_rc=143,
        is_multinode=True,
        formal_multinode_power=True,
        require_power=True,
    )

    assert result.returncode == 143, result.stderr
    adapters = [event for event in _events(tmp_path) if event.startswith("adapter:")]
    assert len(adapters) == 1
    assert "--write-multinode-window running" in adapters[0]
