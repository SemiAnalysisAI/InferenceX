"""Shell-contract tests for the shared single-node AgentX power lifecycle."""

from __future__ import annotations

import os
import re
import signal
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_LIB = REPO_ROOT / "benchmarks" / "benchmark_lib.sh"


def _run_lifecycle(
    tmp_path: Path,
    *,
    replay_rc: int = 0,
    is_multinode: bool = False,
    enable_power: bool = True,
    require_power: bool = False,
    formal_multinode_power: bool = False,
) -> subprocess.CompletedProcess[str]:
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    event_log = tmp_path / "events.log"
    formal_window_dir = str(tmp_path / "power/windows") if formal_multinode_power else ""
    formal_benchmark_type = "custom" if formal_multinode_power else ""
    formal_concurrencies = "8 16" if formal_multinode_power else ""
    formal_result_root = str(tmp_path) if formal_multinode_power else ""
    script = f"""
source {str(BENCHMARK_LIB)!r}
start_gpu_monitor() {{
    printf 'monitor-start:%s\n' "$*" >> {str(event_log)!r}
    printf 'timestamp,index,power.draw [W]\n' > "$2"
}}
stop_gpu_monitor() {{ printf 'monitor-stop\n' >> {str(event_log)!r}; }}
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
        *utils.agentic.aggregation.power_adapter*)
            printf 'adapter:%s\n' "$*" >> {str(event_log)!r}
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
AGENTIC_DIR={str(tmp_path)!r}
INFMAX_CONTAINER_WORKSPACE={str(tmp_path)!r}
AGENTIC_OUTPUT_DIR={str(tmp_path)!r}
RESULT_FILENAME=agg_agentx
AIPERF_FAILED_REQUEST_THRESHOLD=0
TP=3
PP_SIZE=2
PCP_SIZE=2
IS_MULTINODE={'true' if is_multinode else 'false'}
ENABLE_AGENTX_POWER={'1' if enable_power else '0'}
REQUIRE_POWER={'1' if require_power else '0'}
CONC=8
SRT_MEASUREMENT_WINDOW_DIR={formal_window_dir!r}
SRT_MEASUREMENT_WINDOW_BENCHMARK_TYPE={formal_benchmark_type!r}
SRT_MEASUREMENT_WINDOW_CONCURRENCIES={formal_concurrencies!r}
SRT_MEASUREMENT_WINDOW_RESULT_ROOT={formal_result_root!r}
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


@pytest.mark.parametrize(
    ("replay_rc", "expected_rc"),
    [(0, 0), (7, 7)],
)
def test_single_node_monitor_wraps_replay_and_stops_once(
    tmp_path: Path, replay_rc: int, expected_rc: int
):
    result = _run_lifecycle(tmp_path, replay_rc=replay_rc)

    assert result.returncode == expected_rc, result.stderr
    events = _events(tmp_path)
    assert events.count("monitor-stop") == 1
    assert events.index("monitor-start:--output " + str(tmp_path / "results/gpu_metrics.csv")) < events.index(
        "replay"
    )
    assert events.index("replay") < events.index("monitor-stop")
    assert events.index("monitor-stop") < events.index("aggregate")
    assert (tmp_path / "results/gpu_metrics.csv").is_file()
    captured_offset = (tmp_path / "results/agentic_power_timezone_offset.txt").read_text().strip()
    assert re.fullmatch(r"[+-]\d{4}", captured_offset)
    assert events[-1] == "parent-exit"


def test_single_node_invokes_adapter_with_gpu_shape_and_strict_mode(tmp_path: Path):
    result = _run_lifecycle(tmp_path, require_power=True)

    assert result.returncode == 0, result.stderr
    adapter_event = next(event for event in _events(tmp_path) if event.startswith("adapter:"))
    assert "--result-dir " + str(tmp_path / "results") in adapter_event
    assert "--agg-result " + str(tmp_path / "agg_agentx.json") in adapter_event
    assert "--expected-num-gpus 12" in adapter_event
    assert "--require-power" in adapter_event


@pytest.mark.parametrize(
    ("is_multinode", "enable_power"),
    [(True, True), (False, False)],
)
def test_multinode_and_explicit_opt_out_skip_local_power(
    tmp_path: Path, is_multinode: bool, enable_power: bool
):
    result = _run_lifecycle(
        tmp_path,
        is_multinode=is_multinode,
        enable_power=enable_power,
    )

    assert result.returncode == 0, result.stderr
    events = _events(tmp_path)
    assert not any(event.startswith("monitor-") for event in events)
    assert not any(event.startswith("adapter:") for event in events)


def test_multinode_formal_window_wraps_replay_without_local_monitor(tmp_path: Path):
    result = _run_lifecycle(
        tmp_path,
        is_multinode=True,
        formal_multinode_power=True,
        require_power=True,
    )

    assert result.returncode == 0, result.stderr
    events = _events(tmp_path)
    assert not any(event.startswith("monitor-") for event in events)
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


@pytest.mark.parametrize(
    ("sent_signal", "expected_rc"),
    [(signal.SIGINT, 130), (signal.SIGTERM, 143)],
)
def test_signal_stops_monitor_once_without_replacing_parent_trap(
    tmp_path: Path, sent_signal: signal.Signals, expected_rc: int
):
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    event_log = tmp_path / "events.log"
    script = f"""
source {str(BENCHMARK_LIB)!r}
start_gpu_monitor() {{
    printf 'monitor-pid:%s\n' "${{BASHPID:-$$}}" >> {str(event_log)!r}
}}
stop_gpu_monitor() {{ printf 'monitor-stop\n' >> {str(event_log)!r}; }}
fake_replay() {{
    printf 'replay-ready\n' >> {str(event_log)!r}
    exec sleep 30
}}
trap 'printf "parent-exit\\n" >> {str(event_log)!r}' EXIT
trap 'printf "parent-int\\n" >> {str(event_log)!r}; exit 130' INT
trap 'printf "parent-term\\n" >> {str(event_log)!r}; exit 143' TERM
REPLAY_CMD=fake_replay
ENABLE_AGENTX_POWER=1
IS_MULTINODE=false
run_agentic_replay_and_write_outputs {str(result_dir)!r}
"""
    proc = subprocess.Popen(
        ["bash", "-c", script],
        env={**os.environ, "PATH": "/usr/bin:/bin"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        # The monitor starts before the production signal traps are installed.
        # Wait for replay so the signal actually exercises those traps.
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if event_log.exists() and "replay-ready" in event_log.read_text().splitlines():
                break
            time.sleep(0.01)
        else:
            pytest.fail("replay did not start")

        os.killpg(proc.pid, sent_signal)
        _, stderr = proc.communicate(timeout=5)
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.communicate()

    assert proc.returncode == expected_rc, stderr
    events = _events(tmp_path)
    assert events.count("monitor-stop") == 1
    expected_parent_event = "parent-int" if sent_signal == signal.SIGINT else "parent-term"
    assert expected_parent_event in events
    assert events[-1] == "parent-exit"
