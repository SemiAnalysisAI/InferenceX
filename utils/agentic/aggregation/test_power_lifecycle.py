"""Shell-contract tests for the shared single-node AgentX power lifecycle."""

from __future__ import annotations

import csv
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
stop_gpu_monitor() {{
    printf 'monitor-stop:%s\n' "${{AMD_MONITOR_STOP_TIMEOUT_S:-unset}}" >> {str(event_log)!r}
}}
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
    stop_events = [event for event in events if event.startswith("monitor-stop")]
    # Signal teardown must stop exactly once, in abort mode: the coverage wait
    # is skipped by setting AMD_MONITOR_STOP_TIMEOUT_S=0 before stopping.
    assert stop_events == ["monitor-stop:0"]
    expected_parent_event = "parent-int" if sent_signal == signal.SIGINT else "parent-term"
    assert expected_parent_event in events
    assert events[-1] == "parent-exit"


# --------------------------------------------------------------------------- #
# stop_gpu_monitor AMD coverage wait (runs the real helper, no stubs)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("sent_signal", "expected_rc"),
    [(signal.SIGINT, 130), (signal.SIGTERM, 143)],
)
def test_signal_during_amd_coverage_wait_stops_monitor(
    tmp_path: Path, sent_signal: signal.Signals, expected_rc: int
):
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    event_log = tmp_path / "events.log"
    script = f"""
source {str(BENCHMARK_LIB)!r}
start_gpu_monitor() {{
    GPU_METRICS_CSV="$2"
    printf 'timestamp,gpu,socket_power\n1,0,500\n' > "$GPU_METRICS_CSV"
    command sleep 60 >/dev/null 2>&1 &
    GPU_MONITOR_PID=$!
    GPU_MONITOR_VENDOR=amd
    printf 'monitor:%s\nlifecycle:%s\n' "$GPU_MONITOR_PID" "${{BASHPID:-$(exec sh -c 'echo "$PPID"')}}" >> {str(event_log)!r}
}}
sleep() {{
    printf 'coverage-wait\n' >> {str(event_log)!r}
    command sleep "$@"
}}
_write_amd_smi_sidecar() {{ :; }}
fake_replay() {{ :; }}
trap 'printf "parent-exit\\n" >> {str(event_log)!r}' EXIT
REPLAY_CMD=fake_replay
ENABLE_AGENTX_POWER=1
IS_MULTINODE=false
AMD_MONITOR_STOP_TIMEOUT_S=30
run_agentic_replay_and_write_outputs {str(result_dir)!r}
exit $?
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
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            events = _events(tmp_path) if event_log.exists() else []
            if "coverage-wait" in events:
                break
            time.sleep(0.01)
        else:
            pytest.fail("AMD coverage wait did not start")

        pids = dict(event.split(":") for event in events if ":" in event)
        # Signal only the lifecycle shell: signalling the whole group would
        # kill the monitor directly and hide a broken cleanup handler.
        os.kill(int(pids["lifecycle"]), sent_signal)
        _, stderr = proc.communicate(timeout=5)
        assert proc.returncode == expected_rc, stderr
        assert _events(tmp_path)[-1] == "parent-exit"
        with pytest.raises(ProcessLookupError):
            os.kill(int(pids["monitor"]), 0)
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.communicate()


# AMDSMI 26.2.0 `metric -p -c -t -u -w 1 --csv` header (order-faithful subset,
# measured on MI355X; mirrors test_detect_columns_amd_watch_mode_real_header).
_MI355X_WATCH_HEADER = (
    "timestamp,gpu,gfx_activity,umc_activity,mm_activity,vcn_activity,"
    "jpeg_activity,gfx_busy_inst_xcp_0,jpeg_busy_xcp_0,vcn_busy_xcp_0,"
    "socket_power,gfx_voltage,soc_voltage,mem_voltage,throttle_status,"
    "power_management,gfx_0_clk,mem_0_clk,edge,hotspot,mem"
)

# amd-smi quotes list-valued cells with embedded commas; the coverage helper
# must keep the power cell at its header-relative position through them.
_WATCH_ROW_FORMAT = (
    "%s,%s,0,0,N/A,\"['N/A', 'N/A']\",\"['N/A', 'N/A']\",\"[0, 0]\","
    "\"[0, 0]\",\"[0, 0]\",%s,N/A,N/A,N/A,N/A,ENABLED,1404,2000,N/A,40,25\\n"
)


def _bash_single_quote(text: str) -> str:
    return "'" + text.replace("'", "'\\''") + "'"


def _run_amd_stop(
    tmp_path: Path,
    *,
    producer_script: str,
    timeout_s: int | str,
    interval: int = 1,
    setup_script: str = "",
) -> subprocess.CompletedProcess[str]:
    """Run the real stop_gpu_monitor against a scripted AMD telemetry producer."""
    csv_path = tmp_path / "gpu_metrics.csv"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    amd_smi = bin_dir / "amd-smi"
    # Quiet stand-in for the _energy_end sidecar snapshot taken at stop.
    amd_smi.write_text("#!/bin/bash\nprintf 'gpu,total_energy_consumption\\n0,100.0\\n'\n")
    amd_smi.chmod(0o755)
    script = f"""
source {str(BENCHMARK_LIB)!r}
GPU_METRICS_CSV={str(csv_path)!r}
printf '%s\\n' {_bash_single_quote(_MI355X_WATCH_HEADER)} > "$GPU_METRICS_CSV"
emit_row() {{
    printf {_bash_single_quote(_WATCH_ROW_FORMAT)} "$1" "$2" "$3" >> "$GPU_METRICS_CSV"
}}
{setup_script}
( {producer_script} ) &
GPU_MONITOR_PID=$!
printf '%s\\n' "$GPU_MONITOR_PID" > {str(tmp_path / "producer.pid")!r}
GPU_MONITOR_VENDOR=amd
GPU_MONITOR_INTERVAL={interval}
AMD_MONITOR_STOP_TIMEOUT_S={timeout_s}
date +%s > {str(tmp_path / "pre.txt")!r}
stop_gpu_monitor
date +%s > {str(tmp_path / "post.txt")!r}
"""
    return subprocess.run(
        ["bash", "-c", script],
        env={
            **os.environ,
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def _min_covered_tick(csv_path: Path) -> int:
    """Newest usable tick (numeric ts, power > 0) covered by every GPU."""
    newest: dict[str, float] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                timestamp = float((row.get("timestamp") or "").strip())
                power = float((row.get("socket_power") or "").strip())
            except ValueError:
                continue
            if timestamp > 1e12:  # millisecond epoch, mirror _parse_timestamp
                timestamp /= 1000.0
            gpu = (row.get("gpu") or "").strip()
            if not gpu or power <= 0:
                continue
            newest[gpu] = max(newest.get(gpu, 0.0), timestamp)
    assert newest, "no usable telemetry rows"
    return int(min(newest.values()))


def _stop_epochs(tmp_path: Path) -> tuple[int, int]:
    pre = int((tmp_path / "pre.txt").read_text().strip())
    post = int((tmp_path / "post.txt").read_text().strip())
    return pre, post


def _assert_producer_dead(tmp_path: Path) -> None:
    producer_pid = int((tmp_path / "producer.pid").read_text().strip())
    with pytest.raises(ProcessLookupError):
        os.kill(producer_pid, 0)


def test_amd_stop_waits_until_every_gpu_covers_stop_request(tmp_path: Path):
    producer = """
while :; do
    now=$(date +%s)
    emit_row "$now" 0 500
    emit_row "$now" 1 505
    sleep 0.2
done
"""
    started = time.monotonic()
    result = _run_amd_stop(tmp_path, producer_script=producer, timeout_s=30)
    duration = time.monotonic() - started

    assert result.returncode == 0, result.stderr
    assert "never covered the stop request" not in result.stdout + result.stderr
    pre, _ = _stop_epochs(tmp_path)
    # Every GPU has a usable tick at/after the first whole second past stop
    # entry, so any fractional window end before the stop is bracketed.
    assert _min_covered_tick(tmp_path / "gpu_metrics.csv") >= pre + 1
    _assert_producer_dead(tmp_path)
    assert duration < 10


def test_amd_stop_ignores_degenerate_rows_for_coverage(tmp_path: Path):
    setup = """
stale=$(( $(date +%s) - 30 ))
emit_row "$stale" 0 500
emit_row "$stale" 1 505
"""
    producer = """
while :; do
    now=$(date +%s)
    emit_row "$now" 0 N/A
    emit_row "$now" 1 N/A
    sleep 0.2
done
"""
    result = _run_amd_stop(
        tmp_path,
        producer_script=producer,
        timeout_s=2,
        setup_script=setup,
    )

    assert result.returncode == 0, result.stderr
    assert "never covered the stop request" in result.stderr
    pre, post = _stop_epochs(tmp_path)
    assert post - pre >= 2
    _assert_producer_dead(tmp_path)


def test_amd_stop_requires_coverage_per_gpu(tmp_path: Path):
    setup = """
stale=$(( $(date +%s) - 30 ))
emit_row "$stale" 1 505
"""
    producer = """
while :; do
    emit_row "$(date +%s)" 0 500
    sleep 0.2
done
"""
    result = _run_amd_stop(
        tmp_path,
        producer_script=producer,
        timeout_s=2,
        setup_script=setup,
    )

    assert result.returncode == 0, result.stderr
    # GPU 1 never covers the stop request, so min-over-GPUs coverage times out
    # even though GPU 0 keeps producing fresh usable ticks.
    assert "never covered the stop request" in result.stderr
    pre, post = _stop_epochs(tmp_path)
    assert post - pre >= 2
    _assert_producer_dead(tmp_path)


def test_amd_stop_survives_non_integer_timeout(tmp_path: Path):
    producer = """
while :; do
    now=$(date +%s)
    emit_row "$now" 0 500
    emit_row "$now" 1 505
    sleep 0.2
done
"""
    started = time.monotonic()
    result = _run_amd_stop(tmp_path, producer_script=producer, timeout_s="30s")
    duration = time.monotonic() - started

    assert result.returncode == 0, result.stderr
    # A non-integer timeout must not unwind stop_gpu_monitor via a bash
    # arithmetic error (which would leak the monitor and skip tail repair
    # and the energy sidecar): it warns, falls back to 30, and still waits.
    assert "ignoring non-integer AMD_MONITOR_STOP_TIMEOUT_S='30s'" in result.stderr
    assert "never covered the stop request" not in result.stdout + result.stderr
    pre, _ = _stop_epochs(tmp_path)
    assert _min_covered_tick(tmp_path / "gpu_metrics.csv") >= pre + 1
    _assert_producer_dead(tmp_path)
    assert duration < 10


def test_amd_stop_normalizes_millisecond_epoch_timestamps(tmp_path: Path):
    producer = """
while :; do
    now=$(( $(date +%s) * 1000 + 123 ))
    emit_row "$now" 0 500
    emit_row "$now" 1 505
    sleep 0.2
done
"""
    started = time.monotonic()
    result = _run_amd_stop(tmp_path, producer_script=producer, timeout_s=30)
    duration = time.monotonic() - started

    assert result.returncode == 0, result.stderr
    # Raw millisecond epochs (~1.8e12) dwarf any second-scale target, so
    # without normalization the poll would return
    # instantly with zero tail coverage; mirrored _parse_timestamp
    # normalization makes the poll wait for real coverage instead.
    assert "never covered the stop request" not in result.stdout + result.stderr
    pre, _ = _stop_epochs(tmp_path)
    assert _min_covered_tick(tmp_path / "gpu_metrics.csv") >= pre + 1
    _assert_producer_dead(tmp_path)
    assert duration < 10


def test_amd_stop_falls_back_to_fixed_tail_for_iso_timestamps(tmp_path: Path):
    producer = """
while :; do
    emit_row "$(date +%Y-%m-%dT%H:%M:%S)" 0 500
    sleep 0.2
done
"""
    started = time.monotonic()
    result = _run_amd_stop(tmp_path, producer_script=producer, timeout_s=30, interval=0)
    duration = time.monotonic() - started

    assert result.returncode == 0, result.stderr
    assert "never covered the stop request" not in result.stdout + result.stderr
    assert "exited before covering" not in result.stdout + result.stderr
    pre, post = _stop_epochs(tmp_path)
    # Non-epoch timestamps keep the legacy interval+2 fixed tail (one shot).
    assert post - pre >= 2
    assert duration < 10
    _assert_producer_dead(tmp_path)
