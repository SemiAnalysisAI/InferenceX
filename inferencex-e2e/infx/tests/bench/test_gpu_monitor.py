"""GPU monitor against stub SMI tools: telemetry files, workload status, and signal relay."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

from infx.bench import gpu_monitor
from infx.bench.gpu_monitor import GpuMonitor
from infx.tests.bench.stubs import executable

REPO_ROOT = Path(__file__).resolve().parents[3]
# The columns infx.results.power reads from the NVIDIA stream.
NVIDIA_QUERY = (
    "timestamp,index,power.draw,temperature.gpu,clocks.current.sm,clocks.current.memory,"
    "utilization.gpu,utilization.memory"
)
NVIDIA_HEADER = (
    "timestamp, index, power.draw [W], temperature.gpu, clocks.current.sm [MHz], "
    "clocks.current.memory [MHz], utilization.gpu [%], utilization.memory [%]"
)
NVIDIA_SAMPLE = "2026/07/23 12:00:09.000, 0, 490.00 W, 64, 990, 990, 89 %, 9 %"
NVIDIA_FINAL = "2026/07/23 12:00:11.000, 0, 500.00 W, 65, 1000, 1000, 90 %, 10 %"
IDENTITY = (
    "index, uuid, pci.bus_id, name, driver_version\n"
    "0, GPU-device-a, 00000000:01:00.0, NVIDIA Test GPU, 590.00\n"
)


def tool_dir(tmp_path: Path, **stubs: str) -> Path:
    """A PATH with only the shell basics and the given stub tools, so no real SMI is found."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("sh", "sleep"):
        (bin_dir / tool).symlink_to(shutil.which(tool))
    for name, body in stubs.items():
        executable(bin_dir / name.replace("_", "-"), f"#!/bin/sh\n{body}")
    return bin_dir


def nvidia_smi(stream: str, *, interval: int = 1, identity_ok: bool = True, pid_file: Path | None = None) -> str:
    """nvidia-smi answering the identity, streaming, and one-shot sample queries."""
    record_pid = f"echo $$ > {shlex.quote(str(pid_file))}; " if pid_file else ""
    return f"""case "$*" in
    "--query-gpu=index,uuid,pci.bus_id,name,driver_version --format=csv")
        printf '%s' {shlex.quote(IDENTITY)}
        {'' if identity_ok else 'exit 1'} ;;
    "--query-gpu={NVIDIA_QUERY} --format=csv -l {interval}")
        {record_pid}printf '%s' {shlex.quote(stream)}
        exec sleep 30 ;;
    "--query-gpu={NVIDIA_QUERY} --format=csv,noheader")
        printf '%s\\n' {shlex.quote(NVIDIA_FINAL)} ;;
    *) echo "unexpected nvidia-smi $*"; exit 1 ;;
esac
"""


def wait_until(condition, timeout: float = 10) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.02)


def text(path: Path) -> str:
    return path.read_text() if path.is_file() else ""


@pytest.mark.parametrize(("identity_ok", "identity"), [(True, IDENTITY), (False, None)])
def test_nvidia_stream_drops_a_truncated_row_then_appends_the_bracketing_sample(
    tmp_path, monkeypatch, identity_ok, identity
):
    truncated = "2026/07/23 12:00:10.000, 0, 52"
    stream = f"{NVIDIA_HEADER}\n{NVIDIA_SAMPLE}\n{truncated}"
    stub = nvidia_smi(stream, interval=2, identity_ok=identity_ok)
    monkeypatch.setenv("PATH", str(tool_dir(tmp_path, nvidia_smi=stub)))
    (tmp_path / "power").mkdir()
    output = tmp_path / "power" / "node0.csv"

    with GpuMonitor(output, interval=2) as monitor:
        wait_until(lambda: text(output).endswith(truncated))
        assert monitor.vendor == "nvidia"

    assert output.read_text().splitlines() == [NVIDIA_HEADER, NVIDIA_SAMPLE, NVIDIA_FINAL]
    # A failed identity query costs only its sidecar, never the telemetry.
    sidecar = tmp_path / "power" / "node0_identity.csv"
    assert (sidecar.read_text() if sidecar.exists() else None) == identity


def test_amd_stream_keeps_one_header_and_brackets_the_workload(tmp_path, monkeypatch):
    first_snapshot = shlex.quote(str(tmp_path / "energy_start_taken"))
    amd_smi = f"""case "$*" in
    "metric -p -c -t -u -w 3 --csv")
        printf "'CTRL' + 'C' to stop watching output:\\n"
        printf 'timestamp,gpu,socket_power\\n123,0,400\\ntimestamp,gpu,socket_power\\n124,0,410\\n125,0,4'
        exec sleep 30 ;;
    "metric -E --csv")
        if [ -e {first_snapshot} ]; then energy=200; else : > {first_snapshot}; energy=100; fi
        printf 'gpu,total_energy_consumption\\n0,%s\\n' "$energy" ;;
    "static --json") printf '{{"gpu_data": []}}\\n' ;;
    *) echo "unexpected amd-smi $*"; exit 1 ;;
esac
"""
    monkeypatch.setenv("PATH", str(tool_dir(tmp_path, amd_smi=amd_smi)))
    drains = []
    monkeypatch.setattr(gpu_monitor, "time", types.SimpleNamespace(sleep=drains.append))
    output = tmp_path / "gpu_metrics.csv"
    rows = "timestamp,gpu,socket_power\n123,0,400\n124,0,410\n"

    with GpuMonitor(output, interval=3) as monitor:
        # Rows reach the file while amd-smi is still running, not at its exit.
        wait_until(lambda: text(output) == rows)
        assert monitor.vendor == "amd"

    # Two ticks past the workload: amd-smi stamps whole seconds.
    assert drains == [5]
    assert output.read_text() == rows
    assert (tmp_path / "gpu_metrics_energy_start.csv").read_text() == "gpu,total_energy_consumption\n0,100\n"
    assert (tmp_path / "gpu_metrics_energy_end.csv").read_text() == "gpu,total_energy_consumption\n0,200\n"
    assert json.loads((tmp_path / "gpu_metrics_identity.json").read_text()) == {"gpu_data": []}


def test_without_a_gpu_tool_the_workload_runs_unmonitored_and_keeps_its_status(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PATH", str(tool_dir(tmp_path)))
    output = tmp_path / "gpu_metrics.csv"

    assert gpu_monitor.run(output, 1, ["sh", "-c", "exit 7"]) == 7
    assert not output.exists()


RUN_MONITOR = (
    "import sys; from pathlib import Path; from infx.bench.gpu_monitor import run; "
    "sys.exit(run(Path(sys.argv[1]), 1, sys.argv[2:]))"
)


@pytest.mark.parametrize(
    ("sent", "rc"), [(signal.SIGTERM, 143), (signal.SIGHUP, 129)], ids=["TERM", "HUP"]
)
def test_monitor_relays_the_signal_and_still_stops_the_sampler(tmp_path, sent, rc):
    pid_file = tmp_path / "sampler.pid"
    bin_dir = tool_dir(tmp_path, nvidia_smi=nvidia_smi(f"{NVIDIA_HEADER}\n", pid_file=pid_file))
    output, started, relayed = tmp_path / "gpu_metrics.csv", tmp_path / "started", tmp_path / "relayed"
    trap = sent.name.removeprefix("SIG")
    workload = f'trap "echo {trap} > {relayed}; exit 0" {trap}; : > {started}; while :; do sleep 0.05; done'
    process = subprocess.Popen(
        [sys.executable, "-c", RUN_MONITOR, str(output), "sh", "-c", workload],
        env={"PATH": str(bin_dir), "PYTHONPATH": str(REPO_ROOT)},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )  # fmt: skip
    try:
        wait_until(lambda: started.exists() and pid_file.exists())
        process.send_signal(sent)
        _, stderr = process.communicate(timeout=30)
    finally:
        process.kill()

    # The workload handled the signal and exited 0, but an interrupted run never passes.
    assert process.returncode == rc, stderr
    assert relayed.read_text() == f"{trap}\n"
    assert output.read_text().endswith(NVIDIA_FINAL + "\n")
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)
