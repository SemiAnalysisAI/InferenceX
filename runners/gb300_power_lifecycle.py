"""Idle-GPU telemetry diagnostic using the unchanged srt-slurm power stage."""
import json
import math
import os
import re
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import requests
from srtctl.cli.mixins.telemetry_stage import TelemetryStageMixin, read_producer_commit
from srtctl.core.power.contract import CLOCK_SOURCE, SCHEMA_VERSION, atomic_write_json
from srtctl.core.power.samples import read_samples
from srtctl.core.processes import ProcessRegistry
from srtctl.core.runtime import Nodes
from srtctl.core.schema import TelemetryConfig, TelemetryExporterConfig, TelemetryProvider
from srtctl.core.topology import Process

PIN = "80d7203e424f903c9017de4608ee2044afce9574"
PORT = 19401
IMAGE = "/data/home/sa-shared/gharunners/squash/nvcr.io_nvidia_k8s_dcgm-exporter_4.6.0-4.8.3-distroless.sqsh"


def node_probe(out: Path, phase: str) -> None:
    assert read_producer_commit() == PIN
    hostname = socket.gethostname().split(".")[0]
    if phase != "during":
        with socket.socket() as candidate:
            candidate.bind(("0.0.0.0", PORT))
    root_listeners = []
    own_inodes = set()
    owners = []
    for family in ("tcp", "tcp6"):
        for line in Path("/proc/net", family).read_text().splitlines()[1:]:
            fields = line.split()
            if fields[1].split(":")[-1] == f"{9401:04X}" and fields[3] == "0A":
                root_listeners.append({"uid": int(fields[7]), "inode": fields[9]})
            if fields[1].split(":")[-1] == f"{PORT:04X}" and fields[3] == "0A":
                own_inodes.add(fields[9])
    if phase == "during":
        for proc in Path("/proc").iterdir():
            if not proc.name.isdecimal():
                continue
            try:
                matched = any(str(fd.readlink()) in {f"socket:[{inode}]" for inode in own_inodes}
                              for fd in (proc / "fd").iterdir())
                if matched:
                    owners.append({"pid": int(proc.name), "uid": proc.stat().st_uid,
                        "command": (proc / "cmdline").read_bytes().replace(b"\x00", b" ").decode(),
                        "cgroup": (proc / "cgroup").read_text()})
            except OSError:
                continue
        atomic_write_json(out / f"{hostname}-listener-owner.json", {"inodes": sorted(own_inodes), "owners": owners})
        assert own_inodes and len(owners) == 1
        owner = owners[0]
        assert "dcgm-exporter" in owner["command"] and f":{PORT}" in owner["command"]
        assert re.search(r"(?:^|/)job_" + re.escape(os.environ["SLURM_JOB_ID"]) + r"(?:/|$)", owner["cgroup"])

    response = requests.get("http://127.0.0.1:9401/metrics", timeout=3)
    response.raise_for_status()
    service = next(line for line in response.text.splitlines() if line.startswith("nvgpu_exporter_info{"))
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    devices = {index.strip(): uuid.strip() for index, uuid in (line.split(",") for line in raw.splitlines())}
    assert set(devices) == {"0", "1", "2", "3"}
    assert root_listeners and all(listener["uid"] == 0 for listener in root_listeners)
    state = {"hostname": hostname, "devices": devices, "candidate_port_free": PORT if phase != "during" else None,
             "root_listeners": root_listeners, "root_service": service,
             "enroot_list": subprocess.check_output(["enroot", "list"], text=True).splitlines(),
             "producer_sha": PIN, "phase": phase}
    atomic_write_json(out / f"{hostname}-{phase}.json", state)
    if phase == "after":
        before = json.loads((out / f"{hostname}-before.json").read_text())
        assert state["devices"] == before["devices"]
        assert state["root_listeners"] == before["root_listeners"]
        assert state["root_service"] == before["root_service"]


class Diagnostic(TelemetryStageMixin):
    def __init__(self, out: Path, nodes: list[str]):
        telemetry = TelemetryConfig(enabled=True, provider=TelemetryProvider.DCGM_POWER,
            default_frequency=1.0, storage_subdir="power", required=True,
            startup_timeout_seconds=120, request_timeout_seconds=2, collector_join_timeout_seconds=12,
            dcgm_exporter=TelemetryExporterConfig(container_image=IMAGE, port=PORT))
        self.config = SimpleNamespace(telemetry=telemetry,
            benchmark=SimpleNamespace(type="custom", get_concurrency_list=lambda: [1]))
        self.runtime = SimpleNamespace(log_dir=out, job_id=os.environ["SLURM_JOB_ID"],
            run_name=f"idle-power-diagnostic-{os.environ['SLURM_JOB_ID']}", network_interface=None,
            nodes=Nodes(head=nodes[0], bench=nodes[0], infra=nodes[0], worker=tuple(nodes)),
            container_mounts={}, srun_options={"mem": "0", "container-remap-root": ""})
        self._workers = [Process(node=node, gpu_indices=frozenset(range(4)), sys_port=0, http_port=0,
            endpoint_mode="agg", endpoint_index=0, node_rank=i) for i, node in enumerate(nodes)]

    @property
    def backend_processes(self):
        return self._workers


def collect(out: Path) -> None:
    assert read_producer_commit() == PIN
    nodes = subprocess.check_output(["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]], text=True).split()
    assert len(nodes) == 2
    expected_uuids = set()
    for node in nodes:
        evidence = json.loads((out / f"{node}-before.json").read_text())
        assert evidence["candidate_port_free"] == PORT
        expected_uuids.update(evidence["devices"].values())
    assert len(expected_uuids) == 8
    stage = Diagnostic(out, nodes)
    registry = ProcessRegistry(os.environ["SLURM_JOB_ID"])
    rc = 1
    try:
        session = stage.start_power_telemetry(registry)
        atomic_write_json(out / "exporter-processes.json", {
            name: {"pid": process.popen.pid, "command": process.popen.args}
            for name, process in registry.get_all_processes().items()})
        assert session is not None and not stage.power_telemetry_blocks_benchmark()
        subprocess.run(["srun", "--jobid", os.environ["SLURM_JOB_ID"], "--overlap",
            "--nodes=2", "--ntasks=2", "--ntasks-per-node=1", "--cpus-per-task=1",
            f"--output={out}/during-%t.txt", sys.executable, __file__, "during", str(out)], check=True)
        start = time.time()
        time.sleep(30)
        end = time.time()
        result = {"measurement_kind": "idle_gpu_telemetry_diagnostic", "completed": 0,
            "total_input_tokens": 0, "total_output_tokens": 0,
            "benchmark_start_time_unix": start, "benchmark_end_time_unix": end, "duration": end-start}
        atomic_write_json(out / "idle-diagnostic.json", result)
        atomic_write_json(session.windows_dir / "idle-diagnostic.json", {
            **result, "schema_version": SCHEMA_VERSION, "clock_source": CLOCK_SOURCE,
            "benchmark_type": "custom", "concurrency": 1, "result_path": "idle-diagnostic.json",
            "status": "completed", "reason": None})
        time.sleep(3)
        rc = stage.finalize_power_telemetry(0)
        rows, reasons = read_samples(session.samples_path)
        manifest = json.loads(session.manifest_path.read_text())
        assert rc == 0 and not reasons and manifest["publication_valid"]
        assert {row.gpu_uuid for row in rows} == expected_uuids
        assert all(math.isfinite(row.power_w) and row.power_w > 0 for row in rows)
        atomic_write_json(out / "lifecycle-acceptance.json", {
            "telemetry_valid": True, "benchmark_qualified": False, "measurement_kind": "idle_gpu_diagnostic",
            "producer_sha": PIN, "candidate_port": PORT, "gpu_count": 8,
            "sample_count": len(rows), "window_start": start, "window_end": end,
            "window_duration": end-start, "completed_requests": 0, "token_count": 0})
    finally:
        stage.finalize_power_telemetry(rc)
        registry.cleanup()
        atomic_write_json(out / "exporter-cleanup.json", {
            name: {"pid": process.popen.pid, "returncode": process.popen.poll()}
            for name, process in registry.get_all_processes().items()})
        assert all(process.popen.poll() is not None for process in registry.get_all_processes().values())


if __name__ == "__main__":
    mode, directory = sys.argv[1:]
    out = Path(directory)
    def interrupted(signum, frame):
        raise SystemExit(f"diagnostic interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    if mode in ("before", "during", "after"):
        node_probe(out, mode)
    else:
        assert mode == "collect"
        collect(out)
