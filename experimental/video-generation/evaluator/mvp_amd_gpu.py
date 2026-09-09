"""AMD SMI observations normalized to the existing GPU telemetry contract."""
from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

from .mvp_gpu_job import _check_deadline, _command, _now, _proc_identity


def monitor_process(pid: int) -> dict:
    before = _proc_identity(pid)
    status = Path(f"/proc/{pid}/status").read_text()
    cgroup = Path(f"/proc/{pid}/cgroup").read_text()
    with Path(f"/proc/{pid}/cmdline").open("rb") as stream:
        executable = stream.read(4096).split(b"\0", 1)[0].decode()
    after = _proc_identity(pid)
    if (not before or not after or before["start_ticks"] != after["start_ticks"]
            or before["ppid"] != 1 or not re.search(r"^Uid:\s+0\s+0\s+0\s+0\s*$", status, re.MULTILINE)
            or not any(line.endswith(":/system.slice/gpuagent.service") for line in cgroup.splitlines())
            or not executable.startswith("/")):
        raise ValueError("GPU monitor process identity is not established")
    try:
        if os.readlink(f"/proc/{pid}/exe") != executable:
            raise ValueError("GPU monitor executable changed")
        method = "proc exe and systemd ExecStart"
    except PermissionError:
        method = "systemd ExecMainPID/ExecStart and proc argv0; proc exe access denied"
    return {"pid": pid, "start_ticks": before["start_ticks"], "uid": 0,
            "cgroup": "/system.slice/gpuagent.service", "executable": executable,
            "executable_verification": method}


def observe_system_monitor(timeout: float) -> dict:
    record = {"service": "gpuagent.service", "observed_at": _now(), "status": "unverified"}
    try:
        raw = _command(["systemctl", "show", "gpuagent.service", "--property=MainPID,ExecMainPID,ExecStart,ActiveState,SubState,ControlGroup,Type"], timeout=timeout)
        properties = dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
        pid = int(properties["MainPID"])
        executable = re.search(r"(?:^|[ {])path=(/[^ ;}]+)", properties["ExecStart"])
        if (properties["ActiveState"] != "active" or properties["SubState"] != "running"
                or properties["ControlGroup"] != "/system.slice/gpuagent.service"
                or properties["Type"] not in {"simple", "exec", "notify"}
                or pid <= 1 or int(properties["ExecMainPID"]) != pid or executable is None):
            raise ValueError("GPU monitoring service is not an active direct systemd process")
        process = monitor_process(pid)
        binary = Path(executable[1])
        info = binary.stat()
        if process["executable"] != str(binary) or not binary.is_file() or info.st_uid != 0 or info.st_mode & 0o022:
            raise ValueError("GPU monitor executable is not the root-owned service binary")
        record.update(status="verified", process=process,
                      executable_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                      policy="Only this unchanged root service with zero per-process VRAM is excluded from workload contexts; board power still includes its overhead")
    except Exception as error:
        record.update(error_type=type(error).__name__, error=str(error))
    return record


def is_system_monitor(app: dict, receipt: dict | None) -> bool:
    if not receipt or receipt.get("status") != "verified" or app["memory_used_mib"] != 0:
        return False
    expected = receipt.get("process", {})
    if app["pid"] != expected.get("pid"):
        return False
    try:
        return monitor_process(app["pid"]) == expected
    except (OSError, ValueError):
        return False


def smi(option: str, timeout: float) -> object:
    return json.loads(_command(["amd-smi", option, "--json"], timeout=timeout))


def inventory(rows: object) -> dict[str, dict]:
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("AMD SMI GPU inventory unavailable")
    values = {}
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get("uuid"), str)
                or not re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", row["uuid"])
                or type(row.get("gpu")) is not int or not 0 <= row["gpu"] < 8
                or row.get("partition_id") != 0
                or not isinstance(row.get("bdf"), str)
                or not re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", row["bdf"])):
            raise RuntimeError("AMD inventory is malformed or partitioned")
        values[row["uuid"]] = row
    if (len(values) != len(rows) or len({r["gpu"] for r in rows}) != len(rows)
            or len({r["bdf"] for r in rows}) != len(rows)):
        raise RuntimeError("AMD inventory contains duplicate identities")
    return values


def hip_devices() -> list[str]:
    # HIP and AMD SMI ordinals need not agree; join physical devices by PCI BDF.
    observed = inventory(smi("list", 10))
    by_bdf = {row["bdf"]: key for key, row in observed.items()}
    hip = ctypes.CDLL("libamdhip64.so")
    hip.hipGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    hip.hipDeviceGetPCIBusId.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    count = ctypes.c_int()
    if hip.hipGetDeviceCount(ctypes.byref(count)) != 0 or not 1 <= count.value <= 8:
        raise RuntimeError("HIP device enumeration failed")
    devices = []
    for ordinal in range(count.value):
        bdf = ctypes.create_string_buffer(32)
        if hip.hipDeviceGetPCIBusId(bdf, len(bdf), ordinal) != 0:
            raise RuntimeError("HIP PCI device identity unavailable")
        key = bdf.value.decode().lower()
        if key not in by_bdf:
            raise RuntimeError("HIP device is absent from AMD SMI inventory")
        devices.append(by_bdf[key])
    if len(set(devices)) != len(devices):
        raise RuntimeError("HIP device identities are duplicated")
    return devices


def number(value: object, unit: str, *, optional: bool = False) -> float | None:
    if isinstance(value, dict) and value.get("unit") == unit:
        raw = value.get("value")
        if type(raw) in (int, float) and math.isfinite(raw) and raw >= 0:
            return float(raw)
    if optional:
        return None
    raise RuntimeError("AMD telemetry value or unit unavailable: " + unit)


class AmdGpuProbe:
    def __init__(self, devices: list[str], timeout: float):
        self.devices, self.timeout = devices, timeout
        monitor = os.environ.get("H3_AMD_MONITOR_RECEIPT")
        self.monitor = json.loads(Path(monitor).read_text()) if monitor else None
        self.identity = inventory(smi("list", timeout))
        self.static = {row["gpu"]: row for row in smi("static", timeout)}
        if not set(devices) <= self.identity.keys():
            raise RuntimeError("Assigned AMD UUIDs absent from physical inventory")
        for key in devices:
            row = self.identity[key]
            static = self.static[row["gpu"]]
            if static["bus"]["bdf"] != row["bdf"] or static["asic"]["market_name"] != "AMD Instinct MI355X":
                raise RuntimeError("AMD device identity changed or unsupported GPU model")

    def power_configuration(self, *, deadline: float | None = None) -> dict:
        _check_deadline(deadline)
        timeout = self.timeout if deadline is None else min(self.timeout, max(0.001, deadline - time.monotonic()))
        rows = {row["gpu"]: row for row in smi("static", timeout)}
        return {"observed_at": _now(), "query": "amd-smi static --json", "status": "recorded",
                "gpus": [{"uuid": key,
                          "configured_limit_w": number(rows[self.identity[key]["gpu"]]["limit"]["socket_power"], "W", optional=True),
                          "maximum_limit_w": number(rows[self.identity[key]["gpu"]]["limit"]["max_power"], "W", optional=True),
                          "enforced_limit_w": None, "default_limit_w": None} for key in self.devices]}

    def snapshot(self, *, deadline: float | None = None) -> dict:
        def query(option):
            _check_deadline(deadline)
            timeout = self.timeout if deadline is None else min(self.timeout, max(0.001, deadline - time.monotonic()))
            return smi(option, timeout)
        observed = inventory(query("list"))
        if any(observed.get(key) != self.identity[key] for key in self.devices):
            raise RuntimeError("AMD GPU inventory changed during measurement")
        begin, utc = time.monotonic(), _now()
        metric = query("metric")
        end = time.monotonic()
        rows = metric["gpu_data"]
        metrics = {row["gpu"]: row for row in rows}
        processes = query("process")
        by_gpu = {row["gpu"]: row["process_list"] for row in processes}
        if len(metrics) != len(rows) or len(by_gpu) != len(processes):
            raise RuntimeError("AMD telemetry contains duplicate GPU records")
        gpus, apps, monitors = [], [], []
        for key in self.devices:
            index = self.identity[key]["gpu"]
            raw, static = metrics[index], self.static[index]
            # AMD SMI 26.2 labels these MB but divides bytes by 1024**2.
            # See ROCm/amdsmi rocm-7.1.1 amdsmi_commands.py mem_usage.
            gpus.append({"uuid": key, "index": index, "name": static["asic"]["market_name"],
                         "memory_total_mib": number(raw["mem_usage"]["total_vram"], "MB"),
                         "memory_used_mib": number(raw["mem_usage"]["used_vram"], "MB"),
                         "utilization_percent": number(raw["usage"]["gfx_activity"], "%"),
                         "power_watts": number(raw["power"]["socket_power"], "W", optional=True),
                         "temperature_celsius": number(raw["temperature"]["hotspot"], "C", optional=True),
                         "driver_version": static["driver"]["version"], "mig_mode": "N/A",
                         "vendor": "amd", "pci_bdf": self.identity[key]["bdf"]})
            for item in by_gpu[index]:
                proc = item["process_info"]
                if type(proc.get("pid")) is not int or proc["pid"] <= 0:
                    raise RuntimeError("AMD process identity unavailable")
                memory = number(proc["memory_usage"]["vram_mem"], "B", optional=True)
                app = {"gpu_uuid": key, "pid": proc["pid"],
                       "memory_used_mib": memory / 1024**2 if memory is not None else None}
                if is_system_monitor(app, self.monitor):
                    monitors.append({**app, "identity": self.monitor})
                else:
                    apps.append(app)
        return {"at": _now(), "monotonic_seconds": time.monotonic(), "gpus": gpus, "compute_apps": apps,
                "excluded_system_monitor_contexts": monitors,
                "power_query": {"start_utc": utc, "start_monotonic_seconds": begin,
                                "end_monotonic_seconds": end, "field": "amd-smi power.socket_power"}}
