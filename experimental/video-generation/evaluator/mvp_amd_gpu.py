"""AMD SMI observations normalized to the existing GPU telemetry contract."""
from __future__ import annotations

import ctypes
import json
import math
import re
import time

from .mvp_gpu_job import _check_deadline, _command, _now


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
        gpus, apps = [], []
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
                # Keep zero-VRAM contexts until their ownership is established.
                apps.append({"gpu_uuid": key, "pid": proc["pid"],
                             "memory_used_mib": memory / 1024**2 if memory is not None else None})
        return {"at": _now(), "monotonic_seconds": time.monotonic(), "gpus": gpus, "compute_apps": apps,
                "power_query": {"start_utc": utc, "start_monotonic_seconds": begin,
                                "end_monotonic_seconds": end, "field": "amd-smi power.socket_power"}}
