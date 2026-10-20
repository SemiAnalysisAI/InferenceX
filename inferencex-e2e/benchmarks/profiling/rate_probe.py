"""How often each GPU telemetry source updates, and what a read costs."""

import ctypes
import json
import os
import time

from telemetry import NvmlFieldValue, NvmlSample, nvml_number

PROBE_S = 1.0

NVML_PROBE_FIELDS = (("power_instant_mw", 186, 0), ("power_average_mw", 185, 0), ("energy_mj", 83, 0),
                     ("power_instant_module_mw", 186, 1), ("memory_temp_c", 82, 0),
                     ("perf_policy_power", 74, 0), ("perf_policy_thermal", 75, 0),
                     ("pcie_tx_bytes", 197, 0), ("nvlink_data_tx_kib_link0", 138, 0))
NVML_PROBE_SAMPLES = (("samples.power_mw", 0), ("samples.gpu_util_pct", 1), ("samples.sm_clock_mhz", 5),
                      ("samples.module_power_mw", 7))


def spacing(times_ns):
    gaps = sorted((b - a) / 1e6 for a, b in zip(times_ns, times_ns[1:]))
    if not gaps:
        return None
    return {q: round(gaps[min(len(gaps) - 1, int(len(gaps) * f))], 3)
            for q, f in (("p10", 0.1), ("p50", 0.5), ("p90", 0.9))}


def time_source(read, seconds=PROBE_S):
    """Read in a tight loop; the times (mid-read) at which the value changed."""
    reads, changes, last = 0, [], object()
    end = time.perf_counter() + seconds
    t_start = time.perf_counter()
    while time.perf_counter() < end:
        t0 = time.time_ns()
        value = read()
        t1 = time.time_ns()
        reads += 1
        if value is None:
            return None
        if value != last:
            changes.append((t0 + t1) // 2)
            last = value
    elapsed = time.perf_counter() - t_start
    return {"read_us": round(elapsed / reads * 1e6, 1), "reads": reads, "changes": len(changes) - 1,
            "change_ms": spacing(changes[1:]) if len(changes) > 2 else None}


def nvml_sources(lib, handle):
    def field(fid, scope):
        value = NvmlFieldValue(fieldId=fid, scopeId=scope)

        def read():
            if lib.nvmlDeviceGetFieldValues(handle, 1, ctypes.byref(value)) != 0 or value.nvmlReturn != 0:
                return None
            return nvml_number(value.valueType, value.value)
        return read

    def getter(name, ctype, *args):
        fn, out = getattr(lib, name, None), ctype()

        def read():
            if fn is None or fn(handle, *args, ctypes.byref(out)) != 0:
                return None
            return out.value
        return read

    sources = {name: field(fid, scope) for name, fid, scope in NVML_PROBE_FIELDS}
    sources.update({
        "clock_sm_mhz": getter("nvmlDeviceGetClockInfo", ctypes.c_uint, 1),
        "clock_mem_mhz": getter("nvmlDeviceGetClockInfo", ctypes.c_uint, 2),
        "event_reasons": getter("nvmlDeviceGetCurrentClocksEventReasons", ctypes.c_ulonglong),
        "power_usage_mw": getter("nvmlDeviceGetPowerUsage", ctypes.c_uint),
        "energy_consumption_mj": getter("nvmlDeviceGetTotalEnergyConsumption", ctypes.c_ulonglong),
        "gpu_temp_c": getter("nvmlDeviceGetTemperature", ctypes.c_uint, 0),
    })
    return sources


def nvml_samples(lib, handle, kind, seconds=2 * PROBE_S):
    """Driver sample buffer: spacing of its own timestamps, and what a drain costs."""
    value_type, count = ctypes.c_int(), ctypes.c_uint()
    if lib.nvmlDeviceGetSamples(handle, kind, 0, ctypes.byref(value_type), ctypes.byref(count), None) != 0 \
            or not count.value:
        return None
    buf = (NvmlSample * count.value)()
    stamps, last, calls, spent = set(), 0, 0, 0.0
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        count.value = len(buf)
        t0 = time.perf_counter()
        status = lib.nvmlDeviceGetSamples(handle, kind, last, ctypes.byref(value_type), ctypes.byref(count), buf)
        spent += time.perf_counter() - t0
        calls += 1
        if status == 0:
            for s in buf[:count.value]:
                stamps.add(s.timeStamp)
            last = max(stamps, default=0)
        time.sleep(0.05)
    times = sorted(stamps)
    return {"read_us": round(spent / calls * 1e6, 1), "samples": len(times), "buffer": len(buf),
            "stamp_ms": spacing([t * 1000 for t in times])}


def probe(backend, out_path):
    started = time.time()
    lib, handle = backend.lib, backend.handles[0]
    out = {"gpu": 0, "backend": type(backend).__name__, "probe_s": PROBE_S, "sources": {}}
    if out["backend"] == "Nvml":
        for name, read in nvml_sources(lib, handle).items():
            out["sources"][name] = time_source(read)
        for name, kind in NVML_PROBE_SAMPLES:
            out["sources"][name] = nvml_samples(lib, handle, kind)
    else:
        reads, history, last = 0, {}, {}
        end = time.perf_counter() + 4 * PROBE_S
        t_start = time.perf_counter()
        while time.perf_counter() < end:
            t0 = time.time_ns()
            try:
                metrics = lib.amdsmi_get_gpu_metrics_info(handle)
            except Exception:
                break
            t1 = time.time_ns()
            reads += 1
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and last.get(k) != v:
                    history.setdefault(k, []).append((t0 + t1) // 2)
                    last[k] = v
        per_read = round((time.perf_counter() - t_start) / max(reads, 1) * 1e6, 1)
        for k, times in history.items():
            out["sources"][k] = {"read_us": per_read, "reads": reads, "changes": len(times) - 1,
                                 "change_ms": spacing(times[1:]) if len(times) > 2 else None}
    out["seconds"] = round(time.time() - started, 1)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    return out
