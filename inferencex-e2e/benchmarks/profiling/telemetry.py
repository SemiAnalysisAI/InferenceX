"""Every telemetry source a GPU and its host expose, for one profile window."""

import ctypes
import glob
import os
import threading
import time

FIELDS_INTERVAL_S = 1e-3
SAMPLES_INTERVAL_S = 0.25
AMD_INTERVAL_S = 1e-3
HOST_INTERVAL_S = 5e-3
CPUFREQ_EVERY = 20
JOIN_TIMEOUT_S = 5.0

NVML_FIELDS = (
    ("power_instant_mw", 186, 0), ("power_instant_module_mw", 186, 1),
    ("power_average_mw", 185, 0), ("energy_mj", 83, 0),
    ("power_limit_current_mw", 190, 0), ("power_limit_current_module_mw", 190, 1),
    ("power_limit_requested_mw", 192, 0), ("power_limit_min_mw", 187, 0),
    ("power_limit_max_mw", 188, 0), ("power_limit_default_mw", 189, 0),
    ("memory_temp_c", 82, 0), ("tlimit_shutdown_c", 193, 0), ("tlimit_slowdown_c", 194, 0),
    ("tlimit_mem_max_c", 195, 0), ("tlimit_gpu_max_c", 196, 0),
    ("perf_policy_power", 74, 0), ("perf_policy_thermal", 75, 0), ("perf_policy_sync_boost", 76, 0),
    ("perf_policy_board_limit", 77, 0), ("perf_policy_low_utilization", 78, 0),
    ("perf_policy_reliability", 79, 0), ("perf_policy_total_app_clocks", 80, 0),
    ("perf_policy_total_base_clocks", 81, 0),
    ("pcie_tx_bytes", 197, 0), ("pcie_rx_bytes", 198, 0),
)
NVLINK_FIELDS = (("nvlink_data_tx_kib", 138), ("nvlink_data_rx_kib", 139))
NVML_FI_DEV_NVLINK_LINK_COUNT = 91
NVML_SAMPLES = (("power_mw", 0), ("gpu_util_pct", 1), ("mem_util_pct", 2), ("sm_clock_mhz", 5),
                ("mem_clock_mhz", 6), ("module_power_mw", 7))


class NvmlValue(ctypes.Union):
    _fields_ = [("dVal", ctypes.c_double), ("uiVal", ctypes.c_uint), ("ulVal", ctypes.c_ulong),
                ("ullVal", ctypes.c_ulonglong), ("sllVal", ctypes.c_longlong), ("siVal", ctypes.c_int),
                ("usVal", ctypes.c_ushort)]


class NvmlFieldValue(ctypes.Structure):
    _fields_ = [("fieldId", ctypes.c_uint), ("scopeId", ctypes.c_uint), ("timestamp", ctypes.c_longlong),
                ("latencyUsec", ctypes.c_longlong), ("valueType", ctypes.c_int), ("nvmlReturn", ctypes.c_int),
                ("value", NvmlValue)]


class NvmlSample(ctypes.Structure):
    _fields_ = [("timeStamp", ctypes.c_ulonglong), ("sampleValue", NvmlValue)]


def nvml_number(kind, value):
    return {0: value.dVal, 1: value.uiVal, 2: value.ulVal, 3: value.ullVal,
            4: value.sllVal, 5: value.siVal, 6: value.usVal}.get(kind)


class NvmlTelemetry:
    """Readers for one NVML library handle (clock_sampler.Nvml's)."""

    def __init__(self, lib, handles):
        self.lib, self.handles = lib, handles
        self.fields = [self._supported(h) for h in handles]

    def _read_fields(self, handle, specs):
        values = (NvmlFieldValue * len(specs))()
        for k, (_, fid, scope) in enumerate(specs):
            values[k].fieldId, values[k].scopeId = fid, scope
        if self.lib.nvmlDeviceGetFieldValues(handle, len(specs), values) != 0:
            return None
        return values

    def _supported(self, handle):
        """The fields this GPU answers, NVLink throughput per present link."""
        specs = list(NVML_FIELDS)
        links = self._read_fields(handle, [("links", NVML_FI_DEV_NVLINK_LINK_COUNT, 0)])
        if links is not None and links[0].nvmlReturn == 0:
            count = nvml_number(links[0].valueType, links[0].value) or 0
            specs += [(name, fid, link) for link in range(int(count)) for name, fid in NVLINK_FIELDS]
        values = self._read_fields(handle, specs)
        return [] if values is None else [s for s, v in zip(specs, values) if v.nvmlReturn == 0]

    def info(self, i):
        handle, lib, out = self.handles[i], self.lib, {}
        mhz = ctypes.c_uint()
        for name, clock_type in (("graphics", 0), ("sm", 1), ("mem", 2)):
            for what, call in (("max", lambda: lib.nvmlDeviceGetMaxClockInfo(handle, clock_type, ctypes.byref(mhz))),
                               ("app_target", lambda: lib.nvmlDeviceGetClock(handle, clock_type, 1, ctypes.byref(mhz))),
                               ("app_default", lambda: lib.nvmlDeviceGetClock(handle, clock_type, 2, ctypes.byref(mhz))),
                               ("boost_max", lambda: lib.nvmlDeviceGetClock(handle, clock_type, 3, ctypes.byref(mhz)))):
                out[f"{name}_{what}_mhz"] = mhz.value if call() == 0 else None
        out["fields"] = [[name, fid, scope] for name, fid, scope in self.fields[i]]
        return out

    def readers(self, i):
        return [(self._fields_reader(i), FIELDS_INTERVAL_S), (self._samples_reader(i), SAMPLES_INTERVAL_S)]

    def _fields_reader(self, i):
        handle, lib, specs = self.handles[i], self.lib, self.fields[i]
        values = (NvmlFieldValue * len(specs))()
        temp, pstate, fan = ctypes.c_uint(), ctypes.c_int(), ctypes.c_uint()

        class Util(ctypes.Structure):
            _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]
        util = Util()
        get_temp, get_pstate, get_fan, get_util = (getattr(lib, name, None) for name in (
            "nvmlDeviceGetTemperature", "nvmlDeviceGetPerformanceState", "nvmlDeviceGetFanSpeed",
            "nvmlDeviceGetUtilizationRates"))

        def read():
            rows = []
            for k, (_, fid, scope) in enumerate(specs):
                values[k].fieldId, values[k].scopeId = fid, scope
            if specs and lib.nvmlDeviceGetFieldValues(handle, len(specs), values) == 0:
                for (name, _, scope), v in zip(specs, values):
                    if v.nvmlReturn == 0:
                        rows.append((name, scope, v.timestamp, nvml_number(v.valueType, v.value)))
            for sensor in (0, 1):
                if get_temp and get_temp(handle, sensor, ctypes.byref(temp)) == 0:
                    rows.append(("gpu_temp_c", sensor, None, temp.value))
            if get_pstate and get_pstate(handle, ctypes.byref(pstate)) == 0:
                rows.append(("pstate", 0, None, pstate.value))
            if get_fan and get_fan(handle, ctypes.byref(fan)) == 0:
                rows.append(("fan_pct", 0, None, fan.value))
            if get_util and get_util(handle, ctypes.byref(util)) == 0:
                rows += [("util_gpu_pct", 0, None, util.gpu), ("util_mem_pct", 0, None, util.memory)]
            return rows
        return read

    def _samples_reader(self, i):
        handle, lib = self.handles[i], self.lib
        last = {kind: 0 for _, kind in NVML_SAMPLES}
        buffers = {}

        def read():
            rows = []
            for name, kind in NVML_SAMPLES:
                value_type, count = ctypes.c_int(), ctypes.c_uint()
                if kind not in buffers:
                    if lib.nvmlDeviceGetSamples(handle, kind, 0, ctypes.byref(value_type), ctypes.byref(count),
                                                None) != 0 or not count.value:
                        buffers[kind] = None
                        continue
                    buffers[kind] = (NvmlSample * count.value)()
                buf = buffers[kind]
                if buf is None:
                    continue
                count.value = len(buf)
                if lib.nvmlDeviceGetSamples(handle, kind, last[kind], ctypes.byref(value_type),
                                            ctypes.byref(count), buf) != 0:
                    continue
                for s in buf[:count.value]:
                    if s.timeStamp > last[kind]:
                        rows.append((f"samples.{name}", 0, s.timeStamp, nvml_number(value_type.value, s.sampleValue)))
                last[kind] = max([last[kind]] + [s.timeStamp for s in buf[:count.value]])
            return rows
        read.every_row = True
        return read


class AmdTelemetry:
    def __init__(self, lib, handles):
        self.lib, self.handles = lib, handles

    def info(self, i):
        return {}

    def readers(self, i):
        handle, lib = self.handles[i], self.lib

        def read():
            try:
                metrics = lib.amdsmi_get_gpu_metrics_info(handle)
            except Exception:
                return []
            rows = []
            for key, raw in metrics.items():
                for scope, v in (enumerate(raw) if isinstance(raw, (list, tuple)) else ((0, raw),)):
                    if isinstance(v, (int, float)) and v not in (0xFFFF, 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF):
                        rows.append((key, scope, None, v))
            return rows
        return [(read, AMD_INTERVAL_S)]


def host_readers():
    """RAPL energy, hwmon power, mean CPU frequency: whatever this host exposes."""
    files = []
    for zone in sorted(glob.glob("/sys/class/powercap/*/energy_uj")):
        base = os.path.dirname(zone)
        try:
            name = open(os.path.join(base, "name")).read().strip()
        except OSError:
            name = os.path.basename(base)
        files.append((f"rapl.{os.path.basename(base)}.{name}.energy_uj", zone))
    for path in sorted(glob.glob("/sys/class/hwmon/hwmon*/power*_input") +
                       glob.glob("/sys/class/hwmon/hwmon*/power*_average")):
        base = os.path.dirname(path)
        try:
            name = open(os.path.join(base, "name")).read().strip()
        except OSError:
            name = "?"
        label = path[:-len("_input" if path.endswith("_input") else "_average")] + "_label"
        try:
            name += "." + open(label).read().strip()
        except OSError:
            pass
        files.append((f"hwmon.{os.path.basename(base)}.{name}.{os.path.basename(path)}_uw", path))
    readable = []
    for name, path in files:
        try:
            int(open(path).read())
            readable.append((name, path))
        except (OSError, ValueError):
            pass
    freqs = sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq"))
    count = [0]

    def read():
        rows = []
        for name, path in readable:
            try:
                with open(path) as f:
                    rows.append((name, 0, None, int(f.read())))
            except (OSError, ValueError):
                pass
        count[0] += 1
        if freqs and count[0] % CPUFREQ_EVERY == 1:
            khz = []
            for path in freqs:
                try:
                    with open(path) as f:
                        khz.append(int(f.read()))
                except (OSError, ValueError):
                    pass
            if khz:
                rows += [("cpu_freq_mean_khz", 0, None, sum(khz) // len(khz)),
                         ("cpu_freq_min_khz", 0, None, min(khz)), ("cpu_freq_max_khz", 0, None, max(khz))]
        return rows
    return [(read, HOST_INTERVAL_S)] if readable or freqs else [], [n for n, _ in readable]


class Telemetry:
    """All readers for one window: start(window) to stop(), then telemetry<w>.csv."""

    def __init__(self, out_dir, backend):
        self.out_dir = out_dir
        self.gpu = None
        if backend is not None and hasattr(backend, "lib") and hasattr(backend, "handles"):
            kind = NvmlTelemetry if type(backend).__name__ == "Nvml" else AmdTelemetry
            self.gpu = kind(backend.lib, backend.handles)
        self.host, self.host_sources = host_readers()
        self._threads, self._rows, self._stop = [], [], threading.Event()
        self.errors = []

    def info(self):
        out = {"host_sources": self.host_sources}
        if self.gpu is not None:
            for i in range(len(self.gpu.handles)):
                try:
                    out[str(i)] = self.gpu.info(i)
                except Exception as e:
                    out[str(i)] = {"error": str(e)}
        return out

    def start(self, window):
        self._window = window
        self._stop.clear()
        jobs = [("host", read, interval) for read, interval in self.host]
        if self.gpu is not None:
            for i in range(len(self.gpu.handles)):
                jobs += [(i, read, interval) for read, interval in self.gpu.readers(i)]
        self._rows = [[] for _ in jobs]
        self._threads = [threading.Thread(target=self._run, args=(device, read, interval, out), daemon=True)
                         for (device, read, interval), out in zip(jobs, self._rows)]
        for thread in self._threads:
            thread.start()

    def stop(self):
        if not self._threads:
            return 0
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=JOIN_TIMEOUT_S)
            if thread.is_alive():
                self.errors.append("reader still in a read at stop")
        self._threads = []
        rows = 0
        with open(os.path.join(self.out_dir, f"telemetry{self._window}.csv"), "w") as f:
            f.write("t0_ns,t1_ns,device,source,scope,stamp_us,value\n")
            for out in self._rows:
                for t0, t1, device, source, scope, stamp, value in list(out):
                    f.write(f"{t0},{t1},{device},{source},{scope},{'' if stamp is None else stamp},{value}\n")
                rows += len(out)
        self._rows = []
        return rows

    def _run(self, device, read, interval, out):
        last, every_row, stop = {}, getattr(read, "every_row", False), self._stop
        while not stop.is_set():
            t0 = time.time_ns()
            try:
                rows = read()
            except Exception as e:
                self.errors.append(f"{device}: {e}")
                return
            t1 = time.time_ns()
            for source, scope, stamp, value in rows:
                if every_row or last.get((source, scope)) != value:
                    last[(source, scope)] = value
                    out.append((t0, t1, device, source, scope, stamp, value))
            stop.wait(interval)
