"""Sample every GPU's clocks and power, for one profile window: NVML on NVIDIA, amdsmi on AMD."""

import array
import ctypes
import json
import os
import threading
import time

from telemetry import Telemetry

CLOCKS = (("graphics_mhz", 0), ("sm_mhz", 1), ("mem_mhz", 2), ("video_mhz", 3))
FIELDS = ("t0_ns", "t1_ns", "gpu") + tuple(name for name, _ in CLOCKS) + ("event_reasons", "power_mw", "energy_mj")
POLL_INTERVAL_S = 250e-6
POWER_EVERY = 4
POWER_DRAIN_S = 0.5
NVML_TOTAL_POWER_SAMPLES = 0
NVML_FI_DEV_POWER_INSTANT = 186


class _NvmlValue(ctypes.Union):
    _fields_ = [("dVal", ctypes.c_double), ("uiVal", ctypes.c_uint), ("ulVal", ctypes.c_ulong),
                ("ullVal", ctypes.c_ulonglong), ("sllVal", ctypes.c_longlong), ("siVal", ctypes.c_int)]


class _NvmlFieldValue(ctypes.Structure):
    _fields_ = [("fieldId", ctypes.c_uint), ("scopeId", ctypes.c_uint), ("timestamp", ctypes.c_longlong),
                ("latencyUsec", ctypes.c_longlong), ("valueType", ctypes.c_int), ("nvmlReturn", ctypes.c_int),
                ("value", _NvmlValue)]


def _field_number(field):
    kind = field.valueType
    return {0: field.value.dVal, 1: field.value.uiVal, 2: field.value.ulVal, 3: field.value.ullVal,
            4: field.value.sllVal, 5: field.value.siVal}.get(kind)


class Nvml:
    def __init__(self):
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        self._check(self.lib.nvmlInit_v2())
        count = ctypes.c_uint()
        self._check(self.lib.nvmlDeviceGetCount_v2(ctypes.byref(count)))
        self.handles = []
        for i in range(count.value):
            handle = ctypes.c_void_p()
            self._check(self.lib.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(handle)))
            self.handles.append(handle)
        self._reasons = getattr(self.lib, "nvmlDeviceGetCurrentClocksEventReasons", None) or \
            self.lib.nvmlDeviceGetCurrentClocksThrottleReasons
        self._instant = [self._read_instant(h) is not None for h in self.handles]

    def _read_instant(self, handle):
        """NVML_FI_DEV_POWER_INSTANT in mW, or None (older drivers and parts lack it)."""
        field = _NvmlFieldValue(fieldId=NVML_FI_DEV_POWER_INSTANT)
        try:
            if self.lib.nvmlDeviceGetFieldValues(handle, 1, ctypes.byref(field)) != 0 or field.nvmlReturn != 0:
                return None
        except AttributeError:
            return None
        value = _field_number(field)
        return int(value) if value is not None else None

    @staticmethod
    def _check(status):
        if status != 0:
            raise RuntimeError(f"NVML error {status}")

    def gpus(self):
        out = {}
        buf = ctypes.create_string_buffer(96)
        for i, handle in enumerate(self.handles):
            self._check(self.lib.nvmlDeviceGetUUID(handle, buf, len(buf)))
            out[str(i)] = {"uuid": buf.value.decode(), "bdf": None, "backend": "nvml", "power": {
                "source": "NVML_FI_DEV_POWER_INSTANT" if self._instant[i] else "nvmlDeviceGetPowerUsage",
                "averaging": "instantaneous" if self._instant[i] else "averaged over ~tens of ms (driver-defined)",
                "samples_source": "nvmlDeviceGetSamples(NVML_TOTAL_POWER_SAMPLES)",
                "energy_source": "nvmlDeviceGetTotalEnergyConsumption at window start and stop",
                "energy_resolution_mj": 1}}
        return out

    def poller(self, i):
        """A function returning GPU i's (graphics, sm, mem, video MHz, event reasons); -1 on failure."""
        handle, lib, reasons, instant = self.handles[i], self.lib, self._reasons, self._instant[i]
        mhz, mask, mw, mj = ctypes.c_uint(), ctypes.c_ulonglong(), ctypes.c_uint(), ctypes.c_ulonglong()
        mhz_ref, mask_ref, mw_ref, mj_ref = ctypes.byref(mhz), ctypes.byref(mask), ctypes.byref(mw), ctypes.byref(mj)
        count = [0]

        def poll():
            values = []
            for _, clock_type in CLOCKS:
                values.append(mhz.value if lib.nvmlDeviceGetClockInfo(handle, clock_type, mhz_ref) == 0
                              else -1)
            values.append(mask.value if reasons(handle, mask_ref) == 0 else -1)
            count[0] += 1
            if count[0] % POWER_EVERY:
                return values + [-1, -1]
            power = self._read_instant(handle) if instant else None
            if power is None:
                power = mw.value if lib.nvmlDeviceGetPowerUsage(handle, mw_ref) == 0 else -1
            return values + [power, -1]
        return poll

    def power_reader(self, i):
        """drain(): the driver's power samples since the last drain, [(t_ns, mW)]; energy(): (t_ns, mJ) of one energy counter read, or None."""
        from telemetry import NvmlSample, nvml_number

        handle, lib = self.handles[i], self.lib
        value_type, count = ctypes.c_int(), ctypes.c_uint()
        buf = [None]
        last = [0]
        offset = [None]
        mj = ctypes.c_ulonglong()
        window_start = time.time_ns()

        def drain():
            """[(t_ns wall, mW)] taken since the window started, after ("clock", name, offset_ns) once."""
            if lib.nvmlDeviceGetSamples(handle, NVML_TOTAL_POWER_SAMPLES, 0, ctypes.byref(value_type),
                                        ctypes.byref(count), None) != 0 or not count.value:
                return []
            if buf[0] is None or len(buf[0]) < count.value:
                buf[0] = (NvmlSample * count.value)()
            count.value = len(buf[0])
            if lib.nvmlDeviceGetSamples(handle, NVML_TOTAL_POWER_SAMPLES, 0, ctypes.byref(value_type),
                                        ctypes.byref(count), buf[0]) != 0:
                return []
            got = [(s.timeStamp, nvml_number(value_type.value, s.sampleValue)) for s in buf[0][:count.value]
                   if s.timeStamp > last[0]]
            if not got:
                return []
            newest = max(t for t, _ in got)
            meta = []
            if offset[0] is None:
                offset[0], name = stamp_offset_ns(newest * 1000)
                meta = [("clock", name, offset[0])]
            last[0] = newest
            return meta + [(t * 1000 + offset[0], mw) for t, mw in got if t * 1000 + offset[0] >= window_start]

        def energy():
            t0 = time.time_ns()
            if lib.nvmlDeviceGetTotalEnergyConsumption(handle, ctypes.byref(mj)) != 0:
                return None
            return (t0 + time.time_ns()) // 2, mj.value

        return drain, energy


def stamp_offset_ns(stamp_ns):
    """The host clock a driver stamp (just taken) follows, and the ns to add for wall time."""
    now = time.time_ns()
    clocks = {"realtime": now, "monotonic": time.monotonic_ns()}
    if hasattr(time, "CLOCK_BOOTTIME"):
        clocks["boottime"] = time.clock_gettime_ns(time.CLOCK_BOOTTIME)
    for name, reading in clocks.items():
        if 0 <= reading - stamp_ns <= 5 * POWER_DRAIN_S * 1e9:
            return now - reading, name
    return now - stamp_ns, "unknown"


class AmdSmi:
    """amdsmi (ROCm's Python bindings): clocks and throttle status from gpu_metrics."""

    CLOCK_FIELDS = {
        "graphics_mhz": ("current_gfxclk", "current_gfxclks", "average_gfxclk_frequency"),
        "mem_mhz": ("current_uclk", "average_uclk_frequency"),
        "video_mhz": ("current_vclk0", "current_vclk0s", "average_vclk0_frequency"),
        "event_reasons": ("throttle_status", "indep_throttle_status"),
        "power_w": ("current_socket_power", "average_socket_power"),
        "energy": ("energy_accumulator",),
    }

    def __init__(self):
        import amdsmi

        self.lib = amdsmi
        amdsmi.amdsmi_init()
        self.handles = list(amdsmi.amdsmi_get_processor_handles())
        if not self.handles:
            raise RuntimeError("amdsmi found no GPUs")
        sample = amdsmi.amdsmi_get_gpu_metrics_info(self.handles[0])
        self.fields = {column: next((k for k in keys if self._value(sample.get(k)) is not None), None)
                       for column, keys in self.CLOCK_FIELDS.items()}
        self.energy_resolution_uj = []
        for handle in self.handles:
            try:
                self.energy_resolution_uj.append(float(amdsmi.amdsmi_get_energy_count(handle)["counter_resolution"]))
            except Exception:
                self.energy_resolution_uj.append(None)

    @staticmethod
    def _value(raw):
        """A gpu_metrics value as an int: the first valid entry of a per-XCC list."""
        if isinstance(raw, (list, tuple)):
            raw = next((v for v in raw if isinstance(v, int) and 0 < v < 0xFFFF), None)
        return (raw if isinstance(raw, int) and raw not in (0xFFFF, 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF)
                else None)

    def gpus(self):
        out = {}
        for i, handle in enumerate(self.handles):
            info = {"backend": "amdsmi", "fields": self.fields, "power": {
                "source": f"gpu_metrics.{self.fields.get('power_w')}",
                "averaging": "socket power as the SMU reports it (averaged over its own window)",
                "energy_source": f"gpu_metrics.{self.fields.get('energy')}",
                "energy_resolution_uj": self.energy_resolution_uj[i]}}
            for key, call in (("uuid", "amdsmi_get_gpu_device_uuid"), ("bdf", "amdsmi_get_gpu_device_bdf")):
                try:
                    info[key] = str(getattr(self.lib, call)(handle))
                except Exception:
                    info[key] = None
            out[str(i)] = info
        return out

    def poller(self, i):
        handle, lib, fields, value = self.handles[i], self.lib, self.fields, self._value
        resolution = self.energy_resolution_uj[i]
        count = [0]

        def poll():
            try:
                metrics = lib.amdsmi_get_gpu_metrics_info(handle)
            except Exception:
                return [-1] * 7
            got = {c: (value(metrics.get(k)) if k else None) for c, k in fields.items()}
            gfx = got["graphics_mhz"]
            power = energy = None
            count[0] += 1
            if not count[0] % POWER_EVERY:
                power = got["power_w"] * 1000 if got["power_w"] is not None else None
                if got["energy"] is not None and resolution:
                    energy = int(got["energy"] * resolution / 1000)
            return [v if v is not None else -1
                    for v in (gfx, gfx, got["mem_mhz"], got["video_mhz"], got["event_reasons"], power, energy)]
        return poll


class ClockSampler:
    """Polls every GPU, one thread each, between start(window) and stop()."""

    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.nvml = None
        self.error = None
        errors = []
        for backend in (Nvml, AmdSmi):
            try:
                self.nvml = backend()
                break
            except Exception as e:
                errors.append(f"{backend.__name__}: {e}")
        if self.nvml is None:
            self.error = "; ".join(errors)
        else:
            try:
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, "gpus.json"), "w") as f:
                    json.dump(self.nvml.gpus(), f)
            except Exception as e:
                self.error, self.nvml = str(e), None
        self._threads = []
        self._samples = []
        self._power = []
        self._stop = threading.Event()
        self._window = None
        self.polls = 0
        self.telemetry, self.telemetry_rows = None, 0
        if os.environ.get("INFX_PROF_TELEMETRY") != "1":
            return
        try:
            self.telemetry = Telemetry(out_dir, self.nvml)
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "telemetry.json"), "w") as f:
                json.dump(self.telemetry.info(), f)
        except Exception as e:
            self.telemetry, self.error = None, "; ".join(filter(None, (self.error, f"telemetry: {e}")))

    def start(self, window):
        if self.telemetry is not None:
            self.telemetry.start(window)
        if self.nvml is None:
            return
        self._stop.clear()
        self._window = window
        self._samples = [array.array("q") for _ in self.nvml.handles]
        self._threads = [threading.Thread(target=self._run, args=(i,), daemon=True)
                         for i in range(len(self.nvml.handles))]
        self._power = []
        if hasattr(self.nvml, "power_reader"):
            self._power = [[] for _ in self.nvml.handles]
            self._threads += [threading.Thread(target=self._run_power, args=(i,), daemon=True)
                              for i in range(len(self.nvml.handles))]
        for thread in self._threads:
            thread.start()

    def stop(self):
        if self.telemetry is not None:
            self.telemetry_rows = self.telemetry.stop()
        if not self._threads:
            return
        self._stop.set()
        for thread in self._threads:
            thread.join()
        self._threads = []
        width = len(FIELDS)
        self.polls = sum(len(s) // width for s in self._samples)
        with open(os.path.join(self.out_dir, f"window{self._window}.csv"), "w") as f:
            f.write(",".join(FIELDS) + "\n")
            for samples in self._samples:
                for k in range(0, len(samples), width):
                    f.write(",".join(map(str, samples[k:k + width])) + "\n")
        self._samples = []
        if self._power:
            with open(os.path.join(self.out_dir, f"power{self._window}.csv"), "w") as f:
                f.write("t_ns,gpu,kind,value\n")
                for i, rows in enumerate(self._power):
                    for t_ns, kind, value in rows:
                        f.write(f"{t_ns},{i},{kind},{value}\n")
        self._power = []

    def _run_power(self, i):
        out, stop = self._power[i], self._stop
        try:
            drain, energy = self.nvml.power_reader(i)
        except Exception:
            return
        reading = energy()
        if reading:
            out.append((reading[0], "energy_mj", reading[1]))
        def take():
            for row in drain():
                if row[0] == "clock":
                    out.append((time.time_ns(), f"stamp_clock_{row[1]}_offset_ns", row[2]))
                else:
                    out.append((row[0], "sample_mw", row[1]))
        take()
        while not stop.wait(POWER_DRAIN_S):
            take()
        take()
        reading = energy()
        if reading:
            out.append((reading[0], "energy_mj", reading[1]))

    def _run(self, i):
        poll, out, stop = self.nvml.poller(i), self._samples[i], self._stop
        while not stop.is_set():
            t0 = time.time_ns()
            values = poll()
            out.extend((t0, time.time_ns(), i, *values))
            time.sleep(POLL_INTERVAL_S)
