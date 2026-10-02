"""Sample every GPU's clocks, for one profile window: NVML on NVIDIA, amdsmi on AMD.

Runs in the window client's process, not the engine's. One thread per GPU
polls, every POLL_INTERVAL_S, the graphics, SM, memory and video clocks and
the clock event (throttle) reasons. On NVIDIA the reasons are NVML's
nvmlClocksEventReason* bits; on AMD, gpu_metrics' throttle status, and the
graphics clock fills both graphics_mhz and sm_mhz (AMD has no separate SM
clock). A poll is stamped with the wall clock (the one the engines' step log
uses) before and after its calls: those occasionally block for tens of
milliseconds, and the reading is from somewhere inside that interval. Samples
are kept in memory and written when the window's sampling stops. Output,
under OUT_DIR:

  gpus.json          {"<index>": {"uuid", "bdf", "backend", ...}}
  window<w>.csv      t0_ns,t1_ns,gpu,graphics_mhz,sm_mhz,mem_mhz,video_mhz,event_reasons

The extractor puts each kernel's lifetime on this timeline, matching a rank to
its GPU by UUID or PCI address.
"""

import array
import ctypes
import json
import os
import threading
import time

CLOCKS = (("graphics_mhz", 0), ("sm_mhz", 1), ("mem_mhz", 2), ("video_mhz", 3))
FIELDS = ("t0_ns", "t1_ns", "gpu") + tuple(name for name, _ in CLOCKS) + ("event_reasons",)
# Clock changes seen on B200 are milliseconds apart; faster polling only burns a core.
POLL_INTERVAL_S = 250e-6


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
        # Renamed from ClocksThrottleReasons in recent drivers; same bitmask.
        self._reasons = getattr(self.lib, "nvmlDeviceGetCurrentClocksEventReasons", None) or \
            self.lib.nvmlDeviceGetCurrentClocksThrottleReasons

    @staticmethod
    def _check(status):
        if status != 0:
            raise RuntimeError(f"NVML error {status}")

    def gpus(self):
        out = {}
        buf = ctypes.create_string_buffer(96)
        for i, handle in enumerate(self.handles):
            self._check(self.lib.nvmlDeviceGetUUID(handle, buf, len(buf)))
            out[str(i)] = {"uuid": buf.value.decode(), "bdf": None, "backend": "nvml"}
        return out

    def poller(self, i):
        """A function returning GPU i's (graphics, sm, mem, video MHz, event reasons); -1 on failure."""
        handle, lib, reasons = self.handles[i], self.lib, self._reasons
        mhz, mask = ctypes.c_uint(), ctypes.c_ulonglong()
        mhz_ref, mask_ref = ctypes.byref(mhz), ctypes.byref(mask)

        def poll():
            values = []
            for _, clock_type in CLOCKS:
                values.append(mhz.value if lib.nvmlDeviceGetClockInfo(handle, clock_type, mhz_ref) == 0
                              else -1)
            values.append(mask.value if reasons(handle, mask_ref) == 0 else -1)
            return values
        return poll


class AmdSmi:
    """amdsmi (ROCm's Python bindings): clocks and throttle status from gpu_metrics.

    gpu_metrics field names vary with the ASIC and amdsmi version (MI300-class
    parts report per-XCC lists), so the first read picks the fields present.
    """

    CLOCK_FIELDS = {  # column -> candidate gpu_metrics keys, first present wins
        "graphics_mhz": ("current_gfxclk", "current_gfxclks", "average_gfxclk_frequency"),
        "mem_mhz": ("current_uclk", "average_uclk_frequency"),
        "video_mhz": ("current_vclk0", "current_vclk0s", "average_vclk0_frequency"),
        "event_reasons": ("throttle_status", "indep_throttle_status"),
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

    @staticmethod
    def _value(raw):
        """A gpu_metrics value as an int: the first valid entry of a per-XCC list."""
        if isinstance(raw, (list, tuple)):
            raw = next((v for v in raw if isinstance(v, int) and 0 < v < 0xFFFF), None)
        return raw if isinstance(raw, int) and raw != 0xFFFF and raw != 0xFFFFFFFF else None

    def gpus(self):
        out = {}
        for i, handle in enumerate(self.handles):
            info = {"backend": "amdsmi", "fields": self.fields}
            for key, call in (("uuid", "amdsmi_get_gpu_device_uuid"), ("bdf", "amdsmi_get_gpu_device_bdf")):
                try:
                    info[key] = str(getattr(self.lib, call)(handle))
                except Exception:
                    info[key] = None
            out[str(i)] = info
        return out

    def poller(self, i):
        handle, lib, fields, value = self.handles[i], self.lib, self.fields, self._value

        def poll():
            try:
                metrics = lib.amdsmi_get_gpu_metrics_info(handle)
            except Exception:
                return [-1] * 5
            got = {c: (value(metrics.get(k)) if k else None) for c, k in fields.items()}
            gfx = got["graphics_mhz"]
            return [v if v is not None else -1
                    for v in (gfx, gfx, got["mem_mhz"], got["video_mhz"], got["event_reasons"])]
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
        if self.nvml is None:  # no GPU telemetry here: windows still profile, without clocks
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
        self._stop = threading.Event()
        self._window = None
        self.polls = 0

    def start(self, window):
        if self.nvml is None:
            return
        self._stop.clear()
        self._window = window
        self._samples = [array.array("q") for _ in self.nvml.handles]
        self._threads = [threading.Thread(target=self._run, args=(i,), daemon=True)
                         for i in range(len(self.nvml.handles))]
        for thread in self._threads:
            thread.start()

    def stop(self):
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

    def _run(self, i):
        poll, out, stop = self.nvml.poller(i), self._samples[i], self._stop
        while not stop.is_set():
            t0 = time.time_ns()
            values = poll()
            out.extend((t0, time.time_ns(), i, *values))
            time.sleep(POLL_INTERVAL_S)
