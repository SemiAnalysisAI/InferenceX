"""Sample every GPU's clocks through NVML as fast as it answers, for one profile window.

Runs in the window client's process, not the engine's. Each poll reads, per
GPU, the graphics, SM, memory and video clocks and the clock event reasons
(power cap, thermal, sync boost, ...), stamped with the wall clock the
engines' step log uses. Output per window, under OUT_DIR:

  gpus.json          {"<nvml index>": "<uuid>"}
  window<w>.csv      t_ns,gpu,graphics_mhz,sm_mhz,mem_mhz,video_mhz,event_reasons

The extractor puts each kernel's lifetime on this timeline.
"""

import ctypes
import json
import os
import threading
import time

CLOCKS = (("graphics_mhz", 0), ("sm_mhz", 1), ("mem_mhz", 2), ("video_mhz", 3))
FIELDS = ("t_ns", "gpu") + tuple(name for name, _ in CLOCKS) + ("event_reasons",)


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
        self._mhz = ctypes.c_uint()
        self._mask = ctypes.c_ulonglong()

    @staticmethod
    def _check(status):
        if status != 0:
            raise RuntimeError(f"NVML error {status}")

    def uuids(self):
        out = {}
        buf = ctypes.create_string_buffer(96)
        for i, handle in enumerate(self.handles):
            self._check(self.lib.nvmlDeviceGetUUID(handle, buf, len(buf)))
            out[str(i)] = buf.value.decode()
        return out

    def poll(self, i):
        """(graphics, sm, mem, video MHz, event reasons) of GPU i; -1 where NVML fails."""
        handle = self.handles[i]
        values = []
        for _, clock_type in CLOCKS:
            ok = self.lib.nvmlDeviceGetClockInfo(handle, clock_type, ctypes.byref(self._mhz)) == 0
            values.append(self._mhz.value if ok else -1)
        ok = self._reasons(handle, ctypes.byref(self._mask)) == 0
        values.append(self._mask.value if ok else -1)
        return values


class ClockSampler:
    """Polls every GPU in a background thread between start(window) and stop()."""

    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.nvml = None
        self.error = None
        try:
            self.nvml = Nvml()
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "gpus.json"), "w") as f:
                json.dump(self.nvml.uuids(), f)
        except Exception as e:  # no NVML here: windows still profile, without clocks
            self.error = str(e)
        self._thread = None
        self._stop = threading.Event()
        self.polls = 0

    def start(self, window):
        if self.nvml is None:
            return
        self._stop.clear()
        self.polls = 0
        path = os.path.join(self.out_dir, f"window{window}.csv")
        self._thread = threading.Thread(target=self._run, args=(path,), daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            self._thread = None

    def _run(self, path):
        gpus = range(len(self.nvml.handles))
        with open(path, "w") as f:
            f.write(",".join(FIELDS) + "\n")
            while not self._stop.is_set():
                for i in gpus:
                    t0 = time.time_ns()
                    values = self.nvml.poll(i)
                    t = (t0 + time.time_ns()) // 2
                    f.write(f"{t},{i}," + ",".join(map(str, values)) + "\n")
                self.polls += 1
