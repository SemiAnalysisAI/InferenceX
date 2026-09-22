"""Per-op GPU telemetry: fine-grained clocks, power, and throttle events.

A TelemetryProvider samples the device on a background thread while the
runner's timed loop executes, and returns a summary that the runner both
attaches to the result metrics and uses for its throttle-retry policy:
if a POWER/THERMAL throttle event is observed while the SM clock sits
below rated boost (actual capping — the mere power-cap governor bit at
boost clock is normal DVFS and does NOT trigger), the op is re-measured
with increased inter-kernel sleeps, up to OPERATORX_THROTTLE_RETRIES
times, and the best (lowest-median) attempt is kept.

Providers:
  NvmlTelemetry   NVIDIA via pynvml: clocks/power/temp/clock-event
                  reasons polled at OPERATORX_TELEMETRY_MS (default 5),
                  plus the driver's internal sample ring
                  (nvmlDeviceGetSamples) drained at stop() for
                  finer-than-poll power/clock history when supported.
  AmdSmiTelemetry AMD via amdsmi: gpu_metrics table (per-XCD clocks,
                  hardware throttle_status bitmask, socket power).
  NullTelemetry   fallback no-op when no library is available.

metrics["telemetry"] = {
  "provider", "interval_ms", "n_samples",
  "sm_clock_mhz": {"min","p50","max"}, "rated_sm_clock_mhz",
  "power_w": {"min","p50","max"},
  "throttle_reasons": [names observed], "capped": bool,
  "attempts": N, "inter_kernel_sleep_ms": final value  # set by runner
}
Full sample streams are appended per op to
$OPERATORX_TELEMETRY_DIR/telemetry.jsonl when that env var is set.
"""
from __future__ import annotations

import json
import os
import threading
import time

_INTERVAL_MS = float(os.environ.get("OPERATORX_TELEMETRY_MS", "5"))
_TELEMETRY_DIR = os.environ.get("OPERATORX_TELEMETRY_DIR") or None
# clock must sit this fraction below rated boost (while a power/thermal
# reason is active) to count as capped
_CAP_CLOCK_FRACTION = float(os.environ.get("OPERATORX_CAP_CLOCK_FRACTION",
                                           "0.985"))

# NVML clock-event reasons that mean capping when clock is depressed.
# GpuIdle / ApplicationsClocksSetting / SyncBoost are excluded.
_NVML_CAP_BITS = {
    0x0004: "SwPowerCap",
    0x0008: "HwSlowdown",
    0x0020: "SwThermalSlowdown",
    0x0040: "HwThermalSlowdown",
    0x0080: "HwPowerBrakeSlowdown",
}


def _pcts(vals):
    if not vals:
        return {"min": None, "p50": None, "max": None}
    s = sorted(vals)
    return {"min": s[0], "p50": s[len(s) // 2], "max": s[-1]}


class _Report:
    def __init__(self, provider: str, samples: list, rated_mhz: int | None):
        self.provider = provider
        self.samples = samples  # (t_ns, sm_mhz, power_w, reasons_mask|list)
        self.rated_mhz = rated_mhz

    def _cap_samples(self):
        out = []
        for t, clk, pw, reasons in self.samples:
            names = reasons if isinstance(reasons, list) else [
                n for bit, n in _NVML_CAP_BITS.items() if reasons & bit]
            depressed = (self.rated_mhz is not None and clk is not None
                         and clk < self.rated_mhz * _CAP_CLOCK_FRACTION)
            if names and depressed:
                out.append(names)
        return out

    @property
    def capped(self) -> bool:
        return bool(self._cap_samples())

    def summary(self) -> dict:
        clks = [s[1] for s in self.samples if s[1] is not None]
        pws = [s[2] for s in self.samples if s[2] is not None]
        reasons: set[str] = set()
        for _, _, _, r in self.samples:
            if isinstance(r, list):
                reasons.update(r)
            else:
                reasons.update(n for b, n in _NVML_CAP_BITS.items() if r & b)
        return {
            "provider": self.provider,
            "interval_ms": _INTERVAL_MS,
            "n_samples": len(self.samples),
            "sm_clock_mhz": _pcts(clks),
            "rated_sm_clock_mhz": self.rated_mhz,
            "power_w": _pcts([round(p, 1) for p in pws]),
            "throttle_reasons": sorted(reasons),
            "capped": self.capped,
        }

    def dump(self, tag: str) -> None:
        if not _TELEMETRY_DIR:
            return
        try:
            os.makedirs(_TELEMETRY_DIR, exist_ok=True)
            with open(os.path.join(_TELEMETRY_DIR, "telemetry.jsonl"), "a") as f:
                f.write(json.dumps({"op": tag, "provider": self.provider,
                                    "samples": self.samples}) + "\n")
        except OSError:
            pass


class _PollingProvider:
    """Base: background thread calling self._sample() every interval."""

    name = "null"

    def __init__(self):
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._samples: list = []
        self._rated: int | None = None

    def start(self) -> None:
        self._samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> _Report:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._drain()
        return _Report(self.name, self._samples, self._rated)

    def _loop(self) -> None:
        period = _INTERVAL_MS / 1e3
        while not self._stop.is_set():
            try:
                self._samples.append(self._sample())
            except Exception:
                pass
            self._stop.wait(period)

    def _sample(self):  # (t_ns, sm_mhz, power_w, reasons)
        return (time.time_ns(), None, None, 0)

    def _drain(self) -> None:  # provider-specific finer history
        pass


class NullTelemetry(_PollingProvider):
    def start(self) -> None:  # no thread at all
        self._samples = []

    def stop(self) -> _Report:
        return _Report("null", [], None)


class NvmlTelemetry(_PollingProvider):
    name = "nvml"

    def __init__(self, device_index: int = 0):
        super().__init__()
        import pynvml
        self._nv = pynvml
        pynvml.nvmlInit()
        self._h = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        self._rated = pynvml.nvmlDeviceGetMaxClockInfo(
            self._h, pynvml.NVML_CLOCK_SM)
        self._last_sample_ts = time.time_ns() // 1000  # us for GetSamples

    def _sample(self):
        nv, h = self._nv, self._h
        clk = nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM)
        pw = nv.nvmlDeviceGetPowerUsage(h) / 1000.0
        try:
            reasons = nv.nvmlDeviceGetCurrentClocksEventReasons(h)
        except Exception:
            reasons = nv.nvmlDeviceGetCurrentClocksThrottleReasons(h)
        return (time.time_ns(), clk, pw, int(reasons))

    def _drain(self) -> None:
        """Pull the driver's internal sample ring for finer clock/power
        history than our poll interval (best effort; unsupported on some
        driver/SKU combos)."""
        nv, h = self._nv, self._h
        for stype, kind in ((nv.NVML_PROCESSOR_CLK_SAMPLES, "clk"),
                            (nv.NVML_TOTAL_POWER_SAMPLES, "pw")):
            try:
                _, buf = nv.nvmlDeviceGetSamples(h, stype,
                                                 self._last_sample_ts)
            except Exception:
                continue
            for s in buf:
                v = s.sampleValue.uiVal
                self._samples.append(
                    (int(s.timeStamp) * 1000,
                     v if kind == "clk" else None,
                     v / 1000.0 if kind == "pw" else None,
                     0))
        self._samples.sort(key=lambda x: x[0])


class AmdSmiTelemetry(_PollingProvider):
    name = "amdsmi"

    def __init__(self, device_index: int = 0):
        super().__init__()
        import amdsmi
        self._a = amdsmi
        amdsmi.amdsmi_init()
        self._h = amdsmi.amdsmi_get_processor_handles()[device_index]
        try:
            info = amdsmi.amdsmi_get_clock_info(
                self._h, amdsmi.AmdSmiClkType.SYS)
            self._rated = int(info.get("max_clk") or 0) or None
        except Exception:
            self._rated = None

    def _sample(self):
        a, h = self._a, self._h
        clk = pw = None
        names: list[str] = []
        try:
            m = a.amdsmi_get_gpu_metrics_info(h)
            clks = m.get("current_gfxclks") or [m.get("current_gfxclk")]
            clks = [c for c in (clks or []) if c not in (None, 0xFFFF)]
            clk = max(clks) if clks else None
            pw = (m.get("current_socket_power")
                  or m.get("average_socket_power"))
            ts = m.get("throttle_status")
            if ts:  # nonzero hardware throttle bitmask
                names = [f"hw_throttle_status=0x{int(ts):x}"]
        except Exception:
            try:
                clk = a.amdsmi_get_clock_info(
                    h, a.AmdSmiClkType.SYS).get("clk")
            except Exception:
                pass
        return (time.time_ns(), clk, float(pw) if pw else None, names)


def get_provider(platform: str, device_index: int = 0):
    """Best available provider for the platform; never raises."""
    try:
        if platform == "nvidia":
            return NvmlTelemetry(device_index)
        if platform == "amd":
            return AmdSmiTelemetry(device_index)
    except Exception:
        pass
    return NullTelemetry()
