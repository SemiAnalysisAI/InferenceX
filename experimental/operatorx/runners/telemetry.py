"""Per-op GPU telemetry and the power-throttle retry policy.

A provider polls clocks, power, temperatures and throttle reasons every
OPERATORX_TELEMETRY_MS on a background thread while an op is timed, and
reads cumulative counters (energy, time spent in each throttle reason) at
the start and end of the window. An attempt is capped when a power/thermal
throttle reason is active while the SM clock sits more than
OPERATORX_CAP_CLOCK_FRACTION below rated boost; the power-cap reason at
full boost is ordinary DVFS and does not count. Capped attempts are
retried with growing inter-kernel sleeps (OPERATORX_RETRY_SLEEP_MS,
doubling) up to OPERATORX_THROTTLE_RETRIES times, and the attempt with
the lowest median is kept.

metrics["telemetry"] = {
  "provider", "interval_ms", "n_samples", "window_ms", "rated_sm_clock_mhz",
  <sample field>: {"min", "p50", "max"}, ...   # sm_clock_mhz, power_w, ...
  "counters": {...},                           # deltas over the window (100 ms steps)
  "throttle_reasons", "capped", "attempts", "inter_kernel_sleep_ms",
}

With OPERATORX_TELEMETRY_DIR set, the raw samples of every attempt are
appended to <dir>/telemetry-rank<RANK>.jsonl.
"""
from __future__ import annotations

import json
import os
import threading
import time

import torch

_INTERVAL_MS = float(os.environ.get("OPERATORX_TELEMETRY_MS", "5"))
_DIR = os.environ.get("OPERATORX_TELEMETRY_DIR")
_CAP_CLOCK_FRACTION = float(os.environ.get("OPERATORX_CAP_CLOCK_FRACTION", "0.985"))
_RETRIES = int(os.environ.get("OPERATORX_THROTTLE_RETRIES", "3"))
_RETRY_SLEEP_MS = float(os.environ.get("OPERATORX_RETRY_SLEEP_MS", "2"))

_NVML_REASONS = {
    0x001: "GpuIdle",
    0x002: "ApplicationsClocksSetting",
    0x004: "SwPowerCap",
    0x008: "HwSlowdown",
    0x010: "SyncBoost",
    0x020: "SwThermalSlowdown",
    0x040: "HwThermalSlowdown",
    0x080: "HwPowerBrakeSlowdown",
    0x100: "DisplayClockSetting",
}
_NVML_CAP_REASONS = {"SwPowerCap", "HwSlowdown", "SwThermalSlowdown",
                     "HwThermalSlowdown", "HwPowerBrakeSlowdown"}
# cumulative nanosecond counters (field values), read at window start/end
_NVML_TIME_COUNTERS = {
    "throttle_sw_power_cap_ms": "NVML_FI_DEV_CLOCKS_EVENT_REASON_SW_POWER_CAP",
    "throttle_hw_slowdown_ms": "NVML_FI_DEV_CLOCKS_EVENT_REASON_HW_THERM_SLOWDOWN",
    "throttle_sw_thermal_ms": "NVML_FI_DEV_CLOCKS_EVENT_REASON_SW_THERM_SLOWDOWN",
    "throttle_hw_power_brake_ms": "NVML_FI_DEV_CLOCKS_EVENT_REASON_HW_POWER_BRAKE_SLOWDOWN",
    "throttle_sync_boost_ms": "NVML_FI_DEV_CLOCKS_EVENT_REASON_SYNC_BOOST",
    "limit_power_ms": "NVML_FI_DEV_PERF_POLICY_POWER",
    "limit_thermal_ms": "NVML_FI_DEV_PERF_POLICY_THERMAL",
    "limit_board_ms": "NVML_FI_DEV_PERF_POLICY_BOARD_LIMIT",
    "limit_reliability_ms": "NVML_FI_DEV_PERF_POLICY_RELIABILITY",
    "limit_low_utilization_ms": "NVML_FI_DEV_PERF_POLICY_LOW_UTILIZATION",
    "below_app_clocks_ms": "NVML_FI_DEV_PERF_POLICY_TOTAL_APP_CLOCKS",
    "below_base_clocks_ms": "NVML_FI_DEV_PERF_POLICY_TOTAL_BASE_CLOCKS",
}
# The driver advances these counters (and energy) in 100 ms steps, so deltas
# are only meaningful for windows much longer than that.
_COUNTER_MIN_WINDOW_NS = 1_000_000_000


class _Provider:
    """Samples are dicts with "t_ns" plus whichever fields the device reports."""

    name = "null"
    rated_mhz: int | None = None
    cap_reasons: set[str] | None = None  # None: every reported reason caps

    def __init__(self) -> None:
        self._samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0_ns = 0

    def start(self) -> None:
        self._samples = []
        self._t0_ns = time.time_ns()
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> list[dict]:
        self._stop.set()
        self._thread.join()
        self._samples.extend(self._history(self._t0_ns))
        self._samples.sort(key=lambda s: s["t_ns"])
        return self._samples

    def counters(self) -> dict[str, float]:
        return {}

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                self._samples.append(self._sample())
            except Exception:
                pass
            self._stop.wait(_INTERVAL_MS / 1e3)

    def _sample(self) -> dict:
        raise NotImplementedError

    def _history(self, since_ns: int) -> list[dict]:
        return []


class _Null(_Provider):
    def start(self) -> None:
        pass

    def stop(self) -> list[dict]:
        return []


class _Nvml(_Provider):
    # FIXME(hbarclay): switch to DCGM for ~1 ms sampling.
    name = "nvml"
    cap_reasons = _NVML_CAP_REASONS

    def __init__(self, device: int) -> None:
        super().__init__()
        import pynvml as nv
        nv.nvmlInit()
        # NVML ignores CUDA_VISIBLE_DEVICES, so match the device by UUID.
        uuid = str(torch.cuda.get_device_properties(device).uuid)
        self._nv = nv
        self._h = nv.nvmlDeviceGetHandleByUUID(uuid if uuid.startswith("GPU-") else f"GPU-{uuid}")
        self._reasons = getattr(nv, "nvmlDeviceGetCurrentClocksEventReasons", None) \
            or nv.nvmlDeviceGetCurrentClocksThrottleReasons
        self.rated_mhz = nv.nvmlDeviceGetMaxClockInfo(self._h, nv.NVML_CLOCK_SM)
        self._fast_fields = [nv.NVML_FI_DEV_POWER_INSTANT, nv.NVML_FI_DEV_MEMORY_TEMP]
        self._counter_fields = {k: getattr(nv, f) for k, f in _NVML_TIME_COUNTERS.items()
                                if hasattr(nv, f)}
        self._counter_fields["energy_j"] = nv.NVML_FI_DEV_TOTAL_ENERGY_CONSUMPTION

    def _fields(self, ids: list[int]) -> list:
        out = []
        for v in self._nv.nvmlDeviceGetFieldValues(self._h, ids):
            if v.nvmlReturn != 0:
                out.append(None)
            elif v.valueType == 1:
                out.append(v.value.uiVal)
            else:
                out.append(v.value.ullVal)
        return out

    def _sample(self) -> dict:
        nv, h = self._nv, self._h
        mask = self._reasons(h)
        util = nv.nvmlDeviceGetUtilizationRates(h)
        power_inst, mem_temp = self._fields(self._fast_fields)
        return {
            "t_ns": time.time_ns(),
            "sm_clock_mhz": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM),
            "gr_clock_mhz": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_GRAPHICS),
            "mem_clock_mhz": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM),
            "video_clock_mhz": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_VIDEO),
            "power_w": nv.nvmlDeviceGetPowerUsage(h) / 1e3,
            "power_instant_w": power_inst / 1e3 if power_inst is not None else None,
            "gpu_temp_c": nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU),
            "mem_temp_c": mem_temp,
            "pstate": nv.nvmlDeviceGetPerformanceState(h),
            "gpu_util_pct": util.gpu,
            "mem_util_pct": util.memory,
            "reasons": [n for bit, n in _NVML_REASONS.items() if mask & bit],
        }

    def counters(self) -> dict[str, float]:
        vals = self._fields(list(self._counter_fields.values()))
        out = {}
        for key, v in zip(self._counter_fields, vals):
            if v is not None:
                out[key] = v / 1e3 if key == "energy_j" else v / 1e6  # mJ, ns
        return out

    def _history(self, since_ns: int) -> list[dict]:
        """The driver's own sample buffers for the window."""
        nv, out = self._nv, []
        for kind, key, scale in ((nv.NVML_TOTAL_POWER_SAMPLES, "power_w", 1e-3),
                                 (nv.NVML_GPU_UTILIZATION_SAMPLES, "gpu_util_pct", 1),
                                 (nv.NVML_MEMORY_UTILIZATION_SAMPLES, "mem_util_pct", 1)):
            try:
                _, buf = nv.nvmlDeviceGetSamples(self._h, kind, since_ns // 1000)
            except nv.NVMLError:
                continue
            for s in buf:
                t = s.timeStamp * 1000
                if t >= since_ns:
                    out.append({"t_ns": t, key: s.sampleValue.uiVal * scale, "source": "driver"})
        return out


class _AmdSmi(_Provider):
    name = "amdsmi"

    def __init__(self, device: int) -> None:
        super().__init__()
        import amdsmi
        amdsmi.amdsmi_init()
        # amdsmi enumerates every GPU regardless of *_VISIBLE_DEVICES, so
        # match the device by PCI address.
        p = torch.cuda.get_device_properties(device)
        bdf = f"{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}."
        self._a = amdsmi
        self._h = next(h for h in amdsmi.amdsmi_get_processor_handles()
                       if amdsmi.amdsmi_get_gpu_device_bdf(h).lower().startswith(bdf))
        self.rated_mhz = amdsmi.amdsmi_get_clock_info(
            self._h, amdsmi.AmdSmiClkType.SYS).get("max_clk") or None

    def _sample(self) -> dict:
        m = self._a.amdsmi_get_gpu_metrics_info(self._h)

        def val(key):
            v = m.get(key)
            return None if v in (None, "N/A", 0xFFFF, 0xFFFFFFFF) else v

        def top(key):
            vals = [v for v in (m.get(key) or []) if v not in (None, "N/A", 0xFFFF)]
            return max(vals) if vals else None

        status = val("throttle_status")
        power = val("current_socket_power") or val("average_socket_power")
        return {
            "t_ns": time.time_ns(),
            "sm_clock_mhz": top("current_gfxclks") or val("current_gfxclk"),
            "mem_clock_mhz": val("current_uclk"),
            "soc_clock_mhz": top("current_socclks") or val("current_socclk"),
            "power_w": float(power) if power is not None else None,
            "gpu_temp_c": val("temperature_hotspot"),
            "mem_temp_c": val("temperature_mem"),
            "gpu_util_pct": val("average_gfx_activity"),
            "mem_util_pct": val("average_umc_activity"),
            "reasons": [f"throttle_status=0x{int(status):x}"] if status else [],
        }


_PROVIDER: _Provider | None = None


def _provider() -> _Provider:
    global _PROVIDER
    if _PROVIDER is None:
        cls = _AmdSmi if torch.version.hip else _Nvml
        try:
            _PROVIDER = cls(torch.cuda.current_device())
        except Exception:
            _PROVIDER = _Null()
    return _PROVIDER


def _envelope(values: list) -> dict:
    s = sorted(values)
    return {"min": s[0], "p50": s[len(s) // 2], "max": s[-1]}


def _summarize(provider: _Provider, samples: list[dict], counters: dict, window_ns: int) -> dict:
    rated = provider.rated_mhz
    fields = sorted({k for s in samples for k in s} - {"t_ns", "reasons", "source"})
    summary = {
        "provider": provider.name,
        "interval_ms": _INTERVAL_MS,
        "n_samples": len(samples),
        "window_ms": round(window_ns / 1e6, 3),
        "rated_sm_clock_mhz": rated,
    }
    for f in fields:
        vals = [s[f] for s in samples if s.get(f) is not None]
        if vals:
            summary[f] = _envelope([round(v, 1) if isinstance(v, float) else v for v in vals])
    if counters:
        summary["counters"] = {k: round(v, 3) for k, v in counters.items()}
        if "energy_j" in counters and window_ns >= _COUNTER_MIN_WINDOW_NS:
            summary["counters"]["avg_power_w"] = round(counters["energy_j"] / (window_ns / 1e9), 1)

    def caps(s):
        rs = set(s.get("reasons", []))
        return rs & provider.cap_reasons if provider.cap_reasons is not None else rs

    summary["throttle_reasons"] = sorted({r for s in samples for r in s.get("reasons", [])})
    summary["capped"] = rated is not None and any(
        caps(s) and s.get("sm_clock_mhz") is not None
        and s["sm_clock_mhz"] < rated * _CAP_CLOCK_FRACTION for s in samples)
    return summary


def _dump(op, attempt: int, summary: dict, samples: list[dict]) -> None:
    if not _DIR:
        return
    os.makedirs(_DIR, exist_ok=True)
    rec = {"op": {"type": op.type, "backend": op.backend, "args": dict(op.args)},
           "attempt": attempt, "counters": summary.get("counters"), "samples": samples}
    path = os.path.join(_DIR, f"telemetry-rank{os.environ.get('RANK', '0')}.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def measure(op, time_once) -> tuple[float, dict]:
    """Time op with time_once(sleep_s) -> median_us, retrying capped attempts.

    Returns the best median and its telemetry summary."""
    provider = _provider()
    best = None
    for attempt in range(1 + max(_RETRIES, 0)):
        sleep_s = _RETRY_SLEEP_MS * 2 ** (attempt - 1) / 1e3 if attempt else 0.0
        before = provider.counters()
        t0 = time.time_ns()
        provider.start()
        median_us = time_once(sleep_s)
        samples = provider.stop()
        t1 = time.time_ns()
        after = provider.counters()
        delta = {k: after[k] - before[k] for k in after if k in before}
        summary = _summarize(provider, samples, delta, t1 - t0)
        _dump(op, attempt, summary, samples)
        if best is None or median_us < best[0]:
            best = (median_us, summary, sleep_s)
        if not summary["capped"]:
            break
    median_us, summary, sleep_s = best
    summary["attempts"] = attempt + 1
    summary["inter_kernel_sleep_ms"] = sleep_s * 1e3
    return median_us, summary
