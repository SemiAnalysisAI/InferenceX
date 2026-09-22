"""Per-op GPU telemetry and the power-throttle retry policy.

A provider samples SM clock, power and throttle reasons on a background
thread while an op is timed. An attempt is capped when a power/thermal
throttle reason is active while the SM clock sits more than
OPERATORX_CAP_CLOCK_FRACTION below rated boost; the power-cap reason at
full boost is ordinary DVFS and does not count. Capped attempts are
retried with growing inter-kernel sleeps (OPERATORX_RETRY_SLEEP_MS,
doubling) up to OPERATORX_THROTTLE_RETRIES times, and the attempt with
the lowest median is kept.

metrics["telemetry"] = {
  "provider", "interval_ms", "n_samples",
  "sm_clock_mhz": {"min", "p50", "max"}, "rated_sm_clock_mhz",
  "power_w": {"min", "p50", "max"},
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

_NVML_CAP_REASONS = {
    0x04: "SwPowerCap",
    0x08: "HwSlowdown",
    0x20: "SwThermalSlowdown",
    0x40: "HwThermalSlowdown",
    0x80: "HwPowerBrakeSlowdown",
}


class _Provider:
    """Samples are (time_ns, sm_clock_mhz, power_w, throttle_reasons)."""

    name = "null"
    rated_mhz: int | None = None

    def __init__(self) -> None:
        self._samples: list = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0_ns = 0

    def start(self) -> None:
        self._samples = []
        self._t0_ns = time.time_ns()
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> list:
        self._stop.set()
        self._thread.join()
        self._samples.extend(self._history(self._t0_ns))
        self._samples.sort(key=lambda s: s[0])
        return self._samples

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                self._samples.append(self._sample())
            except Exception:
                pass
            self._stop.wait(_INTERVAL_MS / 1e3)

    def _sample(self) -> tuple:
        raise NotImplementedError

    def _history(self, since_ns: int) -> list:
        return []


class _Null(_Provider):
    def start(self) -> None:
        pass

    def stop(self) -> list:
        return []


class _Nvml(_Provider):
    # FIXME(hbarclay): switch to DCGM for ~1 ms sampling.
    name = "nvml"

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

    def _sample(self) -> tuple:
        nv, h = self._nv, self._h
        mask = self._reasons(h)
        return (time.time_ns(),
                nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM),
                nv.nvmlDeviceGetPowerUsage(h) / 1e3,
                [name for bit, name in _NVML_CAP_REASONS.items() if mask & bit])

    def _history(self, since_ns: int) -> list:
        """The driver's own finer-grained clock/power samples for the window."""
        nv, out = self._nv, []
        for kind in (nv.NVML_PROCESSOR_CLK_SAMPLES, nv.NVML_TOTAL_POWER_SAMPLES):
            try:
                _, buf = nv.nvmlDeviceGetSamples(self._h, kind, since_ns // 1000)
            except nv.NVMLError:
                continue
            for s in buf:
                t, v = s.timeStamp * 1000, s.sampleValue.uiVal
                if t < since_ns:
                    continue
                if kind == nv.NVML_PROCESSOR_CLK_SAMPLES:
                    out.append((t, v, None, []))
                else:
                    out.append((t, None, v / 1e3, []))
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

    def _sample(self) -> tuple:
        m = self._a.amdsmi_get_gpu_metrics_info(self._h)
        clocks = [c for c in (m.get("current_gfxclks") or [m.get("current_gfxclk")])
                  if c not in (None, "N/A", 0xFFFF)]
        power = m.get("current_socket_power") or m.get("average_socket_power")
        status = m.get("throttle_status")
        return (time.time_ns(),
                max(clocks) if clocks else None,
                float(power) if power not in (None, "N/A", 0xFFFF) else None,
                [f"throttle_status=0x{int(status):x}"] if status not in (None, "N/A", 0) else [])


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
    if not values:
        return {"min": None, "p50": None, "max": None}
    s = sorted(values)
    return {"min": s[0], "p50": s[len(s) // 2], "max": s[-1]}


def _summarize(provider: _Provider, samples: list) -> dict:
    rated = provider.rated_mhz
    return {
        "provider": provider.name,
        "interval_ms": _INTERVAL_MS,
        "n_samples": len(samples),
        "sm_clock_mhz": _envelope([s[1] for s in samples if s[1] is not None]),
        "rated_sm_clock_mhz": rated,
        "power_w": _envelope([round(s[2], 1) for s in samples if s[2] is not None]),
        "throttle_reasons": sorted({r for s in samples for r in s[3]}),
        "capped": rated is not None and any(
            s[3] and s[1] is not None and s[1] < rated * _CAP_CLOCK_FRACTION
            for s in samples),
    }


def _dump(op, attempt: int, provider: str, samples: list) -> None:
    if not _DIR:
        return
    os.makedirs(_DIR, exist_ok=True)
    rec = {"op": {"type": op.type, "backend": op.backend, "args": dict(op.args)},
           "attempt": attempt, "provider": provider, "samples": samples}
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
        provider.start()
        median_us = time_once(sleep_s)
        samples = provider.stop()
        summary = _summarize(provider, samples)
        _dump(op, attempt, provider.name, samples)
        if best is None or median_us < best[0]:
            best = (median_us, summary, sleep_s)
        if not summary["capped"]:
            break
    median_us, summary, sleep_s = best
    summary["attempts"] = attempt + 1
    summary["inter_kernel_sleep_ms"] = sleep_s * 1e3
    return median_us, summary
