"""Per-rank GPU clock telemetry for one case.

A GPU running below its peers' clock looks exactly like a slow library, so each case records the
clock every rank's GPU ran at over the timed passes and the share of that time its clocks were held
down. Reads happen only between timed bursts, never inside a window: NVML through the driver's own
`libnvidia-ml.so.1` (ctypes, no package) on NVIDIA, and the amdsmi package that ROCm torch imports
on AMD. Every read is best effort: a probe that cannot start records why, a read that fails is
skipped, and neither fails the case.
"""
from __future__ import annotations

import ctypes
import math
import statistics
import time

# Causes of a clock held below the requested clock, across vendors. Each is published as the share
# of the timed passes it was active (null where the GPU does not report it).
THROTTLE_CAUSES = ("power", "thermal", "hbm_thermal", "vr_thermal", "prochot", "sync_boost",
                   "board_limit", "reliability")
# nvmlClocksEventReasons bits. GPU idle (0x1) is left out: a read between bursts may show it.
CLOCK_EVENT_REASONS = (
    (0x2, "applications_clocks_setting"), (0x4, "sw_power_cap"), (0x8, "hw_slowdown"),
    (0x10, "sync_boost"), (0x20, "sw_thermal_slowdown"), (0x40, "hw_thermal_slowdown"),
    (0x80, "hw_power_brake_slowdown"),
)
# Reasons that never appear on a healthy GPU, whatever its load.
ABNORMAL_REASONS = {"hw_slowdown", "sw_thermal_slowdown", "hw_thermal_slowdown", "hw_power_brake_slowdown"}
SOURCES = {1: "nvml", 2: "amdsmi"}
# Ranks of one case run identical work, so their GPUs reach the same clock; a rank whose median
# falls this far below the fastest rank's ran on a slow GPU, not a slow library.
SLOW_GPU_RATIO = 0.95
# A rank its own limits held down for this share of the timed passes ran slow whatever its peers
# did. amdsmi reports no clock-event reasons, so on AMD a tray throttled as a whole shows only here;
# the busiest prefill cases measured spent 0.01-0.02% power-capped. sync_boost is left out: it holds
# a GPU to its sync group's clock, which the ratio already compares.
THROTTLED_FRACTION = 0.05
SLOWING_CAUSES = tuple(cause for cause in THROTTLE_CAUSES if cause != "sync_boost")
# One float per field crosses ranks in a single all_gather; NaN marks a value the rank could not read.
FIELDS = ("source", "sm_clock_min", "sm_clock_p50", "sm_clock_max", "sm_clock_samples",
          "sm_clock_limit", "power_limit_w", "temperature_max_c", "reasons", "span_s",
          *(f"throttle_{cause}" for cause in THROTTLE_CAUSES))


class _ViolationTime(ctypes.Structure):
    _fields_ = [("referenceTime", ctypes.c_ulonglong), ("violationTime", ctypes.c_ulonglong)]


class _Nvml:
    """The few NVML calls the monitor needs, bound to the driver's library by ctypes."""

    code = 1
    _CLOCK_SM = 1
    _TEMPERATURE_GPU = 0
    # nvmlPerfPolicyType_t values: cumulative nanoseconds the driver held the clocks down, by cause.
    POLICIES = (("power", 0), ("thermal", 1), ("sync_boost", 2), ("board_limit", 3), ("reliability", 5))

    def __init__(self, torch, device):
        uuid = str(torch.cuda.get_device_properties(device).uuid)
        # NVML names a GPU "GPU-<uuid>"; torch prints the bare uuid.
        uuid = uuid if uuid.startswith(("GPU-", "MIG-")) else f"GPU-{uuid}"
        lib = ctypes.CDLL("libnvidia-ml.so.1")
        self._check(lib.nvmlInit_v2(), "nvmlInit_v2")
        handle = ctypes.c_void_p()
        self._check(lib.nvmlDeviceGetHandleByUUID(uuid.encode(), ctypes.byref(handle)),
                    "nvmlDeviceGetHandleByUUID")
        self.lib, self.handle = lib, handle
        # Renamed from ...ThrottleReasons in NVML 12.2; older drivers export only the old name.
        self._reasons = (getattr(lib, "nvmlDeviceGetCurrentClocksEventReasons", None)
                         or getattr(lib, "nvmlDeviceGetCurrentClocksThrottleReasons", None))

    @staticmethod
    def _check(rc, call):
        if rc != 0:
            raise OSError(f"{call} returned NVML error {rc}")

    def _read(self, fn, ctype, *args):
        value = ctype()
        return value.value if fn(self.handle, *args, ctypes.byref(value)) == 0 else None

    def sm_clock(self):
        return self._read(self.lib.nvmlDeviceGetClockInfo, ctypes.c_uint, ctypes.c_int(self._CLOCK_SM))

    def sm_clock_limit(self):
        return self._read(self.lib.nvmlDeviceGetMaxClockInfo, ctypes.c_uint, ctypes.c_int(self._CLOCK_SM))

    def power_limit_w(self):
        milliwatts = self._read(self.lib.nvmlDeviceGetEnforcedPowerLimit, ctypes.c_uint)
        return None if milliwatts is None else milliwatts / 1e3

    def temperature(self):
        return self._read(self.lib.nvmlDeviceGetTemperature, ctypes.c_uint,
                          ctypes.c_int(self._TEMPERATURE_GPU))

    def reasons(self):
        return None if self._reasons is None else self._read(self._reasons, ctypes.c_ulonglong)

    def throttle_counters(self):
        """({cause: nanoseconds throttled}, nanoseconds elapsed), cumulative since driver load."""
        counters = {}
        for cause, policy in self.POLICIES:
            sample = _ViolationTime()
            rc = self.lib.nvmlDeviceGetViolationStatus(self.handle, ctypes.c_int(policy), ctypes.byref(sample))
            counters[cause] = sample.violationTime if rc == 0 else None
        return counters, time.monotonic_ns()


class _AmdSmi:
    """ROCm: the amdsmi package, which torch's ROCm build imports for its own device queries."""

    code = 2
    # Throttle-residency accumulators in the GPU metrics table. `accumulation_counter` ticks
    # alongside them, so a residency delta over the counter delta is the share of time throttled.
    RESIDENCY = (("power", "ppt_residency_acc"), ("thermal", "socket_thm_residency_acc"),
                 ("hbm_thermal", "hbm_thm_residency_acc"), ("vr_thermal", "vr_thm_residency_acc"),
                 ("prochot", "prochot_residency_acc"))

    def __init__(self, torch, device):
        import amdsmi

        props = torch.cuda.get_device_properties(device)
        bdf = f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"
        amdsmi.amdsmi_init()
        handles = [handle for handle in amdsmi.amdsmi_get_processor_handles()
                   if amdsmi.amdsmi_get_gpu_device_bdf(handle).lower() == bdf]
        if len(handles) != 1:
            raise OSError(f"amdsmi lists {len(handles)} GPUs at {bdf}")
        self.amdsmi, self.handle = amdsmi, handles[0]

    def _call(self, name, *args):
        try:
            return getattr(self.amdsmi, name)(self.handle, *args)
        except Exception:  # noqa: BLE001 - AmdSmiException, or a query this GPU does not support
            return None

    def _clock_info(self):
        return self._call("amdsmi_get_clock_info", self.amdsmi.AmdSmiClkType.GFX) or {}

    def sm_clock(self):
        return _number(self._clock_info().get("clk"))

    def sm_clock_limit(self):
        return _number(self._clock_info().get("max_clk"))

    def power_limit_w(self):
        microwatts = _number((self._call("amdsmi_get_power_cap_info") or {}).get("power_cap"))
        return None if microwatts is None else microwatts / 1e6

    def temperature(self):
        return _number((self._call("amdsmi_get_gpu_metrics_info") or {}).get("temperature_hotspot"))

    def reasons(self):
        return None

    def throttle_counters(self):
        metrics = self._call("amdsmi_get_gpu_metrics_info") or {}
        return ({cause: _number(metrics.get(key)) for cause, key in self.RESIDENCY},
                _number(metrics.get("accumulation_counter")))


def _number(value):
    """amdsmi reports an unsupported field as the string "N/A"."""
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


class ClockMonitor:
    """One rank's clock readings for one case: `start()` before the first timed pass, `sample()`
    right after timed bursts, while the GPU still runs at the clock the burst ran at, and `finish()`
    after the last timed sample. `vector()` is this rank's share of the cross-rank gather."""

    def __init__(self, torch, device):
        self.probe = None
        self.reason = None
        try:
            hip = getattr(getattr(torch, "version", None), "hip", None)
            self.probe = (_AmdSmi if hip else _Nvml)(torch, device)
        except Exception as error:  # noqa: BLE001 - telemetry is best effort; the case runs without it
            self.reason = f"{type(error).__name__}: {error}"
        self.clocks, self.temperatures, self.reasons = [], [], 0
        self.limit = self.power_limit = self.span_s = None
        self._started = None
        self._counters = None
        self.throttle = {}

    def start(self):
        if self.probe is None:
            return
        self._started = time.monotonic()
        self.limit = self.probe.sm_clock_limit()
        self.power_limit = self.probe.power_limit_w()
        self._counters = self.probe.throttle_counters()
        self._temperature()

    def sample(self, temperature: bool = False):
        if self.probe is None:
            return
        clock = self.probe.sm_clock()
        if clock is not None:
            self.clocks.append(float(clock))
        reasons = self.probe.reasons()
        if reasons:
            self.reasons |= int(reasons)
        if temperature:
            self._temperature()

    def finish(self):
        if self.probe is None or self._started is None:
            return
        self._temperature()
        self.span_s = time.monotonic() - self._started
        (start, start_total), (end, end_total) = self._counters, self.probe.throttle_counters()
        elapsed = None if start_total is None or end_total is None else end_total - start_total
        self.throttle = {
            cause: (end[cause] - start[cause]) / elapsed
            if elapsed and start.get(cause) is not None and end.get(cause) is not None else None
            for cause in start
        }

    def _temperature(self):
        value = self.probe.temperature()
        if value is not None:
            self.temperatures.append(float(value))

    def vector(self) -> list:
        clocks = sorted(self.clocks)
        values = {
            "source": float(self.probe.code) if self.probe is not None else 0.0,
            "sm_clock_min": clocks[0] if clocks else None,
            "sm_clock_p50": statistics.median(clocks) if clocks else None,
            "sm_clock_max": clocks[-1] if clocks else None,
            "sm_clock_samples": float(len(clocks)),
            "sm_clock_limit": self.limit,
            "power_limit_w": self.power_limit,
            "temperature_max_c": max(self.temperatures) if self.temperatures else None,
            "reasons": float(self.reasons),
            "span_s": self.span_s,
            **{f"throttle_{cause}": self.throttle.get(cause) for cause in THROTTLE_CAUSES},
        }
        return [math.nan if values[field] is None else float(values[field]) for field in FIELDS]


def _reason_names(bits) -> list:
    if bits is None:
        return []
    return [name for bit, name in CLOCK_EVENT_REASONS if int(bits) & bit]


def _round(value, digits=0):
    if value is None:
        return None
    return round(value) if digits == 0 else round(value, digits)


def summarize(vectors, unavailable_reason=None) -> dict:
    """`runtime.gpu_clocks` from every rank's `vector()`, in rank order. `sm_clock_ratio_min` is the
    slowest rank's median clock over the fastest's: well below 1 marks a GPU slower than its peers."""
    per_rank = []
    for rank, vector in enumerate(vectors):
        values = {field: (None if math.isnan(value) else value) for field, value in zip(FIELDS, vector)}
        samples = int(values["sm_clock_samples"] or 0)
        throttle = {cause: _round(values[f"throttle_{cause}"], 4) for cause in THROTTLE_CAUSES}
        per_rank.append({
            "rank": rank,
            "source": SOURCES.get(int(values["source"] or 0)),
            "sm_clock_mhz": {
                "min": _round(values["sm_clock_min"]),
                "p50": _round(values["sm_clock_p50"]),
                "max": _round(values["sm_clock_max"]),
                "samples": samples,
            } if samples else None,
            "sm_clock_limit_mhz": _round(values["sm_clock_limit"]),
            "power_limit_w": _round(values["power_limit_w"], 1),
            "temperature_max_c": _round(values["temperature_max_c"]),
            "clock_event_reasons": _reason_names(values["reasons"]),
            "throttle_fraction": throttle if any(value is not None for value in throttle.values()) else None,
            "span_s": _round(values["span_s"], 1),
        })
    sources = {rank["source"] for rank in per_rank}
    medians = [rank["sm_clock_mhz"]["p50"] for rank in per_rank if rank["sm_clock_mhz"]]
    worst = {}
    for cause in THROTTLE_CAUSES:
        observed = [rank["throttle_fraction"][cause] for rank in per_rank
                    if rank["throttle_fraction"] and rank["throttle_fraction"][cause] is not None]
        worst[cause] = max(observed) if observed else None
    source = next(iter(sources)) if len(sources) == 1 else ("mixed" if sources else None)
    # Only the writing rank's own failure text reaches here, so any other rank whose probe did not
    # start is named instead.
    missing = [rank["rank"] for rank in per_rank if rank["source"] is None]
    unnamed = [rank for rank in missing if rank != 0 or not unavailable_reason]
    reasons = ([unavailable_reason] if unavailable_reason and (0 in missing or not per_rank) else [])
    if unnamed:
        reasons.append("no clock probe on rank " + ", ".join(str(rank) for rank in unnamed))
    # Null unless every rank reported: a subset could hide exactly the GPU that failed to.
    ratio = (round(min(medians) / max(medians), 4)
             if per_rank and len(medians) == len(per_rank) and max(medians) > 0 else None)
    return {
        "source": source,
        "unavailable_reason": "; ".join(reasons) or None,
        "sm_clock_ratio_min": ratio,
        "throttle_fraction_max": worst if any(value is not None for value in worst.values()) else None,
        # True when a GPU ran slower than its peers, reported a hardware or thermal slowdown, or spent
        # THROTTLED_FRACTION of the timed passes held down by its own limits.
        "slow_gpu": bool(_problems(ratio, per_rank)) if source else None,
        "per_rank": per_rank,
    }


def _problems(ratio, per_rank) -> list:
    problems = []
    if ratio is not None and ratio < SLOW_GPU_RATIO:
        problems.append(f"the slowest rank's GPU ran at {ratio:.0%} of the fastest rank's SM clock")
    slowed = sorted({(rank["rank"], reason) for rank in per_rank
                     for reason in rank["clock_event_reasons"] if reason in ABNORMAL_REASONS})
    if slowed:
        problems.append("ranks reported " + ", ".join(f"{reason} (rank {rank})" for rank, reason in slowed))
    held = sorted((rank["rank"], cause, fraction) for rank in per_rank
                  for cause, fraction in (rank.get("throttle_fraction") or {}).items()
                  if cause in SLOWING_CAUSES and fraction is not None and fraction >= THROTTLED_FRACTION)
    if held:
        problems.append("ranks were throttled for " + ", ".join(
            f"{fraction:.0%} of the timed passes by {cause} (rank {rank})" for rank, cause, fraction in held))
    return problems


def slow_gpu_warning(summary) -> str | None:
    """The run log's line for a case whose summary says `slow_gpu`."""
    summary = summary or {}
    problems = _problems(summary.get("sm_clock_ratio_min"), summary.get("per_rank", []))
    if not problems:
        return None
    return f"WARNING: {'; '.join(problems)}. Check runtime.gpu_clocks before comparing this case."
