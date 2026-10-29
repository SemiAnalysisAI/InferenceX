"""GPU clock telemetry: the vendor probes, the per-rank monitor and the published summary."""
from __future__ import annotations

import math
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "bench")]

import gpu_clocks  # noqa: E402
import summarize  # noqa: E402


def rank_vector(**values):
    """One rank's gathered vector, NaN wherever the rank could not read a value."""
    return [math.nan if values.get(field) is None else float(values[field]) for field in gpu_clocks.FIELDS]


def nvml_rank(p50, **extra):
    values = {"source": 1, "sm_clock_min": p50 - 15, "sm_clock_p50": p50, "sm_clock_max": p50 + 15,
              "sm_clock_samples": 40, "sm_clock_limit": 1980, "power_limit_w": 700,
              "temperature_max_c": 61, "reasons": 0, "span_s": 812.4}
    return rank_vector(**{**values, **extra})


class Summary(unittest.TestCase):
    def test_a_slow_gpu_shows_as_a_low_ratio_and_a_warning(self):
        summary = gpu_clocks.summarize([nvml_rank(1980), nvml_rank(1410, reasons=0x4 | 0x1)])
        self.assertEqual(summary["source"], "nvml")
        self.assertAlmostEqual(summary["sm_clock_ratio_min"], 1410 / 1980, places=4)
        self.assertTrue(summary["slow_gpu"])
        # The idle bit (0x1) is dropped; a power cap is the reason that explains a low clock.
        self.assertEqual(summary["per_rank"][1]["clock_event_reasons"], ["sw_power_cap"])
        self.assertIn("71%", gpu_clocks.slow_gpu_warning(summary))

    def test_matched_healthy_clocks_raise_nothing(self):
        summary = gpu_clocks.summarize([nvml_rank(1980), nvml_rank(1965, reasons=0x2)])
        self.assertGreater(summary["sm_clock_ratio_min"], gpu_clocks.SLOW_GPU_RATIO)
        self.assertIs(summary["slow_gpu"], False)
        self.assertIsNone(gpu_clocks.slow_gpu_warning(summary))

    def test_a_hardware_slowdown_is_flagged_even_when_every_rank_matches(self):
        # A whole tray clamped together keeps the ratio at 1; its reason bits still give it away.
        summary = gpu_clocks.summarize([nvml_rank(1200, reasons=0x80), nvml_rank(1200, reasons=0x80)])
        self.assertEqual(summary["sm_clock_ratio_min"], 1.0)
        self.assertTrue(summary["slow_gpu"])
        self.assertIn("hw_power_brake_slowdown (rank 0)", gpu_clocks.slow_gpu_warning(summary))

    def test_a_tray_throttled_as_a_whole_is_flagged_where_no_reason_bits_exist(self):
        # amdsmi reports no clock-event reasons and the ratio stays at 1, so the residency decides.
        for power, slow in ((0.12, True), (0.0002, False)):
            with self.subTest(power=power):
                ranks = [nvml_rank(1500, source=2, reasons=None, throttle_power=power,
                                   throttle_sync_boost=0.5) for _ in range(2)]
                summary = gpu_clocks.summarize(ranks)
                self.assertEqual((summary["source"], summary["sm_clock_ratio_min"]), ("amdsmi", 1.0))
                self.assertIs(summary["slow_gpu"], slow)
                warning = gpu_clocks.slow_gpu_warning(summary)
                if slow:
                    self.assertIn("12% of the timed passes by power (rank 0)", warning)
                    self.assertNotIn("sync_boost", warning)
                else:
                    self.assertIsNone(warning)

    def test_throttle_reports_the_worst_rank_per_cause(self):
        summary = gpu_clocks.summarize([
            nvml_rank(1980, throttle_power=0.0, throttle_thermal=0.0),
            nvml_rank(1975, throttle_power=0.0125, throttle_thermal=0.00025),
        ])
        self.assertEqual(summary["throttle_fraction_max"]["power"], 0.0125)
        self.assertEqual(summary["throttle_fraction_max"]["thermal"], 0.0003)
        self.assertIsNone(summary["throttle_fraction_max"]["hbm_thermal"])

    def test_a_rank_without_readings_withholds_the_ratio(self):
        summary = gpu_clocks.summarize([nvml_rank(1980), rank_vector(source=1, sm_clock_samples=0)])
        self.assertIsNone(summary["sm_clock_ratio_min"])
        self.assertIsNone(summary["per_rank"][1]["sm_clock_mhz"])

    def test_an_unavailable_probe_says_why(self):
        summary = gpu_clocks.summarize([rank_vector(source=0, sm_clock_samples=0)],
                                       "OSError: libnvidia-ml.so.1: cannot open shared object file")
        self.assertIsNone(summary["source"])
        self.assertIn("libnvidia-ml", summary["unavailable_reason"])
        self.assertIsNone(summary["sm_clock_ratio_min"])
        self.assertIsNone(summary["throttle_fraction_max"])
        self.assertIsNone(summary["slow_gpu"])

    def test_a_probe_missing_on_another_rank_is_named(self):
        # The writing rank's probe ran; rank 1's did not, and its failure text never left rank 1.
        summary = gpu_clocks.summarize([nvml_rank(1980), rank_vector(source=0, sm_clock_samples=0)])
        self.assertEqual(summary["source"], "mixed")
        self.assertEqual(summary["unavailable_reason"], "no clock probe on rank 1")
        both = gpu_clocks.summarize([rank_vector(source=0, sm_clock_samples=0)] * 2, "OSError: no NVML")
        self.assertEqual(both["unavailable_reason"], "OSError: no NVML; no clock probe on rank 1")
        self.assertIsNone(gpu_clocks.summarize([nvml_rank(1980)] * 2, "unused")["unavailable_reason"])

    def test_nothing_identifies_the_hardware(self):
        # Results are public: per-rank entries carry readings, never a UUID, serial or bus id.
        summary = gpu_clocks.summarize([nvml_rank(1980)])
        self.assertEqual(set(summary["per_rank"][0]), {
            "rank", "source", "sm_clock_mhz", "sm_clock_limit_mhz", "power_limit_w",
            "temperature_max_c", "clock_event_reasons", "throttle_fraction", "span_s",
        })


class _FakeProbe:
    code = 1

    def __init__(self, torch, device):
        self.clocks = iter([1965.0, 1980.0, 1410.0])
        # 10 ms of power throttling over 1 s.
        self.counters = iter([({"power": 1_000_000, "thermal": 0}, 5_000_000_000),
                              ({"power": 11_000_000, "thermal": 0}, 6_000_000_000)])

    def sm_clock(self):
        return next(self.clocks)

    def sm_clock_limit(self):
        return 1980

    def power_limit_w(self):
        return 700.0

    def temperature(self):
        return 58

    def reasons(self):
        return 0x4

    def throttle_counters(self):
        return next(self.counters)


class Monitor(unittest.TestCase):
    def test_readings_become_this_ranks_vector(self):
        torch = SimpleNamespace(version=SimpleNamespace(hip=None))
        with mock.patch.object(gpu_clocks, "_Nvml", _FakeProbe), \
                mock.patch.object(gpu_clocks.time, "monotonic", side_effect=[100.0, 912.5]):
            monitor = gpu_clocks.ClockMonitor(torch, "cuda:3")
            monitor.start()
            for _ in range(3):
                monitor.sample()
            monitor.finish()
        rank = gpu_clocks.summarize([monitor.vector()])["per_rank"][0]
        self.assertEqual(rank["sm_clock_mhz"], {"min": 1410, "p50": 1965, "max": 1980, "samples": 3})
        self.assertEqual(rank["throttle_fraction"]["power"], 0.01)
        self.assertEqual(rank["throttle_fraction"]["thermal"], 0.0)
        self.assertIsNone(rank["throttle_fraction"]["hbm_thermal"])
        self.assertEqual(rank["span_s"], 812.5)
        self.assertEqual(rank["clock_event_reasons"], ["sw_power_cap"])

    def test_a_probe_that_cannot_start_records_the_reason(self):
        torch = SimpleNamespace(cuda=SimpleNamespace())  # no device properties, as in the CPU fakes
        monitor = gpu_clocks.ClockMonitor(torch, "cuda:0")
        monitor.start()
        monitor.sample()
        monitor.finish()
        self.assertIsNone(monitor.probe)
        self.assertIn("AttributeError", monitor.reason)
        self.assertEqual(monitor.vector()[gpu_clocks.FIELDS.index("source")], 0.0)


def _fake_amdsmi(bdfs, metrics):
    """The amdsmi calls the probe makes, as the ROCm 7.2 package answers them on an MI325X."""
    module = types.ModuleType("amdsmi")
    module.AmdSmiClkType = SimpleNamespace(GFX="gfx")
    module.amdsmi_init = lambda: None
    module.amdsmi_get_processor_handles = lambda: list(bdfs)
    module.amdsmi_get_gpu_device_bdf = lambda handle: handle
    module.amdsmi_get_clock_info = lambda handle, kind: {
        "clk": 1540, "min_clk": 500, "max_clk": 2100, "clk_locked": False, "clk_deep_sleep": 133}
    module.amdsmi_get_power_cap_info = lambda handle: {"power_cap": 1_000_000_000}
    module.amdsmi_get_gpu_metrics_info = lambda handle: next(metrics)
    return module


def _metrics(counter, ppt, temperature=49):
    return {"temperature_hotspot": temperature, "throttle_status": "N/A", "accumulation_counter": counter,
            "ppt_residency_acc": ppt, "socket_thm_residency_acc": 0, "hbm_thm_residency_acc": 0,
            "vr_thm_residency_acc": "N/A", "prochot_residency_acc": 0}


class AmdSmiProbe(unittest.TestCase):
    def rocm_torch(self, bus=0x09):
        props = SimpleNamespace(pci_domain_id=0, pci_bus_id=bus, pci_device_id=0)
        return SimpleNamespace(version=SimpleNamespace(hip="7.2"),
                               cuda=SimpleNamespace(get_device_properties=lambda device: props))

    def test_reads_the_gpu_at_this_ranks_bus_id(self):
        metrics = iter([_metrics(1000, 50), _metrics(1000, 50), _metrics(1000, 50),
                        _metrics(3000, 250, temperature=58), _metrics(3000, 250)])
        amdsmi = _fake_amdsmi(["0000:05:00.0", "0000:09:00.0"], metrics)
        with mock.patch.dict(sys.modules, {"amdsmi": amdsmi}):
            monitor = gpu_clocks.ClockMonitor(self.rocm_torch(), "cuda:1")
            monitor.start()
            monitor.sample(temperature=True)
            monitor.finish()
        self.assertEqual(monitor.probe.handle, "0000:09:00.0")
        rank = gpu_clocks.summarize([monitor.vector()])["per_rank"][0]
        self.assertEqual(rank["source"], "amdsmi")
        self.assertEqual(rank["sm_clock_mhz"]["p50"], 1540)
        self.assertEqual(rank["sm_clock_limit_mhz"], 2100)
        self.assertEqual(rank["power_limit_w"], 1000.0)  # amdsmi reports microwatts
        self.assertEqual(rank["temperature_max_c"], 58)
        # 200 residency ticks over 2000 counter ticks; "N/A" stays null.
        self.assertEqual(rank["throttle_fraction"]["power"], 0.1)
        self.assertIsNone(rank["throttle_fraction"]["vr_thermal"])
        self.assertIsNone(rank["throttle_fraction"]["sync_boost"])
        self.assertEqual(rank["clock_event_reasons"], [])

    def test_no_gpu_at_the_bus_id_is_unavailable(self):
        amdsmi = _fake_amdsmi(["0000:05:00.0"], iter([]))
        with mock.patch.dict(sys.modules, {"amdsmi": amdsmi}):
            monitor = gpu_clocks.ClockMonitor(self.rocm_torch(), "cuda:0")
        self.assertIsNone(monitor.probe)
        self.assertIn("0000:09:00.0", monitor.reason)

    def test_a_failing_query_reads_as_missing(self):
        amdsmi = _fake_amdsmi(["0000:09:00.0"], iter([]))

        def unsupported(handle, kind):
            raise RuntimeError("AMDSMI_STATUS_NOT_SUPPORTED")

        amdsmi.amdsmi_get_clock_info = unsupported
        with mock.patch.dict(sys.modules, {"amdsmi": amdsmi}):
            probe = gpu_clocks._AmdSmi(self.rocm_torch(), "cuda:0")
        self.assertIsNone(probe.sm_clock())
        self.assertEqual(probe.throttle_counters(), ({cause: None for cause, _ in probe.RESIDENCY}, None))


class _FakeNvmlLibrary:
    """libnvidia-ml.so.1 for the ctypes binding: each call fills its out-parameter through byref."""

    def __init__(self, init_rc=0):
        self.init_rc = init_rc
        self.uuid = None

    def nvmlInit_v2(self):
        return self.init_rc

    def nvmlDeviceGetHandleByUUID(self, uuid, handle):
        self.uuid = uuid
        handle._obj.value = 0xD00D
        return 0

    def nvmlDeviceGetClockInfo(self, handle, clock_type, out):
        out._obj.value = {1: 1965}[clock_type.value]
        return 0

    def nvmlDeviceGetMaxClockInfo(self, handle, clock_type, out):
        out._obj.value = 1980
        return 0

    def nvmlDeviceGetEnforcedPowerLimit(self, handle, out):
        out._obj.value = 700_000
        return 0

    def nvmlDeviceGetTemperature(self, handle, sensor, out):
        out._obj.value = 61
        return 0

    def nvmlDeviceGetCurrentClocksEventReasons(self, handle, out):
        out._obj.value = 0x4 | 0x1
        return 0

    def nvmlDeviceGetViolationStatus(self, handle, policy, out):
        if policy.value == 5:  # reliability: NVML_ERROR_NOT_SUPPORTED on this part
            return 3
        out._obj.violationTime = {0: 4_000_000}.get(policy.value, 0)
        return 0


class NvmlProbe(unittest.TestCase):
    def cuda_torch(self, uuid):
        props = SimpleNamespace(uuid=uuid)
        return SimpleNamespace(cuda=SimpleNamespace(get_device_properties=lambda device: props))

    def test_out_parameters_and_units(self):
        library = _FakeNvmlLibrary()
        with mock.patch.object(gpu_clocks.ctypes, "CDLL", lambda name: library):
            nvml = gpu_clocks._Nvml(self.cuda_torch("5e5c5a8b-0000-1111-2222-333344445555"), "cuda:0")
        # NVML names devices "GPU-<uuid>"; torch prints the bare uuid.
        self.assertEqual(library.uuid, b"GPU-5e5c5a8b-0000-1111-2222-333344445555")
        self.assertEqual(nvml.sm_clock(), 1965)
        self.assertEqual(nvml.sm_clock_limit(), 1980)
        self.assertEqual(nvml.power_limit_w(), 700.0)  # NVML reports milliwatts
        self.assertEqual(nvml.temperature(), 61)
        self.assertEqual(nvml.reasons(), 0x5)
        counters, elapsed_ns = nvml.throttle_counters()
        self.assertEqual(counters["power"], 4_000_000)
        self.assertIsNone(counters["reliability"])
        self.assertIsInstance(elapsed_ns, int)

    def test_an_already_prefixed_uuid_is_kept(self):
        library = _FakeNvmlLibrary()
        with mock.patch.object(gpu_clocks.ctypes, "CDLL", lambda name: library):
            gpu_clocks._Nvml(self.cuda_torch("GPU-5e5c5a8b"), "cuda:0")
        self.assertEqual(library.uuid, b"GPU-5e5c5a8b")

    def test_a_failed_init_is_an_oserror(self):
        with mock.patch.object(gpu_clocks.ctypes, "CDLL", lambda name: _FakeNvmlLibrary(init_rc=9)), \
                self.assertRaises(OSError):
            gpu_clocks._Nvml(self.cuda_torch("x"), "cuda:0")


class SummaryRender(unittest.TestCase):
    def test_render_flags_a_case_that_ran_a_slow_gpu(self):
        def document(case_id, slow, ratio):
            return {
                "record_type": "case-attempt", "version": 1,
                "identity": {"case_id": case_id, "case_factors": {"sku": "h200-dgxc"}},
                "outcome": {"status": "success"},
                "runtime": {"gpu_clocks": {"slow_gpu": slow, "sm_clock_ratio_min": ratio}},
            }

        documents = [document("slow-case", True, 0.71), document("fine-case", False, 0.99)]
        with mock.patch.object(summarize, "_identity", lambda d: (d["identity"]["case_id"],) * 8), \
                mock.patch.object(summarize, "_headline", lambda d: ("1", "1", "1", "1", "1", True)), \
                mock.patch.object(summarize, "_topology", lambda d: "t"), \
                mock.patch.object(summarize, "_wire_basis", lambda d: "w"):
            text = summarize.render(documents)
        warning = next(line for line in text.splitlines() if "slow or throttled GPU" in line)
        self.assertIn("**1 case(s)", warning)
        self.assertIn("`slow-case` (0.71)", warning)
        self.assertNotIn("fine-case", warning)


if __name__ == "__main__":
    unittest.main()
