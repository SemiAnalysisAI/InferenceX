import importlib.util
import types
import unittest
from pathlib import Path
from unittest import mock

p = Path(__file__).resolve().parents[1] / "scripts/ascend/operatorx_queued_timing.py"
spec = importlib.util.spec_from_file_location("queued", p)
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


class Tests(unittest.TestCase):
    def exercise(self, first=False):
        trace = []
        created = []
        clock = [0]
        sleep = []

        def tick():
            clock[0] += 10_000_000
            return clock[0]

        class Event:
            def __init__(self, enable_timing):
                self.i = len(created)
                created.append(self)

            def record(self):
                trace.append(("event", self.i))

            def elapsed_time(self, end):
                assert end.i == self.i + 10
                return (self.i + 1) * 0.01

        def fn():
            trace.append(("op",))
            return len(trace)

        npu = types.SimpleNamespace(
            Event=Event, synchronize=lambda: trace.append(("sync",))
        )
        cache = types.SimpleNamespace(zero_=lambda: trace.append(("flush",)))
        q.FIRST = first
        with (
            mock.patch.object(q.time, "perf_counter_ns", side_effect=tick),
            mock.patch.object(q.time, "sleep", side_effect=sleep.append),
        ):
            out, t = q.time_op(
                npu,
                fn,
                cache,
                {
                    "samples": 10,
                    "warmups": 5,
                    "warmup_ms": 25,
                    "cache_write_bytes": 268435456,
                },
            )
        return trace, t, sleep

    def test_queue_has_no_intermediate_synchronization(self):
        trace, t, sleep = self.exercise()
        first = trace.index(("event", 0))
        last = trace.index(("event", 19))
        self.assertNotIn(("sync",), trace[first : last + 1])
        self.assertEqual(trace[last + 1 :], [("sync",)])
        self.assertEqual(sum(x == ("flush",) for x in trace), 14)
        self.assertEqual(sum(x[0] == "event" for x in trace), 20)
        self.assertEqual(
            t["device_event_samples_us"],
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        )
        self.assertEqual(t["queued_wall_us_per_call"], t["queued_batch_wall_us"] / 10)
        self.assertNotIn("synchronized_wall_samples_us", t)
        self.assertAlmostEqual(sleep[0], 0.0036)

    def test_first_process_ramp_then_short_ramp(self):
        _, a, _ = self.exercise(True)
        _, b, _ = self.exercise(False)
        self.assertEqual(a["effective_ramp_floor_ms"], 1000)
        self.assertEqual(b["effective_ramp_floor_ms"], 25)
        self.assertGreater(a["actual_warmups"], b["actual_warmups"])

    def test_rejects_accidental_old_configuration(self):
        with self.assertRaises(ValueError):
            q.time_op(None, None, None, {"samples": 20, "warmups": 5, "warmup_ms": 250})


if __name__ == "__main__":
    unittest.main()
