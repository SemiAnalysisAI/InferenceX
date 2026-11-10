import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from infx.bench.umbp_health import FailureIntervals, record_pool


@unittest.skipUnless(sys.platform == "linux", "Requires Linux process identity and groups")
class HealthTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.state = Path(self.directory.name) / "pool.json"
        self.pool = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"])
        record_pool(self.pool.pid, self.state)
        self.running = []

    def tearDown(self):
        for process in self.running:
            if process.poll() is None:
                process.terminate()
            process.communicate(timeout=5)
        self.pool.terminate()
        self.pool.wait(timeout=5)
        self.directory.cleanup()

    def start(self, source):
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "infx.bench",
                "umbp-health",
                "--pool-state",
                str(self.state),
                "--poll-seconds",
                "0.02",
                "--failure-intervals",
                "3",
                "--stop-timeout",
                "0.3",
                "--",
                sys.executable,
                "-u",
                "-c",
                source,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self.running.append(process)
        return process

    def test_pool_death_terminates_live_worker(self):
        pidfile = Path(self.directory.name) / "worker.pid"
        process = self.start(
            f"import os,time,pathlib;pathlib.Path({str(pidfile)!r}).write_text(str(os.getpid()));time.sleep(60)"
        )
        deadline = time.monotonic() + 5
        while not pidfile.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(pidfile.exists())
        self.pool.kill()
        self.pool.wait()
        output, _ = process.communicate(timeout=5)
        self.assertEqual(process.returncode, 70, output)
        self.assertIn(b"umbp_health_failed", output)
        with self.assertRaises(ProcessLookupError):
            os.kill(int(pidfile.read_text()), 0)

    def test_wrong_process_identity_prevents_launch(self):
        state = json.loads(self.state.read_text())
        state["start_ticks"] += 1
        self.state.write_text(json.dumps(state))
        process = self.start("print('UNSAFE_LAUNCH')")
        output, _ = process.communicate(timeout=5)
        self.assertNotEqual(process.returncode, 0)
        self.assertNotIn(b"UNSAFE_LAUNCH", output)

    def test_persistent_failures_stop_worker(self):
        for operation in ("store", "load"):
            with self.subTest(operation=operation):
                line = "KV Transfer metrics: UMBPStoreConnector=" + repr({operation + "_failed": 8})
                process = self.start(f"import time;print(({line!r}+'\\n')*3,end='');time.sleep(60)")
                output, _ = process.communicate(timeout=5)
                self.assertEqual(process.returncode, 70, output)
                self.assertIn(f"UMBP {operation} failed in 3 consecutive reports".encode(), output)

    def test_success_resets_streak_and_preserves_exit(self):
        lines = [
            "KV Transfer metrics: UMBPStoreConnector=" + repr({"store_failed": n})
            for n in (8, 8, 0, 8, 8)
        ]
        process = self.start(f"import sys;print({chr(10).join(lines)!r});sys.exit(7)")
        output, _ = process.communicate(timeout=5)
        self.assertEqual(process.returncode, 7, output)
        self.assertNotIn(b"umbp_health_failed", output)

    def test_signal_stops_worker(self):
        pidfile = Path(self.directory.name) / "worker.pid"
        process = self.start(
            f"import os,time,pathlib;pathlib.Path({str(pidfile)!r}).write_text(str(os.getpid()));time.sleep(60)"
        )
        deadline = time.monotonic() + 5
        while not pidfile.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(pidfile.exists())
        process.send_signal(signal.SIGTERM)
        process.communicate(timeout=5)
        self.assertEqual(process.returncode, 143)
        with self.assertRaises(ProcessLookupError):
            os.kill(int(pidfile.read_text()), 0)

    def test_non_umbp_counters_do_not_trip_gate(self):
        gate = FailureIntervals(1)
        self.assertIsNone(gate.observe(b"OtherConnector={'store_failed': 99}"))
        self.assertIsNone(gate.observe(b"normal worker output"))


if __name__ == "__main__":
    unittest.main()
