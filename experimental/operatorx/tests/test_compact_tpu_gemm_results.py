from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "compact_tpu_gemm_results.py"
)
SPEC = importlib.util.spec_from_file_location("compact_tpu_gemm_results", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CompactTpuGemmResultsTest(unittest.TestCase):
    def test_removes_machine_local_and_high_volume_fields(self) -> None:
        compact = MODULE.compact_results(
            {
                "run": {
                    "hostname": "machine",
                    "argv": ["script.py", "--output", "/tmp/results.json"],
                    "software": {"jax": "0.11.0"},
                },
                "rows": [
                    {
                        "placement": {
                            "validated": True,
                            "input_shardings": ["SingleDeviceSharding(...)"],
                            "output_shardings": ["SingleDeviceSharding(...)"],
                        },
                        "hlo": {
                            "validated": True,
                            "stablehlo_path": "/tmp/stablehlo.txt",
                            "optimized_hlo_path": "/tmp/optimized.txt",
                        },
                        "device_execution_us": {
                            "p50": 12.0,
                            "raw_us": [11.0, 12.0, 13.0],
                            "trace_path": "/tmp/perfetto_trace.json.gz",
                        },
                    }
                ],
            }
        )

        self.assertEqual(compact["run"], {"software": {"jax": "0.11.0"}})
        row = compact["rows"][0]
        self.assertNotIn("input_shardings", row["placement"])
        self.assertNotIn("output_shardings", row["placement"])
        self.assertNotIn("stablehlo_path", row["hlo"])
        self.assertNotIn("optimized_hlo_path", row["hlo"])
        self.assertNotIn("raw_us", row["device_execution_us"])
        self.assertNotIn("trace_path", row["device_execution_us"])
        self.assertTrue(compact["dataset"]["machine_identity_removed"])


if __name__ == "__main__":
    unittest.main()
