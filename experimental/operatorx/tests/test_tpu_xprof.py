from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from operatorx.runners.tpu.xprof import (
    find_perfetto_trace,
    parse_batched_xla_module_durations,
    parse_xla_module_durations,
)


class XprofParserTest(unittest.TestCase):
    def _write_trace(self, directory: Path) -> Path:
        trace_path = directory / "plugins/profile/session/perfetto_trace.json.gz"
        trace_path.parent.mkdir(parents=True)
        trace = {
            "traceEvents": [
                {
                    "name": "operatorx_shape",
                    "ph": "X",
                    "ts": 1,
                    "dur": 10,
                },
                {
                    "name": "jit_dot(fingerprint)",
                    "ph": "X",
                    "args": {"device_duration_ps": "44000000"},
                },
                {
                    "name": "fusion",
                    "ph": "X",
                    "args": {
                        "device_duration_ps": "43000000",
                        "tf_op": "jit(dot)/dot_general:",
                        "hlo_category": "convolution fusion",
                        "model_flops": "128",
                        "raw_bytes_accessed": "64",
                    },
                },
            ]
        }
        with gzip.open(trace_path, "wt") as trace_file:
            json.dump(trace, trace_file)
        return trace_path

    def test_parses_module_critical_path_without_summing_children(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            profile_dir = Path(temporary_directory)
            trace_path = self._write_trace(profile_dir)

            self.assertEqual(find_perfetto_trace(profile_dir), trace_path)
            summary = parse_xla_module_durations(
                trace_path,
                module_name="jit_dot",
                expected_samples=1,
                annotation_name="operatorx_shape",
            )

        self.assertEqual(summary["p50"], 44.0)
        self.assertEqual(summary["samples"], 1)
        self.assertEqual(summary["child_hlo"]["events"], 1)
        self.assertEqual(summary["child_hlo"]["model_flops"], [128])
        self.assertEqual(summary["child_hlo"]["raw_bytes_accessed"], [64])

    def test_rejects_an_unexpected_sample_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            trace_path = self._write_trace(Path(temporary_directory))
            with self.assertRaisesRegex(ValueError, "expected 2"):
                parse_xla_module_durations(
                    trace_path,
                    module_name="jit_dot",
                    expected_samples=2,
                )

    def test_splits_a_synchronized_batch_by_xla_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            trace_path = (
                Path(temporary_directory)
                / "plugins/profile/session/perfetto_trace.json.gz"
            )
            trace_path.parent.mkdir(parents=True)
            trace = {
                "traceEvents": [
                    {
                        "name": "operatorx_batch_000_shape-a",
                        "ph": "X",
                        "ts": 1000,
                        "dur": 100,
                    },
                    {
                        "name": "operatorx_batch_001_shape-b",
                        "ph": "X",
                        "ts": 1200,
                        "dur": 100,
                    },
                    # Device timestamps deliberately precede host annotations.
                    {
                        "name": "jit_dot(shape-a)",
                        "ph": "X",
                        "ts": 900,
                        "dur": 10,
                        "args": {
                            "device_duration_ps": "10000000",
                            "run_id": "10",
                        },
                    },
                    {
                        "name": "jit_dot(shape-a)",
                        "ph": "X",
                        "ts": 920,
                        "dur": 12,
                        "args": {
                            "device_duration_ps": "12000000",
                            "run_id": "11",
                        },
                    },
                    {
                        "name": "jit_dot(shape-b)",
                        "ph": "X",
                        "ts": 1100,
                        "dur": 20,
                        "args": {
                            "device_duration_ps": "20000000",
                            "run_id": "12",
                        },
                    },
                    {
                        "name": "jit_dot(shape-b)",
                        "ph": "X",
                        "ts": 1130,
                        "dur": 22,
                        "args": {
                            "device_duration_ps": "22000000",
                            "run_id": "13",
                        },
                    },
                    {
                        "name": "fusion",
                        "ph": "X",
                        "ts": 901,
                        "dur": 8,
                        "args": {
                            "device_duration_ps": "8000000",
                            "tf_op": "jit(dot)/dot_general:",
                            "model_flops": "128",
                            "raw_bytes_accessed": "64",
                        },
                    },
                    {
                        "name": "fusion",
                        "ph": "X",
                        "ts": 1101,
                        "dur": 18,
                        "args": {
                            "device_duration_ps": "18000000",
                            "tf_op": "jit(dot)/dot_general:",
                            "model_flops": "256",
                            "raw_bytes_accessed": "96",
                        },
                    },
                ]
            }
            with gzip.open(trace_path, "wt") as trace_file:
                json.dump(trace, trace_file)

            result = parse_batched_xla_module_durations(
                trace_path,
                module_name="jit_dot",
                groups=[
                    {
                        "name": "shape-a",
                        "annotation_name": "operatorx_batch_000_shape-a",
                        "expected_samples": 2,
                    },
                    {
                        "name": "shape-b",
                        "annotation_name": "operatorx_batch_001_shape-b",
                        "expected_samples": 2,
                    },
                ],
            )

        shape_a = result["groups"]["shape-a"]
        shape_b = result["groups"]["shape-b"]
        self.assertEqual(shape_a["p50"], 11.0)
        self.assertEqual(shape_a["run_ids"], [10, 11])
        self.assertEqual(shape_a["child_hlo"]["model_flops"], [128])
        self.assertEqual(shape_b["p50"], 21.0)
        self.assertEqual(shape_b["run_ids"], [12, 13])
        self.assertEqual(shape_b["child_hlo"]["model_flops"], [256])
        self.assertEqual(result["total_module_events"], 4)


if __name__ == "__main__":
    unittest.main()
