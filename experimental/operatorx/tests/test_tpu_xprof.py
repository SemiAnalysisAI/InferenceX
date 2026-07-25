from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from operatorx.runners.tpu.xprof import (
    find_perfetto_trace,
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


if __name__ == "__main__":
    unittest.main()
