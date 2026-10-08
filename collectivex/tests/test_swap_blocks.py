"""Measurement behavior and optional real-GPU block-copy checks."""

import argparse
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
import run_swap_blocks as bench


class SwapBlocksMeasurementTests(unittest.TestCase):
    def test_payload_budget_includes_boundary_and_records_exclusions(self):
        points, skipped = bench.plan_cases(["h2d"], [8, 17], [1, 2], 16)
        self.assertEqual(
            points,
            [
                {"direction": "h2d", "block_bytes": 8, "count": 1},
                {"direction": "h2d", "block_bytes": 8, "count": 2},
            ],
        )
        self.assertEqual(
            skipped,
            [
                {
                    "direction": "h2d",
                    "block_bytes": 17,
                    "count": 1,
                    "reason": "exceeds-max-payload-bytes",
                },
                {
                    "direction": "h2d",
                    "block_bytes": 17,
                    "count": 2,
                    "reason": "exceeds-max-payload-bytes",
                },
            ],
        )

    def test_uncapped_selection_keeps_both_directions(self):
        points, skipped = bench.plan_cases(["h2d", "d2h"], [17], [2], None)
        self.assertEqual(
            points,
            [
                {"direction": "h2d", "block_bytes": 17, "count": 2},
                {"direction": "d2h", "block_bytes": 17, "count": 2},
            ],
        )
        self.assertEqual(skipped, [])

    def test_invalid_or_empty_selection_fails(self):
        for sizes, counts, budget in (
            ([8], [1], 0),
            ([0], [1], 8),
            ([8], [0], 8),
            ([17], [1], 16),
        ):
            with (
                self.subTest(sizes=sizes, counts=counts, budget=budget),
                self.assertRaises(ValueError),
            ):
                bench.plan_cases(["h2d"], sizes, counts, budget)

    def test_nearest_rank_latency_and_payload_bandwidth(self):
        result = bench.summarize([8.0, 2.0, 4.0, 1.0], 8000)
        self.assertEqual(result["sample_count"], 4)
        self.assertEqual(
            result["percentiles_us"], {"p50": 2.0, "p90": 8.0, "p95": 8.0, "p99": 8.0}
        )
        self.assertEqual(
            result["payload_gbps_at_latency_percentile"],
            {"p50": 4.0, "p90": 1.0, "p95": 1.0, "p99": 1.0},
        )

    def test_invalid_samples_are_rejected(self):
        for samples in ([], [0.0], [-1.0], [float("nan")], [float("inf")]):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                bench.summarize(samples, 4096)

    def test_wall_timing_excludes_warmup_and_includes_completion(self):
        clock = [0]

        def operation():
            clock[0] += 2000

        def synchronize():
            clock[0] += 3000

        with mock.patch.object(
            bench.time, "perf_counter_ns", side_effect=lambda: clock[0]
        ):
            samples = bench.measure(operation, synchronize, warmup=2, iterations=3)
        self.assertEqual(samples, [5.0, 5.0, 5.0])
        self.assertEqual(clock[0], 34000)

    def test_mapping_copies_each_block_once_without_touching_guards(self):
        self.assertEqual(
            bench.block_pairs(3, "contiguous", 0), [[0, 0], [1, 1], [2, 2]]
        )
        pairs = bench.block_pairs(4, "random", 0)
        source = [11, 22, 33, 44, 55, 66]
        destination = [0] * 6
        for src, dst in pairs:
            destination[dst] = source[src]
        self.assertEqual(destination, [33, 22, 44, 11, 0, 0])

    def test_invalid_inputs(self):
        for value in ("0", "-2"):
            with (
                self.subTest(value=value),
                self.assertRaises(argparse.ArgumentTypeError),
            ):
                bench.positive_int(value)
        for count, layout in ((0, "random"), (1, "bad")):
            with (
                self.subTest(count=count, layout=layout),
                self.assertRaises(ValueError),
            ):
                bench.block_pairs(count, layout, 0)
        with self.assertRaises(ValueError):
            bench.measure(None, None, warmup=-1, iterations=1)


class SwapBlocksGPUTests(unittest.TestCase):
    def test_real_vllm_transfers(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA or ROCm")
        # Missing/broken vLLM on a GPU machine is a failure, not a silent skip.
        from vllm._custom_ops import swap_blocks

        torch.cuda.set_device(0)
        for direction in ("h2d", "d2h", "d2d"):
            with self.subTest(direction=direction):
                row = bench.run_case(
                    torch,
                    swap_blocks,
                    device=0,
                    direction=direction,
                    block_bytes=257,
                    count=4,
                    layout="random",
                    seed=0,
                    warmup=1,
                    iterations=2,
                )
                self.assertTrue(row["correctness_passed"])
                self.assertEqual(row["payload_bytes"], 1028)
                self.assertEqual(row["latency"]["sample_count"], 2)
                self.assertGreater(row["latency"]["percentiles_us"]["p50"], 0)


if __name__ == "__main__":
    unittest.main()
