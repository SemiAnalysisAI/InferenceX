#!/usr/bin/env python3
"""Matrix, subset, and shard-extraction tests."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sweep_matrix  # noqa: E402


def matrix(**options):
    return sweep_matrix.resolve_matrix(**options)


class MatrixTests(unittest.TestCase):
    def test_every_shard_has_an_exact_positive_node_request(self):
        document = matrix(backend="all")
        self.assertTrue(document["include"])
        for shard in document["include"]:
            with self.subTest(shard=shard["id"]):
                self.assertIs(type(shard["nodes"]), int)
                self.assertGreater(shard["nodes"], 0)
                self.assertTrue(shard["cases"])
                self.assertEqual(
                    {case["nodes"] for case in shard["cases"]},
                    {shard["nodes"]},
                )

    def test_only_real_platform_cells_are_unsupported(self):
        platform = {
            "product": "test-gpu", "gpus_per_node": 8, "scale_up_domain": 8,
            "scale_up_transport": "nvlink", "launcher": "test-launcher",
            "backends": {"deepep-v2": [8]},
        }
        with mock.patch.object(sweep_matrix, "PLATFORMS", {"test-sku": platform}), \
                mock.patch.dict(sweep_matrix.SWEEP, {"ep_degrees": [8, 16]}):
            document = matrix(backend="all")
        unsupported = {
            (item["sku"], item["case"]["backend"], item["case"]["ep"])
            for item in document["requested_cases"] if item["disposition"] == "unsupported"
        }
        self.assertEqual(unsupported, {("test-sku", "deepep-v2", 16)})
        self.assertTrue(document["include"])
        for item in document["requested_cases"]:
            self.assertEqual(item["case"]["backend"], "deepep-v2")
        for shard in document["include"]:
            self.assertEqual({case["ep"] for case in shard["cases"]}, {8})

    def test_case_ids_are_unique_across_the_matrix(self):
        # precision is part of case_id, so a cell's bf16 and fp8 attempts are distinct
        # identities. Without precision in the id the two would collide; assert the full
        # matrix carries no duplicate case_id so that identity property stays testable.
        document = matrix(backend="all")
        ids = [item["case"]["case_id"] for item in document["requested_cases"]]
        self.assertEqual(len(ids), len(set(ids)))


    def test_off_path_precisions_require_explicit_opt_in(self):
        with mock.patch.object(sweep_matrix, "OFF_PATH_PRECISIONS", {"deepep-v2": ("fp8",)}):
            default = matrix(backend="deepep-v2")
            opted_in = matrix(backend="deepep-v2", precisions="fp8")
        self.assertEqual(
            {item["case"]["precision"] for item in default["requested_cases"]
             if item["disposition"] == "runnable"},
            {"bf16"},
        )
        self.assertEqual(
            {item["case"]["precision"] for item in opted_in["requested_cases"]
             if item["disposition"] == "runnable"},
            {"fp8"},
        )

    def test_invalid_filters_fail_closed(self):
        for options in (
            {"exclude_skus": "unknown"},
            {"only_sku": "b300", "exclude_skus": "b300"},
            {"ep_sizes": "0"},
            {"ep_sizes": "eight"},
            {"precisions": "fp4"},
            {"modes": "turbo"},
            {"backend": "unknown"},
        ):
            with self.subTest(options=options), self.assertRaises(SystemExit):
                sweep_matrix.resolve_matrix(**options)


class UndeclaredPrecisionsFailClosed(unittest.TestCase):
    # A backend in platform_config but missing from BACKEND_PRECISIONS must stop the matrix
    # rather than resolve to bf16-only: that yields a MISSING case, not a mislabelled one, and
    # run_sweep's non-bf16-dispatch guard can only catch cases that ran.
    def test_a_backend_without_declared_precisions_stops_the_matrix(self):
        pruned = {
            name: value for name, value in sweep_matrix.BACKEND_PRECISIONS.items()
            if name != "deepep-v2"
        }
        with mock.patch.object(sweep_matrix, "BACKEND_PRECISIONS", pruned):
            with self.assertRaises(SystemExit) as caught:
                sweep_matrix.resolve_matrix()
        self.assertIn("deepep-v2", str(caught.exception))
        self.assertIn("BACKEND_PRECISIONS", str(caught.exception))

if __name__ == "__main__":
    unittest.main()
