#!/usr/bin/env python3
"""Matrix, subset, and shard-extraction tests."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sweep_matrix  # noqa: E402


def matrix(**options):
    return sweep_matrix.validate_matrix_document(
        sweep_matrix.resolve_matrix(max_cases=128, **options)
    )


class MatrixTests(unittest.TestCase):
    def test_shard_extraction_is_deterministic_and_preserves_cases(self):
        document = matrix(backend="deepep", only_sku="h100-dgxc")
        cell = document["include"][0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "matrix.json"
            source.write_text(json.dumps(document, sort_keys=True))
            outputs = [
                sweep_matrix.extract_shard(
                    source, cell["id"], root / f"shard-{index}.json",
                    sku=cell["sku"], backend=cell["backend"], nodes=cell["nodes"],
                )
                for index in range(2)
            ]
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual([case["case_id"] for case in outputs[0]["cases"]], cell["case_ids"])

    def test_sku_and_ep_filters_only_remove_cases(self):
        full = matrix(suites="all", backends="all")
        for options, keep in (
            ({"exclude_skus": "b300"}, lambda item: item["sku"] != "b300"),
            ({"ep_sizes": "8"}, lambda item: item["case"]["ep"] == 8),
        ):
            partial = matrix(suites="all", backends="all", **options)
            expected = {
                item["case"]["case_id"]: item for item in full["requested_cases"] if keep(item)
            }
            actual = {item["case"]["case_id"]: item for item in partial["requested_cases"]}
            self.assertEqual(actual, expected)

    def test_invalid_filters_fail_closed(self):
        for options in (
            {"exclude_skus": "unknown"},
            {"only_sku": "b300", "exclude_skus": "b300"},
            {"ep_sizes": "0"},
            {"ep_sizes": "eight"},
        ):
            with self.subTest(options=options), self.assertRaises(SystemExit):
                sweep_matrix.resolve_matrix(**options)


if __name__ == "__main__":
    unittest.main()
