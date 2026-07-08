#!/usr/bin/env python3
"""Matrix, subset, shard extraction, and unsupported-delivery tests."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import contracts  # noqa: E402
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

    def test_unsupported_emission_is_complete_and_delivery_valid(self):
        document = matrix(suites="all", backends="all", only_sku="h100-dgxc")
        expected = [x for x in document["requested_cases"] if x["disposition"] == "unsupported"]
        environment = {
            "COLLECTIVEX_ARTIFACT_NAME": "cxunsupported-123-1",
            "COLLECTIVEX_EXECUTION_ID": "123_1_unsupported",
            "COLLECTIVEX_SOURCE_SHA": "a" * 40,
            "GITHUB_JOB": "setup", "GITHUB_REF_NAME": "collectivex",
            "GITHUB_REPOSITORY": "SemiAnalysisAI/InferenceX",
            "GITHUB_RUN_ATTEMPT": "1", "GITHUB_RUN_ID": "123",
        }
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(
            os.environ, environment, clear=False
        ):
            root = Path(temporary)
            source = root / "matrix.json"
            source.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")))
            paths = sweep_matrix.emit_unsupported(source, root / "unsupported")
            self.assertEqual(
                contracts.validate_delivery(
                    [str(path) for path in paths], str(source), disposition="unsupported"
                ),
                len(expected),
            )
            emitted = [contracts.validate_result(path) for path in paths]
        self.assertEqual(
            {item["identity"]["case_id"] for item in emitted},
            {item["case"]["case_id"] for item in expected},
        )


if __name__ == "__main__":
    unittest.main()
