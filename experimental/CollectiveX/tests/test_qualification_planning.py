#!/usr/bin/env python3
"""CPU-only tests for matrix resolution, subset selection, and shard extraction."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]

import sweep_matrix  # noqa: E402


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode()


class MatrixPlanningTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.matrix = sweep_matrix.validate_matrix_document(
            sweep_matrix.resolve_matrix(
                backend="deepep", only_sku="h100-dgxc", max_cases=128
            )
        )
        cls.shard = next(item for item in cls.matrix["include"] if item["n"] >= 3)

    def test_extract_shard_is_deterministic_and_matrix_preserving(self) -> None:
        matrix_bytes = _canonical(self.matrix) + b"\n"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            matrix_path = root / "matrix.json"
            matrix_path.write_bytes(matrix_bytes)
            first = sweep_matrix.extract_shard(
                matrix_path,
                self.shard["id"],
                root / "first.json",
                sku=self.shard["sku"],
                backend=self.shard["backend"],
                nodes=self.shard["nodes"],
            )
            repeated = sweep_matrix.extract_shard(
                matrix_path,
                self.shard["id"],
                root / "repeated.json",
                sku=self.shard["sku"],
                backend=self.shard["backend"],
                nodes=self.shard["nodes"],
            )
            self.assertEqual(first, repeated)
            self.assertEqual(
                [case["case_id"] for case in first["cases"]], self.shard["case_ids"]
            )
            sweep_matrix.validate_shard_control(
                first,
                sku=self.shard["sku"],
                backend=self.shard["backend"],
                nodes=self.shard["nodes"],
            )
            # Extraction is read-only: the matrix on disk is untouched.
            self.assertEqual(matrix_path.read_bytes(), matrix_bytes)

    def test_full_v1_matrix_emits_every_unsupported_terminal_outcome(self) -> None:
        matrix = sweep_matrix.validate_matrix_document(
            sweep_matrix.resolve_matrix(suites="all", backends="all", max_cases=128)
        )
        expected = [
            item for item in matrix["requested_cases"]
            if item["disposition"] == "unsupported"
        ]
        environment = {
            "COLLECTIVEX_ARTIFACT_NAME": "cxunsupported-123-1",
            "COLLECTIVEX_EXECUTION_ID": "123_1_unsupported",
            "COLLECTIVEX_SOURCE_SHA": "a" * 40,
            "GITHUB_JOB": "setup",
            "GITHUB_REF_NAME": "collectivex",
            "GITHUB_REPOSITORY": "SemiAnalysisAI/InferenceX",
            "GITHUB_RUN_ATTEMPT": "1",
            "GITHUB_RUN_ID": "123",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            matrix_path = root / "matrix.json"
            matrix_path.write_bytes(_canonical(matrix))
            with mock.patch.dict(os.environ, environment, clear=False):
                written = sweep_matrix.emit_unsupported(matrix_path, root / "unsupported")
            documents = [json.loads(path.read_text()) for path in written]

        self.assertEqual(len(documents), len(expected))
        self.assertTrue(documents)
        self.assertEqual(
            {document["outcome"]["reason"] for document in documents},
            {item["reason"] for item in expected},
        )
        self.assertEqual(
            {document["identity"]["case_id"] for document in documents},
            {item["case"]["case_id"] for item in expected},
        )

    def test_exclude_skus_narrows_matrix_to_a_subset(self) -> None:
        full = sweep_matrix.validate_matrix_document(
            sweep_matrix.resolve_matrix(suites="all", backends="all", max_cases=128)
        )
        partial = sweep_matrix.validate_matrix_document(
            sweep_matrix.resolve_matrix(
                suites="all", backends="all", max_cases=128, exclude_skus="b300"
            )
        )
        full_skus = {item["sku"] for item in full["include"]}
        partial_skus = {item["sku"] for item in partial["include"]}
        self.assertIn("b300", full_skus)
        self.assertNotIn("b300", partial_skus)
        self.assertEqual(full_skus - {"b300"}, partial_skus)
        # Excluding a pool only omits its cells; every surviving cell is byte-identical
        # to the full matrix (a partial run cannot invent or reclassify cases).
        def by_id(matrix: dict[str, object]) -> dict[str, object]:
            return {
                item["case"]["case_id"]: item
                for item in matrix["requested_cases"]
                if item.get("sku") != "b300"
            }
        self.assertEqual(by_id(partial), by_id(full))
        self.assertTrue(
            all(item.get("sku") != "b300" for item in partial["requested_cases"])
        )

    def test_exclude_skus_rejects_unknown_and_conflicting_pools(self) -> None:
        with self.assertRaisesRegex(SystemExit, "unknown --exclude-skus"):
            sweep_matrix.resolve_matrix(exclude_skus="nosuchsku")
        with self.assertRaisesRegex(SystemExit, "disjoint pools"):
            sweep_matrix.resolve_matrix(only_sku="b300", exclude_skus="b300")

    def test_ep_sizes_narrows_matrix_to_a_subset(self) -> None:
        full = sweep_matrix.validate_matrix_document(
            sweep_matrix.resolve_matrix(suites="all", backends="all", max_cases=128)
        )
        partial = sweep_matrix.validate_matrix_document(
            sweep_matrix.resolve_matrix(
                suites="all", backends="all", max_cases=128, ep_sizes="8"
            )
        )
        full_eps = {item["case"]["ep"] for item in full["requested_cases"]}
        partial_eps = {item["case"]["ep"] for item in partial["requested_cases"]}
        # The canonical matrix carries EP16; --ep-sizes=8 drops it entirely so a
        # comprehensive run can co-schedule EP8 across every SKU (GB EP8 is two nodes,
        # the 8-GPU SKUs' EP8 is one) with no EP16 leg to wall.
        self.assertIn(16, full_eps)
        self.assertEqual(partial_eps, {8})
        # Keeping a degree only omits the others; every surviving cell is byte-identical
        # to the full matrix (a partial run cannot invent or reclassify cases).
        def ep8_by_id(matrix: dict[str, object]) -> dict[str, object]:
            return {
                item["case"]["case_id"]: item
                for item in matrix["requested_cases"]
                if item["case"]["ep"] == 8
            }
        self.assertEqual(ep8_by_id(partial), ep8_by_id(full))
        self.assertTrue(
            all(item["case"]["ep"] == 8 for item in partial["requested_cases"])
        )

    def test_ep_sizes_rejects_non_positive_or_non_integer_degrees(self) -> None:
        with self.assertRaisesRegex(SystemExit, "invalid --ep-sizes"):
            sweep_matrix.resolve_matrix(ep_sizes="0")
        with self.assertRaisesRegex(SystemExit, "invalid --ep-sizes"):
            sweep_matrix.resolve_matrix(ep_sizes="eight")

    def test_frontend_catalog_covers_every_requested_case_and_point(self) -> None:
        catalog = sweep_matrix.frontend_catalog(self.matrix)
        self.assertEqual(catalog["format"], "collectivex.frontend-catalog.v1")
        self.assertEqual(catalog["case_count"], len(self.matrix["requested_cases"]))
        self.assertEqual(
            catalog["point_count"],
            sum(
                len(item["case"]["ladder"].split())
                for item in self.matrix["requested_cases"]
            ),
        )
        self.assertEqual(
            {item["case_id"] for item in catalog["cases"]},
            {item["case"]["case_id"] for item in self.matrix["requested_cases"]},
        )
        self.assertTrue(all(item["required"] for item in catalog["cases"]))
        self.assertTrue(all(item["backend_generation"] is None for item in catalog["cases"]))
        self.assertEqual(
            {item["resource"]["mode"] for item in catalog["cases"]},
            {"fixed-profile"},
        )
        self.assertLess(len(_canonical(catalog)) + 1, 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
