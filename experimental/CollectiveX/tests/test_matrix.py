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
    return sweep_matrix.resolve_matrix(**options)


class MatrixTests(unittest.TestCase):
    def test_shard_extraction_is_deterministic_and_preserves_cases(self):
        document = matrix(backend="deepep-v2", only_sku="h200-dgxc")
        cell = document["include"][0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "matrix.json"
            source.write_text(json.dumps(document, sort_keys=True))
            outputs = [
                sweep_matrix.extract_shard(
                    source, cell["id"], root / f"shard-{index}.json",
                )
                for index in range(2)
            ]
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[0]["cases"], cell["cases"])

    def test_sku_and_ep_filters_only_remove_cases(self):
        full = matrix(backend="all")
        for options, keep in (
            ({"exclude_skus": "b300"}, lambda item: item["sku"] != "b300"),
            ({"ep_sizes": "8"}, lambda item: item["case"]["ep"] == 8),
            # A precision subset removes only the runnable cases of the other
            # precision; ep-unsupported cells keep their stable bf16 placeholder.
            ({"precisions": "bf16"}, lambda item: item["case"]["precision"] == "bf16"),
            ({"precisions": "fp8"},
             lambda item: item["case"]["precision"] == "fp8"
             or item["disposition"] == "unsupported"),
            # A mode subset removes only the runnable cases of the other mode; the
            # ep-unsupported placeholder is normal-mode and mode-filter-independent, so it
            # survives both selections (mirrors the precision rows above).
            ({"modes": "normal"}, lambda item: item["case"]["mode"] == "normal"),
            ({"modes": "low-latency"},
             lambda item: item["case"]["mode"] == "low-latency"
             or item["disposition"] == "unsupported"),
        ):
            partial = matrix(backend="all", **options)
            expected = {
                item["case"]["case_id"]: item for item in full["requested_cases"] if keep(item)
            }
            actual = {item["case"]["case_id"]: item for item in partial["requested_cases"]}
            self.assertEqual(actual, expected)

    def test_only_real_platform_cells_are_unsupported(self):
        document = matrix(backend="all")
        unsupported = {
            (item["sku"], item["case"]["backend"], item["case"]["ep"])
            for item in document["requested_cases"] if item["disposition"] == "unsupported"
        }
        expected = {
            (sku, backend, ep)
            for sku, platform in sweep_matrix.PLATFORMS.items()
            for backend, runnable_eps in platform["backends"].items()
            for ep in sweep_matrix.SWEEP["ep_degrees"]
            if ep not in runnable_eps
        }
        self.assertEqual(unsupported, expected)
        for item in document["requested_cases"]:
            self.assertIn(item["case"]["backend"], sweep_matrix.PLATFORMS[item["sku"]]["backends"])

    def test_runnable_cases_fan_out_over_backend_precisions(self):
        document = matrix(backend="all")
        runnable = [
            item for item in document["requested_cases"]
            if item["disposition"] == "runnable"
        ]
        # Every runnable case carries a precision its backend supports, and each
        # (sku, backend, ep, phase) cell is realized once per supported precision.
        by_cell: dict[tuple, set[str]] = {}
        for item in runnable:
            case = item["case"]
            self.assertIn(
                case["precision"], sweep_matrix.BACKEND_PRECISIONS[case["backend"]]
            )
            cell = (item["sku"], case["backend"], case["ep"], case["phase"])
            by_cell.setdefault(cell, set()).add(case["precision"])
        for cell, precisions in by_cell.items():
            expected = {
                precision for precision in sweep_matrix.SWEEP["precisions"]
                if precision in sweep_matrix.BACKEND_PRECISIONS[cell[1]]
            }
            self.assertEqual(precisions, expected, cell)
        # Every backend (deepep-v2, mori, uccl-ep) realizes BF16 and FP8.
        self.assertEqual(
            {precision for precisions in by_cell.values() for precision in precisions},
            {"bf16", "fp8"},
        )

    def test_case_ids_are_unique_across_the_matrix(self):
        # precision is part of case_id, so a cell's bf16 and fp8 attempts are distinct
        # identities. Without precision in the id the two would collide; assert the full
        # matrix carries no duplicate case_id so that identity property stays testable.
        document = matrix(backend="all")
        ids = [item["case"]["case_id"] for item in document["requested_cases"]]
        self.assertEqual(len(ids), len(set(ids)))
        # And every id ends in its own precision factor.
        for item in document["requested_cases"]:
            self.assertTrue(item["case"]["case_id"].endswith(item["case"]["precision"]))

    def test_low_latency_is_decode_only_and_capability_gated(self):
        # Low-latency cases are additive: they appear only for (sku, backend, ep) cells
        # listed in the platform registry's ll_backends map, only in the decode phase, and
        # never as unsupported placeholders. Normal-mode cases are unchanged by their
        # presence.
        document = matrix(backend="all")
        ll = [
            item for item in document["requested_cases"]
            if item["case"]["mode"] == "low-latency"
        ]
        self.assertTrue(ll, "expected at least one low-latency cell in the registry")
        for item in ll:
            case = item["case"]
            self.assertEqual(item["disposition"], "runnable")
            self.assertEqual(case["phase"], "decode")
            self.assertIn("low-latency", sweep_matrix.SWEEP["modes"])
            ll_backends = sweep_matrix.PLATFORMS[item["sku"]].get("ll_backends", {})
            self.assertIn(case["ep"], ll_backends.get(case["backend"], []))
            self.assertIn("-low-latency-", case["case_id"])
        # Every low-latency cell realizes exactly its backend's supported precisions.
        by_cell: dict[tuple, set[str]] = {}
        for item in ll:
            case = item["case"]
            cell = (item["sku"], case["backend"], case["ep"])
            by_cell.setdefault(cell, set()).add(case["precision"])
        for cell, precisions in by_cell.items():
            self.assertEqual(
                precisions,
                {p for p in sweep_matrix.SWEEP["precisions"]
                 if p in sweep_matrix.BACKEND_PRECISIONS[cell[1]]},
                cell,
            )

    def test_ll_backends_is_a_well_formed_subset_of_backends(self):
        # A cell can only run low-latency where it can run at all: every ll_backends
        # entry names a real backend of that SKU and a subset of its normal EP degrees.
        for sku, platform in sweep_matrix.PLATFORMS.items():
            ll_backends = platform.get("ll_backends", {})
            for backend, degrees in ll_backends.items():
                with self.subTest(sku=sku, backend=backend):
                    self.assertIn(backend, platform["backends"])
                    self.assertTrue(degrees)
                    self.assertLessEqual(set(degrees), set(platform["backends"][backend]))

    def test_uccl_ep_rollout_shape(self):
        # UCCL-EP's intended rollout, locked here: runnable on exactly the six supported
        # SKUs; EP16 (cross-node) on h100/h200/b200/mi355x; EP8-only on the -tw pair (no
        # cross-node fabric, so their EP16 cells are unsupported placeholders); and no rows
        # at all on b300/gb200/gb300, where the backend is not offered (a missing backends
        # key means "not provided", exactly like NVIDIA SKUs carrying no mori rows). LL
        # (decode) on every supported SKU at EP8.
        document = matrix(backend="all")
        runnable = {
            (item["sku"], item["case"]["ep"])
            for item in document["requested_cases"]
            if item["case"]["backend"] == "uccl-ep" and item["disposition"] == "runnable"
        }
        unsupported = {
            (item["sku"], item["case"]["ep"])
            for item in document["requested_cases"]
            if item["case"]["backend"] == "uccl-ep" and item["disposition"] == "unsupported"
        }
        supported_skus = {
            "h100-dgxc", "h200-dgxc", "b200-dgxc", "mi355x", "mi325x-tw", "mi300x-tw",
        }
        self.assertEqual({sku for sku, _ in runnable}, supported_skus)
        self.assertEqual(
            {sku for sku, ep in runnable if ep == 16},
            {"h100-dgxc", "h200-dgxc", "b200-dgxc", "mi355x"},
        )
        self.assertEqual(unsupported, {("mi325x-tw", 16), ("mi300x-tw", 16)})
        offered = {sku for sku, _ in runnable | unsupported}
        for absent in ("b300", "gb200", "gb300"):
            self.assertNotIn(absent, offered)
        # uccl-ep low-latency is enabled only on NVIDIA; the AMD SKUs keep normal mode but drop
        # LL (UCCL's low-latency kernel trips a warp-group assertion on AMD's CU count).
        ll_skus = {
            item["sku"]
            for item in document["requested_cases"]
            if item["case"]["backend"] == "uccl-ep"
            and item["case"]["mode"] == "low-latency"
        }
        self.assertEqual(ll_skus, {"h100-dgxc", "h200-dgxc", "b200-dgxc"})

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


if __name__ == "__main__":
    unittest.main()
