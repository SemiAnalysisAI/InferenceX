#!/usr/bin/env python3
"""Matrix, subset, and shard-extraction tests."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "runtime"))

import sweep_matrix  # noqa: E402
import config  # noqa: E402


def matrix(**options):
    return sweep_matrix.resolve_matrix(**options)


def cells(document, project=("sku", "ep"), **filters):
    """Projected set of the requested cases matching `filters`.

    The rollout tests all ask the same question -- which (sku, ep) pairs a backend offers, and
    under which disposition -- so they share one query instead of restating the comprehension.
    `project` names the fields to return ("sku" comes from the item, everything else from its
    case); a single field projects to scalars rather than 1-tuples.
    """
    fields = (project,) if isinstance(project, str) else project
    selected = set()
    for item in document["requested_cases"]:
        case = item["case"]
        if any(
            (item["disposition"] if key == "disposition" else case[key]) != value
            for key, value in filters.items()
        ):
            continue
        row = tuple(item["sku"] if f == "sku" else case[f] for f in fields)
        selected.add(row[0] if isinstance(project, str) else row)
    return selected


class MatrixTests(unittest.TestCase):
    def test_sku_and_ep_filters_only_remove_cases(self):
        # Subtractive with ONE deliberate exception: naming an off-path precision explicitly
        # opts its rows back in (see OFF_PATH_PRECISIONS), so the fp8 subset is compared
        # against a baseline that also names fp8 rather than against the default matrix.
        full = matrix(backend="all")
        full_with_off_path = matrix(backend="all", precisions="bf16,fp8")
        for case in (
            ({"exclude_skus": "b300"}, lambda item: item["sku"] != "b300"),
            ({"ep_sizes": "8"}, lambda item: item["case"]["ep"] == 8),
            # A precision subset removes only the runnable cases of the other
            # precision; ep-unsupported cells keep their stable bf16 placeholder.
            ({"precisions": "bf16"}, lambda item: item["case"]["precision"] == "bf16"),
            ({"precisions": "fp8"},
             lambda item: item["case"]["precision"] == "fp8"
             or item["disposition"] == "unsupported", "off_path"),
            # A mode subset removes only the runnable cases of the other mode; the
            # ep-unsupported placeholder is normal-mode and mode-filter-independent, so it
            # survives both selections (mirrors the precision rows above).
            ({"modes": "normal"}, lambda item: item["case"]["mode"] == "normal"),
            ({"modes": "low-latency"},
             lambda item: item["case"]["mode"] == "low-latency"
             or item["disposition"] == "unsupported"),
        ):
            options, keep = case[0], case[1]
            partial = matrix(backend="all", **options)
            baseline = full_with_off_path if len(case) > 2 else full
            expected = {
                item["case"]["case_id"]: item
                for item in baseline["requested_cases"] if keep(item)
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

    def test_flashinfer_ep_rollout_shape(self):
        # FlashInfer one-sided is the transport a GB deployment actually runs: vLLM picks
        # `flashinfer_nvlink_one_sided` on NVLink and `deepep_v2` on RDMA, so it belongs on the
        # MNNVL SKUs and nowhere else this pass.
        #   * GB NVL72 (gb200/gb300): EP8 AND EP16, both inside the 72-GPU scale-up domain.
        #   * No x86 rows. The kernels are grouped upstream under MNNVL and vLLM gates them on an
        #     MNNVL-availability probe, so an HGX 8-GPU NVSwitch node may not qualify; that is an
        #     open question, not an assumed capability, and is left unclaimed until measured.
        #   * No AMD rows (NVIDIA/MNNVL only).
        #   * No low-latency row anywhere: FlashInfer exposes one one-sided A2A kernel family, not
        #     a separate decode kernel, so an ll_backends cell would re-measure the same kernel
        #     under a mode that promises a different one. Decode is still covered by the decode
        #     phase of normal mode.
        # BF16 only in the DEFAULT matrix: the FP8 dispatch is implemented and oracle-validated,
        # but no engine can select it on this transport, so OFF_PATH_PRECISIONS keeps it out
        # unless `--precisions` names fp8 explicitly.
        document = matrix(backend="all")
        cases = [
            item for item in document["requested_cases"]
            if item["case"]["backend"] == "flashinfer-ep"
        ]
        runnable = {
            (item["sku"], item["case"]["ep"])
            for item in cases if item["disposition"] == "runnable"
        }
        self.assertEqual(runnable, {(sku, ep) for sku in ("gb200", "gb300") for ep in (8, 16)})
        # FP8 is dispatch-side only: scales ride as a fourth payload (kMaxPayloads is exactly 4)
        # and combine stays BF16, so none of the 0.6.16+ combine-quant API is needed. It is
        # realizable but off-path, so the DEFAULT matrix carries BF16 alone and naming the
        # precision brings it back for transport comparison.
        self.assertEqual({item["case"]["precision"] for item in cases}, {"bf16"})
        opted_in = cells(
            matrix(backend="flashinfer-ep", precisions="fp8"), "precision",
            disposition="runnable",
        )
        self.assertEqual(opted_in, {"fp8"})
        # Normal mode only — no low-latency cell on any SKU.
        self.assertEqual({item["case"]["mode"] for item in cases}, {"normal"})
        for platform in sweep_matrix.PLATFORMS.values():
            self.assertNotIn("flashinfer-ep", platform.get("ll_backends", {}))

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

    def test_every_scheduled_backend_declares_its_precisions(self):
        for backend in sweep_matrix.SWEEP_BACKENDS:
            self.assertIn(backend, sweep_matrix.BACKEND_PRECISIONS, backend)


class BackendMaturityTests(unittest.TestCase):
    """The registry map and each adapter's `maturity` are two copies of one fact, read by
    different consumers, so they can drift silently: pin coverage, vocabulary and agreement.
    The adapter side is parsed from source rather than imported — importing an adapter pulls
    in torch and the vendor EP library, which the test image does not carry.
    """

    VOCABULARY = {"production", "candidate"}

    @staticmethod
    def _declared_in_source():
        """{backend name: maturity} parsed from the adapter class bodies."""
        import ast

        declared = {}
        for path in sorted((ROOT / "bench").glob("ep_*.py")):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                literals = {}
                for statement in node.body:
                    if not isinstance(statement, ast.Assign):
                        continue
                    if not isinstance(statement.value, ast.Constant):
                        continue
                    for target in statement.targets:
                        if isinstance(target, ast.Name):
                            literals[target.id] = statement.value.value
                # The abstract base declares both as empty defaults; skip it.
                if literals.get("name") and "maturity" in literals:
                    declared[literals["name"]] = literals["maturity"]
        return declared

    def test_registry_covers_every_dispatched_backend(self):
        maturity = sweep_matrix.BACKEND_MATURITY
        for sku, platform in sweep_matrix.PLATFORMS.items():
            for backend in platform["backends"]:
                with self.subTest(sku=sku, backend=backend):
                    self.assertIn(backend, maturity)
                    self.assertIn(maturity[backend], self.VOCABULARY)

    def test_adapters_and_registry_agree(self):
        declared = self._declared_in_source()
        # Every backend the matrix can dispatch must declare a maturity in its adapter,
        # or the artifact it writes would say "unknown" while the registry says otherwise.
        for backend, expected in sweep_matrix.BACKEND_MATURITY.items():
            with self.subTest(backend=backend):
                self.assertIn(backend, declared)
                self.assertEqual(declared[backend], expected)


if __name__ == "__main__":
    unittest.main()
