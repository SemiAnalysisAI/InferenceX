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
        # Every backend that lists FP8 (deepep-v2, mori, uccl-ep) realizes BF16 and FP8;
        # nccl-ep is BF16-only, so the cross-cell union of realized precisions stays {bf16, fp8}.
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

    def test_every_case_carries_the_chain_knobs_in_its_timing_profile(self):
        # The producer half of the codec CaseArgvContract tests the consumer half of: a legacy
        # three-field profile still decodes there while every real case names all six knobs.
        document = matrix(backend="all")
        profiles = {
            tuple(sorted(item["case"]["timing"].items()))
            for item in document["requested_cases"]
        }
        self.assertTrue(profiles)
        expected = {key for key, _ in config._TIMING_FLAGS}
        for profile in sorted(profiles):
            timing = dict(profile)
            with self.subTest(timing=timing):
                self.assertEqual(set(timing), expected)
                self.assertTrue(all(isinstance(v, int) for v in timing.values()), timing)
                # A chain that discards everything it measured leaves nothing to reduce.
                self.assertGreater(timing["chain_iters_per_trial"], timing["chain_drop"])
                self.assertGreater(timing["chain_trials_per_point"], 0)

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

    def test_ep_degrees_pin_the_two_backends_without_a_rollout_test(self):
        # deepep-v2 and mori have no rollout-shape test of their own, so the two facts most
        # likely to drift silently are pinned here: deepep-v2 is the only backend at EP16
        # everywhere, and mori is EP8-only (its EP16 InterNodeV1 combine corrupts, mori#475).
        # Without this, adding 16 to backends.mori passes CI.
        for sku, platform in sweep_matrix.PLATFORMS.items():
            degrees = platform["backends"]
            with self.subTest(sku=sku):
                if "mori" in degrees:
                    self.assertEqual(degrees["mori"], [8])
                if "deepep-v2" in degrees:
                    self.assertEqual(degrees["deepep-v2"], [8, 16])

    def test_uccl_ep_rollout_shape(self):
        # UCCL-EP's rollout, locked here: EP8 runnable on exactly the six supported SKUs, and
        # EP16 an unsupported coverage row on every one of them. uccl-ep is EP8-only: the
        # -tw pair has no cross-node fabric, and on the fabric SKUs cross-node EP16 is
        # functional but its CPU-proxy throughput overruns the standardized per-case
        # wall-clock budget (the internode Config fix landed; EP16 stays scoped out of the
        # sweep, mirroring the mori EP16 re-wall). No rows at all on b300/gb200/gb300, where
        # the backend is not offered. LL (decode) on every NVIDIA supported SKU at EP8.
        document = matrix(backend="all")
        runnable = cells(document, backend="uccl-ep", disposition="runnable")
        unsupported = cells(document, backend="uccl-ep", disposition="unsupported")
        supported_skus = {
            "h100-dgxc", "h200-dgxc", "b200-nscale", "mi355x", "mi325x-tw", "mi300x-tw",
        }
        # EP8 runnable on all six; nothing runnable at EP16.
        self.assertEqual({sku for sku, _ in runnable}, supported_skus)
        self.assertEqual({sku for sku, ep in runnable if ep == 8}, supported_skus)
        self.assertEqual({sku for sku, ep in runnable if ep == 16}, set())
        # EP16 is an honest unsupported coverage row on every supported SKU.
        self.assertEqual(unsupported, {(sku, 16) for sku in supported_skus})
        offered = {sku for sku, _ in runnable | unsupported}
        for absent in ("b300", "gb200", "gb300"):
            self.assertNotIn(absent, offered)
        # uccl-ep low-latency is enabled only on NVIDIA; the AMD SKUs keep normal mode but drop
        # LL: upstream raised kNumMaxTopK 9 -> 16 six days before our pin, and the host assert
        # kNumMaxTopK + 1 <= num_warp_groups * num_warps_per_group cannot hold on AMD, whose
        # kNumMaxWarpGroups is 16 — a dated regression, not a CU-count limit.
        ll_skus = cells(document, "sku", backend="uccl-ep", mode="low-latency")
        self.assertEqual(ll_skus, {"h100-dgxc", "h200-dgxc", "b200-nscale"})

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
        # BF16 only this pass (FP8 dispatch needs the scale payload plumbed and oracle-validated).
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
        # and combine stays BF16, so none of the 0.6.16+ combine-quant API is needed.
        self.assertEqual({item["case"]["precision"] for item in cases}, {"bf16", "fp8"})
        # Normal mode only — no low-latency cell on any SKU.
        self.assertEqual({item["case"]["mode"] for item in cases}, {"normal"})
        for platform in sweep_matrix.PLATFORMS.values():
            self.assertNotIn("flashinfer-ep", platform.get("ll_backends", {}))

    def test_nccl_ep_rollout_shape(self):
        # NCCL-EP's rollout, locked to the on-metal verdict (2026-07-22, all via the real launcher):
        #   * RDMA scale-out SKUs (h100/h200/b200/b300): EP8 runnable, EP16 an UNSUPPORTED coverage
        #     row. EP16 cross-node rides NCCL's kernel-initiated GDAKI GIN, which faults identically
        #     on RoCE (h100/b300) and IB (h200/b200-nscale) — a reproducible NCCL-EP v0.1.0 internode
        #     limitation, so EP16 is scoped out like uccl-ep's.
        #   * GB NVL72 SKUs (gb200/gb300, MNNVL, gb-nv launcher, 4 GPU/node): EP8 AND EP16 runnable —
        #     both stay inside the 72-GPU scale-up domain (world <= scale_up_domain => LSA, no GIN),
        #     so EP16 works over MNNVL where the RDMA-GIN path walls.
        #   * AMD SKUs: no rows (NCCL EP is NVIDIA-only; AMD runs mori).
        #   * LOW-LATENCY: EP8 on all six NVIDIA SKUs. These rows were dropped while every LL leg
        #     wedged, on the reading that only a fixed wheel could restore them. That reading was
        #     wrong about the cause: the wedge needed TWO LL handles live on one group. `buffer_idx`
        #     is per-handle but its buffers are offsets into the per-group rdma_buffer, so same-config
        #     handles alias one another's parity count/flag slots. The adapter now binds a single
        #     handle and rebinds it per shape, which removes the aliasing without a wheel bump —
        #     proven on a stock wheel by ladder [1] (one handle, clean) vs [1, 2] (two handles,
        #     64 dispatch + 6 combine receive timeouts). LL is decode-only and EP8-only: GB stays at
        #     EP8 here even though normal mode runs EP16, because LL adds no EP16 row on any SKU.
        # BF16 only — no FP8 case (NCCL EP FP8 unsupported this release).
        document = matrix(backend="all")
        runnable = cells(document, backend="nccl-ep", disposition="runnable")
        unsupported = cells(document, backend="nccl-ep", disposition="unsupported")
        rdma_skus = {"h100-dgxc", "h200-dgxc", "b200-nscale", "b300"}
        gb_skus = {"gb200", "gb300"}
        # RDMA SKUs: EP8 runnable + EP16 unsupported. GB SKUs: EP8 and EP16 both runnable.
        self.assertEqual(
            runnable,
            {(sku, 8) for sku in rdma_skus} | {(sku, ep) for sku in gb_skus for ep in (8, 16)},
        )
        self.assertEqual(unsupported, {(sku, 16) for sku in rdma_skus})
        # Offered on the six NVIDIA SKUs; never on AMD.
        offered = {sku for sku, _ in runnable | unsupported}
        self.assertEqual(offered, rdma_skus | gb_skus)
        for absent in ("mi355x", "mi325x-tw", "mi300x-tw"):
            self.assertNotIn(absent, offered)
        # Every nccl-ep case is BF16 (FP8 unsupported this release).
        self.assertEqual(cells(document, "precision", backend="nccl-ep"), {"bf16"})
        # Low-latency: EP8 on every NVIDIA SKU, and EP8 only — a stray EP16 LL row would dispatch a
        # shape the mode does not define.
        ll = cells(document, backend="nccl-ep", mode="low-latency", disposition="runnable")
        self.assertEqual(ll, {(sku, 8) for sku in rdma_skus | gb_skus})

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
    # A backend added to platform_config but missing from BACKEND_PRECISIONS used to default
    # to bf16-only. That is a MISSING case, not a mislabelled one: run_sweep's
    # non-bf16-dispatch guard can only catch a case that ran, so the fp8 coverage would just
    # be absent from the matrix with nothing anywhere reporting it.
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
