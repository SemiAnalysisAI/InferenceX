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

    def test_shards_run_on_the_registry_runner_label_else_the_sku(self):
        runners = {shard["sku"]: shard["runner"] for shard in matrix(backend="all")["include"]}
        self.assertEqual(runners["mi325x"], "cluster:mi325x-amds")
        self.assertEqual(runners["h200-dgxc"], "h200-dgxc")

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


class SwapSuiteTests(unittest.TestCase):
    PLATFORMS = {
        "amd-test": {"arch": "gfx942", "launcher": "mi-amds"},
        "cuda-test": {"arch": "sm100", "launcher": "single-slurm",
                      "runner_label": "cluster:cuda-pool"},
    }

    def _swap(self, **options):
        with mock.patch.object(sweep_matrix, "PLATFORMS", self.PLATFORMS):
            return matrix(suites="swap-blocks", **options)

    def test_one_single_gpu_shard_per_pool_on_its_own_launcher(self):
        shards = {shard["sku"]: shard for shard in self._swap()["include"]}
        self.assertEqual(set(shards), {"amd-test", "cuda-test"})
        amd, cuda = shards["amd-test"], shards["cuda-test"]
        self.assertEqual((amd["launcher"], cuda["launcher"]), ("mi-amds", "single-slurm"))
        self.assertEqual(cuda["runner"], "cluster:cuda-pool")
        self.assertEqual(amd["image"], sweep_matrix.SWAP_SWEEP["images"]["amd"])
        self.assertEqual(cuda["image"], sweep_matrix.SWAP_SWEEP["images"]["nvidia"])
        for shard in shards.values():
            self.assertEqual((shard["nodes"], shard["gpus_per_node"]), (1, 1))
            self.assertEqual([case["layout"] for case in shard["cases"]],
                             sweep_matrix.SWAP_SWEEP["layouts"])
            self.assertNotIn("staged_image_dir", shard)

    def test_a_pinned_pool_carries_its_image_and_staged_cache(self):
        pinned = sweep_matrix.SWAP_SWEEP["sku_images"]["h100-dgxc"]
        shard, = matrix(suites="swap-blocks", only_sku="h100-dgxc")["include"]
        self.assertEqual(shard["image"], pinned["image"])
        self.assertEqual(shard["staged_image_dir"], pinned["staged_image_dir"])

    def test_the_profile_sets_the_grid(self):
        for name, profile in sweep_matrix.SWAP_SWEEP["profiles"].items():
            with self.subTest(profile=name):
                case = self._swap(swap_profile=name)["include"][0]["cases"][0]
                self.assertEqual(case["block_bytes"], " ".join(map(str, profile["block_bytes"])))
                self.assertEqual(case["iterations"], profile["iterations"])
                self.assertEqual(case["case_id"], f"amd-test-swap-blocks-{name}-contiguous")

    def test_sku_selection_and_bad_requests_fail_closed(self):
        self.assertEqual([s["sku"] for s in self._swap(exclude_skus="amd-test")["include"]],
                         ["cuda-test"])
        for options in (
            {"swap_profile": "huge"}, {"only_sku": "missing"}, {"exclude_skus": "missing"},
            {"modes": "normal"}, {"ep_sizes": "8"}, {"backend": "mori"},
        ):
            with self.subTest(options=options), self.assertRaises(SystemExit):
                self._swap(**options)
        for suites in ("", "turbo"):
            with self.subTest(suites=suites), self.assertRaises(SystemExit):
                matrix(suites=suites)

    def test_adding_the_suite_leaves_the_ep_shards_unchanged(self):
        ep = matrix()["include"]
        both = matrix(suites="ep,swap-blocks")["include"]
        self.assertEqual([shard for shard in both if shard["backend"] != "swap-blocks"], ep)
        self.assertEqual(len(both) - len(ep), len(sweep_matrix.PLATFORMS))


if __name__ == "__main__":
    unittest.main()
