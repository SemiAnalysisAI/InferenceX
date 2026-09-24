"""Exercise GPU-pool selection using a controlled platform registry."""

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime.swap_nodes import existing_exclusions
from swap_matrix import build_matrix


class SwapMatrixTests(unittest.TestCase):
    def test_selection_preserves_vendor_and_single_gpu_allocations(self):
        platforms = {"amd-test": {"arch": "gfx942"}, "cuda-test": {"arch": "sm100"}}
        self.assertEqual(
            build_matrix(platforms, "", "cuda-test"),
            {
                "include": [
                    {
                        "id": "swap-amd-test",
                        "sku": "amd-test",
                        "backend": "swap-blocks",
                        "nodes": 1,
                        "gpus_per_node": 1,
                        "scale_up_domain": 1,
                        "launcher": "swap-blocks",
                        "vendor": "amd",
                    }
                ]
            },
        )
        self.assertEqual(
            build_matrix(platforms, "cuda-test", "")["include"][0]["vendor"], "nvidia"
        )
        for only, exclude in [
            ("missing", ""),
            ("", "missing"),
            ("amd-test", "amd-test"),
        ]:
            with (
                self.subTest(only=only, exclude=exclude),
                self.assertRaises(ValueError),
            ):
                build_matrix(platforms, only, exclude)


class SwapNodeTests(unittest.TestCase):
    def test_retired_exclusions_do_not_break_allocation_and_live_ones_remain(self):
        with mock.patch(
            "runtime.swap_nodes.subprocess.check_output",
            side_effect=["retired\nquarantined\n", "healthy\nquarantined\n"],
        ):
            self.assertEqual(existing_exclusions("retired,quarantined"), "quarantined")
        with (
            mock.patch(
                "runtime.swap_nodes.subprocess.check_output", side_effect=["old\n", ""]
            ),
            self.assertRaisesRegex(ValueError, "no nodes"),
        ):
            existing_exclusions("old")
