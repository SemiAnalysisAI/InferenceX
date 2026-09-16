"""Exercise GPU-pool selection using a controlled platform registry."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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
