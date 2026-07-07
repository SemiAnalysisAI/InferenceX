#!/usr/bin/env python3
"""CPU-only contract tests for native EP precision adapter wiring."""
from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TESTS))

import ep_precision  # noqa: E402


class PrecisionResolutionTests(unittest.TestCase):
    def test_blank_profile_resolves_to_bf16_control(self):
        profile_id, profile = ep_precision.resolve_precision(
            SimpleNamespace(precision_profile=""),
            backend="nccl-ep",
            mode="normal",
            supported_profiles={"d-bf16.c-bf16"},
        )
        self.assertEqual(profile_id, "d-bf16.c-bf16")
        self.assertEqual(profile["dispatch"]["communication_format"], "bf16")
        self.assertEqual(profile["combine"]["communication_format"], "bf16")

    def test_adapter_profile_rejection_is_fail_closed(self):
        with self.assertRaisesRegex(
            ep_precision.PrecisionError,
            "nccl-ep does not realize precision profile",
        ):
            ep_precision.resolve_precision(
                SimpleNamespace(
                    precision_profile=(
                        "d-fp8-e4m3fn-b128-f32-prequantized.c-bf16"
                    )
                ),
                backend="nccl-ep",
                mode="normal",
                supported_profiles={"d-bf16.c-bf16"},
            )

    def test_profile_mode_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ep_precision.PrecisionError, "not valid in mode"):
            ep_precision.resolve_precision(
                SimpleNamespace(
                    precision_profile="d-fp8-e4m3fn-b128-f32-fused.c-bf16"
                ),
                backend="deepep",
                mode="normal",
                supported_profiles={"d-fp8-e4m3fn-b128-f32-fused.c-bf16"},
            )

    def test_required_native_keyword_is_checked(self):
        def native_api(*, use_fp8=False):
            return use_fp8

        ep_precision.require_keyword(native_api, "use_fp8", api="native_api")
        with self.assertRaisesRegex(ep_precision.PrecisionError, "omits 'use_logfmt'"):
            ep_precision.require_keyword(
                native_api, "use_logfmt", api="native_api"
            )

    def test_bf16_evidence_is_exact_and_has_no_scale_checks(self):
        evidence = ep_precision.exact_axis_evidence()
        self.assertTrue(evidence["passed"])
        self.assertEqual(evidence["max_abs_error"], 0.0)
        self.assertEqual(evidence["max_rel_error"], 0.0)
        self.assertEqual(evidence["saturation_count"], 0)
        self.assertEqual(evidence["saturation_rate"], 0.0)
        self.assertIsNone(evidence["scales_finite"])
        self.assertIsNone(evidence["scales_positive"])


class NativeAdapterWiringTests(unittest.TestCase):
    @staticmethod
    def _tree(name: str) -> ast.Module:
        return ast.parse((TESTS / name).read_text(encoding="utf-8"))

    @staticmethod
    def _keywords(tree: ast.AST, attribute: str) -> list[set[str]]:
        result = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if (
                isinstance(function, ast.Attribute) and function.attr == attribute
            ) or (
                isinstance(function, ast.Name) and function.id == attribute
            ):
                result.append({keyword.arg for keyword in node.keywords})
        return result

    def test_deepep_and_uccl_wire_native_low_latency_controls(self):
        for filename in ("ep_deepep.py", "ep_uccl.py"):
            with self.subTest(filename=filename):
                tree = self._tree(filename)
                dispatch_calls = self._keywords(tree, "low_latency_dispatch")
                combine_calls = self._keywords(tree, "low_latency_combine")
                self.assertTrue(any("use_fp8" in call for call in dispatch_calls))
                self.assertTrue(any("use_logfmt" in call for call in combine_calls))

    def test_elastic_and_hybrid_constructors_enable_native_fp8(self):
        v2 = self._tree("ep_deepep_v2.py")
        hybrid = self._tree("ep_deepep_hybrid.py")
        self.assertTrue(
            any("use_fp8_dispatch" in call for call in self._keywords(v2, "ElasticBuffer"))
        )
        self.assertTrue(
            any("use_fp8" in call for call in self._keywords(hybrid, "HybridEPBuffer"))
        )
        self.assertTrue(
            any("scaling_factor" in call for call in self._keywords(hybrid, "dispatch"))
        )

    def test_mori_wires_dispatch_scales_and_direct_cast_config(self):
        source = (TESTS / "ep_mori.py").read_text(encoding="utf-8")
        harness = (TESTS / "ep_harness.py").read_text(encoding="utf-8")
        precision = (TESTS / "ep_precision.py").read_text(encoding="utf-8")
        self.assertIn('"fp8_direct_cast" if self._direct_cast_combine', source)
        self.assertIn("p.scales,", source)
        self.assertIn("dispatch_scales=_scales", source)
        self.assertIn("dispatch_needs_combine_cleanup = True", source)
        self.assertIn("post=finish_dispatch if dispatch_needs_cleanup else None", harness)
        self.assertEqual(harness.count("view.combine_input = transformed"), 2)
        self.assertIn("transformed = view.combine_input.float()", precision)


if __name__ == "__main__":
    unittest.main()
