#!/usr/bin/env python3
"""CPU-only contract tests for native EP precision adapter wiring."""
from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
