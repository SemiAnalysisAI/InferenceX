#!/usr/bin/env python3
"""Math tests for the bandwidth consumer (unit conversion, alpha/beta fit, fit gating)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1])]

import bandwidth  # noqa: E402

COMPONENTS = bandwidth.COMPONENTS


def _row(tokens, nbytes, latency, passed=True):
    """A measurement row. `latency` is a scalar, or a per-component dict whose None marks
    that component unavailable."""
    lat = latency if isinstance(latency, dict) else dict.fromkeys(COMPONENTS, latency)
    return {
        "tokens_per_rank": tokens,
        "components": {c: {"percentiles_us": None if lat[c] is None else {
            "p50": lat[c], "p90": lat[c], "p95": lat[c], "p99": lat[c] * 2.0}}
            for c in COMPONENTS},
        "byte_provenance": {c: {"total_logical_bytes": nbytes} for c in COMPONENTS},
        "correctness": {"passed": passed},
        "routing": {"locality": {"cross_node_fraction": 0.5}},
    }


def _doc(rows, ep=2, mode="normal"):
    case = {"ep": ep, "backend": "deepep-v2", "precision": "bf16", "phase": "decode",
            "mode": mode, "suite": "s", "routing": "uniform"}
    return {
        "generated_at": "2026-07-25T22:12:55.760511+00:00",
        "identity": {"attempt_ordinal": 1, "allocation_factors": {"run_id": "30177021271"},
                     "case_factors": {"sku": "h100", "case": case}},
        "measurement": {"rows": rows},
        "outcome": {"status": "success"},
    }


def _linear(pairs, passed=True):
    """Rows exactly on latency = 10us + bytes * 2e-6, i.e. alpha=10, beta_agg=500 GB/s."""
    return [_row(t, b, 10.0 + b * 2e-6, passed) for t, b in pairs]


LADDER = ((8, 1e6), (16, 2e6), (32, 3e6))


class BandwidthMath(unittest.TestCase):
    def test_algbw_per_gpu(self):
        # 1e9 bytes in 1000us = 1e12 B/s = 1000 GB/s aggregate; /ep(2) = 500 per GPU.
        self.assertAlmostEqual(bandwidth._algbw_per_gpu(1e9, 1000.0, 2), 500.0)
        self.assertIsNone(bandwidth._algbw_per_gpu(1e9, 0.0, 2))

    def test_fit_recovers_alpha_beta(self):
        fit = bandwidth.fit_alpha_beta(_doc(_linear(LADDER)), "dispatch")
        self.assertAlmostEqual(fit.alpha_us, 10.0, places=4)
        self.assertAlmostEqual(fit.beta_gbps, 250.0, places=4)  # 500 aggregate / ep(2)
        self.assertAlmostEqual(fit.r2, 1.0, places=6)
        self.assertEqual(fit.points, 3)
        self.assertTrue(fit.beta_is_reliable)

    def test_fit_is_none_when_undefensible(self):
        flat = [_row(t, b, 12.0) for t, b in LADDER]            # slope <= 0
        self.assertIsNone(bandwidth.fit_alpha_beta(_doc(_linear(LADDER[:2])), "dispatch"))
        self.assertIsNone(bandwidth.fit_alpha_beta(_doc(flat), "dispatch"))

    def test_noisy_ladder_withholds_beta(self):
        # A positive slope through noise still fits; printing its beta once produced a
        # physically impossible 1018 GB/s per GPU on a B200 at R2 = 0.29.
        rows = [_row(t, b, lat) for t, b, lat in (
            (8, 1e6, 300.0), (16, 2e6, 40.0), (32, 3e6, 260.0),
            (64, 4e6, 60.0), (128, 5e6, 320.0))]
        fit = bandwidth.fit_alpha_beta(_doc(rows), "dispatch")
        self.assertLess(fit.r2, bandwidth.FIT_MIN_R2)
        self.assertFalse(fit.beta_is_reliable)
        out = bandwidth.render([_doc(rows)])
        self.assertIn("beta=unreliable", out)
        self.assertNotIn("GB/s alpha", out)  # no number presented as measured

    def test_latency_bound_ladder_withholds_beta_despite_high_r2(self):
        # Real data gave beta = 3763 GB/s at R2 = 0.92: a near-zero slope explodes beta while
        # the line still fits, so R2 cannot catch it — the transfer-share gate must.
        rows = [_row(t, b, 500.0 + b * 1e-9)
                for t, b in LADDER + ((64, 4e6), (128, 5e6))]
        fit = bandwidth.fit_alpha_beta(_doc(rows), "dispatch")
        self.assertGreater(fit.r2, bandwidth.FIT_MIN_R2)
        self.assertLess(fit.bandwidth_share, bandwidth.FIT_MIN_BANDWIDTH_SHARE)
        self.assertFalse(fit.beta_is_reliable)
        self.assertIn("not bandwidth-bound", bandwidth._format_fit("dispatch", fit))

    def test_alpha_marked_only_when_extrapolated(self):
        prefill = bandwidth.fit_alpha_beta(  # starts at T=1024: intercept is extrapolated
            _doc(_linear(((1024, 1e9), (2048, 2e9), (4096, 4e9), (8192, 8e9)))), "dispatch")
        decode = bandwidth.fit_alpha_beta(   # reaches near zero bytes: alpha stands
            _doc(_linear(((1, 1e5), (64, 6.4e6), (512, 5.12e7)))), "dispatch")
        self.assertTrue(prefill.alpha_extrapolated)
        self.assertFalse(decode.alpha_extrapolated)
        self.assertIn("*", bandwidth._format_fit("dispatch", prefill))
        self.assertNotIn("*", bandwidth._format_fit("dispatch", decode))

    def test_gate_failed_rung_excluded_from_fit_and_marked(self):
        rows = _linear(LADDER) + [_row(64, 4e6, 999.0, passed=False)]
        fit = bandwidth.fit_alpha_beta(_doc(rows), "dispatch")
        self.assertEqual((fit.points, fit.excluded_rows), (3, 1))
        self.assertAlmostEqual(fit.beta_gbps, 250.0, places=4)  # the corrupt rung didn't steer it
        out = bandwidth.render([_doc(rows)])
        self.assertIn("[correctness FAILED]", out)
        self.assertIn("excluded 1 gate-failed rung", out)

    def test_render_marks_unavailable_and_separates_attempts(self):
        unavailable = [_row(t, b, {"dispatch": None, "combine": 5.0, "roundtrip": 6.0})
                       for t, b in LADDER]
        out = bandwidth.render([_doc(unavailable)])
        self.assertIn("dispatch=n/a", out)
        self.assertIn("xnode=  50%", out)
        second = _doc(_linear(LADDER))
        second["identity"]["attempt_ordinal"] = 2
        second["generated_at"] = "2026-07-25T23:00:00.000000+00:00"
        out = bandwidth.render([_doc(_linear(LADDER)), second])
        self.assertIn("attempt 1", out)
        self.assertIn("attempt 2", out)
        self.assertIn("run 30177021271", out)


if __name__ == "__main__":
    unittest.main()
