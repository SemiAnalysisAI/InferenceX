#!/usr/bin/env python3
"""The summary headline's hold on the chained pair period.

`summarize._headline` prefers `components.pair_period` for the starred p50/p99 columns ONLY
when `HEADLINE_PREFERS_PAIR_PERIOD` says the fleet's chained numbers are trustworthy. The
first fleet sweep to carry the field measured it with the six-events-per-pair chain, which
charges four host-side record() calls into the pair window at the bottom of the ladder, so
the constant ships False until a post-fix validation sweep is cross-checked against the
hand-measured references. These tests pin both positions of the switch, so flipping it back
on is a reviewed one-line diff and an accidental flip (or an accidental permanent hold) is a
test failure, not a silent headline change.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]

import summarize  # noqa: E402

ROUNDTRIP = {"p50": 100.0, "p90": 110.0, "p95": 115.0, "p99": 120.0}
PERIOD = {"p50": 60.0, "p90": 66.0, "p95": 69.0, "p99": 72.0}


def document(with_period):
    components = {
        "roundtrip": {"percentiles_us": dict(ROUNDTRIP)},
    }
    if with_period:
        components["pair_period"] = {
            "percentiles_us": dict(PERIOD), "origin": "chained-median",
        }
    return {
        "version": 1,
        "outcome": {"status": "success"},
        "identity": {
            "case_factors": {
                "sku": "stub-sku",
                "case": {
                    "backend": "stub", "suite": "ep-core", "routing": "uniform",
                    "mode": "low-latency", "phase": "decode", "ep": 8,
                    "precision": "bf16",
                },
            },
        },
        "topology": {"gpus_per_node": 8, "scale_up_domain": 8, "nodes": 1},
        "measurement": {
            "rows": [{
                "tokens_per_rank": 64,
                "components": components,
                "logical_copies": {"wire": "per-assignment"},
                "cross_rank_min_us": {"roundtrip": {"percentiles_us": {"p50": 90.0}}},
                "cross_rank_spread_us": {"percentiles_us": {"p50": 5.0}},
            }],
        },
    }


class HeadlineHold(unittest.TestCase):
    def test_the_hold_ships_engaged(self):
        # The intent of the current tree: no inflated fleet period reaches the top line.
        self.assertFalse(summarize.HEADLINE_PREFERS_PAIR_PERIOD)

    def test_under_the_hold_the_headline_is_the_roundtrip_even_when_a_period_exists(self):
        with mock.patch.object(summarize, "HEADLINE_PREFERS_PAIR_PERIOD", False):
            tokens, p50, p99, _, _, carries = summarize._headline(document(with_period=True))
        self.assertEqual((tokens, p50, p99), (64, ROUNDTRIP["p50"], ROUNDTRIP["p99"]))
        # Presence is still reported, so the footnote can say a period exists unheadlined.
        self.assertTrue(carries)

    def test_the_hold_footnote_names_the_unheadlined_periods(self):
        with mock.patch.object(summarize, "HEADLINE_PREFERS_PAIR_PERIOD", False):
            rendered = summarize.render([document(with_period=True)])
        self.assertIn("drained `roundtrip`", rendered)
        self.assertIn("NOT headlined", rendered)
        self.assertIn(f"| {ROUNDTRIP['p50']} | {ROUNDTRIP['p99']} |", rendered)

    def test_released_the_headline_prefers_the_period_again(self):
        with mock.patch.object(summarize, "HEADLINE_PREFERS_PAIR_PERIOD", True):
            _, p50, p99, _, _, carries = summarize._headline(document(with_period=True))
            rendered = summarize.render([document(with_period=True)])
        self.assertEqual((p50, p99), (PERIOD["p50"], PERIOD["p99"]))
        self.assertTrue(carries)
        self.assertIn("chained pair period", rendered)

    def test_a_row_without_a_period_falls_back_to_the_roundtrip_in_both_positions(self):
        for prefers in (False, True):
            with self.subTest(prefers=prefers), \
                    mock.patch.object(summarize, "HEADLINE_PREFERS_PAIR_PERIOD", prefers):
                _, p50, p99, _, _, carries = summarize._headline(document(with_period=False))
            self.assertEqual((p50, p99), (ROUNDTRIP["p50"], ROUNDTRIP["p99"]))
            self.assertFalse(carries)


if __name__ == "__main__":
    unittest.main()
