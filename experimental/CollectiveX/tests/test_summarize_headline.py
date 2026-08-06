#!/usr/bin/env python3
"""The summary headline: `components.pair_period` when a row carries one, the drained
`roundtrip` otherwise, with the table footnoting which quantity the starred columns hold.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

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


class Headline(unittest.TestCase):
    def test_the_headline_is_the_pair_period_when_a_row_carries_one(self):
        tokens, p50, p99, _, _, carries = summarize._headline(document(with_period=True))
        self.assertEqual((tokens, p50, p99), (64, PERIOD["p50"], PERIOD["p99"]))
        self.assertTrue(carries)
        self.assertIn("chained pair period", summarize.render([document(with_period=True)]))

    def test_a_row_without_a_period_falls_back_to_the_roundtrip(self):
        _, p50, p99, _, _, carries = summarize._headline(document(with_period=False))
        self.assertEqual((p50, p99), (ROUNDTRIP["p50"], ROUNDTRIP["p99"]))
        self.assertFalse(carries)
        self.assertIn(
            "no row here carries a chained pair period",
            summarize.render([document(with_period=False)]),
        )

    def test_a_mixed_table_footnotes_the_fallback_rows_as_incomparable(self):
        rendered = summarize.render(
            [document(with_period=True), document(with_period=False)]
        )
        self.assertIn("1 of 2 row(s) predate it", rendered)
        self.assertIn("do not rank across them", rendered)


if __name__ == "__main__":
    unittest.main()
