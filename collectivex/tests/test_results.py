#!/usr/bin/env python3
"""results/cases.json is the committed matrix the CollectiveX upload validates results against, so it
must match the configs. Result files themselves are uploaded to the database, never committed."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sweep_matrix  # noqa: E402

RESULTS = ROOT / "results"
CASES = RESULTS / "cases.json"
SUITES = "ep,swap-blocks"


class CasesJson(unittest.TestCase):
    def test_cases_json_matches_the_configs(self) -> None:
        resolved = sweep_matrix.resolve_matrix(suites=SUITES)
        expected = {key: resolved[key] for key in ("version", "requested_cases")}
        self.assertEqual(
            json.loads(CASES.read_text(encoding="utf-8")), expected,
            f"regenerate: python3 sweep_matrix.py --suites {SUITES} "
            "--cases-json results/cases.json",
        )


if __name__ == "__main__":
    unittest.main()
