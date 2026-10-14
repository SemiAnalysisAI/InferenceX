#!/usr/bin/env python3
"""Committed results are what the dashboard publishes: each file must be a current, runnable
case measured from committed code, and must carry no cluster identifiers."""
from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sweep_matrix  # noqa: E402

RESULTS = ROOT / "results"
CASES = RESULTS / "cases.json"
SUITES = "ep,swap-blocks"
MAX_BYTES = 1 << 20
SHA = re.compile(r"[0-9a-f]{40}")
IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
# Fields that name machines; harnesses must not write them into a result.
HOST_KEYS = {"host", "hosts", "hostname", "nodelist", "slurm_nodelist", "master_addr"}


def _strings_and_keys(value, keys):
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(key.lower())
            yield from _strings_and_keys(item, keys)
    elif isinstance(value, list):
        for item in value:
            yield from _strings_and_keys(item, keys)
    elif isinstance(value, str):
        yield value


def _identity(document: dict) -> tuple[str, str, object, dict]:
    """(case_id, sku, version, provenance) for an EP case attempt or a swap-blocks result."""
    if document.get("schema") == "collectivex-swap-blocks-v1":
        runtime = document["runtime"]
        return document["case_id"], runtime["sku"], None, runtime
    identity = document["identity"]
    return (identity["case_id"], identity["case_factors"]["sku"], document["version"],
            document["provenance"])


class CommittedResults(unittest.TestCase):
    def test_cases_json_matches_the_configs(self) -> None:
        resolved = sweep_matrix.resolve_matrix(suites=SUITES)
        expected = {key: resolved[key] for key in ("version", "requested_cases")}
        self.assertEqual(
            json.loads(CASES.read_text(encoding="utf-8")), expected,
            f"regenerate: python3 sweep_matrix.py --suites {SUITES} "
            "--cases-json results/cases.json",
        )

    def test_every_result_is_a_runnable_case_from_committed_code(self) -> None:
        cases = json.loads(CASES.read_text(encoding="utf-8"))
        runnable = {(row["sku"], row["case"]["case_id"]) for row in cases["requested_cases"]
                    if row["disposition"] == "runnable"}
        for path in sorted(RESULTS.glob("*/*.json")):
            with self.subTest(result=str(path.relative_to(RESULTS))):
                raw = path.read_bytes()
                self.assertLessEqual(len(raw), MAX_BYTES)
                document = json.loads(raw)
                case_id, sku, version, provenance = _identity(document)
                self.assertEqual((path.parent.name, path.stem), (sku, case_id))
                self.assertIn((sku, case_id), runnable, "not a runnable case in cases.json")
                if version is not None:
                    self.assertEqual(version, cases["version"])
                self.assertRegex(provenance.get("source_sha") or "", SHA)
                self.assertIs(provenance.get("source_dirty"), False,
                              "measured with uncommitted collectivex/ edits")
                keys: set[str] = set()
                leaked = [text for text in _strings_and_keys(document, keys) if IPV4.search(text)]
                self.assertEqual(leaked, [], "IP addresses in a result")
                self.assertEqual(keys & HOST_KEYS, set(), "host-identifying fields in a result")


if __name__ == "__main__":
    unittest.main()
