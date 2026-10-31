#!/usr/bin/env python3
"""perf-changelog.yaml is the record an upload links its run to, so its entries must parse, stay
append-only and select real cases."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import changelog  # noqa: E402

ENTRY = b"""- date: 2026-11-01
  pr-link: https://github.com/SemiAnalysisAI/InferenceX/pull/5000
  pools: [b300]
  backends: [deepep-v2]
  description:
    - Build AWS's EFA fork.
  description-zh:
    - \xe6\x94\xb9\xe7\x94\xa8 AWS fork\xe3\x80\x82
"""


def case(sku, **fields):
    return {"sku": sku, "case": {"case_id": f"{sku}-{fields.get('backend', 'x')}", **fields}}


class CommittedChangelog(unittest.TestCase):
    def test_the_committed_file_parses_and_every_entry_selects_cases(self):
        entries = changelog.parse(changelog.CHANGELOG.read_bytes())
        self.assertTrue(entries)
        changelog.check_coverage(entries, changelog.load_cases(), changelog.registry_pools())


class Schema(unittest.TestCase):
    def test_a_well_formed_entry_parses(self):
        (entry,) = changelog.parse(ENTRY)
        self.assertEqual(changelog.pr_number(entry), 5000)

    def test_rejects_unknown_keys_bad_links_and_missing_translations(self):
        for bad in (
            ENTRY.replace(b"backends", b"backend"),
            ENTRY.replace(b"SemiAnalysisAI/InferenceX/pull/5000", b"other/repo/pull/5000"),
            ENTRY.split(b"  description-zh")[0],
            ENTRY.replace(b"2026-11-01", b"Nov 1"),
            ENTRY.replace(b"pools: [b300]", b"pools: []"),
        ):
            with self.subTest(bad=bad[:60]), self.assertRaises(changelog.ChangelogError):
                changelog.parse(bad)

    def test_dates_must_not_go_backwards(self):
        older = ENTRY.replace(b"2026-11-01", b"2026-10-01")
        changelog.parse(older + ENTRY)
        with self.assertRaises(changelog.ChangelogError):
            changelog.parse(ENTRY + older)


class Selection(unittest.TestCase):
    def test_pools_and_selectors_narrow_the_cases(self):
        (entry,) = changelog.parse(ENTRY)
        cases = [case("b300", backend="deepep-v2"), case("b300", backend="nccl-ep"),
                 case("h100-dgxc", backend="deepep-v2")]
        self.assertEqual(changelog.selected_cases(entry, cases), ["b300-deepep-v2"])

    def test_an_entry_selecting_nothing_fails_unless_its_pools_retired(self):
        (entry,) = changelog.parse(ENTRY)
        with self.assertRaises(changelog.ChangelogError):
            changelog.check_coverage([entry], [case("b300", backend="mori")], {"b300"})
        changelog.check_coverage([entry], [], {"h100-dgxc"})


class AppendOnly(unittest.TestCase):
    def test_new_entries_go_at_the_end_and_link_this_pr(self):
        later = ENTRY.replace(b"pull/5000", b"pull/5001")
        self.assertEqual(len(changelog.check_append_only(ENTRY, ENTRY + later, 5001)), 1)
        with self.assertRaises(changelog.ChangelogError):
            changelog.check_append_only(ENTRY, later + ENTRY, 5001)
        with self.assertRaises(changelog.ChangelogError):
            changelog.check_append_only(ENTRY, ENTRY + later, 5002)

    def test_the_pr_creating_the_file_may_seed_earlier_prs(self):
        self.assertEqual(len(changelog.check_append_only(b"", ENTRY, 4999)), 1)


if __name__ == "__main__":
    unittest.main()
