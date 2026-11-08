#!/usr/bin/env python3
"""vLLM's Kimi-K3.1 transfer geometry, block tables and verification (numpy stands in for the
device)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "bench")]

import kv_workload  # noqa: E402

ROW = 18_800_640


class Geometry(unittest.TestCase):
    """Kimi-K3.1-Pro on one TP16 rank, as the kimi-k31 image registers it. A GB200 production run
    logged num_regions=88, block lengths {425472, 589824, 69632} and KDA state (32256, 393216)."""

    def test_the_registered_kinds_are_vllms_88_regions_and_the_kda_parts(self):
        kinds = kv_workload.PRESETS["k31"]["kinds"]
        regions, parts = kinds[:88], kinds[88:]
        self.assertEqual({size for _, size in regions}, {425_472, 589_824, 69_632})
        self.assertEqual([sum(size == n for _, size in regions)
                          for n in (425_472, 69_632, 589_824)], [35, 24, 29])
        # a KDA state is read as three conv projections and the recurrent state
        self.assertEqual(len(parts), 35 * 4)
        self.assertEqual([size for _, size in parts[:4]], [10_752, 10_752, 10_752, 393_216])
        self.assertEqual(sum(size for _, size in parts[:4]), 425_472)
        self.assertEqual((10_752 * 3, 393_216), (32_256, 393_216))
        # the attention pages tile the row from byte 0; the row is padded to its stride
        attention = sorted(regions[35:])
        self.assertEqual(attention[0][0], 0)
        for (offset, size), (following, _) in zip(attention, attention[1:]):
            self.assertEqual(offset + size, following)
        self.assertEqual(sum(size for _, size in attention), 18_776_064)
        self.assertEqual(kv_workload.PRESETS["k31"]["row_bytes"], 18_800_640)

    def test_a_request_moves_every_prompt_block_and_one_state_block_per_kda_group(self):
        # descs = 53 B + 276 and bytes = 18,776,064 B + 29,357,568 for B = ceil((isl - 1) / 1024).
        # The production run averaged 5,007 descriptors and 1.705 GB per rank at 91.5k new tokens.
        for isl, blocks in ((2048, 2), (1025, 1), (1026, 2), (91_500, 90), (262_144, 256)):
            with self.subTest(isl=isl):
                cfg = kv_workload.plan_config("k31", "fp8", isl)
                self.assertEqual([g["blocks"] for g in cfg["groups"]], [blocks, 1, 1])
                self.assertEqual(cfg["rows"], blocks + 2)
                self.assertEqual(cfg["descs"], 53 * blocks + 276)
                self.assertEqual(cfg["req_bytes"], 18_776_064 * blocks + 29_357_568)
                rows = kv_workload.block_table(cfg, 5)
                local, remote, sizes = kv_workload.nixl_entries(cfg, rows, rows[::-1])
                self.assertEqual((len(sizes), int(sizes.sum())),
                                 (53 * blocks + 276, cfg["req_bytes"]))

    def test_entries_sit_in_their_own_rows_and_never_overlap(self):
        cfg = kv_workload.plan_config("k31", "fp8", 4096)
        local_rows = kv_workload.block_table(cfg, 5)
        remote_rows = kv_workload.block_table(cfg, 6)
        local, remote, sizes = kv_workload.nixl_entries(cfg, local_rows, remote_rows)
        row = cfg["row_bytes"]
        # attention blocks first (53 pages each), then kda-a's 35 x 4 parts, then kda-b's 34 x 4
        owners = [*np.repeat(np.arange(4), 53), *[4] * 140, *[5] * 136]
        self.assertEqual((local // row).tolist(), local_rows[owners].tolist())
        self.assertEqual((remote // row).tolist(), remote_rows[owners].tolist())
        np.testing.assert_array_equal(local % row, remote % row)
        spans = sorted(zip(local.tolist(), sizes.tolist()))
        for (start, size), (following, _) in zip(spans, spans[1:]):
            self.assertLessEqual(start + size, following)
        self.assertTrue((sizes % 8 == 0).all() and (local % 8 == 0).all())

    def test_only_the_served_precision_plans(self):
        with self.assertRaises(ValueError):
            kv_workload.plan_config("k31", "bf16", 2048)


class Tables(unittest.TestCase):
    def test_deterministic_and_distinct_per_side_and_seed(self):
        cfg = kv_workload.plan_config("k31", "fp8", 8192)
        local = kv_workload.table_seed(cfg, "local", 67)
        remote = kv_workload.table_seed(cfg, "remote", 67)
        np.testing.assert_array_equal(kv_workload.block_table(cfg, local),
                                      kv_workload.block_table(cfg, local))
        self.assertFalse(np.array_equal(kv_workload.block_table(cfg, local),
                                        kv_workload.block_table(cfg, remote)))
        self.assertNotEqual(local, kv_workload.table_seed(cfg, "local", 68))

    def test_every_table_set_and_request_takes_disjoint_rows(self):
        cfg = kv_workload.plan_config("k31", "fp8", 2048, batch_max=4)
        rows = np.concatenate([kv_workload.block_table(cfg, 5, r, k)
                               for k in range(cfg["table_sets"]) for r in range(4)])
        self.assertEqual(len(rows), cfg["table_sets"] * 4 * cfg["rows"])
        self.assertEqual(len(rows), len(set(rows.tolist())))
        self.assertLess(rows.max(), cfg["pool_rows"])

    def test_a_request_beyond_the_pool_fails_closed(self):
        cfg = kv_workload.plan_config("k31", "fp8", 2048, table_sets=1)
        with self.assertRaises(ValueError):
            kv_workload.block_table(cfg, 5, request=0, table_set=5)


class Verify(unittest.TestCase):
    SRC_SALT, DST_SALT = 1, 0

    def _setup(self):
        cfg = kv_workload.plan_config("k31", "fp8", 2048, table_sets=1)
        dst_rows = kv_workload.block_table(cfg, kv_workload.table_seed(cfg, "local", 67))
        src_rows = kv_workload.block_table(cfg, kv_workload.table_seed(cfg, "remote", 67))
        dst, src, sizes = kv_workload.nixl_entries(cfg, dst_rows, src_rows)
        words = cfg["pool_bytes"] // 8
        pool = kv_workload.pattern_words(0, words, self.DST_SALT)
        src_pool = kv_workload.pattern_words(0, words, self.SRC_SALT)
        for d, s, n in zip(dst.astype(np.int64) // 8, src.astype(np.int64) // 8,
                           sizes.astype(np.int64) // 8):
            pool[d:d + n] = src_pool[s:s + n]
        return pool, dst, src, sizes

    def test_a_faithful_transfer_verifies(self):
        pool, dst, src, sizes = self._setup()
        ok, detail = kv_workload.verify_entries(pool, dst, src, sizes, self.SRC_SALT)
        self.assertTrue(ok, detail)

    def test_wrong_transfers_fail(self):
        pool, dst, src, sizes = self._setup()
        untouched = kv_workload.pattern_words(0, len(pool), self.DST_SALT)
        wiped = np.full_like(pool, 0x5A5A5A5A5A5A5A5A)
        shifted = pool.copy()
        first = int(dst[0]) // 8
        shifted[first:first + int(sizes[0]) // 8] = np.roll(
            shifted[first:first + int(sizes[0]) // 8], 1)
        truncated = pool.copy()
        truncated[first + int(sizes[0]) // 8 - 1] = untouched[first + int(sizes[0]) // 8 - 1]
        for name, args in (
            # a rep that never happened leaves the destination's own pattern
            ("never happened", (untouched, dst, src, sizes, self.SRC_SALT)),
            # the sentinel wipe before a verified rep survives an empty rep
            ("left the sentinel", (wiped, dst, src, sizes, self.SRC_SALT)),
            ("tables swapped", (pool, src, dst, sizes, self.SRC_SALT)),
            ("loopback", (pool, dst, src, sizes, self.DST_SALT)),
            # one descriptor shifted by a single word, or missing its tail
            ("shifted by 8 bytes", (shifted, dst, src, sizes, self.SRC_SALT)),
            ("tail not moved", (truncated, dst, src, sizes, self.SRC_SALT)),
        ):
            with self.subTest(name):
                ok, detail = kv_workload.verify_entries(*args)
                self.assertFalse(ok)
                self.assertIn("expected", detail)

    def test_bulk_verifies_and_catches_a_stale_buffer(self):
        words = 1 << 16
        src = kv_workload.pattern_words(0, words, self.SRC_SALT)
        self.assertTrue(kv_workload.verify_bulk(src, words * 8, self.SRC_SALT)[0])
        stale = kv_workload.pattern_words(0, words, self.DST_SALT)
        self.assertFalse(kv_workload.verify_bulk(stale, words * 8, self.SRC_SALT)[0])

    def test_no_two_words_of_a_pool_share_a_value(self):
        words = kv_workload.pattern_words(0, 1 << 20, 3)
        self.assertEqual(len(np.unique(words)), len(words))


class Percentiles(unittest.TestCase):
    def test_pcts(self):
        stats = kv_workload.pcts([5.0, 1.0, 3.0, 2.0, 4.0])
        self.assertEqual(stats["p50"], 3.0)
        # Nearest-rank, as the EP suite: an even count takes the lower middle, p95 the 19th of 20.
        self.assertEqual(kv_workload.pcts([4.0, 1.0, 3.0, 2.0])["p50"], 2.0)
        self.assertEqual(kv_workload.pcts([float(i) for i in range(1, 21)])["p95"], 19.0)
        self.assertEqual(stats["min"], 1.0)
        self.assertEqual(stats["max"], 5.0)
        self.assertEqual(stats["n"], 5)


if __name__ == "__main__":
    unittest.main()
