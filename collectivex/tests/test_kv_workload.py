#!/usr/bin/env python3
"""vLLM 0.32.0's DSV4 transfer geometry, block tables and verification (numpy stands in
for the device)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "bench")]

import kv_workload  # noqa: E402

ROW = 1_435_968


class Geometry(unittest.TestCase):
    def test_pages_follow_vllms_576_byte_padding(self):
        # round_up(block / tokens_per_state * bytes_per_state, 576), per
        # vLLM's kv_cache_interface: C4A 64 x 584, its indexer 64 x 132, C128A
        # 2 x 584, SWA 64 x 584, the fp32 compressor states 4 x 8192,
        # 4 x 2048 and 8 x 4096.
        self.assertEqual({c: kv_workload.page_bytes(c) for c in kv_workload.DSV4_CACHES}, {
            "c4a": 37_440, "c4a-indexer": 8_640, "c128a": 1_728, "swa": 37_440,
            "c4a-state": 32_832, "c4a-indexer-state": 8_640, "c128a-state": 32_832})

    def test_the_row_is_the_widest_group(self):
        groups = kv_workload.PRESETS["dsv4"]["groups"]
        self.assertEqual({g["name"]: kv_workload.group_block_bytes(g) for g in groups}, {
            "mla": 30 * 37_440 + 30 * 8_640 + 31 * 1_728,       # 1,435,968
            "swa-a": 31 * 37_440, "swa-b": 30 * 37_440,
            "c4a-state": 30 * 32_832 + 30 * 8_640, "c128a-state": 31 * 32_832})
        cfg = kv_workload.plan_config("dsv4", "fp8", 2048)
        self.assertEqual(cfg["row_bytes"], ROW)

    def test_windowed_groups_move_only_their_live_tail(self):
        # vLLM's group order; MLA moves every block. vLLM 0.32.0's prefill tails
        # were [3, 3, 17, 3] ids for the windowed groups at 2048/8192/32768, each
        # led by one null block; the decode side lists only its live blocks and
        # the connectors end-trim to that, so 2, 2, 16 and 2 ids move.
        for isl, blocks, descs in ((2048, [2, 2, 8, 16, 2], 30),
                                   (131072, [2, 2, 512, 16, 2], 534),
                                   (524288, [2, 2, 2048, 16, 2], 2070)):
            with self.subTest(isl=isl):
                cfg = kv_workload.plan_config("dsv4", "fp8", isl)
                self.assertEqual([g["blocks"] for g in cfg["groups"]], blocks)
                self.assertEqual((cfg["descs"], cfg["req_bytes"]), (descs, descs * ROW))

    def test_a_block_leaves_once_the_next_tokens_window_passes_it(self):
        # get_num_skipped_tokens(isl) = isl - window + 1: 191 tokens free swa's
        # first 64-token block (and one block of each state group), 190 do not;
        # a prompt inside every window keeps only the blocks it fills.
        for isl, blocks in ((190, [3, 3, 1, 17, 3]), (191, [2, 2, 1, 16, 2]),
                            (8, [1, 1, 1, 1, 2])):
            with self.subTest(isl=isl):
                cfg = kv_workload.plan_config("dsv4", "fp8", isl)
                self.assertEqual([g["blocks"] for g in cfg["groups"]], blocks)

    def test_only_the_architectural_precision_plans(self):
        with self.assertRaises(ValueError):
            kv_workload.plan_config("dsv4", "bf16", 2048)


class Entries(unittest.TestCase):
    """What one request posts: NIXL's descriptors and Mooncake's transfer plan."""

    def test_nixl_moves_one_whole_row_per_id(self):
        # vLLM 0.32.0 registers the backing allocation as ONE region (no
        # per-layer view of a padded page is the page) and posts a row per id.
        cfg = kv_workload.plan_config("dsv4", "fp8", 2048)
        rows = kv_workload.block_table(cfg, 5)
        local, remote, sizes = kv_workload.nixl_entries(cfg, rows, rows + 1)
        self.assertEqual(local.tolist(), (rows * ROW).tolist())
        self.assertEqual(remote.tolist(), ((rows + 1) * ROW).tolist())
        self.assertEqual(sizes.tolist(), [ROW] * 30)

    def test_mooncake_folds_the_window_views_and_promotes_whole_rows(self):
        # Five transfer regions: swa-a and swa-b fold into one whole row over
        # both groups' ids, swa-a's 31st page (30 x 37,440 in, 584 x 64 long)
        # stays a page region, and MLA plus both state groups tile their rows.
        self.assertEqual(kv_workload.mooncake_regions("dsv4"), (
            ((0, 1), 0, None), ((0,), 1_123_200, 37_376),
            ((2,), 0, None), ((3,), 0, None), ((4,), 0, None)))

    def test_a_mooncake_request_follows_its_regions(self):
        # Scattered tables: a whole row per id, plus swa-a's ids again as pages.
        cfg = kv_workload.plan_config("dsv4", "fp8", 2048)
        local = kv_workload.block_table(cfg, kv_workload.table_seed(cfg, "local", 67))
        remote = kv_workload.block_table(cfg, kv_workload.table_seed(cfg, "remote", 67))
        lo, re, sizes = kv_workload.mooncake_entries(cfg, local, remote)
        self.assertEqual((len(sizes), int(sizes.sum())), (32, 30 * ROW + 2 * 37_376))
        pages = sizes == 37_376
        self.assertEqual(lo[pages].tolist(), (local[:2] * ROW + 1_123_200).tolist())
        self.assertEqual(re[pages].tolist(), (remote[:2] * ROW + 1_123_200).tolist())
        # Runs coalesce where both sides advance by one: across swa-a into swa-b
        # (one region), never on the page region, never across regions (the
        # state groups' rows continue each other here), and they break where
        # only the remote side jumps (inside the MLA ids).
        local = np.array([10, 11, 12, 13, *range(30, 38), *range(40, 56), 56, 57])
        remote = np.array([110, 111, 112, 113, 130, 131, 132, 133, 170, 171, 172, 173,
                           *range(140, 156), 156, 157])
        lo, re, sizes = kv_workload.mooncake_entries(cfg, local, remote)
        self.assertEqual((sizes // ROW).tolist(), [4, 0, 0, 4, 4, 16, 2])
        self.assertEqual(lo[sizes < ROW].tolist(), [row * ROW + 1_123_200 for row in (10, 11)])
        self.assertEqual(re[sizes // ROW == 4].tolist(), [110 * ROW, 130 * ROW, 170 * ROW])


class Tables(unittest.TestCase):
    def test_deterministic_and_distinct_per_side_and_seed(self):
        cfg = kv_workload.plan_config("dsv4", "fp8", 8192)
        local = kv_workload.table_seed(cfg, "local", 67)
        remote = kv_workload.table_seed(cfg, "remote", 67)
        np.testing.assert_array_equal(kv_workload.block_table(cfg, local),
                                      kv_workload.block_table(cfg, local))
        self.assertFalse(np.array_equal(kv_workload.block_table(cfg, local),
                                        kv_workload.block_table(cfg, remote)))
        self.assertNotEqual(local, kv_workload.table_seed(cfg, "local", 68))

    def test_every_table_set_and_request_takes_disjoint_rows(self):
        cfg = kv_workload.plan_config("dsv4", "fp8", 2048, batch_max=4)
        rows = np.concatenate([kv_workload.block_table(cfg, 5, r, k)
                               for k in range(cfg["table_sets"]) for r in range(4)])
        self.assertEqual(len(rows), cfg["table_sets"] * 4 * 30)
        self.assertEqual(len(rows), len(set(rows.tolist())))
        self.assertLess(rows.max(), cfg["pool_rows"])

    def test_a_request_beyond_the_pool_fails_closed(self):
        cfg = kv_workload.plan_config("dsv4", "fp8", 2048, table_sets=1)
        with self.assertRaises(ValueError):
            kv_workload.block_table(cfg, 5, request=0, table_set=5)


class Verify(unittest.TestCase):
    SRC_SALT, DST_SALT = 1, 0

    def _setup(self, entries="rows"):
        cfg = kv_workload.plan_config("dsv4", "fp8", 2048)
        dst_rows = kv_workload.block_table(cfg, kv_workload.table_seed(cfg, "local", 67))
        src_rows = kv_workload.block_table(cfg, kv_workload.table_seed(cfg, "remote", 67))
        if entries == "rows":
            dst, src, sizes = kv_workload.nixl_entries(cfg, dst_rows, src_rows)
        else:
            dst, src, sizes = kv_workload.mooncake_entries(cfg, dst_rows, src_rows)
        words = cfg["pool_bytes"] // 8
        pool = kv_workload.pattern_words(0, words, self.DST_SALT)
        src_pool = kv_workload.pattern_words(0, words, self.SRC_SALT)
        for d, s, n in zip(dst.astype(np.int64) // 8, src.astype(np.int64) // 8,
                           sizes.astype(np.int64) // 8):
            pool[d:d + n] = src_pool[s:s + n]
        return pool, dst, src, sizes

    def test_a_faithful_transfer_verifies(self):
        for entries in ("rows", "mooncake"):
            with self.subTest(entries=entries):
                pool, dst, src, sizes = self._setup(entries)
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
