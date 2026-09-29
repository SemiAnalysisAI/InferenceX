#!/usr/bin/env python3
"""Geometry and correctness math of the KV-transfer workload model.

The contract is vLLM's (32ad1400d7) DSV4 KV layout: five cache groups over
one shared block-row allocation as wide as the widest group, whole-row NIXL
descriptors, per-group window tails, per-layer pages for layer-registering
connectors, seed-keyed disjoint block tables, and a unique-word per-rank
salted pattern whose every word follows from its source offset and source
rank alone. The torch fill path runs on metal (a wrong fill fails every verify
row); here a numpy pool stands in for the device.
"""
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
        cfg = kv_workload.plan_config("dsv4", "fp8", 2048, 256)
        self.assertEqual((cfg["row_bytes"], cfg["page_bytes"]), (ROW, ROW))

    def test_block_counts_follow_the_window_tail_clip(self):
        # MLA: every block; windowed groups: cdiv(window, block) + 1, clipped
        # to the prompt. 2048 -> [8, 3, 3, 3, 17]; the table vLLM's own
        # geometry derivation produced for the ladder.
        for isl, blocks, descs in ((2048, [8, 3, 3, 3, 17], 34),
                                   (131072, [512, 3, 3, 3, 17], 538),
                                   (524288, [2048, 3, 3, 3, 17], 2074)):
            with self.subTest(isl=isl):
                cfg = kv_workload.plan_config("dsv4", "fp8", isl, 256)
                self.assertEqual([g["blocks"] for g in cfg["groups"]], blocks)
                self.assertEqual((cfg["descs"], cfg["req_bytes"]), (descs, descs * ROW))
        # a prompt shorter than a window keeps only the blocks it fills
        short = kv_workload.plan_config("dsv4", "fp8", 8, 256)
        self.assertEqual([g["blocks"] for g in short["groups"]], [1, 1, 1, 2, 1])

    def test_per_layer_pages_tile_each_groups_share_of_the_row(self):
        for group in kv_workload.layer_layout("dsv4"):
            with self.subTest(group=group["name"]):
                offset = 0
                for start, size in group["layers"]:
                    self.assertEqual(start, offset)
                    offset += size
        # the MLA group lays out C128A, then the indexer, then C4A from byte 0
        mla = kv_workload.layer_layout("dsv4")[0]["layers"]
        self.assertEqual((mla[0], mla[31], mla[61], len(mla)),
                         ((0, 1_728), (53_568, 8_640), (312_768, 37_440), 91))

    def test_layer_entries_count_and_bytes(self):
        # 91 x ceil(L/256) + 890 entries (31x3 + 30x3 + 60x3 + 31x17 window
        # pages); pages carry no row padding, so bytes fall below descs x row
        for isl, entries, nbytes in ((2048, 1_618, 39_374_208),
                                     (524288, 187_258, 2_968_748_928)):
            with self.subTest(isl=isl):
                cfg = kv_workload.plan_config("dsv4", "fp8", isl, 256)
                offsets, sizes = kv_workload.layer_entries(cfg, kv_workload.block_table(cfg, 1))
                self.assertEqual((len(offsets), int(sizes.sum())), (entries, nbytes))

    def test_only_the_served_block_size_and_precision_plan(self):
        with self.assertRaises(ValueError):
            kv_workload.plan_config("dsv4", "fp8", 2048, 128)
        with self.assertRaises(ValueError):
            kv_workload.plan_config("dsv4", "bf16", 2048, 256)

    def test_the_pool_holds_every_table_set_and_batch_disjointly(self):
        one = kv_workload.plan_config("dsv4", "fp8", 8192, 256, table_sets=1)
        four = kv_workload.plan_config("dsv4", "fp8", 8192, 256, batch_max=8)
        self.assertEqual(one["pool_rows"], int(58 * 2.0) + 8)
        self.assertEqual(four["pool_rows"], int(58 * 4 * 8 * 1.25) + 8)


class Tables(unittest.TestCase):
    def test_deterministic_and_distinct_per_side_and_seed(self):
        cfg = kv_workload.plan_config("dsv4", "fp8", 8192, 256)
        local = kv_workload.table_seed(cfg, "local", 67)
        remote = kv_workload.table_seed(cfg, "remote", 67)
        np.testing.assert_array_equal(kv_workload.block_table(cfg, local),
                                      kv_workload.block_table(cfg, local))
        self.assertFalse(np.array_equal(kv_workload.block_table(cfg, local),
                                        kv_workload.block_table(cfg, remote)))
        self.assertNotEqual(local, kv_workload.table_seed(cfg, "local", 68))

    def test_every_table_set_and_request_takes_disjoint_rows(self):
        cfg = kv_workload.plan_config("dsv4", "fp8", 2048, 256, batch_max=4)
        rows = np.concatenate([kv_workload.block_table(cfg, 5, r, k)
                               for k in range(cfg["table_sets"]) for r in range(4)])
        self.assertEqual(len(rows), len(set(rows.tolist())))
        self.assertLess(rows.max(), cfg["pool_rows"])

    def test_a_request_beyond_the_pool_fails_closed(self):
        cfg = kv_workload.plan_config("dsv4", "fp8", 2048, 256, table_sets=1)
        with self.assertRaises(ValueError):
            kv_workload.block_table(cfg, 5, request=0, table_set=5)


class Verify(unittest.TestCase):
    SRC_SALT, DST_SALT = 1, 0

    def _setup(self, entries="rows"):
        cfg = kv_workload.plan_config("dsv4", "fp8", 2048, 256)
        dst_rows = kv_workload.block_table(cfg, kv_workload.table_seed(cfg, "local", 67))
        src_rows = kv_workload.block_table(cfg, kv_workload.table_seed(cfg, "remote", 67))
        if entries == "rows":
            dst, src = (kv_workload.page_offsets(cfg, r) for r in (dst_rows, src_rows))
            sizes = kv_workload.desc_sizes(cfg)
        else:
            dst, sizes = kv_workload.layer_entries(cfg, dst_rows)
            src, _ = kv_workload.layer_entries(cfg, src_rows)
        words = cfg["pool_bytes"] // 8
        pool = kv_workload.pattern_words(0, words, self.DST_SALT)
        src_pool = kv_workload.pattern_words(0, words, self.SRC_SALT)
        for d, s, n in zip(dst.astype(np.int64) // 8, src.astype(np.int64) // 8,
                           sizes.astype(np.int64) // 8):
            pool[d:d + n] = src_pool[s:s + n]
        return pool, dst, src, sizes

    def test_a_faithful_transfer_verifies(self):
        for entries in ("rows", "layers"):
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


class SweepConfigConsistency(unittest.TestCase):
    def test_every_scheduled_workload_point_is_plannable(self):
        # sweep_matrix stays stdlib-only, so it schedules from kv_sweep.json's
        # workload -> precision map without importing this model; a precision
        # or block size plan_config rejects would kill every kv leg at its
        # first grid point.
        import sweep_matrix

        for workload, precisions in sweep_matrix.KV_SWEEP["workloads"].items():
            for precision in precisions:
                for block in sweep_matrix.KV_SWEEP["page_tokens"]:
                    with self.subTest(workload=workload, precision=precision, block=block):
                        kv_workload.plan_config(workload.removeprefix("kv-"), precision,
                                                512, block)


class Percentiles(unittest.TestCase):
    def test_pcts(self):
        stats = kv_workload.pcts([5.0, 1.0, 3.0, 2.0, 4.0])
        self.assertEqual(stats["p50"], 3.0)
        self.assertEqual(stats["min"], 1.0)
        self.assertEqual(stats["max"], 5.0)
        self.assertEqual(stats["n"], 5)


if __name__ == "__main__":
    unittest.main()
