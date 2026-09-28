#!/usr/bin/env python3
"""Geometry and correctness math of the KV-transfer workload model.

The packed block-major layout is the contract: per cache-group region, one
contiguous descriptor covers all the group's layers for one physical block
(vLLM's packed DSV4 NIXL shape), block tables are seed-keyed permutations both
ranks derive independently (batched requests slicing disjoint ranges of one
permutation), and an offset-derived, per-rank-salted pattern makes any byte's
expected value computable from its source offset and source rank alone.
These tests pin that math with hand-computed cases validated against vLLM commit 32ad1400d7 (state content 584 B, page
padded to a 576 B multiple at block granularity, one descriptor per packed
block); the torch fill path is exercised on metal by the suite itself (a wrong
fill fails every verify row loudly).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "bench")]

import kv_workload  # noqa: E402


def _read8(pool: np.ndarray):
    return lambda offset: pool[offset : offset + 8].tobytes()


class Geometry(unittest.TestCase):
    def test_dsv4_regions_by_hand(self):
        # isl=512, block=256. Every token-state is 584 B (448 NoPE + 128 RoPE
        # + 8 fp8 scale); pages pad to a 576 B multiple at BLOCK granularity.
        # C4A: 64 states -> round_up(64*584, 576) = 37,440; its indexer keeps
        # 132 B states -> round_up(64*132, 576) = 8,640; C128A: 2 states ->
        # round_up(2*584, 576) = 1,728; the sliding window's block is fixed at
        # 64 tokens (it shares C4A's physical tensor) -> 37,440 on all 61
        # layers, capped at 128 window tokens. One descriptor per block spans
        # the group's layers.
        cfg = kv_workload.plan_config("dsv4", "fp8", 512, 256)
        regions = {r["name"]: r for r in cfg["regions"]}
        self.assertEqual([r["name"] for r in cfg["regions"]],
                         ["c4a", "c4a-idx", "c128a", "swa"])
        self.assertEqual(
            (regions["c4a"]["layers"], regions["c4a"]["page_bytes"],
             regions["c4a"]["packed_bytes"], regions["c4a"]["blocks_req"]),
            (30, 37_440, 30 * 37_440, 2))
        self.assertEqual(
            (regions["c4a-idx"]["layers"], regions["c4a-idx"]["page_bytes"],
             regions["c4a-idx"]["blocks_req"]), (30, 8_640, 2))
        self.assertEqual(
            (regions["c128a"]["layers"], regions["c128a"]["page_bytes"],
             regions["c128a"]["blocks_req"]), (31, 1_728, 2))
        # the window shares C4A's physical tensor, so its page equals C4A's
        self.assertEqual(
            (regions["swa"]["layers"], regions["swa"]["block_tokens"],
             regions["swa"]["page_bytes"], regions["swa"]["blocks_req"]),
            (61, 64, 37_440, 2))
        self.assertEqual(cfg["descs"], 2 + 2 + 2 + 2)
        self.assertEqual(cfg["req_bytes"],
                         2 * (30 * 37_440 + 30 * 8_640 + 31 * 1_728 + 61 * 37_440))
        # regions tile one contiguous pool
        self.assertEqual(cfg["pool_bytes"],
                         sum(r["pool_blocks"] * r["packed_bytes"]
                             for r in cfg["regions"]))

    def test_a_short_request_uses_only_its_own_window_tokens(self):
        # min(isl, 128) = 64 tokens -> one 64-token window block
        small = kv_workload.plan_config("dsv4", "fp8", 64, 256)
        self.assertEqual({r["name"]: r for r in small["regions"]}["swa"]["blocks_req"], 1)

    def test_block_sizes_that_split_a_state_fail_closed(self):
        # C128A's 128-token states force the model block size to a multiple
        # of 128; vLLM serves DSV4 at 256. The old 16/64-token sweep values
        # cannot hold a whole HCA state and must be rejected.
        for block in (16, 64, 192):
            with self.assertRaises(ValueError):
                kv_workload.plan_config("dsv4", "fp8", 512, block)
        self.assertEqual(
            {r["name"]: r for r in
             kv_workload.plan_config("dsv4", "fp8", 512, 128)["regions"]
             }["c128a"]["page_bytes"], 1_152)  # 1 state, 584 -> padded

    def test_dsv4_precision_is_architectural(self):
        with self.assertRaises(ValueError):
            kv_workload.plan_config("dsv4", "bf16", 512, 256)

    def test_partial_last_block_rounds_up(self):
        # 300 tokens at 256/block -> 2 blocks for every non-window group.
        cfg = kv_workload.plan_config("dsv4", "fp8", 300, 256)
        self.assertEqual(cfg["regions"][0]["blocks_req"], 2)

    def test_batch_max_grows_the_pool_for_disjoint_requests(self):
        cfg = kv_workload.plan_config("dsv4", "fp8", 512, 256, batch_max=16)
        for region in cfg["regions"]:
            self.assertGreaterEqual(region["pool_blocks"], 16 * region["blocks_req"])


class Tables(unittest.TestCase):
    def test_deterministic_and_distinct_per_side_and_seed(self):
        cfg = kv_workload.plan_config("dsv4", "fp8", 4096, 256)

        def table(side, seed=67):
            return kv_workload.block_table(cfg, kv_workload.table_seed(cfg, side, seed))

        local, remote, again, reseeded = table("local"), table("remote"), table("local"), \
            table("local", seed=68)
        for region in cfg["regions"]:
            name, blocks_req = region["name"], region["blocks_req"]
            self.assertTrue((local[name] == again[name]).all())
            self.assertFalse((local[name] == remote[name]).all())
            self.assertFalse((local[name] == reseeded[name]).all())
            # distinct in-range blocks (fragmented, never aliased)
            self.assertEqual(len(set(local[name].tolist())), blocks_req)
            self.assertTrue((local[name] < region["pool_blocks"]).all())

    def test_batched_requests_slice_disjoint_blocks(self):
        cfg = kv_workload.plan_config("dsv4", "fp8", 512, 256, batch_max=4)
        seed = kv_workload.table_seed(cfg, "local", 67)
        tables = [kv_workload.block_table(cfg, seed, request=r) for r in range(4)]
        for region in cfg["regions"]:
            blocks = [t[region["name"]].tolist() for t in tables]
            union = set().union(*map(set, blocks))
            self.assertEqual(len(union), 4 * region["blocks_req"])

    def test_a_request_beyond_the_pool_fails_closed(self):
        cfg = kv_workload.plan_config("dsv4", "fp8", 512, 256)  # slack for ~2 requests
        with self.assertRaises(ValueError):
            kv_workload.block_table(cfg, 1, request=8)

    def test_desc_array_carries_per_region_packed_sizes(self):
        cfg = dict(regions=[
            dict(name="a", packed_bytes=256, blocks_req=2, pool_blocks=4, base=0),
            dict(name="b", packed_bytes=132, blocks_req=1, pool_blocks=4, base=1024),
        ], descs=3)
        tables = {"a": np.array([1, 3]), "b": np.array([2])}
        descs = kv_workload.desc_array(10_000, cfg, tables, dev=5)
        self.assertEqual(descs[:, 0].tolist(),
                         [10_000 + 256, 10_000 + 768, 10_000 + 1024 + 264])
        self.assertEqual(descs[:, 1].tolist(), [256, 256, 132])
        self.assertEqual(descs[:, 2].tolist(), [5, 5, 5])


class Verify(unittest.TestCase):
    SRC_SALT, DST_SALT = 1, 0

    @staticmethod
    def _pattern(nbytes, salt):
        # The expected byte model, written independently of kv_workload.
        chunks = np.arange(nbytes, dtype=np.int64) >> 8
        return ((chunks * 131 + 7 + 101 * salt) & 0xFF).astype(np.uint8)

    def _painted_destination(self, cfg, dst_tables, src_tables):
        """A destination pool where every dst block holds its src block's pattern."""
        pool = self._pattern(cfg["pool_bytes"], self.DST_SALT)
        src_pool = self._pattern(cfg["pool_bytes"], self.SRC_SALT)
        for region in cfg["regions"]:
            size = region["packed_bytes"]
            for dst, src in zip(dst_tables[region["name"]], src_tables[region["name"]]):
                dst_off = int(dst) * size + region["base"]
                src_off = int(src) * size + region["base"]
                pool[dst_off : dst_off + size] = src_pool[src_off : src_off + size]
        return pool

    def _setup(self):
        cfg = kv_workload.plan_config("dsv4", "fp8", 512, 256)
        dst = kv_workload.block_table(cfg, kv_workload.table_seed(cfg, "local", 67))
        src = kv_workload.block_table(cfg, kv_workload.table_seed(cfg, "remote", 67))
        return cfg, dst, src, self._painted_destination(cfg, dst, src)

    def _verify(self, pool, cfg, dst, src, salt=SRC_SALT):
        return kv_workload.verify_transfer(_read8(pool), cfg, dst, src, src_salt=salt)

    def test_a_faithful_transfer_verifies_across_unaligned_pages(self):
        # dsv4's page sizes are 576 B multiples, never 256 B multiples, so
        # per-layer probes land at any byte alignment and exercise the
        # per-byte expectation model.
        cfg, dst, src, pool = self._setup()
        ok, detail = self._verify(pool, cfg, dst, src)
        self.assertTrue(ok, detail)

    def test_wrong_transfers_fail(self):
        cfg, dst, src, pool = self._setup()
        untouched = self._pattern(cfg["pool_bytes"], self.DST_SALT)
        for name, args in (
            # a transfer that never happened leaves the destination's own
            # repainted pattern, which the source salt tells apart
            ("never happened", (untouched, cfg, dst, src)),
            # dst blocks hold the src pattern, not their own
            ("tables swapped", (pool, cfg, src, dst)),
            # a loopback read of the destination's own pool
            ("wrong source rank", (pool, cfg, dst, src, self.DST_SALT)),
        ):
            with self.subTest(name):
                ok, detail = self._verify(*args)
                self.assertFalse(ok)
                self.assertIn("expected", detail)

    def test_the_fabric_pool_tile_matches_the_verify_model(self):
        # FabricPool (the mnnvl fill path) doubles pattern_tile across the pool;
        # it must agree byte for byte with the verify model, or every mnnvl row
        # fails verify.
        for salt in (0, 1):
            tile = kv_workload.pattern_tile(salt)
            self.assertEqual(tile.nbytes, kv_workload.PATTERN_PERIOD)
            for offset in (0, 8, 256, 1016, kv_workload.PATTERN_PERIOD - 8):
                expected = bytes(kv_workload._chunk_byte(offset + j, salt) for j in range(8))
                self.assertEqual(tile[offset : offset + 8].tobytes(), expected, (salt, offset))
            # periodic: the byte one period on is the same byte
            self.assertEqual(kv_workload._chunk_byte(kv_workload.PATTERN_PERIOD + 300, salt),
                             kv_workload._chunk_byte(300, salt))


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
