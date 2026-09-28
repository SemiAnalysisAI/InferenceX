#!/usr/bin/env python3
"""Geometry and correctness math of the KV-transfer workload model.

The packed block-major layout is the contract: per cache-group region, one
contiguous descriptor covers all the group's layers for one physical block
(vLLM's packed DSV4 NIXL shape), block tables are seed-keyed permutations both
ranks derive independently (batched requests slicing disjoint ranges of one
permutation), and an offset-derived pattern makes any byte's expected value
computable from its offset alone. These tests pin that math with hand-computed
cases validated against vLLM commit 32ad1400d7 (state content 584 B, page
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
        self.assertEqual(
            (regions["swa"]["layers"], regions["swa"]["block_tokens"],
             regions["swa"]["blocks_req"]), (61, 64, 2))
        self.assertEqual(cfg["descs"], 2 + 2 + 2 + 2)
        self.assertEqual(cfg["req_bytes"],
                         2 * (30 * 37_440 + 30 * 8_640 + 31 * 1_728 + 61 * 37_440))
        # regions tile one contiguous pool
        self.assertEqual(cfg["pool_bytes"],
                         sum(r["pool_blocks"] * r["packed_bytes"]
                             for r in cfg["regions"]))

    def test_alignment_pads_the_page_not_each_state(self):
        # 64 states * 584 B = 37,376 -> padded once per page to 37,440. The
        # old per-entry 576 B model would give 64 * 576 = 36,864 — vLLM pads
        # at page granularity, not per state.
        cfg = kv_workload.plan_config("dsv4", "fp8", 512, 256)
        c4a = {r["name"]: r for r in cfg["regions"]}["c4a"]
        self.assertEqual(c4a["page_bytes"], 37_440)
        self.assertNotEqual(c4a["page_bytes"], 64 * 576)

    def test_swa_shares_the_c4a_page_size(self):
        # Both block types live in one physical tensor: a 64-token window
        # block (1 token/state) and a 256-token C4A block (4 tokens/state)
        # are the same 64 states -> byte-identical pages.
        cfg = kv_workload.plan_config("dsv4", "fp8", 512, 256)
        regions = {r["name"]: r for r in cfg["regions"]}
        self.assertEqual(regions["swa"]["page_bytes"], regions["c4a"]["page_bytes"])

    def test_one_descriptor_per_block_at_the_big_isl(self):
        # 512k tokens at block 256: 2048 blocks per non-window group + 2
        # window blocks = 6,146 descriptors per request — the packed shape
        # vLLM's connector asserts, not a per-(layer, page) explosion.
        cfg = kv_workload.plan_config("dsv4", "fp8", 524_288, 256)
        self.assertEqual(cfg["descs"], 2048 * 3 + 2)

    def test_dsv4_window_caps_at_128_tokens(self):
        small = kv_workload.plan_config("dsv4", "fp8", 64, 256)
        large = kv_workload.plan_config("dsv4", "fp8", 32_768, 256)
        window = {r["name"]: r for r in large["regions"]}["swa"]
        self.assertEqual(window["blocks_req"], 2)  # 128 tokens / 64 per block
        self.assertEqual({r["name"]: r for r in small["regions"]}["swa"]["blocks_req"],
                         1)  # min(isl, 128) = 64 tokens

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
    def test_deterministic_and_distinct_per_side(self):
        cfg = kv_workload.plan_config("dsv4", "fp8", 4096, 256)
        local = kv_workload.block_table(cfg, kv_workload.table_seed(cfg, "local"))
        remote = kv_workload.block_table(cfg, kv_workload.table_seed(cfg, "remote"))
        again = kv_workload.block_table(cfg, kv_workload.table_seed(cfg, "local"))
        for region in cfg["regions"]:
            name, blocks_req = region["name"], region["blocks_req"]
            self.assertTrue((local[name] == again[name]).all())
            self.assertFalse((local[name] == remote[name]).all())
            # distinct in-range blocks (fragmented, never aliased)
            self.assertEqual(len(set(local[name].tolist())), blocks_req)
            self.assertTrue((local[name] < region["pool_blocks"]).all())

    def test_batched_requests_slice_disjoint_blocks(self):
        cfg = kv_workload.plan_config("dsv4", "fp8", 512, 256, batch_max=4)
        seed = kv_workload.table_seed(cfg, "local")
        tables = [kv_workload.block_table(cfg, seed, request=r) for r in range(4)]
        for region in cfg["regions"]:
            blocks = [t[region["name"]].tolist() for t in tables]
            union = set().union(*map(set, blocks))
            self.assertEqual(len(union), 4 * region["blocks_req"])

    def test_a_request_beyond_the_pool_fails_closed(self):
        cfg = kv_workload.plan_config("dsv4", "fp8", 512, 256)  # slack for ~2 requests
        with self.assertRaises(ValueError):
            kv_workload.block_table(cfg, 1, request=8)

    def test_block_major_offsets(self):
        # One offset per packed block: block b sits at b * packed_bytes.
        cfg = dict(regions=[dict(name="kv", packed_bytes=512, blocks_req=2,
                                 pool_blocks=3, base=0)], descs=2)
        offsets = kv_workload.page_offsets(cfg, {"kv": np.array([2, 0])})
        self.assertEqual(offsets.tolist(), [2 * 512, 0])

    def test_second_region_offsets_start_at_its_base(self):
        cfg = dict(regions=[
            dict(name="a", packed_bytes=256, blocks_req=1, pool_blocks=2, base=0),
            dict(name="b", packed_bytes=128, blocks_req=1, pool_blocks=2, base=512),
        ], descs=2)
        offsets = kv_workload.page_offsets(cfg, {"a": np.array([1]), "b": np.array([1])})
        self.assertEqual(offsets.tolist(), [256, 512 + 128])

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
    def _painted_destination(self, cfg, dst_tables, src_tables):
        """A destination pool where every dst block holds its src block's pattern."""
        pool = np.zeros(cfg["pool_bytes"], dtype=np.uint8)
        for region in cfg["regions"]:
            size = region["packed_bytes"]
            for dst, src in zip(dst_tables[region["name"]], src_tables[region["name"]]):
                dst_off = int(dst) * size + region["base"]
                src_off = int(src) * size + region["base"]
                src_bytes = src_off + np.arange(size, dtype=np.int64)
                pool[dst_off : dst_off + size] = ((src_bytes >> 8) * 131 + 7) & 0xFF
        return pool

    def _tables(self, cfg):
        dst = kv_workload.block_table(cfg, kv_workload.table_seed(cfg, "local"))
        src = kv_workload.block_table(cfg, kv_workload.table_seed(cfg, "remote"))
        return dst, src

    def test_a_faithful_transfer_verifies_across_unaligned_pages(self):
        # dsv4's page sizes are 576 B multiples, never 256 B multiples, so
        # per-layer probes land at any byte alignment and exercise the
        # per-byte expectation model.
        cfg = kv_workload.plan_config("dsv4", "fp8", 512, 256)
        dst, src = self._tables(cfg)
        pool = self._painted_destination(cfg, dst, src)
        ok, detail = kv_workload.verify_transfer(_read8(pool), cfg, dst, src)
        self.assertTrue(ok, detail)

    def test_one_missing_transfer_fails_with_its_coordinates(self):
        cfg = kv_workload.plan_config("dsv4", "fp8", 512, 256)
        dst, src = self._tables(cfg)
        pool = self._painted_destination(cfg, dst, src)
        pool[:] = 0  # a transfer that never happened
        ok, detail = kv_workload.verify_transfer(_read8(pool), cfg, dst, src)
        self.assertFalse(ok)
        self.assertIn("expected", detail)

    def test_direction_matters(self):
        # Verifying with the tables swapped must fail: dst blocks hold src
        # pattern, not their own.
        cfg = kv_workload.plan_config("dsv4", "fp8", 512, 256)
        dst, src = self._tables(cfg)
        pool = self._painted_destination(cfg, dst, src)
        ok, _ = kv_workload.verify_transfer(_read8(pool), cfg, src, dst)
        self.assertFalse(ok)

    def test_fabric_pool_pattern_matches_the_verify_model(self):
        # kv_pool's host-built pattern (the mnnvl fill path) and the verify
        # model must agree byte for byte, or every mnnvl row fails verify.
        import kv_pool

        pattern = kv_pool._pattern(1024)
        for offset in (0, 8, 256, 512, 1016):
            expected = kv_workload._chunk_byte(offset)
            self.assertTrue((pattern[offset : offset + 8] == expected).all(), offset)


class SweepConfigConsistency(unittest.TestCase):
    def test_kv_sweep_precisions_match_the_workload_model(self):
        # sweep_matrix schedules from the JSON map (it must stay stdlib-only);
        # the workload model owns the truth and plan_config fail-closes on a
        # mismatch at runtime. This pins the two together at PR time.
        import json

        sweep = json.loads((ROOT / "configs" / "kv_sweep.json").read_text())
        for workload, precisions in sweep["workloads"].items():
            preset = kv_workload.PRESETS[workload.removeprefix("kv-")]
            self.assertEqual(tuple(precisions), preset["precisions"], workload)

    def test_kv_sweep_block_sizes_are_plannable(self):
        # A sweep block size the model rejects (splitting an HCA state) would
        # kill every kv leg at the first grid point.
        import json

        sweep = json.loads((ROOT / "configs" / "kv_sweep.json").read_text())
        for workload, precisions in sweep["workloads"].items():
            for block in sweep["page_tokens"]:
                kv_workload.plan_config(workload.removeprefix("kv-"),
                                        precisions[0], 512, block)


class Percentiles(unittest.TestCase):
    def test_pcts(self):
        stats = kv_workload.pcts([5.0, 1.0, 3.0, 2.0, 4.0])
        self.assertEqual(stats["p50"], 3.0)
        self.assertEqual(stats["min"], 1.0)
        self.assertEqual(stats["max"], 5.0)
        self.assertEqual(stats["n"], 5)


if __name__ == "__main__":
    unittest.main()
