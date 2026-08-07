#!/usr/bin/env python3
"""Geometry and correctness math of the KV-transfer workload model.

The paged layout is the contract: layer-major offsets over `[layer][page]`
pools, seed-keyed block tables both ranks derive independently, and an
offset-derived pattern that makes any page's expected contents computable
without knowing which config painted the pool. These tests pin that math with
hand-computed cases; the torch fill path is exercised on metal by the suite
itself (a wrong fill fails every verify row loudly).
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
    def test_mla_bf16_shapes(self):
        cfg = kv_workload.plan_config("mla", "bf16", 512, 16)
        self.assertEqual(cfg["page_bytes"], 16 * 576 * 2)
        self.assertEqual(cfg["pages_req"], 32)
        self.assertEqual(cfg["descs"], 61 * 32)
        self.assertEqual(cfg["req_bytes"], 61 * 32 * cfg["page_bytes"])
        self.assertGreaterEqual(cfg["pool_pages"], 2 * cfg["pages_req"])

    def test_fp8_halves_page_bytes(self):
        bf16 = kv_workload.plan_config("gqa", "bf16", 4096, 64)
        fp8 = kv_workload.plan_config("gqa", "fp8", 4096, 64)
        self.assertEqual(fp8["page_bytes"] * 2, bf16["page_bytes"])
        self.assertEqual(fp8["req_bytes"] * 2, bf16["req_bytes"])
        self.assertEqual(fp8["descs"], bf16["descs"])

    def test_partial_last_page_rounds_up(self):
        cfg = kv_workload.plan_config("mla", "bf16", 100, 64)
        self.assertEqual(cfg["pages_req"], 2)

    def test_page_bytes_must_hold_the_pattern_alignment(self):
        # mla fp8 with a 1-token page is 576 B — not a multiple of 256.
        with self.assertRaises(ValueError):
            kv_workload.plan_config("mla", "fp8", 512, 1)


class Tables(unittest.TestCase):
    def test_deterministic_and_distinct_per_side(self):
        cfg = kv_workload.plan_config("mla", "bf16", 4096, 16)
        local = kv_workload.block_table(cfg, kv_workload.table_seed(cfg, "local"))
        remote = kv_workload.block_table(cfg, kv_workload.table_seed(cfg, "remote"))
        again = kv_workload.block_table(cfg, kv_workload.table_seed(cfg, "local"))
        self.assertTrue((local == again).all())
        self.assertFalse((local == remote).all())
        # a block table is a set of distinct in-range pages (fragmented, never aliased)
        self.assertEqual(len(set(local.tolist())), cfg["pages_req"])
        self.assertTrue((local < cfg["pool_pages"]).all())

    def test_layer_major_offsets(self):
        cfg = dict(layers=2, pool_pages=3, page_bytes=512, descs=4)
        table = np.array([2, 0])
        offsets = kv_workload.page_offsets(cfg, table)
        # layer 0 pages 2,0 then layer 1 pages 2,0 — each layer pool_pages wide
        self.assertEqual(offsets.tolist(), [2 * 512, 0, (3 + 2) * 512, 3 * 512])

    def test_desc_array_carries_base_len_dev(self):
        cfg = dict(layers=1, pool_pages=4, page_bytes=256, descs=2)
        table = np.array([1, 3])
        descs = kv_workload.desc_array(10_000, cfg, table, dev=5)
        self.assertEqual(descs[:, 0].tolist(), [10_000 + 256, 10_000 + 768])
        self.assertEqual(descs[:, 1].tolist(), [256, 256])
        self.assertEqual(descs[:, 2].tolist(), [5, 5])


class Verify(unittest.TestCase):
    def _painted_destination(self, cfg, dst_table, src_table):
        """A destination pool where every dst page holds its src page's pattern."""
        pool = np.zeros(cfg["layers"] * cfg["pool_pages"] * cfg["page_bytes"], dtype=np.uint8)
        for layer in range(cfg["layers"]):
            for dst, src in zip(dst_table, src_table):
                dst_off = (layer * cfg["pool_pages"] + int(dst)) * cfg["page_bytes"]
                src_off = (layer * cfg["pool_pages"] + int(src)) * cfg["page_bytes"]
                pool[dst_off : dst_off + cfg["page_bytes"]] = ((src_off >> 8) * 131 + 7) & 0xFF
        return pool

    def test_a_faithful_transfer_verifies(self):
        cfg = kv_workload.plan_config("mla", "bf16", 512, 16)
        dst = kv_workload.block_table(cfg, kv_workload.table_seed(cfg, "local"))
        src = kv_workload.block_table(cfg, kv_workload.table_seed(cfg, "remote"))
        pool = self._painted_destination(cfg, dst, src)
        ok, detail = kv_workload.verify_transfer(_read8(pool), cfg, dst, src)
        self.assertTrue(ok, detail)

    def test_one_corrupted_page_fails_with_its_coordinates(self):
        cfg = kv_workload.plan_config("mla", "bf16", 512, 16)
        dst = kv_workload.block_table(cfg, kv_workload.table_seed(cfg, "local"))
        src = kv_workload.block_table(cfg, kv_workload.table_seed(cfg, "remote"))
        pool = self._painted_destination(cfg, dst, src)
        pool[:] = 0  # a transfer that never happened
        ok, detail = kv_workload.verify_transfer(_read8(pool), cfg, dst, src)
        self.assertFalse(ok)
        self.assertIn("expected", detail)

    def test_direction_matters(self):
        # Verifying with the tables swapped must fail: dst pages hold src
        # pattern, not their own.
        cfg = kv_workload.plan_config("mla", "bf16", 512, 16)
        dst = kv_workload.block_table(cfg, kv_workload.table_seed(cfg, "local"))
        src = kv_workload.block_table(cfg, kv_workload.table_seed(cfg, "remote"))
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


class Percentiles(unittest.TestCase):
    def test_pcts(self):
        stats = kv_workload.pcts([5.0, 1.0, 3.0, 2.0, 4.0])
        self.assertEqual(stats["p50"], 3.0)
        self.assertEqual(stats["min"], 1.0)
        self.assertEqual(stats["max"], 5.0)
        self.assertEqual(stats["n"], 5)


if __name__ == "__main__":
    unittest.main()
