#!/usr/bin/env python3
"""Geometry and correctness math of the KV-transfer workload model.

The paged layout is the contract: per-region layer-major offsets over
`[layer][page]` pools, seed-keyed block tables both ranks derive independently
(batched requests slicing disjoint ranges of one permutation), and an
offset-derived pattern that makes any byte's expected value computable from its
offset alone. These tests pin that math with hand-computed cases; the torch
fill path is exercised on metal by the suite itself (a wrong fill fails every
verify row loudly).
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
        # isl=512, page=16 tokens. CSA compresses 4 tokens/entry into vLLM's
        # 576 B slot -> 4 entries/page; its indexer keeps 132 B/entry; HCA
        # compresses 128 tokens/entry (a 16-token page holds one entry); the
        # 128-token sliding window is uncompressed 576 B slots on all 61 layers.
        cfg = kv_workload.plan_config("dsv4", "fp8", 512, 16)
        regions = {r["name"]: r for r in cfg["regions"]}
        self.assertEqual([r["name"] for r in cfg["regions"]],
                         ["csa", "csa-idx", "hca", "window"])
        self.assertEqual((regions["csa"]["layers"], regions["csa"]["page_bytes"],
                          regions["csa"]["pages_req"]), (30, 4 * 576, 32))
        self.assertEqual((regions["csa-idx"]["layers"], regions["csa-idx"]["page_bytes"],
                          regions["csa-idx"]["pages_req"]), (30, 4 * 132, 32))
        self.assertEqual((regions["hca"]["layers"], regions["hca"]["page_bytes"],
                          regions["hca"]["pages_req"]), (31, 576, 4))
        self.assertEqual((regions["window"]["layers"], regions["window"]["page_bytes"],
                          regions["window"]["pages_req"]), (61, 16 * 576, 8))
        self.assertEqual(cfg["descs"], 30 * 32 + 30 * 32 + 31 * 4 + 61 * 8)
        self.assertEqual(cfg["req_bytes"],
                         30 * 32 * 2304 + 30 * 32 * 528 + 31 * 4 * 576 + 61 * 8 * 9216)
        # regions tile one contiguous pool
        self.assertEqual(cfg["pool_bytes"],
                         sum(r["layers"] * r["pool_pages"] * r["page_bytes"]
                             for r in cfg["regions"]))

    def test_dsv4_window_caps_at_128_tokens(self):
        small = kv_workload.plan_config("dsv4", "fp8", 64, 16)
        large = kv_workload.plan_config("dsv4", "fp8", 32768, 16)
        window = {r["name"]: r for r in large["regions"]}["window"]
        self.assertEqual(window["pages_req"], 8)  # 128 tokens / 16 per page
        self.assertEqual({r["name"]: r for r in small["regions"]}["window"]["pages_req"],
                         4)  # min(isl, 128) = 64 tokens

    def test_dsv4_precision_is_architectural(self):
        with self.assertRaises(ValueError):
            kv_workload.plan_config("dsv4", "bf16", 512, 16)

    def test_partial_last_page_rounds_up(self):
        # 100 tokens at 64/page: CSA holds 25 entries (16/page) -> 2 pages.
        cfg = kv_workload.plan_config("dsv4", "fp8", 100, 64)
        self.assertEqual(cfg["regions"][0]["pages_req"], 2)

    def test_batch_max_grows_the_pool_for_disjoint_requests(self):
        cfg = kv_workload.plan_config("dsv4", "fp8", 512, 16, batch_max=16)
        for region in cfg["regions"]:
            self.assertGreaterEqual(region["pool_pages"], 16 * region["pages_req"])


class Tables(unittest.TestCase):
    def test_deterministic_and_distinct_per_side(self):
        cfg = kv_workload.plan_config("dsv4", "fp8", 4096, 16)
        local = kv_workload.block_table(cfg, kv_workload.table_seed(cfg, "local"))
        remote = kv_workload.block_table(cfg, kv_workload.table_seed(cfg, "remote"))
        again = kv_workload.block_table(cfg, kv_workload.table_seed(cfg, "local"))
        for region in cfg["regions"]:
            name, pages_req = region["name"], region["pages_req"]
            self.assertTrue((local[name] == again[name]).all())
            self.assertFalse((local[name] == remote[name]).all())
            # distinct in-range pages (fragmented, never aliased)
            self.assertEqual(len(set(local[name].tolist())), pages_req)
            self.assertTrue((local[name] < region["pool_pages"]).all())

    def test_batched_requests_slice_disjoint_pages(self):
        cfg = kv_workload.plan_config("dsv4", "fp8", 512, 16, batch_max=4)
        seed = kv_workload.table_seed(cfg, "local")
        tables = [kv_workload.block_table(cfg, seed, request=r) for r in range(4)]
        for region in cfg["regions"]:
            pages = [t[region["name"]].tolist() for t in tables]
            union = set().union(*map(set, pages))
            self.assertEqual(len(union), 4 * region["pages_req"])

    def test_a_request_beyond_the_pool_fails_closed(self):
        cfg = kv_workload.plan_config("dsv4", "fp8", 512, 16)  # slack for ~2 requests
        with self.assertRaises(ValueError):
            kv_workload.block_table(cfg, 1, request=8)

    def test_layer_major_offsets(self):
        cfg = dict(regions=[dict(name="kv", layers=2, pool_pages=3, page_bytes=512,
                                 pages_req=2, base=0)], descs=4)
        offsets = kv_workload.page_offsets(cfg, {"kv": np.array([2, 0])})
        # layer 0 pages 2,0 then layer 1 pages 2,0 — each layer pool_pages wide
        self.assertEqual(offsets.tolist(), [2 * 512, 0, (3 + 2) * 512, 3 * 512])

    def test_second_region_offsets_start_at_its_base(self):
        cfg = dict(regions=[
            dict(name="a", layers=1, pool_pages=2, page_bytes=256, pages_req=1, base=0),
            dict(name="b", layers=1, pool_pages=2, page_bytes=128, pages_req=1, base=512),
        ], descs=2)
        offsets = kv_workload.page_offsets(cfg, {"a": np.array([1]), "b": np.array([1])})
        self.assertEqual(offsets.tolist(), [256, 512 + 128])

    def test_desc_array_carries_per_region_sizes(self):
        cfg = dict(regions=[
            dict(name="a", layers=1, pool_pages=4, page_bytes=256, pages_req=2, base=0),
            dict(name="b", layers=1, pool_pages=4, page_bytes=132, pages_req=1, base=1024),
        ], descs=3)
        tables = {"a": np.array([1, 3]), "b": np.array([2])}
        descs = kv_workload.desc_array(10_000, cfg, tables, dev=5)
        self.assertEqual(descs[:, 0].tolist(),
                         [10_000 + 256, 10_000 + 768, 10_000 + 1024 + 264])
        self.assertEqual(descs[:, 1].tolist(), [256, 256, 132])
        self.assertEqual(descs[:, 2].tolist(), [5, 5, 5])


class Verify(unittest.TestCase):
    def _painted_destination(self, cfg, dst_tables, src_tables):
        """A destination pool where every dst page holds its src page's pattern."""
        pool = np.zeros(cfg["pool_bytes"], dtype=np.uint8)
        for region in cfg["regions"]:
            size = region["page_bytes"]
            for layer in range(region["layers"]):
                for dst, src in zip(dst_tables[region["name"]], src_tables[region["name"]]):
                    dst_off = (layer * region["pool_pages"] + int(dst)) * size + region["base"]
                    src_off = (layer * region["pool_pages"] + int(src)) * size + region["base"]
                    src_bytes = src_off + np.arange(size, dtype=np.int64)
                    pool[dst_off : dst_off + size] = ((src_bytes >> 8) * 131 + 7) & 0xFF
        return pool

    def _tables(self, cfg):
        dst = kv_workload.block_table(cfg, kv_workload.table_seed(cfg, "local"))
        src = kv_workload.block_table(cfg, kv_workload.table_seed(cfg, "remote"))
        return dst, src

    def test_a_faithful_transfer_verifies_across_unaligned_regions(self):
        # dsv4 includes the 132 B-entry indexer region: page starts land at any
        # byte alignment, so this exercises the per-byte expectation model.
        cfg = kv_workload.plan_config("dsv4", "fp8", 512, 16)
        dst, src = self._tables(cfg)
        pool = self._painted_destination(cfg, dst, src)
        ok, detail = kv_workload.verify_transfer(_read8(pool), cfg, dst, src)
        self.assertTrue(ok, detail)

    def test_one_missing_transfer_fails_with_its_coordinates(self):
        cfg = kv_workload.plan_config("dsv4", "fp8", 512, 16)
        dst, src = self._tables(cfg)
        pool = self._painted_destination(cfg, dst, src)
        pool[:] = 0  # a transfer that never happened
        ok, detail = kv_workload.verify_transfer(_read8(pool), cfg, dst, src)
        self.assertFalse(ok)
        self.assertIn("expected", detail)

    def test_direction_matters(self):
        # Verifying with the tables swapped must fail: dst pages hold src
        # pattern, not their own.
        cfg = kv_workload.plan_config("dsv4", "fp8", 512, 16)
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


class Percentiles(unittest.TestCase):
    def test_pcts(self):
        stats = kv_workload.pcts([5.0, 1.0, 3.0, 2.0, 4.0])
        self.assertEqual(stats["p50"], 3.0)
        self.assertEqual(stats["min"], 1.0)
        self.assertEqual(stats["max"], 5.0)
        self.assertEqual(stats["n"], 5)


if __name__ == "__main__":
    unittest.main()
