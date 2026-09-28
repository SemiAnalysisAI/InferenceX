#!/usr/bin/env python3
"""The kv-transfer suite's scheduling, control-plane, grid, and summary contracts.

sweep_matrix emits kv shards only where the registry carries `kv_backends` (and
never perturbs the EP matrix); run_kv's pure helpers (verdict exchange, UCX
selectors, burst timing, grid budgets, registration layout) are exercised on
CPU; summarize renders kv documents in their own table. The argv codec's
round trip lives with the other suites' in test_runtime.CaseArgvContract.
"""
from __future__ import annotations

import argparse
import bisect
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "bench"), str(ROOT / "tests")]

import ep_harness  # noqa: E402
import kv_nixl  # noqa: E402
import kv_workload  # noqa: E402
import run_kv  # noqa: E402
import summarize  # noqa: E402
import sweep_matrix  # noqa: E402
from kv_backend import time_bursts  # noqa: E402
from test_chain import document as ep_document  # noqa: E402


class KVMatrix(unittest.TestCase):
    _BASE = {"scale_up_domain": 8, "scale_up_transport": "nvlink"}
    PLATFORMS = {
        # a fabric list runs the full sweep
        "full": {**_BASE, "launcher": "single-slurm", "product": "full",
                 "kv_backends": {"nixl": ["rdma", "mnnvl"]}},
        # an object restricts it
        "restricted": {**_BASE, "launcher": "mi-amds", "product": "restricted",
                       "kv_backends": {"mooncake": {
                           "fabrics": ["rdma"], "ops": ["push"], "image": "pinned:tag",
                           "device": "rdma{gpu}", "pool_budget": 123}}},
        # no entry, no legs
        "absent": {**_BASE, "launcher": "single-slurm", "product": "absent"},
    }

    def _kv(self):
        sweep = {**sweep_matrix.KV_SWEEP, "scheduling": {
            "default": {"allocation_minutes": 100, "run_timeout": 5000},
            "restricted": {"allocation_minutes": 690, "run_timeout": 39600},
        }}
        with mock.patch.object(sweep_matrix, "PLATFORMS", self.PLATFORMS), \
                mock.patch.object(sweep_matrix, "KV_SWEEP", sweep):
            return sweep_matrix.resolve_matrix(suites="kv-transfer")

    def test_registry_entries_drive_the_shards(self):
        matrix = self._kv()
        shards = {(s["sku"], s["backend"], s["cases"][0]["mode"]): s for s in matrix["include"]}
        self.assertEqual(set(shards), {("full", "nixl", "rdma"), ("full", "nixl", "mnnvl"),
                                       ("restricted", "mooncake", "rdma")})
        self.assertEqual(len(matrix["requested_cases"]),
                         sum(len(s["cases"]) for s in shards.values()))
        for (sku, _, fabric), shard in shards.items():
            with self.subTest(sku=sku, fabric=fabric):
                self.assertEqual((shard["nodes"], shard["gpus_per_node"]), (2, 1))
                self.assertEqual(shard["launcher"], self.PLATFORMS[sku]["launcher"])
                for case in shard["cases"]:
                    self.assertEqual((case["suite"], case["ep"], case["mode"]),
                                     ("kv-transfer", 2, fabric))
                    self.assertEqual(case["case_id"], ep_harness.case_id(sku, case))
        full = shards[("full", "nixl", "rdma")]
        self.assertNotIn("image", full)
        self.assertEqual({(c["ops"], c["kv_device"], "pool_budget" in c) for c in full["cases"]},
                         {("pull push", "", False)})
        restricted = shards[("restricted", "mooncake", "rdma")]
        self.assertEqual(restricted["image"], "pinned:tag")
        self.assertEqual({(c["ops"], c["kv_device"], c["pool_budget"])
                          for c in restricted["cases"]}, {("push", "rdma{gpu}", 123)})
        # scheduling: the default vs a per-pool override; the GitHub job ceiling is
        # max(350, allocation + 30) so it always outlives the allocation
        self.assertEqual((full["allocation_minutes"], full["run_timeout"],
                          full["job_timeout_minutes"]), (100, 5000, 350))
        self.assertEqual((restricted["allocation_minutes"], restricted["run_timeout"],
                          restricted["job_timeout_minutes"]), (690, 39600, 720))

    def test_the_real_registry_keeps_each_guard_inside_its_allocation(self):
        shards = sweep_matrix.resolve_matrix(suites="kv-transfer")["include"]
        self.assertTrue(shards, "registry carries kv_backends but no shard resolved")
        for shard in shards:
            with self.subTest(shard=shard["id"]):
                self.assertLess(shard["run_timeout"], shard["allocation_minutes"] * 60)
                self.assertGreater(shard["job_timeout_minutes"], shard["allocation_minutes"])

    def test_kv_never_perturbs_the_ep_matrix(self):
        ep_only = sweep_matrix.resolve_matrix()
        both = sweep_matrix.resolve_matrix(suites="ep,kv-transfer")
        self.assertEqual(
            ep_only["include"],
            [s for s in both["include"] if s.get("suite") != "kv-transfer"])
        self.assertEqual(
            ep_only["requested_cases"],
            [c for c in both["requested_cases"] if c["case"].get("suite") != "kv-transfer"])


class _StubDist:
    """all_gather_object across a simulated 2-rank pair."""

    def __init__(self, other_value):
        self.other = other_value

    def all_gather_object(self, out, mine):
        out[0], out[1] = mine, self.other


class VerdictExchange(unittest.TestCase):
    """Bulk rows have no verifying side; that path crashed on the metal (gb200
    smoke 22840: StopIteration on both ranks) before this contract existed."""

    def test_every_rank_returns_the_verifying_ranks_verdict(self):
        bad = {"passed": False, "detail": "bad page"}

        def never():
            raise AssertionError("must not verify")

        for name, other, role, side, verify, expected in (
            ("verifying rank", None, "initiator", "initiator", lambda: (False, "bad page"), bad),
            ("other rank", bad, "target", "initiator", never, bad),
            ("no verifying side", None, "initiator", "none", never,
             {"passed": True, "detail": ""}),
        ):
            with self.subTest(name):
                self.assertEqual(
                    run_kv.exchange_verdict(_StubDist(other), role, side, verify), expected)


class UCXSelectors(unittest.TestCase):
    """run_kv pins UCX to the operator's validated RDMA selectors — UCX
    auto-selection is a wrong-fabric trap (b200-nscale's aux quad-port card) —
    while explicit UCX_* values always win, except over a case's own NIC pin."""

    # (environment, kv_device pin) -> the UCX variables export_ucx_selectors leaves.
    CASES = (
        # registry selectors map to UCX, ports default to 1 and pass through
        ({"COLLX_RDMA_DEVICES": "mlx5_0,mlx5_10", "COLLX_IB_GID_INDEX": "3"}, "",
         {"UCX_NET_DEVICES": "mlx5_0:1,mlx5_10:1", "UCX_IB_GID_INDEX": "3"}),
        ({"COLLX_RDMA_DEVICES": "mlx5_18:1,mlx5_19"}, "",
         {"UCX_NET_DEVICES": "mlx5_18:1,mlx5_19:1"}),
        # explicit UCX env wins over the inventory
        ({"COLLX_RDMA_DEVICES": "mlx5_0", "UCX_NET_DEVICES": "rdma0:1",
          "COLLX_IB_GID_INDEX": "3", "UCX_IB_GID_INDEX": "1"}, "",
         {"UCX_NET_DEVICES": "rdma0:1", "UCX_IB_GID_INDEX": "1"}),
        # a case pin narrows the inventory and overrides a host-inherited blanket value
        # (forwarded by srun --export=ALL from /etc/environment), which would swallow it
        ({"COLLX_RDMA_DEVICES": "mlx5_0,mlx5_1"}, "mlx5_0", {"UCX_NET_DEVICES": "mlx5_0:1"}),
        ({"UCX_NET_DEVICES": "rdma0:1"}, "mlx5_0", {"UCX_NET_DEVICES": "mlx5_0:1"}),
        # A positive UCX_TLS list without cuda (a cluster-wide UCX_TLS=rc) closes the cuda
        # mds and NIXL VRAM registration fails; extending it segfaults rkey resolution on the
        # first GET, so it is dropped. Lists already covering cuda stay untouched.
        ({"UCX_TLS": "rc"}, "", {}),
        ({"UCX_TLS": "^tcp"}, "", {"UCX_TLS": "^tcp"}),
        ({"UCX_TLS": "rc,cuda_copy"}, "", {"UCX_TLS": "rc,cuda_copy"}),
        ({"UCX_TLS": "all"}, "", {"UCX_TLS": "all"}),
        ({}, "", {}),
    )

    def test_selectors(self):
        for env, device, expected in self.CASES:
            with self.subTest(env=env, device=device):
                env = dict(env)
                run_kv.export_ucx_selectors(env, device=device)
                self.assertEqual({k: v for k, v in env.items() if k.startswith("UCX_")}, expected)


class BurstTiming(unittest.TestCase):
    def test_a_burst_posts_every_request_before_waiting_on_any(self):
        order = []
        pairs = [(lambda i=i: order.append(("post", i)),
                  lambda i=i: order.append(("wait", i))) for i in range(3)]
        burst_ms, request_ms = time_bursts(pairs, warmup=1, reps=2)
        self.assertEqual(order[:6], [("post", 0), ("post", 1), ("post", 2),
                                     ("wait", 0), ("wait", 1), ("wait", 2)])
        # warmups dropped; one completion mark per request per kept rep, offsets
        # from the burst start (never decreasing), and the burst is its last mark
        self.assertEqual((len(burst_ms), len(request_ms)), (2, 2 * 3))
        for rep, burst in enumerate(burst_ms):
            marks = request_ms[3 * rep : 3 * rep + 3]
            self.assertEqual(marks, sorted(marks))
            self.assertEqual(burst, marks[-1])


def _grid_args(**overrides):
    base = dict(workload_name="kv-dsv4", precision="fp8",
                isl_ladder="8192 32768 131072 524288", page_tokens="256",
                batch_sizes="1 2 4 8 16 32 64", pool_slack=2.0,
                pool_budget=run_kv.POOL_BUDGET, seed=67)
    base.update(overrides)
    return argparse.Namespace(**base)


class KVGrid(unittest.TestCase):
    def test_the_packed_grid_sheds_only_where_the_pool_budget_bites(self):
        # Packed block-major geometry: a 512k-ISL block-256 request is 6,146
        # descriptors, so no batch on this ladder nears DESC_BUDGET. Only the
        # 512k point sheds, and via the pool budget: its batch-32 pool plans
        # ~118 GB against the 64 GiB budget, batch 16 fits at ~59 GB.
        points, isls, batches = run_kv._grid(_grid_args())
        self.assertEqual((isls, batches),
                         ([8192, 32768, 131072, 524288], [1, 2, 4, 8, 16, 32, 64]))
        allowed = {cfg["isl"]: allowed for cfg, allowed in points}
        self.assertEqual(allowed, {8192: batches, 32768: batches, 131072: batches,
                                   524288: [1, 2, 4, 8, 16]})
        for cfg, batch_list in points:
            # one descriptor per packed block: 3 full-ISL groups + 2 window blocks
            self.assertEqual(cfg["descs"], 3 * -(-cfg["isl"] // 256) + 2)
            self.assertLessEqual(cfg["pool_bytes"], run_kv.POOL_BUDGET)

    def test_descriptor_budget_sheds_batches_but_keeps_a_chartable_ladder(self):
        # DESC_BUDGET stays as the fail-closed guard for future presets whose
        # bursts are descriptor-bound. Pin it to 4 requests' descriptors at
        # the largest ISL: batches above the per-point allowance shed, but the
        # LADDER_FLOOR smallest batches always survive so every point keeps a
        # chartable batch ladder.
        probe = kv_workload.plan_config("dsv4", "fp8", 524288, 256)
        with mock.patch.object(run_kv, "DESC_BUDGET", 4 * probe["descs"]):
            points, _isls, _batches = run_kv._grid(_grid_args())
        allowed = {cfg["isl"]: allowed for cfg, allowed in points}
        self.assertEqual(allowed[8192], [1, 2, 4, 8, 16, 32, 64])   # 98 descs/req
        self.assertEqual(allowed[32768], [1, 2, 4, 8, 16, 32])      # 386
        self.assertEqual(allowed[131072], [1, 2, 4, 8, 16])         # 1538, floor
        self.assertEqual(allowed[524288], [1, 2, 4, 8, 16])         # 6146, floor

    def test_pool_budget_sheds_largest_batches_even_below_the_ladder_floor(self):
        # The budget is a hard memory limit, so it sheds batches the descriptor
        # floor keeps: pinned to the 512k point's batch-1 pool, only [1] remains.
        for isl, batches, fit_batch, expected in ((32768, "1 4 16", 4, [1, 4]),
                                                  (524288, "1 2 4 8 16 32 64", 1, [1])):
            with self.subTest(isl=isl):
                args = _grid_args(isl_ladder=str(isl), batch_sizes=batches)
                args.pool_budget = kv_workload.plan_config(
                    "dsv4", "fp8", isl, 256, 2.0, batch_max=fit_batch)["pool_bytes"]
                points, _isls, _batches = run_kv._grid(args)
                self.assertEqual(points[0][1], expected)
                self.assertLessEqual(points[0][0]["pool_bytes"], args.pool_budget)


class RegistrationChunking(unittest.TestCase):
    # The NIXL adapter registers an oversized pool in pieces. The pieces must
    # never cut through a descriptor of ANY planned config, which _harmonize
    # guarantees by giving every config one shared region layout.

    def test_harmonize_makes_region_bases_config_invariant(self):
        points, _isls, _batches = run_kv._grid(_grid_args())
        layout = run_kv._harmonize(points)
        total = sum(nbytes for _, _, nbytes in layout)
        running = 0
        for base, _packed, nbytes in layout:
            self.assertEqual(base, running)
            running += nbytes
        for cfg, _ in points:
            self.assertEqual(cfg["pool_bytes"], total)
            for region, (base, packed, nbytes) in zip(cfg["regions"], layout):
                self.assertEqual((region["base"], region["packed_bytes"], region["pool_blocks"]),
                                 (base, packed, nbytes // packed))
                self.assertLessEqual(region["blocks_req"], region["pool_blocks"])

    def test_spans_tile_the_pool_and_no_descriptor_straddles_a_cut(self):
        # A tiny cap on a small grid forces many cuts; every block any config
        # can address must land whole inside one registered piece.
        cap = 1 << 24
        points, _isls, _batches = run_kv._grid(
            _grid_args(isl_ladder="2048 8192", batch_sizes="1 4"))
        layout = run_kv._harmonize(points)
        total = sum(nbytes for _, _, nbytes in layout)
        spans = kv_nixl.reg_spans(total, layout, cap=cap)
        self.assertGreater(len(spans), len(layout))
        # exact in-order coverage, no gap, no overlap, each piece within the cap
        self.assertEqual(spans[0][0], 0)
        for (a_off, a_len), (b_off, _) in zip(spans, spans[1:]):
            self.assertEqual(a_off + a_len, b_off)
        self.assertEqual(sum(length for _, length in spans), total)
        for off, length in spans:
            packed = next(entry for entry in reversed(layout) if entry[0] <= off)[1]
            self.assertLessEqual(length, max(cap, packed))
        starts = [off for off, _ in spans]
        straddles = []
        for cfg, _ in points:
            for region in cfg["regions"]:
                packed = region["packed_bytes"]
                for block in range(region["pool_blocks"]):
                    off = region["base"] + block * packed
                    s_off, s_len = spans[bisect.bisect_right(starts, off) - 1]
                    if off + packed > s_off + s_len:
                        straddles.append((region["name"], block))
        self.assertEqual(straddles, [])

    def test_without_a_layout_the_pool_registers_whole(self):
        for layout in (None, []):
            self.assertEqual(kv_nixl.reg_spans(123456, layout), [(0, 123456)])


def _kv_document(status="success"):
    def row(kind, page, op, gbps, p50, batch=1):
        return {"kind": kind, "isl": 32768, "page_tokens": page, "op": op, "batch": batch,
                "latency_ms": {"p50": p50}, "gbps_p50": gbps}

    return {
        "version": 1,
        "identity": {"case_factors": {"sku": "b200-nscale", "case": {
            "suite": "kv-transfer", "backend": "nixl", "workload": "kv-dsv4",
            "mode": "rdma", "precision": "fp8"}}},
        "measurement": {"rows": [
            row("paged", 256, "pull", 43.4, 53.1),
            row("paged", 256, "pull", 96.2, 21.4, batch=16),
            # a smaller measured block must lose to the production block size
            row("paged", 128, "pull", 12.4, 185.2),
            row("bulk", None, "pull", 48.3, 47.7),
            row("paged", 256, "push", 48.4, 47.6),
        ]},
        "outcome": {"status": status, "reasons": []},
    }


class KVSummary(unittest.TestCase):
    def test_kv_documents_render_their_own_table(self):
        text = summarize.render([_kv_document()])
        self.assertIn("KV-transfer results", text)
        self.assertNotIn("EP results", text)
        self.assertIn("| pull | 43.4 | 96.2 | 48.3 | 53.1 |", text)

    def test_a_push_only_document_reads_its_push_lane(self):
        doc = _kv_document()
        doc["measurement"]["rows"] = [
            row for row in doc["measurement"]["rows"] if row["op"] == "push"
        ]
        text = summarize.render([doc])
        self.assertIn("| push | 48.4 |", text)
        self.assertNotIn("INVALID", text)

    def test_kv_invalid_counts_in_the_banner(self):
        self.assertIn("INVALID", summarize.render([_kv_document(status="invalid")]))

    def test_mixed_documents_render_one_table_per_suite(self):
        text = summarize.render([ep_document(with_period=True), _kv_document()])
        self.assertIn("EP results", text)
        self.assertIn("KV-transfer results", text)
        self.assertLess(text.index("EP results"), text.index("KV-transfer results"))

    def test_an_empty_render_has_only_the_ep_table(self):
        text = summarize.render([])
        self.assertIn("EP results", text)
        self.assertNotIn("KV-transfer results", text)


if __name__ == "__main__":
    unittest.main()
