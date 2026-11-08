#!/usr/bin/env python3
"""kv-transfer scheduling, run_kv's control-plane helpers, grid and summary table."""
from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "bench"), str(ROOT / "tests")]

import kv_workload  # noqa: E402
import run_kv  # noqa: E402
import summarize  # noqa: E402
import sweep_matrix  # noqa: E402
from test_chain import document as ep_document  # noqa: E402


class KVMatrix(unittest.TestCase):
    _BASE = {"scale_up_domain": 8, "scale_up_transport": "nvlink"}
    PLATFORMS = {
        "full": {**_BASE, "launcher": "single-slurm", "product": "full",
                 "kv_backends": {"nixl": ["rdma", "mnnvl"]}},
        "push-only": {**_BASE, "launcher": "mi-amds", "product": "push-only",
                      "kv_backends": {"mooncake": ["rdma"]}},
        # no entry, no legs
        "absent": {**_BASE, "launcher": "single-slurm", "product": "absent"},
    }

    def _kv(self):
        sweep = {**sweep_matrix.KV_SWEEP, "allocation_minutes": 100,
                 "ops": {"nixl": ["pull", "push"], "mooncake": ["push"]}}
        with mock.patch.object(sweep_matrix, "PLATFORMS", self.PLATFORMS), \
                mock.patch.object(sweep_matrix, "KV_SWEEP", sweep):
            return sweep_matrix.resolve_matrix(suites="kv-transfer")

    def test_registry_entries_drive_the_shards(self):
        matrix = self._kv()
        shards = {(s["sku"], s["backend"], s["cases"][0]["mode"]): s for s in matrix["include"]}
        self.assertEqual(set(shards), {("full", "nixl", "rdma"), ("full", "nixl", "mnnvl"),
                                       ("push-only", "mooncake", "rdma")})
        self.assertEqual(len(matrix["requested_cases"]),
                         sum(len(s["cases"]) for s in shards.values()))
        for (sku, _, fabric), shard in shards.items():
            with self.subTest(sku=sku, fabric=fabric):
                self.assertEqual((shard["nodes"], shard["gpus_per_node"]), (2, 1))
                self.assertEqual(shard["launcher"], self.PLATFORMS[sku]["launcher"])
                for case in shard["cases"]:
                    self.assertEqual((case["suite"], case["ep"], case["mode"]),
                                     ("kv-transfer", 2, fabric))
                    # the app labels a GB rdma leg by its scale-out transport
                    self.assertEqual(case["scale_out_transport"],
                                     "rdma" if fabric == "rdma" else None)
        # each library runs the directions the sweep config gives it
        self.assertEqual(shards[("full", "nixl", "rdma")]["cases"][0]["ops"], "pull push")
        self.assertEqual(shards[("push-only", "mooncake", "rdma")]["cases"][0]["ops"], "push")
        # run.sh reads the allocation (COLLX_TIME) and exports the fabric off the shard row
        for shard in shards.values():
            self.assertEqual(shard["allocation_minutes"], 100)
            self.assertEqual(shard["fabric"], shard["cases"][0]["mode"])

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

    def __init__(self, other_value, rank):
        self.other, self.rank = other_value, rank

    def all_gather_object(self, out, mine):
        out[self.rank], out[1 - self.rank] = mine, self.other


class VerdictExchange(unittest.TestCase):
    def test_every_rank_returns_the_verifying_ranks_verdict(self):
        bad = {"passed": False, "detail": "bad page"}

        def never():
            raise AssertionError("must not verify")

        def raises():
            raise RuntimeError("boom")

        good = {"passed": True, "detail": ""}
        failed = {"passed": False, "detail": "transfer raised boom"}
        for name, other, role, side, verify, error, expected in (
            ("pull: the initiator verifies", None, "initiator", "initiator",
             lambda: (False, "bad page"), "", bad),
            ("pull: the target reads it", bad, "target", "initiator", never, "", bad),
            ("push: the target verifies", None, "target", "target", lambda: (True, ""), "", good),
            ("push: the initiator reads it", good, "initiator", "target", never, "", good),
            # a rank's own failure wins over the other rank's passing check
            ("initiator transfer failed", good, "initiator", "target", never,
             "transfer raised boom", failed),
            ("target reads the failure", failed, "target", "target", lambda: (True, ""), "",
             failed),
            ("a raising check fails the row", None, "target", "target", raises, "",
             {"passed": False, "detail": "verify raised RuntimeError('boom')"}),
        ):
            with self.subTest(name):
                dist = _StubDist(other, 0 if role == "target" else 1)
                self.assertEqual(
                    run_kv.exchange_verdict(dist, role, side, verify, error), expected)


class UCXSelectors(unittest.TestCase):
    """run_kv pins UCX to the operator's validated RDMA selectors (auto-selection
    is a wrong-fabric trap on b200-nscale's aux quad-port card); UCX_* values the
    environment already sets win."""

    CASES = (
        # registry selectors map to UCX, ports default to 1 and pass through
        ({"COLLX_RDMA_DEVICES": "mlx5_0,mlx5_10", "COLLX_IB_GID_INDEX": "3"},
         {"UCX_NET_DEVICES": "mlx5_0:1,mlx5_10:1", "UCX_IB_GID_INDEX": "3"}),
        ({"COLLX_RDMA_DEVICES": "mlx5_18:1,mlx5_19"}, {"UCX_NET_DEVICES": "mlx5_18:1,mlx5_19:1"}),
        # explicit UCX env wins over the inventory
        ({"COLLX_RDMA_DEVICES": "mlx5_0", "UCX_NET_DEVICES": "rdma0:1",
          "COLLX_IB_GID_INDEX": "3", "UCX_IB_GID_INDEX": "1"},
         {"UCX_NET_DEVICES": "rdma0:1", "UCX_IB_GID_INDEX": "1"}),
        ({}, {}),
    )

    def test_selectors(self):
        for env, expected in self.CASES:
            with self.subTest(env=env):
                env = dict(env)
                run_kv.export_ucx_selectors(env)
                self.assertEqual({k: v for k, v in env.items() if k.startswith("UCX_")}, expected)


class BurstTiming(unittest.TestCase):
    def test_a_burst_posts_every_request_before_waiting_on_any(self):
        order, built, settled = [], [], []

        def build(rep):
            built.append(rep)
            return [(lambda i=i: order.append(("post", i)),
                     lambda i=i: order.append(("wait", i))) for i in range(3)]

        burst_ms, request_ms = run_kv.time_bursts(build, warmup=1, reps=2,
                                           settle=lambda: settled.append(1), rep0=10)
        self.assertEqual(order[:6], [("post", 0), ("post", 1), ("post", 2),
                                     ("wait", 0), ("wait", 1), ("wait", 2)])
        # every rep builds its own burst (fresh handles and tables) from rep0,
        # and handles are released after every burst
        self.assertEqual((built, len(settled)), ([10, 11, 12], 3))
        # warmups dropped; one completion mark per request per kept rep, offsets
        # from the burst start (never decreasing), and the burst is its last mark
        self.assertEqual((len(burst_ms), len(request_ms)), (2, 2 * 3))
        for rep, burst in enumerate(burst_ms):
            marks = request_ms[3 * rep : 3 * rep + 3]
            self.assertEqual(marks, sorted(marks))
            self.assertEqual(burst, marks[-1])


def _grid_args(**overrides):
    base = dict(workload_name="kv-dsv4", precision="fp8",
                isl_ladder="2048 8192 32768 65536 131072 524288",
                batch_sizes="1 2 4 8 16 32", pool_slack=2.0, max_burst_tokens=131072)
    base.update(overrides)
    return argparse.Namespace(**base)


class KVGrid(unittest.TestCase):
    def test_the_burst_token_cap_shapes_the_ladder(self):
        # batch x isl <= 131072 prompt tokens per burst; batch 1 always runs; 0 = no cap.
        full = [1, 2, 4, 8, 16, 32]
        for cap, expected in ((131072, [full, full[:5], full[:3], full[:2], [1], [1]]),
                              (0, [full] * 6)):
            with self.subTest(cap=cap):
                points, _isls, _batches = run_kv._grid(_grid_args(max_burst_tokens=cap))
                self.assertEqual([allowed for _, allowed in points], expected)

    def test_every_point_shares_one_pool_its_tables_fit(self):
        points, _isls, _batches = run_kv._grid(_grid_args())
        self.assertEqual(len({cfg["pool_rows"] for cfg, _ in points}), 1)
        for cfg, allowed in points:
            # the last request of the last table set still draws inside the pool
            last = kv_workload.block_table(cfg, 5, allowed[-1] - 1, cfg["table_sets"] - 1)
            self.assertLess(last.max(), cfg["pool_rows"])

    def test_batch_one_always_runs(self):
        points, _isls, batches = run_kv._grid(_grid_args(batch_sizes="2 4"))
        self.assertEqual(batches, [1, 2, 4])


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
            row("bulk", None, "pull", 48.3, 47.7),
            row("paged", 256, "push", 48.4, 47.6),
        ]},
        "implementation": {"library_version": "1.3.2", "transport": "ucx"},
        "topology": {"network": "infiniband"},
        "outcome": {"status": status, "reasons": []},
    }


class KVSummary(unittest.TestCase):
    def test_kv_documents_render_their_own_table(self):
        text = summarize.render([_kv_document()])
        self.assertIn("KV-transfer results", text)
        self.assertNotIn("EP results", text)
        self.assertIn("| `nixl` | 1.3.2 | rdma | infiniband | ucx | kv-dsv4 | success | pull "
                      "| 43.4 | 96.2 @b16 | 48.3 | 53.1 |", text)

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
