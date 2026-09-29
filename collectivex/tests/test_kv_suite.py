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
        order, built, settled = [], [], []

        def build(rep):
            built.append(rep)
            return [(lambda i=i: order.append(("post", i)),
                     lambda i=i: order.append(("wait", i))) for i in range(3)]

        burst_ms, request_ms = time_bursts(build, warmup=1, reps=2,
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
                isl_ladder="2048 8192 32768 65536 131072 524288", page_tokens="256",
                batch_sizes="1 2 4 8 16 32", pool_slack=2.0, max_burst_tokens=131072,
                table_sets=kv_workload.DEFAULT_TABLE_SETS,
                pool_budget=run_kv.POOL_BUDGET, seed=67)
    base.update(overrides)
    return argparse.Namespace(**base)


class KVGrid(unittest.TestCase):
    def test_the_burst_token_cap_shapes_the_ladder(self):
        # batch x isl <= 131072 prompt tokens per burst; batch 1 always runs.
        points, isls, batches = run_kv._grid(_grid_args())
        allowed = {cfg["isl"]: allowed for cfg, allowed in points}
        self.assertEqual(allowed, {2048: [1, 2, 4, 8, 16, 32], 8192: [1, 2, 4, 8, 16],
                                   32768: [1, 2, 4], 65536: [1, 2], 131072: [1],
                                   524288: [1]})
        for cfg, _ in points:
            self.assertEqual(cfg["table_sets"], kv_workload.DEFAULT_TABLE_SETS)
            self.assertLessEqual(cfg["pool_bytes"], run_kv.POOL_BUDGET)

    def test_no_cap_runs_the_full_ladder(self):
        points, _isls, batches = run_kv._grid(_grid_args(max_burst_tokens=0))
        self.assertTrue(all(allowed == batches for _, allowed in points
                            if _["isl"] <= 32768))

    def test_the_pool_budget_sheds_batches_then_table_sets(self):
        # Pinned to the 512k point's batch-1, two-table-set pool: batches are
        # already [1], so the table sets shed from four to two.
        args = _grid_args(isl_ladder="524288")
        args.pool_budget = kv_workload.plan_config(
            "dsv4", "fp8", 524288, 256, 2.0, batch_max=1, table_sets=2)["pool_bytes"]
        (cfg, allowed), = run_kv._grid(args)[0]
        self.assertEqual((allowed, cfg["table_sets"]), ([1], 2))
        # a multi-batch point sheds its batches before its table sets
        args = _grid_args(isl_ladder="2048", max_burst_tokens=0)
        args.pool_budget = kv_workload.plan_config(
            "dsv4", "fp8", 2048, 256, 2.0, batch_max=8)["pool_bytes"]
        (cfg, allowed), = run_kv._grid(args)[0]
        self.assertEqual((allowed, cfg["table_sets"]),
                         ([1, 2, 4, 8], kv_workload.DEFAULT_TABLE_SETS))

    def test_descriptor_budget_stays_the_fail_closed_guard(self):
        probe = kv_workload.plan_config("dsv4", "fp8", 524288, 256)
        with mock.patch.object(run_kv, "DESC_BUDGET", probe["descs"] - 1):
            points, _isls, _batches = run_kv._grid(_grid_args(isl_ladder="2048 524288"))
        self.assertEqual([cfg["isl"] for cfg, _ in points], [2048])


class RegistrationChunking(unittest.TestCase):
    # The NIXL adapter registers an oversized pool in pieces cut on the row
    # grid; every descriptor is a whole row, so no cut may split a row.

    def test_harmonize_gives_every_point_the_one_shared_pool(self):
        points, _isls, _batches = run_kv._grid(_grid_args())
        (base, row, nbytes), = run_kv._harmonize(points)
        self.assertEqual((base, row), (0, points[0][0]["row_bytes"]))
        for cfg, _ in points:
            self.assertEqual((cfg["pool_rows"] * row, cfg["pool_bytes"]), (nbytes, nbytes))

    def test_spans_tile_the_pool_on_the_row_grid(self):
        points, _isls, _batches = run_kv._grid(_grid_args(isl_ladder="2048 8192"))
        layout = run_kv._harmonize(points)
        total = layout[0][2]
        spans = kv_nixl.reg_spans(total, layout, cap=1 << 24)
        self.assertGreater(len(spans), 1)
        self.assertEqual(spans[0][0], 0)
        for (a_off, a_len), (b_off, _) in zip(spans, spans[1:]):
            self.assertEqual(a_off + a_len, b_off)
        self.assertEqual(sum(length for _, length in spans), total)
        row = layout[0][1]
        self.assertTrue(all(off % row == 0 and length % row == 0 for off, length in spans))

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
