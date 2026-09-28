#!/usr/bin/env python3
"""The kv-transfer suite's scheduling, argv codec, and summary contracts.

Three seams keep KV legs honest end to end: sweep_matrix must emit kv shards
only for SKUs whose registry carries `kv_backends` (and must not perturb the EP
matrix at all); config.py must encode a kv case into run_kv argv behind the
`--entrypoint` marker the rank wrapper dispatches on; and summarize must render
kv documents in their own table instead of crashing the EP renderer.
"""
from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "bench"), str(ROOT / "runtime")]

import config as runtime_config  # noqa: E402
import ep_harness  # noqa: E402
import summarize  # noqa: E402
import sweep_matrix  # noqa: E402


class KVMatrix(unittest.TestCase):
    def test_kv_shards_only_where_the_registry_enables_them(self):
        shards = sweep_matrix.resolve_matrix(suites="kv-transfer")["include"]
        self.assertTrue(shards, "registry carries kv_backends but no shard resolved")
        enabled = {
            sku for sku, platform in sweep_matrix.PLATFORMS.items()
            if platform.get("kv_backends")
        }
        self.assertEqual({shard["sku"] for shard in shards}, enabled)
        default = sweep_matrix.KV_SWEEP["scheduling"]["default"]
        for shard in shards:
            self.assertEqual((shard["nodes"], shard["gpus_per_node"]), (2, 1))
            self.assertEqual(shard["launcher"], sweep_matrix.PLATFORMS[shard["sku"]]["launcher"])
            scheduling = sweep_matrix.KV_SWEEP["scheduling"].get(shard["sku"], default)
            self.assertEqual(shard["allocation_minutes"], scheduling["allocation_minutes"])
            self.assertEqual(shard["run_timeout"], scheduling["run_timeout"])
            # The guard fires inside the allocation, and the GitHub job outlives it.
            self.assertLess(shard["run_timeout"], shard["allocation_minutes"] * 60)
            self.assertGreater(shard["job_timeout_minutes"], shard["allocation_minutes"])
            # the suite sweeps DeepSeek-V4-Pro's shape; its dtype mix is
            # architectural, so one workload x one precision
            self.assertEqual(
                {(c["workload"], c["precision"]) for c in shard["cases"]},
                {("kv-dsv4", "fp8")})
            if shard["sku"] == "mi355x" and shard["backend"] == "mooncake":
                # AMD's atom-dev build: push-only (upstream ionic RDMA READ is
                # broken), GPU-paired NIC filter, shipped inside a pinned image,
                # and a pool the ionic NICs can register.
                self.assertEqual({c["ops"] for c in shard["cases"]}, {"push"})
                self.assertEqual({c["kv_device"] for c in shard["cases"]}, {"rdma{gpu}"})
                self.assertTrue(shard["image"].startswith("rocm/atom-dev:"))
                self.assertEqual({c["pool_budget"] for c in shard["cases"]}, {20 << 30})
            else:
                self.assertEqual({c["ops"] for c in shard["cases"]}, {"pull push"})
                self.assertNotIn("image", shard)
                self.assertEqual({c["kv_device"] for c in shard["cases"]}, {""})
                self.assertFalse([c for c in shard["cases"] if "pool_budget" in c])
            for case in shard["cases"]:
                self.assertEqual(case["suite"], "kv-transfer")
                self.assertEqual(case["ep"], 2)
                self.assertEqual(
                    case["case_id"], ep_harness.case_id(shard["sku"], case))

    def test_kv_never_perturbs_the_ep_matrix(self):
        ep_only = sweep_matrix.resolve_matrix()
        both = sweep_matrix.resolve_matrix(suites="ep,kv-transfer")
        self.assertEqual(
            ep_only["include"],
            [s for s in both["include"] if s.get("suite") != "kv-transfer"])
        self.assertEqual(
            ep_only["requested_cases"],
            [c for c in both["requested_cases"] if c["case"].get("suite") != "kv-transfer"])

    def test_unknown_suite_fails_closed(self):
        with self.assertRaises(SystemExit):
            sweep_matrix.resolve_matrix(suites="kv-transfr")

    def test_precision_filter_applies_to_kv(self):
        # dsv4 is fp8-only, so a bf16-scoped dispatch has no kv legs at all.
        matrix = sweep_matrix.resolve_matrix(suites="kv-transfer", precisions="bf16")
        self.assertEqual(matrix["include"], [])
        matrix = sweep_matrix.resolve_matrix(suites="kv-transfer", precisions="fp8")
        self.assertTrue(matrix["include"])

    def test_ep_only_filters_need_the_ep_suite(self):
        for options in ({"backend": "deepep-v2"}, {"modes": "normal"}, {"ep_sizes": "8"}):
            with self.subTest(options=options), self.assertRaises(SystemExit):
                sweep_matrix.resolve_matrix(suites="kv-transfer", **options)

    def test_the_workload_map_matches_the_presets(self):
        # sweep_matrix stays stdlib-only for the bare-runner matrix step, so it cannot import
        # kv_workload; pin its workload -> precision map to the workload model instead.
        import kv_workload

        for workload, precisions in sweep_matrix.KV_SWEEP["workloads"].items():
            preset = workload.removeprefix("kv-")
            with self.subTest(workload=workload):
                self.assertIn(preset, kv_workload.PRESETS)
                for precision in precisions:
                    kv_workload.plan_config(preset, precision, 2048, 256)

    def test_every_kv_backend_passes_its_launcher_identity_gate(self):
        # The launchers collx_die on unknown COLLX_BENCH values before anything
        # runs; a registry kv backend its launcher rejects is a dead shard
        # (this exact gap shipped once — every kv leg died at the gate).
        launchers = Path(sweep_matrix.__file__).parent / "launchers"
        for sku, platform in sweep_matrix.PLATFORMS.items():
            source = (launchers / f"launch_{platform['launcher']}.sh").read_text()
            for backend in platform.get("kv_backends", {}):
                with self.subTest(sku=sku, backend=backend):
                    self.assertRegex(source, rf"(^|[ |]){backend}( |\)|\s*\|)")


class KVArgvCodec(unittest.TestCase):
    def _shard(self, backend="nixl"):
        shards = sweep_matrix.resolve_matrix(suites="kv-transfer")["include"]
        return next(shard for shard in shards if shard["backend"] == backend)

    @staticmethod
    def _captured_argv(case, sku):
        class _Stdout:
            buffer = io.BytesIO()

        saved, sys.stdout = sys.stdout, _Stdout()
        try:
            runtime_config._emit_argv(case, 1, sku, "20260807", 0)
            return sys.stdout.buffer.getvalue().decode().split("\0")[:-1]
        finally:
            sys.stdout = saved

    def _parsed(self, shard):
        import argparse

        import run_kv

        argv = self._captured_argv(shard["cases"][0], shard["sku"])
        self.assertEqual(argv[:2], ["--entrypoint", "run_kv"])
        parser = argparse.ArgumentParser()
        parser.add_argument("--backend", required=True, choices=["nixl", "mori-io", "mooncake"])
        run_kv.add_kv_args(parser)
        return parser.parse_args(argv[2:])

    def test_kv_case_round_trips_through_the_run_kv_parser(self):
        import run_kv

        shard = self._shard()
        case = shard["cases"][0]
        args = self._parsed(shard)
        self.assertEqual((args.backend, args.workload_name, args.precision, args.fabric),
                         (case["backend"], case["workload"], case["precision"], case["mode"]))
        self.assertEqual((args.warmup, args.reps, args.trials),
                         (case["warmup"], case["reps"], case["trials"]))
        self.assertEqual((args.batch_sizes, args.kv_device, args.ops),
                         (case["batch_sizes"], case["kv_device"], case["ops"]))
        self.assertEqual(args.case_id, case["case_id"])
        self.assertEqual(args.pool_budget, run_kv.POOL_BUDGET)
        self.assertEqual(args.out, f"results/{case['case_id']}_20260807-c000.json")
        # run_kv recomputes the identity from the same factors and refuses a mismatch.
        self.assertEqual(ep_harness.case_id(shard["sku"], run_kv.kv_case(args)), case["case_id"])

    def test_a_pool_budget_reaches_run_kv(self):
        shard = next(s for s in sweep_matrix.resolve_matrix(suites="kv-transfer")["include"]
                     if s["cases"][0].get("pool_budget"))
        self.assertEqual(self._parsed(shard).pool_budget, shard["cases"][0]["pool_budget"])


class _StubDist:
    """all_gather_object across a simulated 2-rank pair."""

    def __init__(self, other_value):
        self.other = other_value

    def all_gather_object(self, out, mine):
        out[0], out[1] = mine, self.other


class VerdictExchange(unittest.TestCase):
    """Bulk rows have no verifying side; that path crashed on the metal (gb200
    smoke 22840: StopIteration on both ranks) before this contract existed."""

    def test_the_verifying_rank_supplies_the_verdict(self):
        import run_kv

        verdict = run_kv.exchange_verdict(
            _StubDist(None), "initiator", "initiator", lambda: (False, "bad page"))
        self.assertEqual(verdict, {"passed": False, "detail": "bad page"})

    def test_the_other_rank_receives_it(self):
        import run_kv

        verdict = run_kv.exchange_verdict(
            _StubDist({"passed": False, "detail": "bad page"}), "target", "initiator",
            lambda: (True, ""))
        self.assertEqual(verdict["passed"], False)

    def test_a_row_with_no_verifying_side_passes_without_a_gather_crash(self):
        import run_kv

        verdict = run_kv.exchange_verdict(
            _StubDist(None), "initiator", "none",
            lambda: (_ for _ in ()).throw(AssertionError("must not verify")))
        self.assertEqual(verdict, {"passed": True, "detail": ""})


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
        import run_kv

        for env, device, expected in self.CASES:
            with self.subTest(env=env, device=device):
                env = dict(env)
                run_kv.export_ucx_selectors(env, device=device)
                self.assertEqual({k: v for k, v in env.items() if k.startswith("UCX_")}, expected)


def _kv_document(status="success", sku="b200-nscale"):
    def row(kind, page, op, gbps, p50, batch=1):
        return {"kind": kind, "preset": "dsv4", "isl": 32768, "page_tokens": page,
                "op": op, "descs": 1, "req_bytes": 1, "batch": batch, "prep_ms": 0.1,
                "latency_ms": {"p50": p50, "p95": p50, "min": p50, "max": p50, "n": 48},
                "request_ms": {"p50": p50, "p95": p50, "min": p50, "max": p50,
                               "n": 48 * batch},
                "gbps_p50": gbps, "gbps_p50_incl_prep": gbps,
                "verify": {"passed": status == "success", "detail": ""}}

    return {
        "version": 1,
        "record_type": "case-attempt",
        "identity": {"case_factors": {"sku": sku, "case": {
            "suite": "kv-transfer", "backend": "nixl", "workload": "kv-dsv4",
            "mode": "rdma", "phase": "xfer", "ep": 2, "routing": "paged",
            "precision": "fp8"}}},
        "measurement": {"rows": [
            row("paged", 256, "pull", 43.4, 53.1),
            row("paged", 256, "pull", 96.2, 21.4, batch=16),
            # a smaller measured block must lose to the production block size
            row("paged", 128, "pull", 12.4, 185.2),
            row("bulk", None, "pull", 48.3, 47.7),
            row("paged", 256, "push", 48.4, 47.6),
        ]},
        "topology": {"gpus_per_node": 1, "scale_up_domain": 8, "nodes": 2},
        "outcome": {"status": status, "reasons": []},
    }


class BurstTiming(unittest.TestCase):
    def test_a_burst_posts_every_request_before_waiting_on_any(self):
        from kv_backend import time_bursts

        order = []
        pairs = [(lambda i=i: order.append(("post", i)),
                  lambda i=i: order.append(("wait", i))) for i in range(3)]
        burst_ms, request_ms = time_bursts(pairs, warmup=1, reps=2)
        self.assertEqual(len(burst_ms), 2)
        # one completion mark per request per kept rep, in posting order
        self.assertEqual(len(request_ms), 2 * 3)
        self.assertEqual(order[:6], [("post", 0), ("post", 1), ("post", 2),
                                     ("wait", 0), ("wait", 1), ("wait", 2)])

    def test_the_burst_sample_is_the_last_request_mark(self):
        from kv_backend import time_bursts

        pairs = [(lambda: None, lambda: None)] * 2
        burst_ms, request_ms = time_bursts(pairs, warmup=0, reps=1)
        self.assertEqual(burst_ms[0], request_ms[-1])
        # marks are offsets from the burst start, so they never decrease
        self.assertEqual(request_ms, sorted(request_ms))


class KVGrid(unittest.TestCase):
    @staticmethod
    def _args(**overrides):
        import argparse

        base = dict(workload_name="kv-dsv4", precision="fp8",
                    isl_ladder="8192 32768 131072 524288", page_tokens="256",
                    batch_sizes="1 2 4 8 16 32 64", pool_slack=2.0)
        import run_kv

        base["pool_budget"] = run_kv.POOL_BUDGET
        base.update(overrides)
        return argparse.Namespace(**base)

    def test_the_packed_grid_sheds_only_where_the_pool_budget_bites(self):
        # Packed block-major geometry: a 512k-ISL block-256 request is 6,146
        # descriptors, so no batch on this ladder nears DESC_BUDGET. Only the
        # 512k point sheds, and via the pool budget: its batch-32 pool plans
        # ~118 GB against the 64 GiB budget, batch 16 fits at ~59 GB.
        import run_kv

        points, isls, batches = run_kv._grid(self._args())
        self.assertEqual((isls, batches),
                         ([8192, 32768, 131072, 524288], [1, 2, 4, 8, 16, 32, 64]))
        allowed = {cfg["isl"]: allowed for cfg, allowed in points}
        self.assertEqual(allowed[8192], [1, 2, 4, 8, 16, 32, 64])
        self.assertEqual(allowed[32768], [1, 2, 4, 8, 16, 32, 64])
        self.assertEqual(allowed[131072], [1, 2, 4, 8, 16, 32, 64])
        self.assertEqual(allowed[524288], [1, 2, 4, 8, 16])
        for cfg, batch_list in points:
            self.assertEqual(cfg["descs"], 3 * -(-cfg["isl"] // 256) + 2)
            self.assertLessEqual(cfg["pool_bytes"], run_kv.POOL_BUDGET)
            for batch in batch_list[run_kv.LADDER_FLOOR:]:
                self.assertLessEqual(batch * cfg["descs"], run_kv.DESC_BUDGET)

    def test_descriptor_budget_sheds_batches_but_keeps_a_chartable_ladder(self):
        # DESC_BUDGET stays as the fail-closed guard for future presets whose
        # bursts are descriptor-bound. Pin it to 4 requests' descriptors at
        # the largest ISL: batches above the per-point allowance shed, but the
        # LADDER_FLOOR smallest batches always survive so every point keeps a
        # chartable batch ladder (the frontier draws its line through the
        # ladder at the largest measured ISL).
        import kv_workload
        import run_kv

        probe = kv_workload.plan_config("dsv4", "fp8", 524288, 256)
        saved, run_kv.DESC_BUDGET = run_kv.DESC_BUDGET, 4 * probe["descs"]
        try:
            points, _isls, _batches = run_kv._grid(self._args())
        finally:
            run_kv.DESC_BUDGET = saved
        allowed = {cfg["isl"]: allowed for cfg, allowed in points}
        self.assertEqual(allowed[8192], [1, 2, 4, 8, 16, 32, 64])   # 98 descs/req
        self.assertEqual(allowed[32768], [1, 2, 4, 8, 16, 32])      # 386
        self.assertEqual(allowed[131072], [1, 2, 4, 8, 16])         # 1538, floor
        self.assertEqual(allowed[524288], [1, 2, 4, 8, 16])         # 6146, floor

    def test_pool_budget_sheds_largest_batches_even_below_the_ladder_floor(self):
        # A point whose largest batch cannot fit the pool budget survives with the batches that
        # do. The budget is a hard memory limit, so it sheds batches the descriptor floor keeps:
        # pinned to the 512k point's batch-1 pool, only [1] remains.
        import kv_workload
        import run_kv

        for isl, batches, fit_batch, expected in ((32768, "1 4 16", 4, [1, 4]),
                                                  (524288, "1 2 4 8 16 32 64", 1, [1])):
            with self.subTest(isl=isl):
                args = self._args(isl_ladder=str(isl), batch_sizes=batches)
                args.pool_budget = kv_workload.plan_config(
                    "dsv4", "fp8", isl, 256, 2.0, batch_max=fit_batch)["pool_bytes"]
                points, _isls, _batches = run_kv._grid(args)
                self.assertEqual(points[0][1], expected)
                self.assertLessEqual(points[0][0]["pool_bytes"], args.pool_budget)


class RegistrationChunking(unittest.TestCase):
    # b300's NICs refuse cuda registrations past ~8 GiB, so the NIXL adapter
    # registers the pool in pieces. The pieces must never cut through a
    # descriptor of ANY planned config, which _harmonize guarantees by giving
    # every config one shared region layout.

    def test_harmonize_makes_region_bases_config_invariant(self):
        import run_kv

        points, _isls, _batches = run_kv._grid(KVGrid._args())
        layout = run_kv._harmonize(points)
        total = sum(nbytes for _, _, nbytes in layout)
        running = 0
        for base, _packed, nbytes in layout:
            self.assertEqual(base, running)
            running += nbytes
        for cfg, _ in points:
            self.assertEqual(cfg["pool_bytes"], total)
            for region, (base, packed, nbytes) in zip(cfg["regions"], layout):
                self.assertEqual(region["base"], base)
                self.assertEqual(region["packed_bytes"], packed)
                self.assertEqual(region["pool_blocks"], nbytes // packed)
                self.assertLessEqual(region["blocks_req"], region["pool_blocks"])

    def test_reg_spans_cut_each_region_on_its_own_packed_grid(self):
        import kv_nixl
        import run_kv

        points, _isls, _batches = run_kv._grid(KVGrid._args())
        layout = run_kv._harmonize(points)
        total = sum(nbytes for _, _, nbytes in layout)
        spans = kv_nixl.reg_spans(total, layout)
        # The full test grid plans a pool far past one chunk.
        self.assertGreater(len(spans), 1)
        # Exact in-order coverage, no gap, no overlap.
        self.assertEqual(spans[0][0], 0)
        for (a_off, a_len), (b_off, _) in zip(spans, spans[1:]):
            self.assertEqual(a_off + a_len, b_off)
        self.assertEqual(sum(length for _, length in spans), total)
        for off, length in spans:
            base, packed, _ = next(entry for entry in reversed(layout)
                                   if entry[0] <= off)
            self.assertEqual((off - base) % packed, 0)
            self.assertLessEqual(length, max(kv_nixl.REG_CHUNK_BYTES, packed))

    def test_no_descriptor_straddles_a_registration_cut(self):
        # Every block any config can ever address must land whole inside one
        # registered piece; a tiny cap on a small grid forces many cuts.
        import bisect

        import kv_nixl
        import run_kv

        args = KVGrid._args(isl_ladder="2048 8192", batch_sizes="1 4")
        points, _isls, _batches = run_kv._grid(args)
        layout = run_kv._harmonize(points)
        total = sum(nbytes for _, _, nbytes in layout)
        spans = kv_nixl.reg_spans(total, layout, cap=1 << 24)
        self.assertGreater(len(spans), len(layout))
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
        import kv_nixl

        self.assertEqual(kv_nixl.reg_spans(123456, None), [(0, 123456)])
        self.assertEqual(kv_nixl.reg_spans(123456, []), [(0, 123456)])


class KVSummary(unittest.TestCase):
    def test_kv_documents_render_their_own_table(self):
        text = summarize.render([_kv_document()])
        self.assertIn("KV-transfer results", text)
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
        text = summarize.render([_kv_document(status="invalid")])
        self.assertIn("INVALID", text)

    def test_ep_documents_do_not_grow_a_kv_table(self):
        self.assertNotIn("KV-transfer results", summarize.render([]))


if __name__ == "__main__":
    unittest.main()
