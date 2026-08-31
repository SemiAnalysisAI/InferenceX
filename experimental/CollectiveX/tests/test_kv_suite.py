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
        matrix = sweep_matrix.resolve_matrix(suites="kv-transfer")
        shards = matrix["include"]
        self.assertTrue(shards, "registry carries kv_backends but no shard resolved")
        enabled = {
            sku for sku, platform in sweep_matrix.PLATFORMS.items()
            if platform.get("kv_backends")
        }
        self.assertEqual({shard["sku"] for shard in shards}, enabled)
        for shard in shards:
            self.assertEqual(shard["nodes"], 2)
            self.assertEqual(shard["gpus_per_node"], 1)
            # the suite sweeps DeepSeek-V4-Pro's shape; its dtype mix is
            # architectural, so one workload x one precision
            self.assertEqual(
                {(c["workload"], c["precision"]) for c in shard["cases"]},
                {("kv-dsv4", "fp8")})
            if shard["sku"] == "mi355x" and shard["backend"] == "mooncake":
                # AMD's atom-dev build: push-only (upstream ionic RDMA READ is
                # broken), GPU-paired NIC filter, shipped inside a pinned image.
                self.assertEqual({c["ops"] for c in shard["cases"]}, {"push"})
                self.assertEqual({c["kv_device"] for c in shard["cases"]}, {"rdma{gpu}"})
                self.assertTrue(shard["image_ref"].startswith("rocm/atom-dev:"))
            elif shard["sku"] == "b300":
                # b300 pods expose two RDMA rails that are not cross-routable.
                # mooncake: the image's engine draws the peer NIC per request
                # blind to endpoint health, so every cross-rail draw stalls ~1s.
                # nixl: unpinned UCX pull degrades reproducibly across node
                # pairs while push stays at line rate; pinning one rail takes
                # multi-rail selection out of the measurement.
                self.assertEqual({c["ops"] for c in shard["cases"]}, {"pull push"})
                self.assertEqual({c["kv_device"] for c in shard["cases"]}, {"mlx5_0"})
                self.assertEqual(shard["image_ref"], "")
            else:
                self.assertEqual({c["ops"] for c in shard["cases"]}, {"pull push"})
                self.assertEqual(shard["image_ref"], "")
                self.assertEqual({c["kv_device"] for c in shard["cases"]}, {""})
            for case in shard["cases"]:
                self.assertEqual(case["suite"], "kv-transfer")
                self.assertEqual(case["ep"], 2)
                self.assertEqual(
                    case["case_id"], ep_harness.case_id(shard["sku"], case))

    def test_kv_never_perturbs_the_ep_matrix(self):
        ep_only = sweep_matrix.resolve_matrix(suites="ep-core")
        both = sweep_matrix.resolve_matrix()
        ep_ids = [s["id"] for s in ep_only["include"]]
        both_ep_ids = [s["id"] for s in both["include"] if "-kv-" not in s["id"]]
        self.assertEqual(sorted(ep_ids), sorted(both_ep_ids))
        self.assertEqual(
            [c for c in ep_only["requested_cases"]],
            [c for c in both["requested_cases"]
             if c["case"].get("suite") != "kv-transfer"])

    def test_unknown_suite_fails_closed(self):
        with self.assertRaises(SystemExit):
            sweep_matrix.resolve_matrix(suites="kv-transfr")

    def test_precision_filter_applies_to_kv(self):
        # dsv4 is fp8-only, so a bf16-scoped dispatch has no kv legs at all.
        matrix = sweep_matrix.resolve_matrix(suites="kv-transfer", precisions="bf16")
        self.assertEqual(matrix["include"], [])
        matrix = sweep_matrix.resolve_matrix(suites="kv-transfer", precisions="fp8")
        self.assertTrue(matrix["include"])

    def test_a_backend_scoped_dispatch_is_ep_only(self):
        matrix = sweep_matrix.resolve_matrix(backend="deepep-v2")
        self.assertFalse([s for s in matrix["include"] if "-kv-" in s["id"]])

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
    def _case(self):
        matrix = sweep_matrix.resolve_matrix(suites="kv-transfer")
        return matrix["include"][0]

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

    def test_kv_case_encodes_behind_the_entrypoint_marker(self):
        shard = self._case()
        case = shard["cases"][0]
        argv = self._captured_argv(case, shard["sku"])
        self.assertEqual(argv[:2], ["--entrypoint", "run_kv"])
        pairs = dict(zip(argv[2::2], argv[3::2]))
        self.assertEqual(pairs["--backend"], case["backend"])
        self.assertEqual(pairs["--workload-name"], case["workload"])
        self.assertEqual(pairs["--fabric"], case["mode"])
        self.assertEqual(pairs["--case-id"], case["case_id"])
        self.assertEqual(
            (pairs["--warmup"], pairs["--reps"], pairs["--trials"]),
            tuple(case["timing"].split(":")))
        self.assertEqual(pairs["--batch-sizes"], case["batch_sizes"])
        self.assertEqual(pairs["--kv-device"], case["kv_device"])
        self.assertEqual(pairs["--ops"], case["ops"])
        self.assertIn(shard["sku"], pairs["--out"])
        self.assertTrue(pairs["--out"].startswith("results/"))

    def test_a_malformed_kv_timing_profile_fails_closed(self):
        case = dict(self._case()["cases"][0], timing="2:8")
        with self.assertRaises(SystemExit):
            self._captured_argv(case, "sku")


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
    auto-selection is a wrong-fabric trap (b200-nscale's aux quad-port card,
    b300's storage IB) — while explicit UCX_* values always win."""

    def test_registry_selectors_map_to_ucx(self):
        import run_kv

        env = {"COLLX_RDMA_DEVICES": "mlx5_0,mlx5_10", "COLLX_IB_GID_INDEX": "3"}
        run_kv.export_ucx_selectors(env)
        self.assertEqual(env["UCX_NET_DEVICES"], "mlx5_0:1,mlx5_10:1")
        self.assertEqual(env["UCX_IB_GID_INDEX"], "3")

    def test_explicit_ucx_env_wins(self):
        import run_kv

        env = {"COLLX_RDMA_DEVICES": "mlx5_0", "UCX_NET_DEVICES": "rdma0:1",
               "COLLX_IB_GID_INDEX": "3", "UCX_IB_GID_INDEX": "1"}
        run_kv.export_ucx_selectors(env)
        self.assertEqual(env["UCX_NET_DEVICES"], "rdma0:1")
        self.assertEqual(env["UCX_IB_GID_INDEX"], "1")

    def test_registry_device_pin_narrows_the_inventory(self):
        # A kv_device pin on a UCX-backed case wins over the operator's full
        # RDMA inventory: rail-isolated pods (b300) publish one-rail rows.
        import run_kv

        env = {"COLLX_RDMA_DEVICES": "mlx5_0,mlx5_1"}
        run_kv.export_ucx_selectors(env, device="mlx5_0")
        self.assertEqual(env["UCX_NET_DEVICES"], "mlx5_0:1")

    def test_device_pin_overrides_host_inherited_ucx_env(self):
        # b300 ships a blanket 16-device UCX_NET_DEVICES in /etc/environment
        # (forwarded by srun --export=ALL); left standing it silently swallows
        # the registry pin, so the pin wins — same treatment as UCX_TLS=rc.
        import run_kv

        env = {"UCX_NET_DEVICES": "rdma0:1"}
        run_kv.export_ucx_selectors(env, device="mlx5_0")
        self.assertEqual(env["UCX_NET_DEVICES"], "mlx5_0:1")

    def test_empty_device_pin_keeps_the_inventory_path(self):
        import run_kv

        env = {"COLLX_RDMA_DEVICES": "mlx5_0,mlx5_1"}
        run_kv.export_ucx_selectors(env, device="")
        self.assertEqual(env["UCX_NET_DEVICES"], "mlx5_0:1,mlx5_1:1")

    def test_ports_in_selectors_pass_through(self):
        import run_kv

        env = {"COLLX_RDMA_DEVICES": "mlx5_18:1,mlx5_19"}
        run_kv.export_ucx_selectors(env)
        self.assertEqual(env["UCX_NET_DEVICES"], "mlx5_18:1,mlx5_19:1")

    def test_positive_tls_list_without_cuda_is_dropped(self):
        # b300 exports UCX_TLS=rc cluster-wide; an RC-only context closes the
        # cuda mds and NIXL registration of VRAM fails with NIXL_ERR_BACKEND.
        # Extending the list with cuda transports segfaults UCX rkey-config
        # resolution on the first GET, so the list is dropped entirely.
        import run_kv

        env = {"UCX_TLS": "rc"}
        run_kv.export_ucx_selectors(env)
        self.assertNotIn("UCX_TLS", env)

    def test_tls_lists_already_covering_cuda_stay_untouched(self):
        import run_kv

        for tls in ("^tcp", "rc,cuda_copy", "all"):
            env = {"UCX_TLS": tls}
            run_kv.export_ucx_selectors(env)
            self.assertEqual(env["UCX_TLS"], tls)
        env = {}
        run_kv.export_ucx_selectors(env)
        self.assertNotIn("UCX_TLS", env)


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

    def test_pool_budget_sheds_largest_batches_not_the_point(self):
        # A point whose largest batch cannot fit the pool budget must survive
        # with the batches that do: pin the budget between the batch-4 and
        # batch-16 pool sizes.
        import kv_workload
        import run_kv

        args = self._args(isl_ladder="32768", batch_sizes="1 4 16")
        budget = kv_workload.plan_config("dsv4", "fp8", 32768, 256,
                                         2.0, batch_max=4)["pool_bytes"]
        saved, run_kv.POOL_BUDGET = run_kv.POOL_BUDGET, budget
        try:
            points, _isls, _batches = run_kv._grid(args)
        finally:
            run_kv.POOL_BUDGET = saved
        self.assertEqual(points[0][1], [1, 4])
        self.assertLessEqual(points[0][0]["pool_bytes"], budget)

    def test_pool_budget_overrides_the_ladder_floor(self):
        # The descriptor floor keeps the LADDER_FLOOR smallest batches, but
        # the pool budget is a hard memory limit and must still shed a
        # floor-kept batch. Pin the budget to the 512k point's batch-1 pool
        # size: every larger batch survives the descriptor floor, then the
        # pool loop must drop them all, leaving [1].
        import kv_workload
        import run_kv

        args = self._args(isl_ladder="524288")
        budget = kv_workload.plan_config("dsv4", "fp8", 524288, 256,
                                         2.0, batch_max=1)["pool_bytes"]
        saved, run_kv.POOL_BUDGET = run_kv.POOL_BUDGET, budget
        try:
            points, _isls, _batches = run_kv._grid(args)
        finally:
            run_kv.POOL_BUDGET = saved
        self.assertEqual(points[0][1], [1])
        self.assertLessEqual(points[0][0]["pool_bytes"], budget)


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
