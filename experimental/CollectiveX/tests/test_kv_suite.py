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
            self.assertEqual(len(shard["cases"]), 4)  # 2 workloads x 2 precisions
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
        matrix = sweep_matrix.resolve_matrix(suites="kv-transfer", precisions="bf16")
        for shard in matrix["include"]:
            self.assertEqual({c["precision"] for c in shard["cases"]}, {"bf16"})

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

    def test_ports_in_selectors_pass_through(self):
        import run_kv

        env = {"COLLX_RDMA_DEVICES": "mlx5_18:1,mlx5_19"}
        run_kv.export_ucx_selectors(env)
        self.assertEqual(env["UCX_NET_DEVICES"], "mlx5_18:1,mlx5_19:1")


def _kv_document(status="success", sku="b200-nscale"):
    def row(kind, page, op, gbps, p50):
        return {"kind": kind, "preset": "mla", "isl": 32768, "page_tokens": page,
                "op": op, "descs": 1, "req_bytes": 1, "prep_ms": 0.1,
                "latency_ms": {"p50": p50, "p95": p50, "min": p50, "max": p50, "n": 24},
                "gbps_p50": gbps, "verify": {"passed": status == "success", "detail": ""}}

    return {
        "version": 1,
        "record_type": "case-attempt",
        "identity": {"case_factors": {"sku": sku, "case": {
            "suite": "kv-transfer", "backend": "nixl", "workload": "kv-mla",
            "mode": "rdma", "phase": "xfer", "ep": 2, "routing": "paged",
            "precision": "bf16"}}},
        "measurement": {"rows": [
            row("paged", 64, "pull", 43.4, 53.1),
            row("paged", 16, "pull", 12.4, 185.2),
            row("bulk", None, "pull", 48.3, 47.7),
            row("paged", 64, "push", 48.4, 47.6),
        ]},
        "topology": {"gpus_per_node": 1, "scale_up_domain": 8, "nodes": 2},
        "outcome": {"status": status, "reasons": []},
    }


class KVSummary(unittest.TestCase):
    def test_kv_documents_render_their_own_table(self):
        text = summarize.render([_kv_document()])
        self.assertIn("KV-transfer results", text)
        self.assertIn("| 43.4 | 12.4 | 48.3 | 53.1 |", text)
        self.assertNotIn("INVALID", text)

    def test_kv_invalid_counts_in_the_banner(self):
        text = summarize.render([_kv_document(status="invalid")])
        self.assertIn("INVALID", text)

    def test_ep_documents_do_not_grow_a_kv_table(self):
        self.assertNotIn("KV-transfer results", summarize.render([]))


if __name__ == "__main__":
    unittest.main()
