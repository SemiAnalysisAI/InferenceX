#!/usr/bin/env python3
"""Measurement-model tests: the low-latency correctness oracle, and the bandwidth math."""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "bench")]
sys.path[:0] = [str(Path(__file__).resolve().parents[1])]

import bandwidth  # noqa: E402


# ---- from test_ll_oracle.py -------------------------------------------------------
try:
    import torch as _torch
except Exception:  # torch is absent in the plain CPU test image; runs on GPU CI
    _torch = None

import ep_harness  # noqa: E402  (stdlib-only at import)


class _FakeLLBackend:
    """Correct single-rank low-latency backend over pure CPU torch (ep_size == 1, so
    every expert is local). Delivers one slot per (source token, expert) assignment in
    per-expert order and combines by the source-side gate-weighted sum the kernel does."""

    name = "fake-ll"
    receive_layout = "token-expert"
    combine_weight_semantics = "weighted-kernel-sum"

    def __init__(self, experts_per_rank: int, seed: int):
        self.experts_per_rank = experts_per_rank
        self.seed = seed

    def semantic_payload(self, x):
        return x  # BF16 control: no quantization round-trip

    def dispatch(self, p):
        torch = _torch
        idx = p.topk_idx  # [T, topk] global expert ids (all local at ep_size 1)
        slot_token, slot_expert = [], []
        for e in range(self.experts_per_rank):
            tokens = (idx == e).any(dim=1).nonzero(as_tuple=True)[0]
            for t in tokens.tolist():
                slot_token.append(t)
                slot_expert.append(e)
        slot_token = torch.tensor(slot_token, dtype=torch.int64)
        slot_expert = torch.tensor(slot_expert, dtype=torch.int64)
        return types.SimpleNamespace(
            slot_token=slot_token,
            slot_expert=slot_expert,
            counts=torch.bincount(slot_expert, minlength=self.experts_per_rank),
        )

    def recv_tokens(self, h):
        return int(h.slot_token.numel())

    def inspect_dispatch(self, p, h):
        return types.SimpleNamespace(
            payload=p.x.index_select(0, h.slot_token),
            expert_ids=h.slot_expert,  # rank 0 -> local id == global id
            local_expert_counts=h.counts,
        )

    def combine_transformed(self, p, h, transformed):
        torch = _torch
        T = p.x.shape[0]
        out = torch.zeros((T, p.x.shape[1]), dtype=torch.float32)
        for i in range(int(h.slot_token.numel())):
            token = int(h.slot_token[i])
            expert = int(h.slot_expert[i])
            gate = float(p.topk_weights[token][(p.topk_idx[token] == expert).nonzero()[0]])
            out[token] += gate * transformed[i].float()
        return out.to(p.x.dtype)


@unittest.skipUnless(_torch is not None, "low-latency oracle e2e check requires torch")
class LowLatencyOracleEndToEnd(unittest.TestCase):
    def _problem(self, T: int, hidden: int, topk: int, experts: int, seed: int):
        import routing

        idx_g, w_g = routing.build_global_routing(T, experts, topk, "uniform", seed)
        x = routing.activations_for_source_ids(
            _torch.arange(T, dtype=_torch.int64), hidden, seed, _torch.bfloat16
        )
        problem = types.SimpleNamespace(
            x=x, topk_idx=idx_g.clone(), topk_weights=w_g.clone()
        )
        return problem, idx_g, w_g

    def test_corrupted_combine_trips_the_gate(self):
        import routing

        torch = _torch
        T, hidden, topk, experts = 8, 128, 4, 16
        problem, idx_g, w_g = self._problem(T, hidden, topk, experts, seed=67)
        backend = _FakeLLBackend(experts_per_rank=experts, seed=67)
        # A backend that returns the unweighted sum (skips the gate) must fail combine_values.
        backend.combine_transformed = lambda p, h, transformed: sum(
            transformed[i] for i in range(int(h.slot_token.numel()))
        )  # wrong shape/values on purpose
        with mock.patch.object(torch.cuda, "synchronize", lambda *a, **k: None):
            report = ep_harness._run_ll_expert_oracle(
                torch, routing, backend, problem, idx_g, w_g,
                rank=0, experts_per_rank=experts, scale_up_domain=1, seed=67,
            )
        self.assertFalse(report["checks"]["combine_values"])


# ---- from test_bandwidth.py -------------------------------------------------------
COMPONENTS = bandwidth.COMPONENTS


def _row(tokens, nbytes, latency, passed=True):
    """A measurement row. `latency` is a scalar, or a per-component dict whose None marks
    that component unavailable."""
    lat = latency if isinstance(latency, dict) else dict.fromkeys(COMPONENTS, latency)
    return {
        "tokens_per_rank": tokens,
        "components": {c: {"percentiles_us": None if lat[c] is None else {
            "p50": lat[c], "p90": lat[c], "p95": lat[c], "p99": lat[c] * 2.0}}
            for c in COMPONENTS},
        "byte_provenance": {c: {"total_logical_bytes": nbytes} for c in COMPONENTS},
        "correctness": {"passed": passed},
        "routing": {"locality": {"cross_node_fraction": 0.5}},
    }


def _doc(rows, ep=2, mode="normal"):
    case = {"ep": ep, "backend": "deepep-v2", "precision": "bf16", "phase": "decode",
            "mode": mode, "suite": "s", "routing": "uniform"}
    return {
        "generated_at": "2026-07-25T22:12:55.760511+00:00",
        "identity": {"attempt_ordinal": 1, "allocation_factors": {"run_id": "30177021271"},
                     "case_factors": {"sku": "h100", "case": case}},
        "measurement": {"rows": rows},
        "outcome": {"status": "success"},
    }


def _linear(pairs, passed=True):
    """Rows exactly on latency = 10us + bytes * 2e-6, i.e. alpha=10, beta_agg=500 GB/s."""
    return [_row(t, b, 10.0 + b * 2e-6, passed) for t, b in pairs]


LADDER = ((8, 1e6), (16, 2e6), (32, 3e6))


class BandwidthMath(unittest.TestCase):
    def test_fit_is_none_when_undefensible(self):
        flat = [_row(t, b, 12.0) for t, b in LADDER]            # slope <= 0
        self.assertIsNone(bandwidth.fit_alpha_beta(_doc(_linear(LADDER[:2])), "dispatch"))
        self.assertIsNone(bandwidth.fit_alpha_beta(_doc(flat), "dispatch"))

    def test_noisy_ladder_withholds_beta(self):
        # A positive slope through noise still fits; printing its beta once produced a
        # physically impossible 1018 GB/s per GPU on a B200 at R2 = 0.29.
        rows = [_row(t, b, lat) for t, b, lat in (
            (8, 1e6, 300.0), (16, 2e6, 40.0), (32, 3e6, 260.0),
            (64, 4e6, 60.0), (128, 5e6, 320.0))]
        fit = bandwidth.fit_alpha_beta(_doc(rows), "dispatch")
        self.assertLess(fit.r2, bandwidth.FIT_MIN_R2)
        self.assertFalse(fit.beta_is_reliable)
        out = bandwidth.render([_doc(rows)])
        self.assertIn("beta=unreliable", out)
        self.assertNotIn("GB/s alpha", out)  # no number presented as measured

    def test_gate_failed_rung_excluded_from_fit_and_marked(self):
        rows = _linear(LADDER) + [_row(64, 4e6, 999.0, passed=False)]
        fit = bandwidth.fit_alpha_beta(_doc(rows), "dispatch")
        self.assertEqual((fit.points, fit.excluded_rows), (3, 1))
        self.assertAlmostEqual(fit.beta_gbps, 250.0, places=4)  # the corrupt rung didn't steer it
        out = bandwidth.render([_doc(rows)])
        self.assertIn("[correctness FAILED]", out)
        self.assertIn("excluded 1 gate-failed rung", out)

    def test_wire_basis_bytes_are_preferred_over_the_deduplicated_basis(self):
        # A token-expert LL row moves one copy per (token, expert): dividing GB/s from the
        # rank-deduplicated `byte_provenance` published a LOWER BOUND as if it were the wire
        # rate (34% low on nccl-ep LL EP8 at T=128). With `wire_byte_provenance` present the
        # fit must use it; the pre-wire fixture rows above pin the fallback path.
        rows = _linear(LADDER)
        for row in rows:
            row["wire_byte_provenance"] = {
                c: {"total_logical_bytes":
                    row["byte_provenance"][c]["total_logical_bytes"] * 3 // 2}
                for c in COMPONENTS
            }
        baseline = bandwidth.fit_alpha_beta(_doc(_linear(LADDER)), "dispatch")
        fit = bandwidth.fit_alpha_beta(_doc(rows), "dispatch")
        self.assertAlmostEqual(fit.beta_gbps, baseline.beta_gbps * 1.5, places=4)


if __name__ == "__main__":
    unittest.main()
