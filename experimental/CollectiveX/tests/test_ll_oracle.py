#!/usr/bin/env python3
"""End-to-end check of the low-latency per-slot correctness oracle on CPU.

The real low-latency kernels only run on GPU, but the oracle's plumbing —
per-(source, expert) slot normalization, the delivered-assignment multiset check,
per-expert counts, and the gate-weighted combine comparison — is platform-independent
Python. This drives `_run_ll_expert_oracle` against a single-rank CPU fake that
implements CORRECT low-latency semantics (deliver one slot per local (token, expert)
assignment; combine = source-side gate-weighted sum), so a structural bug in the oracle
or a divergence between its expected-combine model and true low-latency behavior fails
here rather than silently reding every GPU leg.
"""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "bench")]

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

    def test_correct_ll_backend_passes_every_oracle_check(self):
        import routing

        torch = _torch
        T, hidden, topk, experts = 8, 128, 4, 16  # ep_size 1 -> experts_per_rank == experts
        problem, idx_g, w_g = self._problem(T, hidden, topk, experts, seed=67)
        backend = _FakeLLBackend(experts_per_rank=experts, seed=67)
        with mock.patch.object(torch.cuda, "synchronize", lambda *a, **k: None):
            report = ep_harness._run_ll_expert_oracle(
                torch, routing, backend, problem, idx_g, w_g,
                rank=0, experts_per_rank=experts, scale_up_domain=1, seed=67,
            )
        self.assertTrue(report["passed"], report["checks"])
        for name, ok in report["checks"].items():
            self.assertTrue(ok, f"check {name} failed: {report}")
        self.assertLess(report["max_elementwise_relative_error"], ep_harness.COMBINE_REL_TOL)

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


if __name__ == "__main__":
    unittest.main()
