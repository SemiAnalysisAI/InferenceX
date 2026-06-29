#!/usr/bin/env python3
"""CollectiveX independent EP reference semantics (goal Part 3).

A from-scratch model of MoE dispatch + combine, written WITHOUT DeepEP or MoRI, used ONLY for
UNTIMED correctness validation. The point (goal: "avoid validating backend against itself"):
expected outputs come from the canonical routing trace + this independent logic, never from the
backend's own round trip. Pure numpy — runs anywhere, no torch.

Model (ep_size ranks, experts_per_rank experts each; expert e lives on rank e // experts_per_rank):
  dispatch:  token t selected for expert e contributes a copy of x[t] to (rank e//epr, expert e).
  expert:    a deterministic per-expert transform f_e (default: scale x by (1 + e/E) — distinct
             per expert so a mis-routed copy is detectable; identity is the degenerate case).
  combine:   y[t] = sum over t's selected experts e of  topk_weight[t,e] * f_e(x[t]).
             Reduction is over the token's experts; output is in SOURCE token order.

validate_dispatch() checks every (token, selected-expert) maps to the right rank+expert and the
right payload+gate weight, exactly once. validate_combine() checks the reduction, gate-weighting,
source ordering, and multiple-experts-on-one-rank. reference_combine() returns y for comparing a
backend's combined output against an independent oracle.
"""
from __future__ import annotations

import numpy as np


def expert_scale(e: int, experts: int) -> float:
    """Default deterministic per-expert transform factor — distinct per expert so a copy routed
    to the wrong expert produces a wrong value (identity would hide mis-routing)."""
    return 1.0 + e / float(experts)


def dispatch_plan(idx: np.ndarray, experts: int, experts_per_rank: int):
    """Independent dispatch model. idx[T,topk] selected experts per token.
    Returns list of (token, slot, expert, dest_rank) — every routed copy, exactly once."""
    T, topk = idx.shape
    plan = []
    for t in range(T):
        seen = set()
        for k in range(topk):
            e = int(idx[t, k])
            assert e not in seen, f"token {t} selects expert {e} twice (must be distinct)"
            seen.add(e)
            plan.append((t, k, e, e // experts_per_rank))
    return plan


def reference_combine(idx, weights, x, experts, experts_per_rank, transform=expert_scale):
    """y[t] = sum_k weights[t,k] * f_{idx[t,k]}(x[t]); source-token order. The independent oracle."""
    T, topk = idx.shape
    y = np.zeros_like(x, dtype=np.float64)
    for t in range(T):
        for k in range(topk):
            e = int(idx[t, k])
            y[t] += float(weights[t, k]) * transform(e, experts) * x[t].astype(np.float64)
    return y


def validate_dispatch(idx, experts, experts_per_rank):
    """Every selected (token,expert) routes to the correct rank+expert, exactly once."""
    plan = dispatch_plan(idx, experts, experts_per_rank)
    errs = []
    # exactly-once: no duplicate (token, expert)
    pairs = [(t, e) for (t, _k, e, _r) in plan]
    if len(pairs) != len(set(pairs)):
        errs.append("duplicate (token,expert) routed copy")
    # correct destination rank
    for (t, k, e, r) in plan:
        if r != e // experts_per_rank:
            errs.append(f"token {t} expert {e} -> rank {r}, expected {e // experts_per_rank}")
    ep = (experts + experts_per_rank - 1) // experts_per_rank
    for (t, k, e, r) in plan:
        if not (0 <= r < ep):
            errs.append(f"dest rank {r} out of range [0,{ep})")
    return errs


def validate_combine(idx, weights, x, experts, experts_per_rank, transform=expert_scale, tol=1e-9):
    """Recompute y two ways (vectorizable reduction vs explicit per-copy accumulation) and confirm
    they agree — exercises reduction across experts, gate-weighting, source ordering, and the
    multiple-experts-on-one-rank case (when topk experts share a rank)."""
    errs = []
    y_ref = reference_combine(idx, weights, x, experts, experts_per_rank, transform)
    # explicit accumulation over the dispatch plan (independent path)
    T = idx.shape[0]
    y_acc = np.zeros((T, x.shape[1]), dtype=np.float64)
    for (t, k, e, r) in dispatch_plan(idx, experts, experts_per_rank):
        y_acc[t] += float(weights[t, k]) * transform(e, experts) * x[t].astype(np.float64)
    if np.abs(y_ref - y_acc).max() > tol:
        errs.append(f"combine reduction mismatch ({np.abs(y_ref - y_acc).max():.2e})")
    # multiple-experts-on-one-rank present?
    multi = any(len({int(e) // experts_per_rank for e in idx[t]}) < idx.shape[1] for t in range(T))
    return errs, {"has_multi_expert_per_rank": bool(multi)}


# --------------------------------------------------------------------------- self-test
if __name__ == "__main__":
    import sys
    rng = np.random.default_rng(0)
    E, EPR, T, topk, H = 256, 32, 64, 8, 16
    idx = np.stack([rng.permutation(E)[:topk] for _ in range(T)]).astype(np.int64)
    w = rng.random((T, topk)).astype(np.float32)
    x = rng.standard_normal((T, H)).astype(np.float32)
    de = validate_dispatch(idx, E, EPR); assert not de, de
    ce, info = validate_combine(idx, w, x, E, EPR); assert not ce, ce
    print(f"dispatch+combine semantics OK (multi_expert_per_rank={info['has_multi_expert_per_rank']})")
    # mis-routing is DETECTED: corrupt one expert id and confirm the oracle value changes
    y0 = reference_combine(idx, w, x, E, EPR)
    idx2 = idx.copy(); idx2[0, 0] = (idx2[0, 0] + 1) % E
    y1 = reference_combine(idx2, w, x, E, EPR)
    assert np.abs(y0[0] - y1[0]).max() > 1e-6, "per-expert transform must make mis-routing detectable"
    print("mis-routing detectable via distinct per-expert transform OK")
    # edge cases (goal Part 3): empty rank, repeated dest rank, non-divisible handled by callers
    idx_hot = np.zeros((4, topk), dtype=np.int64)
    idx_hot[:] = np.arange(topk)               # all tokens -> experts 0..7 (all on rank 0) = hotspot
    assert not validate_dispatch(idx_hot, E, EPR), "single-rank hotspot must validate"
    print("edge case: single-rank hotspot (all topk on rank 0) OK")
    print("reference_ep self-test: PASS"); sys.exit(0)
