#!/usr/bin/env python3
"""CollectiveX — deterministic, platform-independent MoE routing trace.

Fair-comparison fix #1: routing (per-token expert IDs + gate weights) is generated
ONCE from a fixed seed over the *global* token batch, indexed by global token id, and
is identical on every SKU for the same (seed, routing, global_tokens, experts, top-k,
experts_per_rank). Each rank materializes its slice `[rank*T,(rank+1)*T)`. Activations
are per-rank (same rank ⇒ same x on any platform), so a given global token id has
identical activation everywhere without materializing a global activation tensor.

Trace classes (the rank fan-out — #destination ranks a token's top-k experts touch —
is the property that makes an EP workload representative; review caught the old
default having fan-out 1):

  * uniform   — top-k distinct experts drawn uniformly per token. The DEFAULT.
                Expected fan-out for top-k=8, 256 experts, EP8 (32 experts/rank) ≈
                8·(1 − C(224,8)/C(256,8)) ≈ 5.3 ranks/token. Load ~ Poisson.
  * balanced  — load-equalized AND maximally spread: token i, slot j →
                (i + j·experts_per_rank) mod E, so the 8 experts sit one-per-rank
                (fan-out = ep_size) and every expert is hit equally. The high-fan-out,
                perfectly-balanced reference.
  * balanced-rank-local — the OLD degenerate "balanced": (i·top_k + j) mod E, i.e.
                top_k consecutive experts, which (top_k ≤ experts/rank, aligned) all
                land on ONE rank ⇒ fan-out 1, minimum communication. Kept as an
                explicit edge case, honestly named.
  * zipf      — expert popularity ∝ 1/rank (skewed load), uniform-ish fan-out.

Always publish the realized fan-out so the workload is never misread again
(`routing_stats`).
"""
from __future__ import annotations

import hashlib

import torch

_RANK_SUBSEED = 7919


def _cpu_gen(seed: int) -> "torch.Generator":
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return g


def build_global_routing(global_tokens: int, experts: int, topk: int,
                         routing: str, seed: int, experts_per_rank: int):
    """(idx[gt, topk] int64, weights[gt, topk] float32) on CPU — deterministic,
    independent of world/EP/platform, experts distinct within a token."""
    if topk > experts:
        raise ValueError(f"topk ({topk}) > experts ({experts})")
    gt = int(global_tokens)
    g = _cpu_gen(seed)
    if routing == "uniform":
        keys = torch.rand(gt, experts, generator=g)
        idx = keys.argsort(dim=1)[:, :topk].contiguous().to(torch.int64)
    elif routing == "balanced":
        # one expert per rank ⇒ fan-out = ep_size, perfectly balanced load.
        i = torch.arange(gt, dtype=torch.int64).unsqueeze(1)
        j = torch.arange(topk, dtype=torch.int64).unsqueeze(0)
        idx = (i + j * int(experts_per_rank)) % experts
    elif routing == "balanced-rank-local":
        # top_k consecutive (mod E) ⇒ all on ONE rank ⇒ fan-out 1 (min comm). Edge case.
        i = torch.arange(gt, dtype=torch.int64).unsqueeze(1)
        j = torch.arange(topk, dtype=torch.int64).unsqueeze(0)
        idx = (i * topk + j) % experts
    elif routing == "zipf":
        p = 1.0 / torch.arange(1, experts + 1, dtype=torch.float32)
        p = (p / p.sum()).expand(gt, experts)
        idx = torch.multinomial(p, topk, replacement=False, generator=g).to(torch.int64)
    else:
        raise ValueError(f"unknown routing '{routing}' (uniform|balanced|balanced-rank-local|zipf)")
    weights = torch.softmax(torch.randn(gt, topk, generator=g), dim=1).to(torch.float32)
    return idx, weights


def rank_slice(idx, weights, rank: int, tokens_per_rank: int):
    lo = rank * tokens_per_rank
    return idx[lo:lo + tokens_per_rank].contiguous(), weights[lo:lo + tokens_per_rank].contiguous()


def rank_activations(tokens: int, hidden: int, seed: int, rank: int, device, dtype=torch.bfloat16):
    g = _cpu_gen(int(seed) * _RANK_SUBSEED + int(rank) + 1)
    return torch.randn(tokens, hidden, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)


def routing_stats(idx, experts: int, experts_per_rank: int, weights=None) -> dict:
    """Realized routing properties for the GLOBAL trace — published per point so the
    fan-out / load can never be silently misread. idx is the global [gt, topk] tensor;
    weights the matching [gt, topk] gate weights (hashed too for workload identity).
    """
    ep = max(1, experts // max(1, experts_per_rank))
    ranks = (idx // experts_per_rank)                       # [gt, topk] destination rank per assignment
    # unique destination ranks per token (fan-out)
    onehot = torch.zeros(idx.shape[0], ep, dtype=torch.bool)
    onehot.scatter_(1, ranks.clamp(max=ep - 1), True)
    fanout = onehot.sum(dim=1)                              # [gt]
    hist = torch.bincount(fanout, minlength=ep + 1)[1:ep + 1].tolist()  # counts for fan-out 1..ep
    load = torch.bincount(idx.reshape(-1), minlength=experts).float()
    # token-copies SENT to each destination rank (the "send histogram", review #3).
    rank_load = torch.bincount(ranks.reshape(-1).clamp(max=ep - 1), minlength=ep).tolist()
    # SHA-256 workload identity over BOTH topk_idx and gate weights (review #3): a chart
    # point's routing is provably identical across SKUs only if both hashes match.
    idx_bytes = idx.to(torch.int32).cpu().numpy().tobytes()
    idx_hash = hashlib.sha256(idx_bytes).hexdigest()[:16]
    if weights is not None:
        w_bytes = weights.to(torch.float32).cpu().numpy().tobytes()
        w_hash = hashlib.sha256(w_bytes).hexdigest()[:16]
        routing_hash = hashlib.sha256(idx_bytes + w_bytes).hexdigest()[:16]  # combined identity
    else:
        w_hash, routing_hash = None, idx_hash
    return {
        "fanout_mean": float(fanout.float().mean()),
        "fanout_min": int(fanout.min()), "fanout_max": int(fanout.max()),
        "fanout_hist": hist,                               # index k-1 = #tokens with fan-out k
        "rank_load_hist": rank_load,                       # token-copies sent to each dest rank
        "routed_copies": int(fanout.sum()),                # total (token, dest-rank) pairs
        "expert_load_min": int(load.min()), "expert_load_max": int(load.max()),
        "expert_load_mean": float(load.mean()),
        "routing_hash": routing_hash, "idx_hash": idx_hash, "weights_hash": w_hash,
    }
