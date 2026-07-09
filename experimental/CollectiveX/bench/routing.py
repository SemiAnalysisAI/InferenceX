#!/usr/bin/env python3
"""CollectiveX — deterministic, platform-independent MoE routing trace.

Fair-comparison fix #1: routing (per-token expert IDs + gate weights) is generated
ONCE from a fixed seed over the *global* token batch, indexed by global token id, and
is identical on every SKU for the same (seed, routing, global_tokens, experts, top-k).
Each rank materializes its slice `[rank*T,(rank+1)*T)`. Activations
are per-rank (same rank ⇒ same x on any platform), so a given global token id has
identical activation everywhere without materializing a global activation tensor.

The v1 suite uses a single routing distribution:

  * uniform   — top-k distinct experts drawn uniformly per token. The DEFAULT.
                Expected fan-out for top-k=8, 256 experts, EP8 (32 experts/rank) ≈
                8·(1 − C(224,8)/C(256,8)) ≈ 5.3 ranks/token. Load ~ Poisson.

Always publish the realized fan-out so the workload is never misread again
(`routing_stats`).
"""
from __future__ import annotations

import torch

SOURCE_ID_BITS = 32
SOURCE_ID_COLUMNS = SOURCE_ID_BITS
SOURCE_ID_CONTRACT = "bounded-sign-bit-source-v2"


def build_global_routing(
    global_tokens: int,
    experts: int,
    topk: int,
    routing: str,
    seed: int,
    *,
    token_offset: int = 0,
):
    """Return one byte-stable counter-generated routing window on CPU."""
    import workload

    indices, weights = workload.canonical_routing_rows(
        int(global_tokens),
        int(experts),
        int(topk),
        routing,
        int(seed),
        token_offset=token_offset,
    )
    return (
        torch.tensor(indices, dtype=torch.int64),
        torch.tensor(weights, dtype=torch.float32),
    )


def rank_slice(idx, weights, rank: int, tokens_per_rank: int):
    lo = rank * tokens_per_rank
    return idx[lo:lo + tokens_per_rank].contiguous(), weights[lo:lo + tokens_per_rank].contiguous()


def rank_activations(tokens: int, hidden: int, seed: int, rank: int, device,
                     dtype=torch.bfloat16):
    """Exact counter-derived inputs with a quantization-safe source-token prefix."""
    source = torch.arange(tokens, device=device, dtype=torch.int64) + rank * tokens
    return activations_for_source_ids(source, hidden, seed, dtype)


def activations_for_source_ids(source, hidden: int, seed: int, dtype=torch.bfloat16):
    """Materialize canonical activations for arbitrary global source-token IDs."""
    if hidden < SOURCE_ID_COLUMNS:
        raise ValueError(f"hidden must be at least {SOURCE_ID_COLUMNS}")
    source = source.to(torch.int64)
    column = torch.arange(hidden, device=source.device, dtype=torch.int64)
    values = (source[:, None] * 131 + column[None, :] * 17 + int(seed) * 19) % 257 - 128
    output = values.to(dtype).mul_(1 / 64)
    if bool((source < 0).any().item()) or bool((source >= (1 << SOURCE_ID_BITS)).any().item()):
        raise ValueError("source token ID is outside the bounded identity contract")
    source_columns = torch.arange(SOURCE_ID_BITS, device=source.device, dtype=torch.int64)
    source_bits = ((source[:, None] >> source_columns[None, :]) & 1) * 2 - 1
    # Magnitude one sits inside the ordinary [-2, 2] activation range, so the identity cannot set
    # an FP8 block scale. Decode depends only on sign and remains stable after dequantization.
    output[:, :SOURCE_ID_BITS] = source_bits.to(dtype)
    return output


def decode_source_ids(payload, seed: int):
    """Decode and validate source IDs carried by rank_activations."""
    if payload.ndim != 2 or payload.shape[1] < SOURCE_ID_COLUMNS:
        raise ValueError("received payload cannot carry the source-token prefix")
    prefix = payload[:, :SOURCE_ID_COLUMNS].float()
    if not bool(torch.isfinite(prefix).all().item()) or bool((prefix.abs() < 0.25).any().item()):
        raise ValueError("received source-token prefix is not quantization-stable")
    bits = prefix >= 0
    powers = 1 << torch.arange(SOURCE_ID_BITS, device=payload.device, dtype=torch.int64)
    source = (bits[:, :SOURCE_ID_BITS].to(torch.int64) * powers).sum(dim=1)
    return source


def routing_locality(idx, experts_per_rank: int, ep_size: int, tokens_per_rank: int,
                     gpus_per_node: int, scale_up_domain: int = None) -> dict:
    """Locality of rank-deduplicated payload copies under packed placement."""
    import torch as _t
    gt = idx.shape[0]
    assignments = (idx // experts_per_rank).clamp(max=ep_size - 1)
    destinations = _t.zeros((gt, ep_size), dtype=_t.bool)
    destinations.scatter_(1, assignments, True)
    token, dest = destinations.nonzero(as_tuple=True)
    src = (token // max(1, tokens_per_rank)).clamp(max=ep_size - 1)
    sud = scale_up_domain or (gpus_per_node * ep_size)                  # default: all one domain
    phys = _t.arange(ep_size, dtype=_t.int64)
    pd, ps = phys[dest], phys[src]
    local = (dest == src)
    same_node = (pd // gpus_per_node) == (ps // gpus_per_node)
    same_dom = (pd // sud) == (ps // sud)
    n = dest.numel()
    return {
        "placement": "packed",
        "local_rank_fraction": float(local.float().mean()),
        "same_node_fraction": float(same_node.float().mean()),
        "same_scaleup_domain_fraction": float(same_dom.float().mean()),
        "cross_node_fraction": float((~same_node).float().mean()),
        "cross_domain_fraction": float((~same_dom).float().mean()),
        "gpus_per_node": gpus_per_node, "scale_up_domain": sud, "copies": int(n),
    }


def routing_stats(idx, experts: int, experts_per_rank: int, weights=None) -> dict:
    """Realized routing properties for the GLOBAL trace — published per point so the
    fan-out / load can never be silently misread. idx is the global [gt, topk] tensor;
    weights the matching [gt, topk] gate weights.
    """
    ep = max(1, experts // max(1, experts_per_rank))
    ranks = (idx // experts_per_rank)                       # [gt, topk] destination rank per assignment
    # unique destination ranks per token (fan-out)
    onehot = torch.zeros(idx.shape[0], ep, dtype=torch.bool)
    onehot.scatter_(1, ranks.clamp(max=ep - 1), True)
    fanout = onehot.sum(dim=1)                              # [gt]
    hist = torch.bincount(fanout, minlength=ep + 1)[1:ep + 1].tolist()  # counts for fan-out 1..ep
    load = torch.bincount(idx.reshape(-1), minlength=experts).float()
    # Keep expert assignments (compute load) separate from rank-deduplicated payload copies
    # (network load). Conflating them overstates traffic when two experts share a rank.
    assignment_load = torch.bincount(
        ranks.reshape(-1).clamp(max=ep - 1), minlength=ep
    ).float()
    payload_load = onehot.sum(dim=0).float()
    # One-number imbalance summaries so a row is self-describing for the distribution-sensitivity
    # suite (no need to read the full histograms): CV = std/mean of the load; hotspot_ratio =
    # worst expert load over the mean.
    def _cv(t):
        m = float(t.mean())
        return float(t.std(unbiased=False) / m) if m > 0 else 0.0
    expert_load_cv = _cv(load)
    assignment_rank_cv = _cv(assignment_load)
    payload_rank_cv = _cv(payload_load)
    hotspot_ratio = float(load.max() / load.mean()) if float(load.mean()) > 0 else 0.0
    # Empty experts capture compute skew; empty destination ranks capture network skew.
    empty_expert_count = int((load == 0).sum())
    empty_rank_count = int((payload_load == 0).sum())
    return {
        "fanout_mean": float(fanout.float().mean()),
        "fanout_min": int(fanout.min()), "fanout_max": int(fanout.max()),
        "fanout_hist": hist,                               # index k-1 = #tokens with fan-out k
        "expert_assignments_per_rank": [int(x) for x in assignment_load.tolist()],
        "payload_copies_per_rank": [int(x) for x in payload_load.tolist()],
        "routed_copies": int(fanout.sum()),                # total (token, dest-rank) pairs
        "expert_load_min": int(load.min()), "expert_load_max": int(load.max()),
        "expert_load_mean": float(load.mean()), "expert_load_cv": expert_load_cv,
        "expert_assignment_rank_cv": assignment_rank_cv,
        "payload_rank_cv": payload_rank_cv, "hotspot_ratio": hotspot_ratio,
        "empty_expert_count": empty_expert_count, "empty_rank_count": empty_rank_count,
    }


# --------------------------------------------------------------------------- self-test
if __name__ == "__main__":
    import sys
    E, TOPK, EPR, GT = 256, 8, 32, 4096
    ui, _ = build_global_routing(GT, E, TOPK, "uniform", 67)
    assert all(len(set(row.tolist())) == TOPK for row in ui[:16])
    uniform = routing_stats(ui, E, EPR)
    assert uniform["hotspot_ratio"] >= 1.0
    dev = torch.device("cpu")
    first = rank_activations(8, 256, 67, 0, dev, dtype=torch.float32)
    second = rank_activations(8, 256, 67, 0, dev, dtype=torch.float32)
    assert torch.equal(first, second) and torch.isfinite(first).all()
    print("routing self-test: PASS")
    sys.exit(0)
