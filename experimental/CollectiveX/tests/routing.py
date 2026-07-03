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
  * zipf      — expert popularity proportional to 1/rank (skewed load), uniform-ish fan-out.
  * hotspot-single — expert 0 is present in every token's top-k (receive-concentration probe).

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
                         routing: str, seed: int, experts_per_rank: int, step: int = 0):
    """(idx[gt, topk] int64, weights[gt, topk] float32) on CPU — deterministic,
    independent of world/EP/platform, with experts distinct within a token. `step` is retained only
    for legacy call compatibility and must be zero."""
    if topk > experts:
        raise ValueError(f"topk ({topk}) > experts ({experts})")
    if int(step) != 0:
        raise ValueError("nonzero routing step requires a stateful trace-replay benchmark")
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
    elif routing == "hotspot-single":
        # One hot expert is in every token's top-k; the others are uniform and distinct.
        hot = 0
        others = [e for e in range(experts) if e != hot]
        others_t = torch.tensor(others, dtype=torch.int64)
        rest = torch.stack([others_t[torch.randperm(experts - 1, generator=g)[:topk - 1]]
                            for _ in range(gt)]).to(torch.int64)
        idx = torch.cat([torch.full((gt, 1), hot, dtype=torch.int64), rest], dim=1)
    else:
        raise ValueError(
            f"unknown routing '{routing}' "
            "(uniform|balanced|balanced-rank-local|zipf|hotspot-single)")
    weights = torch.softmax(torch.randn(gt, topk, generator=g), dim=1).to(torch.float32)
    return idx, weights


# Activation VALUE distributions (goal Part 2 "activation-value sensitivity"). Under bf16 combine
# these are latency-neutral (bf16 is value-independent — the ratio is ~1.0, the expected null
# result); they become latency-relevant only under a quantized combine (PR311), where amax /
# outliers / saturation drive scale computation. Kept here so the rig is ready + the value
# identity (activation_identity) is honest about which distribution was used.
ACTIVATION_PROFILES = ("normal", "zeros", "small-amplitude", "wide-dynamic-range", "fp8-saturation")
_FP8_E4M3_MAX = 448.0   # e4m3 max magnitude — fp8-saturation pushes values to/over this


def rank_slice(idx, weights, rank: int, tokens_per_rank: int):
    lo = rank * tokens_per_rank
    return idx[lo:lo + tokens_per_rank].contiguous(), weights[lo:lo + tokens_per_rank].contiguous()


def rank_activations(tokens: int, hidden: int, seed: int, rank: int, device,
                     dtype=torch.bfloat16, profile: str = "normal"):
    """Per-rank expert-input activations. Deterministic from (seed, rank) so a given global
    token has identical activation on every platform. `profile` selects the VALUE distribution
    (goal Part 2): normal N(0,1); zeros; small-amplitude (×0.01); wide-dynamic-range (heavy-tailed
    with rare large outliers); fp8-saturation (values scaled to straddle the e4m3 max so an fp8
    cast saturates). All seeded identically per rank — only the value shape changes."""
    g = _cpu_gen(int(seed) * _RANK_SUBSEED + int(rank) + 1)
    if profile == "zeros":
        x = torch.zeros(tokens, hidden, dtype=torch.float32)
    elif profile == "small-amplitude":
        x = torch.randn(tokens, hidden, generator=g, dtype=torch.float32) * 0.01
    elif profile == "wide-dynamic-range":
        # heavy-tailed: N(0,1) base with a sparse (~1%) set of large (×~250) outliers, so amax
        # per block swings widely token-to-token (the case that stresses per-block fp8 scaling).
        x = torch.randn(tokens, hidden, generator=g, dtype=torch.float32)
        spikes = (torch.rand(tokens, hidden, generator=g) < 0.01).float()
        x = x + spikes * torch.randn(tokens, hidden, generator=g, dtype=torch.float32) * 250.0
    elif profile == "fp8-saturation":
        # uniform in [-1,1] scaled to ~1.5× the e4m3 max so a naive fp8 cast clips/saturates.
        u = torch.rand(tokens, hidden, generator=g, dtype=torch.float32) * 2.0 - 1.0
        x = u * (_FP8_E4M3_MAX * 1.5)
    elif profile == "normal":
        x = torch.randn(tokens, hidden, generator=g, dtype=torch.float32)
    else:
        raise ValueError(f"unknown activation profile '{profile}' (one of {ACTIVATION_PROFILES})")
    return x.to(device=device, dtype=dtype)


def placement_perm(ep_size: int, gpus_per_node: int, placement: str) -> list:
    """phys[logical_rank] -> physical slot, per placement kind (goal Part 2 placement matrix).
    The physical slot's node = slot // gpus_per_node, domain = slot // scale_up_domain. Single
    node (ep <= gpus_per_node) makes every placement identical (everything is same-node).

      packed         identity — fill one node/domain before crossing (latency-oriented default).
      runtime-native identity for now — reproduces the serving placement (link via recipe meta).
      striped        round-robin logical ranks across nodes (exposes inter-node transport).
      adversarial    a deterministic scatter that maximizes cross-node/-domain copies.
    """
    n = ep_size
    if gpus_per_node <= 0 or gpus_per_node >= n or placement in ("packed", "runtime-native"):
        return list(range(n))
    nodes = (n + gpus_per_node - 1) // gpus_per_node
    if placement == "striped":
        # logical r -> node (r % nodes), intra-node slot (r // nodes): spreads neighbors apart.
        return [min(n - 1, (r % nodes) * gpus_per_node + (r // nodes)) for r in range(n)]
    if placement == "adversarial":
        # reverse within the rank space, then stripe — pushes a rank's neighbors to far nodes.
        return [min(n - 1, ((n - 1 - r) % nodes) * gpus_per_node + ((n - 1 - r) // nodes))
                for r in range(n)]
    return list(range(n))


def routing_locality(idx, experts_per_rank: int, ep_size: int, tokens_per_rank: int,
                     gpus_per_node: int, scale_up_domain: int = None,
                     placement: str = "packed") -> dict:
    """Locality of the routed (token, dest-rank) copies (goal Part 2 topology section).
    A token's SOURCE rank is global_id // tokens_per_rank; its DEST ranks are idx // epr. The
    PLACEMENT maps each logical rank to a physical slot, so node/domain membership — and thus the
    same-node / same-domain / cross-* fractions — depend on packed vs striped vs adversarial."""
    import torch as _t
    gt = idx.shape[0]
    dest = (idx // experts_per_rank).clamp(max=ep_size - 1)             # [gt, topk] dest logical rank
    src = (_t.arange(gt) // max(1, tokens_per_rank)).clamp(max=ep_size - 1).unsqueeze(1)
    src = src.expand_as(dest)
    sud = scale_up_domain or (gpus_per_node * ep_size)                  # default: all one domain
    # physical slot of each logical rank, per placement -> node / domain it lives in.
    perm = placement_perm(ep_size, gpus_per_node, placement)
    phys = _t.tensor(perm, dtype=_t.int64)
    pd, ps = phys[dest], phys[src]
    local = (dest == src)
    same_node = (pd // gpus_per_node) == (ps // gpus_per_node)
    same_dom = (pd // sud) == (ps // sud)
    n = dest.numel()
    return {
        "placement": placement,
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
    rank_load_t = torch.bincount(ranks.reshape(-1).clamp(max=ep - 1), minlength=ep).float()
    rank_load = [int(x) for x in rank_load_t.tolist()]
    # One-number imbalance summaries so a row is self-describing for the distribution-sensitivity
    # suite (no need to read the full histograms): CV = std/mean of the load; hotspot_ratio =
    # worst expert load over the mean. uniform -> CV≈0, hotspot_ratio≈1; zipf / hotspot-single ->
    # high CV and hotspot_ratio (≫1). Population std (unbiased=False) over the full realized trace.
    def _cv(t):
        m = float(t.mean())
        return float(t.std(unbiased=False) / m) if m > 0 else 0.0
    expert_load_cv = _cv(load)
    rank_load_cv = _cv(rank_load_t)
    hotspot_ratio = float(load.max() / load.mean()) if float(load.mean()) > 0 else 0.0
    # Empty-expert / empty-rank counts (goal P2 "report full load and fanout statistics"):
    # how many experts/dest-ranks received ZERO token-copies (the dark side of skew — idle
    # units while the hot rank stalls). dest-rank load max/mean make the rank histogram
    # self-describing without re-reading rank_load_hist.
    empty_expert_count = int((load == 0).sum())
    empty_rank_count = int((rank_load_t == 0).sum())
    dest_rank_load_max = int(rank_load_t.max())
    dest_rank_load_mean = float(rank_load_t.mean())
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
        "expert_load_mean": float(load.mean()), "expert_load_cv": expert_load_cv,
        "rank_load_cv": rank_load_cv, "hotspot_ratio": hotspot_ratio,
        "dest_rank_load_max": dest_rank_load_max, "dest_rank_load_mean": dest_rank_load_mean,
        "empty_expert_count": empty_expert_count, "empty_rank_count": empty_rank_count,
        "routing_hash": routing_hash, "idx_hash": idx_hash, "weights_hash": w_hash,
    }


# --------------------------------------------------------------------------- self-test
if __name__ == "__main__":  # needs torch; verifies routing stats and value profiles
    import sys
    E, TOPK, EPR, GT = 256, 8, 32, 4096
    # (1) static hotspot pins expert zero and keeps every token's top-k distinct.
    si, _ = build_global_routing(GT, E, TOPK, "hotspot-single", 67, EPR)
    assert (si[:, 0] == 0).all(), "hotspot-single must pin expert 0 on every step"
    assert all(len(set(r.tolist())) == TOPK for r in si[:16]), "hotspot top-k must stay distinct"
    # (2) uniform has low concentration while hotspot is visibly concentrated.
    su = routing_stats(build_global_routing(GT, E, TOPK, "uniform", 67, EPR)[0], E, EPR)
    sh = routing_stats(si, E, EPR)
    assert su["hotspot_ratio"] < 1.5 and sh["hotspot_ratio"] > 5, "hotspot_ratio must separate uniform/hotspot"
    assert sh["empty_expert_count"] >= 0 and "empty_rank_count" in sh and "dest_rank_load_max" in sh
    print(f"routing stats OK (uniform hotspot_ratio={su['hotspot_ratio']:.2f} "
          f"hotspot empty_experts={sh['empty_expert_count']} dest_rank_max={sh['dest_rank_load_max']})")
    # (3) value profiles: distinct value shapes, all finite, fp8-saturation exceeds e4m3 max.
    dev = torch.device("cpu")
    z = rank_activations(8, 256, 67, 0, dev, dtype=torch.float32, profile="zeros")
    assert float(z.abs().max()) == 0.0, "zeros profile must be all-zero"
    sat = rank_activations(8, 256, 67, 0, dev, dtype=torch.float32, profile="fp8-saturation")
    assert float(sat.abs().max()) > _FP8_E4M3_MAX, "fp8-saturation must exceed e4m3 max"
    sm = rank_activations(8, 256, 67, 0, dev, dtype=torch.float32, profile="small-amplitude")
    assert float(sm.abs().max()) < 1.0, "small-amplitude must be tiny"
    for prof in ACTIVATION_PROFILES:
        v = rank_activations(8, 256, 67, 0, dev, dtype=torch.float32, profile=prof)
        assert torch.isfinite(v).all(), f"{prof} produced non-finite values"
    print(f"activation profiles OK ({', '.join(ACTIVATION_PROFILES)})")
    print("routing self-test: PASS")
    sys.exit(0)
