#!/usr/bin/env python3
"""CollectiveX — EPLB (Expert-Parallel Load Balancer), the DeepSeek-style remedy for
skewed (zipf) expert load.

Under skewed routing, the ranks hosting hot logical experts receive far more token-copies
than the rest; dispatch/combine latency is gated by that busiest rank (the cross-rank MAX
the harness measures), so the whole collective stalls on it. EPLB REPLICATES hot experts
onto extra physical slots and PLACES the slots so every rank carries ~equal load.

This module is backend-agnostic: it is purely a transform of the deterministic routing
trace. The trick that keeps every adapter unchanged — DeepEP/MoRI both route expert i to
rank `i // experts_per_rank` (contiguous block placement) — is to number the physical slots
RANK-MAJOR (rank r owns physical ids [r*spp, (r+1)*spp)), so the standard contiguous mapping
reproduces EPLB's balanced placement. The harness then runs with `experts = num_physical`
and the remapped (physical) trace; nothing else changes.

  num_physical = num_logical + redundant   (redundant rounded up to a multiple of ep_size)
  build_plan(): greedy replicate-by-load + equal-cardinality balanced packing onto ep_size ranks
  remap_idx():  each token's logical targets -> physical replicas, spread by global token id

Pure-Python planner (no torch) so it unit-tests on a login node; remap_idx needs torch.
"""
from __future__ import annotations

import hashlib
import json


def physical_count(num_logical: int, num_redundant: int, ep_size: int) -> int:
    """num_logical + redundant, with redundant rounded UP to a multiple of ep_size so the
    physical experts divide evenly across ranks (symmetric dispatch)."""
    r = ((max(0, num_redundant) + ep_size - 1) // ep_size) * ep_size
    return num_logical + r


def _contiguous_rank_load(logical_load, ep_size):
    """Per-rank received load WITHOUT EPLB: logical experts placed contiguously
    (experts_per_rank = num_logical/ep_size), so rank r carries its block's total."""
    n = len(logical_load)
    per = n // ep_size
    return [sum(logical_load[r * per:(r + 1) * per]) for r in range(ep_size)]


def build_plan(logical_load, num_physical: int, ep_size: int) -> dict:
    """logical_load: list[float] length num_logical (token-copies per logical expert).
    Returns the replication+placement plan (all pure-Python lists) + before/after balance."""
    num_logical = len(logical_load)
    assert num_physical >= num_logical, "num_physical must be >= num_logical"
    assert num_physical % ep_size == 0, "num_physical must divide ep_size"
    assert num_logical % ep_size == 0, "num_logical must divide ep_size"
    spp = num_physical // ep_size                      # physical slots per rank (fixed)

    # 1) Replica allocation — start one slot per logical expert, then hand each redundant
    #    slot to the expert with the highest CURRENT per-replica load (greedy min-max).
    replicas = [1] * num_logical
    for _ in range(num_physical - num_logical):
        best, best_lps = 0, -1.0
        for e in range(num_logical):
            lps = logical_load[e] / replicas[e]
            if lps > best_lps:
                best, best_lps = e, lps
        replicas[best] += 1

    # 2) Slots = (per-replica load, logical expert), one per replica.
    slots = []
    for e in range(num_logical):
        lps = logical_load[e] / replicas[e]
        slots.extend((lps, e) for _ in range(replicas[e]))

    # 3) Balanced packing into ep_size bins of EQUAL cardinality (spp each), minimizing the
    #    max per-rank load: heaviest slot first -> least-loaded rank that still has capacity.
    slots.sort(reverse=True)
    rank_slots = [[] for _ in range(ep_size)]
    rank_load = [0.0] * ep_size
    for lps, e in slots:
        r = min((r for r in range(ep_size) if len(rank_slots[r]) < spp),
                key=lambda r: rank_load[r])
        rank_slots[r].append(e)
        rank_load[r] += lps

    # 4) Rank-major physical numbering -> contiguous placement == this balanced placement.
    phys2log, rank_of_phys = [], []
    for r in range(ep_size):
        for e in rank_slots[r]:
            phys2log.append(e)
            rank_of_phys.append(r)
    log2phys = [[] for _ in range(num_logical)]
    for pid, e in enumerate(phys2log):
        log2phys[e].append(pid)

    before = _contiguous_rank_load(logical_load, ep_size)
    total = sum(logical_load) or 1.0
    mean = total / ep_size
    return {
        "num_logical": num_logical, "num_physical": num_physical, "ep_size": ep_size,
        "slots_per_rank": spp, "replicas": replicas, "max_replicas": max(replicas),
        "phys2log": phys2log, "rank_of_phys": rank_of_phys, "log2phys": log2phys,
        "rank_load_after": rank_load, "rank_load_before": before,
        # imbalance = busiest rank / mean (1.0 = perfect). This is the number EPLB cuts.
        "imbalance_before": max(before) / mean, "imbalance_after": max(rank_load) / mean,
        "replicated_experts": sum(1 for r in replicas if r > 1),
    }


def mapping_hash(plan: dict) -> str:
    """Hash the placement fields that fully determine the logical-to-physical remap."""
    payload = {
        "phys2log": plan["phys2log"],
        "rank_of_phys": plan["rank_of_phys"],
        "replicas": plan["replicas"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def remap_rows(indices: list[list[int]], plan: dict) -> list[list[int]]:
    """Pure-Python equivalent of remap_idx for contract verification."""
    replicas = plan["log2phys"]
    return [
        [replicas[expert][token % len(replicas[expert])] for expert in row]
        for token, row in enumerate(indices)
    ]


def remap_idx(idx_logical, plan):
    """idx_logical: torch [gt, topk] int64 logical-expert ids (global trace).
    Returns idx_physical [gt, topk]: each token's logical target -> one of that expert's
    physical replicas, SPREAD by global token id (row) so a hot expert's tokens fan out
    across its replicas (= across ranks). Replicas of distinct logical experts are disjoint,
    so a token's top-k physical ids stay distinct (dispatch invariant preserved)."""
    import torch
    replicas = plan["replicas"]
    num_logical = len(replicas)
    max_rc = plan["max_replicas"]
    rc = torch.tensor(replicas, dtype=torch.int64)
    # padded [num_logical, max_rc] table of physical ids (pad with replica 0; never indexed
    # past rc[e] because the replica index is taken mod rc[e]).
    padded = torch.zeros(num_logical, max_rc, dtype=torch.int64)
    for e, phys in enumerate(plan["log2phys"]):
        for k in range(max_rc):
            padded[e, k] = phys[k] if k < len(phys) else phys[0]
    gt = idx_logical.shape[0]
    rows = torch.arange(gt, dtype=torch.int64).unsqueeze(1)     # [gt,1] global token id
    e = idx_logical.to(torch.int64)                             # [gt,topk]
    ridx = rows % rc[e]                                         # [gt,topk] replica index
    return padded[e, ridx]                                      # [gt,topk] physical ids


# --------------------------------------------------------------------------- self-test
if __name__ == "__main__":
    # Synthetic zipf load (popularity ∝ 1/(e+1)) — the case EPLB targets. No torch needed.
    import sys
    NUM_LOGICAL, EP, REDUNDANT = 256, 8, 32
    load = [1.0 / (e + 1) for e in range(NUM_LOGICAL)]
    nphys = physical_count(NUM_LOGICAL, REDUNDANT, EP)
    plan = build_plan(load, nphys, EP)
    print(f"num_logical={NUM_LOGICAL} ep={EP} num_physical={nphys} slots/rank={plan['slots_per_rank']}")
    print(f"replicated experts={plan['replicated_experts']} max_replicas={plan['max_replicas']} "
          f"(hottest expert 0 replicas={plan['replicas'][0]})")
    print(f"per-rank load BEFORE (contiguous): {[round(x,3) for x in plan['rank_load_before']]}")
    print(f"per-rank load AFTER  (EPLB):       {[round(x,3) for x in plan['rank_load_after']]}")
    print(f"imbalance (max/mean)  BEFORE={plan['imbalance_before']:.2f}x  AFTER={plan['imbalance_after']:.2f}x")
    # Gates: equal slot cardinality, every logical expert placed, big imbalance cut.
    assert all(plan["replicas"][e] >= 1 for e in range(NUM_LOGICAL))
    assert sum(plan["replicas"]) == nphys
    assert len(plan["phys2log"]) == nphys
    assert all(len(plan["log2phys"][e]) == plan["replicas"][e] for e in range(NUM_LOGICAL))
    # rank-major numbering => contiguous block per rank => rank_of_phys is non-decreasing
    assert plan["rank_of_phys"] == sorted(plan["rank_of_phys"])
    assert plan["imbalance_after"] < plan["imbalance_before"], "EPLB must reduce imbalance"
    assert plan["imbalance_after"] < 1.30, f"EPLB should get within ~30% of perfect, got {plan['imbalance_after']:.2f}"
    # remap (if torch present): distinctness + balanced receive on a sampled zipf trace.
    try:
        import torch
        g = torch.Generator().manual_seed(0)
        p = torch.tensor(load)
        p = (p / p.sum()).expand(4096, NUM_LOGICAL)
        idx_l = torch.multinomial(p, 8, replacement=False, generator=g).to(torch.int64)
        idx_p = remap_idx(idx_l, plan)
        assert idx_p.shape == idx_l.shape
        # top-k physical ids distinct per token
        assert all(len(set(row.tolist())) == 8 for row in idx_p), "physical top-k must stay distinct"
        spp = plan["slots_per_rank"]
        recv_before = [0] * EP
        recv_after = [0] * EP
        per_log = NUM_LOGICAL // EP
        for row_l, row_p in zip(idx_l.tolist(), idx_p.tolist()):
            for e in row_l:
                recv_before[e // per_log] += 1
            for pid in row_p:
                recv_after[pid // spp] += 1
        ib = max(recv_before) / (sum(recv_before) / EP)
        ia = max(recv_after) / (sum(recv_after) / EP)
        print(f"sampled-trace receive imbalance BEFORE={ib:.2f}x  AFTER={ia:.2f}x")
        assert ia < ib and ia < 1.35, "remap must balance per-rank receive load"
        print("remap self-test: OK")
    except ImportError:
        print("(torch absent — skipped remap self-test; planner gates passed)")
    print("EPLB self-test: PASS")
    sys.exit(0)
