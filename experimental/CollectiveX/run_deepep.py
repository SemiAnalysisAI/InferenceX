#!/usr/bin/env python3
"""CollectiveX spike — DeepEP MoE dispatch+combine (normal mode), B200 first.

One decode-shaped dispatch+combine point, correctness-gated, CUDA-event timed,
emitting the same flat-JSON provenance shape as run_nccl.py.

Scope (plan §Milestone 0): normal mode only — low-latency (LL) mode is the
known-broken/blocked IBGDA path and is out of scope for the spike. B200
(x86_64) first; GB200 is the fast-follow once the aarch64 rebuild-deepep path
is proven.

  !!! DeepEP's Python API is VERSION-SENSITIVE (the plan notes V2 changed
  NVSHMEM->NCCL, unified the APIs, and removed zero-SM LL mode). The
  dispatch/combine block below follows the documented normal-mode intranode
  API and is marked "ADAPT HERE" — validate the call signatures against the
  DeepEP commit actually built by rebuild-deepep at job time, and record that
  commit in provenance. Build is done at job setup, not shipped in the image.

Launch (one process per GPU), e.g. single-node 8x B200:
    torchrun --nproc_per_node=8 run_deepep.py \\
        --runner b200-dgxc --topology-class b200-nvlink-island --transport nvlink \\
        --env-json results/env.json --out results/b200_deepep.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys

SCHEMA_VERSION = 1
MEASUREMENT_CONTRACT = "deepep-normal-v1"


def _percentile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    i = max(0, min(len(s) - 1, int(round(q / 100.0 * (len(s) - 1)))))
    return s[i]


def comparison_key(meta: dict) -> str:
    parts = [
        meta["op"], meta["backend"], meta["mode"], str(meta["world_size"]),
        str(meta["nodes"]), meta["topology_class"], meta["comparison_class"],
        meta["measurement_contract"], str(meta["shape"]),
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser(description="CollectiveX DeepEP dispatch+combine (normal mode)")
    # shape (decode-ish default from the plan)
    ap.add_argument("--tokens-per-rank", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=7168)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--dispatch-dtype", default="fp8", choices=["fp8", "bf16"])
    ap.add_argument("--routing", default="uniform", choices=["uniform", "zipf"])
    ap.add_argument("--seed", type=int, default=67)
    # measurement
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--num-sms", type=int, default=24, help="communication SMs (standardized budget)")
    # provenance
    ap.add_argument("--runner", required=True)
    ap.add_argument("--topology-class", required=True)
    ap.add_argument("--transport", default="")
    ap.add_argument("--comparison-class", default="standardized")
    ap.add_argument("--deepep-commit", default=os.environ.get("DEEPEP_COMMIT", "unknown"))
    ap.add_argument("--env-json")
    ap.add_argument("--timestamp")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # ---- imports guarded so a missing build fails loudly, not cryptically ----
    try:
        import torch
        import torch.distributed as dist
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: torch unavailable: {exc!r}", file=sys.stderr)
        return 3
    try:
        from deep_ep import Buffer  # type: ignore
    except Exception as exc:  # pragma: no cover
        print(
            "ERROR: deep_ep import failed — DeepEP must be built at job setup "
            f"(rebuild-deepep). {exc!r}",
            file=sys.stderr,
        )
        return 3

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    group = dist.group.WORLD
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(args.seed + rank)

    n = args.tokens_per_rank
    H = args.hidden
    topk = args.topk
    E = args.experts

    # Input tokens + routing. Weights sum to 1 per token so that a pure
    # dispatch->combine round trip (no expert compute) reconstructs x.
    x = torch.randn((n, H), dtype=torch.bfloat16, device=device)
    if args.routing == "uniform":
        topk_idx = torch.stack([
            torch.randperm(E, device=device)[:topk] for _ in range(n)
        ]).to(torch.int64)
    else:  # zipf-ish skew toward low expert ids
        probs = (1.0 / torch.arange(1, E + 1, device=device).float())
        topk_idx = torch.multinomial(probs.expand(n, E), topk, replacement=False).to(torch.int64)
    topk_weights = torch.softmax(torch.randn((n, topk), device=device, dtype=torch.float32), dim=-1)

    # Buffer sizing: intranode uses NVLink buffer only (no RDMA for single node).
    # Numbers follow DeepEP's intranode test guidance; tune per build.
    num_nvl_bytes = 1024 * 1024 * 1024
    num_rdma_bytes = 0
    buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)
    # Apply the standardized communication-SM budget so the recorded
    # num_comm_sms reflects the actual run (best-effort across DeepEP versions).
    try:
        Buffer.set_num_sms(args.num_sms)
    except Exception as exc:  # pragma: no cover - API/version dependent
        if rank == 0:
            print(f"WARN: could not set num_sms={args.num_sms}: {exc!r}", file=sys.stderr)

    def run_once():
        # ===================== ADAPT HERE (DeepEP API) =======================
        # Normal-mode intranode dispatch/combine. Signatures below match the
        # documented DeepEP normal API; confirm against the built commit.
        (num_tokens_per_rank, _, num_tokens_per_expert,
         is_token_in_rank, _) = buffer.get_dispatch_layout(topk_idx, E)
        recv_x, recv_topk_idx, recv_topk_weights, _, handle, _ = buffer.dispatch(
            x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_tokens_per_rank=num_tokens_per_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
        )
        combined_x, _, _ = buffer.combine(recv_x, handle, topk_weights=recv_topk_weights)
        # =====================================================================
        return combined_x, num_tokens_per_expert, is_token_in_rank

    # ---- correctness gate (run before timing; a fast wrong answer is invalid) ----
    combined_x, num_tokens_per_expert, is_token_in_rank = run_once()
    torch.cuda.synchronize()
    expected_routed = n * topk
    routed = int(torch.as_tensor(num_tokens_per_expert).sum().item())
    token_conservation = (routed == expected_routed)
    # DeepEP combine sums one copy of each token per destination RANK, so the
    # dispatch->combine round trip reconstructs x only after dividing by the
    # number of ranks each token was sent to (per DeepEP's own check in
    # tests/legacy/test_intranode.py: combined_x / is_token_in_rank.sum(dim=1)).
    ranks_per_token = is_token_in_rank.sum(dim=1, keepdim=True).clamp(min=1).float()
    check_x = combined_x.float() / ranks_per_token
    max_abs = (check_x - x.float()).abs().max().item()
    max_rel = (max_abs / (x.float().abs().max().item() + 1e-6))
    combine_ok = max_rel < 2e-2  # bf16 dispatch/combine round-trip tolerance
    correct = bool(token_conservation and combine_ok)

    # ---- timing (CUDA events; per-rank; reduce for slowest rank) ----
    def time_ms(fn, warmup, iters) -> list[float]:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        out = []
        for _ in range(iters):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            out.append(s.elapsed_time(e) * 1000.0)  # ms -> us
        return out

    def dispatch_only():
        (npr, _, npe, itir, _) = buffer.get_dispatch_layout(topk_idx, E)
        buffer.dispatch(x, topk_idx=topk_idx, topk_weights=topk_weights,
                        num_tokens_per_rank=npr, is_token_in_rank=itir,
                        num_tokens_per_expert=npe)

    trials = []
    for _ in range(args.trials):
        rt = time_ms(run_once, args.warmup, args.iters)      # dispatch+combine round trip
        dp = time_ms(dispatch_only, args.warmup, args.iters)  # dispatch only
        trials.append({
            "roundtrip_us_p50": _percentile(rt, 50), "roundtrip_us_p99": _percentile(rt, 99),
            "dispatch_us_p50": _percentile(dp, 50),
        })

    local_rt_p50 = sum(t["roundtrip_us_p50"] for t in trials) / len(trials)
    # slowest rank across the world
    t = torch.tensor([local_rt_p50], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    slowest_rank_us = float(t.item())

    if rank == 0:
        shape = {
            "tokens_per_rank": n, "hidden": H, "topk": topk, "experts": E,
            "dispatch_dtype": args.dispatch_dtype, "routing": args.routing,
            "num_comm_sms": args.num_sms,
        }
        meta = {
            "op": "dispatch-combine", "backend": "deepep", "mode": "normal",
            "world_size": world_size, "nodes": int(os.environ.get("SLURM_NNODES", "1")),
            "topology_class": args.topology_class, "comparison_class": args.comparison_class,
            "measurement_contract": MEASUREMENT_CONTRACT, "shape": shape,
        }
        tokens_total = n * world_size
        rt_p50 = sum(t["roundtrip_us_p50"] for t in trials) / len(trials)
        env = None
        if args.env_json and os.path.exists(args.env_json):
            with open(args.env_json) as _fh:
                env = json.load(_fh)
        doc = {
            "schema_version": SCHEMA_VERSION,
            "family": "moe",
            "generated_by": "run_deepep.py",
            "generated_at": args.timestamp or _dt.datetime.now().astimezone().isoformat(),
            "runner": args.runner,
            "transport": args.transport,
            "status": "valid" if correct else "invalid",
            "comparison_key": comparison_key(meta),
            "backend_provenance": {"deepep_commit": args.deepep_commit},
            **meta,
            "correctness": {
                "passed": correct, "token_conservation": token_conservation,
                "combine_within_tol": combine_ok, "max_abs_error": max_abs, "max_rel_error": max_rel,
            },
            "metrics": {
                "roundtrip_us_p50": rt_p50,
                "roundtrip_us_p99": sum(t["roundtrip_us_p99"] for t in trials) / len(trials),
                "dispatch_us_p50": sum(t["dispatch_us_p50"] for t in trials) / len(trials),
                "slowest_rank_roundtrip_us": slowest_rank_us,
                "tokens_per_second": (tokens_total / (rt_p50 * 1e-6)) if rt_p50 else None,
            },
            "trials": trials,
            "environment": env,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(doc, fh, indent=2)
            fh.write("\n")
        print(
            f"deepep dispatch-combine: status={doc['status']} "
            f"rt_p50={rt_p50:.1f}us slowest_rank={slowest_rank_us:.1f}us "
            f"correct={correct} -> {args.out}"
        )

    dist.barrier()
    dist.destroy_process_group()
    return 0 if correct else 1


if __name__ == "__main__":
    raise SystemExit(main())
