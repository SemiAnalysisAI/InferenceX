#!/usr/bin/env python3
"""CollectiveX — shared EP (expert-parallel) dispatch/combine benchmark harness.

Backend-agnostic core for the EP benchmark. The per-backend adapters
(`ep_deepep.py`, `ep_mori.py`) implement a small duck-typed protocol; this module
owns everything else: the source-tokens-per-rank sweep, the SEPARATED dispatch /
combine / round-trip timing, the correctness gate, and the provenance-tagged JSON
doc the summarizer + plotter consume.

Measurement model (see the CollectiveX EP framework notes):
  * Primary x-axis is SOURCE TOKENS PER RANK, T in {1,2,4,8,...}. One row per T.
    Only T varies along a line; everything else (backend, ep degree, phase,
    precision, top-k, experts, hidden, routing, mode, comm-SMs) is FIXED and
    identifies the line.
  * Dispatch and combine are SEPARATE measurements. The combine timing window
    contains ONLY combine(): the dispatch that produces its handle/layout (and
    the "expert outputs" staged into the combine input) runs UNTIMED. The
    round-trip is a third, distinct measurement (dispatch + combine).
  * Both x values are recorded per row — tokens_per_rank and
    global_tokens = T * ep_size — so a frontend can toggle weak-scaling (fixed
    tokens/rank) vs strong-scaling (fixed global tokens) without re-running.

stdlib-only at module top (torch is passed in by the entrypoint after a guarded
import) so this file `py_compile`s on a machine without torch.

Backend protocol (see ep_deepep.py / ep_mori.py):
    name: str                      # "deepep" | "mori"
    mode: str                      # "normal" | "ll"
    measurement_contract: str      # e.g. "deepep-normal-v1"
    combine_needs_redispatch: bool # True if combine consumes the dispatch state
    backend_provenance: dict
    buffer_cap(args) -> int|None   # max T the backend's buffers can hold (None = unbounded)
    make_problem(T) -> problem     # build x[T,H], topk_idx[T,topk], topk_weights, scales
    dispatch(problem) -> handle    # ONLY the dispatch comm op (timed for dispatch-only)
    stage(problem, handle)         # untimed: place "expert outputs" into combine input
    combine(problem, handle) -> tensor   # ONLY the combine comm op (timed for combine-only)
    expected(problem, handle) -> (tensor, n_compare)   # reference for the gate
    recv_tokens(handle) -> int     # realized tokens received this rank (comm volume)
    finalize(rc) -> int|NoReturn   # clean shutdown (mori hard-exits)
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os

SCHEMA_VERSION = 1

# Phase-default sweeps. Decode: a handful of active sequences per rank (small T).
# Prefill: a chunk of context tokens per rank (large T). Powers of two so the
# x-axis is even on a log scale. Either is overridable via --tokens-ladder; both
# get clamped to the backend's buffer ceiling (MoRI's registerable heap).
DECODE_LADDER = [1, 2, 4, 8, 16, 32, 64, 128]
PREFILL_LADDER = [128, 256, 512, 1024, 2048, 4096]

# bytes per element of the dispatch payload, for the comm-volume / algbw estimate.
_DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp8": 1}


def add_common_args(ap: argparse.ArgumentParser) -> None:
    """CLI args shared by every backend (the entrypoint adds --backend)."""
    # workload shape — FIXED params identify the line; only --tokens-ladder sweeps.
    ap.add_argument("--phase", default="decode", choices=["decode", "prefill"],
                    help="decode (small T) or prefill (large T); picks the default ladder")
    ap.add_argument("--tokens-ladder", default="",
                    help="space/comma-separated source-tokens-per-rank sweep; blank = phase default")
    ap.add_argument("--hidden", type=int, default=7168)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--experts", type=int, default=256, help="TOTAL experts (fixed across ep degrees)")
    ap.add_argument("--dispatch-dtype", default="bf16", choices=["bf16", "fp8"])
    ap.add_argument("--routing", default="balanced", choices=["balanced", "uniform", "zipf"])
    ap.add_argument("--num-comm-sms", type=int, default=24, help="standardized communication-SM budget")
    ap.add_argument("--num-ep-groups", type=int, default=1,
                    help="concurrent EP groups on the node (1 = the ordinary line; >1 is a distinct experiment)")
    ap.add_argument("--seed", type=int, default=67)
    # measurement
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    # provenance
    ap.add_argument("--runner", required=True)
    ap.add_argument("--topology-class", required=True)
    ap.add_argument("--transport", default="")
    ap.add_argument("--comparison-class", default="standardized")
    ap.add_argument("--env-json")
    ap.add_argument("--timestamp")
    ap.add_argument("--out", required=True)


def token_ladder(spec: str, phase: str, cap: int | None) -> tuple[list[int], list[int]]:
    """Return (ladder, dropped). Parse an explicit spec else the phase default;
    keep only positive ints; clamp to `cap` (backend buffer ceiling) and report
    what was dropped so truncation is never silent."""
    if spec and spec.strip():
        raw = [t.strip() for t in spec.replace(",", " ").split()]
        want = [int(t) for t in raw if t]
    else:
        want = DECODE_LADDER if phase == "decode" else PREFILL_LADDER
    want = sorted({t for t in want if t > 0})
    if cap is not None:
        kept = [t for t in want if t <= cap]
        dropped = [t for t in want if t > cap]
    else:
        kept, dropped = want, []
    return kept, dropped


def percentile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    i = max(0, min(len(s) - 1, int(round(q / 100.0 * (len(s) - 1)))))
    return s[i]


def time_us(torch, fn, warmup: int, iters: int, pre=None) -> list[float]:
    """CUDA-event timing in microseconds.

    Without `pre`: times `fn()`. With `pre`: runs `pre()` UNTIMED each iteration
    (with a sync before the start event so its GPU work cannot bleed into the
    measured window), then times `fn(pre_result)`. `pre` is how combine is
    isolated for a backend whose combine consumes the dispatch state and so needs
    a fresh dispatch+stage before every combine sample.
    """
    def sample():
        arg = None
        if pre is not None:
            arg = pre()
            torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn(arg) if pre is not None else fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) * 1000.0  # ms -> us

    for _ in range(max(0, warmup)):
        if pre is not None:
            a = pre()
            torch.cuda.synchronize()
            fn(a)
        else:
            fn()
    torch.cuda.synchronize()
    return [sample() for _ in range(iters)]


def comparison_key(meta: dict) -> str:
    """Machine key gating which rows share a curve. Built from the FIXED config
    ONLY — tokens_per_rank is the x-axis and MUST NOT be in the key, or every
    sweep point would read as a different line. ep_size, num_ep_groups, phase and
    topology-class ARE in the key, so EP4 vs EP8, decode vs prefill, and a
    concurrent-groups run are labelled distinct rather than silently overlaid."""
    parts = [
        meta["op"], meta["backend"], meta["mode"], meta["phase"],
        str(meta["ep_size"]), str(meta["num_ep_groups"]), str(meta["nodes"]),
        meta["topology_class"], meta["comparison_class"], meta["measurement_contract"],
        json.dumps(meta["shape"], sort_keys=True),
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _reduce_max(torch, dist, device, vals: list[float]) -> list[float]:
    t = torch.tensor(vals, device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return [float(x) for x in t.tolist()]


def _reduce_min_int(torch, dist, device, v: int) -> int:
    t = torch.tensor([v], device=device, dtype=torch.int64)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return int(t.item())


def run_sweep(args, backend, torch, dist, device, rank: int, world_size: int) -> int:
    """Drive the source-tokens-per-rank sweep for one fully-specified line.

    For each T: build the problem, run one untimed dispatch->stage->combine for
    the correctness gate, then take three SEPARATE timings — dispatch-only,
    combine-only (dispatch+stage untimed), and the round trip. Latencies are
    reduced MAX across ranks (a collective finishes with its slowest rank);
    correctness is reduced MIN (any rank failing fails the point). Rank 0 writes
    one JSON doc with a row per T. Returns a process exit code.
    """
    ep_size = world_size // max(1, args.num_ep_groups)
    if args.experts % ep_size != 0:
        if rank == 0:
            print(f"ERROR: experts ({args.experts}) must divide ep_size ({ep_size})")
        return 2
    experts_per_rank = args.experts // ep_size
    elem_bytes = _DTYPE_BYTES.get(args.dispatch_dtype, 2)

    cap = backend.buffer_cap(args)
    ladder, dropped = token_ladder(args.tokens_ladder, args.phase, cap)
    if rank == 0 and dropped:
        print(f"NOTE: dropped tokens/rank {dropped} — exceed {backend.name} buffer cap {cap} "
              f"(hidden={args.hidden}); not silently truncated.")
    if not ladder:
        if rank == 0:
            print(f"ERROR: empty token ladder (phase={args.phase}, cap={cap})")
        return 2
    # Some backends (MoRI) wedge on a COLD dispatch that jumps straight to a large
    # token count; they set needs_gradual_ramp so the sweep approaches its max T
    # through a geometric ramp from 1 (validated on MI355X to avoid the hang while
    # still reaching 512). A naturally-gradual ladder (decode) is unchanged.
    if getattr(backend, "needs_gradual_ramp", False):
        top, ramp, t = ladder[-1], [], 1
        while t < top:
            ramp.append(t)
            t *= 2
        ramp.append(top)
        if rank == 0 and ramp != ladder:
            print(f"NOTE: {backend.name} sweep ramped gradually 1..{top} (cold-jump-safe): {ramp}")
        ladder = ramp

    rows: list[dict] = []
    for T in ladder:
        problem = backend.make_problem(T)

        # ---- correctness gate (untimed): dispatch -> stage experts -> combine ----
        h = backend.dispatch(problem)
        backend.stage(problem, h)
        combined = backend.combine(problem, h)
        torch.cuda.synchronize()
        recv_local = backend.recv_tokens(h)
        exp, n_cmp = backend.expected(problem, h)
        got = combined[:n_cmp].float()
        max_abs = (got - exp[:n_cmp].float()).abs().max().item()
        denom = exp[:n_cmp].float().abs().max().item() + 1e-6
        max_rel = max_abs / denom
        local_ok = 1 if (max_rel < 2e-2 and recv_local > 0) else 0

        # ---- three separate timings ----
        disp = time_us(torch, lambda p=problem: backend.dispatch(p), args.warmup, args.iters)

        def prep(p=problem):
            hh = backend.dispatch(p)
            backend.stage(p, hh)
            return hh

        if backend.combine_needs_redispatch:
            comb = time_us(torch, lambda hh, p=problem: backend.combine(p, hh),
                           args.warmup, args.iters, pre=prep)
        else:
            hh = prep()
            comb = time_us(torch, lambda p=problem, hx=hh: backend.combine(p, hx),
                           args.warmup, args.iters)

        def roundtrip(p=problem):
            hh = backend.dispatch(p)
            backend.stage(p, hh)
            return backend.combine(p, hh)

        rt = time_us(torch, roundtrip, args.warmup, args.iters)

        # ---- reduce across ranks ----
        d50, d99 = percentile(disp, 50), percentile(disp, 99)
        c50, c99 = percentile(comb, 50), percentile(comb, 99)
        r50, r99 = percentile(rt, 50), percentile(rt, 99)
        (d50, d99, c50, c99, r50, r99) = _reduce_max(
            torch, dist, device, [d50, d99, c50, c99, r50, r99])
        recv = int(_reduce_max(torch, dist, device, [float(recv_local)])[0])
        global_ok = _reduce_min_int(torch, dist, device, local_ok)
        max_rel = _reduce_max(torch, dist, device, [max_rel])[0]

        global_tokens = T * ep_size
        dispatch_bytes = recv * args.hidden * elem_bytes
        # Algorithmic bandwidth: realized received payload / dispatch time. Labelled
        # "alg" (not bus) — an EP bus-bandwidth model is backend-specific and out of
        # scope; latency is the primary metric, this is a comm-volume sanity figure.
        disp_algbw = (dispatch_bytes / (d50 * 1e3)) if d50 > 0 else 0.0
        tps = (global_tokens / (r50 * 1e-6)) if r50 > 0 else None

        rows.append({
            "tokens_per_rank": T,
            "global_tokens": global_tokens,
            "dispatch_us_p50": d50, "dispatch_us_p99": d99,
            "combine_us_p50": c50, "combine_us_p99": c99,
            "roundtrip_us_p50": r50, "roundtrip_us_p99": r99,
            "recv_tokens": recv,
            "dispatch_bytes": dispatch_bytes,
            "dispatch_algbw_gbps": disp_algbw,
            "tokens_per_second": tps,
            "correct": bool(global_ok),
            "max_rel_error": max_rel,
        })
        if rank == 0:
            print(f"  T={T:<5} disp={d50:8.2f}us combine={c50:8.2f}us rt={r50:8.2f}us "
                  f"recv={recv:<6} correct={bool(global_ok)}")

    if rank != 0:
        return 0

    all_ok = bool(rows) and all(r["correct"] for r in rows)
    shape = {
        "hidden": args.hidden, "topk": args.topk, "experts": args.experts,
        "experts_per_rank": experts_per_rank, "dispatch_dtype": args.dispatch_dtype,
        "routing": args.routing, "num_comm_sms": args.num_comm_sms,
    }
    meta = {
        "op": "ep-dispatch-combine", "backend": backend.name, "mode": backend.mode,
        "phase": args.phase, "world_size": world_size, "ep_size": ep_size,
        "num_ep_groups": args.num_ep_groups,
        "nodes": int(os.environ.get("SLURM_NNODES", "1")),
        "topology_class": args.topology_class, "comparison_class": args.comparison_class,
        "measurement_contract": backend.measurement_contract, "shape": shape,
    }
    headline = next((r for r in rows if r["tokens_per_rank"] == 64), rows[len(rows) // 2])
    env = None
    if args.env_json and os.path.exists(args.env_json):
        with open(args.env_json) as fh:
            env = json.load(fh)
    doc = {
        "schema_version": SCHEMA_VERSION, "family": "moe", "generated_by": "tests/run_ep.py",
        "generated_at": args.timestamp or _dt.datetime.now().astimezone().isoformat(),
        "runner": args.runner, "transport": args.transport,
        "status": "valid" if all_ok else "invalid",
        "comparison_key": comparison_key(meta),
        "x_axis": {"primary": "tokens_per_rank",
                   "global_relation": "global_tokens = tokens_per_rank * ep_size"},
        "backend_provenance": backend.backend_provenance,
        **meta,
        "correctness": {"passed": all_ok,
                        "max_rel_error": max((r["max_rel_error"] for r in rows), default=None),
                        "points": len(rows)},
        "metrics": {
            "headline_tokens_per_rank": headline["tokens_per_rank"],
            "dispatch_us_p50": headline["dispatch_us_p50"],
            "combine_us_p50": headline["combine_us_p50"],
            "roundtrip_us_p50": headline["roundtrip_us_p50"],
            "roundtrip_us_p99": headline["roundtrip_us_p99"],
            "tokens_per_second": headline["tokens_per_second"],
        },
        "rows": rows,
        "environment": env,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")
    print(f"{backend.name} ep-dispatch-combine [{args.phase}]: status={doc['status']} "
          f"{len(rows)} points, headline T={headline['tokens_per_rank']} "
          f"disp={headline['dispatch_us_p50']:.1f}us combine={headline['combine_us_p50']:.1f}us "
          f"-> {args.out}")
    return 0 if all_ok else 1
