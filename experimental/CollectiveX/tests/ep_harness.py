#!/usr/bin/env python3
"""CollectiveX — shared EP (expert-parallel) dispatch/combine benchmark harness.

Backend-agnostic core. The per-backend adapters (`ep_deepep.py`, `ep_mori.py`)
implement a small duck-typed protocol; this module owns the source-tokens-per-rank
sweep, the timing, the correctness gate, and the provenance-tagged JSON doc.

Fair-comparison contract (hardened after review — see notes.md / plan.md):
  * **Deterministic shared routing trace** (`routing.py`): the per-token expert IDs +
    gate weights are generated once from a fixed seed over the *global* batch and are
    identical on every SKU; each rank materializes its slice. So every platform runs
    the *same* problem (no per-rank/per-platform RNG in the adapters).
  * **Communication-only timing**: dispatch and combine are each timed as pure comm
    with all staging (expert-output placement) done UNTIMED; round-trip is the SUM of
    the two comm-only medians (no mixed timed region), so backend-specific staging
    never enters a timed window. `measurement_contract = "comm-only-v1"`.
  * **Correct collective percentile**: each iteration's latency is reduced MAX across
    ranks first (a collective finishes with its slowest rank), THEN percentiled —
    `median_i(max_r)`, not `max_r(median_i)`.
  * **One line = one fixed config**; only T varies. Both `tokens_per_rank` and
    `global_tokens = T * ep_size` are recorded for the weak/strong-scaling x toggle.

stdlib-only at module top (torch is passed in by the entrypoint; `routing` is imported
lazily inside run_sweep) so this file `py_compile`s without torch.

Backend protocol:
    name, mode, combine_needs_redispatch, backend_provenance(dict)
    buffer_cap(args) -> int|None
    make_problem(T, idx, weights, x) -> problem   # materialize this rank's trace slice
    dispatch(problem) -> handle                   # pure dispatch comm (timed)
    stage(problem, handle)                        # untimed expert-output placement
    combine(problem, handle) -> tensor            # pure combine comm (timed)
    expected(problem, handle) -> (tensor, n_cmp)  # correctness reference
    recv_tokens(handle) -> int                    # realized tokens received this rank
    finalize(rc) -> int|NoReturn
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os

SCHEMA_VERSION = 2  # bumped: comm-only contract, deterministic trace, corrected percentile

# Phase-default sweeps — token-size regimes, NOT distinct kernels (both run normal
# mode; "decode"/"prefill" name the small/large-token regime). Powers of two for a
# clean log x-axis; clamped to the backend buffer ceiling (MoRI's registerable heap).
DECODE_LADDER = [1, 2, 4, 8, 16, 32, 64, 128]
PREFILL_LADDER = [128, 256, 512, 1024, 2048, 4096]

_DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp8": 1}


def add_common_args(ap: argparse.ArgumentParser) -> None:
    """CLI args shared by every backend (the entrypoint adds --backend)."""
    ap.add_argument("--phase", default="decode", choices=["decode", "prefill"],
                    help="token-size regime: decode (small T) / prefill (large T) — picks the default ladder")
    ap.add_argument("--tokens-ladder", default="",
                    help="space/comma-separated source-tokens-per-rank sweep; blank = phase default")
    ap.add_argument("--hidden", type=int, default=7168)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--experts", type=int, default=256, help="TOTAL experts (fixed across EP degrees)")
    ap.add_argument("--dispatch-dtype", default="bf16", choices=["bf16", "fp8"])
    # uniform = realistic top-k (fan-out ≈5.3 over EP8); balanced = load-equalized,
    # one-expert-per-rank (fan-out = ep_size); balanced-rank-local = fan-out 1 (min
    # comm) edge case; zipf = skewed. Default to the REALISTIC one.
    ap.add_argument("--routing", default="uniform",
                    choices=["uniform", "balanced", "balanced-rank-local", "zipf"])
    ap.add_argument("--mode", default="normal", choices=["normal", "ll"],
                    help="kernel path: normal or low-latency (LL); LL is backend-dependent")
    ap.add_argument("--num-sms", type=int, default=24,
                    help="communication-SM budget for DeepEP (recorded as the actual budget; MoRI uses block_num/warps)")
    ap.add_argument("--seed", type=int, default=67)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200, help=">=100 so p99 is meaningful")
    ap.add_argument("--allow-unknown-provenance", action="store_true",
                    help="permit a run with unpinned backend commit/version (default: fail)")
    # provenance / output
    ap.add_argument("--runner", required=True)
    ap.add_argument("--topology-class", required=True)
    ap.add_argument("--transport", default="")
    ap.add_argument("--comparison-class", default="standardized")
    ap.add_argument("--env-json")
    ap.add_argument("--timestamp")
    ap.add_argument("--out", required=True)


def token_ladder(spec: str, phase: str, cap: int | None) -> tuple[list[int], list[int]]:
    """Return (ladder, dropped): explicit spec else the phase default; positive ints;
    clamped to `cap` with dropped points reported (never silently truncated)."""
    if spec and spec.strip():
        want = [int(t) for t in spec.replace(",", " ").split() if t]
    else:
        want = DECODE_LADDER if phase == "decode" else PREFILL_LADDER
    want = sorted({t for t in want if t > 0})
    if cap is not None:
        return [t for t in want if t <= cap], [t for t in want if t > cap]
    return want, []


def percentile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    i = max(0, min(len(s) - 1, int(round(q / 100.0 * (len(s) - 1)))))
    return s[i]


def time_us(torch, fn, warmup: int, iters: int, pre=None) -> list[float]:
    """Per-iteration CUDA-event latencies (µs) for THIS rank.

    Without `pre`: times `fn()`. With `pre`: runs `pre()` UNTIMED each iteration (sync
    before the start event so its GPU work can't bleed in), then times `fn(pre_result)`
    — how combine is isolated when it consumes the dispatch state and needs a fresh
    untimed dispatch+stage before every sample. Returns the raw per-iteration series;
    the caller reduces across ranks per iteration before percentiling.
    """
    def sample():
        arg = pre() if pre is not None else None
        if pre is not None:
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
            a = pre(); torch.cuda.synchronize(); fn(a)
        else:
            fn()
    torch.cuda.synchronize()
    return [sample() for _ in range(iters)]


def comparison_key(meta: dict) -> str:
    """Machine key gating which rows share a curve — built from the FIXED config ONLY
    (tokens_per_rank is the x-axis and is excluded). op/backend/mode/phase/ep_size/
    topology are in the key, so EP4 vs EP8, normal vs LL, decode vs prefill, and
    different SKUs are labelled distinct, never silently overlaid."""
    parts = [
        meta["op"], meta["backend"], meta["mode"], meta["phase"],
        str(meta["ep_size"]), str(meta["nodes"]),
        meta["topology_class"], meta["comparison_class"], meta["measurement_contract"],
        json.dumps(meta["shape"], sort_keys=True),
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _reduce_vec(torch, dist, device, vals, op):
    t = torch.tensor(vals, device=device, dtype=torch.float64)
    dist.all_reduce(t, op=op)
    return [float(x) for x in t.tolist()]


def _reduce_int(torch, dist, device, v: int, op) -> int:
    t = torch.tensor([int(v)], device=device, dtype=torch.int64)
    dist.all_reduce(t, op=op)
    return int(t.item())


def _provenance_unknown(prov: dict) -> list[str]:
    return [k for k, v in prov.items() if isinstance(v, str) and v.strip().lower() == "unknown"]


def run_sweep(args, backend, torch, dist, device, rank: int, world_size: int) -> int:
    """Drive the source-tokens-per-rank sweep for one fully-specified line."""
    import routing  # torch-based; imported lazily so the module byte-compiles without torch

    ep_size = world_size  # num_ep_groups removed (was metadata-only; no real subgroups)
    if args.experts % ep_size != 0:
        if rank == 0:
            print(f"ERROR: experts ({args.experts}) must divide ep_size ({ep_size})")
        return 2
    experts_per_rank = args.experts // ep_size
    elem_bytes = _DTYPE_BYTES.get(args.dispatch_dtype, 2)

    # Provenance gate (review #1): refuse a comparison run with unpinned backend info.
    unknown = _provenance_unknown(backend.backend_provenance)
    if unknown and not args.allow_unknown_provenance:
        if rank == 0:
            print(f"ERROR: unpinned provenance {unknown} in {backend.backend_provenance}; "
                  f"set the commit/version env or pass --allow-unknown-provenance.")
        return 4

    cap = backend.buffer_cap(args)
    ladder, dropped = token_ladder(args.tokens_ladder, args.phase, cap)
    if rank == 0 and dropped:
        print(f"NOTE: dropped tokens/rank {dropped} — exceed {backend.name} buffer cap {cap} "
              f"(hidden={args.hidden}); not silently truncated.")
    if not ladder:
        if rank == 0:
            print(f"ERROR: empty token ladder (phase={args.phase}, cap={cap})")
        return 2
    # MoRI wedges on a COLD dispatch that jumps straight to a large T; it sets
    # needs_gradual_ramp so the sweep approaches its max T via a geometric ramp from 1
    # (validated on MI355X). A naturally-gradual ladder (decode) is unchanged.
    if getattr(backend, "needs_gradual_ramp", False):
        top, ramp, t = ladder[-1], [], 1
        while t < top:
            ramp.append(t); t *= 2
        ramp.append(top)
        if rank == 0 and ramp != ladder:
            print(f"NOTE: {backend.name} sweep ramped gradually 1..{top} (cold-jump-safe): {ramp}")
        ladder = ramp

    MAX, MIN, SUM = dist.ReduceOp.MAX, dist.ReduceOp.MIN, dist.ReduceOp.SUM

    # Fabric/clock warm-up BEFORE any timed point (review: H200 had an anomalous cold
    # first point and a 40% decode-vs-prefill mismatch at the shared T=128). Gradually
    # ramp through the small ladder shapes untimed — warms clocks/fabric for everyone
    # and is also cold-jump-safe for MoRI.
    warm_T = min(ladder[-1], 128)
    for wt in [t for t in ladder if t <= warm_T] or [ladder[0]]:
        wi, ww = routing.build_global_routing(wt * ep_size, args.experts, args.topk,
                                              args.routing, args.seed, experts_per_rank)
        wsi, wsw = routing.rank_slice(wi, ww, rank, wt)
        wx = routing.rank_activations(wt, args.hidden, args.seed, rank, device, torch.bfloat16)
        wp = backend.make_problem(wt, wsi.to(device), wsw.to(device), wx)
        for _ in range(8):
            wh = backend.dispatch(wp); backend.stage(wp, wh); backend.combine(wp, wh)
    torch.cuda.synchronize()
    try:
        dist.barrier()
    except Exception:
        pass

    rows: list[dict] = []
    for T in ladder:
        gt = T * ep_size
        idx_g, w_g = routing.build_global_routing(gt, args.experts, args.topk, args.routing,
                                                  args.seed, experts_per_rank)
        rstats = routing.routing_stats(idx_g, args.experts, experts_per_rank)
        idx_s, w_s = routing.rank_slice(idx_g, w_g, rank, T)
        x = routing.rank_activations(T, args.hidden, args.seed, rank, device, torch.bfloat16)
        problem = backend.make_problem(T, idx_s.to(device), w_s.to(device), x)

        # ---- correctness gate (untimed): dispatch -> stage -> combine ----
        h = backend.dispatch(problem)
        backend.stage(problem, h)
        combined = backend.combine(problem, h)
        torch.cuda.synchronize()
        recv_local = backend.recv_tokens(h)
        exp, n_cmp = backend.expected(problem, h)
        max_abs = (combined[:n_cmp].float() - exp[:n_cmp].float()).abs().max().item()
        denom = exp[:n_cmp].float().abs().max().item() + 1e-6
        max_rel = max_abs / denom
        # Correctness = this rank's OWN tokens reconstruct (combine round-trip). A rank
        # may legitimately RECEIVE 0 tokens at small T under balanced routing (not every
        # rank is a destination), so recv==0 is NOT a per-rank failure — only the GLOBAL
        # total recv must be > 0 (gated below), to catch a truly silent no-op.
        local_ok = 1 if max_rel < 5e-2 else 0

        # ---- comm-only timing: dispatch-only + combine-only (staging untimed) ----
        disp_iters = time_us(torch, lambda p=problem: backend.dispatch(p), args.warmup, args.iters)

        def prep(p=problem):
            hh = backend.dispatch(p)
            backend.stage(p, hh)
            return hh

        if backend.combine_needs_redispatch:
            comb_iters = time_us(torch, lambda hh, p=problem: backend.combine(p, hh),
                                 args.warmup, args.iters, pre=prep)
        else:
            hh = prep()
            comb_iters = time_us(torch, lambda p=problem, hx=hh: backend.combine(p, hx),
                                 args.warmup, args.iters)

        # ---- per-iteration cross-rank MAX, THEN percentile  (= median_i(max_r)) ----
        d_iter = _reduce_vec(torch, dist, device, disp_iters, MAX)
        c_iter = _reduce_vec(torch, dist, device, comb_iters, MAX)
        d50, d99 = percentile(d_iter, 50), percentile(d_iter, 99)
        c50, c99 = percentile(c_iter, 50), percentile(c_iter, 99)
        # SERIAL dispatch+combine = the SUM of the two separately-measured comm-only
        # medians. NOT an independently-measured chained op: it cannot reveal shared
        # sync cost, launch amortization, or dispatch/combine overlap. Named honestly.
        s50, s99 = d50 + c50, d99 + c99

        # ---- realized comm volume (from the known trace) + recv distribution ----
        recv_total = _reduce_int(torch, dist, device, recv_local, SUM)
        recv_max = _reduce_int(torch, dist, device, recv_local, MAX)
        recv_min = _reduce_int(torch, dist, device, recv_local, MIN)
        global_ok = _reduce_int(torch, dist, device, local_ok, MIN)
        max_rel = _reduce_vec(torch, dist, device, [max_rel], MAX)[0]
        point_ok = bool(global_ok) and recv_total > 0  # reconstruct on all ranks + non-silent

        routed_bytes_total = recv_total * args.hidden * elem_bytes  # all ranks, one direction
        # Algorithmic bandwidth: total routed payload across ranks / collective latency.
        # Payload-only (excludes indices/weights/scales); serial-RT moves it ~twice.
        disp_algbw = (routed_bytes_total / (d50 * 1e3)) if d50 > 0 else 0.0
        serial_algbw = (2 * routed_bytes_total / (s50 * 1e3)) if s50 > 0 else 0.0
        # tokens/s is throughput at THIS global-token count — only compare across
        # configs at a MATCHED global_tokens (the global-tokens x-axis), not equal T.
        tps = (gt / (s50 * 1e-6)) if s50 > 0 else None

        rows.append({
            "tokens_per_rank": T, "global_tokens": gt,
            "dispatch_us_p50": d50, "dispatch_us_p99": d99,
            "combine_us_p50": c50, "combine_us_p99": c99,
            "serial_us_p50": s50, "serial_us_p99": s99,  # = dispatch + combine (sum, not chained)
            "recv_tokens_max": recv_max, "recv_tokens_min": recv_min,
            "recv_tokens_mean": recv_total / world_size, "recv_tokens_total": recv_total,
            "routed_bytes_total": routed_bytes_total,
            "dispatch_algbw_gbps": disp_algbw, "serial_algbw_gbps": serial_algbw,
            "tokens_per_second": tps,
            # realized routing properties (published so fan-out is never misread):
            "fanout_mean": rstats["fanout_mean"], "fanout_max": rstats["fanout_max"],
            "routed_copies": rstats["routed_copies"], "expert_load_max": rstats["expert_load_max"],
            "routing_hash": rstats["routing_hash"],
            "correct": point_ok, "max_rel_error": max_rel,
        })
        if rank == 0:
            print(f"  T={T:<5} disp={d50:8.2f}us combine={c50:8.2f}us serial={s50:8.2f}us "
                  f"fanout={rstats['fanout_mean']:.2f} recv[min/mean/max]="
                  f"{recv_min}/{recv_total // world_size}/{recv_max} correct={point_ok}")

    if rank != 0:
        return 0

    all_ok = bool(rows) and all(r["correct"] for r in rows)
    shape = {  # FIXED line identity (no T, no per-backend resource knobs)
        "hidden": args.hidden, "topk": args.topk, "experts": args.experts,
        "experts_per_rank": experts_per_rank, "dispatch_dtype": args.dispatch_dtype,
        "routing": args.routing,
    }
    meta = {
        "op": "ep-dispatch-combine", "backend": backend.name, "mode": args.mode,
        "phase": args.phase, "world_size": world_size, "ep_size": ep_size,
        "nodes": int(os.environ.get("SLURM_NNODES", "1")),
        "topology_class": args.topology_class, "comparison_class": args.comparison_class,
        "measurement_contract": "comm-only-v1", "shape": shape,
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
                        "tolerance": 5e-2, "points": len(rows)},
        "routing_profile": {  # realized fan-out for the whole sweep (so it can't be misread)
            "routing": args.routing,
            "fanout_mean": sum(r["fanout_mean"] for r in rows) / len(rows),
            "fanout_max": max(r["fanout_max"] for r in rows),
            "headline_hash": headline["routing_hash"],
        },
        "metrics": {
            "headline_tokens_per_rank": headline["tokens_per_rank"],
            "dispatch_us_p50": headline["dispatch_us_p50"],
            "combine_us_p50": headline["combine_us_p50"],
            "serial_us_p50": headline["serial_us_p50"],
            "serial_us_p99": headline["serial_us_p99"],
            "tokens_per_second": headline["tokens_per_second"],
        },
        "rows": rows, "environment": env,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")
    print(f"{backend.name} ep-dispatch-combine [{args.phase}/{args.mode}]: status={doc['status']} "
          f"{len(rows)} points, headline T={headline['tokens_per_rank']} "
          f"disp={headline['dispatch_us_p50']:.1f}us combine={headline['combine_us_p50']:.1f}us "
          f"-> {args.out}")
    return 0 if all_ok else 1
