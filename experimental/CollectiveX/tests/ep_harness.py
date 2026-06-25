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
  * **Explicit measurement contract** (review #3): adapters conform to a NAMED timing
    boundary, they do not each choose their own. layout-and-dispatch-v1 times the
    routing-layout step inside dispatch (the only contract MoRI can honor); cached-
    layout-comm-only-v1 hoists it out (DeepEP). Combine excludes staging in both.
    Serial = SUM of the two isolated medians (NOT a measured chained op).
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

SCHEMA_VERSION = 3  # v3: explicit contracts, pooled trials p50/p90/p99, routing-identity proof, separated logical bytes

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
    # Measurement contract — the EXPLICIT timing boundary every adapter must conform to
    # (review #3: adapters must not each decide their own boundary). Backends declare
    # SUPPORTED_CONTRACTS; run_ep.py rejects an unsupported one.
    #   layout-and-dispatch-v1   — dispatch timing INCLUDES routing-layout generation
    #                              (the only contract MoRI can honor; its layout is
    #                              computed inside the kernel and cannot be hoisted).
    #   cached-layout-comm-only-v1 — layout computed ONCE untimed; dispatch times pure
    #                              comm (DeepEP-only; matches DeepEP's own benchmark).
    # Combine excludes staging in BOTH (staging is untimed for every backend).
    ap.add_argument("--measurement-contract", default="layout-and-dispatch-v1",
                    choices=["layout-and-dispatch-v1", "cached-layout-comm-only-v1"])
    ap.add_argument("--num-sms", type=int, default=24,
                    help="DeepEP comm-SM budget in 'default' resource-mode (MoRI uses block_num/warps)")
    # Resource regime (review: budgets were neither normalized nor tuned):
    #   normalized — each backend restricted to ~sm_fraction of its device's units
    #                (DeepEP set_num_sms(frac·SMs); MoRI block_num≈frac·CUs). Fraction-
    #                based, recorded — an approximate apples-to-apples, not identical work.
    #   tuned      — each backend's recommended/auto launch config (best achievable).
    #   default    — DeepEP --num-sms / MoRI 80 blocks (the bring-up budget).
    ap.add_argument("--resource-mode", default="normalized",
                    choices=["normalized", "tuned", "default"])
    ap.add_argument("--sm-fraction", type=float, default=0.18,
                    help="normalized mode: fraction of device SMs/CUs dedicated to comms (~24/132)")
    ap.add_argument("--num-ep-groups", type=int, default=1,
                    help="concurrent EP groups; >1 is REJECTED (real subgroup PGs unimplemented)")
    ap.add_argument("--seed", type=int, default=67)
    # 32: B300/Blackwell needs ~30 untimed iters to reach steady-state GPU clocks +
    # establish NVLink/NVSHMEM connections — at warmup=8 its dispatch read ~1787us
    # (cold), at warmup>=30 it settles to ~85us (faster than H100, reproducible within
    # ~2.5%). H100/MI355X reach steady state much sooner; the extra iters are harmless.
    ap.add_argument("--warmup", type=int, default=32)
    ap.add_argument("--iters", type=int, default=200,
                    help="timed iterations PER TRIAL; pooled across trials for percentiles")
    # review #3: p99 from ~50 samples is just the max. Pool iters x trials, randomize the
    # token-order each trial so warmup/clock drift doesn't correlate with T, report p50/
    # p90/p99 (p99 is the headline). 3 trials x 200 iters = 600 pooled samples per point.
    ap.add_argument("--trials", type=int, default=3,
                    help="independent timed trials, token-order randomized per trial; samples pooled")
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
        str(meta["ep_size"]), str(meta["nodes"]), meta.get("resource_mode", "default"),
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
    warm_shapes = [t for t in ladder if t <= warm_T] or [ladder[0]]
    for wt in warm_shapes:
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
    # Per-point clock-ramp burst (set up below, applied inside the loop): a ONE-TIME burst
    # warms clocks, but on Blackwell (B300) the tiny small-T points let clocks drop again,
    # so a mid-sweep T=64 reads ~20x cold. Re-ramping at EACH shape keeps every timed point
    # steady-state. Gated by backend.wants_warm_burst — MoRI WEDGES on a sustained burst
    # (and is already steady at warmup=8), so it opts out. CX_FABRIC_WARM_BURST overrides.
    warm_burst = int(os.environ.get("CX_FABRIC_WARM_BURST", "40"))
    do_burst = warm_burst > 0 and getattr(backend, "wants_warm_burst", False)

    import random as _random
    elem_dispatch = elem_bytes          # fp8=1 / bf16=2 (dispatch payload element size)
    tol = getattr(backend, "tolerance", 5e-2)

    # ---- Pass 1: build the per-T problem ONCE (deterministic trace + cached layout per
    # contract), run the correctness gate ONCE. Timing is Pass 2 (pooled over trials). ----
    problems, gate = {}, {}
    routing_hashes = set()
    for T in ladder:
        gt = T * ep_size
        idx_g, w_g = routing.build_global_routing(gt, args.experts, args.topk, args.routing,
                                                  args.seed, experts_per_rank)
        rstats = routing.routing_stats(idx_g, args.experts, experts_per_rank, weights=w_g)
        routing_hashes.add(rstats["routing_hash"])
        idx_s, w_s = routing.rank_slice(idx_g, w_g, rank, T)
        x = routing.rank_activations(T, args.hidden, args.seed, rank, device, torch.bfloat16)
        problem = backend.make_problem(T, idx_s.to(device), w_s.to(device), x)
        h = backend.dispatch(problem); backend.stage(problem, h)
        combined = backend.combine(problem, h)
        torch.cuda.synchronize()
        recv_local = backend.recv_tokens(h)
        exp, n_cmp = backend.expected(problem, h)
        max_abs = (combined[:n_cmp].float() - exp[:n_cmp].float()).abs().max().item()
        max_rel = max_abs / (exp[:n_cmp].float().abs().max().item() + 1e-6)
        problems[T] = problem
        gate[T] = {"rstats": rstats, "recv_local": recv_local,
                   "max_rel": max_rel, "local_ok": 1 if max_rel < tol else 0}

    # ---- Pass 2: N timed trials. Token order is randomized PER TRIAL (seeded ⇒ identical
    # on every rank, so collectives stay lock-step) so warmup/clock drift can't correlate
    # with T. Per-iteration cross-rank MAX samples are POOLED across trials, then
    # percentiled (review #3: p99 from one 50-iter run is just the max). MoRI keeps
    # ascending order — it wedges on a cold jump to a large T. ----
    disp_pool = {T: [] for T in ladder}
    comb_pool = {T: [] for T in ladder}
    order = list(ladder)
    rng = _random.Random(args.seed)
    shuffle_ok = not getattr(backend, "needs_gradual_ramp", False)
    for trial in range(max(1, args.trials)):
        if shuffle_ok:
            rng.shuffle(order)
        for T in order:
            problem = problems[T]
            if do_burst:   # re-ramp clocks at THIS shape before timing (Blackwell)
                for _ in range(warm_burst):
                    bh = backend.dispatch(problem); backend.stage(problem, bh); backend.combine(problem, bh)
                torch.cuda.synchronize()
            disp_iters = time_us(torch, lambda p=problem: backend.dispatch(p), args.warmup, args.iters)

            def prep(p=problem):
                hh = backend.dispatch(p); backend.stage(p, hh); return hh
            if backend.combine_needs_redispatch:
                comb_iters = time_us(torch, lambda hh, p=problem: backend.combine(p, hh),
                                     args.warmup, args.iters, pre=prep)
            else:
                hh = prep()
                comb_iters = time_us(torch, lambda p=problem, hx=hh: backend.combine(p, hx),
                                     args.warmup, args.iters)
            # per-iteration cross-rank MAX (the distributed-op latency per iter), pooled.
            disp_pool[T] += _reduce_vec(torch, dist, device, disp_iters, MAX)
            comb_pool[T] += _reduce_vec(torch, dist, device, comb_iters, MAX)

    # ---- Pass 3: percentiles from pooled samples + realized bytes + row ----
    rows = []
    for T in ladder:
        gt = T * ep_size
        g = gate[T]; rstats = g["rstats"]
        d, c = disp_pool[T], comb_pool[T]
        d50, d90, d99 = percentile(d, 50), percentile(d, 90), percentile(d, 99)
        c50, c90, c99 = percentile(c, 50), percentile(c, 90), percentile(c, 99)
        # "Sum of isolated medians" — NOT an independently-measured chained dispatch->combine
        # op (cannot reveal shared sync, launch amortization, or overlap). Named so in the UI.
        s50, s90, s99 = d50 + c50, d90 + c90, d99 + c99
        recv_total = _reduce_int(torch, dist, device, g["recv_local"], SUM)
        recv_max = _reduce_int(torch, dist, device, g["recv_local"], MAX)
        recv_min = _reduce_int(torch, dist, device, g["recv_local"], MIN)
        global_ok = _reduce_int(torch, dist, device, g["local_ok"], MIN)
        max_rel = _reduce_vec(torch, dist, device, [g["max_rel"]], MAX)[0]
        point_ok = bool(global_ok) and recv_total > 0
        # Logical routed payload (NOT wire/bus bandwidth): realized token-copies received
        # across all ranks x hidden x element size. Dispatch and combine counted SEPARATELY
        # at their REAL dtypes; excludes scales/indices/metadata/padding/protocol. The
        # plot reports a "logical routed payload rate", never an algBW/busBW claim.
        dispatch_logical_bytes = recv_total * args.hidden * elem_dispatch
        combine_logical_bytes = recv_total * args.hidden * 2   # combine input is bf16
        rows.append({
            "tokens_per_rank": T, "global_tokens": gt,
            "dispatch_us_p50": d50, "dispatch_us_p90": d90, "dispatch_us_p99": d99,
            "combine_us_p50": c50, "combine_us_p90": c90, "combine_us_p99": c99,
            "serial_us_p50": s50, "serial_us_p90": s90, "serial_us_p99": s99,  # sum of isolated medians
            "samples_pooled": len(d), "trials": max(1, args.trials),
            "recv_tokens_max": recv_max, "recv_tokens_min": recv_min,
            "recv_tokens_mean": recv_total / world_size, "recv_tokens_total": recv_total,
            "dispatch_logical_bytes": dispatch_logical_bytes,
            "combine_logical_bytes": combine_logical_bytes,
            "byte_contract": "logical-routed-payload-v1",
            "tokens_per_second": (gt / (s50 * 1e-6)) if s50 > 0 else None,
            "fanout_mean": rstats["fanout_mean"], "fanout_max": rstats["fanout_max"],
            "routed_copies": rstats["routed_copies"], "expert_load_max": rstats["expert_load_max"],
            "routing_hash": rstats["routing_hash"],
            "correct": point_ok, "max_rel_error": max_rel,
        })
        if rank == 0:
            print(f"  T={T:<5} disp p50/p99={d50:7.1f}/{d99:7.1f}us combine p50/p99={c50:7.1f}/{c99:7.1f}us "
                  f"n={len(d)} fanout={rstats['fanout_mean']:.2f} recv[min/mean/max]="
                  f"{recv_min}/{recv_total // world_size}/{recv_max} correct={point_ok}")

    # Cross-rank workload-identity proof: every rank must have built the SAME global routing
    # (one hash per T here); confirm all ranks agree by hashing the per-T hash set and
    # MIN/MAX-reducing it — a mismatch means NVIDIA and AMD did NOT run identical routing.
    trace_sig = int(hashlib.sha256("|".join(sorted(routing_hashes)).encode()).hexdigest()[:15], 16)
    sig_min = _reduce_int(torch, dist, device, trace_sig, MIN)
    sig_max = _reduce_int(torch, dist, device, trace_sig, MAX)
    routing_consistent = (sig_min == sig_max == trace_sig)

    if rank != 0:
        return 0

    # status=valid requires correctness AND a proven-identical routing trace across ranks.
    all_ok = bool(rows) and all(r["correct"] for r in rows) and routing_consistent
    shape = {  # FIXED line identity (no T, no per-backend resource knobs)
        "hidden": args.hidden, "topk": args.topk, "experts": args.experts,
        "experts_per_rank": experts_per_rank, "dispatch_dtype": args.dispatch_dtype,
        "routing": args.routing,
    }
    meta = {
        "op": "ep-dispatch-combine", "backend": backend.name, "mode": args.mode,
        "phase": args.phase, "world_size": world_size, "ep_size": ep_size,
        "resource_mode": args.resource_mode,
        "nodes": int(os.environ.get("SLURM_NNODES", "1")),
        "topology_class": args.topology_class, "comparison_class": args.comparison_class,
        # honest contract name (was the misleading "comm-only-v1": dispatch INCLUDES layout
        # under layout-and-dispatch-v1). Adapters declare which they conform to.
        "measurement_contract": args.measurement_contract, "shape": shape,
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
        "reproduction": {
            "command": getattr(args, "reproduction_command", ""),
            "image": getattr(args, "image", "") or None,
            "image_digest": getattr(args, "image_digest", "") or None,
            "git_run": getattr(args, "git_run", None),   # GHA run id/attempt/sha (review #1)
            "seed": args.seed, "warmup": args.warmup, "iters": args.iters,
            "trials": max(1, args.trials), "samples_per_point": (max(1, args.trials) * args.iters),
            "measurement_contract": args.measurement_contract,
            "dispatch_dtype": args.dispatch_dtype, "mode": args.mode,
            "fp8_quant_in_timing": getattr(backend, "fp8_in_timing", None),
        },
        **meta,
        "correctness": {"passed": all_ok,
                        "max_rel_error": max((r["max_rel_error"] for r in rows), default=None),
                        "tolerance": getattr(backend, "tolerance", 5e-2), "points": len(rows),
                        # honest scope: round-trip reconstruction + non-silent recv, NOT a full
                        # per-token routing/ordering/weight/padding proof (review #3).
                        "scope": "roundtrip-reconstruction-smoke-v1"},
        "routing_identity": {   # cryptographic workload-identity proof (review #3)
            "consistent_across_ranks": routing_consistent,
            "trace_signature": f"{trace_sig:015x}",
            "distinct_per_T_hashes": sorted(routing_hashes),
        },
        "routing_profile": {
            "routing": args.routing,
            "fanout_mean": sum(r["fanout_mean"] for r in rows) / len(rows),
            "fanout_max": max(r["fanout_max"] for r in rows),
            "headline_hash": headline["routing_hash"],
        },
        "metrics": {   # p99 is the headline percentile (review #3); p50/p90 also kept
            "headline_tokens_per_rank": headline["tokens_per_rank"],
            "headline_percentile": "p99",
            "dispatch_us_p50": headline["dispatch_us_p50"], "dispatch_us_p99": headline["dispatch_us_p99"],
            "combine_us_p50": headline["combine_us_p50"], "combine_us_p99": headline["combine_us_p99"],
            "serial_us_p50": headline["serial_us_p50"], "serial_us_p99": headline["serial_us_p99"],
            "serial_label": "sum of isolated medians (not a measured chained op)",
            "tokens_per_second": headline["tokens_per_second"],
        },
        "rows": rows, "environment": env,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")
    print(f"{backend.name} ep-dispatch-combine [{args.phase}/{args.mode}/{args.measurement_contract}]: "
          f"status={doc['status']} {len(rows)} pts, routing_consistent={routing_consistent}, "
          f"headline T={headline['tokens_per_rank']} disp_p99={headline['dispatch_us_p99']:.1f}us "
          f"-> {args.out}")
    return 0 if all_ok else 1
