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

# Phase profiles (goal P2 "decode/prefill representation"): decode/prefill are token-size REGIMES
# that also carry distinct serving semantics — NOT merely ladder aliases. Emitted into the doc so a
# T=128 point launched under "prefill" is never silently read as decode (the shared-T overlap is
# the same kernel at the same T; the phase records what serving situation it stands in). Each point
# is ONE MoE layer, ONE step, a SINGLE dispatch+combine collective pair — not a whole model or
# several concurrent layers.
PHASE_PROFILE = {
    "decode": {"regime": "decode", "tokens_per_iter": "1 (or few) per active sequence",
               "microbatch": "one decode step across the active sequences",
               "routing_variability": "varies step-to-step (temporal routing modes model this)",
               "represents": "one MoE layer · one decode step · one dispatch+combine collective"},
    "prefill": {"regime": "prefill", "chunk": "chunked-prefill — many tokens/sequence per MoE layer",
                "request_mixture": "tokens of one chunk entering a single MoE layer at once",
                "represents": "one MoE layer · one prefill chunk · one dispatch+combine collective"},
}


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
    # Combine-path precision/quant is a SEPARATE axis from dispatch (review: don't let
    # dispatch_dtype=fp8 imply the whole EP path is quantized). Today every backend combines
    # bf16 with no quant (combine_quant_mode=none); a future quantized combine (e.g. ROCm/MoRI
    # PR311) sets these WITHOUT changing --dispatch-dtype. Defaults reproduce today exactly;
    # capability.py gates unsupported values.
    ap.add_argument("--combine-dtype", default="bf16", choices=["bf16", "fp8"],
                    help="combine-input precision (today bf16 everywhere; fp8 = future quant combine)")
    ap.add_argument("--combine-quant-mode", default="none",
                    help="combine quantization mode; 'none' today. capability.py rejects unwired modes")
    # Activation VALUE distribution of expert inputs (goal P2). normal = seeded N(0,1) (the only
    # latency-relevant one under bf16 combine — bf16 is value-independent); the others stress a
    # FUTURE quantized combine's scale computation (amax/outliers/saturation). routing.py owns
    # the generators; capability.py gates which a backend/mode admits.
    ap.add_argument("--activation-profile", default="normal",
                    choices=["normal", "zeros", "small-amplitude", "wide-dynamic-range", "fp8-saturation"],
                    help="value distribution of expert inputs (routing.ACTIVATION_PROFILES)")
    # uniform = realistic top-k (fan-out ≈5.3 over EP8); balanced = load-equalized,
    # one-expert-per-rank (fan-out = ep_size); balanced-rank-local = fan-out 1 (min
    # comm) edge case; zipf = skewed; hotspot-* = adversarial single hot expert (static
    # or moving across steps); alternating-groups = expert halves that toggle by step.
    ap.add_argument("--routing", default="uniform",
                    choices=["uniform", "balanced", "balanced-rank-local", "zipf",
                             "zipf-mild", "zipf-moderate", "zipf-heavy", "hotspot-single",
                             "hotspot-moving", "alternating-groups"])
    # Temporal snapshot index for the moving/alternating distributions (goal P2 "temporal routing
    # changes"). One run = one step; a temporal suite launches steps 0..N and analyze_ep compares
    # them. Folds into workload_id only when non-zero (preserves existing canonical ids).
    ap.add_argument("--routing-step", type=int, default=0,
                    help="temporal step for hotspot-moving / alternating-groups (0 = first/static)")
    # Uneven source-token allocation (goal P2 "support uneven source-token allocation"): per-rank
    # token counts vary (global may not divide EP); empty-source-rank case included. Default 'none'
    # = every rank gets exactly the ladder T (perfectly even; source-token CV 0) — no behavior
    # change for existing runs. 'linear' ramps counts ~0.5T..1.5T; 'empty-rank' zeroes rank 0.
    ap.add_argument("--uneven-tokens", default="none", choices=["none", "linear", "empty-rank"],
                    help="per-rank source-token allocation skew (records source_token_stats)")
    # EPLB (Expert-Parallel Load Balancer): replicate hot experts onto redundant physical
    # slots + balanced-place so per-rank load equalizes. A pure routing-trace transform
    # (tests/eplb.py); experts becomes num_logical+redundant. The remedy for `zipf` skew.
    ap.add_argument("--eplb", action="store_true",
                    help="apply EPLB expert replication/placement to the routing trace")
    ap.add_argument("--num-redundant-experts", type=int, default=32,
                    help="EPLB: redundant physical expert slots (rounded up to a multiple of ep_size)")
    # Canonical serialized workload (goal P1): consume pre-generated trace bytes instead of the
    # seeded runtime generator, so a result is provably the SAME workload as another machine's
    # (checksum match). Points at a dir of <workload_id>.npz/.manifest.json (make_workloads.py).
    ap.add_argument("--workload-dir", default="",
                    help="dir of canonical workload traces; empty = seeded runtime generation (dev)")
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
    #   runtime-visible-v1       — the serving-realistic boundary: dispatch starts from what the
    #                              runtime has right after routing and INCLUDES required quant /
    #                              scale creation / layout / packing / comm / sync; combine starts
    #                              from expert outputs and ends when token outputs are consumable.
    #                              (DeepEP-only today; the FP8 cast moves INSIDE the timed window.)
    ap.add_argument("--measurement-contract", default="layout-and-dispatch-v1",
                    choices=["layout-and-dispatch-v1", "cached-layout-comm-only-v1",
                             "runtime-visible-v1"])
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
    # Anomaly waiver (goal P1: roundtrip/isolated_sum threshold -> diagnostic unless explicitly
    # waived). Without this, a measured roundtrip implausibly larger/smaller than its components
    # (e.g. the open LL-FP8 anomaly) demotes the result to 'diagnostic'. Pass to keep it
    # comparable-experimental/official AFTER the cause is understood + documented.
    ap.add_argument("--waive-anomaly", action="store_true",
                    help="do not let a flagged timing anomaly demote publication_status to diagnostic")
    ap.add_argument("--roundtrip-anomaly-threshold", type=float, default=3.0,
                    help="roundtrip p99 > threshold x isolated_sum p99 is flagged as an anomaly")
    # provenance / output
    ap.add_argument("--runner", required=True)
    ap.add_argument("--topology-class", required=True)
    ap.add_argument("--transport", default="")
    ap.add_argument("--comparison-class", default="standardized")
    # Structured placement metadata (goal P2 topology): GPUs/node + scale-up domain + placement
    # kind let routing locality (local/same-node/cross-domain copy fractions) be computed and let
    # packed/striped/adversarial be distinguished. gpus-per-node=0 -> single node (= ep_size).
    ap.add_argument("--gpus-per-node", type=int, default=0)
    ap.add_argument("--scale-up-domain", type=int, default=0, help="0 = gpus_per_node*ep (one domain)")
    ap.add_argument("--placement", default="packed",
                    choices=["packed", "striped", "runtime-native", "adversarial"])
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


def source_token_counts(nominal_T: int, ep_size: int, mode: str) -> list[int]:
    """Per-rank source-token counts for the uneven-allocation study (goal P2). 'none' = even
    (every rank nominal_T; global = nominal_T*ep). 'linear' = a deterministic ramp ~0.5T..1.5T
    (mean ≈ T, so global tokens stay ~the same but ranks are imbalanced). 'empty-rank' = rank 0
    gets 0 and the rest share evenly (the empty-source-rank case). Deterministic => identical on
    every rank. Counts are clamped to >=0; total need not divide ep_size."""
    if mode == "none" or ep_size <= 1:
        return [nominal_T] * ep_size
    if mode == "empty-rank":
        if ep_size < 2:
            return [nominal_T]
        # rank 0 empty; spread ep_size*T across the remaining ranks (keeps ~global constant).
        total = nominal_T * ep_size
        per = max(1, total // (ep_size - 1))
        return [0] + [per] * (ep_size - 1)
    # linear ramp from ~0.5T to ~1.5T across ranks (mean ≈ T). At least 1 token/rank.
    if ep_size == 1:
        return [nominal_T]
    lo, hi = 0.5 * nominal_T, 1.5 * nominal_T
    return [max(1, int(round(lo + (hi - lo) * r / (ep_size - 1)))) for r in range(ep_size)]


def _stats_vec(xs: list[int]) -> dict:
    """min/mean/max/CV (+ empty count) of a per-rank count vector — self-describing source-token
    or load summary without dumping the full vector."""
    n = len(xs) or 1
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    cv = (var ** 0.5 / mean) if mean > 0 else 0.0
    return {"min": min(xs) if xs else 0, "mean": round(mean, 3),
            "max": max(xs) if xs else 0, "cv": round(cv, 4),
            "empty_ranks": sum(1 for x in xs if x == 0), "total": sum(xs), "ranks": n}


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
        # sync EACH warmup iteration, not just once after the loop: the measured-roundtrip fn
        # interleaves dispatch+combine on a backend's persistent comm buffer, so back-to-back
        # un-synced warmup iterations let iter N+1's dispatch race iter N's combine (CUDA abort
        # on a rank -> NCCL-watchdog SIGABRT). Cheap (warmup is small); timed samples already sync.
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


def _allgather_floats(torch, dist, device, v: float) -> list[float]:
    """Gather one scalar from every rank -> list indexed by rank (for per-rank diagnostics:
    which rank is the straggler, the rank spread). all_reduce can't do this — it collapses."""
    world = dist.get_world_size()
    out = [torch.zeros(1, device=device, dtype=torch.float64) for _ in range(world)]
    dist.all_gather(out, torch.tensor([float(v)], device=device, dtype=torch.float64))
    return [float(x.item()) for x in out]


def _histogram(xs: list[float], nbins: int = 40) -> dict:
    """Compact distribution of pooled cross-rank-max samples (for p99-spike debugging without
    storing every sample). Equal-width bins between min and max."""
    if not xs:
        return {"n": 0}
    lo, hi = min(xs), max(xs)
    if hi <= lo:
        return {"n": len(xs), "min": lo, "max": hi, "bins": nbins, "counts": [len(xs)]}
    counts = [0] * nbins
    span = hi - lo
    for x in xs:
        b = min(nbins - 1, int((x - lo) / span * nbins))
        counts[b] += 1
    return {"n": len(xs), "min": round(lo, 3), "max": round(hi, 3), "bins": nbins, "counts": counts}


def _provenance_unknown(prov: dict) -> list[str]:
    return [k for k, v in prov.items() if isinstance(v, str) and v.strip().lower() == "unknown"]


def _resource_profile(prov: dict, args) -> dict:
    """Map backend-specific provenance onto the backend-INDEPENDENT resource vocabulary (goal P3):
    requested vs achieved comm-unit fraction, configured units/warps, and a conformance class.
    DeepEP units = SMs (num_sms); MoRI units = CU blocks (block_num)."""
    dev = prov.get("device_sms") or prov.get("device_cus")
    cfg = prov.get("num_sms") if prov.get("num_sms") is not None else prov.get("block_num")
    requested = args.sm_fraction if args.resource_mode == "normalized" else None
    achieved = (cfg / dev) if (cfg and dev) else None
    floored = bool(prov.get("block_num_floored"))
    # FIXED-KERNEL split (goal P3 / immediate P0): a kernel whose comm occupancy is fixed by the
    # library and NOT a normalized/tuned SM/CU budget (DeepEP LL: num_sms=None, low_latency_mode,
    # tuned_source=ll-fixed-kernel) is NOT a resource-constrained run. It gets resource_class=
    # fixed-kernel + conformance not-applicable, and is excluded from resource-Pareto comparisons.
    fixed_kernel = bool(prov.get("low_latency_mode")) or ("fixed-kernel" in str(prov.get("tuned_source", "")))
    if fixed_kernel:
        resource_class, cls = "fixed-kernel", "not-applicable"
    elif floored:
        resource_class, cls = "resource-constrained", "minimum-functional"  # needed MORE than requested
    elif args.resource_mode == "normalized":
        resource_class, cls = "resource-constrained", "resource-conforming"
    elif args.resource_mode == "tuned":
        resource_class = "backend-tuned"
        cls = "best-known" if "default" not in str(prov.get("tuned_source", "")) else "backend-default"
    else:
        resource_class, cls = "backend-default", "backend-default"
    # within tolerance? (normalized only — did we hit the requested fraction?)
    tol = 0.10
    target_achieved = (requested is not None and achieved is not None
                       and abs(achieved - requested) <= tol) if requested else None
    return {
        "comm_units_kind": "sm" if prov.get("num_sms") is not None else "cu_block",
        "requested_fraction": requested, "configured_units": cfg, "device_units": dev,
        "achieved_fraction": round(achieved, 4) if achieved else None,
        "warps_dispatch": prov.get("dispatch_warps"), "warps_combine": prov.get("combine_warps"),
        "qps_per_rank": prov.get("num_qps_per_rank"),
        "persistent_bytes": prov.get("num_nvl_bytes") or prov.get("num_rdma_bytes") or prov.get("heap_size"),
        "tuned_source": prov.get("tuned_source"),
        # resource_class: fixed-kernel | resource-constrained | backend-tuned | backend-default.
        # fixed-kernel + backend-* are NOT normalized resource-constrained runs (excluded from Pareto).
        "resource_class": resource_class,
        "conformance_class": cls, "tolerance": tol, "target_achieved_within_tol": target_achieved,
        "nonconforming": floored, "fixed_kernel": fixed_kernel,
        "pareto_eligible": (resource_class == "resource-constrained" and not floored),
    }


def _derive_publication_status(v: dict) -> str:
    """Machine-derive the publication state from the validity dimensions (goal P1). No caller
    may hand-label a result 'official' — it must earn every gate here."""
    if v["execution_status"] != "complete":
        return "failed"
    if v["semantic_correctness"] != "pass" or v["measurement_conformance"] != "conformant" \
       or v["workload_identity"] == "inconsistent":
        return "invalid"
    sound = (v["semantic_correctness"] == "pass"
             and v["workload_identity"].startswith("consistent")
             and v["measurement_conformance"] == "conformant")
    # resource-nonconforming but otherwise sound -> diagnostic (not a fair cross-platform point)
    if v["resource_conformance"].endswith("nonconforming"):
        return "diagnostic"
    # contract-level anomaly (goal P1-e/f): a flagged roundtrip/isolated_sum mismatch demotes to
    # diagnostic unless explicitly waived (validity.anomaly_free reflects the waiver).
    if not v.get("anomaly_free", True):
        return "diagnostic"
    if sound and v["provenance_complete"] and v["workload_source"] == "canonical-serialized":
        return "official"
    if sound:
        return "comparable-experimental"   # measurement sound, missing a publication requirement
    return "diagnostic"


def run_sweep(args, backend, torch, dist, device, rank: int, world_size: int) -> int:
    """Drive the source-tokens-per-rank sweep for one fully-specified line."""
    import routing  # torch-based; imported lazily so the module byte-compiles without torch
    import eplb     # stdlib planner + torch remap (the EPLB transform)

    ep_size = world_size  # num_ep_groups removed (was metadata-only; no real subgroups)
    # EPLB (if on): run_ep.py already bumped args.experts to the PHYSICAL count and stashed the
    # logical count, so experts_per_rank below is physical. The trace is built over LOGICAL
    # experts then remapped to physical (build_trace), so the whole sweep runs over the
    # balanced physical placement with no adapter change.
    eplb_on = getattr(args, "eplb", False)
    num_logical = getattr(args, "num_logical_experts", args.experts)
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
    # temporal snapshot index — defined BEFORE the EPLB block (which builds a reference trace with
    # step=routing_step); the EPLB path runs only when eplb_on, so a late definition raised an
    # UnboundLocalError on zipf+eplb canonical runs (caught as a preserved failed-case).
    routing_step = int(getattr(args, "routing_step", 0))

    # EPLB plan (once): estimate logical load from the global logical trace at the largest
    # ladder T (most samples), then replicate+place. Held fixed across all T (as real EPLB
    # plans from an observed load estimate). build_trace builds the LOGICAL trace and remaps
    # to physical when the plan is present; otherwise it's the identity (logical == physical).
    eplb_plan = None
    if eplb_on:
        ref_idx, _ = routing.build_global_routing(max(ladder) * ep_size, num_logical, args.topk,
                                                  args.routing, args.seed, num_logical // ep_size,
                                                  step=routing_step)
        load = torch.bincount(ref_idx.reshape(-1), minlength=num_logical).float().tolist()
        eplb_plan = eplb.build_plan(load, args.experts, ep_size)
        if rank == 0:
            print(f"NOTE: EPLB {num_logical}->{args.experts} experts ({ep_size}x{experts_per_rank}); "
                  f"per-rank load imbalance {eplb_plan['imbalance_before']:.2f}x -> "
                  f"{eplb_plan['imbalance_after']:.2f}x; {eplb_plan['replicated_experts']} experts "
                  f"replicated (hottest {eplb_plan['max_replicas']}x)")

    canonical = bool(getattr(args, "workload_dir", ""))
    uneven = getattr(args, "uneven_tokens", "none")
    if canonical and uneven != "none":
        if rank == 0:
            print(f"ERROR: --uneven-tokens={uneven} is incompatible with --workload-dir "
                  f"(canonical workloads are serialized at a fixed global-token count per id); "
                  f"use seeded-runtime for the uneven-allocation study.")
        return 2
    loaded_workload_ids, loaded_checksums = [], {}
    if canonical:
        import workload as _wl

    def build_trace(gt):
        # canonical: load pre-serialized trace bytes (verified by checksum) so this run is
        # provably the SAME workload as any other consuming the same files. else: seeded gen.
        if canonical:
            wid = _wl.compute_workload_id(args.routing, args.hidden, args.topk, num_logical, gt,
                                          args.seed, step=routing_step)
            idx_np, w_np, man = _wl.load_workload(os.path.join(args.workload_dir, f"{wid}.npz"), verify=True)
            idx_l = torch.from_numpy(idx_np).to(torch.int64)
            w = torch.from_numpy(w_np).to(torch.float32)
            if wid not in loaded_workload_ids:
                loaded_workload_ids.append(wid)
                loaded_checksums[wid] = man.get("checksums")
        else:
            idx_l, w = routing.build_global_routing(gt, num_logical, args.topk, args.routing,
                                                    args.seed, num_logical // ep_size, step=routing_step)
        return (eplb.remap_idx(idx_l, eplb_plan) if eplb_plan is not None else idx_l), w

    # Fabric/clock warm-up BEFORE any timed point (review: H200 had an anomalous cold
    # first point and a 40% decode-vs-prefill mismatch at the shared T=128). Gradually
    # ramp through the small ladder shapes untimed — warms clocks/fabric for everyone
    # and is also cold-jump-safe for MoRI.
    warm_T = min(ladder[-1], 128)
    warm_shapes = [t for t in ladder if t <= warm_T] or [ladder[0]]
    for wt in warm_shapes:
        wi, ww = build_trace(wt * ep_size)
        wsi, wsw = routing.rank_slice(wi, ww, rank, wt)
        wx = routing.rank_activations(wt, args.hidden, args.seed, rank, device, torch.bfloat16,
                                      profile=args.activation_profile)
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
    problems, gate, gts = {}, {}, {}
    routing_hashes = set()
    for T in ladder:
        # Per-rank source-token counts (goal P2 uneven allocation). mode 'none' => [T]*ep,
        # gt = T*ep, offsets = 0,T,2T,... — byte-identical to the even path. Otherwise counts
        # vary (global may not divide ep) and rank 0 may be empty.
        counts = source_token_counts(T, ep_size, uneven)
        offsets = [sum(counts[:r]) for r in range(ep_size)]
        gt = sum(counts)
        gts[T] = gt
        idx_g, w_g = build_trace(gt)
        rstats = routing.routing_stats(idx_g, args.experts, experts_per_rank, weights=w_g)
        gpn = args.gpus_per_node or ep_size
        # placement-aware locality (goal P2): packed/striped/adversarial change which physical
        # node/domain a rank sits on, so the local/same-node/cross-domain copy fractions differ.
        rstats["locality"] = routing.routing_locality(idx_g, experts_per_rank, ep_size, max(1, T),
                                                      gpn, args.scale_up_domain or None,
                                                      placement=args.placement)
        rstats["source_token_stats"] = _stats_vec(counts)
        routing_hashes.add(rstats["routing_hash"])
        my_off, my_cnt = offsets[rank], counts[rank]
        idx_s = idx_g[my_off:my_off + my_cnt].contiguous()
        w_s = w_g[my_off:my_off + my_cnt].contiguous()
        x = routing.rank_activations(my_cnt, args.hidden, args.seed, rank, device, torch.bfloat16,
                                     profile=args.activation_profile)
        problem = backend.make_problem(my_cnt, idx_s.to(device), w_s.to(device), x)
        h = backend.dispatch(problem); backend.stage(problem, h)
        combined = backend.combine(problem, h)
        torch.cuda.synchronize()
        recv_local = backend.recv_tokens(h)
        exp, n_cmp = backend.expected(problem, h)
        # empty source rank (my_cnt==0): nothing to reconstruct locally — gate passes vacuously.
        if n_cmp > 0:
            max_abs = (combined[:n_cmp].float() - exp[:n_cmp].float()).abs().max().item()
            max_rel = max_abs / (exp[:n_cmp].float().abs().max().item() + 1e-6)
        else:
            max_rel = 0.0
        problems[T] = problem
        gate[T] = {"rstats": rstats, "recv_local": recv_local,
                   "max_rel": max_rel, "local_ok": 1 if max_rel < tol else 0}

    # ---- Pass 2: N timed trials. Token order is randomized PER TRIAL (seeded ⇒ identical
    # on every rank, so collectives stay lock-step) so warmup/clock drift can't correlate
    # with T. Per-iteration cross-rank MAX samples are POOLED across trials, then
    # percentiled (review #3: p99 from one 50-iter run is just the max). MoRI keeps
    # ascending order — it wedges on a cold jump to a large T. ----
    disp_pool = {T: [] for T in ladder}     # pooled per-iteration cross-rank MAX (dispatch)
    comb_pool = {T: [] for T in ladder}     # ... combine
    rt_pool = {T: [] for T in ladder}       # ... INDEPENDENTLY-MEASURED round trip (goal P1)
    disp_local = {T: [] for T in ladder}    # THIS rank's own dispatch samples (per-rank diag)
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
            # MEASURED round trip (goal P1: not a sum of percentiles): one timed region over
            # dispatch -> stage (no-op "expert" transform) -> combine -> output ready. Captures
            # shared sync / launch amortization / overlap that the isolated_sum cannot.
            def rt_once(p=problem):
                hh = backend.dispatch(p); backend.stage(p, hh); return backend.combine(p, hh)
            rt_iters = time_us(torch, lambda p=problem: rt_once(p), args.warmup, args.iters)
            # per-iteration cross-rank MAX (the distributed-op latency per iter), pooled.
            disp_pool[T] += _reduce_vec(torch, dist, device, disp_iters, MAX)
            comb_pool[T] += _reduce_vec(torch, dist, device, comb_iters, MAX)
            rt_pool[T] += _reduce_vec(torch, dist, device, rt_iters, MAX)
            disp_local[T] += disp_iters

    # ---- Pass 3: percentiles (p50/p90/p95/p99, nearest-rank) from pooled samples + bytes + row ----
    def pcts(xs):
        return {"p50": percentile(xs, 50), "p90": percentile(xs, 90),
                "p95": percentile(xs, 95), "p99": percentile(xs, 99)}
    rows = []
    all_anomalies = []                                       # contract-level anomalies (goal P1)
    thr_rt = float(getattr(args, "roundtrip_anomaly_threshold", 3.0))
    for T in ladder:
        gt = gts[T]
        g = gate[T]; rstats = g["rstats"]
        d, c, rt = disp_pool[T], comb_pool[T], rt_pool[T]
        dp, cp, rtp = pcts(d), pcts(c), pcts(rt)
        # isolated_sum = SUM of the isolated dispatch+combine percentiles. NOT a measured op
        # (can't reveal shared sync / launch amortization / overlap) — do NOT use for throughput
        # or SLO capacity. The MEASURED round trip (rtp) is the real chained latency.
        isum = {k: dp[k] + cp[k] for k in dp}
        recv_total = _reduce_int(torch, dist, device, g["recv_local"], SUM)
        recv_max = _reduce_int(torch, dist, device, g["recv_local"], MAX)
        recv_min = _reduce_int(torch, dist, device, g["recv_local"], MIN)
        global_ok = _reduce_int(torch, dist, device, g["local_ok"], MIN)
        max_rel = _reduce_vec(torch, dist, device, [g["max_rel"]], MAX)[0]
        point_ok = bool(global_ok) and recv_total > 0
        # Per-rank diagnostics: gather each rank's own dispatch median -> spread + straggler.
        per_rank_med = _allgather_floats(torch, dist, device, percentile(disp_local[T], 50))
        slowest_rank = max(range(len(per_rank_med)), key=lambda i: per_rank_med[i])
        rmean = sum(per_rank_med) / len(per_rank_med)
        # Canonical LOGICAL payload byte contracts (from the routing trace, NOT backend recv
        # tensors): token-rank = one copy per unique (token,dest-rank); token-expert = one copy
        # per routed (token,expert). routed_copies = token-rank copies; gt*topk = token-expert.
        token_rank_copies = rstats["routed_copies"]
        token_expert_copies = gt * args.topk
        H = args.hidden
        # Bandwidth semantics (goal P1 "distinguish all bandwidth concepts"): the ONLY rates we can
        # defensibly publish are logical-payload (canonical routed bytes / latency) and backend-
        # buffer (recv-tensor bytes / latency). algorithm/bus/wire bandwidth are NULL — EP
        # dispatch/combine have no standard busBW model and we have no transport counters, so we
        # must NOT imply physical NVLink/XGMI/RDMA utilization.
        def _rate(nbytes, us):
            return round(nbytes / (us * 1e3), 3) if (us and us > 0) else None
        disp_bytes_l = token_rank_copies * H * elem_dispatch
        comb_bytes_l = token_rank_copies * H * 2
        buf_disp = recv_max * H * elem_dispatch
        buf_comb = recv_max * H * 2
        bandwidth = {
            "logical_payload_rate_gbps": {
                "dispatch": _rate(disp_bytes_l, dp["p50"]), "combine": _rate(comb_bytes_l, cp["p50"]),
                "roundtrip": _rate(disp_bytes_l + comb_bytes_l, rtp["p50"])},
            "backend_buffer_rate_gbps": {
                "dispatch": _rate(buf_disp, dp["p50"]), "combine": _rate(buf_comb, cp["p50"])},
            "algorithm_bandwidth_gbps": None, "bus_bandwidth_gbps": None, "wire_utilization": None,
            "basis": ("logical = canonical routed-payload copies x hidden x dtype / latency; "
                      "buffer = backend recv tensor / latency; alg/bus/wire = null (no defined "
                      "EP busBW formula, no transport counters) — NOT physical link utilization"),
        }
        # Contract-level anomaly checks (goal P1) — attached to the ROW and rolled into validity.
        #   roundtrip_gt_isolated_sum: measured RT p99 >> Σ(isolated dispatch+combine) p99 — a
        #     chained op shouldn't be far larger than its parts (the open LL-FP8 case).
        #   roundtrip_lt_component_floor: measured RT p50 < max(dispatch,combine) p50 — a chained
        #     op can't finish faster than its slowest required component (sync semantics violated).
        row_anoms = []
        if isum["p99"] > 0 and rtp["p99"] > thr_rt * isum["p99"]:
            row_anoms.append({"type": "roundtrip_gt_isolated_sum", "T": T,
                              "roundtrip_p99": round(rtp["p99"], 2), "isolated_sum_p99": round(isum["p99"], 2),
                              "ratio": round(rtp["p99"] / isum["p99"], 2), "threshold": thr_rt})
        floor = max(dp["p50"], cp["p50"])
        if rtp["p50"] > 0 and floor > 0 and rtp["p50"] < 0.95 * floor:
            row_anoms.append({"type": "roundtrip_lt_component_floor", "T": T,
                              "roundtrip_p50": round(rtp["p50"], 2), "component_floor_p50": round(floor, 2)})
        all_anomalies.extend(row_anoms)
        rows.append({
            "tokens_per_rank": T, "global_tokens": gt,
            "dispatch": dp, "combine": cp, "roundtrip": rtp, "isolated_sum": isum,
            # flat aliases kept for back-compat with v3 readers
            "dispatch_us_p50": dp["p50"], "dispatch_us_p90": dp["p90"], "dispatch_us_p99": dp["p99"],
            "combine_us_p50": cp["p50"], "combine_us_p90": cp["p90"], "combine_us_p99": cp["p99"],
            "roundtrip_us_p50": rtp["p50"], "roundtrip_us_p90": rtp["p90"],
            "roundtrip_us_p95": rtp["p95"], "roundtrip_us_p99": rtp["p99"],
            "isolated_sum_us_p50": isum["p50"], "isolated_sum_us_p99": isum["p99"],
            "samples_pooled": len(d), "trials": max(1, args.trials),
            "percentile_interpolation": "nearest-rank",
            "recv_tokens_max": recv_max, "recv_tokens_min": recv_min,
            "recv_tokens_mean": recv_total / world_size, "recv_tokens_total": recv_total,
            "per_rank_dispatch_us": {"min": min(per_rank_med), "mean": rmean,
                                     "max": max(per_rank_med), "spread": max(per_rank_med) - min(per_rank_med),
                                     "slowest_rank": slowest_rank},
            # dispatch carries its dtype's element size; combine input is bf16 (2B).
            "dispatch_logical_bytes": token_rank_copies * H * elem_dispatch,
            "combine_logical_bytes": token_rank_copies * H * 2,
            "byte_contracts": {
                "token_rank_payload_copies": token_rank_copies,
                "token_expert_payload_copies": token_expert_copies,
                "dispatch_bytes": token_rank_copies * H * elem_dispatch,
                "combine_bytes": token_rank_copies * H * 2,
                "fp8_scale_bytes": (token_rank_copies * (H // 128) * 4) if elem_dispatch == 1 else 0,
                "routing_index_bytes": token_expert_copies * 4,   # int32 topk_idx
                "gate_weight_bytes": token_expert_copies * 4,     # f32 topk_weights
            },
            "byte_contract": "logical-routed-payload-v1",
            # throughput from the MEASURED round trip ONLY (not isolated_sum).
            "roundtrip_tokens_per_second": (gt / (rtp["p50"] * 1e-6)) if rtp["p50"] > 0 else None,
            "raw_samples": {"dispatch": _histogram(d), "combine": _histogram(c), "roundtrip": _histogram(rt)},
            # distinguished bandwidth concepts (goal P1) — logical + buffer real, alg/bus/wire null.
            "bandwidth": bandwidth,
            # full load + fanout statistics in EVERY row (goal P2 "report full load and fanout"):
            "fanout_mean": rstats["fanout_mean"], "fanout_max": rstats["fanout_max"],
            "fanout_min": rstats["fanout_min"], "fanout_hist": rstats["fanout_hist"],
            "routed_copies": rstats["routed_copies"],
            "expert_load_min": rstats["expert_load_min"], "expert_load_max": rstats["expert_load_max"],
            "expert_load_mean": rstats["expert_load_mean"], "expert_load_cv": rstats["expert_load_cv"],
            "rank_load_cv": rstats["rank_load_cv"], "hotspot_ratio": rstats["hotspot_ratio"],
            "dest_rank_load_max": rstats["dest_rank_load_max"],
            "dest_rank_load_mean": rstats["dest_rank_load_mean"],
            "empty_expert_count": rstats["empty_expert_count"],
            "empty_rank_count": rstats["empty_rank_count"],
            "rank_load_hist": rstats["rank_load_hist"],
            "source_token_stats": rstats.get("source_token_stats"),
            "routing_hash": rstats["routing_hash"], "locality": rstats.get("locality"),
            "anomalies": row_anoms,
            "correct": point_ok, "max_rel_error": max_rel,
        })
        if rank == 0:
            print(f"  T={T:<5} disp p50/p99={dp['p50']:7.1f}/{dp['p99']:7.1f} comb {cp['p50']:6.1f}/{cp['p99']:6.1f} "
                  f"RT p50/p99={rtp['p50']:7.1f}/{rtp['p99']:7.1f}us n={len(d)} fanout={rstats['fanout_mean']:.2f} "
                  f"recv[min/mean/max]={recv_min}/{recv_total // world_size}/{recv_max} "
                  f"straggler=r{slowest_rank} correct={point_ok}")

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

    # ---- Multi-dimensional validity (goal P1) -> MACHINE-DERIVED publication_status. Adapters
    # never self-label "official"; status is a pure function of these gates. ----
    prov = backend.backend_provenance
    prov_unknown = _provenance_unknown(prov)
    repro = getattr(args, "reproduction_full", {})
    git_run = getattr(args, "git_run", None)
    provenance_complete = (not prov_unknown
                           and bool(getattr(args, "image_digest", ""))
                           and bool(git_run) and all((git_run or {}).get(k) for k in ("run_id", "source_sha")))
    floored = bool(prov.get("block_num_floored"))
    # fixed-kernel (DeepEP LL) is NOT a normalized resource-constrained run -> conformance N/A
    # (immediate P0 "split LL fixed-kernel from normalized-resource"). Not a conformance failure.
    fixed_kernel = bool(prov.get("low_latency_mode")) or ("fixed-kernel" in str(prov.get("tuned_source", "")))
    resource_conformance = ("not-applicable" if fixed_kernel
                            else "minimum-functional-nonconforming" if floored
                            else ("resource-conforming" if args.resource_mode == "normalized"
                                  else "backend-default" if args.resource_mode in ("tuned", "default")
                                  else "unspecified"))
    # record the canonical workload identity consumed (one trace per T -> set of ids/checksums).
    if canonical and loaded_workload_ids:
        args.workload_id = (loaded_workload_ids[0] if len(loaded_workload_ids) == 1
                            else f"set:{len(loaded_workload_ids)}:{loaded_workload_ids[0]}")
        args.workload_checksums = loaded_checksums
    canonical_workload = bool(getattr(args, "workload_id", None))
    # Activation-value identity (scaffold): today activations are seeded N(0,1) and NOT serialized,
    # so identity is the deterministic descriptor (profile|seed|hidden|generator). When a value rig
    # (lognormal / model-trace) lands, this becomes the byte-hash of the serialized activations.
    activation_identity = hashlib.sha256(
        f"{args.activation_profile}|seed={args.seed}|hidden={args.hidden}|gen=collectivex-activation-v1"
        .encode()).hexdigest()[:16]
    # EPLB mapping identity hash (goal P2) — over the replica placement, not just the counts.
    eplb_mapping_hash = None
    if eplb_plan is not None:
        eplb_mapping_hash = hashlib.sha256(json.dumps(
            {"phys2log": eplb_plan["phys2log"], "rank_of_phys": eplb_plan["rank_of_phys"],
             "replicas": eplb_plan["replicas"]}, sort_keys=True).encode()).hexdigest()[:16]
    # Anomaly roll-up (goal P1-e/f): any flagged row anomaly demotes publication_status to
    # diagnostic, unless --waive-anomaly (set AFTER the cause is understood + documented).
    waived = bool(getattr(args, "waive_anomaly", False))
    anomaly_free = (len(all_anomalies) == 0) or waived
    validity = {
        "execution_status": "complete" if rows else "failed",
        "semantic_correctness": "pass" if (rows and all(r["correct"] for r in rows)) else "fail",
        "workload_identity": "consistent-across-ranks" if routing_consistent else "inconsistent",
        "workload_source": "canonical-serialized" if canonical_workload else "seeded-runtime",
        "measurement_conformance": "conformant",   # run_ep gate rejects nonconformant pre-run
        "resource_conformance": resource_conformance,
        "provenance_complete": provenance_complete,
        # anomaly-free unless a contract-level timing anomaly fired (then diagnostic, see above).
        "anomaly_free": anomaly_free,
    }
    publication_status = _derive_publication_status(validity)

    shape = {  # FIXED line identity (no T, no per-backend resource knobs)
        "hidden": args.hidden, "topk": args.topk, "experts": args.experts,
        "experts_per_rank": experts_per_rank, "dispatch_dtype": args.dispatch_dtype,
        "routing": args.routing, "eplb": bool(eplb_plan), "num_logical_experts": num_logical,
        # temporal snapshot + uneven allocation change the realized workload, so they are part of
        # the line identity (fold into comparison_key). Default 0/none reproduce the prior key for
        # non-temporal even runs in spirit (the value is recorded either way).
        "routing_step": routing_step, "uneven_tokens": uneven,
        # value distribution of expert inputs — part of the workload identity (review: quant
        # combine can be value-sensitive). "normal" today; folds into comparison_key.
        "activation_profile": args.activation_profile,
        # Combine contract, SEPARATE from dispatch. Today bf16/none for every backend regardless
        # of dispatch_dtype; a quant-combine backend (PR311) reports its actuals via attrs. In
        # shape so it folds into comparison_key — a quant-combine run is never compared to a bf16 one.
        "quant": {
            "combine_input_dtype": getattr(backend, "combine_input_dtype", args.combine_dtype),
            "combine_accum_dtype": getattr(backend, "combine_accum_dtype", "fp32"),
            "combine_output_dtype": getattr(backend, "combine_output_dtype", "bf16"),
            "combine_quant_mode": getattr(backend, "combine_quant_mode", args.combine_quant_mode),
            "scale_layout": getattr(backend, "scale_layout", None),
        },
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
        # structured placement metadata (goal P2 topology) — replaces the bare topology string.
        "placement": {
            "kind": args.placement, "nodes": int(os.environ.get("SLURM_NNODES", "1")),
            "gpus_per_node": args.gpus_per_node or ep_size,
            "scale_up_domain": args.scale_up_domain or ((args.gpus_per_node or ep_size) * 1),
            "ranks": ep_size, "transport": args.transport,
        },
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
        # Multi-dimensional validity + machine-derived publication status (goal P1). `status`
        # is a back-compat alias (legacy v3 readers) — publication_status is authoritative.
        "validity": validity,
        "publication_status": publication_status,
        "status": "valid" if all_ok else "invalid",
        "workload": {
            "source": validity["workload_source"],
            "workload_id": getattr(args, "workload_id", None),
            "manifest_checksums": getattr(args, "workload_checksums", None),
            "trace_signature": f"{trace_sig:015x}",
            "distinct_per_T_hashes": sorted(routing_hashes),
            # within-run (cross-rank) identity is PROVEN here; cross-hardware identity holds
            # only if another run records the SAME trace_signature / workload_id.
            "cross_rank_consistent": routing_consistent,
            # value-distribution identity of the expert inputs (scaffold; see activation_identity above).
            "activation_profile": args.activation_profile,
            "activation_identity": activation_identity,
        },
        "comparison_key": comparison_key(meta),
        "x_axis": {"primary": "tokens_per_rank",
                   "global_relation": "global_tokens = tokens_per_rank * ep_size"},
        "backend_provenance": backend.backend_provenance,
        # backend-independent resource vocabulary + conformance class (goal P3).
        "resource_profile": _resource_profile(backend.backend_provenance, args),
        "reproduction": {
            "command": getattr(args, "reproduction_command", ""),
            "image": getattr(args, "image", "") or None,
            "image_digest": getattr(args, "image_digest", "") or None,
            "image_arch": getattr(args, "image_arch", None),
            "squash_sha256": getattr(args, "squash_sha256", None),
            "git_run": getattr(args, "git_run", None),   # repo/run/attempt/ref/sha/job/artifact
            # redaction (goal P1): command + provenance carry NO hostnames/IPs/UUIDs/private paths;
            # per-node env (hostnames, GPU UUIDs, NIC GUIDs) lives in the separate gitignored
            # env_json (CI uploads it as a workflow artifact), never inlined into this record.
            "redaction": "no hostnames/IPs/UUIDs/private-paths in command or provenance",
            "seed": args.seed, "warmup": args.warmup, "iters": args.iters,
            "trials": max(1, args.trials), "samples_per_point": (max(1, args.trials) * args.iters),
            "measurement_contract": args.measurement_contract,
            "dispatch_dtype": args.dispatch_dtype, "mode": args.mode,
            "combine_dtype": args.combine_dtype, "combine_quant_mode": args.combine_quant_mode,
            "activation_profile": args.activation_profile,
            "routing_step": routing_step, "uneven_tokens": uneven,
            "waive_anomaly": waived,
            "roundtrip_anomaly_threshold": thr_rt,
            # whether (de)quantization is inside the timed window. fp8_quant_in_timing kept as a
            # back-compat alias (dispatch-side fp8); combine_* are the quant-combine generalization
            # (None today — no quant combine is wired). A backend sets these when it quantizes.
            "fp8_quant_in_timing": getattr(backend, "fp8_in_timing", None),
            "combine_quant_in_timing": getattr(backend, "combine_quant_in_timing", None),
            "combine_dequant_in_timing": getattr(backend, "combine_dequant_in_timing", None),
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
        # EPLB plan + the per-rank load imbalance it removes (the headline of the zipf+EPLB
        # comparison). enabled=False when the run did not apply EPLB.
        # EPLB mapping IDENTITY (goal P2): logical/physical counts + a hash of the replica
        # placement (phys2log/rank_of_phys/replicas). Two EPLB runs are only an official comparison
        # if their mapping_hash matches (cohort.py enforces); zipf vs zipf+eplb is a RECOVERY
        # experiment, not the same raw workload.
        "eplb": ({"enabled": True, "num_logical_experts": num_logical,
                  "num_physical_experts": args.experts,
                  "num_redundant": args.experts - num_logical,
                  "imbalance_before": eplb_plan["imbalance_before"],
                  "imbalance_after": eplb_plan["imbalance_after"],
                  "replicated_experts": eplb_plan["replicated_experts"],
                  "max_replicas": eplb_plan["max_replicas"],
                  "mapping_hash": eplb_mapping_hash}
                 if eplb_plan else {"enabled": False}),
        "routing_profile": {
            "routing": args.routing,
            "fanout_mean": sum(r["fanout_mean"] for r in rows) / len(rows),
            "fanout_max": max(r["fanout_max"] for r in rows),
            "headline_hash": headline["routing_hash"],
        },
        "metrics": {   # p99 is the headline percentile (review #3); p50/p90/p95 also kept per row
            "headline_tokens_per_rank": headline["tokens_per_rank"],
            "headline_percentile": "p99",
            "dispatch_us_p50": headline["dispatch_us_p50"], "dispatch_us_p99": headline["dispatch_us_p99"],
            "combine_us_p50": headline["combine_us_p50"], "combine_us_p99": headline["combine_us_p99"],
            "roundtrip_us_p50": headline["roundtrip_us_p50"], "roundtrip_us_p99": headline["roundtrip_us_p99"],
            "isolated_sum_us_p50": headline["isolated_sum_us_p50"], "isolated_sum_us_p99": headline["isolated_sum_us_p99"],
            "isolated_sum_label": "sum of isolated dispatch+combine percentiles — NOT a measured chained op",
            "roundtrip_tokens_per_second": headline["roundtrip_tokens_per_second"],
        },
        # phase semantics (goal P2): decode/prefill are regimes with distinct serving meaning, not
        # just ladder aliases — a point is one MoE layer / one step / one collective.
        "phase_profile": PHASE_PROFILE.get(args.phase, {"regime": args.phase}),
        # source-token allocation across ranks (goal P2 uneven allocation). 'none' = even.
        "source_allocation": {
            "mode": uneven, "routing_step": routing_step,
            "note": ("even — every rank gets the ladder T (global = T*ep_size)" if uneven == "none"
                     else "uneven — per-rank source-token counts vary; see rows[].source_token_stats "
                          "(global may not divide ep_size; empty-source-rank possible)"),
        },
        # contract-level timing anomalies (goal P1) — aggregate of the per-row flags; demotes
        # publication_status to diagnostic unless --waive-anomaly (validity.anomaly_free).
        "anomalies": all_anomalies,
        "anomaly_summary": {"count": len(all_anomalies), "waived": waived,
                            "types": sorted({a["type"] for a in all_anomalies})},
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
