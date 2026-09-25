"""Case-attempt artifact schema, byte accounting, and atomic publication."""
from __future__ import annotations

import datetime as _dt
import json
import os
import re

from ep_measurement import _pcts, _reduce_int, _reduce_vec


_CASE_ID = re.compile(r"^[a-z0-9][a-z0-9.-]*$")
_NON_SLUG = re.compile(r"[^a-z0-9]+")


def is_case_id(value) -> bool:
    return bool(isinstance(value, str) and _CASE_ID.fullmatch(value))


def case_id(sku: str, case: dict) -> str:
    parts = (
        sku,
        case["backend"],
        case["workload"],
        case["mode"],
        case["phase"],
        f"ep{int(case['ep'])}",
        case["routing"],
        case["precision"],
    )
    values = [_NON_SLUG.sub("-", str(part).lower()).strip("-") for part in parts]
    if not all(values):
        raise ValueError("case ID contains an empty factor")
    return "-".join(values)


def logical_byte_provenance(
    logical_copies: int,
    hidden: int,
    value_bytes: int = 2,
    scale_bytes_per_copy: int = 0,
) -> dict[str, int]:
    """Return comparable logical activation bytes for one direction.

    BF16 moves 2 bytes/value with no scale payload (``scale_bytes`` zero). An FP8
    dispatch moves 1 byte/value; a blockwise codec (DeepEP) also carries per-block
    FP32 scales (``scale_bytes_per_copy`` > 0), while a plain e4m3 tensor cast (MoRI)
    carries none. Combine is always BF16.
    """
    if logical_copies < 0 or hidden < 0:
        raise ValueError("logical byte dimensions must be non-negative")
    # Every realized dispatch moves at least one byte per value; scale bytes may be zero
    # (BF16, and MoRI's scale-free e4m3 cast).
    if value_bytes <= 0 or scale_bytes_per_copy < 0:
        raise ValueError("value_bytes must be positive and scale bytes non-negative")
    activation_data_bytes = logical_copies * hidden * value_bytes
    scale_bytes = logical_copies * scale_bytes_per_copy
    return {
        "activation_data_bytes": activation_data_bytes,
        "scale_bytes": scale_bytes,
        "total_logical_bytes": activation_data_bytes + scale_bytes,
    }


# Consumer contract, not labels: the frontend and durable store key the headline on
# `components.pair_period` carrying exactly CHAIN_PERIOD_ORIGIN, so a typo fails silently.
CHAIN_PERIOD_ORIGIN = "chained-median"
CHAIN_FLOOR_ORIGIN = "chained-cross-rank-min"


def _component(percentiles, count, *, derived=False, origin=None):
    """One component block: availability, the reduction behind it, percentiles, sample count.

    `origin` names that reduction wherever it is not this suite's default per-iteration cross-rank
    MAX. The chained families set it, because they share this block shape while being
    differently-reduced statistics a consumer must not have to infer from the field name.
    """
    if percentiles is None:
        return {"availability": "unavailable", "origin": None,
                "percentiles_us": None, "sample_count": 0}
    return {
        "availability": "derived" if derived else "measured",
        "origin": origin or ("derived-percentile-sum" if derived else "measured"),
        "percentiles_us": percentiles,
        "sample_count": 0 if derived else count,
    }


# The exact routing fields each row publishes — a whitelist so a new stat in
# routing.routing_stats never leaks into the artifact unreviewed.
_ROUTING_FIELDS = (
    "empty_expert_count", "empty_rank_count", "expert_assignment_rank_cv",
    "expert_assignments_per_rank", "expert_load_cv", "expert_load_max",
    "expert_load_mean", "expert_load_min", "fanout_histogram", "fanout_max",
    "fanout_mean", "fanout_min", "hotspot_ratio", "locality",
    "payload_copies_per_rank", "payload_rank_cv", "routed_copies",
)


def _write_bytes_atomic(path: str, payload: bytes) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = f"{path}.tmp-{os.getpid()}"
    try:
        with open(temporary, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _write_json_atomic(path: str, value) -> None:
    payload = json.dumps(
        value, allow_nan=False, ensure_ascii=False, separators=(",", ":")
    ).encode() + b"\n"
    _write_bytes_atomic(path, payload)


def kernel_generation(backend) -> str:
    """Return the adapter's declared kernel family."""
    return getattr(backend, "kernel_generation", None) or "n-a"


def write_results(args, backend, torch, dist, device, rank, world_size,
                  spec, samples, gate, routing_consistent, chain_output_applicable):
    """Reduce each point, write the case artifact on rank zero, and agree the exit status."""
    ladder, dropped, cap = spec.ladder, spec.dropped, spec.cap
    ep_size = world_size
    gpn, scale_up_domain = args.gpus_per_node, args.scale_up_domain
    mode, suite, workload_name = args.mode, args.suite, args.workload_name
    MAX, MIN, SUM = dist.ReduceOp.MAX, dist.ReduceOp.MIN, dist.ReduceOp.SUM

    # ---- Pass 4: percentiles (p50/p90/p95/p99, nearest-rank) from pooled samples + bytes + row ----
    rows = []
    for T in ladder:
        gt = T * world_size
        g = gate[T]
        rstats = g["rstats"]
        d, s, c, rt = samples[T].dispatch, samples[T].stage, samples[T].combine, samples[T].roundtrip
        dp, sp, cp, rtp = _pcts(d), _pcts(s), _pcts(c), _pcts(rt)
        # isolated_sum = SUM of the isolated dispatch+stage+combine percentiles. Stage contributes
        # zero when it is explicitly not applicable. This is NOT a measured chained operation
        # (can't reveal shared sync / launch amortization / overlap) — do NOT use for throughput
        # or SLO capacity. The MEASURED round trip (rtp) is the real chained latency.
        isum = (
            {key: dp[key] + (sp[key] if sp is not None else 0.0) + cp[key] for key in dp}
            if dp and cp else None
        )
        recv_total = _reduce_int(torch, dist, device, g["recv_local"], SUM)
        recv_max = _reduce_int(torch, dist, device, g["recv_local"], MAX)
        recv_min = _reduce_int(torch, dist, device, g["recv_local"], MIN)
        global_ok = _reduce_int(torch, dist, device, g["local_ok"], MIN)
        # Agreed across ranks like `passed`, not rank 0's local view.
        post_chain_state_passed = bool(
            _reduce_int(torch, dist, device, g["chain_local_ok"], MIN)
        )
        # null where the check does not apply (staging hoisted): the artifact says "not
        # asked", never a bare False that a reader would mistake for a failed comparison.
        # The reduce still runs on every rank so the collective stays aligned.
        chain_last_output_passed = bool(
            _reduce_int(torch, dist, device, g["chain_output_local_ok"], MIN)
        )
        # Published whether or not the verdict passed. Without it the artifact records THAT the
        # chained output differed but never BY HOW MUCH, which is the difference between a
        # transport corruption and a tolerance set too tight for a backend's accumulator.
        chain_output_error = _reduce_vec(
            torch, dist, device, [g["chain_output_error"]], MAX
        )[0]
        if not chain_output_applicable:
            chain_last_output_passed, chain_output_error = None, None
        max_rel = _reduce_vec(torch, dist, device, [g["max_rel"]], MAX)[0]
        point_ok = bool(global_ok) and recv_total > 0
        throughput = {
            percentile_name: gt / (latency_us * 1e-6)
            for percentile_name, latency_us in rtp.items()
        }
        # Canonical LOGICAL payload bytes come from the routing trace (NOT backend recv
        # tensors): one copy per unique (token, dest-rank) pair. Dispatch carries the
        # backend's realized precision (BF16, or 1-byte FP8 + optional scales); combine
        # is always BF16. The roundtrip is their per-field sum and stage moves nothing.
        dispatch_bytes = logical_byte_provenance(
            rstats["routed_copies"], args.hidden,
            backend.dispatch_value_bytes, backend.dispatch_scale_bytes_per_copy,
        )
        combine_bytes = logical_byte_provenance(rstats["routed_copies"], args.hidden)
        # Second byte basis, for backends whose wire carries one copy per (token, expert). Which
        # applies is a property of the RECEIVE, not the mode -- MoRI's LL kernels deduplicate
        # where the other low-latency kernels do not -- so key it on the declared
        # receive layout. `routed_copies` stays the canonical comparable basis.
        assignment_copies = int(sum(rstats["expert_assignments_per_rank"]))
        wire_basis = (
            "per-assignment"
            if backend.receive_layout == "token-expert"
            else "rank-deduplicated"
        )
        roundtrip_bytes = {
            field: dispatch_bytes[field] + combine_bytes[field] for field in dispatch_bytes
        }
        stage_bytes = dict.fromkeys(dispatch_bytes, 0)
        # WIRE bytes: what the kernels actually move, on the basis `wire_basis` declares.
        # For token-expert receives this is the per-assignment count (topk/fanout above the
        # rank-deduplicated basis, +34% observed on nccl-ep LL EP8 at T=128); for token-rank
        # receives it equals the canonical figures. A bandwidth divided from `byte_provenance`
        # on a token-expert backend is a LOWER BOUND, not the wire rate, and is not comparable
        # across backends -- consumers computing GB/s must divide from THESE bytes.
        wire_copies = (
            assignment_copies if wire_basis == "per-assignment"
            else int(rstats["routed_copies"])
        )
        wire_dispatch_bytes = logical_byte_provenance(
            wire_copies, args.hidden,
            backend.dispatch_value_bytes, backend.dispatch_scale_bytes_per_copy,
        )
        wire_combine_bytes = logical_byte_provenance(wire_copies, args.hidden)
        wire_roundtrip_bytes = {
            field: wire_dispatch_bytes[field] + wire_combine_bytes[field]
            for field in wire_dispatch_bytes
        }
        spread = samples[T].spread
        chain = samples[T].chain
        chain_spread = samples[T].chain_spread
        dfloor = samples[T].dispatch_floor
        cfloor = samples[T].combine_floor
        chain_gap = samples[T].gap
        chain_settle = samples[T].settle
        chainp = _pcts(chain)
        rows.append({
            "components": {
                "combine": _component(cp, len(c)),
                "dispatch": _component(dp, len(d)),
                "isolated_sum": _component(isum, 0, derived=True),
                # What a serving decode loop pays per MoE layer: the steady-state period of
                # back-to-back dispatch->combine pairs, every backend, cross-rank median. Not
                # `roundtrip` (drained around every pair, an idle-pipeline latency). Do not sum it.
                "pair_period": _component(chainp, len(chain), origin=CHAIN_PERIOD_ORIGIN),
                "roundtrip": _component(rtp, len(rt)),
                "stage": _component(sp, len(s)),
            },
            # Per-op floors from the FLOORS sibling chain: cross-rank MINIMUM of each op's window,
            # the last-entering rank's. Not the chained cost of dispatch/combine -- only the
            # minimum excludes the parked wait. Tracks profiler kernel time to ~10%.
            "chain_floor_us": {
                "combine": _component(_pcts(cfloor), len(cfloor), origin=CHAIN_FLOOR_ORIGIN),
                "dispatch": _component(_pcts(dfloor), len(dfloor), origin=CHAIN_FLOOR_ORIGIN),
            },
            # Whether the chain was the steady state the period claims. `pair_spread_us` large
            # next to `pair_period` means a paced rank; `interpair_gap_us` growth is the
            # measurement loop contaminating the chain, not the fabric; `settle_drift_us` is the
            # convergence proof `chain_drop` assumes but cannot show. Pass 2b has the reductions.
            "chain_health": {
                "interpair_gap_us": _component(_pcts(chain_gap), len(chain_gap)),
                "pair_spread_us": _component(_pcts(chain_spread), len(chain_spread)),
                "settle_drift_us": _component(_pcts(chain_settle), len(chain_settle)),
            },
            # Skew-excluded companion to `components`: same iterations reduced with cross-rank
            # MIN instead of MAX. Compare against `components` to see how much of a point is the
            # operation and how much is rank stagger; a curve that dips in MAX but not in MIN was
            # never the operation getting faster.
            "cross_rank_min_us": {
                "combine": _component(_pcts(samples[T].combine_min), len(samples[T].combine_min)),
                "dispatch": _component(_pcts(samples[T].dispatch_min), len(samples[T].dispatch_min)),
                "roundtrip": _component(_pcts(samples[T].roundtrip_min), len(samples[T].roundtrip_min)),
            },
            # Diagnostic, NOT a latency: per-iteration cross-rank (max-min) of the round trip.
            # Small => ranks entered together and the reported MAX is the operation's cost.
            # Large relative to the roundtrip => the point is skew-inflated; read it with care.
            "cross_rank_spread_us": _component(_pcts(spread), len(spread)),
            "correctness": {
                # Whether the free-running chain's OWN final combined output (per trial)
                # matched a drained pair through the identical code path, within the combine
                # tolerance. Proves the last pair of each chain, not every interior pair --
                # validating those would put device work inside the timed loops.
                # Folded into `passed` WHERE IT APPLIES; null where it does not, which is
                # wherever staging is hoisted out of the chain (every FP8 adapter by default).
                # There the staged stand-in is decoupled from each pair's dispatch, so chained
                # and drained are not comparable and the question is not asked -- see the Pass
                # 2b call site for the A/B that established that. Read it beside
                # `chain_last_output_error`, never alone.
                "chain_last_output_passed": chain_last_output_passed,
                # How far apart they actually were (cross-rank MAX), published whether or not
                # the verdict passed. A bool alone cannot separate a transport corruption from
                # a tolerance too tight for a backend's accumulator, and reading it as the
                # former without this number is a mistake this field exists to prevent.
                "chain_last_output_error": chain_output_error,
                # Whether the full oracle also passed against the state the free-running chain
                # left behind (a FRESH dispatch+combine after the final trial). Renamed from
                # `chain_regime_passed`, which overclaimed: older artifacts carry that name,
                # and `null` there meant the chain never ran (a state the budget gate has since
                # made impossible). Folded into `passed`.
                "post_chain_state_passed": post_chain_state_passed,
                # Max elementwise relative error (COMBINE_MAG_FLOOR-clamped)
                # against the BF16-faithful expected combine.
                "max_relative_error": max_rel,
                "passed": point_ok,
            },
            "global_tokens": gt,
            "byte_provenance": {
                "combine": combine_bytes,
                "dispatch": dispatch_bytes,
                "roundtrip": roundtrip_bytes,
                "stage": stage_bytes,
            },
            # Same fields on the wire basis (`logical_copies.wire`). Identical to
            # `byte_provenance` for token-rank receives; per-assignment for the LL kernels
            # that move one copy per (token, expert). Bandwidth = wire bytes / latency.
            "wire_byte_provenance": {
                "combine": wire_combine_bytes,
                "dispatch": wire_dispatch_bytes,
                "roundtrip": wire_roundtrip_bytes,
                "stage": stage_bytes,
            },
            # Copy counts behind the byte figures above, so a reader can rebase them: `routed` is
            # the basis they use, `assignments` the per-(token, expert) count, `wire` which the
            # kernels move. Kept out of `byte_provenance`, whose values are all per-component.
            "logical_copies": {
                "routed": int(rstats["routed_copies"]),
                "assignments": assignment_copies,
                "wire": wire_basis,
            },
            "receive": {
                "max": recv_max,
                "mean": recv_total / world_size,
                "min": recv_min,
                "total": recv_total,
            },
            "routing": {key: rstats[key] for key in _ROUTING_FIELDS},
            "token_rate_at_latency_percentile": throughput,
            "tokens_per_rank": T,
        })
        if rank == 0:
            component_log = (f"disp p50/p99={dp['p50']:7.1f}/{dp['p99']:7.1f} "
                             f"comb {cp['p50']:6.1f}/{cp['p99']:6.1f} " if dp and cp
                             else "components=unavailable ")
            period_log = f"period={chainp['p50']:7.1f}us " if chainp else "period=n/a "
            print(f"  T={T:<5} {component_log}{period_log}"
                  f"RT p50/p99={rtp['p50']:7.1f}/{rtp['p99']:7.1f}us n={len(rt)} fanout={rstats['fanout_mean']:.2f} "
                  f"recv[min/mean/max]={recv_min}/{recv_total // world_size}/{recv_max} "
                  f"correct={point_ok}")

    # status=valid requires correctness AND a proven-identical routing trace across ranks.
    all_ok = bool(rows) and all(r["correctness"]["passed"] for r in rows) and routing_consistent

    generated_at = _dt.datetime.now().astimezone().isoformat()
    nodes = int(os.environ.get("SLURM_NNODES", "1"))
    scheduled_case = {
            "backend": backend.name,
            "ep": ep_size,
            "experts": args.experts,
            "gpus_per_node": gpn,
            "hidden": args.hidden,
            "ladder": " ".join(map(str, ladder)),
            "mode": mode,
            "nodes": nodes,
            "phase": args.phase,
            "precision": args.precision,
            "routing": args.routing,
            "scale_up_domain": scale_up_domain,
            "scale_up_transport": args.scale_up_transport,
            "scale_out_transport": args.scale_out_transport or None,
            "scope": args.scope,
            "suite": suite,
            "topk": args.topk,
            "topology_class": args.topology_class,
            "transport": args.transport,
            "workload": workload_name,
    }
    case_factors = {"case": scheduled_case, "sku": args.runner}
    computed_case_id = case_id(args.runner, scheduled_case)
    if args.case_id != computed_case_id:
        raise ValueError(
            f"scheduled case ID does not match realized factors: {args.case_id} != {computed_case_id}"
        )
    git_run = getattr(args, "git_run", None) or {}
    allocation_factors = {
        "run_attempt": git_run.get("run_attempt"),
        "run_id": git_run.get("run_id"),
        "source_sha": git_run.get("source_sha"),
    }
    try:
        attempt_ordinal = int(os.environ.get("COLLX_ATTEMPT_ID", "1"))
    except ValueError:
        attempt_ordinal = 0
    if attempt_ordinal <= 0:
        raise ValueError("COLLX_ATTEMPT_ID must be a positive integer")
    doc = {
        "version": args.version,
        "record_type": "case-attempt",
        "generated_at": generated_at,
        "identity": {
            "allocation_factors": allocation_factors,
            "attempt_ordinal": attempt_ordinal,
            "case_factors": case_factors,
            "case_id": args.case_id,
        },
        "workload": {
            "cross_rank_consistent": routing_consistent,
            # The ladder actually measured, plus any requested point the backend's cap excluded.
            # In stdout only, a clamped ladder was invisible to anyone reading the artifact.
            "ladder_measured": list(ladder),
            "ladder_dropped": list(dropped),
            "ladder_cap": cap,
        },
        "measurement": {
            "combine_dtype": backend.combine_dtype,
            "combine_semantics": "activation-only",
            "dispatch_dtype": backend.dispatch_dtype,
            "payload_unit": "token-rank",
            "rows": rows,
            "sampling": {
                # The fresh-entry family. The chained family below is sampled separately and its
                # counts are not derivable from these.
                "iterations_per_trial": args.iters,
                "samples_per_component": args.iters * args.trials,
                "trials": args.trials,
                "warmup_iterations": args.warmup,
                # `pair_period`, `chain_floor_us` and `chain_health` sampling. Emitted because
                # `sample_count` cannot be decomposed back into them: 128x4 is not 512x1.
                "chain_drop": args.chain_drop,
                "chain_iterations_per_trial": args.chain_iters,
                "chain_trials": args.chain_trials,
            },
        },
        "implementation": {
            # Which production FP8 consumption path the chained roundtrip modelled; see
            # EPBackend.fp8_consume. Only meaningful when the case dispatches FP8.
            "fp8_consume": getattr(backend, "fp8_consume", None),
            "kernel_generation": kernel_generation(backend),
            # Which reduction the correctness oracle held the kernel to. A backend may
            # pick this per installed library version (flashinfer-ep does), so without it
            # a wheel bump silently changes the arithmetic behind `passed` with no trace.
            "combine_reduction": getattr(backend, "combine_reduction", "domain-fp32"),
            # The library version the line above was decided FROM: without it a reader cannot tell
            # a correct selection from a mis-parse. None where a backend does not report one.
            "library_version": getattr(backend, "library_version", None),
            # Whether `roundtrip` excludes expert-output staging. It always does now, unless the
            # CX_FP8_CONSUME=dequant hatch is set; older rows carried the staging copy inside the
            # chain for MoRI and FlashInfer BF16, and without this field they look identical.
            "stage_excluded_from_roundtrip": bool(
                getattr(backend, "stage_excluded_from_roundtrip", False)
            ),
            # Whether this document's rows carry the chained family. Consumers key the headline on
            # presence, as for `stage_excluded_from_roundtrip`; the sweep `version` does not move.
            "chained_period": True,
            # See EPBackend.maturity: a "candidate" row measures the library, not a deployment.
            "maturity": getattr(backend, "maturity", None) or "unknown",
            "name": backend.name,
        },
        "topology": {
            "device_product": getattr(args, "runtime_device_product", None),
            "gpus_per_node": gpn,
            "nodes": nodes,
            "placement": "packed",
            "scale_up_domain": scale_up_domain,
            "scale_up_transport": args.scale_up_transport,
            "scale_out_transport": args.scale_out_transport or None,
            "scope": args.scope,
            "topology_class": args.topology_class,
            "transport": args.transport,
            "world_size": world_size,
        },
        "runtime": getattr(args, "runtime", {}),
        "provenance": {
            "image": getattr(args, "image", "") or None,
            "source_sha": git_run.get("source_sha"),
        },
        "outcome": {
            "reasons": [] if all_ok else ["semantic correctness or routing identity failed"],
            "status": "success" if all_ok else "invalid",
        },
    }
    if rank == 0:
        _write_json_atomic(args.out, doc)
        # Ladder ends + two interior points — one mid-ladder headline hides the
        # low-token (startup-dominated) behavior.
        summary_rows = []
        for tokens in (ladder[0], 8, 64, ladder[-1]):
            row = next((r for r in rows if r["tokens_per_rank"] == tokens), None)
            if row is not None and row not in summary_rows:
                summary_rows.append(row)

        def _point_summary(row):
            period = row["components"]["pair_period"]["percentiles_us"]
            period_summary = f" period_p50={period['p50']:.1f}us" if period else ""
            percentiles = row["components"]["dispatch"]["percentiles_us"]
            if not percentiles:
                return f"T={row['tokens_per_rank']}:n/a{period_summary}"
            return (f"T={row['tokens_per_rank']}:disp_p99={percentiles['p99']:.1f}us"
                    f"{period_summary}")

        component_summary = " ".join(_point_summary(row) for row in summary_rows)
        print(f"{backend.name} ep-dispatch-combine [{args.phase}/{mode}]: "
              f"status={doc['outcome']['status']} {len(rows)} pts, routing_consistent={routing_consistent}, "
              f"{component_summary} "
              f"-> {args.out}")
    # CI honesty: run_sweep's return code is the only success signal run_ep_cases (and thus CI)
    # reads — the doc is uploaded regardless, via the launcher's always() stage step. A captured
    # `invalid` outcome (semantic correctness or cross-rank routing identity failed) must therefore
    # fail the leg, not ride as a green success; otherwise a persistent oracle failure is invisible
    # in CI and could autopublish an invalid doc. Agree the verdict across ranks (MIN) so every
    # rank exits identically and the distributed case fails as one.
    outcome_ok = bool(_reduce_int(torch, dist, device, int(all_ok), dist.ReduceOp.MIN))
    return 0 if outcome_ok else 3
