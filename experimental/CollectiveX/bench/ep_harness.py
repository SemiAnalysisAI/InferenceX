"""Coordinate input validation, correctness passes, and EP measurements."""
from __future__ import annotations

# Keep the existing import surface for callers while the implementation lives with its owner.
from ep_case import add_common_args, case_id, format_collective_version, is_case_id, token_ladder
from ep_measurement import (
    PointSamples, _gather_scalar, _pcts, _reduce_int, _reduce_vec,
    _reduce_vec_median_spread, _same_tensors_across_ranks, percentile, time_us, trial_order,
)
from ep_oracle import (
    COMBINE_MAG_FLOOR, COMBINE_REL_TOL, MODE_ALLOWED_SEMANTICS, ORACLE_MODELED_CONTRACTS,
    _ORACLE_CHECKS, _chain_output_matches, _expected_transformed_combine,
    _expert_transform, _oracle_report, _run_expert_oracle, _run_ll_expert_oracle,
    _topk_slot_tree_combine,
)
from ep_results import (
    CHAIN_FLOOR_ORIGIN, CHAIN_PERIOD_ORIGIN, _component, kernel_generation,
    logical_byte_provenance, write_results,
)

# Workload and timing values arrive from configs/sweep.json through the matrix.
CONDITIONING_ROUNDS_PER_SHAPE = 8

def run_sweep(args, backend, torch, dist, device, rank: int, world_size: int) -> int:
    """Drive the source-tokens-per-rank sweep for one fully-specified line."""
    mode = args.mode
    if mode not in MODE_ALLOWED_SEMANTICS:
        if rank == 0:
            print(f"ERROR: unknown CollectiveX case mode {mode!r}")
        return 2
    if min(args.iters, args.trials, args.warmup) <= 0:
        if rank == 0:
            print(f"ERROR: iters/trials/warmup must be positive; got "
                  f"{args.iters}:{args.trials}:{args.warmup}")
        return 2
    # Fail closed: a drop within one pair of the iteration count leaves nothing (or a single
    # pair, whose start-to-start series is empty) and would publish `pair_period` degenerate and
    # `chain_health` as "unavailable", indistinguishable from a backend that cannot be chained.
    # Requiring two kept pairs here is what lets Pass 2b compute the health scalars
    # unconditionally and Pass 3 assert the chained oracle ran.
    if (min(args.chain_iters, args.chain_trials) <= 0
            or not 0 <= args.chain_drop <= args.chain_iters - 2):
        if rank == 0:
            print(f"ERROR: chain iters/trials must be positive and 0 <= drop <= iters - 2; got "
                  f"{args.chain_iters}:{args.chain_trials}:{args.chain_drop}")
        return 2
    import routing  # torch-based; imported lazily so the module byte-compiles without torch

    ep_size = world_size
    if args.experts % ep_size != 0:
        if rank == 0:
            print(f"ERROR: experts ({args.experts}) must divide ep_size ({ep_size})")
        return 2
    experts_per_rank = args.experts // ep_size
    gpn = args.gpus_per_node
    scale_up_domain = args.scale_up_domain
    if getattr(backend, "mode", None) != mode:
        if rank == 0:
            print(f"ERROR: backend mode {getattr(backend, 'mode', None)!r} != {mode!r}")
        return 2
    allowed_semantics = MODE_ALLOWED_SEMANTICS[mode]
    if getattr(backend, "combine_weight_semantics", None) not in allowed_semantics:
        if rank == 0:
            print(
                f"ERROR: {mode} requires combine semantics in {sorted(allowed_semantics)}; "
                f"backend declares {getattr(backend, 'combine_weight_semantics', None)!r}"
            )
        return 2
    # Layout and weighting are declared independently; only two pairings have a
    # correctness oracle (ORACLE_MODELED_CONTRACTS). Fail closed on the rest, so a
    # backend whose declarations diverge errors with the missing model named instead
    # of being verified against the wrong oracle and publishing the wrong wire basis.
    declared_contract = (
        getattr(backend, "receive_layout", "token-rank"),
        getattr(backend, "combine_weight_semantics", None),
    )
    if declared_contract not in ORACLE_MODELED_CONTRACTS:
        if rank == 0:
            print(
                "ERROR: no correctness oracle models receive_layout="
                f"{declared_contract[0]!r} with combine semantics {declared_contract[1]!r}"
            )
        return 2
    # A non-control precision must realize a non-BF16 dispatch wire format. Otherwise a
    # backend that lists the precision in SUPPORTED_PRECISIONS but never overrode its
    # encode hooks would run the case in BF16 and emit an artifact mislabeled with the
    # scheduled precision — fail closed rather than publish a mislabeled measurement.
    if args.precision != "bf16" and backend.dispatch_dtype == "bf16":
        if rank == 0:
            print(
                f"ERROR: precision {args.precision!r} did not realize a non-BF16 dispatch "
                f"dtype (backend reports {backend.dispatch_dtype!r})"
            )
        return 2

    spec = backend.make_inputs(args)
    if not spec.ok:
        if rank == 0:
            print(f"ERROR: {spec.message}")
        return spec.rc
    cap = spec.cap
    ladder, dropped = spec.ladder, spec.dropped
    if rank == 0 and dropped:
        print(f"NOTE: dropped tokens/rank {dropped} — exceed {backend.name} buffer cap {cap} "
              f"(hidden={args.hidden}); not silently truncated.")
    MAX, MIN = dist.ReduceOp.MAX, dist.ReduceOp.MIN

    # Inputs determine the communicator capacity.
    backend.create_buffer(spec)

    # ---- Pass 1: per shape, ascending (a cold-jump-safe ramp): warm untimed,
    # then prove workload identity and run the expert oracle. The untimed warm
    # rounds settle clocks/fabric BEFORE anything gate-bearing runs at that shape
    # and are never measured or emitted. ----
    problems, gate, global_traces, input_snapshots = {}, {}, {}, {}
    routing_consistent = True
    for T in ladder:
        point = spec.points[T]
        problem = backend.make_problem(
            T, point.topk_idx.to(device), point.topk_weights.to(device), point.activations
        )
        backend.warm(problem, CONDITIONING_ROUNDS_PER_SHAPE)
        torch.cuda.synchronize()
        problems[T] = problem
        idx_g, w_g = point.global_idx, point.global_weights
        rstats = routing.routing_stats(idx_g, args.experts, experts_per_rank)
        rstats["locality"] = routing.routing_locality(
            idx_g, experts_per_rank, ep_size, max(1, T), gpn, scale_up_domain
        )
        point_routing_consistent = _same_tensors_across_ranks(
            torch, dist, device, idx_g, w_g
        )
        routing_consistent = routing_consistent and point_routing_consistent
        input_snapshots[T] = (
            problem.x.clone(), problem.topk_idx.clone(), problem.topk_weights.clone()
        )
        oracle = _run_expert_oracle(
            torch, routing, backend, problem, idx_g, w_g, rank, experts_per_rank,
            scale_up_domain, args.seed,
        )
        before_x, before_idx, before_weights = input_snapshots[T]
        pre_input_unchanged = (
            torch.equal(problem.x, before_x)
            and torch.equal(problem.topk_idx, before_idx)
            and torch.equal(problem.topk_weights, before_weights)
        )
        global_traces[T] = (idx_g, w_g)
        gate[T] = {
            "rstats": rstats,
            "recv_local": oracle["receive_count"],
            "max_rel": oracle["max_elementwise_relative_error"] or 0.0,
            "local_ok": int(oracle["passed"]),
            "oracle_pre": oracle,
            # Filled by Pass 2b after that point's last chain trial; the budget gate above
            # guarantees at least one trial, so Pass 3 asserts this is no longer None.
            "oracle_chain": None,
            # ANDed across chain trials by Pass 2b: each trial's final chained output against
            # a drained pair through the same code path.
            "chain_output_local_ok": 1,
            # Worst chained-vs-drained relative error seen at this point, kept even when the
            # verdict passes: a magnitude creeping toward the tolerance is the early warning a
            # bool cannot give, and the only way to tell a real corruption from a tight gate.
            "chain_output_error": 0.0,
            "pre_input_unchanged": pre_input_unchanged,
        }

    # The chained-output A/B is only defined when the chain stages per pair. Under the hoist
    # (every FP8 adapter by default, since stage_device_work IS the fp8 flag) the staged
    # stand-in is decoupled from each pair's dispatch, so chained and drained are not
    # comparable -- see the call site for the measurement that established this.
    chain_output_applicable = not backend.stage_excluded_from_roundtrip

    # ---- Pass 2: every backend uses the same rotated point order.
    # Per-iteration cross-rank MAX samples are pooled across trials. ----
    # One object per ladder point, so a point's sample series travel together instead of as
    # fourteen parallel dicts that must be kept in step by hand. Every list is per-iteration
    # cross-rank reduced and pooled across trials; the reduction differs per field and is what
    # the field name records (MAX for the published latencies, MIN for the skew-excluded floors,
    # MEDIAN for the chained period -- see the reduction sites below).
    samples = {T: PointSamples() for T in ladder}

    for trial_index in range(args.trials):
        order = trial_order(list(ladder), trial_index)
        for T in order:
            problem = problems[T]
            # timed_components() encodes whether stage launches device work once, in
            # the base class.
            component_order = trial_order(backend.timed_components(), trial_index)
            measured = {name: [] for name in ("dispatch", "stage", "combine", "roundtrip")}
            for component_name in component_order:
                # The base template gives every component the same synchronized
                # full-roundtrip warm-up before its timed trial and encodes the
                # fresh-pair rule (dispatch drain, combine re-dispatch) internally.
                measured[component_name] = backend.benchmark_component(
                    component_name, problem, args.warmup, args.iters
                )
            # per-iteration cross-rank MAX (the distributed-op latency per iter), pooled.
            if measured["dispatch"]:
                samples[T].dispatch += _reduce_vec(torch, dist, device, measured["dispatch"], MAX)
                samples[T].combine += _reduce_vec(torch, dist, device, measured["combine"], MAX)
            if measured["stage"]:
                samples[T].stage += _reduce_vec(torch, dist, device, measured["stage"], MAX)
            rt_max = _reduce_vec(torch, dist, device, measured["roundtrip"], MAX)
            samples[T].roundtrip += rt_max
            # Cross-rank SPREAD (max-min) of the same iterations. A collective cannot finish
            # before its slowest participant, so when ranks enter together every rank measures
            # nearly the same duration and the spread is small; a large spread means the ranks
            # were staggered and the reported MAX is charging one rank's wait for the others.
            # Emitted as a diagnostic so a skew-inflated point is visible in the artifact
            # instead of being mistaken for the operation getting slower.
            rt_min = _reduce_vec(torch, dist, device, measured["roundtrip"], MIN)
            samples[T].spread += [hi - lo for hi, lo in zip(rt_max, rt_min)]
            samples[T].roundtrip_min += rt_min
            if measured["dispatch"]:
                samples[T].dispatch_min += _reduce_vec(torch, dist, device, measured["dispatch"], MIN)
                samples[T].combine_min += _reduce_vec(torch, dist, device, measured["combine"], MIN)

    # ---- Pass 2b: the chained family, on its own trial count. A separate loop because one call
    # already yields chain_iters free-running pairs, so a handful of trials out-samples the
    # fresh-entry components' 256 for a fraction of the wall clock. Ladder order still rotates
    # per trial, as above. ----
    for trial_index in range(args.chain_trials):
        final_chain_trial = trial_index == args.chain_trials - 1
        for T in trial_order(list(ladder), trial_index):
            chained = backend.benchmark_chain(
                problems[T], args.warmup, args.chain_iters, args.chain_drop
            )
            # The chain's OWN final output against a drained pair through the identical
            # dispatch->combine path, outside every timed region. This is the only chained
            # output the run can inspect without putting device work inside the timed loops:
            # each pair overwrites its predecessor's, so interior pairs are unvalidated by
            # design (see methodology, Correctness). The drained pair is collective, issued in
            # the same (trial, T) order on every rank, so the group stays aligned.
            #
            # Skipped entirely where staging is HOISTED, because there the comparison has no
            # meaning. The hoist captures one warm-up dispatch's staged stand-in and reuses it
            # for every pair, so neither the chain's final combine nor the drained reference
            # consumes an input matching its OWN dispatch -- they are two differently
            # mismatched pairs, and nothing requires them to agree. Measured, not assumed:
            # h100/deepep-v2/EP8, identical in every other respect, native (hoisted) vs
            # dequant (staged per pair) --
            #     hoisted:   chain_last_output_error 31..93   (1000x-2966x tolerance)
            #     per-pair:  chain_last_output_error 0.0      (bit-identical, every rung)
            # in BOTH normal and low-latency mode (runs 31180411148, 31185184372, 31185233991).
            # Passing the chain's staged input to the drained pair was tried first and is NOT
            # enough -- it makes the two share an input, but a shared input that matches
            # neither dispatch. Only per-pair staging makes the regimes comparable, which is
            # exactly the case this guard admits, so the drained pair stages inline here and
            # `benchmark_chain`'s staged value has no consumer. Gating under the hoist reddened
            # every FP8 leg fleet-wide for a harness artifact.
            drained = backend.run_roundtrip(problems[T])
            torch.cuda.synchronize()
            if chain_output_applicable:
                output_ok, output_error = _chain_output_matches(chained["combined"], drained)
                gate[T]["chain_output_local_ok"] &= int(output_ok)
                gate[T]["chain_output_error"] = max(
                    gate[T]["chain_output_error"], output_error
                )
            pair = chained["pair"]
            pair_median, pair_spread = _reduce_vec_median_spread(torch, dist, device, pair)
            samples[T].chain += pair_median
            samples[T].chain_spread += pair_spread
            # Per-op: cross-rank MIN only, from the FLOORS sibling chain (the period chain carries
            # no per-op events). The chained windows park each rank's inter-rank wait, so only the
            # minimum -- the last-entering rank's -- is the operation with the wait excluded.
            samples[T].dispatch_floor += _reduce_vec(torch, dist, device, chained["dispatch"], MIN)
            samples[T].combine_floor += _reduce_vec(torch, dist, device, chained["combine"], MIN)
            # Chain-health scalars, one per trial. `interpair_gap_us` = start-to-start minus pair
            # window: the per-pair cost outside the published window, the in-artifact guard against
            # instrumentation self-charging. `settle_drift_us` = late-half minus early-half period.
            # Median across ranks for the gap; signed max-magnitude for the drift. Unconditional:
            # the budget gate guarantees two kept pairs, so both series are non-degenerate.
            s2s_p50 = _pcts(chained["start_to_start"])["p50"]
            gaps = _gather_scalar(
                torch, dist, device, s2s_p50 - _pcts(pair)["p50"]
            )
            samples[T].gap.append(_pcts(gaps)["p50"])
            half = len(pair) // 2
            drifts = _gather_scalar(
                torch, dist, device,
                _pcts(pair[half:])["p50"] - _pcts(pair[:half])["p50"],
            )
            samples[T].settle.append(max(drifts, key=abs))
            if final_chain_trial:
                # Gate the regime we publish: Passes 1 and 3 only check drained calls, so without
                # this a backend that corrupts under free-running pairs would present as the
                # fastest in the suite. Pass 3's machinery against the state this point's chain
                # left behind, once per ladder point -- the failure mode is all-or-nothing.
                idx_g, w_g = global_traces[T]
                gate[T]["oracle_chain"] = _run_expert_oracle(
                    torch, routing, backend, problems[T], idx_g, w_g, rank,
                    experts_per_rank, scale_up_domain, args.seed,
                )

    # ---- Pass 3: prove timed inputs were immutable and repeat the full oracle. ----
    for T in ladder:
        problem = problems[T]
        before_x, before_idx, before_weights = input_snapshots[T]
        input_unchanged = gate[T]["pre_input_unchanged"] and (
            torch.equal(problem.x, before_x)
            and torch.equal(problem.topk_idx, before_idx)
            and torch.equal(problem.topk_weights, before_weights)
        )
        idx_g, w_g = global_traces[T]
        post = _run_expert_oracle(
            torch, routing, backend, problem, idx_g, w_g, rank, experts_per_rank,
            scale_up_domain, args.seed,
        )
        pre = gate[T]["oracle_pre"]
        # The chained ORACLE is ANDed in like the other two, so a chained-regime failure reds the
        # leg. The budget gate rejects chain_trials=0 up front, so a missing chained oracle is a
        # harness bug, not a configuration.
        chain_oracle = gate[T]["oracle_chain"]
        assert chain_oracle is not None, "chained oracle missing despite a validated budget"
        chain_ok = bool(chain_oracle["passed"])
        # The chained-OUTPUT check gates again, on a measured magnitude rather than a verdict.
        # It was briefly demoted on the theory its tolerance was too tight for FP8; probe
        # 31180411148 (h100, deepep-v2, EP8, low-latency) falsified that:
        #   bf16  chain_last_output_error = 0.0 at every rung -- bit-identical
        #   fp8   chain_last_output_error = 31..93 -- 1000x to 2966x COMBINE_REL_TOL
        # A mis-set tolerance lands JUST outside; this is three orders of magnitude past, and
        # the bf16 control proves the comparison itself is exact. Meanwhile every oracle passes
        # (max_relative_error ~0.0039), which is precisely the signature this check exists for:
        # a difference invisible to drained oracles. So FP8's chained output really does
        # disagree with a drained pair, and a leg that cannot reproduce its own chained result
        # should not publish a period from it.
        chain_output_ok = bool(gate[T]["chain_output_local_ok"])
        gate[T].update({
            "input_unchanged": input_unchanged,
            "local_ok": int(
                pre["passed"] and post["passed"] and chain_ok and input_unchanged
                and (chain_output_ok or not chain_output_applicable)
            ),
            "chain_local_ok": int(chain_ok),
            "max_rel": max(
                pre["max_elementwise_relative_error"] or 0.0,
                post["max_elementwise_relative_error"] or 0.0,
                chain_oracle["max_elementwise_relative_error"] or 0.0,
            ),
            "oracle_post": post,
        })


    return write_results(
        args, backend, torch, dist, device, rank, world_size, spec, samples, gate,
        routing_consistent, chain_output_applicable,
    )
