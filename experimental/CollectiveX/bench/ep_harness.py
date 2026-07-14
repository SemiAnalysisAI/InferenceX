#!/usr/bin/env python3
"""Shared EP timing, correctness, and result generation."""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import re


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
    )
    values = [_NON_SLUG.sub("-", str(part).lower()).strip("-") for part in parts]
    if not all(values):
        raise ValueError("case ID contains an empty factor")
    return "-".join(values)


# Workload and timing values arrive from configs/sweep.json through the matrix.
CONDITIONING_ROUNDS_PER_SHAPE = 8
# Dispatch and combine are fixed BF16, so the combine oracle uses one frozen gate.
# _expected_transformed_combine reproduces the two-level (intra-domain FP32,
# per-domain BF16 scale-out partial) reduction, so a correct backend's only
# residual is the accumulation-order ambiguity the model cannot pin down: at most
# topk (8) BF16 stores at one ulp (2^-8) each. Below the magnitude floor the gate
# is effectively absolute (cancellation makes relative error meaningless there).
COMBINE_REL_TOL = 8 * 2.0 ** -8
COMBINE_MAG_FLOOR = 2e-2

def logical_byte_provenance(logical_copies: int, hidden: int) -> dict[str, int]:
    """Return comparable logical BF16 activation bytes for one direction.

    Dispatch and combine both move BF16 (2 bytes/value) with no separate scale
    payload, so ``scale_bytes`` is always zero.
    """
    if logical_copies < 0 or hidden < 0:
        raise ValueError("logical byte dimensions must be non-negative")
    activation_data_bytes = logical_copies * hidden * 2
    return {
        "activation_data_bytes": activation_data_bytes,
        "scale_bytes": 0,
        "total_logical_bytes": activation_data_bytes,
    }

def format_collective_version(raw) -> str:
    """Normalize PyTorch's tuple or packed NCCL/RCCL version representation."""
    if isinstance(raw, int):
        if raw < 10_000:
            return f"{raw // 1000}.{raw // 100 % 10}.{raw % 100}"
        return f"{raw // 10_000}.{raw // 100 % 100}.{raw % 100}"
    if isinstance(raw, (tuple, list)):
        return ".".join(map(str, raw))
    return str(raw) if raw not in (None, "") else "unknown"


def add_common_args(ap: argparse.ArgumentParser) -> None:
    """Add the varying v1 inputs; fixed profile values are not CLI axes."""
    ap.add_argument("--mode", required=True, choices=["normal"])
    ap.add_argument("--phase", required=True, choices=["decode", "prefill"],
                    help="token-size regime label: decode (small T) / prefill (large T)")
    ap.add_argument("--tokens-ladder", required=True,
                    help="space/comma-separated source-tokens-per-rank sweep; the matrix "
                         "supplies the workload's phase ladder from configs/sweep.json")
    ap.add_argument("--hidden", type=int, required=True)
    ap.add_argument("--topk", type=int, required=True)
    ap.add_argument("--experts", type=int, required=True,
                    help="TOTAL experts (fixed across EP degrees)")
    ap.add_argument("--routing", required=True, choices=["uniform"])
    ap.add_argument("--case-id", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--workload-name", required=True)
    ap.add_argument("--seed", type=int, required=True,
                    help="routing-trace seed; part of the workload identity in configs/sweep.json")
    ap.add_argument(
        "--version",
        type=int,
        required=True,
        help="iterable benchmark version copied verbatim into the emitted result",
    )
    # The single cross-SKU profile lives in configs/sweep.json
    # `timing:`; the matrix bakes it into every scheduled case.
    ap.add_argument("--warmup", type=int, required=True,
                    help="untimed full roundtrips before each trial/point")
    ap.add_argument("--iters", type=int, required=True,
                    help="timed iterations per trial")
    ap.add_argument("--trials", type=int, required=True,
                    help="timed trials")
    # provenance / output
    ap.add_argument("--runner", required=True)
    ap.add_argument("--topology-class", required=True)
    ap.add_argument("--transport", required=True)
    ap.add_argument("--scope", required=True, choices=["scale-up", "scale-out"])
    ap.add_argument("--scale-up-transport", required=True)
    ap.add_argument("--scale-out-transport", required=True)
    ap.add_argument("--gpus-per-node", type=int, required=True)
    ap.add_argument("--scale-up-domain", type=int, required=True)
    ap.add_argument("--out", required=True)


def token_ladder(spec: str, cap: int | None) -> tuple[list[int], list[int]]:
    """Return (ladder, dropped) from an explicit spec (there is no default — the
    model-specific ladders live in configs/sweep.json); positive ints; clamped to
    `cap` with dropped points reported (never silently truncated)."""
    want = sorted({t for t in (int(t) for t in spec.replace(",", " ").split() if t) if t > 0})
    if cap is not None:
        return [t for t in want if t <= cap], [t for t in want if t > cap]
    return want, []


def trial_order(values: list, trial_index: int) -> list:
    """Rotate and reverse values so each occupies every timing position."""
    if not values or len(values) != len(set(values)):
        raise ValueError("trial order requires non-empty unique values")
    if type(trial_index) is not int or trial_index < 0:
        raise ValueError("trial_index must be a non-negative integer")
    cycle, offset = divmod(trial_index, len(values))
    base = list(values) if cycle % 2 == 0 else list(reversed(values))
    return base[offset:] + base[:offset]


def percentile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    i = max(0, min(len(s) - 1, math.ceil(q / 100.0 * len(s)) - 1))
    return s[i]


def _pcts(xs):
    return ({"p50": percentile(xs, 50), "p90": percentile(xs, 90),
             "p95": percentile(xs, 95), "p99": percentile(xs, 99)} if xs else None)


def _component(percentiles, count, *, derived=False):
    if percentiles is None:
        return {"availability": "unavailable", "origin": None,
                "percentiles_us": None, "sample_count": 0}
    return {
        "availability": "derived" if derived else "measured",
        "origin": "derived-percentile-sum" if derived else "measured",
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


def time_us(torch, fn, warmup: int, iters: int, pre=None, post=None) -> list[float]:
    """Per-iteration CUDA-event latencies (µs) for THIS rank.

    Without `pre`: times `fn()`. With `pre`: runs `pre()` UNTIMED each iteration (sync
    before the start event so its GPU work can't bleed in), then times `fn(pre_result)`.
    `post(result)` runs after the end event and synchronization, so stateful backends can
    consume/reset a timed operation without charging that cleanup to its latency. Returns
    the raw per-iteration series; the caller reduces across ranks per iteration before
    percentiling.
    """
    def sample():
        arg = pre() if pre is not None else None
        if pre is not None:
            torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        result = fn(arg) if pre is not None else fn()
        e.record()
        torch.cuda.synchronize()
        elapsed = s.elapsed_time(e) * 1000.0  # ms -> us
        if post is not None:
            post(result)
            torch.cuda.synchronize()
        return elapsed

    for _ in range(max(0, warmup)):
        if pre is not None:
            a = pre()
            torch.cuda.synchronize()
            fn(a)
        else:
            fn()
        # sync EACH warmup iteration, not just once after the loop: the measured-roundtrip fn
        # interleaves dispatch+combine on a backend's persistent comm buffer, so back-to-back
        # un-synced warmup iterations let iter N+1's dispatch race iter N's combine (CUDA abort
        # on a rank -> NCCL-watchdog SIGABRT). Cheap (warmup is small); timed samples already sync.
        torch.cuda.synchronize()
    return [sample() for _ in range(iters)]


def kernel_generation(backend) -> str:
    """Return the adapter's declared kernel family."""
    return getattr(backend, "kernel_generation", None) or "n-a"


def _reduce_vec(torch, dist, device, vals, op):
    t = torch.tensor(vals, device=device, dtype=torch.float64)
    dist.all_reduce(t, op=op)
    return [float(x) for x in t.tolist()]


def _reduce_int(torch, dist, device, v: int, op) -> int:
    t = torch.tensor([int(v)], device=device, dtype=torch.int64)
    dist.all_reduce(t, op=op)
    return int(t.item())


def _same_tensors_across_ranks(torch, dist, device, *tensors) -> bool:
    matches = True
    for tensor in tensors:
        observed = tensor.to(device=device, non_blocking=False)
        reference = observed.clone() if dist.get_rank() == 0 else torch.empty_like(observed)
        dist.broadcast(reference, src=0)
        matches = matches and bool(torch.equal(observed, reference))
    result = torch.tensor([int(matches)], device=device, dtype=torch.int64)
    dist.all_reduce(result, op=dist.ReduceOp.MIN)
    return bool(result.item())


def _normalized_expert_metadata(torch, expert_ids, weights):
    """Sort each row by global expert ID while keeping -1 sentinels last."""
    valid = expert_ids >= 0
    keys = torch.where(valid, expert_ids.to(torch.int64), torch.full_like(expert_ids, 1 << 30))
    order = torch.argsort(keys, dim=1, stable=True)
    sorted_ids = torch.gather(expert_ids.to(torch.int64), 1, order)
    sorted_weights = torch.gather(weights.to(torch.float32), 1, order)
    sorted_valid = sorted_ids >= 0
    return (
        torch.where(sorted_valid, sorted_ids, torch.full_like(sorted_ids, -1)),
        sorted_weights.masked_fill(~sorted_valid, 0),
    )


def _expert_coefficients(torch, expert):
    """Per-expert affine coefficients — the transform and its independently derived
    expectation must use these exact formulas, so they exist only here."""
    scale = ((expert * 17 + 5) % 31 + 1).to(torch.float32) / 32
    offset_a = (((expert * 29 + 7) % 37) - 18).to(torch.float32) / 64
    offset_b = (((expert * 43 + 11) % 41) - 20).to(torch.float32) / 128
    return scale, offset_a, offset_b


def _column_pattern(torch, ncols, device):
    columns = torch.arange(ncols, device=device, dtype=torch.int64)
    return (((columns * 13) % 17) - 8).to(torch.float32) / 8


def _expert_transform(torch, payload, expert_ids, weights, combine_weight_semantics):
    """Build one local expert aggregate for the v1 unweighted combine contract."""
    if combine_weight_semantics != "unweighted-rank-sum":
        raise ValueError("benchmark requires unweighted rank-sum combine")
    valid = expert_ids >= 0
    expert = expert_ids.clamp(min=0).to(torch.int64)
    gate = weights.to(torch.float32).masked_fill(~valid, 0)
    scale, offset_a, offset_b = _expert_coefficients(torch, expert)
    scale_sum = (gate * scale).sum(dim=1, keepdim=True)
    offset_a_sum = (gate * offset_a).sum(dim=1, keepdim=True)
    offset_b_sum = (gate * offset_b).sum(dim=1, keepdim=True)
    pattern = _column_pattern(torch, payload.shape[1], payload.device)
    transformed = (
        payload.float() * scale_sum + offset_a_sum + offset_b_sum * pattern.unsqueeze(0)
    )
    return transformed.to(payload.dtype)


def _expected_transformed_combine(torch, problem, experts_per_rank, scale_up_domain):
    """Reproduce the two-level reduction combine actually performs, so the
    expectation carries the same BF16 rounding a correct backend does rather than
    hiding it in a wide tolerance. Each destination rank casts its FP32 local
    aggregate to the payload dtype (as _expert_transform does). Ranks sharing a
    scale-up domain (NVLink/MNNVL) then reduce in FP32, and each domain casts its
    aggregate to the payload dtype for the scale-out send before those
    communicated BF16 partials are summed. When the whole EP group fits in one
    scale-up domain (ep_size <= scale_up_domain — every EP8 case and the MNNVL
    EP16 cases) there is a single domain and no scale-out rounding; a multi-node
    RoCE EP16 group has one BF16 partial per node, and omitting that cast is what
    left the scale-out combine ~0.048 off a single-domain reference."""
    semantic_x = getattr(problem, "oracle_x", problem.x)
    expert_ids = problem.topk_idx.to(torch.int64)
    weights = problem.topk_weights.to(torch.float32)
    pattern = _column_pattern(torch, semantic_x.shape[1], semantic_x.device)
    dtype = semantic_x.dtype
    destination = expert_ids // experts_per_rank
    ranks_per_domain = max(1, scale_up_domain)
    domains: dict[int, object] = {}
    scale, offset_a, offset_b = _expert_coefficients(torch, expert_ids)
    for rank_id in destination.unique().tolist():
        gate = weights * (destination == rank_id)
        # Per-rank BF16 output, FP32-accumulated within its scale-up domain.
        contribution = (
            semantic_x.float() * (gate * scale).sum(dim=1, keepdim=True)
            + (gate * offset_a).sum(dim=1, keepdim=True)
            + (gate * offset_b).sum(dim=1, keepdim=True) * pattern.unsqueeze(0)
        ).to(dtype).float()
        domain = rank_id // ranks_per_domain
        if domain in domains:
            domains[domain] += contribution
        else:
            domains[domain] = contribution
    # Each domain's aggregate is cast to the communicated payload dtype (the
    # scale-out send) before the partials are summed. Unrouted tokens carry an
    # exact zero through every level (all gates zero) — no mask needed.
    expected = torch.zeros_like(semantic_x, dtype=torch.float32)
    for domain in sorted(domains):
        expected += domains[domain].to(dtype).float()
    return expected


_ORACLE_CHECKS = (
    "combine_values", "counts", "metadata", "multiplicity", "payload",
    "source_set", "weights",
)


def _oracle_report(**fields):
    """One report shape for both the fail-soft and full oracle paths, so every
    emitted correctness dict carries identical keys."""
    report = {
        "passed": False,
        "rel_tol": COMBINE_REL_TOL,
        "mag_floor": COMBINE_MAG_FLOOR,
        "combine_weight_semantics": "undeclared",
        "receive_count": 0,
        "max_absolute_error": None,
        "max_elementwise_relative_error": None,
        "max_weight_error": None,
        "checks": dict.fromkeys(_ORACLE_CHECKS, False),
    }
    assert set(fields) <= set(report), sorted(set(fields) - set(report))
    report.update(fields)
    assert set(report["checks"]) == set(_ORACLE_CHECKS)
    return report


def _run_expert_oracle(
    torch,
    routing,
    backend,
    problem,
    global_idx,
    global_weights,
    rank: int,
    experts_per_rank: int,
    scale_up_domain: int,
    seed: int,
):
    """Verify one real dispatch/transform/combine without entering a timed region."""
    handle = backend.dispatch(problem)
    torch.cuda.synchronize()
    try:
        view = backend.inspect_dispatch(problem, handle)
        source_ids = routing.decode_source_ids(view.payload, seed)
    except Exception as inspection_error:
        # Drain the in-flight dispatch before reporting: an abandoned handle
        # would deadlock the other ranks.
        try:
            problem.recv_tokens = backend.recv_tokens(handle)
            backend.stage(problem, handle)
            backend.combine(problem, handle)
            torch.cuda.synchronize()
        except Exception as cleanup_error:
            raise inspection_error from cleanup_error
        return _oracle_report(
            combine_weight_semantics=getattr(
                backend, "combine_weight_semantics", "undeclared"
            ),
        )

    receive_count = int(view.payload.shape[0])
    shape_ok = (
        view.payload.ndim == 2
        and view.expert_ids.shape == (receive_count, problem.topk_idx.shape[1])
        and view.weights.shape == view.expert_ids.shape
    )
    source_range = bool(
        receive_count == 0
        or ((source_ids >= 0) & (source_ids < global_idx.shape[0])).all().item()
    )
    if source_range:
        expected_idx = global_idx.to(problem.x.device).index_select(0, source_ids)
        expected_weights = global_weights.to(problem.x.device).index_select(0, source_ids)
        local = (expected_idx // experts_per_rank) == rank
        expected_ids = torch.where(local, expected_idx, torch.full_like(expected_idx, -1))
        expected_weights = expected_weights.masked_fill(~local, 0)
        expected_payload = routing.activations_for_source_ids(
            source_ids, problem.x.shape[1], seed, problem.x.dtype
        )
    else:
        expected_ids = torch.full_like(view.expert_ids, -1)
        expected_weights = torch.zeros_like(view.weights)
        expected_payload = torch.empty_like(view.payload)
    actual_ids, actual_weights = _normalized_expert_metadata(
        torch, view.expert_ids, view.weights
    )
    expected_ids, expected_weights = _normalized_expert_metadata(
        torch, expected_ids, expected_weights
    )
    expected_sources = (
        ((global_idx // experts_per_rank) == rank).any(dim=1).nonzero(as_tuple=True)[0]
    ).to(problem.x.device)
    source_set_ok = (
        source_range
        and source_ids.numel() == torch.unique(source_ids).numel()
        and torch.equal(torch.sort(source_ids).values, expected_sources)
    )
    payload_ok = source_range and torch.equal(view.payload, expected_payload)
    metadata_ok = shape_ok and torch.equal(actual_ids, expected_ids)
    max_weight_error = (
        float((actual_weights - expected_weights).abs().max().item())
        if actual_weights.numel()
        else 0.0
    )
    weights_ok = max_weight_error == 0.0
    valid_expected = expected_ids >= 0
    expected_local = expected_ids[valid_expected] - rank * experts_per_rank
    expected_counts = torch.bincount(expected_local, minlength=experts_per_rank)
    counts_ok = torch.equal(
        view.local_expert_counts.to(torch.int64), expected_counts.to(torch.int64)
    )
    multiplicity_ok = torch.equal(
        (actual_ids >= 0).sum(dim=1), (expected_ids >= 0).sum(dim=1)
    )
    problem.recv_tokens = receive_count
    combine_weight_semantics = backend.combine_weight_semantics
    transformed = _expert_transform(
        torch, view.payload, actual_ids, actual_weights, combine_weight_semantics
    )
    view.combine_input = transformed
    combined = backend.combine_transformed(problem, handle, transformed)
    torch.cuda.synchronize()
    expected_combined = _expected_transformed_combine(
        torch, problem, experts_per_rank, scale_up_domain
    )
    if combined.shape == expected_combined.shape:
        # Zero errors stand when the rank legitimately combined nothing.
        max_absolute_error = max_elementwise_relative_error = 0.0
        combine_values_ok = True
        if combined.numel():
            absolute_error = (combined.float() - expected_combined).abs()
            max_absolute_error = float(absolute_error.max().item())
            max_elementwise_relative_error = float(
                (absolute_error / expected_combined.abs().clamp_min(COMBINE_MAG_FLOOR))
                .max().item()
            )
            combine_values_ok = max_elementwise_relative_error < COMBINE_REL_TOL
    else:
        max_absolute_error = max_elementwise_relative_error = None
        combine_values_ok = False
    checks = {
        "combine_values": combine_values_ok,
        "counts": counts_ok,
        "metadata": metadata_ok,
        "multiplicity": multiplicity_ok,
        "payload": payload_ok,
        "source_set": source_set_ok,
        "weights": weights_ok,
    }
    return _oracle_report(
        passed=all(checks.values()),
        combine_weight_semantics=combine_weight_semantics,
        receive_count=receive_count,
        max_absolute_error=max_absolute_error,
        max_elementwise_relative_error=max_elementwise_relative_error,
        max_weight_error=max_weight_error,
        checks=checks,
    )


def run_sweep(args, backend, torch, dist, device, rank: int, world_size: int) -> int:
    """Drive the source-tokens-per-rank sweep for one fully-specified line."""
    mode = args.mode
    if mode != "normal":
        if rank == 0:
            print(f"ERROR: unknown CollectiveX case mode {mode!r}")
        return 2
    if min(args.iters, args.trials, args.warmup) <= 0:
        if rank == 0:
            print(f"ERROR: iters/trials/warmup must be positive; got "
                  f"{args.iters}:{args.trials}:{args.warmup}")
        return 2
    import routing  # torch-based; imported lazily so the module byte-compiles without torch

    ep_size = world_size
    num_logical = getattr(args, "num_logical_experts", args.experts)
    if args.experts % ep_size != 0:
        if rank == 0:
            print(f"ERROR: experts ({args.experts}) must divide ep_size ({ep_size})")
        return 2
    experts_per_rank = args.experts // ep_size
    gpn = args.gpus_per_node
    scale_up_domain = args.scale_up_domain
    suite = args.suite
    workload_name = args.workload_name
    if getattr(backend, "mode", None) != mode:
        if rank == 0:
            print(f"ERROR: backend mode {getattr(backend, 'mode', None)!r} != {mode!r}")
        return 2
    expected_weight_semantics = "unweighted-rank-sum"
    if getattr(backend, "combine_weight_semantics", None) != expected_weight_semantics:
        if rank == 0:
            print(
                f"ERROR: {mode} requires combine semantics {expected_weight_semantics}"
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
    MAX, MIN, SUM = dist.ReduceOp.MAX, dist.ReduceOp.MIN, dist.ReduceOp.SUM

    # Inputs determine the communicator capacity.
    backend.create_buffer(spec)

    # ---- Pass 1: per shape, ascending (a cold-jump-safe ramp): warm untimed,
    # then prove workload identity and run the expert oracle. The untimed warm
    # rounds settle clocks/fabric BEFORE anything gate-bearing runs at that shape
    # and are never measured or emitted. ----
    problems, gate, gts, global_traces, input_snapshots = {}, {}, {}, {}, {}
    routing_consistent = True
    for T in ladder:
        gt = T * ep_size
        gts[T] = gt
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
            "pre_input_unchanged": pre_input_unchanged,
        }

    # ---- Pass 2: every backend uses the same rotated point order.
    # Per-iteration cross-rank MAX samples are pooled across trials. ----
    disp_pool = {T: [] for T in ladder}     # pooled per-iteration cross-rank MAX (dispatch)
    stage_pool = {T: [] for T in ladder}    # measured only when stage launches device work
    comb_pool = {T: [] for T in ladder}     # ... combine
    rt_pool = {T: [] for T in ladder}       # independently measured round trip
    for trial_index in range(args.trials):
        order = trial_order(list(ladder), trial_index)
        for T in order:
            problem = problems[T]
            # timed_components() encodes the roundtrip-only vs full-component contract
            # (and whether stage launches device work) once, in the base class.
            component_order = trial_order(backend.timed_components(), trial_index)
            measured = {name: [] for name in ("dispatch", "stage", "combine", "roundtrip")}
            for component_name in component_order:
                # The base template gives every component the same synchronized
                # full-roundtrip warm-up before its timed trial and encodes the two
                # branch rules (dispatch cleanup, combine re-dispatch) internally.
                measured[component_name] = backend.benchmark_component(
                    component_name, problem, args.warmup, args.iters
                )
            # per-iteration cross-rank MAX (the distributed-op latency per iter), pooled.
            if measured["dispatch"]:
                disp_pool[T] += _reduce_vec(torch, dist, device, measured["dispatch"], MAX)
                comb_pool[T] += _reduce_vec(torch, dist, device, measured["combine"], MAX)
            if measured["stage"]:
                stage_pool[T] += _reduce_vec(torch, dist, device, measured["stage"], MAX)
            rt_pool[T] += _reduce_vec(torch, dist, device, measured["roundtrip"], MAX)

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
        gate[T].update({
            "input_unchanged": input_unchanged,
            "local_ok": int(pre["passed"] and post["passed"] and input_unchanged),
            "max_rel": max(
                pre["max_elementwise_relative_error"] or 0.0,
                post["max_elementwise_relative_error"] or 0.0,
            ),
            "oracle_post": post,
        })

    # ---- Pass 4: percentiles (p50/p90/p95/p99, nearest-rank) from pooled samples + bytes + row ----
    rows = []
    for T in ladder:
        gt = gts[T]
        g = gate[T]
        rstats = g["rstats"]
        d, s, c, rt = disp_pool[T], stage_pool[T], comb_pool[T], rt_pool[T]
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
        max_rel = _reduce_vec(torch, dist, device, [g["max_rel"]], MAX)[0]
        point_ok = bool(global_ok) and recv_total > 0
        throughput = {
            percentile_name: gt / (latency_us * 1e-6)
            for percentile_name, latency_us in rtp.items()
        }
        # Canonical LOGICAL payload bytes come from the routing trace (NOT backend recv
        # tensors): one copy per unique (token, dest-rank) pair. Dispatch and combine
        # move the same logical payload; the roundtrip is their sum and stage moves nothing.
        one_way_bytes = logical_byte_provenance(rstats["routed_copies"], args.hidden)
        roundtrip_bytes = {field: 2 * value for field, value in one_way_bytes.items()}
        stage_bytes = dict.fromkeys(one_way_bytes, 0)
        rows.append({
            "components": {
                "combine": _component(cp, len(c)),
                "dispatch": _component(dp, len(d)),
                "isolated_sum": _component(isum, 0, derived=True),
                "roundtrip": _component(rtp, len(rt)),
                "stage": _component(sp, len(s)),
            },
            "correctness": {
                # Max elementwise relative error (COMBINE_MAG_FLOOR-clamped)
                # against the BF16-faithful expected combine.
                "max_relative_error": max_rel,
                "passed": point_ok,
            },
            "global_tokens": gt,
            "byte_provenance": {
                "combine": one_way_bytes,
                "dispatch": one_way_bytes,
                "roundtrip": roundtrip_bytes,
                "stage": stage_bytes,
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
            print(f"  T={T:<5} {component_log}"
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
            "experts": num_logical,
            "gpus_per_node": gpn,
            "hidden": args.hidden,
            "ladder": " ".join(map(str, ladder)),
            "mode": mode,
            "nodes": nodes,
            "phase": args.phase,
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
        },
        "measurement": {
            "combine_dtype": "bf16",
            "combine_semantics": "activation-only",
            "dispatch_dtype": "bf16",
            "payload_unit": "token-rank",
            "rows": rows,
            "sampling": {
                "iterations_per_trial": args.iters,
                "samples_per_component": args.iters * args.trials,
                "trials": args.trials,
                "warmup_iterations": args.warmup,
            },
        },
        "implementation": {
            "kernel_generation": kernel_generation(backend),
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
            percentiles = row["components"]["dispatch"]["percentiles_us"]
            if not percentiles:
                return f"T={row['tokens_per_rank']}:n/a"
            return f"T={row['tokens_per_rank']}:disp_p99={percentiles['p99']:.1f}us"

        component_summary = " ".join(_point_summary(row) for row in summary_rows)
        print(f"{backend.name} ep-dispatch-combine [{args.phase}/{mode}]: "
              f"status={doc['outcome']['status']} {len(rows)} pts, routing_consistent={routing_consistent}, "
              f"{component_summary} "
              f"-> {args.out}")
    # CI honesty: run_sweep's return code is the only success signal collx_run_shard (and thus CI)
    # reads — the doc is uploaded regardless, via the launcher's always() stage step. A captured
    # `invalid` outcome (semantic correctness or cross-rank routing identity failed) must therefore
    # fail the leg, not ride as a green success; otherwise a persistent oracle failure is invisible
    # in CI and could autopublish an invalid doc. Agree the verdict across ranks (MIN) so every
    # rank exits identically and the distributed case fails as one.
    outcome_ok = bool(_reduce_int(torch, dist, device, int(all_ok), dist.ReduceOp.MIN))
    return 0 if outcome_ok else 3
