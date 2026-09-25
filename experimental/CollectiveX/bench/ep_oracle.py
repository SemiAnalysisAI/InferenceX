"""Independent dispatch and combine correctness models for both receive layouts."""
from __future__ import annotations



# Combine is always BF16, and an FP8 dispatch is modeled exactly (the same quant->
# dequant round-trip is applied to the oracle's semantic payload), so the combine
# oracle keeps one frozen gate regardless of dispatch precision.
# _expected_transformed_combine reproduces the two-level (intra-domain FP32,
# per-domain BF16 scale-out partial) reduction, so a correct backend's only
# residual is the accumulation-order ambiguity the model cannot pin down: at most
# topk (8) BF16 stores at one ulp (2^-8) each. Below the magnitude floor the gate
# is effectively absolute (cancellation makes relative error meaningless there).
#
# The same bound covers the topk-slot-tree model (FlashInfer below 0.6.16, which rounds at
# every level of the reduction tree) without widening: a pairwise tree over topk=8 leaves is
# depth 3, so it compounds at most 3 roundings against the 8 this budget was sized for --
# treewise summation is provably no worse than sequential. Both models are gated against this
# one constant, so a future model must be checked against it rather than assumed to fit.
COMBINE_REL_TOL = 8 * 2.0 ** -8
COMBINE_MAG_FLOOR = 2e-2

# The combine contract(s) each mode may realize. The combine semantics is a BACKEND
# fact, not a pure function of mode: DeepEP's legacy-Buffer decode combine applies the
# top-k gate weights INSIDE the kernel ("weighted-kernel-sum" — the benchmark stages the
# unweighted per-expert transform and the kernel multiplies by the gate), whereas MoRI's
# decode kernels (IntraNodeLL/AsyncLL) keep the plain additive rank sum and reduce the
# gate weights in parallel ("unweighted-rank-sum", identical to normal mode). Normal mode
# is frozen to the unweighted v1 contract. The backend declares which it realizes;
# run_sweep only checks that the declared value is ALLOWED for the mode (so a mislabeled
# adapter fails closed), and the oracle keys on that same declared value.
MODE_ALLOWED_SEMANTICS = {
    "normal": {"unweighted-rank-sum"},
    "low-latency": {"weighted-kernel-sum", "unweighted-rank-sum"},
}

# The (receive_layout, combine_weight_semantics) pairs the correctness oracles model.
# The two axes are independent declarations, but only these combinations have an
# expected-combine model; run_sweep fails closed on any other pairing. A backend
# declaring an unmodeled pair may be correctly implemented -- it needs an oracle
# written for it, not a silent verification against the wrong expectation.
ORACLE_MODELED_CONTRACTS = {
    ("token-rank", "unweighted-rank-sum"),
    ("token-expert", "weighted-kernel-sum"),
}


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
    """Build the per-received-token combine input the staged combine consumes.

    Both contracts apply the same per-expert affine coefficients; they differ only in
    WHERE the top-k gate weight enters. Under ``unweighted-rank-sum`` (normal mode) the
    gate is folded in here (coefficient = the routing weight) because the combine kernel
    only sums. Under ``weighted-kernel-sum`` (low-latency decode) the combine kernel
    multiplies by the gate itself, so the staged payload must be the UNWEIGHTED expert
    transform (coefficient = 1 for each valid assignment) or the gate would be applied
    twice. Each received row still carries at most one valid expert on the low-latency
    padded layout; the sum over the top-k axis then collapses to that single expert.
    """
    valid = expert_ids >= 0
    expert = expert_ids.clamp(min=0).to(torch.int64)
    if combine_weight_semantics == "unweighted-rank-sum":
        coefficient = weights.to(torch.float32).masked_fill(~valid, 0)
    elif combine_weight_semantics == "weighted-kernel-sum":
        coefficient = valid.to(torch.float32)
    else:
        raise ValueError(f"unknown combine semantics {combine_weight_semantics!r}")
    scale, offset_a, offset_b = _expert_coefficients(torch, expert)
    scale_sum = (coefficient * scale).sum(dim=1, keepdim=True)
    offset_a_sum = (coefficient * offset_a).sum(dim=1, keepdim=True)
    offset_b_sum = (coefficient * offset_b).sum(dim=1, keepdim=True)
    pattern = _column_pattern(torch, payload.shape[1], payload.device)
    transformed = (
        payload.float() * scale_sum + offset_a_sum + offset_b_sum * pattern.unsqueeze(0)
    )
    return transformed.to(payload.dtype)


def _topk_slot_tree_combine(torch, destination, valid, messages, dtype):
    """Reduce the per-rank messages the way a payload-dtype accumulator does.

    Most combine kernels accumulate in FP32 and narrow once. FlashInfer's one-sided kernel
    (<= 0.6.15) instead holds its top-k accumulators IN the payload dtype and reduces them
    with a hand-unrolled pairwise tree, so every level rounds:

        acc[k] = message of destination[k], or 0 if a lower k already claimed that rank
        (a0+=a1) (a2+=a3) (a4+=a5) (a6+=a7); (a0+=a2) (a4+=a6); (a0+=a4)   -- and store

    Three BF16 roundings on partials near a contribution's own magnitude is a few ulps of
    error, which is the whole gap a plain FP32 sum leaves against this backend. Operands sit
    at their ORIGINAL top-k slot -- the kernel blanks duplicate-rank slots in place rather
    than compacting -- so the tree's shape depends on the routing, not just the rank count.
    The generic halving below reproduces the unrolled K=6/8/10 trees exactly.

    Unlike the domain reduction, which folds into one accumulator, this holds a message per
    rank AND a slot per top-k position, so oracle memory is O(ep_size * tokens * hidden):
    ~8 GiB at EP16 with the 8192-token prefill rung. Fine against 180+ GiB HBM at the EP
    sizes here, but it is the term that would need streaming before EP32.
    """
    tokens = torch.arange(destination.shape[0], device=destination.device)
    zero = torch.zeros_like(messages[0])
    slots = []
    for slot in range(destination.shape[1]):
        rank_id = destination[:, slot]
        claimed = valid[:, slot].clone()
        for earlier in range(slot):
            claimed &= ~(valid[:, earlier] & (destination[:, earlier] == rank_id))
        slots.append(torch.where(claimed.unsqueeze(1), messages[rank_id, tokens], zero))
    while len(slots) > 1:
        merged = [
            (slots[i] + slots[i + 1]).to(dtype).float()
            for i in range(0, len(slots) - 1, 2)
        ]
        if len(slots) % 2:
            merged.append(slots[-1])
        slots = merged
    return slots[0]


def _expected_transformed_combine(
    torch, problem, experts_per_rank, scale_up_domain, combine_weight_semantics,
    combine_reduction="domain-fp32",
):
    """Reproduce the reduction combine actually performs so the expectation carries the
    same BF16 rounding a correct backend does rather than hiding it in a wide tolerance.

    Two reduction shapes, one per combine contract:

    ``weighted-kernel-sum`` (low-latency decode): every routed expert returns its own
    BF16 message and the source rank multiplies each by that assignment's gate weight
    and FP32-accumulates. The rounding granularity is therefore per (token, expert): cast
    each expert's affine transform to the payload dtype, scale by the gate in FP32, and
    sum. There is no per-domain intermediate — the low-latency kernels reduce at the
    source, so scale-up vs scale-out topology does not change the model.

    ``unweighted-rank-sum`` (normal mode): each destination rank casts its FP32 local
    aggregate to the payload dtype. Ranks sharing a scale-up domain (NVLink/MNNVL) reduce
    in FP32, and each domain casts its aggregate to the payload dtype for the scale-out
    send before those communicated BF16 partials are summed. When the whole EP group fits
    in one scale-up domain (ep_size <= scale_up_domain — every EP8 case and the MNNVL EP16
    cases) there is a single domain and no scale-out rounding; a multi-node RoCE EP16 group
    has one BF16 partial per node, and omitting that cast is what left the scale-out
    combine ~0.048 off a single-domain reference.

    A backend whose accumulator is the payload dtype rather than FP32 declares
    ``combine_reduction = "topk-slot-tree"`` and takes the model in
    :func:`_topk_slot_tree_combine` instead of the domain reduction below.
    """
    semantic_x = getattr(problem, "oracle_x", problem.x)
    expert_ids = problem.topk_idx.to(torch.int64)
    weights = problem.topk_weights.to(torch.float32)
    pattern = _column_pattern(torch, semantic_x.shape[1], semantic_x.device)
    dtype = semantic_x.dtype
    if combine_weight_semantics == "weighted-kernel-sum":
        # Per-assignment BF16 message, gate-scaled in FP32 at the source and summed.
        valid = expert_ids >= 0
        scale, offset_a, offset_b = _expert_coefficients(torch, expert_ids.clamp(min=0))
        expected = torch.zeros_like(semantic_x, dtype=torch.float32)
        for slot in range(expert_ids.shape[1]):
            transform = (
                semantic_x.float() * scale[:, slot:slot + 1]
                + offset_a[:, slot:slot + 1]
                + offset_b[:, slot:slot + 1] * pattern.unsqueeze(0)
            ).to(dtype).float()
            gate = (weights[:, slot:slot + 1] * valid[:, slot:slot + 1].to(torch.float32))
            expected += gate * transform
        return expected
    if combine_weight_semantics != "unweighted-rank-sum":
        raise ValueError(f"unknown combine semantics {combine_weight_semantics!r}")
    valid = expert_ids >= 0
    destination = torch.where(valid, expert_ids, torch.zeros_like(expert_ids))
    destination //= experts_per_rank
    scale, offset_a, offset_b = _expert_coefficients(torch, expert_ids)

    def rank_message(rank_id):
        """The one BF16 row this destination rank stages back for every token.

        The narrowing here is the ADAPTER's — torch producing the staged combine input —
        not the kernel's, so it is always round-to-nearest regardless of what the kernel
        does with its own accumulator.
        """
        gate = weights * (destination == rank_id) * valid
        return (
            semantic_x.float() * (gate * scale).sum(dim=1, keepdim=True)
            + (gate * offset_a).sum(dim=1, keepdim=True)
            + (gate * offset_b).sum(dim=1, keepdim=True) * pattern.unsqueeze(0)
        ).to(dtype).float()

    present = sorted(destination[valid].unique().tolist())
    if combine_reduction == "topk-slot-tree":
        messages = torch.zeros(
            (max(present, default=0) + 1,) + semantic_x.shape,
            dtype=torch.float32, device=semantic_x.device,
        )
        for rank_id in present:
            messages[rank_id] = rank_message(rank_id)
        return _topk_slot_tree_combine(torch, destination, valid, messages, dtype)
    if combine_reduction != "domain-fp32":
        raise ValueError(f"unknown combine reduction {combine_reduction!r}")
    ranks_per_domain = max(1, scale_up_domain)
    domains: dict[int, object] = {}
    for rank_id in present:
        # Per-rank BF16 output, FP32-accumulated within its scale-up domain.
        domain = rank_id // ranks_per_domain
        contribution = rank_message(rank_id)
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


def _chain_output_matches(chained, drained):
    """Whether the chain's final combined output matches a drained pair, same code path.

    A regime A/B, not an oracle: both tensors come from the backend's own dispatch->combine,
    differing only in whether the pair ran free-running or drained, so a mismatch is corruption
    that only manifests under back-to-back pairs (the stale-parity / aliased-signal class) --
    invisible to the drained oracles and to the fresh post-chain check alike. Judged with the
    oracle's elementwise tolerance rather than bit equality because a combine kernel is not
    required to be order-deterministic across invocations; a regime defect produces errors
    orders of magnitude past COMBINE_REL_TOL, never inside it -- an assumption the returned
    magnitude exists to CHECK rather than assert, since a verdict alone cannot distinguish a
    corrupt result from one that merely landed just outside the tolerance.

    Returns (within_tolerance, worst_relative_error).
    """
    if chained.shape != drained.shape:
        return False, float("inf")
    if not chained.numel():
        return True, 0.0
    error = (chained.float() - drained.float()).abs()
    relative = error / drained.float().abs().clamp_min(COMBINE_MAG_FLOOR)
    worst = float(relative.max().item())
    return worst < COMBINE_REL_TOL, worst


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
    # A per-(source, expert) slot receive breaks this oracle's rank-deduplicated
    # assumptions. Route those layouts to the dedicated per-slot oracle; keying on the
    # declared receive layout -- not the combine weighting, which is an independent
    # declaration -- keeps the token-rank path below untouched.
    if getattr(backend, "receive_layout", "token-rank") == "token-expert":
        return _run_ll_expert_oracle(
            torch, routing, backend, problem, global_idx, global_weights,
            rank, experts_per_rank, scale_up_domain, seed,
        )
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
        expected_payload = backend.semantic_payload(
            routing.activations_for_source_ids(
                source_ids, problem.x.shape[1], seed, problem.x.dtype
            )
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
        torch, problem, experts_per_rank, scale_up_domain, combine_weight_semantics,
        getattr(backend, "combine_reduction", "domain-fp32"),
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


def _run_ll_expert_oracle(
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
    """Correctness oracle for the low-latency per-expert-slot dispatch/combine layout.

    Normal mode delivers a rank-deduplicated payload: one row per source token per rank,
    carrying every one of that token's experts that live on the rank, combined by an
    unweighted rank-sum. The low-latency decode kernels instead deliver one row per
    (source token, expert) ASSIGNMENT — a token routed to two experts on this rank
    appears in two rows — and the combine kernel applies the top-k gate weights itself.

    The adapter's low-latency ``inspect_dispatch`` therefore exposes a flat per-slot view:
      * ``payload``            [N, hidden]  activations of each slot's source token
      * ``expert_ids``         [N]          global expert id owning the slot (from the
                                            padded layout's leading dimension)
      * ``local_expert_counts``[experts_per_rank]
    Source identity is decoded from the payload bytes. The low-latency dispatch does NOT
    transport per-slot gate weights — the combine kernel applies the top-k weights at the
    source — so there is no receive-side weight to check here; weight correctness is
    covered end-to-end by the combine-values comparison. The staged combine input is the
    UNWEIGHTED per-expert transform (the kernel multiplies by the gate), so the expected
    combine sums the gate-scaled per-expert BF16 messages (see
    _expected_transformed_combine's weighted-kernel-sum branch)."""
    handle = backend.dispatch(problem)
    torch.cuda.synchronize()
    try:
        view = backend.inspect_dispatch(problem, handle)
        source_ids = routing.decode_source_ids(view.payload, seed)
    except Exception as inspection_error:
        # Drain the in-flight dispatch before reporting (an abandoned handle would
        # deadlock the peer ranks), mirroring the normal-mode oracle's fail-soft path.
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

    device = problem.x.device
    count = int(view.payload.shape[0])
    expert_ids = view.expert_ids.to(torch.int64).reshape(-1)
    local_lo = rank * experts_per_rank
    shape_ok = view.payload.ndim == 2 and tuple(expert_ids.shape) == (count,)
    source_range = bool(
        count == 0
        or ((source_ids >= 0) & (source_ids < global_idx.shape[0])).all().item()
    )
    expert_range = bool(
        count == 0
        or ((expert_ids >= local_lo) & (expert_ids < local_lo + experts_per_rank)).all().item()
    )
    # Every (source, local-expert) assignment the global trace routes to this rank —
    # the exact multiset the per-slot dispatch must deliver.
    local_assignment = (global_idx.to(device) // experts_per_rank) == rank
    exp_source, exp_slot = local_assignment.nonzero(as_tuple=True)
    expected_expert = global_idx.to(device)[exp_source, exp_slot]

    if source_range:
        expected_payload = backend.semantic_payload(
            routing.activations_for_source_ids(
                source_ids, problem.x.shape[1], seed, problem.x.dtype
            )
        )
        payload_ok = torch.equal(view.payload, expected_payload)
    else:
        payload_ok = False

    # Compare the delivered (source, expert) pairs against the expected assignment
    # multiset, and the per-slot gate weight against the trace. A 20-bit expert shift is
    # safe: total experts stay far below 2^20.
    if (
        source_range and expert_range
        and count == int(expected_expert.numel())
    ):
        got_key = source_ids.to(torch.int64) * (1 << 20) + expert_ids
        want_key = exp_source.to(torch.int64) * (1 << 20) + expected_expert
        source_set_ok = bool(
            torch.equal(torch.sort(got_key).values, torch.sort(want_key).values)
        )
    else:
        source_set_ok = False
    # No receive-side weight is transported under low latency (the combine applies the
    # gate at the source), so there is nothing to check here; weight correctness is
    # verified by combine_values below. Report a zero weight error so the artifact field
    # stays populated and comparable with normal mode.
    weights_ok = True
    max_weight_error = 0.0

    actual_counts = view.local_expert_counts.to(torch.int64)
    if source_range and expert_range:
        expected_counts = torch.bincount(
            (expected_expert - local_lo), minlength=experts_per_rank
        ).to(torch.int64)
        counts_ok = tuple(actual_counts.shape) == (experts_per_rank,) and torch.equal(
            actual_counts, expected_counts
        )
    else:
        counts_ok = False
    metadata_ok = shape_ok and source_range and expert_range
    # Each source token must appear once per local expert it routes to; source_set_ok
    # already verified the exact (source, expert) multiset, so multiplicity rides on it.
    multiplicity_ok = source_set_ok

    problem.recv_tokens = count
    combine_weight_semantics = backend.combine_weight_semantics
    # Per-slot unweighted transform: one valid expert per row, unit coefficient (the
    # kernel applies the gate). weights arg is unused under weighted-kernel-sum.
    slot_expert = expert_ids.reshape(count, 1)
    slot_weight = torch.ones((count, 1), dtype=torch.float32, device=device)
    transformed = _expert_transform(
        torch, view.payload, slot_expert, slot_weight, combine_weight_semantics
    )
    view.combine_input = transformed
    combined = backend.combine_transformed(problem, handle, transformed)
    torch.cuda.synchronize()
    expected_combined = _expected_transformed_combine(
        torch, problem, experts_per_rank, scale_up_domain, combine_weight_semantics,
    )
    if combined.shape == expected_combined.shape:
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
        receive_count=count,
        max_absolute_error=max_absolute_error,
        max_elementwise_relative_error=max_elementwise_relative_error,
        max_weight_error=max_weight_error,
        checks=checks,
    )
