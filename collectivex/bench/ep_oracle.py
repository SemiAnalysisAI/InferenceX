"""Correctness oracles: expected combine models and the dispatch/combine verification pass."""
from __future__ import annotations

import math

# Combine is always BF16 and an FP8 dispatch is modeled exactly (the oracle applies the same
# quant->dequant round-trip), so one frozen gate covers every precision. The models reproduce each
# kernel's rounding, leaving only accumulation-order ambiguity: at most topk (8) BF16 stores at one
# ulp (2^-8) each. The pairwise topk-slot-tree (depth 3) compounds at most 3 roundings against the
# 8 this budget was sized for, so it fits without widening; a new model must be checked against it.
# Below the magnitude floor the gate is effectively absolute.
COMBINE_REL_TOL = 8 * 2.0 ** -8
COMBINE_MAG_FLOOR = 2e-2

# Combine semantics is a BACKEND fact: DeepEP's legacy-Buffer decode combine applies the gate
# inside the kernel, MoRI's decode kernels keep the plain rank sum. Normal mode is frozen to the
# unweighted contract; run_sweep checks the declared value is allowed for the mode.
MODE_ALLOWED_SEMANTICS = {
    "normal": {"unweighted-rank-sum"},
    "low-latency": {"weighted-kernel-sum", "unweighted-rank-sum"},
}

# The (receive_layout, combine_weight_semantics) pairs an oracle models; run_sweep fails closed on
# any other pairing rather than verifying against the wrong expectation.
ORACLE_MODELED_CONTRACTS = {
    ("token-rank", "unweighted-rank-sum"),
    ("token-expert", "weighted-kernel-sum"),
}

ORACLE_CHECKS = (
    "combine_values", "counts", "metadata", "multiplicity", "payload", "source_set", "weights",
)


def oracle_report(**fields):
    """One report shape for the fail-soft and full oracle paths."""
    report = {
        "passed": False,
        "rel_tol": COMBINE_REL_TOL,
        "mag_floor": COMBINE_MAG_FLOOR,
        "combine_weight_semantics": "undeclared",
        "receive_count": 0,
        "max_absolute_error": None,
        "max_elementwise_relative_error": None,
        "max_weight_error": None,
        "checks": dict.fromkeys(ORACLE_CHECKS, False),
    }
    assert set(fields) <= set(report), sorted(set(fields) - set(report))
    report.update(fields)
    assert set(report["checks"]) == set(ORACLE_CHECKS)
    return report


def _expert_coefficients(torch, expert):
    """Per-expert affine coefficients shared by the transform and its expectation."""
    scale = ((expert * 17 + 5) % 31 + 1).to(torch.float32) / 32
    offset_a = (((expert * 29 + 7) % 37) - 18).to(torch.float32) / 64
    offset_b = (((expert * 43 + 11) % 41) - 20).to(torch.float32) / 128
    return scale, offset_a, offset_b


def _column_pattern(torch, ncols, device):
    columns = torch.arange(ncols, device=device, dtype=torch.int64)
    return (((columns * 13) % 17) - 8).to(torch.float32) / 8


def _unique_rank_slots(destination, valid, slot):
    """Whether top-k `slot` is the first to claim its destination rank (kernels blank duplicates)."""
    rank_id = destination[:, slot]
    claimed = valid[:, slot].clone()
    for earlier in range(slot):
        claimed &= ~(valid[:, earlier] & (destination[:, earlier] == rank_id))
    return rank_id, claimed


class CombineModel:
    """What a correct combine returns for the oracle's per-expert affine transform.

    `semantics` is the published `combine_weight_semantics`: whether the gate is folded into the
    staged payload (rank sum) or applied by the combine kernel (weighted kernel sum).
    """

    semantics = ""

    def coefficient(self, torch, weights, valid):
        raise NotImplementedError

    def transform(self, torch, payload, expert_ids, weights):
        """The per-received-token combine input the staged combine consumes."""
        valid = expert_ids >= 0
        expert = expert_ids.clamp(min=0).to(torch.int64)
        coefficient = self.coefficient(torch, weights, valid)
        scale, offset_a, offset_b = _expert_coefficients(torch, expert)
        scale_sum = (coefficient * scale).sum(dim=1, keepdim=True)
        offset_a_sum = (coefficient * offset_a).sum(dim=1, keepdim=True)
        offset_b_sum = (coefficient * offset_b).sum(dim=1, keepdim=True)
        pattern = _column_pattern(torch, payload.shape[1], payload.device)
        transformed = payload.float() * scale_sum + offset_a_sum + offset_b_sum * pattern.unsqueeze(0)
        return transformed.to(payload.dtype)

    def expected(self, torch, problem, experts_per_rank, scale_up_domain):
        raise NotImplementedError


class WeightedKernelSum(CombineModel):
    """Low-latency decode: each expert returns its own BF16 message and the source rank scales it
    by the gate and FP32-accumulates, so there is no per-domain intermediate."""

    semantics = "weighted-kernel-sum"

    def coefficient(self, torch, weights, valid):
        return valid.to(torch.float32)

    def expected(self, torch, problem, experts_per_rank, scale_up_domain):
        x = getattr(problem, "oracle_x", problem.x)
        expert_ids = problem.topk_idx.to(torch.int64)
        weights = problem.topk_weights.to(torch.float32)
        pattern = _column_pattern(torch, x.shape[1], x.device)
        valid = expert_ids >= 0
        scale, offset_a, offset_b = _expert_coefficients(torch, expert_ids.clamp(min=0))
        expected = torch.zeros_like(x, dtype=torch.float32)
        for slot in range(expert_ids.shape[1]):
            transform = (
                x.float() * scale[:, slot:slot + 1]
                + offset_a[:, slot:slot + 1]
                + offset_b[:, slot:slot + 1] * pattern.unsqueeze(0)
            ).to(x.dtype).float()
            expected += (weights[:, slot:slot + 1] * valid[:, slot:slot + 1].to(torch.float32)) * transform
        return expected


class RankSum(CombineModel):
    """Normal mode: each destination rank stages one gate-weighted BF16 row per token and the
    combine sums those rows. Subclasses differ only in how the rows are reduced."""

    semantics = "unweighted-rank-sum"

    def coefficient(self, torch, weights, valid):
        return weights.to(torch.float32).masked_fill(~valid, 0)

    def expected(self, torch, problem, experts_per_rank, scale_up_domain):
        x = getattr(problem, "oracle_x", problem.x)
        expert_ids = problem.topk_idx.to(torch.int64)
        weights = problem.topk_weights.to(torch.float32)
        pattern = _column_pattern(torch, x.shape[1], x.device)
        valid = expert_ids >= 0
        destination = torch.where(valid, expert_ids, torch.zeros_like(expert_ids))
        destination //= experts_per_rank
        scale, offset_a, offset_b = _expert_coefficients(torch, expert_ids)

        def rank_message(rank_id):
            # The adapter's narrowing (torch staging the combine input), so always round-to-nearest.
            gate = weights * (destination == rank_id) * valid
            return (
                x.float() * (gate * scale).sum(dim=1, keepdim=True)
                + (gate * offset_a).sum(dim=1, keepdim=True)
                + (gate * offset_b).sum(dim=1, keepdim=True) * pattern.unsqueeze(0)
            ).to(x.dtype).float()

        present = sorted(destination[valid].unique().tolist())
        return self.reduce(torch, x, destination, valid, present, rank_message, scale_up_domain)

    def reduce(self, torch, x, destination, valid, present, rank_message, scale_up_domain):
        raise NotImplementedError

    @staticmethod
    def _messages(torch, x, present, rank_message):
        messages = torch.zeros(
            (max(present, default=0) + 1,) + x.shape, dtype=torch.float32, device=x.device,
        )
        for rank_id in present:
            messages[rank_id] = rank_message(rank_id)
        return messages


class DomainFP32(RankSum):
    """Ranks sharing a scale-up domain reduce in FP32; each domain casts its aggregate to BF16 for
    the scale-out send before the partials are summed (omitting that cast left multi-node EP16
    ~0.048 off). One domain, so no scale-out rounding, whenever ep_size <= scale_up_domain."""

    def reduce(self, torch, x, destination, valid, present, rank_message, scale_up_domain):
        ranks_per_domain = max(1, scale_up_domain)
        domains: dict[int, object] = {}
        for rank_id in present:
            domain = rank_id // ranks_per_domain
            contribution = rank_message(rank_id)
            if domain in domains:
                domains[domain] += contribution
            else:
                domains[domain] = contribution
        expected = torch.zeros_like(x, dtype=torch.float32)
        for domain in sorted(domains):
            expected += domains[domain].to(x.dtype).float()
        return expected


class RankFP32(RankSum):
    """NCCL-EP LL rank-major: one BF16 row per unique destination rank, summed in FP32 in top-k
    order, with no per-domain cast."""

    def reduce(self, torch, x, destination, valid, present, rank_message, scale_up_domain):
        return topk_rank_fp32_combine(
            torch, destination, valid, self._messages(torch, x, present, rank_message)
        )


class TopkSlotTree(RankSum):
    """A payload-dtype accumulator reduced by a pairwise tree (FlashInfer <= 0.6.15)."""

    def reduce(self, torch, x, destination, valid, present, rank_message, scale_up_domain):
        return topk_slot_tree_combine(
            torch, destination, valid, self._messages(torch, x, present, rank_message), x.dtype
        )


_RANK_SUM_REDUCTIONS = {"domain-fp32": DomainFP32, "rank-fp32": RankFP32, "topk-slot-tree": TopkSlotTree}


def combine_model(semantics, reduction="domain-fp32") -> CombineModel:
    """The model for a backend's declared (combine_weight_semantics, combine_reduction)."""
    if semantics == WeightedKernelSum.semantics:
        return WeightedKernelSum()
    if semantics != RankSum.semantics:
        raise ValueError(f"unknown combine semantics {semantics!r}")
    if reduction not in _RANK_SUM_REDUCTIONS:
        raise ValueError(f"unknown combine reduction {reduction!r}")
    return _RANK_SUM_REDUCTIONS[reduction]()


def topk_slot_tree_combine(torch, destination, valid, messages, dtype):
    """Reduce per-rank messages the way FlashInfer's payload-dtype accumulator does.

    acc[k] = message of destination[k] (0 if a lower k claimed that rank), then
    (a0+=a1)(a2+=a3)(a4+=a5)(a6+=a7); (a0+=a2)(a4+=a6); (a0+=a4), rounding at every level.
    Operands stay at their ORIGINAL top-k slot, so the tree shape depends on the routing. Memory is
    O(ep_size * tokens * hidden), ~8 GiB at EP16/8192 tokens: the term to stream before EP32.
    """
    tokens = torch.arange(destination.shape[0], device=destination.device)
    zero = torch.zeros_like(messages[0])
    slots = []
    for slot in range(destination.shape[1]):
        rank_id, claimed = _unique_rank_slots(destination, valid, slot)
        slots.append(torch.where(claimed.unsqueeze(1), messages[rank_id, tokens], zero))
    while len(slots) > 1:
        merged = [(slots[i] + slots[i + 1]).to(dtype).float() for i in range(0, len(slots) - 1, 2)]
        if len(slots) % 2:
            merged.append(slots[-1])
        slots = merged
    return slots[0]


def topk_rank_fp32_combine(torch, destination, valid, messages):
    """Unique-rank BF16 rows summed in FP32 in top-k order."""
    tokens = torch.arange(destination.shape[0], device=destination.device)
    combined = torch.zeros_like(messages[0])
    zero = torch.zeros_like(combined)
    for slot in range(destination.shape[1]):
        rank_id, claimed = _unique_rank_slots(destination, valid, slot)
        combined += torch.where(claimed.unsqueeze(1), messages[rank_id, tokens], zero)
    return combined


def chain_output_matches(chained, drained):
    """Whether a free-running chain's final output matches a drained pair through the same path.

    A regime A/B rather than an oracle: a mismatch is corruption that only appears under back-to-
    back pairs. Judged at the oracle tolerance (combines need not be order-deterministic); the
    returned magnitude separates a corruption from a result just outside it.
    Returns (within_tolerance, worst_relative_error).
    """
    if chained.shape != drained.shape:
        return False, float("inf")
    if not chained.numel():
        return True, 0.0
    error = (chained.float() - drained.float()).abs()
    worst = float((error / drained.float().abs().clamp_min(COMBINE_MAG_FLOOR)).max().item())
    # NaN would vanish from the cross-rank MAX and publish "failed, error 0.0".
    if not math.isfinite(worst):
        return False, float("inf")
    return worst < COMBINE_REL_TOL, worst


def _normalized_expert_metadata(torch, expert_ids, weights):
    """Sort each row by global expert ID while keeping -1 sentinels last."""
    keys = torch.where(expert_ids >= 0, expert_ids.to(torch.int64), torch.full_like(expert_ids, 1 << 30))
    order = torch.argsort(keys, dim=1, stable=True)
    sorted_ids = torch.gather(expert_ids.to(torch.int64), 1, order)
    sorted_weights = torch.gather(weights.to(torch.float32), 1, order)
    sorted_valid = sorted_ids >= 0
    return (
        torch.where(sorted_valid, sorted_ids, torch.full_like(sorted_ids, -1)),
        sorted_weights.masked_fill(~sorted_valid, 0),
    )


def _check_token_rank(torch, routing, backend, problem, view, source_ids, global_idx,
                      global_weights, rank, experts_per_rank, seed):
    """Rank-deduplicated receive: one row per source token, carrying its local experts."""
    receive_count = int(view.payload.shape[0])
    shape_ok = (
        view.payload.ndim == 2
        and view.expert_ids.shape == (receive_count, problem.topk_idx.shape[1])
        and view.weights.shape == view.expert_ids.shape
    )
    source_range = bool(
        receive_count == 0 or ((source_ids >= 0) & (source_ids < global_idx.shape[0])).all().item()
    )
    if source_range:
        expected_idx = global_idx.to(problem.x.device).index_select(0, source_ids)
        expected_weights = global_weights.to(problem.x.device).index_select(0, source_ids)
        local = (expected_idx // experts_per_rank) == rank
        expected_ids = torch.where(local, expected_idx, torch.full_like(expected_idx, -1))
        expected_weights = expected_weights.masked_fill(~local, 0)
        expected_payload = backend.semantic_payload(
            routing.activations_for_source_ids(source_ids, problem.x.shape[1], seed, problem.x.dtype)
        )
    else:
        expected_ids = torch.full_like(view.expert_ids, -1)
        expected_weights = torch.zeros_like(view.weights)
        expected_payload = torch.empty_like(view.payload)
    actual_ids, actual_weights = _normalized_expert_metadata(torch, view.expert_ids, view.weights)
    expected_ids, expected_weights = _normalized_expert_metadata(torch, expected_ids, expected_weights)
    expected_sources = (
        ((global_idx // experts_per_rank) == rank).any(dim=1).nonzero(as_tuple=True)[0]
    ).to(problem.x.device)
    max_weight_error = (
        float((actual_weights - expected_weights).abs().max().item()) if actual_weights.numel() else 0.0
    )
    expected_local = expected_ids[expected_ids >= 0] - rank * experts_per_rank
    expected_counts = torch.bincount(expected_local, minlength=experts_per_rank)
    checks = {
        "counts": torch.equal(view.local_expert_counts.to(torch.int64), expected_counts.to(torch.int64)),
        "metadata": shape_ok and torch.equal(actual_ids, expected_ids),
        "multiplicity": torch.equal((actual_ids >= 0).sum(dim=1), (expected_ids >= 0).sum(dim=1)),
        "payload": source_range and torch.equal(view.payload, expected_payload),
        "source_set": (
            source_range
            and source_ids.numel() == torch.unique(source_ids).numel()
            and torch.equal(torch.sort(source_ids).values, expected_sources)
        ),
        "weights": max_weight_error == 0.0,
    }
    return receive_count, actual_ids, actual_weights, max_weight_error, checks


def _check_token_expert(torch, routing, backend, problem, view, source_ids, global_idx,
                        global_weights, rank, experts_per_rank, seed):
    """Per-assignment receive (low-latency): one row per (source token, local expert).

    No gate weight is transported (the kernel applies it at the source), so weight correctness
    rides on combine_values and the weight check reports a populated zero error.
    """
    device = problem.x.device
    count = int(view.payload.shape[0])
    expert_ids = view.expert_ids.to(torch.int64).reshape(-1)
    local_lo = rank * experts_per_rank
    shape_ok = view.payload.ndim == 2 and tuple(expert_ids.shape) == (count,)
    source_range = bool(
        count == 0 or ((source_ids >= 0) & (source_ids < global_idx.shape[0])).all().item()
    )
    expert_range = bool(
        count == 0 or ((expert_ids >= local_lo) & (expert_ids < local_lo + experts_per_rank)).all().item()
    )
    # The exact (source, local-expert) multiset the global trace routes to this rank.
    exp_source, exp_slot = ((global_idx.to(device) // experts_per_rank) == rank).nonzero(as_tuple=True)
    expected_expert = global_idx.to(device)[exp_source, exp_slot]
    payload_ok = source_range and torch.equal(view.payload, backend.semantic_payload(
        routing.activations_for_source_ids(source_ids, problem.x.shape[1], seed, problem.x.dtype)
    ))
    source_set_ok = False
    if source_range and expert_range and count == int(expected_expert.numel()):
        # A 20-bit expert shift is safe: total experts stay far below 2^20.
        got_key = source_ids.to(torch.int64) * (1 << 20) + expert_ids
        want_key = exp_source.to(torch.int64) * (1 << 20) + expected_expert
        source_set_ok = bool(torch.equal(torch.sort(got_key).values, torch.sort(want_key).values))
    counts_ok = False
    if source_range and expert_range:
        actual_counts = view.local_expert_counts.to(torch.int64)
        expected_counts = torch.bincount(expected_expert - local_lo, minlength=experts_per_rank).to(torch.int64)
        counts_ok = tuple(actual_counts.shape) == (experts_per_rank,) and torch.equal(actual_counts, expected_counts)
    checks = {
        "counts": counts_ok,
        "metadata": shape_ok and source_range and expert_range,
        "multiplicity": source_set_ok,  # the exact multiset already fixes per-source multiplicity
        "payload": payload_ok,
        "source_set": source_set_ok,
        "weights": True,
    }
    # One valid expert per row with a unit coefficient: the kernel applies the gate.
    slot_weight = torch.ones((count, 1), dtype=torch.float32, device=device)
    return count, expert_ids.reshape(count, 1), slot_weight, 0.0, checks


def run_expert_oracle(torch, routing, backend, problem, global_idx, global_weights,
                      rank: int, experts_per_rank: int, scale_up_domain: int, seed: int):
    """Verify one real dispatch/transform/combine without entering a timed region."""
    handle = backend.dispatch(problem)
    torch.cuda.synchronize()
    try:
        view = backend.inspect_dispatch(problem, handle)
        source_ids = routing.decode_source_ids(view.payload, seed)
    except Exception as inspection_error:
        # Drain the in-flight dispatch before reporting: an abandoned handle deadlocks the peers.
        try:
            problem.recv_tokens = backend.recv_tokens(handle)
            backend.stage(problem, handle)
            backend.combine(problem, handle)
            torch.cuda.synchronize()
        except Exception as cleanup_error:
            raise inspection_error from cleanup_error
        return oracle_report(combine_weight_semantics=backend.combine_weight_semantics)

    check = _check_token_expert if backend.receive_layout == "token-expert" else _check_token_rank
    receive_count, ids, weights, max_weight_error, checks = check(
        torch, routing, backend, problem, view, source_ids, global_idx, global_weights,
        rank, experts_per_rank, seed,
    )
    problem.recv_tokens = receive_count
    model = backend.combine_model
    transformed = model.transform(torch, view.payload, ids, weights)
    combined = backend.combine_transformed(problem, handle, transformed)
    torch.cuda.synchronize()
    expected = model.expected(torch, problem, experts_per_rank, scale_up_domain)
    max_absolute_error = max_relative_error = None
    checks["combine_values"] = False
    if combined.shape == expected.shape:
        # Zero errors stand when the rank legitimately combined nothing.
        max_absolute_error = max_relative_error = 0.0
        checks["combine_values"] = True
        if combined.numel():
            absolute_error = (combined.float() - expected).abs()
            max_absolute_error = float(absolute_error.max().item())
            max_relative_error = float(
                (absolute_error / expected.abs().clamp_min(COMBINE_MAG_FLOOR)).max().item()
            )
            checks["combine_values"] = max_relative_error < COMBINE_REL_TOL
    return oracle_report(
        passed=all(checks.values()),
        combine_weight_semantics=model.semantics,
        receive_count=receive_count,
        max_absolute_error=max_absolute_error,
        max_elementwise_relative_error=max_relative_error,
        max_weight_error=max_weight_error,
        checks={name: checks[name] for name in ORACLE_CHECKS},
    )
