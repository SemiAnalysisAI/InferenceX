#!/usr/bin/env python3
"""The EP sweep driver: case identity, the sampling passes and the result document."""
from __future__ import annotations

import argparse
import datetime as _dt
from dataclasses import dataclass, field
import json
import math
import os
import re

from ep_oracle import (
    MODE_ALLOWED_SEMANTICS,
    ORACLE_CHECKS,
    ORACLE_MODELED_CONTRACTS,
    chain_output_matches,
    run_expert_oracle,
)

_CASE_ID = re.compile(r"^[a-z0-9][a-z0-9.-]*$")
_NON_SLUG = re.compile(r"[^a-z0-9]+")


def is_case_id(value) -> bool:
    return bool(isinstance(value, str) and _CASE_ID.fullmatch(value))


def slug_id(parts) -> str:
    values = [_NON_SLUG.sub("-", str(part).lower()).strip("-") for part in parts]
    if not all(values):
        raise ValueError("case ID contains an empty factor")
    return "-".join(values)


def case_id(sku: str, case: dict) -> str:
    return slug_id((
        sku, case["backend"], case["workload"], case["mode"], case["phase"],
        f"ep{int(case['ep'])}", case["routing"], case["precision"],
    ))


# Workload and timing values arrive from configs/sweep.json through the matrix.
CONDITIONING_ROUNDS_PER_SHAPE = 8


def logical_byte_provenance(logical_copies: int, hidden: int, value_bytes: int = 2,
                            scale_bytes_per_copy: int = 0) -> dict[str, int]:
    """Comparable logical activation bytes for one direction. BF16 moves 2 bytes/value with no
    scales; FP8 moves 1 byte/value, plus per-block FP32 scales for a blockwise codec (DeepEP) and
    none for a plain e4m3 cast (MoRI). Combine is always BF16."""
    if logical_copies < 0 or hidden < 0:
        raise ValueError("logical byte dimensions must be non-negative")
    if value_bytes <= 0 or scale_bytes_per_copy < 0:
        raise ValueError("value_bytes must be positive and scale bytes non-negative")
    activation_data_bytes = logical_copies * hidden * value_bytes
    scale_bytes = logical_copies * scale_bytes_per_copy
    return {
        "activation_data_bytes": activation_data_bytes,
        "scale_bytes": scale_bytes,
        "total_logical_bytes": activation_data_bytes + scale_bytes,
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
    ap.add_argument("--mode", required=True, choices=["normal", "low-latency"])
    ap.add_argument("--precision", required=True, choices=["bf16", "fp8"],
                    help="dispatch payload precision; combine is always BF16")
    ap.add_argument("--phase", required=True, choices=["decode", "prefill"],
                    help="token-size regime label: decode (small T) / prefill (large T)")
    ap.add_argument("--tokens-ladder", required=True,
                    help="space/comma-separated source-tokens-per-rank sweep from configs/sweep.json")
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
    ap.add_argument("--version", type=int, required=True,
                    help="iterable benchmark version copied verbatim into the emitted result")
    # The single cross-SKU timing profile (configs/sweep.json `timing:`), baked in by the matrix.
    ap.add_argument("--warmup", type=int, required=True,
                    help="untimed full roundtrips before each trial/point")
    ap.add_argument("--iters", type=int, required=True, help="timed iterations per trial")
    ap.add_argument("--trials", type=int, required=True, help="timed trials")
    # One chain call already yields chain_iters pairs, so the chain needs far fewer trials.
    ap.add_argument("--chain-iters", type=int, default=128,
                    help="free-running dispatch->combine pairs per chain trial")
    ap.add_argument("--chain-trials", type=int, default=4, help="chain trials per ladder point")
    ap.add_argument("--chain-drop", type=int, default=16,
                    help="head pairs discarded per chain trial (pipeline fill, not period)")
    ap.add_argument("--runner", required=True)
    ap.add_argument("--topology-class", required=True)
    ap.add_argument("--transport", required=True)
    ap.add_argument("--scope", required=True, choices=["scale-up", "scale-out"])
    ap.add_argument("--scale-up-transport", required=True)
    ap.add_argument("--scale-out-transport", required=True)
    ap.add_argument("--gpus-per-node", type=int, required=True)
    ap.add_argument("--scale-up-domain", type=int, required=True)
    ap.add_argument("--out", required=True)


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
    return s[max(0, min(len(s) - 1, math.ceil(q / 100.0 * len(s)) - 1))]


def _pcts(xs):
    return ({"p50": percentile(xs, 50), "p90": percentile(xs, 90),
             "p95": percentile(xs, 95), "p99": percentile(xs, 99)} if xs else None)


# Consumer contract, not labels: the frontend and durable store key the headline on
# `components.pair_period` carrying exactly CHAIN_PERIOD_ORIGIN, so a typo fails silently.
CHAIN_PERIOD_ORIGIN = "chained-median"
CHAIN_FLOOR_ORIGIN = "chained-cross-rank-min"
CUDA_GRAPH_ORIGIN = "cuda-graph-replay"


def _published_tails(percentiles, graph_replay):
    """Under graph replay only the median: a rank whose host is late to launch its replay stalls
    the others, so graphed fresh-entry tails measure host jitter (gb200 flashinfer T=1 roundtrip
    p99 858us vs combine p99 30us)."""
    if not graph_replay or percentiles is None:
        return percentiles
    return {key: (value if key == "p50" else None) for key, value in percentiles.items()}


def _component(percentiles, count, *, derived=False, origin=None):
    """One component block. `origin` names the reduction wherever it is not the default
    per-iteration cross-rank MAX, so a consumer never infers it from the field name."""
    if percentiles is None:
        return {"availability": "unavailable", "origin": None, "percentiles_us": None, "sample_count": 0}
    return {
        "availability": "derived" if derived else "measured",
        "origin": origin or ("derived-percentile-sum" if derived else "measured"),
        "percentiles_us": percentiles,
        "sample_count": 0 if derived else count,
    }


# The routing fields each row publishes: a whitelist so a new stat never leaks in unreviewed.
_ROUTING_FIELDS = (
    "empty_expert_count", "empty_rank_count", "expert_assignment_rank_cv",
    "expert_assignments_per_rank", "expert_load_cv", "expert_load_max",
    "expert_load_mean", "expert_load_min", "fanout_histogram", "fanout_max",
    "fanout_mean", "fanout_min", "hotspot_ratio", "locality",
    "payload_copies_per_rank", "payload_rank_cv", "routed_copies",
)


def _write_json_atomic(path: str, value) -> None:
    payload = json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":")).encode() + b"\n"
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


def kernel_generation(backend) -> str:
    """The adapter's kernel family; `-cudagraph` keeps replayed rows a separate series."""
    family = backend.kernel_generation or "n-a"
    return f"{family}-cudagraph" if backend.cuda_graph_enabled else family


class _Collectives:
    """The cross-rank reductions the sweep publishes, over one torch.distributed group."""

    def __init__(self, torch, dist, device):
        self.torch, self.dist, self.device = torch, dist, device

    def vec(self, vals, op):
        t = self.torch.tensor(vals, device=self.device, dtype=self.torch.float64)
        self.dist.all_reduce(t, op=op)
        return [float(x) for x in t.tolist()]

    def median_spread(self, vals):
        """Per-element cross-rank (MEDIAN, MAX-MIN) from one all_gather. The pair period is a rate:
        MAX would publish whichever rank hiccuped, while the median is the agreed cadence and a
        large spread marks a paced rank. Rank-ordered, so every rank writes the same artifact."""
        stacked = self.torch.stack(self._gather(self.torch.tensor(vals, device=self.device, dtype=self.torch.float64)))
        median = stacked.median(dim=0).values
        spread = stacked.max(dim=0).values - stacked.min(dim=0).values
        return [float(x) for x in median.tolist()], [float(x) for x in spread.tolist()]

    def scalars(self, val):
        """Every rank's copy of one per-rank scalar, in rank order."""
        return [float(g.item()) for g in self._gather(
            self.torch.tensor([float(val)], device=self.device, dtype=self.torch.float64))]

    def integer(self, v, op) -> int:
        t = self.torch.tensor([int(v)], device=self.device, dtype=self.torch.int64)
        self.dist.all_reduce(t, op=op)
        return int(t.item())

    def all(self, flag) -> bool:
        return bool(self.integer(int(bool(flag)), self.dist.ReduceOp.MIN))

    def _gather(self, local):
        gathered = [self.torch.empty_like(local) for _ in range(self.dist.get_world_size())]
        self.dist.all_gather(gathered, local)
        return gathered


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


class CaseError(Exception):
    """A case the harness refuses to run: rank 0 prints it and the leg exits 2."""


@dataclass
class Point:
    """One ladder point: its problem and trace, every sample series, and every correctness verdict.

    Fresh-entry series carry one value per timed iteration pooled across trials: MAX-reduced for
    the published latencies, MIN-reduced (`_min`) for the skew-excluded companions, `spread` the
    roundtrip's max-min. The chained family runs on its own trials: `chain`/`chain_spread` are
    per-iteration, `gap` and `settle` one scalar per trial.
    """

    T: int
    problem: object
    global_idx: object
    global_weights: object
    snapshot: tuple
    rstats: dict
    oracle_pre: dict
    pre_input_unchanged: bool
    oracle_chain: "dict | None" = None
    oracle_post: "dict | None" = None
    # Chained output vs a drained pair, ANDed across chain trials; the worst error is kept even
    # when passing, since a magnitude creeping toward the tolerance is the early warning.
    chain_output_ok: bool = True
    chain_output_error: float = 0.0
    graph_output_ok: bool = True
    graph_output_error: float = 0.0
    graph_output_rewritten: bool = True
    dispatch: list = field(default_factory=list)
    stage: list = field(default_factory=list)
    combine: list = field(default_factory=list)
    roundtrip: list = field(default_factory=list)
    spread: list = field(default_factory=list)
    dispatch_min: list = field(default_factory=list)
    combine_min: list = field(default_factory=list)
    roundtrip_min: list = field(default_factory=list)
    chain: list = field(default_factory=list)
    chain_spread: list = field(default_factory=list)
    dispatch_floor: list = field(default_factory=list)
    combine_floor: list = field(default_factory=list)
    gap: list = field(default_factory=list)
    settle: list = field(default_factory=list)

    def inputs_unchanged(self, torch):
        x, idx, weights = self.snapshot
        p = self.problem
        return torch.equal(p.x, x) and torch.equal(p.topk_idx, idx) and torch.equal(p.topk_weights, weights)


class Sweep:
    """Drive the source-tokens-per-rank sweep for one fully specified case.

    Pass 1 warms each shape and runs the expert oracle; Pass 2 samples the fresh-entry components;
    graph mode then value-checks a poisoned replay; Pass 2b samples the chained family and gates
    its final state; Pass 3 proves the inputs were immutable and repeats the oracle; the rows and
    document follow. Every collective runs in the same order on every rank.
    """

    def __init__(self, args, backend, torch, dist, device, rank: int, world_size: int):
        self.args, self.backend = args, backend
        self.torch, self.dist, self.device = torch, dist, device
        self.rank, self.world_size = rank, world_size

    def run(self) -> int:
        try:
            self._validate()
        except CaseError as error:
            if self.rank == 0:
                print(f"ERROR: {error}")
            return 2
        spec = self.backend.make_inputs(self.args)
        if not spec.ok:
            if self.rank == 0:
                print(f"ERROR: {spec.message}")
            return spec.rc
        self.cap, self.ladder, self.dropped = spec.cap, spec.ladder, spec.dropped
        if self.rank == 0 and self.dropped:
            print(f"NOTE: dropped tokens/rank {self.dropped} — exceed {self.backend.name} buffer cap "
                  f"{self.cap} (hidden={self.args.hidden}); not silently truncated.")
        self.graph = self.backend.cuda_graph_enabled
        self.backend.create_buffer(spec)
        # create_buffer may set the reduction (FlashInfer picks it by wheel version), so resolve the
        # oracle's combine model here: a bad declaration fails on every rank before any dispatch,
        # not inside the oracle with one in flight.
        try:
            self.backend.combine_model
        except ValueError as error:
            if self.rank == 0:
                print(f"ERROR: {error}")
            return 2
        # The chained-output A/B is defined only when each pair stages its own input: under the
        # hoist (every FP8 adapter by default) the staged stand-in matches neither pair's dispatch,
        # so chained and drained are two differently mismatched pairs (h100 deepep-v2 EP8: hoisted
        # error 31..93, per-pair 0.0).
        self.chain_output_applicable = not self.backend.stage_excluded_from_roundtrip
        self.reduce = _Collectives(self.torch, self.dist, self.device)
        prepared = [self._prepare(spec.points[T], T) for T in self.ladder]
        self.points = [point for point, _ in prepared]
        # status=valid also requires a proven-identical routing trace across ranks.
        self.routing_consistent = all([consistent for _, consistent in prepared])
        self._sample_fresh()
        if self.graph:
            self._check_graph_replay()
        self._sample_chain()
        for point in self.points:
            self._final_oracle(point)
        rows = [self._row(point) for point in self.points]
        all_ok = bool(rows) and all(r["correctness"]["passed"] for r in rows) and self.routing_consistent
        doc = self._document(rows, all_ok)
        if self.rank == 0:
            _write_json_atomic(self.args.out, doc)
            self._print_summary(rows, doc)
        # The return code is CI's only success signal (the doc uploads regardless), so a captured
        # `invalid` outcome must fail the leg; agreed across ranks so the case fails as one.
        return 0 if self.reduce.all(all_ok) else 3

    def _validate(self):
        args, backend = self.args, self.backend
        if args.mode not in MODE_ALLOWED_SEMANTICS:
            raise CaseError(f"unknown CollectiveX case mode {args.mode!r}")
        if min(args.iters, args.trials, args.warmup) <= 0:
            raise CaseError(f"iters/trials/warmup must be positive; got "
                            f"{args.iters}:{args.trials}:{args.warmup}")
        # Two kept pairs, or `pair_period` publishes degenerate and the health scalars vanish.
        if (min(args.chain_iters, args.chain_trials) <= 0
                or not 0 <= args.chain_drop <= args.chain_iters - 2):
            raise CaseError(f"chain iters/trials must be positive and 0 <= drop <= iters - 2; got "
                            f"{args.chain_iters}:{args.chain_trials}:{args.chain_drop}")
        import routing  # torch-based; imported lazily so the module byte-compiles without torch

        self.routing = routing
        if args.experts % self.world_size != 0:
            raise CaseError(f"experts ({args.experts}) must divide ep_size ({self.world_size})")
        self.experts_per_rank = args.experts // self.world_size
        if backend.mode != args.mode:
            raise CaseError(f"backend mode {backend.mode!r} != {args.mode!r}")
        allowed = MODE_ALLOWED_SEMANTICS[args.mode]
        if backend.combine_weight_semantics not in allowed:
            raise CaseError(f"{args.mode} requires combine semantics in {sorted(allowed)}; "
                            f"backend declares {backend.combine_weight_semantics!r}")
        contract = (backend.receive_layout, backend.combine_weight_semantics)
        if contract not in ORACLE_MODELED_CONTRACTS:
            raise CaseError("no correctness oracle models receive_layout="
                            f"{contract[0]!r} with combine semantics {contract[1]!r}")
        # A non-control precision must realize a non-BF16 wire, or the artifact is mislabeled.
        if args.precision != "bf16" and backend.dispatch_dtype == "bf16":
            raise CaseError(f"precision {args.precision!r} did not realize a non-BF16 dispatch "
                            f"dtype (backend reports {backend.dispatch_dtype!r})")

    def _oracle(self, point):
        return run_expert_oracle(
            self.torch, self.routing, self.backend, point.problem, point.global_idx,
            point.global_weights, self.rank, self.experts_per_rank, self.args.scale_up_domain,
            self.args.seed,
        )

    def _prepare(self, inputs, T):
        """Pass 1, ascending: warm untimed so clocks and fabric settle, prove the routing trace is
        identical across ranks, then run the expert oracle."""
        torch, args = self.torch, self.args
        problem = self.backend.make_problem(
            T, inputs.topk_idx.to(self.device), inputs.topk_weights.to(self.device), inputs.activations
        )
        self.backend.warm(problem, CONDITIONING_ROUNDS_PER_SHAPE)
        torch.cuda.synchronize()
        idx_g, w_g = inputs.global_idx, inputs.global_weights
        rstats = self.routing.routing_stats(idx_g, args.experts, self.experts_per_rank)
        rstats["locality"] = self.routing.routing_locality(
            idx_g, self.experts_per_rank, self.world_size, max(1, T), args.gpus_per_node,
            args.scale_up_domain,
        )
        consistent = _same_tensors_across_ranks(torch, self.dist, self.device, idx_g, w_g)
        point = Point(
            T=T, problem=problem, global_idx=idx_g, global_weights=w_g,
            snapshot=(problem.x.clone(), problem.topk_idx.clone(), problem.topk_weights.clone()),
            rstats=rstats, oracle_pre=None, pre_input_unchanged=False,
        )
        point.oracle_pre = self._oracle(point)
        point.pre_input_unchanged = point.inputs_unchanged(torch)
        return point, consistent

    def _sample_fresh(self):
        """Pass 2: fresh-entry components in a rotated point and component order, reduced per
        iteration across ranks. The spread (max-min) marks skew-inflated points: when ranks enter
        together every rank measures nearly the same duration."""
        MAX, MIN = self.dist.ReduceOp.MAX, self.dist.ReduceOp.MIN
        by_T = {point.T: point for point in self.points}
        for trial_index in range(self.args.trials):
            for T in trial_order(list(self.ladder), trial_index):
                point = by_T[T]
                measured = {name: [] for name in ("dispatch", "stage", "combine", "roundtrip")}
                for name in trial_order(self.backend.timed_components(), trial_index):
                    measured[name] = self.backend.benchmark_component(
                        name, point.problem, self.args.warmup, self.args.iters
                    )
                if measured["dispatch"]:
                    point.dispatch += self.reduce.vec(measured["dispatch"], MAX)
                    point.combine += self.reduce.vec(measured["combine"], MAX)
                if measured["stage"]:
                    point.stage += self.reduce.vec(measured["stage"], MAX)
                rt_max = self.reduce.vec(measured["roundtrip"], MAX)
                point.roundtrip += rt_max
                rt_min = self.reduce.vec(measured["roundtrip"], MIN)
                point.spread += [hi - lo for hi, lo in zip(rt_max, rt_min)]
                point.roundtrip_min += rt_min
                if measured["dispatch"]:
                    point.dispatch_min += self.reduce.vec(measured["dispatch"], MIN)
                    point.combine_min += self.reduce.vec(measured["combine"], MIN)

    def _check_graph_replay(self):
        """The timed replays must have rewritten their poisoned output, and a poisoned replay with
        staging inside the capture must match a drained pair. Untimed, in ladder order."""
        for point in self.points:
            point.graph_output_rewritten &= bool(getattr(point.problem, "_cuda_graph_output_rewritten", False))
            replayed = self.backend.graph_replay_output(point.problem)
            drained = self.backend.run_roundtrip(point.problem)
            self.torch.cuda.synchronize()
            ok, error = chain_output_matches(replayed, drained)
            point.graph_output_ok &= ok
            point.graph_output_error = max(point.graph_output_error, error)

    def _sample_chain(self):
        """Pass 2b: the chained family on its own trial count, then the chained oracle against the
        state each point's last chain left behind. The drained reference pair is collective and
        runs in the same (trial, T) order on every rank even where the A/B does not apply."""
        MIN = self.dist.ReduceOp.MIN
        by_T = {point.T: point for point in self.points}
        for trial_index in range(self.args.chain_trials):
            final = trial_index == self.args.chain_trials - 1
            for T in trial_order(list(self.ladder), trial_index):
                point = by_T[T]
                chained = self.backend.benchmark_chain(
                    point.problem, self.args.warmup, self.args.chain_iters, self.args.chain_drop
                )
                drained = self.backend.run_roundtrip(point.problem)
                self.torch.cuda.synchronize()
                if self.chain_output_applicable:
                    ok, error = chain_output_matches(chained["combined"], drained)
                    point.chain_output_ok &= ok
                    point.chain_output_error = max(point.chain_output_error, error)
                pair = chained["pair"]
                median, spread = self.reduce.median_spread(pair)
                point.chain += median
                point.chain_spread += spread
                # Per-op floors: cross-rank MIN only, since chained windows park each rank's wait.
                point.dispatch_floor += self.reduce.vec(chained["dispatch"], MIN)
                point.combine_floor += self.reduce.vec(chained["combine"], MIN)
                # Health, one scalar per trial: the per-pair cost outside the window (median across
                # ranks), and the late-minus-early-half drift (signed max magnitude).
                gaps = self.reduce.scalars(_pcts(chained["start_to_start"])["p50"] - _pcts(pair)["p50"])
                point.gap.append(_pcts(gaps)["p50"])
                half = len(pair) // 2
                drifts = self.reduce.scalars(_pcts(pair[half:])["p50"] - _pcts(pair[:half])["p50"])
                point.settle.append(max(drifts, key=abs))
                if final:
                    point.oracle_chain = self._oracle(point)

    def _final_oracle(self, point):
        """Pass 3: the inputs were never written, and the full oracle still passes."""
        point.input_unchanged = point.pre_input_unchanged and point.inputs_unchanged(self.torch)
        point.oracle_post = self._oracle(point)
        assert point.oracle_chain is not None, "chained oracle missing despite a validated budget"
        point.chain_ok = bool(point.oracle_chain["passed"])
        point.graph_ok = point.graph_output_rewritten and (point.graph_output_ok or not self.graph)
        point.local_ok = (
            point.oracle_pre["passed"] and point.oracle_post["passed"] and point.chain_ok
            and point.input_unchanged
            and (point.chain_output_ok or not self.chain_output_applicable)
            and point.graph_ok
        )
        point.max_rel = max(
            report["max_elementwise_relative_error"] or 0.0
            for report in (point.oracle_pre, point.oracle_post, point.oracle_chain)
        )

    def _bytes(self, copies):
        """(dispatch, combine, roundtrip) byte blocks for `copies` logical copies."""
        backend, hidden = self.backend, self.args.hidden
        dispatch = logical_byte_provenance(copies, hidden, backend.dispatch_value_bytes,
                                           backend.dispatch_scale_bytes_per_copy)
        combine = logical_byte_provenance(copies, hidden)
        return dispatch, combine, {key: dispatch[key] + combine[key] for key in dispatch}

    def _correctness(self, point):
        """The correctness block, every verdict agreed across ranks (MIN, or MAX for magnitudes)."""
        reduce, MAX = self.reduce, self.dist.ReduceOp.MAX
        oracle_passed, oracle_failed_checks = {}, {}
        for name, report in (("pre", point.oracle_pre), ("chained", point.oracle_chain),
                             ("post", point.oracle_post)):
            oracle_passed[name] = reduce.all(report["passed"])
            oracle_failed_checks[name] = [
                check for check in ORACLE_CHECKS if not reduce.all(report["checks"][check])
            ]
        post_chain_state_passed = reduce.all(point.chain_ok)
        # Reduced on every rank to keep collectives aligned; null where the A/B does not apply, so
        # "not asked" never reads as a failed comparison.
        chain_passed = reduce.all(point.chain_output_ok)
        chain_error = reduce.vec([point.chain_output_error], MAX)[0]
        if not self.chain_output_applicable:
            chain_passed = chain_error = None
        graph_rewritten = graph_passed = graph_error = None
        if self.graph:
            graph_rewritten = reduce.all(point.graph_output_rewritten)
            graph_passed = reduce.all(point.graph_output_ok)
            graph_error = reduce.vec([point.graph_output_error], MAX)[0]
        return {
            # The chain's own final output vs a drained pair: proves the last pair of each chain,
            # never the interior ones. Read beside the error, never alone.
            "chain_last_output_passed": chain_passed,
            "chain_last_output_error": chain_error,
            # The full oracle against the state the chain left behind (a fresh pair afterwards).
            "post_chain_state_passed": post_chain_state_passed,
            "cuda_graph_output_rewritten": graph_rewritten,
            "cuda_graph_last_output_passed": graph_passed,
            "cuda_graph_last_output_error": graph_error,
            "max_relative_error": None,  # filled after `passed` below keeps collective order
            "oracle_passed": oracle_passed,
            "oracle_failed_checks": oracle_failed_checks,
            "passed": None,
        }

    def _row(self, point):
        reduce, ops = self.reduce, self.dist.ReduceOp
        T, rstats, graph = point.T, point.rstats, self.graph
        global_tokens = T * self.world_size
        dp, sp, cp, rtp = _pcts(point.dispatch), _pcts(point.stage), _pcts(point.combine), _pcts(point.roundtrip)
        chainp = _pcts(point.chain)
        origin = CUDA_GRAPH_ORIGIN if graph else None

        def pub(pcts):
            return _published_tails(pcts, graph)

        def measured(samples, pcts=None):
            return _component(pub(pcts or _pcts(samples)), len(samples), origin=origin)

        # isolated_sum adds the isolated percentiles; not a measured chained operation, never a
        # throughput basis. Stage contributes zero where it is not timed.
        isum = ({key: dp[key] + (sp[key] if sp is not None else 0.0) + cp[key] for key in dp}
                if dp and cp else None)
        received = point.oracle_pre["receive_count"]
        recv_total = reduce.integer(received, ops.SUM)
        recv_max = reduce.integer(received, ops.MAX)
        recv_min = reduce.integer(received, ops.MIN)
        global_ok = reduce.integer(point.local_ok, ops.MIN)
        correctness = self._correctness(point)
        correctness["max_relative_error"] = reduce.vec([point.max_rel], ops.MAX)[0]
        correctness["passed"] = point_ok = bool(global_ok) and recv_total > 0
        dispatch_bytes, combine_bytes, roundtrip_bytes = self._bytes(rstats["routed_copies"])
        # The wire basis is a property of the RECEIVE (MoRI's LL kernels deduplicate where the other
        # LL kernels do not). Bandwidth must divide the wire bytes; `byte_provenance` is the
        # comparable, rank-deduplicated basis and a lower bound for per-assignment receives.
        assignment_copies = int(sum(rstats["expert_assignments_per_rank"]))
        wire_basis = "per-assignment" if self.backend.receive_layout == "token-expert" else "rank-deduplicated"
        wire_copies = assignment_copies if wire_basis == "per-assignment" else int(rstats["routed_copies"])
        wire_dispatch, wire_combine, wire_roundtrip = self._bytes(wire_copies)
        stage_bytes = dict.fromkeys(dispatch_bytes, 0)
        throughput = {name: (global_tokens / (latency * 1e-6) if latency is not None else None)
                      for name, latency in pub(rtp).items()}
        row = {
            "components": {
                "combine": measured(point.combine, cp),
                "dispatch": measured(point.dispatch, dp),
                "isolated_sum": _component(pub(isum), 0, derived=True),
                # What a decode loop pays per MoE layer: the steady-state period of back-to-back
                # pairs, cross-rank median. Do not sum it.
                "pair_period": _component(chainp, len(point.chain), origin=CHAIN_PERIOD_ORIGIN),
                "roundtrip": measured(point.roundtrip, rtp),
                "stage": _component(sp, len(point.stage)),
            },
            # The last-entering rank's op windows from the floors chain; tracks kernel time ~10%.
            "chain_floor_us": {
                "combine": _component(_pcts(point.combine_floor), len(point.combine_floor), origin=CHAIN_FLOOR_ORIGIN),
                "dispatch": _component(_pcts(point.dispatch_floor), len(point.dispatch_floor), origin=CHAIN_FLOOR_ORIGIN),
            },
            # Whether the chain was the steady state the period claims.
            "chain_health": {
                "interpair_gap_us": _component(_pcts(point.gap), len(point.gap)),
                "pair_spread_us": _component(_pcts(point.chain_spread), len(point.chain_spread)),
                "settle_drift_us": _component(_pcts(point.settle), len(point.settle)),
            },
            # Same iterations reduced by cross-rank MIN: how much of a point is rank stagger.
            "cross_rank_min_us": {
                "combine": measured(point.combine_min),
                "dispatch": measured(point.dispatch_min),
                "roundtrip": measured(point.roundtrip_min),
            },
            # Diagnostic, not a latency: large next to the roundtrip means skew-inflated.
            "cross_rank_spread_us": _component(_pcts(point.spread), len(point.spread)),
            "correctness": correctness,
            "global_tokens": global_tokens,
            "byte_provenance": {
                "combine": combine_bytes, "dispatch": dispatch_bytes,
                "roundtrip": roundtrip_bytes, "stage": stage_bytes,
            },
            "wire_byte_provenance": {
                "combine": wire_combine, "dispatch": wire_dispatch,
                "roundtrip": wire_roundtrip, "stage": stage_bytes,
            },
            "logical_copies": {
                "routed": int(rstats["routed_copies"]),
                "assignments": assignment_copies,
                "wire": wire_basis,
            },
            "receive": {"max": recv_max, "mean": recv_total / self.world_size, "min": recv_min, "total": recv_total},
            "routing": {key: rstats[key] for key in _ROUTING_FIELDS},
            "token_rate_at_latency_percentile": throughput,
            "tokens_per_rank": T,
        }
        if self.rank == 0:
            component_log = (f"disp p50/p99={dp['p50']:7.1f}/{dp['p99']:7.1f} "
                             f"comb {cp['p50']:6.1f}/{cp['p99']:6.1f} " if dp and cp
                             else "components=unavailable ")
            period_log = f"period={chainp['p50']:7.1f}us " if chainp else "period=n/a "
            print(f"  T={T:<5} {component_log}{period_log}"
                  f"RT p50/p99={rtp['p50']:7.1f}/{rtp['p99']:7.1f}us n={len(point.roundtrip)} "
                  f"fanout={rstats['fanout_mean']:.2f} "
                  f"recv[min/mean/max]={recv_min}/{recv_total // self.world_size}/{recv_max} "
                  f"correct={point_ok}")
        return row

    def _document(self, rows, all_ok):
        args, backend = self.args, self.backend
        nodes = int(os.environ.get("SLURM_NNODES", "1"))
        transport = {
            "scale_up_domain": args.scale_up_domain,
            "scale_up_transport": args.scale_up_transport,
            "scale_out_transport": args.scale_out_transport or None,
            "scope": args.scope,
        }
        scheduled_case = {
            "backend": backend.name, "ep": self.world_size, "experts": args.experts,
            "gpus_per_node": args.gpus_per_node, "hidden": args.hidden,
            "ladder": " ".join(map(str, self.ladder)), "mode": args.mode, "nodes": nodes,
            "phase": args.phase, "precision": args.precision, "routing": args.routing,
            **transport, "suite": args.suite, "topk": args.topk,
            "topology_class": args.topology_class, "transport": args.transport,
            "workload": args.workload_name,
        }
        computed = case_id(args.runner, scheduled_case)
        if args.case_id != computed:
            raise ValueError(f"scheduled case ID does not match realized factors: {args.case_id} != {computed}")
        git_run = getattr(args, "git_run", None) or {}
        try:
            attempt_ordinal = int(os.environ.get("COLLX_ATTEMPT_ID", "1"))
        except ValueError:
            attempt_ordinal = 0
        if attempt_ordinal <= 0:
            raise ValueError("COLLX_ATTEMPT_ID must be a positive integer")
        return {
            "version": args.version,
            "record_type": "case-attempt",
            "generated_at": _dt.datetime.now().astimezone().isoformat(),
            "identity": {
                "allocation_factors": {key: git_run.get(key) for key in ("run_attempt", "run_id", "source_sha")},
                "attempt_ordinal": attempt_ordinal,
                "case_factors": {"case": scheduled_case, "sku": args.runner},
                "case_id": args.case_id,
            },
            "workload": {
                "cross_rank_consistent": self.routing_consistent,
                "ladder_measured": list(self.ladder),
                "ladder_dropped": list(self.dropped),
                "ladder_cap": self.cap,
            },
            "measurement": {
                "combine_dtype": backend.combine_dtype,
                "combine_semantics": "activation-only",
                "dispatch_dtype": backend.dispatch_dtype,
                "payload_unit": "token-rank",
                "rows": rows,
                "sampling": {
                    "iterations_per_trial": args.iters,
                    "samples_per_component": args.iters * args.trials,
                    "trials": args.trials,
                    "warmup_iterations": args.warmup,
                    # The chained family's own sampling: 128x4 is not 512x1.
                    "chain_drop": args.chain_drop,
                    "chain_iterations_per_trial": args.chain_iters,
                    "chain_trials": args.chain_trials,
                },
            },
            "implementation": {
                "fp8_consume": backend.fp8_consume,
                "kernel_generation": kernel_generation(backend),
                # The reduction the oracle held the kernel to, and the library version it was
                # chosen from: a wheel bump must not silently change the arithmetic behind `passed`.
                "combine_reduction": backend.combine_reduction,
                "library_version": backend.library_version,
                "stage_excluded_from_roundtrip": bool(backend.stage_excluded_from_roundtrip),
                "chained_period": True,
                "cuda_graph_replay": self.graph,
                "cuda_graph_supported": bool(backend.cuda_graph_supported),
                # A "candidate" row measures the library, not a deployment (EPBackend.maturity).
                "maturity": backend.maturity or "unknown",
                "name": backend.name,
            },
            "topology": {
                "device_product": getattr(args, "runtime_device_product", None),
                "gpus_per_node": args.gpus_per_node,
                "nodes": nodes,
                "placement": "packed",
                **{key: transport[key] for key in ("scale_up_domain", "scale_up_transport", "scale_out_transport", "scope")},
                "topology_class": args.topology_class,
                "transport": args.transport,
                "world_size": self.world_size,
            },
            "runtime": getattr(args, "runtime", {}),
            "provenance": {"image": getattr(args, "image", "") or None, "source_sha": git_run.get("source_sha")},
            "outcome": {
                "reasons": [] if all_ok else ["semantic correctness or routing identity failed"],
                "status": "success" if all_ok else "invalid",
            },
        }

    def _print_summary(self, rows, doc):
        # Ladder ends plus two interior points: one mid-ladder headline hides low-token behavior.
        summary_rows = []
        for tokens in (self.ladder[0], 8, 64, self.ladder[-1]):
            row = next((r for r in rows if r["tokens_per_rank"] == tokens), None)
            if row is not None and row not in summary_rows:
                summary_rows.append(row)

        def point_summary(row):
            period = row["components"]["pair_period"]["percentiles_us"]
            period_summary = f" period_p50={period['p50']:.1f}us" if period else ""
            percentiles = row["components"]["dispatch"]["percentiles_us"]
            if not percentiles:
                return f"T={row['tokens_per_rank']}:n/a{period_summary}"
            if percentiles.get("p99") is None:
                return f"T={row['tokens_per_rank']}:disp_p50={percentiles['p50']:.1f}us{period_summary}"
            return f"T={row['tokens_per_rank']}:disp_p99={percentiles['p99']:.1f}us{period_summary}"

        print(f"{self.backend.name} ep-dispatch-combine [{self.args.phase}/{self.args.mode}]: "
              f"status={doc['outcome']['status']} {len(rows)} pts, routing_consistent={self.routing_consistent}, "
              f"{' '.join(point_summary(row) for row in summary_rows)} "
              f"-> {self.args.out}")


def run_sweep(args, backend, torch, dist, device, rank: int, world_size: int) -> int:
    """Drive the source-tokens-per-rank sweep for one fully-specified line."""
    return Sweep(args, backend, torch, dist, device, rank, world_size).run()
