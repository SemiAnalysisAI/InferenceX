"""CUDA-event timing, sample storage, and cross-rank reductions."""
from __future__ import annotations

from dataclasses import dataclass, field
import math

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


def time_us(torch, fn, warmup: int, iters: int, pre=None, post=None) -> list[float]:
    """Per-iteration CUDA-event latencies (µs) for THIS rank.

    Without `pre`: times `fn()`. With `pre`: runs `pre()` UNTIMED each iteration, then times
    `fn(pre_result)`. `post(result)` runs after the end event and synchronization, so stateful
    backends can consume/reset a timed operation without charging that cleanup to its latency.
    Returns the raw per-iteration series; the caller reduces across ranks per iteration before
    percentiling.

    There is deliberately NO host sync between `pre()` and the start event. Stream ordering
    already keeps pre()'s work out of the s->e window: `s` is enqueued behind pre()'s kernels,
    so the event timestamps when the stream REACHES it, not when the host recorded it. A sync
    here adds no guarantee and costs correctness: it drains the GPU, which puts the host's launch
    of `fn` inside the measured window and, because `fn` is a collective, lets per-rank launch
    jitter desynchronise ranks that pre() had just aligned. Each rank then blocks on the slowest
    peer and run_sweep's cross-rank MAX reports that stagger as latency. Measured on b200 uccl-ep
    low-latency combine: 113.4us with a sync against 87.4us without, at T=1, and no difference at
    T=32 -- which is what produces a *falling* latency curve as tokens grow.

    The end-of-iteration sync stays: the warmup note below documents why iterations must not
    overlap (iter N+1's dispatch races iter N's combine on the persistent comm buffer), so only
    one pre/fn pair is ever in flight.
    """
    def sample():
        arg = pre() if pre is not None else None
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


def time_cuda_graph_phase_us(
    torch, fn, warmup: int, iters: int, interval
) -> list[float]:
    """Time one event-record interval captured inside graph replay."""
    for _ in range(max(0, warmup)):
        fn()
        torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        fn()
        torch.cuda.synchronize()
        samples.append(interval[0].elapsed_time(interval[1]) * 1000.0)
    return samples



def _reduce_vec(torch, dist, device, vals, op):
    t = torch.tensor(vals, device=device, dtype=torch.float64)
    dist.all_reduce(t, op=op)
    return [float(x) for x in t.tolist()]


def _reduce_vec_median_spread(torch, dist, device, vals):
    """Per-element cross-rank (MEDIAN, MAX-MIN) for a chained series, from one all_gather.

    The pair period is a RATE, not a completion cost: every rank runs the same phase-locked
    free-running loop, so MAX would publish whichever rank hiccuped as the pipeline's speed. The
    median is the agreed cadence; a spread large next to it means one rank was paced -- distrust.

    One gather rather than three reductions, so the two cannot disagree about which iterations
    they describe (and MEDIAN is not a `ReduceOp`); every rank gathers the same matrix in rank
    order, so the artifact does not depend on which rank wrote it.
    """
    local = torch.tensor(vals, device=device, dtype=torch.float64)
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    stacked = torch.stack(gathered)
    median = stacked.median(dim=0).values
    spread = stacked.max(dim=0).values - stacked.min(dim=0).values
    return [float(x) for x in median.tolist()], [float(x) for x in spread.tolist()]


def _gather_scalar(torch, dist, device, val):
    """Every rank's copy of one per-rank scalar, in rank order, identical on every rank.

    The chain-health caller reduces it two ways (median and max-magnitude) from this one list.
    """
    local = torch.tensor([float(val)], device=device, dtype=torch.float64)
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    return [float(g.item()) for g in gathered]


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


@dataclass
class PointSamples:
    """Every sample series for one ladder point.

    Fresh-entry components carry one value per timed iteration, pooled across trials;
    `spread` is the per-iteration cross-rank max-minus-min of the roundtrip; the `_min`
    fields are the same iterations reduced by cross-rank MIN (the last rank into a
    collective waited least, so its duration is the operation with entry skew excluded).
    The chained family runs on its own much smaller trial count: `chain` and `chain_spread`
    are per-iteration, while `gap` and `settle` are one scalar per trial.
    """

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


class EPTiming:
    """Measure an EPBackend using its dispatch, stage, combine, and pairing contract."""

    # ---- Timing template methods -----------------------------------------------------

    def timed_components(self):
        """Components measured for this backend: roundtrip, dispatch and combine
        always; stage only when it launches device work."""
        components = ["roundtrip", "dispatch", "combine"]
        if self.stage_device_work and not self.cuda_graph_enabled:
            components.append("stage")
        return components

    def warm(self, problem, count, stage_every=False):
        """Untimed synchronized full round trips (fabric/clock warm-up; cold-jump-safe).

        Caches the dynamic receive cardinality once so adapters never read a device
        scalar during a timed trial (the count is stable for a fixed routing trace).

        `stage_every` re-materialises the combine input on every iteration; the default hoists it
        after the first, mirroring `benchmark_roundtrip`. Where staging is excluded from the chain
        the timed region stages nothing, so warming it warms work the measurement never performs
        -- ~247us per FP8 dequant against a 61us roundtrip, the leg's largest single cost.
        `benchmark_stage` opts in, because there staging is the timed operation.
        """
        import torch

        staged = None
        for _ in range(count):
            handle = self.dispatch(problem)
            if not hasattr(problem, "recv_tokens"):
                problem.recv_tokens = self.recv_tokens(handle)
            if staged is None:
                self.stage(problem, handle)
                if not stage_every and self.stage_excluded_from_roundtrip:
                    staged = handle.combine_input
            else:
                handle.combine_input = staged
            self.combine(problem, handle)
            torch.cuda.synchronize()

    def run_roundtrip(self, problem, staged=None):
        """One chained round trip; returns combined activations.

        `staged` supplies a pre-materialised combine input so staging stays out of the timed
        region -- the default wherever `stage()` does device work (see
        `stage_excluded_from_roundtrip`). It is None where `stage()` is a bare pointer
        assignment, or under the `CX_FP8_CONSUME=dequant` hatch that wants it back in the chain.
        """
        handle = self.dispatch(problem)
        if staged is None:
            self.stage(problem, handle)
        else:
            handle.combine_input = staged
        return self.combine(problem, handle)

    def benchmark_chain(self, problem, warmup, iters, drop):
        """Free-running dispatch->combine pairs, no host sync: a floors chain, then a period chain.

        This is what a serving stack pays: a decode loop never stops between layers, so entry
        skew amortises across the chain instead of landing on one op the way `roundtrip`'s
        drained windows charge it. The pairing is `run_roundtrip`'s, so paired-API backends stay
        in contract; every backend is measured.

        Two chains, because per-op events inside a chained pair execute immediately on an idle
        stream, landing the host's record() cost in the pair window: six events per pair
        published a flat +10-30us host constant on every vendor (+20-38% at T=1, decaying with
        T). So the floors chain carries op-window events only, the period chain one outer pair
        with nothing between its two collectives, and `chain_health.interpair_gap_us`
        (start-to-start median minus window median) guards that defect in-artifact.

        Only the pair period and the per-op minima are publishable: each rank's inter-rank wait
        parks in whichever op window it blocks in while the period is conserved, so `run_sweep`
        enforces pair -> cross-rank median, per-op -> cross-rank minimum, never a chained per-op
        median or p99.

        `drop` discards each chain's head (pipeline fill, not period). The chain's own final
        combined output is returned under `combined` -- cloned after the closing synchronize, so
        the copy is untimed and detached from any double-buffered receive the next dispatch would
        overwrite. `run_sweep` checks it against a drained pair through this same code path and
        separately reruns the full expert oracle against the state the chain leaves behind; both
        fold into the point's verdict. Interior pairs stay unvalidated by design -- each pair
        overwrites its predecessor's output, and holding or reducing every output would put
        device work inside the timed loops (see methodology, Correctness).
        Free-running is safe fleet-wide: every backend double-buffers per dispatch or completes
        each op on a reusable handle, and deepep-v2 NORMAL probed clean with 256 un-synced pairs
        (T=128, EP8+EP16, both precisions, 2026-08-06, pin 01dc3aaa). Returns post-`drop` series
        in microseconds: `pair` and `start_to_start` from the period chain (the latter one
        element shorter), `dispatch` and `combine` from the floors chain.
        """
        import torch

        self.warm(problem, warmup)
        staged = None
        if self.stage_excluded_from_roundtrip:
            # The same hoist `benchmark_roundtrip` performs, so the chain is dispatch -> combine
            # and nothing else. The `CX_FP8_CONSUME=dequant` hatch leaves `staged` None, putting
            # the conversion inside the pair period and inside neither per-op window -- where
            # work between the two collectives belongs.
            handle = self.dispatch(problem)
            self.stage(problem, handle)
            staged = handle.combine_input
            self.combine(problem, handle)  # drain the pair backends require
            torch.cuda.synchronize()
        # Events are allocated BEFORE the loops: an allocation between two record() calls is host
        # work inside a window meant to belong to the stream, a measurable fraction of the period
        # at the bottom of the ladder.
        def events():
            return [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

        dispatch_start, dispatch_end = events(), events()
        combine_start, combine_end = events(), events()
        pair_start, pair_end = events(), events()

        # ---- Floors chain: op windows only, pair boundaries uninstrumented. ----
        for i in range(iters):
            dispatch_start[i].record()
            handle = self.dispatch(problem)
            dispatch_end[i].record()
            if staged is None:
                self.stage(problem, handle)
            else:
                handle.combine_input = staged
            combine_start[i].record()
            self.combine(problem, handle)
            combine_end[i].record()
        torch.cuda.synchronize()

        # ---- Period chain: nothing between the pair's collectives but the pair itself. ----
        for i in range(iters):
            pair_start[i].record()
            handle = self.dispatch(problem)
            if staged is None:
                self.stage(problem, handle)
            else:
                handle.combine_input = staged
            combined = self.combine(problem, handle)
            pair_end[i].record()
        torch.cuda.synchronize()

        def series(starts, ends):
            return [
                start.elapsed_time(end) * 1000.0  # ms -> us
                for start, end in zip(starts[drop:], ends[drop:])
            ]

        return {
            "pair": series(pair_start, pair_end),
            "start_to_start": series(pair_start[:-1], pair_start[1:]),
            "dispatch": series(dispatch_start, dispatch_end),
            "combine": series(combine_start, combine_end),
            # The period chain's final combined output, produced IN the free-running regime.
            # Cloned post-sync (untimed, stream-ordered ahead of any later dispatch) so the
            # caller can compare it against a drained pair without racing the buffers.
            "combined": combined.clone(),
        }

    def benchmark_component(self, component, problem, warmup, iters):
        """Measure one named component; every component gets the same warm-up first."""
        if self.cuda_graph_enabled:
            # Re-capture the roundtrip for each component, adding timing nodes only
            # around that phase so roundtrip replay stays uninstrumented.
            return self.benchmark_roundtrip(problem, warmup, iters, component)
        if component == "roundtrip":
            return self.benchmark_roundtrip(problem, warmup, iters)
        if component == "dispatch":
            return self.benchmark_dispatch(problem, warmup, iters)
        if component == "stage":
            return self.benchmark_stage(problem, warmup, iters)
        if component == "combine":
            return self.benchmark_combine(problem, warmup, iters)
        raise RuntimeError(f"unknown timed component {component!r}")

    def benchmark_roundtrip(self, problem, warmup, iters, graph_component="roundtrip"):
        import torch

        self.warm(problem, warmup)
        staged = None
        if self.stage_excluded_from_roundtrip:
            # Materialise the expert-output stand-in ONCE, untimed, so the chained measurement is
            # dispatch -> combine and nothing else. Routing is fixed for a ladder point, so the
            # same staged tensor is valid for every iteration -- MoRI's is the dispatch output at
            # BF16 or a `[:rows]` BF16 cast under FP8, FlashInfer's the workspace combine region,
            # which sits past the end of every dispatch receive plane. Read back through
            # `handle.combine_input` rather than constructed, so an adapter's non-torch payload
            # (nccl-ep) would round-trip unchanged if one ever reached here.
            handle = self.dispatch(problem)
            self.stage(problem, handle)
            staged = handle.combine_input
            self.combine(problem, handle)  # drain the pair backends require
            torch.cuda.synchronize()
        if self.cuda_graph_enabled:
            # Capture replaces the existing roundtrip callable in place. Capture and its warmup
            # are excluded; the ordinary time_us event pipeline measures replay directly.
            import torch.distributed as dist

            dist.barrier()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            interval = (
                (
                    torch.cuda.Event(enable_timing=True, external=True),
                    torch.cuda.Event(enable_timing=True, external=True),
                )
                if graph_component != "roundtrip" else None
            )
            with torch.cuda.graph(graph, capture_error_mode="relaxed"):
                if graph_component == "dispatch":
                    interval[0].record()
                handle = self.dispatch(problem)
                if graph_component == "dispatch":
                    interval[1].record()
                if staged is None:
                    self.stage(problem, handle)
                else:
                    handle.combine_input = staged
                if graph_component == "combine":
                    interval[0].record()
                combined = self.combine(problem, handle)
                if graph_component == "combine":
                    interval[1].record()
            torch.cuda.synchronize()
            if interval is None:
                samples = time_us(torch, graph.replay, warmup, iters)
            else:
                samples = time_cuda_graph_phase_us(
                    torch, graph.replay, warmup, iters, interval
                )

            # Prove replay, rather than capture, writes the output used by the correctness gate.
            combined.fill_(float("nan"))
            torch.cuda.synchronize()
            graph.replay()
            torch.cuda.synchronize()
            replayed = combined.clone()
            problem._cuda_graph_output = replayed
            problem._cuda_graph_output_rewritten = bool(
                torch.isfinite(replayed).all().item()
            )
            return samples
        return time_us(torch, lambda p=problem: self.run_roundtrip(p, staged), 0, iters)

    def benchmark_dispatch(self, problem, warmup, iters):
        import torch

        self.warm(problem, warmup)

        def finish_dispatch(hh, p=problem):
            self.stage(p, hh)
            self.combine(p, hh)

        return time_us(
            torch, lambda p=problem: self.dispatch(p), 0, iters,
            post=finish_dispatch if self.requires_fresh_pair else None,
        )

    def benchmark_stage(self, problem, warmup, iters):
        import torch

        # Staging is the timed operation here, so it must be warmed on every iteration.
        self.warm(problem, warmup, stage_every=True)

        def prep_stage(p=problem):
            return self.dispatch(p)

        def stage_op(hh, p=problem):
            self.stage(p, hh)
            return hh

        # Drain each timed stage's dispatch with an untimed combine where the
        # backend requires the pair (same rule as benchmark_dispatch).
        return time_us(
            torch, stage_op, 0, iters, pre=prep_stage,
            post=(lambda hh, p=problem: self.combine(p, hh))
            if self.requires_fresh_pair else None,
        )

    def benchmark_combine(self, problem, warmup, iters):
        import torch

        self.warm(problem, warmup)

        def prep_combine(p=problem):
            hh = self.dispatch(p)
            self.stage(p, hh)
            return hh

        if self.requires_fresh_pair:
            return time_us(
                torch, lambda hh, p=problem: self.combine(p, hh), 0, iters, pre=prep_combine,
            )
        hh = prep_combine()
        torch.cuda.synchronize()
        return time_us(torch, lambda p=problem, hx=hh: self.combine(p, hx), 0, iters)
