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
