"""Timing strategies: eager CUDA-event windows and CUDA graph replay, one object per regime."""
from __future__ import annotations

import functools
import gc
import itertools
import os
import statistics
import time


# Iterations of the CPU probe: a loop that allocates nothing, so no garbage collection can land in
# it and only the host's speed moves its wall time (~20 us on a b200-nscale core at full clock).
PROBE_LOOPS = 2048


def cpu_probe_us() -> float:
    """Wall time (us) of a fixed slice of interpreter work: how fast this host is running now."""
    start = time.perf_counter_ns()
    for _ in itertools.repeat(None, PROBE_LOOPS):
        pass
    return (time.perf_counter_ns() - start) / 1e3


class HostGate:
    """Keeps a throttled host out of the windows of a dispatch that waits on the host.

    Such a dispatch relaunches its transfer from inside the window once the receive counts arrive,
    so the host's speed is part of the sample. While uccl-ep EP16 runs, b200-nscale nodes
    intermittently run every core at ~200 MHz effective (pinned Python runs ~16x slow) for 100-200
    ms, every few seconds, and take another 10-20 ms to wind back up through 3-7x, while the GPUs
    keep their clocks: the relaunch then takes 1-2 ms and every rank waits on it. The base and eager
    arms see it at the same rate. Before a sample opens, `hold()` times a CPU probe
    and, while it runs more than `HOLD_FACTOR` times slower than calibrated, keeps probing instead
    of opening the sample; the other ranks wait at the alignment barrier, so the window opens once
    the host runs at speed again. After the sample's final sync `check()` probes again. A sample is
    **held** when its start waited longer than `HELD_S`, and **stalled** when a probe beside its
    window ran more than `STALL_FACTOR` times slower: a throttle that began inside the window, or
    one that outlasted the hold limits. Both probes run outside the window.
    """

    # Away from throttles a ~50 us slice of Python on a rank's own core runs over 2.5x its median
    # in at most ~0.2% of samples and over 5x in ~0.001%: holding at 2.5x costs an occasional extra
    # probe, while only a real throttle reaches the stall bar.
    HOLD_FACTOR = 2.5
    STALL_FACTOR = 5.0
    CALIBRATION = 16
    # A shorter hold was an interrupt on one probe, not a throttle, and is not counted as held.
    HELD_S = 1e-3
    # The longest one sample waits for its host (the b200 throttle lasts 100-200 ms), and the
    # run-wide total after which samples stop waiting and are only counted, so a host throttled
    # for good cannot stall the run.
    HOLD_S = 0.5
    BUDGET_S = 60.0

    def __init__(self):
        self.probe_us = None
        self.held = []
        self.stalled = []
        self.spent_s = 0.0
        self._opened_throttled = False

    def calibrate(self):
        """Start a series. The probe's unthrottled wall time is the run's lowest per-series median,
        so a calibration that itself lands in a throttle cannot raise the bar for the rest of the
        run."""
        median = statistics.median(cpu_probe_us() for _ in range(self.CALIBRATION))
        self.probe_us = median if self.probe_us is None else min(self.probe_us, median)
        self.held, self.stalled = [], []

    def hold(self):
        """Before a sample opens: wait while this host runs slow, within the limits."""
        bar = self.HOLD_FACTOR * self.probe_us
        probe = cpu_probe_us()
        waited = 0.0
        if probe > bar and self.spent_s < self.BUDGET_S:
            start = time.perf_counter()
            limit = min(self.HOLD_S, self.BUDGET_S - self.spent_s)
            while probe > bar and waited <= limit:
                probe = cpu_probe_us()
                waited = time.perf_counter() - start
            self.spent_s += waited
        self.held.append(int(waited > self.HELD_S))
        self._opened_throttled = probe > self.STALL_FACTOR * self.probe_us

    def check(self):
        """After a sample's final sync: count it stalled if its host runs throttled now, or if the
        hold ran out while it still was."""
        throttled = cpu_probe_us() > self.STALL_FACTOR * self.probe_us
        self.stalled.append(int(throttled or self._opened_throttled))


def time_us(torch, fn, iters: int, pre=None, post=None, align=None, leads=None,
            gate=None, lead_through_fn=False, align_after_pre=False) -> list[float]:
    """Per-iteration CUDA-event latencies (us) for THIS rank.

    `align()` opens each sample by enqueueing the rank alignment ahead of everything else;
    `leads`, when given, receives each sample's host time (us) from the start of `align()` to the
    start event's record, or to `fn()`'s return with `lead_through_fn`, which the alignment's spin
    must outlast: a host that is still enqueuing the timed op when the barrier releases leaves its
    GPU idle inside the window. `gate`, when given, holds each sample until this rank's host runs
    at speed and checks it again after the final sync (`HostGate`), both outside the window and
    outside the lead. `pre()` runs untimed before each sample and its result is passed to `fn`;
    `post(result)` runs after the end event and sync. There is deliberately NO sync between
    `pre()` and the start event: stream order already keeps pre()'s work out of the window, and a
    sync would put the host's launch of the collective inside it and let launch jitter stagger the
    ranks (b200 uccl-ep LL combine: 113.4us with a sync vs 87.4us without at T=1). A `pre()` that
    itself waits on the host (a host-synced dispatch) runs before the alignment instead
    (`align_after_pre`), or its wait would spend the spin. Each iteration ends synchronized:
    iteration N+1's dispatch must not race iteration N's combine on the persistent buffer.
    """
    def sample():
        if gate is not None:
            gate.hold()
        arg = pre() if pre is not None and align_after_pre else None
        opened = time.perf_counter()
        if align is not None:
            align()
        if pre is not None and not align_after_pre:
            arg = pre()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        if leads is not None and not lead_through_fn:
            leads.append((time.perf_counter() - opened) * 1e6)
        result = fn(arg) if pre is not None else fn()
        if leads is not None and lead_through_fn:
            leads.append((time.perf_counter() - opened) * 1e6)
        e.record()
        torch.cuda.synchronize()
        elapsed = s.elapsed_time(e) * 1000.0  # ms -> us
        if gate is not None:
            gate.check()
        if post is not None:
            post(result)
            torch.cuda.synchronize()
        return elapsed

    return [sample() for _ in range(iters)]


def time_cuda_graph_phase_us(torch, fn, warmup: int, iters: int, interval, leads=None) -> list[float]:
    """Time one event-record interval captured inside graph replay.

    The graph opens with a spin and an all-reduce ahead of the timing events. `leads`, when given,
    receives each replay call's host time (us): HIP submits a replay node by node from the host,
    so a call that outlasts the spin may still be enqueuing the timed work when the barrier
    releases the other ranks.
    """
    for _ in range(max(0, warmup)):
        fn()
        torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        opened = time.perf_counter()
        fn()
        if leads is not None:
            leads.append((time.perf_counter() - opened) * 1e6)
        torch.cuda.synchronize()
        samples.append(interval[0].elapsed_time(interval[1]) * 1000.0)
    return samples


def _series(starts, ends, drop):
    return [start.elapsed_time(end) * 1000.0 for start, end in zip(starts[drop:], ends[drop:])]


def _chain_result(pair, floors, drop, combined):
    pair_start, pair_end = pair
    return {
        "pair": _series(pair_start, pair_end, drop),
        "start_to_start": _series(pair_start[:-1], pair_start[1:], drop),
        "dispatch": _series(*floors["dispatch"], drop),
        "combine": _series(*floors["combine"], drop),
        # The period chain's final output, cloned post-sync so the caller can compare it against a
        # drained pair without racing the buffers.
        "combined": combined.clone(),
    }


class _SpinAlignment:
    """The alignment both regimes share: a GPU spin, then a device all-reduce, ahead of every
    fresh-entry sample, the order DeepEP's bench_kineto uses. The spin outlasts the host's enqueue
    of the sample, so the last rank to arrive already has its timed work queued when the
    all-reduce releases everyone; the window then opens on every rank at the barrier's exit
    instead of at each host's launch, which the cross-rank MAX would otherwise report as latency.
    A sample whose host took longer than the spin is counted as missed (`missed_samples`)."""

    # Wall time the spin must outlast: the host's enqueue of one sample (alignment, untimed setup,
    # events and the timed operation), stalls included. 500 us covered the ~50-70 us enqueue but
    # not its occasional stalls, which doubled h200 uccl-ep decode p99; DeepEP spins ~10 ms.
    SPIN_US = 2000.0

    def __init__(self, backend):
        self.backend = backend
        self.spin_us = float(os.environ.get("COLLX_ALIGN_SPIN_US") or self.SPIN_US)
        self.spin_cycles = None
        self.leads = []
        self._align_token = None

    def missed_samples(self):
        """1 per sample of the last timed series whose host outran the spin: that rank's window may
        have opened after the barrier released the others."""
        return [int(lead > self.spin_us) for lead in self.leads]

    def calibrate_spin(self):
        """Size the spin to `spin_us` of wall time: `_sleep` counts SM cycles, not time."""
        import torch

        probe = 200_000
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda._sleep(probe // 10)  # ramp clocks before the measured spin
        start.record()
        torch.cuda._sleep(probe)
        end.record()
        torch.cuda.synchronize()
        elapsed_us = max(start.elapsed_time(end) * 1000.0, 1e-3)
        self.spin_cycles = max(1, int(probe * self.spin_us / elapsed_us))

    def _all_reduce(self):
        import torch
        import torch.distributed as dist

        if self._align_token is None:
            self._align_token = torch.zeros(1, device=self.backend.device)
        dist.all_reduce(self._align_token)


class EagerTiming(_SpinAlignment):
    """CUDA events recorded on the stream around eagerly launched operations.

    Each timed sample opens with the spin and the all-reduce (`_SpinAlignment`). A series whose
    timed dispatch waits on the host for its receive counts keeps the host's relaunch inside the
    window, so its samples also pass through a `HostGate`: each opens only once this rank's host
    runs at speed (`host_holds`), and one the host ran throttled around anyway is counted as
    host-stalled (`host_stalls`). Its chains pass through the same gate (`chain`).
    """

    graph = False
    # Chain siblings a throttled host may restart, per trial, before the trial publishes as stalled.
    CHAIN_RETRIES = 3

    def __init__(self, backend):
        super().__init__(backend)
        # One gate per run, so its calibration and its hold budget span every series.
        self.gate = HostGate()
        self._gated = False

    @property
    def alignment(self):
        return {"method": "gpu-spin-then-all-reduce", "spin_us": self.spin_us}

    def host_holds(self):
        """1 per sample of the last timed series whose start waited for this rank's throttled host
        to recover. None for a series whose dispatch does not wait on the host."""
        return list(self.gate.held) if self._gated else None

    def host_stalls(self):
        """1 per sample of the last timed series that this rank's host ran throttled around anyway,
        so the relaunch inside its window may have run slow. None for a series whose dispatch does
        not wait on the host, which has no host work in its window."""
        return list(self.gate.stalled) if self._gated else None

    def align(self):
        """Enqueue the spin, then a device all-reduce, on the current stream; no host sync."""
        import torch

        if self.spin_cycles is None:
            self.calibrate_spin()
        torch.cuda._sleep(self.spin_cycles)
        self._all_reduce()

    def components(self):
        """Roundtrip, dispatch and combine always; stage only when it launches device work."""
        names = ["roundtrip", "dispatch", "combine"]
        return names + ["stage"] if self.backend.stage_device_work else names

    def _time(self, fn, iters, host_synced=False, **kwargs):
        """Time one series, every sample aligned. The spin is calibrated here, after the caller's
        warm-up, because clocks move with load and temperature. `host_synced` marks a series whose
        timed call waits on the host for its own device work; only those pass through the gate, and
        only their lead stops at the start event, since their call returns after the device work.
        Garbage collection is off for the series, so a collection cannot stall a host inside a
        window."""
        import torch

        self.calibrate_spin()
        self.leads = []
        self._gated = host_synced
        if host_synced:
            self.gate.calibrate()
        collecting = gc.isenabled()
        gc.disable()
        try:
            return time_us(torch, fn, iters, align=self.align, leads=self.leads,
                           gate=self.gate if host_synced else None,
                           lead_through_fn=not host_synced, **kwargs)
        finally:
            if collecting:
                gc.enable()

    def component(self, name, problem, warmup, iters):
        b = self.backend
        if name == "roundtrip":
            staged = b.warm_and_hoist_stage(problem, warmup)
            return self._time(lambda: b.run_roundtrip(problem, staged), iters,
                              host_synced=b.host_synced_dispatch)
        if name == "dispatch":
            b.warm(problem, warmup)

            def finish(handle):
                b.stage(problem, handle)
                b.combine(problem, handle)

            return self._time(lambda: b.dispatch(problem), iters,
                              host_synced=b.host_synced_dispatch,
                              post=finish if b.requires_fresh_pair else None)
        if name == "stage":
            # Staging is the timed operation here, so it must be warmed on every iteration.
            b.warm(problem, warmup, stage_every=True)

            def stage(handle):
                b.stage(problem, handle)
                return handle

            return self._time(stage, iters, pre=lambda: b.dispatch(problem),
                              align_after_pre=b.host_synced_dispatch,
                              post=(lambda h: b.combine(problem, h)) if b.requires_fresh_pair else None)
        if name == "combine":
            # A fresh pair with the hoisted stage, as graph replay and the roundtrip time it: the
            # window opens behind its own dispatch and inherits that dispatch's exit skew on every
            # backend. A dispatch that waits on the host runs before the alignment, or its wait
            # would spend the spin.
            staged = b.warm_and_hoist_stage(problem, warmup)

            def staged_dispatch():
                handle = b.dispatch(problem)
                if staged is None:
                    b.stage(problem, handle)
                else:
                    handle.combine_input = staged
                return handle

            return self._time(lambda h: b.combine(problem, h), iters, pre=staged_dispatch,
                              align_after_pre=b.host_synced_dispatch)
        raise RuntimeError(f"unknown timed component {name!r}")

    def chain(self, problem, staged, iters, drop):
        """A floors chain (op windows only), then a period chain (one outer window per pair).

        Two siblings because per-op events inside a pair land the host's record() cost in the pair
        window: six events per pair published a +10-30us host constant on every vendor.
        Events are allocated before the loops so no allocation lands inside a window.

        Each sibling opens with the all-reduce alone. A spin in front would let the host queue a
        backlog that the GPU then drains faster than the host can feed it, so a host-bound chain's
        kept pairs would mix two speeds (mi300x MoRI eager: settle drift +25us of a 78us period).
        A backend whose dispatch waits on the host also passes each sibling through the host gate:
        it opens once this host runs at speed and restarts, on every rank together, if any rank's
        host throttled around it, up to `CHAIN_RETRIES` times.
        """
        import torch

        b = self.backend

        def events():
            return [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

        floors = {"dispatch": (events(), events()), "combine": (events(), events())}
        pair = (events(), events())
        out = {}

        def floors_chain():
            for i in range(iters):
                floors["dispatch"][0][i].record()
                handle = b.dispatch(problem)
                floors["dispatch"][1][i].record()
                if staged is None:
                    b.stage(problem, handle)
                else:
                    handle.combine_input = staged
                floors["combine"][0][i].record()
                b.combine(problem, handle)
                floors["combine"][1][i].record()

        def period_chain():
            for i in range(iters):
                pair[0][i].record()
                handle = b.dispatch(problem)
                if staged is None:
                    b.stage(problem, handle)
                else:
                    handle.combine_input = staged
                out["combined"] = b.combine(problem, handle)
                pair[1][i].record()

        gated = b.host_synced_dispatch
        if gated:
            self.gate.calibrate()
        retries = stalled = 0
        # Off for the chains too: a host-bound chain's pair window carries the host's pace.
        collecting = gc.isenabled()
        gc.disable()
        try:
            for sibling in (floors_chain, period_chain):
                for attempt in range(self.CHAIN_RETRIES + 1):
                    if gated:
                        self.gate.hold()
                    self._all_reduce()
                    sibling()
                    torch.cuda.synchronize()
                    if not gated or not self._host_stalled_anywhere():
                        break
                    if attempt == self.CHAIN_RETRIES:
                        stalled += 1
                    else:
                        retries += 1
        finally:
            if collecting:
                gc.enable()
        result = _chain_result(pair, floors, drop, out["combined"])
        result["host_retries"] = retries if gated else None
        result["host_stalled"] = bool(stalled) if gated else None
        return result

    def _host_stalled_anywhere(self):
        """Whether any rank's host ran throttled around the sibling it just finished; every rank
        gets the same answer, so they restart (or not) together."""
        import torch
        import torch.distributed as dist

        self.gate.check()
        flag = torch.tensor([self.gate.stalled[-1]], device=self.backend.device, dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())


def graph_event():
    """A timing event whose record() can be captured into a graph. ROCm torch < 2.13 rejects
    `external=True`, so there the event is recorded once to exist and its captured record goes
    through hipEventRecordWithFlags, as torch 2.13 does (pytorch#178264)."""
    import torch

    if not getattr(torch.version, "hip", None):
        return torch.cuda.Event(enable_timing=True, external=True)
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    event._collx_hip_external = True
    return event


def record_graph_event(event):
    import ctypes

    import torch

    if not getattr(event, "_collx_hip_external", False):
        return event.record()
    rc = _hip_runtime().hipEventRecordWithFlags(
        ctypes.c_void_p(event.cuda_event),
        ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
        ctypes.c_uint(0x1),  # hipEventRecordExternal
    )
    if rc != 0:
        raise RuntimeError(f"hipEventRecordWithFlags(external) failed with hipError {rc}")


@functools.cache
def _hip_runtime():
    import ctypes

    import torch

    try:
        return ctypes.CDLL("libamdhip64.so")
    except OSError:  # pip ROCm torch bundles the runtime in torch/lib
        return ctypes.CDLL(os.path.join(os.path.dirname(torch.__file__), "lib", "libamdhip64.so"))


def poison(tensor):
    """Overwrite a tensor with 0xFF bytes: NaN for bf16/fp16/fp32/fp8-e4m3, -1 for ints."""
    import torch

    if tensor is None:
        return
    if isinstance(tensor, (tuple, list)):
        for part in tensor:
            poison(part)
        return
    if not isinstance(tensor, torch.Tensor) or not tensor.numel():
        return
    try:
        tensor.view(torch.uint8).fill_(0xFF)
    except RuntimeError:
        tensor.fill_(float("nan") if tensor.is_floating_point() else -1)


class GraphTiming(_SpinAlignment):
    """Measure dispatch/combine latency with CUDA events inside a replayed CUDA graph.

    Each replay opens with the same spin and all-reduce as an eager sample, captured into the graph
    ahead of the timing events, so their cost stays out of the window. CUDA launches a replay
    whole, but HIP submits it node by node from the host, so on ROCm a host that lags after the
    all-reduce's nodes would otherwise start its dispatch late while its peers' windows are already
    open (mi300x MoRI decode p99/p50 1.30-1.55 graphed against 1.02-1.14 eager on the same node).
    The spin gives the host time to submit the whole replay first, and a replay call that outlasts
    it counts as missed, exactly as an eager sample does.
    """

    graph = True
    # Handle attributes dispatch and the captured stage write; the replay value check poisons them,
    # so a replay that skipped the stage cannot pass on the payload the upload replay staged.
    DISPATCH_OUTPUT_FIELDS = ("recv_x", "recv_scales", "dispatch_output", "combine_input")
    # The replay value check rescales the inputs by this exact power of two, so a replay that
    # re-sent stale data cannot match the drained pair run on the rescaled inputs.
    REPLAY_INPUT_SCALE = 2.0

    @property
    def alignment(self):
        return {"method": "in-graph-spin-then-all-reduce", "spin_us": self.spin_us}

    def host_holds(self):
        """No graphed dispatch waits on the host."""
        return None

    def host_stalls(self):
        """No graphed dispatch waits on the host."""
        return None

    def components(self):
        return ["roundtrip", "dispatch", "combine"]

    def capture_pairs(self, problem, staged, pairs, marks):
        """Capture `pairs` back-to-back pairs into one graph; `marks` picks the windows that get
        event nodes ("pair", "dispatch", "combine"). Returns (graph, {mark: (starts, ends)},
        last combined output, last dispatch handle)."""
        import torch
        import torch.distributed as dist

        b = self.backend
        if self._align_token is None:
            # Created and reduced once outside any capture, so the communicator initialises eagerly.
            self._all_reduce()
        if self.spin_cycles is None:
            self.calibrate_spin()
        stamps = {mark: ([graph_event() for _ in range(pairs)], [graph_event() for _ in range(pairs)])
                  for mark in marks}

        def record(mark, edge, i):
            if mark in stamps:
                record_graph_event(stamps[mark][edge][i])

        dist.barrier()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        combined = handle = None
        # torch's default capture mode, as the engines capture: an API call that is unsafe during
        # capture fails here instead of being recorded into a graph no engine could build.
        with torch.cuda.graph(graph):
            # A separate all-reduce followed by a host-launched graph can let CPU launch
            # delays spread the ranks apart again, inflating the measured communication time.
            # Capture the spin, then the all-reduce and its stream dependencies, so each replay
            # reaches the timed work without another host launch after alignment. Place them
            # before the first timing event to exclude alignment cost, and only once per graph
            # to preserve back-to-back dispatch/combine pairs.
            torch.cuda._sleep(self.spin_cycles)
            dist.all_reduce(self._align_token)
            for i in range(pairs):
                record("pair", 0, i)
                record("dispatch", 0, i)
                handle = b.dispatch(problem)
                record("dispatch", 1, i)
                b.stage_or_reuse(problem, handle, staged)
                record("combine", 0, i)
                combined = b.combine(problem, handle)
                record("combine", 1, i)
                record("pair", 1, i)
        torch.cuda.synchronize()
        return graph, stamps, combined, handle

    def component(self, name, problem, warmup, iters):
        """One captured pair with event nodes around `name`; the poisoned output must be rewritten
        by the replay, proving replay rather than capture wrote what the correctness gate reads."""
        import torch

        if name not in ("roundtrip", "dispatch", "combine"):
            raise RuntimeError(f"unknown timed component {name!r}")
        staged = self.backend.warm_and_hoist_stage(problem, warmup)
        # Calibrated after the warm-up and before capture: the spin's cycle count is baked in.
        self.calibrate_spin()
        mark = "pair" if name == "roundtrip" else name
        graph, stamps, combined, _ = self.capture_pairs(problem, staged, 1, (mark,))
        starts, ends = stamps[mark]
        self.leads = []
        collecting = gc.isenabled()
        gc.disable()
        try:
            samples = time_cuda_graph_phase_us(
                torch, graph.replay, warmup, iters, (starts[0], ends[0]), leads=self.leads
            )
        finally:
            if collecting:
                gc.enable()
        combined.fill_(float("nan"))
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        replayed = combined.clone()
        problem._cuda_graph_output = replayed
        # Every timed capture must rewrite its poisoned output, not only the last one.
        problem._cuda_graph_output_rewritten = (
            getattr(problem, "_cuda_graph_output_rewritten", True)
            and bool(torch.isfinite(replayed).all().item())
        )
        return samples

    def chain(self, problem, staged, iters, drop):
        """Each sibling is ONE graph of `iters` unrolled pairs, the shape of a decode graph, replayed
        once untimed (upload) and once timed, with alignment captured before the first event."""
        import torch

        self.calibrate_spin()
        floors, floor_stamps, _, _ = self.capture_pairs(problem, staged, iters, ("dispatch", "combine"))
        period, period_stamps, combined, _ = self.capture_pairs(problem, staged, iters, ("pair",))
        for graph in (floors, period):
            graph.replay()
            torch.cuda.synchronize()
            graph.replay()
            torch.cuda.synchronize()
        result = _chain_result(period_stamps["pair"], floor_stamps, drop, combined)
        result["host_retries"] = result["host_stalled"] = None
        return result

    def replay_output(self, problem):
        """An untimed capture with `stage` INSIDE the graph. After the upload replay the inputs are
        rescaled in place and dispatch's output, the staged combine input and the result are
        poisoned, so the checked replay's output matches the drained pair the caller runs next on
        the same rescaled inputs only if that replay re-read its inputs and re-ran dispatch, stage
        and combine. The caller restores the inputs afterwards (`restore_replay_inputs`)."""
        import torch
        import torch.distributed as dist

        self.backend.warm(problem, 1)
        graph, _, combined, handle = self.capture_pairs(problem, None, 1, ())
        graph.replay()  # first launch uploads the graph; its output is discarded
        torch.cuda.synchronize()
        problem._replay_rescaled = rescale_inputs(problem, self.REPLAY_INPUT_SCALE)
        for name in self.DISPATCH_OUTPUT_FIELDS:
            poison(getattr(handle, name, None))
        poison(combined)
        torch.cuda.synchronize()
        # Peers write straight into each other's receive buffers: without the barrier a fast rank's
        # replay lands before a slow peer's poison, which then overwrites fresh data.
        dist.barrier()
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        return combined.clone()

    def restore_replay_inputs(self, problem):
        """Undo `replay_output`'s rescale: an exact power of two, so the inputs come back bitwise."""
        for tensor in getattr(problem, "_replay_rescaled", ()):
            tensor.mul_(1.0 / self.REPLAY_INPUT_SCALE)
        problem._replay_rescaled = ()


def rescale_inputs(problem, factor):
    """Multiply the problem's floating-point activations in place (each distinct tensor once) and
    return the tensors touched. FP8 tensors are skipped: an adapter that pre-quantizes outside the
    timed path then reads unchanged bytes in both the replay and the drained pair."""
    import torch

    floating = {getattr(torch, name, None) for name in ("bfloat16", "float16", "float32")} - {None}
    touched, seen = [], set()
    for name in ("x", "dispatch_x"):
        tensor = getattr(problem, name, None)
        if getattr(tensor, "dtype", None) not in floating or tensor.data_ptr() in seen:
            continue
        seen.add(tensor.data_ptr())
        tensor.mul_(factor)
        touched.append(tensor)
    return touched
