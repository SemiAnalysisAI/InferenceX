"""Timing strategies: eager CUDA-event windows and CUDA graph replay, one object per regime."""
from __future__ import annotations

import functools
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
            gate=None) -> list[float]:
    """Per-iteration CUDA-event latencies (us) for THIS rank.

    `align()` opens each sample by enqueueing the rank alignment ahead of everything else;
    `leads`, when given, receives each sample's host time (us) from the start of `align()` to the
    start event's record, which the alignment's spin must outlast. `gate`, when given, holds each
    sample until this rank's host runs at speed and checks it again after the final sync
    (`HostGate`), both outside the window and outside the lead. `pre()` runs untimed before each
    sample and its result is passed to `fn`; `post(result)` runs after the end event and sync.
    There is deliberately NO sync between `pre()` and the start event: stream order already keeps
    pre()'s work out of the window, and a sync would put the host's launch of the collective inside
    it and let launch jitter stagger the ranks (b200 uccl-ep LL combine: 113.4us with a sync vs
    87.4us without at T=1). Each iteration ends synchronized: iteration N+1's dispatch must not race
    iteration N's combine on the persistent buffer.
    """
    def sample():
        if gate is not None:
            gate.hold()
        opened = time.perf_counter()
        if align is not None:
            align()
        arg = pre() if pre is not None else None
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        if leads is not None:
            leads.append((time.perf_counter() - opened) * 1e6)
        result = fn(arg) if pre is not None else fn()
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


def time_cuda_graph_phase_us(torch, fn, warmup: int, iters: int, interval) -> list[float]:
    """Time one event-record interval captured inside graph replay.

    The graph contains an all-reduce before the timing events to reduce rank entry skew.
    """
    for _ in range(max(0, warmup)):
        fn()
        torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        fn()
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


class EagerTiming:
    """CUDA events recorded on the stream around eagerly launched operations.

    Each timed sample, and each chain sibling, opens with a GPU spin and then a device all-reduce,
    the order DeepEP's bench_kineto uses. The spin outlasts the host's launch of the whole sample,
    so the last rank to arrive already has its timed work queued when the all-reduce releases
    everyone; the window then starts on every rank at the barrier's exit instead of at each host's
    launch, which the cross-rank MAX would otherwise report as latency. A sample whose host took
    longer than the spin to reach its start event is counted as missed (`missed_samples`).

    A series whose timed dispatch waits on the host for its receive counts keeps the host's
    relaunch inside the window, so its samples also pass through a `HostGate`: each opens only once
    this rank's host runs at speed (`host_holds`), and one the host ran throttled around anyway is
    counted as host-stalled (`host_stalls`).
    """

    graph = False
    # Wall time the spin must outlast: the host's enqueue of one sample (alignment, untimed setup,
    # events and the timed operation), stalls included. 500 us covered the ~50-70 us enqueue but
    # not its occasional stalls, which doubled h200 uccl-ep decode p99; DeepEP spins ~10 ms.
    SPIN_US = 2000.0

    def __init__(self, backend):
        self.backend = backend
        self.spin_us = float(os.environ.get("COLLX_ALIGN_SPIN_US") or self.SPIN_US)
        self.spin_cycles = None
        self.leads = []
        # One gate per run, so its calibration and its hold budget span every series.
        self.gate = HostGate()
        self._gated = False
        self._align_token = None

    @property
    def alignment(self):
        return {"method": "gpu-spin-then-all-reduce", "spin_us": self.spin_us}

    def missed_samples(self):
        """1 per sample of the last timed series whose host outran the spin before recording the
        start event: that rank's window may have opened after the barrier released the others."""
        return [int(lead > self.spin_us) for lead in self.leads]

    def host_holds(self):
        """1 per sample of the last timed series whose start waited for this rank's throttled host
        to recover. None for a series whose dispatch does not wait on the host."""
        return list(self.gate.held) if self._gated else None

    def host_stalls(self):
        """1 per sample of the last timed series that this rank's host ran throttled around anyway,
        so the relaunch inside its window may have run slow. None for a series whose dispatch does
        not wait on the host, which has no host work in its window."""
        return list(self.gate.stalled) if self._gated else None

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

    def align(self):
        """Enqueue the spin, then a device all-reduce, on the current stream; no host sync."""
        import torch
        import torch.distributed as dist

        if self._align_token is None:
            self._align_token = torch.zeros(1, device=self.backend.device)
        if self.spin_cycles is None:
            self.calibrate_spin()
        torch.cuda._sleep(self.spin_cycles)
        dist.all_reduce(self._align_token)

    def components(self):
        """Roundtrip, dispatch and combine always; stage only when it launches device work."""
        names = ["roundtrip", "dispatch", "combine"]
        return names + ["stage"] if self.backend.stage_device_work else names

    def _time(self, fn, iters, host_synced=False, **kwargs):
        """Time one series, every sample aligned. The spin is calibrated here, after the caller's
        warm-up, because clocks move with load and temperature. `host_synced` marks a series whose
        timed call waits on the host for its own device work; only those pass through the gate."""
        import torch

        self.calibrate_spin()
        self.leads = []
        self._gated = host_synced
        if host_synced:
            self.gate.calibrate()
        return time_us(torch, fn, iters, align=self.align, leads=self.leads,
                       gate=self.gate if host_synced else None, **kwargs)

    def component(self, name, problem, warmup, iters):
        import torch

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
                              post=(lambda h: b.combine(problem, h)) if b.requires_fresh_pair else None)
        if name == "combine":
            b.warm(problem, warmup)

            def staged_dispatch():
                handle = b.dispatch(problem)
                b.stage(problem, handle)
                return handle

            if b.requires_fresh_pair:
                return self._time(lambda h: b.combine(problem, h), iters, pre=staged_dispatch)
            handle = staged_dispatch()
            torch.cuda.synchronize()
            return self._time(lambda: b.combine(problem, handle), iters)
        raise RuntimeError(f"unknown timed component {name!r}")

    def chain(self, problem, staged, iters, drop):
        """A floors chain (op windows only), then a period chain (one outer window per pair).

        Two siblings because per-op events inside a pair land the host's record() cost in the pair
        window: six events per pair published a +10-30us host constant on every vendor.
        Events are allocated before the loops so no allocation lands inside a window.
        """
        import torch

        b = self.backend

        def events():
            return [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

        floors = {"dispatch": (events(), events()), "combine": (events(), events())}
        pair = (events(), events())
        self.calibrate_spin()
        self.align()
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
        torch.cuda.synchronize()
        self.align()
        for i in range(iters):
            pair[0][i].record()
            handle = b.dispatch(problem)
            if staged is None:
                b.stage(problem, handle)
            else:
                handle.combine_input = staged
            combined = b.combine(problem, handle)
            pair[1][i].record()
        torch.cuda.synchronize()
        return _chain_result(pair, floors, drop, combined)


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


class GraphTiming:
    """Measure dispatch/combine latency with CUDA events inside a replayed CUDA graph.

    Each replay first runs an all-reduce to bring ranks closer together before timed work
    begins. The all-reduce is part of the graph but precedes the timing events, so its cost
    is excluded from the reported latency.
    """

    graph = True
    # Handle attributes dispatch writes; the replay value check poisons them.
    DISPATCH_OUTPUT_FIELDS = ("recv_x", "recv_scales", "dispatch_output")

    def __init__(self, backend):
        self.backend = backend
        self._align_token = None

    @property
    def alignment(self):
        return {"method": "in-graph-all-reduce", "spin_us": None}

    def missed_samples(self):
        """Replays are launched whole, so there is no host step between barrier and window."""
        return None

    def host_holds(self):
        """No host work runs inside a replayed window."""
        return None

    def host_stalls(self):
        """No host work runs inside a replayed window."""
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
            self._align_token = torch.zeros(1, device=b.device)
            dist.all_reduce(self._align_token)
        stamps = {mark: ([graph_event() for _ in range(pairs)], [graph_event() for _ in range(pairs)])
                  for mark in marks}

        def record(mark, edge, i):
            if mark in stamps:
                record_graph_event(stamps[mark][edge][i])

        dist.barrier()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        combined = handle = None
        with torch.cuda.graph(graph, capture_error_mode="relaxed"):
            # A separate all-reduce followed by a host-launched graph can let CPU launch
            # delays spread the ranks apart again, inflating the measured communication time.
            # Capture the all-reduce and its stream dependencies in this graph so each replay
            # reaches the timed work without another host launch after alignment. Place it
            # before the first timing event to exclude alignment cost, and only once per graph
            # to preserve back-to-back dispatch/combine pairs. This reduces entry skew; it
            # does not guarantee that every rank starts the timed work simultaneously.
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
        mark = "pair" if name == "roundtrip" else name
        graph, stamps, combined, _ = self.capture_pairs(problem, staged, 1, (mark,))
        starts, ends = stamps[mark]
        samples = time_cuda_graph_phase_us(
            torch, graph.replay, warmup, iters, (starts[0], ends[0])
        )
        combined.fill_(float("nan"))
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        replayed = combined.clone()
        problem._cuda_graph_output = replayed
        problem._cuda_graph_output_rewritten = bool(torch.isfinite(replayed).all().item())
        return samples

    def chain(self, problem, staged, iters, drop):
        """Each sibling is ONE graph of `iters` unrolled pairs, the shape of a decode graph, replayed
        once untimed (upload) and once timed, with alignment captured before the first event."""
        import torch

        floors, floor_stamps, _, _ = self.capture_pairs(problem, staged, iters, ("dispatch", "combine"))
        period, period_stamps, combined, _ = self.capture_pairs(problem, staged, iters, ("pair",))
        for graph in (floors, period):
            graph.replay()
            torch.cuda.synchronize()
            graph.replay()
            torch.cuda.synchronize()
        return _chain_result(period_stamps["pair"], floor_stamps, drop, combined)

    def replay_output(self, problem):
        """An untimed capture with `stage` INSIDE the graph. Dispatch's output and the result are
        poisoned before the only replay, so the output is valid only if that replay re-ran
        dispatch, stage and combine."""
        import torch
        import torch.distributed as dist

        self.backend.warm(problem, 1)
        graph, _, combined, handle = self.capture_pairs(problem, None, 1, ())
        graph.replay()  # first launch uploads the graph; its output is discarded
        torch.cuda.synchronize()
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
