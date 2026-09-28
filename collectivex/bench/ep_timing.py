"""Timing strategies: eager CUDA-event windows and CUDA graph replay, one object per regime."""
from __future__ import annotations

import functools
import os


def time_us(torch, fn, iters: int, pre=None, post=None) -> list[float]:
    """Per-iteration CUDA-event latencies (us) for THIS rank.

    `pre()` runs untimed before each sample and its result is passed to `fn`; `post(result)` runs
    after the end event and sync. There is deliberately NO sync between `pre()` and the start
    event: stream order already keeps pre()'s work out of the window, and a sync would put the
    host's launch of the collective inside it and let launch jitter stagger the ranks (b200 uccl-ep
    LL combine: 113.4us with a sync vs 87.4us without at T=1). Each iteration ends synchronized:
    iteration N+1's dispatch must not race iteration N's combine on the persistent buffer.
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

    return [sample() for _ in range(iters)]


def time_cuda_graph_phase_us(torch, fn, warmup: int, iters: int, interval, align) -> list[float]:
    """Time one event-record interval captured inside graph replay.

    `align()` enqueues a device-side rank barrier before each replay, so replays start together
    rather than ~75us apart (b200 EP16), which the cross-rank MAX would report as latency.
    """
    for _ in range(max(0, warmup)):
        fn()
        torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        align()
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
    """CUDA events recorded on the stream around eagerly launched operations."""

    graph = False

    def __init__(self, backend):
        self.backend = backend

    def components(self):
        """Roundtrip, dispatch and combine always; stage only when it launches device work."""
        names = ["roundtrip", "dispatch", "combine"]
        return names + ["stage"] if self.backend.stage_device_work else names

    def component(self, name, problem, warmup, iters):
        import torch

        b = self.backend
        if name == "roundtrip":
            staged = b.warm_and_hoist_stage(problem, warmup)
            return time_us(torch, lambda: b.run_roundtrip(problem, staged), iters)
        if name == "dispatch":
            b.warm(problem, warmup)

            def finish(handle):
                b.stage(problem, handle)
                b.combine(problem, handle)

            return time_us(torch, lambda: b.dispatch(problem), iters,
                           post=finish if b.requires_fresh_pair else None)
        if name == "stage":
            # Staging is the timed operation here, so it must be warmed on every iteration.
            b.warm(problem, warmup, stage_every=True)

            def stage(handle):
                b.stage(problem, handle)
                return handle

            return time_us(torch, stage, iters, pre=lambda: b.dispatch(problem),
                           post=(lambda h: b.combine(problem, h)) if b.requires_fresh_pair else None)
        if name == "combine":
            b.warm(problem, warmup)

            def staged_dispatch():
                handle = b.dispatch(problem)
                b.stage(problem, handle)
                return handle

            if b.requires_fresh_pair:
                return time_us(torch, lambda h: b.combine(problem, h), iters, pre=staged_dispatch)
            handle = staged_dispatch()
            torch.cuda.synchronize()
            return time_us(torch, lambda: b.combine(problem, handle), iters)
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
        for i in range(iters):
            floors["dispatch"][0][i].record()
            handle = b.dispatch(problem)
            floors["dispatch"][1][i].record()
            b.stage_or_reuse(problem, handle, staged)
            floors["combine"][0][i].record()
            b.combine(problem, handle)
            floors["combine"][1][i].record()
        torch.cuda.synchronize()
        for i in range(iters):
            pair[0][i].record()
            handle = b.dispatch(problem)
            b.stage_or_reuse(problem, handle, staged)
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
    """Captured dispatch -> combine pairs timed by event nodes inside the graph, each timed replay
    started behind a device-side rank barrier so the cross-rank MAX is the operation, not launch
    skew."""

    graph = True
    # Handle attributes dispatch writes; the replay value check poisons them.
    DISPATCH_OUTPUT_FIELDS = ("recv_x", "recv_scales", "dispatch_output")

    def __init__(self, backend):
        self.backend = backend
        self.align_cycles = None
        self._align_token = None

    def components(self):
        return ["roundtrip", "dispatch", "combine"]

    def calibrate_align_spin(self, spin_us=100.0):
        """Size the post-barrier spin to `spin_us` of wall time on this GPU.

        The spin lets every host enqueue its replay before the stream reaches it. `_sleep` counts
        SM cycles, so a fixed count left ~15us of skew between differently clocked gb200 ranks.
        """
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
        self.align_cycles = max(1, int(probe * spin_us / elapsed_us))

    def align(self):
        """Enqueue a device-side rank barrier on the current stream, without a host sync."""
        import torch
        import torch.distributed as dist

        if self._align_token is None:
            self._align_token = torch.zeros(1, device=self.backend.device)
        if self.align_cycles is None:
            self.calibrate_align_spin()
        dist.all_reduce(self._align_token)
        torch.cuda._sleep(self.align_cycles)

    def capture_pairs(self, problem, staged, pairs, marks):
        """Capture `pairs` back-to-back pairs into one graph; `marks` picks the windows that get
        event nodes ("pair", "dispatch", "combine"). Returns (graph, {mark: (starts, ends)},
        last combined output, last dispatch handle)."""
        import torch
        import torch.distributed as dist

        b = self.backend
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
        self.calibrate_align_spin()  # clocks move with load and temperature: per timed series
        starts, ends = stamps[mark]
        samples = time_cuda_graph_phase_us(
            torch, graph.replay, warmup, iters, (starts[0], ends[0]), align=self.align
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
        once untimed (upload) and once aligned and timed."""
        import torch

        floors, floor_stamps, _, _ = self.capture_pairs(problem, staged, iters, ("dispatch", "combine"))
        period, period_stamps, combined, _ = self.capture_pairs(problem, staged, iters, ("pair",))
        self.calibrate_align_spin()
        for graph in (floors, period):
            graph.replay()
            torch.cuda.synchronize()
            self.align()
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
