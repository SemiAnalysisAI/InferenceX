"""Warmup, isolated-component windows, and free-running EP timing templates."""
from __future__ import annotations

from ep_measurement import time_us


class EPTiming:
    """Measure an EPBackend using its dispatch, stage, combine, and pairing contract."""

    # ---- Timing template methods -----------------------------------------------------

    def timed_components(self):
        """Components measured for this backend: roundtrip, dispatch and combine
        always; stage only when it launches device work."""
        components = ["roundtrip", "dispatch", "combine"]
        if self.stage_device_work:
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
        if component == "roundtrip":
            return self.benchmark_roundtrip(problem, warmup, iters)
        if component == "dispatch":
            return self.benchmark_dispatch(problem, warmup, iters)
        if component == "stage":
            return self.benchmark_stage(problem, warmup, iters)
        if component == "combine":
            return self.benchmark_combine(problem, warmup, iters)
        raise RuntimeError(f"unknown timed component {component!r}")

    def benchmark_roundtrip(self, problem, warmup, iters):
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
