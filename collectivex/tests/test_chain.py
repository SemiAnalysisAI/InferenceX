#!/usr/bin/env python3
"""The chained pair period end to end: the timing primitive, its publication through run_sweep, and the headline it feeds.

Two torch doubles live here on purpose. The trace_* family logs record()/sync()/all_reduce against a controllable clock, so event PLACEMENT is assertable; the value_* family carries real tensor arithmetic, so the published NUMBERS are. Merging them would be more complex than either."""
from __future__ import annotations

import contextlib
import io
import itertools
import sys
import types
import unittest
from pathlib import Path
from unittest import mock
import copy
import gc
import json
import math
import os
import statistics
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "bench"), str(ROOT / "tests")]

import ep_backend  # noqa: E402
import ep_harness  # noqa: E402
import ep_oracle  # noqa: E402
import ep_timing  # noqa: E402
import gpu_clocks  # noqa: E402
import summarize  # noqa: E402
from _fakes import COMBINE_MS, DISPATCH_MS, _ChainBackend, _Combined  # noqa: E402


class _TraceEvent:
    """torch.cuda.Event stand-in; logs record() into the shared call trace so placement is
    assertable."""

    def __init__(self, clock, log=None):
        self._clock = clock
        self._log = log
        self.t = None

    def record(self, *_args, **_kwargs):
        self.t = self._clock.now_ms
        if self._log is not None:
            self._log.append("record")

    def elapsed_time(self, other):
        if self.t is None or other.t is None:
            raise AssertionError("elapsed_time on an event that was never recorded")
        return other.t - self.t


class _TraceGraph:
    def __init__(self, clock, log):
        self.clock, self.log = clock, log

    def replay(self):
        self.log.append("graph_replay")
        self.clock.advance(DISPATCH_MS + COMBINE_MS)


@contextlib.contextmanager
def trace_torch(clock, log):
    """Install a stub `torch`/`torch.distributed` that logs the calls this contract is about."""
    dist = types.SimpleNamespace(
        all_reduce=lambda *args, **kwargs: log.append("all_reduce"),
        barrier=lambda *args, **kwargs: log.append("dist_barrier"),
    )

    @contextlib.contextmanager
    def graph_context(_graph, **kwargs):
        if kwargs.get("capture_error_mode", "global") != "global":
            raise AssertionError("graph capture must use torch's default (global) mode")
        log.append("capture_begin")
        yield
        log.append("capture_end")

    torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            Event=lambda *args, **kwargs: _TraceEvent(clock, log),
            CUDAGraph=lambda: _TraceGraph(clock, log),
            graph=graph_context,
            synchronize=lambda *args, **kwargs: log.append("sync"),
            _sleep=lambda *args, **kwargs: log.append("align_spin"),
        ),
        distributed=dist,
        version=types.SimpleNamespace(hip=None),
        zeros=lambda *args, **kwargs: object(),
        isfinite=lambda _value: types.SimpleNamespace(
            all=lambda: types.SimpleNamespace(item=lambda: True)
        ),
    )
    with mock.patch.dict(sys.modules, {"torch": torch, "torch.distributed": dist}):
        yield torch


def new_problem():
    """A problem the backend can hang cached state on -- `warm` caches recv_tokens there."""
    return types.SimpleNamespace()


def run_chain(backend, iters, drop):
    with trace_torch(backend.clock, backend.calls):
        return backend.benchmark_chain(new_problem(), 0, iters, drop)


def timed_tail(calls, iters, per_pair):
    """The period chain's ops: the last `per_pair * iters` op entries, records and syncs removed."""
    trace = [entry for entry in calls if entry != "record"]
    while trace and trace[-1] in ("sync", "all_reduce", "dist_barrier"):
        trace.pop()
    return trace[-per_pair * iters:]


def ops_only(calls):
    """Just the backend operations, in order."""
    return [entry for entry in calls if entry in ("dispatch", "stage", "combine")]


def chain_sections(calls):
    """(floors_chain, period_chain) raw-trace slices; the chains sit between the last three syncs."""
    sync_idx = [i for i, entry in enumerate(calls) if entry == "sync"]
    end_period, end_floors = sync_idx[-1], sync_idx[-2]
    start_floors = sync_idx[-3] + 1 if len(sync_idx) >= 3 else 0
    return calls[start_floors:end_floors], calls[end_floors + 1:end_period]


class ChainedPairPeriod(unittest.TestCase):
    def test_the_loop_is_free_running_dispatch_combine_pairs(self):
        # A host sync inside the loop drains the GPU and turns the period back into a sequence of
        # drained roundtrips; a cross-rank call between pairs re-aligns the ranks and buys back
        # the very stagger the chain exists to amortise. Neither may appear. Each sibling is
        # aligned once, before its first event, by the all-reduce alone: a spin in front would let
        # a host-bound chain queue a backlog the GPU then drains at a second speed.
        iters = 6
        backend = _ChainBackend()
        run_chain(backend, iters, 2)
        self.assertEqual(
            timed_tail(backend.calls, iters, 2), ["dispatch", "combine"] * iters
        )
        for section in chain_sections(backend.calls):
            first_event = section.index("record")
            self.assertEqual(section[:first_event], ["all_reduce"])
            self.assertNotIn("all_reduce", section[first_event:])
        self.assertNotIn("dist_barrier", backend.calls)
        self.assertNotIn("align_spin", backend.calls)

    def test_a_hoisted_stage_runs_once_and_stays_out_of_every_pair_window(self):
        # The conversion is materialised once, untimed, so the pair is dispatch -> combine in
        # both chains. The dequant hatch is fp8-only, so a bf16 row keeps its hoist regardless.
        for precision, consume in (("bf16", "native"), ("fp8", "native"), ("bf16", "dequant")):
            with self.subTest(precision=precision, consume=consume):
                iters = 6
                backend = _ChainBackend(
                    stage_device_work=True, fp8_consume=consume, precision=precision
                )
                self.assertTrue(backend.stage_excluded_from_roundtrip)
                series = run_chain(backend, iters, 2)
                self.assertEqual(backend.calls.count("stage"), 1)
                self.assertEqual(backend.consumed, ["staged-by-stage"] * (2 * iters + 1))
                floors, period = chain_sections(backend.calls)
                self.assertEqual(ops_only(floors), ["dispatch", "combine"] * iters)
                self.assertEqual(ops_only(period), ["dispatch", "combine"] * iters)
                # The staged cost is absent from the pair window, not merely from the trace.
                for value in series["pair"]:
                    self.assertAlmostEqual(value, 8000.0)  # 3ms dispatch + 5ms combine
                for value in series["start_to_start"]:
                    self.assertAlmostEqual(value, 8000.0)
                for value in series["dispatch"]:
                    self.assertAlmostEqual(value, 3000.0)
                for value in series["combine"]:
                    self.assertAlmostEqual(value, 5000.0)

    def test_returns_one_sample_per_kept_iteration(self):
        for iters, drop, kept, gaps in ((8, 0, 8, 7), (8, 2, 6, 5), (6, 5, 1, 0)):
            with self.subTest(iters=iters, drop=drop):
                series = run_chain(_ChainBackend(), iters, drop)
                self.assertEqual(
                    sorted(series),
                    ["combine", "combined", "dispatch", "host_retries", "host_stalled", "pair",
                     "start_to_start"],
                )
                # Only a dispatch that waits on the host passes its chain through the host gate.
                self.assertEqual((series["host_retries"], series["host_stalled"]), (None, None))
                for key in ("pair", "dispatch", "combine"):
                    self.assertEqual(len(series[key]), kept)
                # Start-to-start is a difference series: one fewer than the kept pairs.
                self.assertEqual(len(series["start_to_start"]), gaps)

    def test_the_dropped_iterations_are_the_head_of_each_chain(self):
        # `drop` discards pipeline fill, so it must cut the head of both chains -- the period
        # chain refills after the inter-chain synchronize.
        iters, drop = 6, 2
        slow_head = [50.0] * drop + [DISPATCH_MS] * (iters - drop)
        backend = _ChainBackend(
            stage_device_work=False, fp8_consume="native", precision="bf16",
            dispatch_schedule=slow_head * 2,  # floors chain runs first, then the period chain
        )
        series = run_chain(backend, iters, drop)
        self.assertEqual(len(series["dispatch"]), 4)
        for value in series["dispatch"]:
            self.assertAlmostEqual(value, 3000.0)
        for value in series["pair"]:
            self.assertAlmostEqual(value, 15000.0)  # 3ms dispatch + 7ms stage + 5ms combine


class ChainHostGate(unittest.TestCase):
    """A chain whose dispatch waits on the host relaunches from the host inside every pair, so each
    sibling opens only once this host runs at speed and restarts, on every rank together, when any
    rank's host throttled around it."""

    def _gated_chain(self, verdicts, iters=4, drop=1):
        backend = _ChainBackend(stage_device_work=False, fp8_consume="native", precision="bf16")
        backend.host_synced_dispatch = True
        script = iter(verdicts)

        def stalled_anywhere(_timing):
            backend.calls.append("stall_check")
            return next(script)

        with mock.patch.object(ep_timing, "cpu_probe_us",
                               lambda: backend.calls.append("probe") or 20.0), \
                mock.patch.object(ep_timing.EagerTiming, "_host_stalled_anywhere", stalled_anywhere):
            series = run_chain(backend, iters, drop)
        self.assertRaises(StopIteration, next, script)  # every verdict was consumed
        return backend, series

    def test_each_sibling_is_held_then_aligned_and_a_stalled_one_restarts(self):
        backend, series = self._gated_chain([False, True, False])
        calls = backend.calls
        barriers = [i for i, call in enumerate(calls) if call == "all_reduce"]
        # Floors once; the period sibling twice, the first attempt discarded.
        self.assertEqual(len(barriers), 3)
        for i in barriers:
            self.assertEqual(calls[i - 1], "probe")  # the hold, outside every window
            self.assertEqual(calls[i + 1], "record")
            self.assertEqual(calls.index("stall_check", i), calls.index("sync", i) + 1)
        self.assertEqual(calls.count("stall_check"), 3)
        self.assertEqual((series["host_retries"], series["host_stalled"]), (1, False))
        self.assertEqual(len(series["pair"]), 3)

    def test_a_sibling_still_stalled_after_its_last_restart_marks_the_trial(self):
        attempts = ep_timing.EagerTiming.CHAIN_RETRIES + 1
        backend, series = self._gated_chain([True] * (2 * attempts))
        self.assertEqual(backend.calls.count("all_reduce"), 2 * attempts)
        self.assertEqual(series["host_retries"], 2 * (attempts - 1))
        self.assertIs(series["host_stalled"], True)

    def test_every_rank_restarts_when_any_ranks_host_stalled(self):
        # The verdict is a MAX all-reduce of every rank's own check, so all ranks run the same
        # number of attempts and their chains stay paired.
        for peer, expected in ((0, False), (1, True)):
            with self.subTest(peer_stalled=peer):
                class Flag:
                    def __init__(self, data, **_kwargs):
                        self.value = data[0]

                    def item(self):
                        return self.value

                def all_reduce(flag, op=None):
                    self.assertEqual(op, "max")
                    flag.value = max(flag.value, peer)

                dist = types.SimpleNamespace(all_reduce=all_reduce,
                                             ReduceOp=types.SimpleNamespace(MAX="max"))
                fake = types.SimpleNamespace(tensor=Flag, int32="int32", distributed=dist)
                timing = ep_timing.EagerTiming(_ChainBackend())
                timing.gate.probe_us = 20.0
                with mock.patch.dict(sys.modules, {"torch": fake, "torch.distributed": dist}), \
                        mock.patch.object(ep_timing, "cpu_probe_us", lambda: 20.0):
                    self.assertIs(timing._host_stalled_anywhere(), expected)
                self.assertEqual(timing.gate.stalled, [0])  # this rank's own check ran

    def test_garbage_collection_is_off_inside_both_sibling_chains(self):
        backend = _ChainBackend()
        seen = []
        original = backend.dispatch
        backend.dispatch = lambda problem: seen.append(gc.isenabled()) or original(problem)
        run_chain(backend, 4, 1)
        self.assertEqual(seen[-8:], [False] * 8)
        self.assertTrue(gc.isenabled())

    def test_a_chain_without_a_host_synced_dispatch_takes_no_probe(self):
        backend = _ChainBackend()
        with mock.patch.object(ep_timing, "cpu_probe_us",
                               lambda: self.fail("a probe on a chain with no host wait")):
            series = run_chain(backend, 4, 1)
        self.assertEqual((series["host_retries"], series["host_stalled"]), (None, None))


class CudaGraphRoundtrip(unittest.TestCase):
    def test_each_graph_component_uses_its_own_capture(self):
        backend = self._graph_backend(stage_device_work=True)
        problem = new_problem()
        samples = {}
        with mock.patch.dict(os.environ, {}, clear=True), \
                trace_torch(backend.clock, backend.calls):
            for component in ("roundtrip", "dispatch", "combine"):
                samples[component] = backend.benchmark_component(
                    component, problem, warmup=1, iters=4
                )

        self.assertEqual(samples["roundtrip"], [8000.0] * 4)
        self.assertEqual(samples["dispatch"], [3000.0] * 4)
        self.assertEqual(samples["combine"], [5000.0] * 4)
        self.assertEqual(backend.calls.count("capture_begin"), 3)
        self.assertEqual(backend.calls.count("capture_end"), 3)
        self.assertEqual(backend.calls.count("graph_replay"), 18)
        self.assertTrue(problem._cuda_graph_output.cloned)
        self.assertTrue(problem._cuda_graph_output_rewritten)

    def test_one_unrewritten_capture_reds_the_point_however_later_ones_did(self):
        backend = self._graph_backend()
        problem = new_problem()
        problem._cuda_graph_output_rewritten = False  # an earlier component's capture
        with mock.patch.dict(os.environ, {}, clear=True), \
                trace_torch(backend.clock, backend.calls):
            backend.benchmark_component("dispatch", problem, warmup=1, iters=2)
        self.assertIs(problem._cuda_graph_output_rewritten, False)

    def _graph_backend(self, stage_device_work=False, precision="bf16"):
        backend = _ChainBackend(stage_device_work=stage_device_work, precision=precision)
        backend.mode, backend.CUDA_GRAPH_MODES = "normal", ("normal",)
        return backend

    def test_a_spin_then_the_alignment_are_captured_before_the_timed_window(self):
        # HIP submits a replay node by node from the host, so the captured spin is what keeps a
        # host lagging behind the all-reduce from opening its peers' windows early.
        backend = self._graph_backend()
        with mock.patch.dict(os.environ, {}, clear=True), \
                trace_torch(backend.clock, backend.calls):
            backend.benchmark_component("roundtrip", new_problem(), warmup=2, iters=3)
        calls = backend.calls
        begin, end = calls.index("capture_begin"), calls.index("capture_end")
        first_event = calls.index("record", begin)
        self.assertEqual(calls[begin + 1:first_event], ["align_spin", "all_reduce"])
        self.assertNotIn("all_reduce", calls[first_event:end])
        self.assertNotIn("align_spin", calls[first_event:end])
        self.assertNotIn("all_reduce", calls[end:])
        # The spin's cycle count is calibrated after the warm-up and before the capture.
        calibration = calls.index("align_spin")
        self.assertEqual(calls[calibration:calibration + 5],
                         ["align_spin", "record", "align_spin", "record", "sync"])
        self.assertLess(calls.index("combine"), calibration)
        self.assertLess(calibration, begin)

    def test_a_replay_call_that_outlasts_the_spin_counts_as_missed(self):
        # One fake millisecond per perf_counter() read: every replay call's lead is 1000 us.
        for spin_us, missed in (("500", 1), ("2000", 0)):
            with self.subTest(spin_us=spin_us):
                backend = self._graph_backend()
                ticks = (tick / 1000.0 for tick in itertools.count())
                with mock.patch.dict(os.environ, {"COLLX_ALIGN_SPIN_US": spin_us}, clear=True), \
                        trace_torch(backend.clock, backend.calls), \
                        mock.patch.object(ep_timing.time, "perf_counter", lambda: next(ticks)):
                    backend.benchmark_component("roundtrip", new_problem(), warmup=1, iters=4)
                self.assertEqual(backend.alignment_misses(), [missed] * 4)
                self.assertEqual(backend.timing.alignment,
                                 {"method": "in-graph-spin-then-all-reduce",
                                  "spin_us": float(spin_us)})

    def test_the_graph_chain_is_one_capture_of_unrolled_pairs_per_sibling(self):
        iters, drop = 5, 1
        backend = self._graph_backend()
        with mock.patch.dict(os.environ, {}, clear=True):
            series = run_chain(backend, iters, drop)
        calls = backend.calls
        begin = [i for i, call in enumerate(calls) if call == "capture_begin"]
        end = [i for i, call in enumerate(calls) if call == "capture_end"]
        # Floors and period siblings: one capture each holding every pair, each replayed once
        # untimed and once behind the barrier.
        self.assertEqual(len(begin), 2)
        for lo, hi in zip(begin, end):
            self.assertEqual(ops_only(calls[lo:hi]), ["dispatch", "stage", "combine"] * iters)
        self.assertEqual(calls.count("graph_replay"), 4)
        for lo, hi in zip(begin, end):
            first_event = calls.index("record", lo)
            self.assertEqual(calls[lo + 1:first_event], ["align_spin", "all_reduce"])
            self.assertNotIn("all_reduce", calls[first_event:hi])
            self.assertNotIn("align_spin", calls[first_event:hi])
        self.assertEqual({len(series[k]) for k in ("pair", "dispatch", "combine")}, {iters - drop})
        self.assertEqual(len(series["start_to_start"]), iters - drop - 1)
        self.assertTrue(series["combined"].cloned)
        self.assertEqual((series["host_retries"], series["host_stalled"]), (None, None))

    def test_external_switch_restores_the_eager_component_pipeline(self):
        backend = self._graph_backend(stage_device_work=True, precision="fp8")
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                backend.timed_components(), ["roundtrip", "dispatch", "combine"]
            )
        with mock.patch.dict(os.environ, {"COLLX_CUDA_GRAPH": "0"}, clear=True):
            self.assertEqual(
                backend.timed_components(), ["roundtrip", "dispatch", "combine", "stage"]
            )
        with mock.patch.dict(os.environ, {"COLLX_CUDA_GRAPH": "maybe"}, clear=True), \
                self.assertRaisesRegex(ValueError, "COLLX_CUDA_GRAPH"):
            backend.timed_components()


class EagerAlignment(unittest.TestCase):
    """Eager samples are aligned the way DeepEP's bench_kineto aligns them: a GPU spin, then a
    device all-reduce, both queued ahead of the sample, with no host sync before the window."""

    def _eager_component(self, component, iters=3, backend=None, env=None):
        backend = backend or _ChainBackend()
        with mock.patch.dict(os.environ, {"COLLX_CUDA_GRAPH": "0", **(env or {})}, clear=True), \
                trace_torch(backend.clock, backend.calls):
            backend.benchmark_component(component, new_problem(), warmup=2, iters=iters)
        return backend.calls

    def test_every_eager_sample_opens_with_a_spin_then_a_device_all_reduce(self):
        iters = 3
        for component in ("roundtrip", "dispatch", "combine"):
            with self.subTest(component=component):
                calls = self._eager_component(component, iters)
                barriers = [i for i, call in enumerate(calls) if call == "all_reduce"]
                self.assertEqual(len(barriers), iters)
                for i in barriers:
                    self.assertEqual(calls[i - 1], "align_spin")
                    # The window opens with no host sync after the barrier.
                    window_start = calls.index("record", i)
                    self.assertNotIn("sync", calls[i:window_start])

    def test_the_spin_is_calibrated_after_the_warm_up_and_before_the_first_sample(self):
        for component in ("roundtrip", "dispatch", "combine"):
            with self.subTest(component=component):
                calls = self._eager_component(component)
                # `_sleep` counts SM cycles: a clock ramp, then a spin bracketed by two events and
                # a sync converts the spin's wall time to cycles at the clocks warm-up left behind.
                calibration = calls.index("align_spin")
                self.assertEqual(calls[calibration:calibration + 5],
                                 ["align_spin", "record", "align_spin", "record", "sync"])
                self.assertIn("combine", calls[:calibration])
                self.assertLess(calibration, calls.index("all_reduce"))

    def test_combine_is_timed_behind_its_own_untimed_dispatch(self):
        # Every combine sample is a fresh pair, as graph replay and the roundtrip time it, so the
        # window opens behind its own dispatch on every backend. The barrier releases that
        # untimed dispatch, and the stage, hoisted once untimed, appears in no sample.
        for fresh in (True, False):
            with self.subTest(requires_fresh_pair=fresh):
                backend = _ChainBackend()
                backend.requires_fresh_pair = fresh
                self.assertTrue(backend.stage_excluded_from_roundtrip)
                iters = 3
                calls = self._eager_component("combine", iters, backend=backend)
                barriers = [i for i, call in enumerate(calls) if call == "all_reduce"]
                self.assertEqual(len(barriers), iters)
                for i in barriers:
                    self.assertEqual(calls[i - 1:i + 6], ["align_spin", "all_reduce", "dispatch",
                                                          "record", "combine", "record", "sync"])
                self.assertEqual(backend.consumed[-iters:], ["staged-by-stage"] * iters)

    def test_a_dispatch_that_waits_on_the_host_runs_before_the_alignment(self):
        # Its wait for the receive counts would spend the spin and count every sample as missed;
        # run first, it leaves the barrier to release the timed op alone.
        for component in ("combine", "stage"):
            with self.subTest(component=component):
                backend = _ChainBackend(stage_device_work=True, fp8_consume="dequant")
                backend.host_synced_dispatch = True
                iters = 3
                calls = self._eager_component(component, iters, backend=backend)
                barriers = [i for i, call in enumerate(calls) if call == "all_reduce"]
                self.assertEqual(len(barriers), iters)
                for i in barriers:
                    opened = calls.index("record", i)
                    self.assertEqual(calls[i - 1], "align_spin")
                    self.assertIn("dispatch", calls[i - 4:i - 1])
                    self.assertNotIn("dispatch", calls[i:opened])
                    self.assertEqual(calls[opened + 1], component)

    def test_a_sample_counts_as_missed_when_its_host_outruns_the_spin(self):
        # One fake millisecond per perf_counter() read: every sample's lead is 1000 us.
        for spin_us, missed in (("500", 1), ("2000", 0)):
            with self.subTest(spin_us=spin_us):
                backend = _ChainBackend()
                ticks = (tick / 1000.0 for tick in itertools.count())
                with mock.patch.object(ep_timing.time, "perf_counter", lambda: next(ticks)):
                    self._eager_component("roundtrip", 4, backend=backend,
                                          env={"COLLX_ALIGN_SPIN_US": spin_us})
                self.assertEqual(backend.alignment_misses(), [missed] * 4)
                self.assertEqual(backend.timing.alignment["spin_us"], float(spin_us))

    def test_the_lead_covers_the_timed_ops_enqueue_unless_it_waits_on_the_host(self):
        # A host still enqueuing the timed op when the barrier releases leaves its GPU idle inside
        # the window, so the lead runs through fn()'s return. A dispatch that waits on the host
        # returns after its own device work, so its lead stops at the start event instead and
        # the host gate covers it.
        for host_synced, missed in ((False, 1), (True, 0)):
            with self.subTest(host_synced=host_synced):
                backend = _ChainBackend()
                backend.host_synced_dispatch = host_synced
                host = [0.0]
                original = backend.dispatch

                def slow_enqueue(problem):
                    host[0] += 3e-3  # 3 ms of host time, past the 2 ms spin
                    return original(problem)

                backend.dispatch = slow_enqueue
                with contextlib.ExitStack() as stack:
                    stack.enter_context(
                        mock.patch.object(ep_timing.time, "perf_counter", lambda: host[0]))
                    stack.enter_context(mock.patch.object(ep_timing, "cpu_probe_us", lambda: 20.0))
                    self._eager_component("dispatch", 3, backend=backend)
                self.assertEqual(backend.alignment_misses(), [missed] * 3)

    def test_garbage_collection_is_off_inside_a_timed_series(self):
        backend = _ChainBackend()
        seen = []
        original = backend.dispatch
        backend.dispatch = lambda problem: seen.append(gc.isenabled()) or original(problem)
        self.assertTrue(gc.isenabled())
        self._eager_component("roundtrip", 3, backend=backend)
        self.assertEqual(seen[-3:], [False] * 3)
        self.assertTrue(gc.isenabled())

    def test_graph_replay_counts_misses_but_has_no_host_gate(self):
        timing = ep_timing.GraphTiming(_ChainBackend())
        self.assertEqual(timing.missed_samples(), [])
        self.assertIsNone(timing.host_holds())
        self.assertIsNone(timing.host_stalls())

    # Calibrated at 20 us, a sample holds above 2.5 x 20 = 50 us and counts stalled above 100 us.
    CALIBRATION = [20.0] * ep_timing.HostGate.CALIBRATION

    def _gated_component(self, component, probe_us, iters=4, backend=None, tick_s=None):
        """Run one eager series on a backend whose dispatch waits on the host, every CPU probe
        logged into the trace and timed from `probe_us` (calibration reads first); with `tick_s`,
        every perf_counter() read advances a fake clock by that much."""
        backend = backend or _ChainBackend()
        backend.host_synced_dispatch = True
        readings = iter(probe_us)

        def probe():
            backend.calls.append("probe")
            return next(readings)

        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(ep_timing, "cpu_probe_us", probe))
            if tick_s is not None:
                ticks = (tick * tick_s for tick in itertools.count())
                stack.enter_context(
                    mock.patch.object(ep_timing.time, "perf_counter", lambda: next(ticks)))
            self._eager_component(component, iters, backend=backend)
        self.assertRaises(StopIteration, next, readings)  # every reading was taken
        return backend

    def test_a_host_synced_series_probes_the_cpu_around_each_sample_outside_its_window(self):
        calibration = ep_timing.HostGate.CALIBRATION
        for component, ops in (("roundtrip", ["dispatch", "combine"]), ("dispatch", ["dispatch"])):
            with self.subTest(component=component):
                iters = 3
                readings = [20.0] * (calibration + 2 * iters)
                backend = self._gated_component(component, readings, iters)
                calls = backend.calls
                first = calls.index("all_reduce") - 2
                self.assertEqual(calls[first - calibration:first], ["probe"] * calibration)
                # Before the spin, and after the sample's final sync: never between its events.
                sample = ["probe", "align_spin", "all_reduce", "record", *ops, "record", "sync",
                          "probe"]
                self.assertEqual(calls[first:], sample * iters)
                self.assertEqual(backend.host_holds(), [0] * iters)
                self.assertEqual(backend.host_stalls(), [0] * iters)

    def test_a_series_without_a_host_synced_dispatch_takes_no_probe(self):
        backend = self._gated_component("combine", [], iters=3)
        self.assertNotIn("probe", backend.calls)
        self.assertIsNone(backend.host_holds())
        self.assertIsNone(backend.host_stalls())
        backend = _ChainBackend()
        self._eager_component("roundtrip", 3, backend=backend)
        self.assertIsNone(backend.host_holds())
        self.assertIsNone(backend.host_stalls())

    def test_a_sample_opens_only_once_its_slow_host_runs_at_speed_again(self):
        # The second sample's first probe runs throttled (400 us) and the next one still winding
        # down (100 us, above the 50 us hold bar): its spin and barrier are enqueued only after the
        # probe back under the bar, 2 fake ms later.
        backend = self._gated_component(
            "dispatch", self.CALIBRATION + [20.0, 20.0, 400.0, 100.0, 40.0, 20.0, 20.0, 20.0],
            iters=3, tick_s=1e-3)
        calls = backend.calls
        second = calls.index("all_reduce", calls.index("all_reduce") + 1)
        self.assertEqual(calls[second - 6:second + 1],
                         ["sync", "probe", "probe", "probe", "probe", "align_spin", "all_reduce"])
        self.assertEqual(backend.host_holds(), [0, 1, 0])
        self.assertEqual(backend.host_stalls(), [0, 0, 0])
        self.assertEqual(backend.alignment_misses(), [0, 0, 0])

    def test_a_hold_shorter_than_a_millisecond_was_an_interrupt_and_is_not_counted(self):
        backend = self._gated_component(
            "dispatch", self.CALIBRATION + [20.0, 20.0, 60.0, 20.0, 20.0, 20.0, 20.0], iters=3,
            tick_s=1e-4)
        self.assertEqual(backend.calls.count("probe"), ep_timing.HostGate.CALIBRATION + 7)
        self.assertEqual(backend.host_holds(), [0, 0, 0])
        self.assertEqual(backend.host_stalls(), [0, 0, 0])

    def test_a_throttle_that_begins_inside_the_window_counts_the_sample_stalled(self):
        # Opened at speed, throttled by the probe after the final sync: the relaunch inside the
        # window may have run slow, and nothing held it. A probe under the stall bar is not.
        backend = self._gated_component(
            "dispatch", self.CALIBRATION + [20.0, 20.0, 20.0, 400.0, 20.0, 99.0, 20.0, 20.0])
        self.assertEqual(backend.host_holds(), [0, 0, 0, 0])
        self.assertEqual(backend.host_stalls(), [0, 1, 0, 0])

    def test_a_hold_that_runs_out_opens_the_sample_and_counts_it_stalled_if_still_throttled(self):
        # Every perf_counter() read advances 300 fake ms against a 0.5 s hold limit: the second
        # sample gives up still throttled, the third still winding down (60 us, under the stall
        # bar).
        backend = self._gated_component(
            "dispatch", self.CALIBRATION + [20.0, 20.0, 400.0, 400.0, 400.0, 20.0,
                                            60.0, 60.0, 60.0, 20.0], iters=3, tick_s=0.3)
        self.assertEqual(backend.host_holds(), [0, 1, 1])
        self.assertEqual(backend.host_stalls(), [0, 1, 0])

    def test_the_hold_budget_spans_the_run(self):
        # Every perf_counter() read advances 300 fake ms: the first series' hold spends 0.6 s of a
        # 0.5 s budget, so the next series' throttled sample opens without waiting, counted stalled.
        backend = _ChainBackend()
        with mock.patch.object(ep_timing.HostGate, "BUDGET_S", 0.5):
            self._gated_component(
                "dispatch", self.CALIBRATION + [400.0, 400.0, 20.0, 20.0, 20.0, 20.0], iters=2,
                backend=backend, tick_s=0.3)
            self.assertEqual((backend.host_holds(), backend.host_stalls()), ([1, 0], [0, 0]))
            self._gated_component("dispatch", self.CALIBRATION + [400.0, 20.0, 20.0, 20.0],
                                  iters=2, backend=backend, tick_s=0.3)
        self.assertEqual((backend.host_holds(), backend.host_stalls()), ([0, 0], [1, 0]))

    def test_a_calibration_taken_while_throttled_cannot_raise_the_bar(self):
        # The run keeps its lowest calibration: a series calibrated inside a throttle (400 us)
        # still judges its samples against the 20 us an earlier series measured at speed.
        backend = self._gated_component("dispatch", self.CALIBRATION + [20.0] * 8)
        self._gated_component("dispatch", [400.0] * ep_timing.HostGate.CALIBRATION
                              + [20.0, 150.0] + [20.0] * 6, backend=backend)
        self.assertEqual(backend.host_holds(), [0, 0, 0, 0])
        self.assertEqual(backend.host_stalls(), [1, 0, 0, 0])
        self.assertEqual(backend.timing.gate.probe_us, 20.0)


class GraphAlignmentAndValueCheck(unittest.TestCase):
    def test_the_replay_value_check_poisons_what_dispatch_wrote_before_replaying(self):
        backend = _ChainBackend(stage_device_work=True, fp8_consume="native", precision="fp8")
        order, combined = [], _Combined(1.0)
        handle = types.SimpleNamespace(recv_x="recv", recv_scales=None, combine_input="staged")
        graph = types.SimpleNamespace(replay=lambda: order.append("replay"))
        backend.warm = lambda problem, count: order.append("warm")
        timing = ep_timing.GraphTiming(backend)
        timing.capture_pairs = lambda problem, staged, pairs, marks: (
            order.append(("capture", staged)) or (graph, {}, combined, handle)
        )
        dist = types.SimpleNamespace(barrier=lambda: order.append("barrier"))
        fake = types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=lambda: None),
                                     distributed=dist)
        with mock.patch.dict(sys.modules, {"torch": fake, "torch.distributed": dist}), \
                mock.patch.object(ep_timing, "poison", lambda tensor: order.append(("poison", tensor))):
            result = timing.replay_output(new_problem())
        # Staging runs INSIDE the capture (staged=None). Between the upload replay and the
        # returned one, dispatch's output, the staged combine input and the result are poisoned,
        # then every rank barriers: peers write into each other's buffers, so an unbarriered
        # replay races a slow poison.
        self.assertEqual(order, [
            "warm", ("capture", None), "replay",
            ("poison", "recv"), ("poison", None), ("poison", None), ("poison", "staged"),
            ("poison", combined), "barrier", "replay",
        ])
        self.assertTrue(result.cloned)

    def test_the_checked_replay_runs_on_rescaled_inputs_that_come_back_bitwise(self):
        # A replay that re-sent its captured inputs instead of re-reading them would still produce
        # finite output; scaling the inputs by an exact power of two between the upload replay and
        # the checked one makes it disagree with the drained pair run on the same scaled inputs.
        class Tensor:
            def __init__(self, dtype, ptr, order):
                self.dtype, self.ptr, self.order, self.scale = dtype, ptr, order, 1.0

            def data_ptr(self):
                return self.ptr

            def mul_(self, factor):
                self.order.append(("scale", self.ptr, factor))
                self.scale *= factor
                return self

        backend = _ChainBackend()
        order, combined = [], _Combined(1.0)
        handle = types.SimpleNamespace(recv_x="recv", recv_scales=None, combine_input=None)
        graph = types.SimpleNamespace(replay=lambda: order.append("replay"))
        backend.warm = lambda problem, count: None
        timing = ep_timing.GraphTiming(backend)
        timing.capture_pairs = lambda problem, staged, pairs, marks: (graph, {}, combined, handle)
        dist = types.SimpleNamespace(barrier=lambda: order.append("barrier"))
        fake = types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=lambda: None),
                                     distributed=dist, bfloat16="bf16", float16="fp16",
                                     float32="fp32")
        x = Tensor("bf16", 1, order)
        problem = types.SimpleNamespace(x=x, dispatch_x=x)  # one tensor under two names
        fp8 = types.SimpleNamespace(x=Tensor("fp8", 2, order), dispatch_x=None)
        with mock.patch.dict(sys.modules, {"torch": fake, "torch.distributed": dist}), \
                mock.patch.object(ep_timing, "poison", lambda tensor: order.append("poison")):
            timing.replay_output(problem)
            self.assertEqual(order[:3], ["replay", ("scale", 1, 2.0), "poison"])
            self.assertEqual(x.scale, 2.0)  # scaled once, not once per name
            timing.restore_replay_inputs(problem)
            self.assertEqual(x.scale, 1.0)
            timing.restore_replay_inputs(problem)  # idempotent: nothing left to undo
            self.assertEqual(x.scale, 1.0)
            # FP8 inputs are left alone: a pre-quantized copy would not see the scale.
            self.assertEqual(ep_timing.rescale_inputs(fp8, 2.0), [])


class RocmGraphEventRecord(unittest.TestCase):
    """ROCm torch < 2.13 rejects Event(external=True); the capture records through HIP instead."""

    def _record(self, rc):
        records, calls = [], []

        class Event:
            cuda_event = 0xE0

            def __init__(self, **kwargs):
                assert not kwargs.get("external"), "External events are disallowed in rocm"

            def record(self):
                records.append("record")

        fake = types.SimpleNamespace(version=types.SimpleNamespace(hip="7.2"), cuda=types.SimpleNamespace(
            Event=Event, current_stream=lambda: types.SimpleNamespace(cuda_stream=0x5)))
        hip = types.SimpleNamespace(hipEventRecordWithFlags=lambda e, s, f: calls.append(
            (e.value, s.value, f.value)) or rc)
        with mock.patch.dict(sys.modules, {"torch": fake}), \
                mock.patch.object(ep_timing, "_hip_runtime", lambda: hip):
            ep_timing.record_graph_event(ep_timing.graph_event())
        return records, calls

    def test_rocm_events_are_recorded_once_outside_then_captured_through_hip(self):
        # One record materializes the event outside capture; the captured one goes through HIP.
        self.assertEqual(self._record(0), (["record"], [(0xE0, 0x5, 0x1)]))

    def test_a_failed_hip_record_raises(self):
        with self.assertRaisesRegex(RuntimeError, "hipEventRecordWithFlags"):
            self._record(1)


class EventPlacement(unittest.TestCase):
    def test_the_period_pairs_carry_only_the_outer_events(self):
        # One record before the dispatch, one after the combine, nothing between: both records'
        # host cost lands in the inter-pair gap, outside the published window. The stub charges
        # host work nothing, so this asserts record placement rather than window values.
        backend = _ChainBackend(stage_device_work=False, fp8_consume="native", precision="bf16")
        run_chain(backend, 4, 1)
        _, period = chain_sections(backend.calls)
        self.assertEqual(
            period, ["all_reduce"] + ["record", "dispatch", "stage", "combine", "record"] * 4)


class ChainBudgetGate(unittest.TestCase):
    """A budget that could publish a degenerate chain must stop the leg first: zero kept pairs
    serialises as "unavailable", indistinguishable from a backend that cannot chain at all, and
    a single kept pair would publish a period whose health scalars are silently unavailable."""

    @staticmethod
    def _args(**updates):
        values = dict(
            mode="normal", iters=8, trials=256, warmup=32,
            chain_iters=128, chain_trials=4, chain_drop=16,
        )
        values.update(updates)
        return types.SimpleNamespace(**values)

    def _gate(self, **updates):
        """rc and rank-0 output; None stands in for every device-side argument."""
        with contextlib.redirect_stdout(io.StringIO()) as out:
            rc = ep_harness.run_sweep(self._args(**updates), None, None, None, None, 0, 1)
        return rc, out.getvalue()

    def test_an_unusable_budget_fails_closed_naming_its_fields(self):
        for label, updates, fields in (
            ("no iterations", dict(chain_iters=0), "chain"),
            ("no trials", dict(chain_trials=0), "chain"),
            # A single kept pair has an empty start-to-start series: interpair gap and settle
            # drift would silently publish as "unavailable" health for a measured period.
            ("drop leaves a single pair", dict(chain_iters=8, chain_drop=7), "chain"),
            ("negative drop", dict(chain_drop=-1), "chain"),
            # The fresh-entry budget is still checked, and names its own fields.
            ("no fresh-entry iterations", dict(iters=0), "iters/trials/warmup"),
        ):
            with self.subTest(budget=label):
                rc, output = self._gate(**updates)
                self.assertEqual(rc, 2)
                self.assertIn(fields, output)


class ChainComponentContract(unittest.TestCase):
    """Component behavior on the paths no chained row takes."""

    def test_an_overridden_origin_leaves_the_rest_of_the_component_alone(self):
        # Every pre-chain row also flows through `_component`, so omitting the override must
        # reproduce the old strings exactly or the chain reclassifies unrelated rows.
        percentiles = {"p50": 1.0, "p90": 2.0, "p95": 3.0, "p99": 4.0}
        self.assertEqual(ep_harness._component(percentiles, 3)["origin"], "measured")
        self.assertEqual(
            ep_harness._component(percentiles, 0, derived=True)["origin"],
            "derived-percentile-sum",
        )
        self.assertIsNone(ep_harness._component(None, 0)["origin"])

        overridden = ep_harness._component(percentiles, 3, origin="chained-median")
        self.assertEqual(overridden["origin"], "chained-median")
        self.assertEqual(overridden["availability"], "measured")
        self.assertEqual(overridden["percentiles_us"], percentiles)
        self.assertEqual(overridden["sample_count"], 3)


LADDER = [4, 8]
CHAIN_ITERS, CHAIN_DROP, CHAIN_TRIALS = 8, 2, 2
# What the stub backend reports for every chained iteration, distinct so a published number is
# traceable to the op it came from; start_to_start sits a fixed GAP above the pair window.
PAIR_US, DISPATCH_FLOOR_US, COMBINE_FLOOR_US = 50.0, 20.0, 25.0
GAP_US, DRIFT_US = 4.0, 8.0
UNAVAILABLE = {
    "availability": "unavailable", "origin": None, "percentiles_us": None, "sample_count": 0,
}


class _ValueTensor:
    """Enough tensor for the reductions run_sweep performs: gather, stack, median/max/min, sub."""

    def __init__(self, data):
        self.data = data

    def tolist(self):
        return self.data

    def item(self):
        return self.data[0]

    def clone(self):
        return _ValueTensor(copy.deepcopy(self.data))

    def to(self, *args, **kwargs):
        return self

    def __sub__(self, other):
        return _ValueTensor([a - b for a, b in zip(self.data, other.data)])

    def _reduce(self, fn):
        columns = (
            [[row[i] for row in self.data] for i in range(len(self.data[0]))]
            if self.data else []
        )
        return SimpleNamespace(values=_ValueTensor([fn(column) for column in columns]))

    def median(self, dim=0):
        return self._reduce(statistics.median)

    def max(self, dim=0):
        return self._reduce(max)

    def min(self, dim=0):
        return self._reduce(min)


class _FakeDist:
    """World size 1, so every collective is the identity and the artifact is this rank's view."""

    ReduceOp = SimpleNamespace(MAX="max", MIN="min", SUM="sum")

    @staticmethod
    def get_world_size():
        return 1

    @staticmethod
    def get_rank():
        return 0

    @staticmethod
    def all_reduce(tensor, op=None):
        return None

    @staticmethod
    def all_gather(out, local):
        out[0].data = list(local.data)

    @staticmethod
    def broadcast(tensor, src=0):
        return None


def value_torch():
    torch = types.ModuleType("torch")
    torch.float64, torch.int64 = "f64", "i64"
    torch.cuda = SimpleNamespace(synchronize=lambda: None)
    torch.tensor = lambda values, device=None, dtype=None: _ValueTensor(list(values))
    torch.empty_like = lambda x: _ValueTensor(list(x.data))
    torch.stack = lambda xs: _ValueTensor([x.data for x in xs])
    torch.equal = lambda a, b: a.data == b.data
    return torch


def fake_routing():
    routing = types.ModuleType("routing")
    routing.routing_stats = lambda idx, experts, per_rank: {
        "empty_expert_count": 0, "empty_rank_count": 0, "expert_assignment_rank_cv": 0.0,
        "expert_assignments_per_rank": [8], "expert_load_cv": 0.0, "expert_load_max": 1,
        "expert_load_mean": 1.0, "expert_load_min": 1, "fanout_histogram": {}, "fanout_max": 1,
        "fanout_mean": 1.0, "fanout_min": 1, "hotspot_ratio": 1.0,
        "payload_copies_per_rank": [1], "payload_rank_cv": 0.0, "routed_copies": 8,
    }
    routing.routing_locality = lambda *args, **kwargs: 1.0
    return routing


class _SweepBackend(ep_backend.EPBackend):
    """Constant-cost backend; every timed call returns a value unique to what it measures."""

    name = "stub"
    maturity = "candidate"

    def __init__(self):
        self.mode = "normal"
        self.precision = "bf16"
        self.stage_device_work = False
        self.fp8_consume = "native"
        self.device = "cuda:0"
        self.events = []

    def make_inputs(self, args):
        spec = ep_backend.WorkloadSpec(
            ep_size=1, experts_per_rank=256, cap=None, dropped=[],
            max_tokens_per_rank=max(LADDER), ladder=list(LADDER),
        )
        for tokens in spec.ladder:
            spec.points[tokens] = ep_backend.RankInputs(
                tokens_per_rank=tokens, topk_idx=_ValueTensor([0]),
                topk_weights=_ValueTensor([1.0]), activations=_ValueTensor([1.0]),
                global_idx=_ValueTensor([0]), global_weights=_ValueTensor([1.0]),
            )
        return spec

    def make_problem(self, T, idx, weights, x):
        return SimpleNamespace(T=T, x=x, dispatch_x=x, topk_idx=idx, topk_weights=weights)

    def create_buffer(self, spec):
        return None

    def warm(self, problem, count, stage_every=False):
        return None

    def benchmark_component(self, component, problem, warmup, iters):
        return [10.0] * iters

    def pair_series(self, kept):
        return [PAIR_US] * kept

    def benchmark_chain(self, problem, warmup, iters, drop):
        self.events.append(("chain", problem.T))
        kept = iters - drop
        pair = self.pair_series(kept)
        return {
            "pair": pair,
            "start_to_start": [value + GAP_US for value in pair[:-1]],
            "dispatch": [DISPATCH_FLOOR_US] * kept,
            "combine": [COMBINE_FLOOR_US] * kept,
            "combined": f"chained-{problem.T}",
        }

    def dispatch(self, problem):
        # Only the harness's drained reference pair reaches this: warm is a no-op, the
        # components are constants, and the oracle is mocked out in `_sweep`.
        self.events.append(("drained", problem.T))
        return SimpleNamespace(combine_input=None)

    def stage(self, problem, handle):
        return None

    def combine(self, problem, handle):
        return f"drained-{problem.T}"

    def recv_tokens(self, handle):
        return 8

    def inspect_dispatch(self, problem, handle):
        return {}

    def combine_transformed(self, problem, handle, transformed):
        return transformed


class _MisalignedSweepBackend(_SweepBackend):
    """Every roundtrip sample was entered by some rank after its spin ran out; nothing else was."""

    def benchmark_component(self, component, problem, warmup, iters):
        self._timed = (component, iters)
        return super().benchmark_component(component, problem, warmup, iters)

    def alignment_misses(self):
        component, iters = self._timed
        return [int(component == "roundtrip")] * iters


class _HostStalledSweepBackend(_SweepBackend):
    """A dispatch that waits on the host; `held` names the series whose every sample some rank held
    for its throttled host, `stalled` those whose every sample some rank's host ran throttled
    around anyway. Only roundtrip and dispatch carry a host-synced dispatch in the window."""

    host_synced_dispatch = True
    held = ()
    stalled = ("dispatch",)

    def benchmark_component(self, component, problem, warmup, iters):
        self._timed = (component, iters)
        return super().benchmark_component(component, problem, warmup, iters)

    def alignment_misses(self):
        return [0] * self._timed[1]

    def _flags(self, names):
        component, iters = self._timed
        if component not in ("roundtrip", "dispatch"):
            return None
        return [int(component in names)] * iters

    def host_holds(self):
        return self._flags(self.held)

    def host_stalls(self):
        return self._flags(self.stalled)


class _HeldSweepBackend(_HostStalledSweepBackend):
    held = ("roundtrip", "dispatch")
    stalled = ()


class _GraphSweepBackend(_SweepBackend):
    CUDA_GRAPH_MODES = ("normal",)

    def graph_replay_output(self, problem):
        self.events.append(("graph-value-check", problem.T))
        return f"graph-{problem.T}"

    def benchmark_component(self, component, problem, warmup, iters):
        self.events.append(("graph", component, problem.T))
        problem._cuda_graph_output = f"graph-{problem.T}"
        problem._cuda_graph_output_rewritten = True
        latency = {"roundtrip": 30.0, "dispatch": 11.0, "combine": 13.0}[component]
        return [latency] * iters


class _MissedGraphSweepBackend(_GraphSweepBackend):
    """Every roundtrip replay call outlasted the captured spin on some rank; nothing else did."""

    def benchmark_component(self, component, problem, warmup, iters):
        self._timed = (component, iters)
        return super().benchmark_component(component, problem, warmup, iters)

    def alignment_misses(self):
        component, iters = self._timed
        return [int(component == "roundtrip")] * iters


class _RestoringGraphSweepBackend(_GraphSweepBackend):
    def graph_replay_restore(self, problem):
        self.events.append(("graph-restore", problem.T))


class _ChainStalledSweepBackend(_SweepBackend):
    """A dispatch that waits on the host whose every chain trial restarted one sibling and was
    still throttled after the last restart."""

    host_synced_dispatch = True

    def benchmark_chain(self, problem, warmup, iters, drop):
        result = super().benchmark_chain(problem, warmup, iters, drop)
        result.update(host_retries=1, host_stalled=True)
        return result


class _StagedSweepBackend(_SweepBackend):
    """An eager backend whose stage launches device work, so it is timed apart."""

    def __init__(self):
        super().__init__()
        self.stage_device_work = True

    def benchmark_component(self, component, problem, warmup, iters):
        return [{"roundtrip": 30.0, "dispatch": 11.0, "combine": 13.0, "stage": 7.0}[component]] * iters


class _ReportingSweepBackend(_SweepBackend):
    def implementation_details(self):
        return {"num_sms": 64, "allow_multiple_reduction": False}


class _BrokenGraphSweepBackend(_GraphSweepBackend):
    def benchmark_component(self, component, problem, warmup, iters):
        samples = super().benchmark_component(component, problem, warmup, iters)
        problem._cuda_graph_output_rewritten = False
        return samples


def make_args(out):
    return SimpleNamespace(
        mode="normal", precision="bf16", phase="decode",
        tokens_ladder=" ".join(map(str, LADDER)),
        hidden=7168, topk=8, experts=256, routing="uniform",
        case_id="sku-stub-deepseek-v3-normal-decode-ep1-uniform-bf16",
        suite="ep-core", workload_name="deepseek-v3", seed=67, version=1,
        warmup=2, iters=4, trials=2,
        chain_iters=CHAIN_ITERS, chain_trials=CHAIN_TRIALS, chain_drop=CHAIN_DROP,
        runner="sku", topology_class="tc", transport="nvlink", scope="scale-up",
        scale_up_transport="nvlink", scale_out_transport="", gpus_per_node=1,
        scale_up_domain=1, out=str(out), runtime={}, image="", git_run=None,
    )


def phases_by_index(oracle_count, points):
    """Which pass each oracle call belongs to, by position: Pass 1 opens and Pass 3 closes with one
    per point, the middle is the chained gate."""
    return (
        ["pre"] * points
        + ["chain"] * (oracle_count - 2 * points)
        + ["post"] * points
    )


def _sweep(fail_indices, error_indices, chain_error, backend_factory=None,
           chain_output_ok=True):
    """One full run_sweep against the stubs; failures scripted by oracle call index."""
    backend = (backend_factory or _SweepBackend)()
    events = backend.events
    oracle_calls = []
    output_checks = []

    def fake_oracle(torch_, routing_, backend_, problem, *rest):
        index = len(oracle_calls)
        events.append(("oracle", problem.T))
        oracle_calls.append((index, problem.T, (problem, *rest)))
        passed = index not in fail_indices
        return ep_oracle.oracle_report(
            passed=passed,
            receive_count=8,
            max_elementwise_relative_error=chain_error if index in error_indices else 0.0,
            checks=dict.fromkeys(ep_oracle.ORACLE_CHECKS, passed),
        )

    def fake_output_match(chained, drained):
        output_checks.append((chained, drained))
        return chain_output_ok, 0.0 if chain_output_ok else 1.0

    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory) / "result.json"
        stdout = io.StringIO()
        with mock.patch.dict(sys.modules, {"routing": fake_routing()}), \
                mock.patch.dict(os.environ, {"COLLX_ATTEMPT_ID": "1"}), \
                mock.patch.object(ep_harness, "run_expert_oracle", fake_oracle), \
                mock.patch.object(ep_harness, "chain_output_matches", fake_output_match), \
                contextlib.redirect_stdout(stdout):
            rc = ep_harness.run_sweep(
                make_args(out), backend, value_torch(), _FakeDist(), "cuda:0", 0, 1
            )
        doc = json.loads(out.read_text())
    return SimpleNamespace(
        rc=rc, doc=doc, rows=doc["measurement"]["rows"], events=events,
        oracle_calls=oracle_calls, output_checks=output_checks,
        stdout=stdout.getvalue(), backend=backend,
        phases=phases_by_index(len(oracle_calls), len(LADDER)),
    )


def drive(*, fail_phases=(), chain_error=0.0, backend_factory=None, chain_output_ok=True):
    """Run the sweep, optionally failing every oracle of a given pass; a clean probe run first
    learns the oracle call count, so failures are selected by phase rather than by hardcoded index."""
    probe = _sweep(frozenset(), frozenset(), 0.0, backend_factory)
    if not fail_phases and not chain_error and chain_output_ok:
        return probe
    selected = lambda wanted: frozenset(  # noqa: E731
        index for index, phase in enumerate(probe.phases) if phase in wanted
    )
    return _sweep(
        selected(set(fail_phases)),
        selected({"chain"}) if chain_error else frozenset(), chain_error, backend_factory,
        chain_output_ok=chain_output_ok,
    )


class CombineModelGate(unittest.TestCase):
    def test_a_bad_reduction_from_create_buffer_fails_before_any_dispatch(self):
        # create_buffer may set the reduction (FlashInfer does, by wheel version), so the harness
        # resolves the model after it; the oracle would otherwise raise with a dispatch in flight.
        class _BadReduction(_SweepBackend):
            def create_buffer(self, spec):
                self.combine_reduction = "not-a-reduction"

        backend = _BadReduction()
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.dict(sys.modules, {"routing": fake_routing()}), \
                mock.patch.dict(os.environ, {"COLLX_ATTEMPT_ID": "1"}), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            rc = ep_harness.run_sweep(
                make_args(Path(directory) / "result.json"), backend, value_torch(), _FakeDist(),
                "cuda:0", 0, 1,
            )
        self.assertEqual(rc, 2)
        self.assertIn("not-a-reduction", out.getvalue())
        self.assertEqual(backend.events, [])


class ChainedRegimeOracleGate(unittest.TestCase):
    """The published regime has to be the gated one: Passes 1 and 3 only ever check drained calls,
    so a backend that corrupts only under free-running pairs would present as the suite's fastest."""

    def test_a_chain_output_mismatch_reds_the_case_on_its_own(self):
        # The two chained verdicts are independent: a chain whose own final output is wrong
        # reds the case even when the state it leaves behind still passes a fresh oracle --
        # exactly the shape a stale-parity/aliased-signal defect presents.
        run = drive(chain_output_ok=False)
        self.assertEqual(run.rc, 3)
        self.assertEqual(run.doc["outcome"]["status"], "invalid")
        for row in run.rows:
            with self.subTest(tokens=row["tokens_per_rank"]):
                self.assertIs(row["correctness"]["chain_last_output_passed"], False)
                self.assertIs(row["correctness"]["post_chain_state_passed"], True)
                self.assertIs(row["correctness"]["passed"], False)
                # The magnitude rides along so the verdict can be judged, not just believed.
                self.assertGreater(row["correctness"]["chain_last_output_error"], 0.0)

    def test_the_drained_oracles_still_red_the_case_on_their_own(self):
        # The chained gate is an addition, not a replacement.
        for phase in ("pre", "post"):
            with self.subTest(phase=phase):
                run = drive(fail_phases=(phase,))
                self.assertEqual(run.rc, 3)
                self.assertEqual(run.doc["outcome"]["status"], "invalid")
                for row in run.rows:
                    self.assertIs(row["correctness"]["passed"], False)
                    self.assertIs(row["correctness"]["post_chain_state_passed"], True)
                    self.assertIs(row["correctness"]["chain_last_output_passed"], True)
                    # The per-pass verdicts place the failure in the pass that produced it.
                    self.assertEqual(
                        row["correctness"]["oracle_passed"],
                        {"pre": phase != "pre", "chained": True, "post": phase != "post"},
                    )

    def test_the_output_check_is_skipped_where_staging_is_hoisted(self):
        # Under the hoist the chain captures one warm-up dispatch's staged stand-in and reuses
        # it for every pair, so neither the chain's final combine nor the drained reference
        # consumes an input matching its OWN dispatch. They are two differently mismatched
        # pairs and nothing requires them to agree, so the question is not asked.
        class _Hoisting(_SweepBackend):
            def __init__(self):
                super().__init__()
                self.precision, self.stage_device_work = "fp8", True

        backend = _Hoisting()
        self.assertTrue(backend.stage_excluded_from_roundtrip)
        run = drive(backend_factory=lambda: backend)
        self.assertEqual(run.rc, 0)
        self.assertEqual(run.output_checks, [], "the comparison must not run under the hoist")
        for row in run.rows:
            with self.subTest(tokens=row["tokens_per_rank"]):
                # null, not False: the artifact says "not asked", never something a reader
                # could mistake for a comparison that ran and failed.
                self.assertIsNone(row["correctness"]["chain_last_output_passed"])
                self.assertIsNone(row["correctness"]["chain_last_output_error"])
                self.assertIs(row["correctness"]["passed"], True)

    def test_the_chained_error_is_folded_into_max_relative_error(self):
        # Maxed in like the other two oracles', so a chained regime that is within tolerance
        # but worse than the drained one stays visible.
        run = drive(chain_error=0.25)
        self.assertEqual(run.rc, 0)
        for row in run.rows:
            with self.subTest(tokens=row["tokens_per_rank"]):
                self.assertAlmostEqual(row["correctness"]["max_relative_error"], 0.25)


class ChainedPublication(unittest.TestCase):
    """What a free-running chain actually emits: values, origins, counts and placement."""

    @classmethod
    def setUpClass(cls):
        cls.swept = drive()

    def test_every_published_chained_field(self):
        # One walk over the emitted rows covering the whole chained family: the period and its
        # origin, the per-op floors, and the three health scalars. Each carries its OWN origin
        # and sample count because they are reduced differently -- median for the period, MIN
        # for the floors, per-trial for the gap and drift -- and mixing those up is the defect
        # this pins.
        for row in self.swept.rows:
            with self.subTest(tokens=row["tokens_per_rank"]):
                period = row["components"]["pair_period"]
                self.assertEqual(period["percentiles_us"]["p50"], PAIR_US)
                self.assertIsNotNone(period["percentiles_us"]["p99"])
                self.assertEqual(period["origin"], "chained-median")
                self.assertEqual(period["availability"], "measured")
                self.assertEqual(period["sample_count"], 12)  # six kept pairs in each of two trials
                for op, expected in (
                    ("dispatch", DISPATCH_FLOOR_US), ("combine", COMBINE_FLOOR_US),
                ):
                    floor = row["chain_floor_us"][op]
                    self.assertEqual(floor["percentiles_us"]["p50"], expected)
                    self.assertEqual(floor["origin"], "chained-cross-rank-min")
                health = row["chain_health"]
                # One rank, so the ranks trivially agree; what matters is that the spread is
                # emitted and component-shaped, since a wide spread disqualifies a period.
                self.assertEqual(health["pair_spread_us"]["percentiles_us"]["p50"], 0.0)
                self.assertEqual(set(health["pair_spread_us"]), set(UNAVAILABLE))
                # start-to-start median minus pair-window median: the per-pair cost OUTSIDE the
                # published window, so instrumentation creeping back into the loop shows here.
                gap = health["interpair_gap_us"]
                self.assertEqual(gap["percentiles_us"]["p50"], GAP_US)
                self.assertEqual(gap["availability"], "measured")
                self.assertEqual(gap["sample_count"], CHAIN_TRIALS)
                drift = health["settle_drift_us"]
                self.assertEqual(drift["percentiles_us"]["p50"], 0.0)
                self.assertEqual(drift["sample_count"], CHAIN_TRIALS)
                # No dispatch here waits on the host, so the chain has no host counts to report.
                self.assertIsNone(health["host_retried_siblings"])
                self.assertIsNone(health["host_stalled_trials"])
                # Additive: the fresh-entry family keeps its meaning alongside the chain.
                self.assertEqual(row["components"]["roundtrip"]["origin"], "measured")
                self.assertEqual(row["components"]["roundtrip"]["percentiles_us"]["p50"], 10.0)
                self.assertEqual(row["components"]["dispatch"]["percentiles_us"]["p50"], 10.0)

    def test_the_doc_and_the_per_point_line_record_the_chain(self):
        self.assertIs(self.swept.doc["implementation"]["chained_period"], True)
        sampling = self.swept.doc["measurement"]["sampling"]
        self.assertEqual(sampling["chain_iterations_per_trial"], CHAIN_ITERS)
        self.assertEqual(sampling["chain_trials"], CHAIN_TRIALS)
        self.assertEqual(sampling["chain_drop"], CHAIN_DROP)
        self.assertIn("period=", self.swept.stdout)
        self.assertNotIn("period=n/a", self.swept.stdout)


class AlignmentMissPublication(unittest.TestCase):
    """Eager rows publish how many samples some rank entered after its spin ran out, and withhold
    the tails those samples could fill."""

    PERCENTILES = {"p50": 1.0, "p90": 2.0, "p95": 3.0, "p99": 4.0}

    def test_a_tail_is_withheld_once_missed_samples_could_fill_it(self):
        for fraction, expected in ((0.0, [1.0, 2.0, 3.0, 4.0]), (0.009, [1.0, 2.0, 3.0, 4.0]),
                                   (0.01, [1.0, 2.0, 3.0, None]), (0.05, [1.0, 2.0, None, None]),
                                   (0.1, [1.0, None, None, None]), (1.0, [1.0, None, None, None])):
            with self.subTest(fraction=fraction):
                published = ep_harness._published_tails(self.PERCENTILES, fraction)
                self.assertEqual(list(published.values()), expected)

    def test_a_clean_eager_row_keeps_its_tails_and_records_the_spin(self):
        swept = drive()
        self.assertEqual(swept.doc["implementation"]["rank_alignment"],
                         {"method": "gpu-spin-then-all-reduce", "spin_us": 2000.0})
        for row in swept.rows:
            with self.subTest(tokens=row["tokens_per_rank"]):
                self.assertEqual(row["rank_alignment"], {
                    "host_held_fraction": None,
                    "host_held_samples": None,
                    "host_stalled_fraction": None,
                    "host_stalled_samples": None,
                    "missed_fraction": 0.0,
                    "missed_samples": {"combine": 0, "dispatch": 0, "roundtrip": 0},
                })
                self.assertIsNotNone(row["components"]["roundtrip"]["percentiles_us"]["p99"])

    def test_missed_roundtrips_withhold_the_row_tails(self):
        swept = drive(backend_factory=_MisalignedSweepBackend)
        self.assertEqual(swept.rc, 0)
        for row in swept.rows:
            with self.subTest(tokens=row["tokens_per_rank"]):
                self.assertEqual(row["rank_alignment"], {
                    "host_held_fraction": None,
                    "host_held_samples": None,
                    "host_stalled_fraction": None,
                    "host_stalled_samples": None,
                    "missed_fraction": 1.0,
                    "missed_samples": {"combine": 0, "dispatch": 0, "roundtrip": 8},
                })
                for block in (row["components"]["roundtrip"], row["components"]["dispatch"],
                              row["cross_rank_min_us"]["roundtrip"]):
                    tails = block["percentiles_us"]
                    self.assertEqual(tails["p50"], 10.0)
                    self.assertEqual([tails[k] for k in ("p90", "p95", "p99")], [None] * 3)
                self.assertIsNone(row["token_rate_at_latency_percentile"]["p99"])
                # The chained family aligns once per sibling and keeps its tails.
                self.assertIsNotNone(row["components"]["pair_period"]["percentiles_us"]["p99"])

    def test_host_stalled_dispatches_withhold_the_row_tails(self):
        # A host throttled around a host-synced dispatch stretched the relaunch inside its window
        # on every rank, so those samples gate the tails the way missed ones do; misses stay apart.
        swept = drive(backend_factory=_HostStalledSweepBackend)
        self.assertEqual(swept.rc, 0)
        for row in swept.rows:
            with self.subTest(tokens=row["tokens_per_rank"]):
                self.assertEqual(row["rank_alignment"], {
                    "host_held_fraction": 0.0,
                    "host_held_samples": {"dispatch": 0, "roundtrip": 0},
                    "host_stalled_fraction": 1.0,
                    "host_stalled_samples": {"dispatch": 8, "roundtrip": 0},
                    "missed_fraction": 0.0,
                    "missed_samples": {"combine": 0, "dispatch": 0, "roundtrip": 0},
                })
                for block in (row["components"]["roundtrip"], row["components"]["dispatch"],
                              row["cross_rank_min_us"]["dispatch"]):
                    tails = block["percentiles_us"]
                    self.assertEqual(tails["p50"], 10.0)
                    self.assertEqual([tails[k] for k in ("p90", "p95", "p99")], [None] * 3)
                self.assertIsNone(row["token_rate_at_latency_percentile"]["p99"])
                self.assertIsNotNone(row["components"]["pair_period"]["percentiles_us"]["p99"])
        self.assertIn("missed=0.00% held=0.00% stalled=100.00% correct=True", swept.stdout)

    def test_held_samples_opened_at_speed_and_keep_the_row_tails(self):
        # A held sample waited for its host before the window opened, so it is counted but gates
        # nothing: every tail stays published.
        swept = drive(backend_factory=_HeldSweepBackend)
        self.assertEqual(swept.rc, 0)
        for row in swept.rows:
            with self.subTest(tokens=row["tokens_per_rank"]):
                self.assertEqual(row["rank_alignment"], {
                    "host_held_fraction": 1.0,
                    "host_held_samples": {"dispatch": 8, "roundtrip": 8},
                    "host_stalled_fraction": 0.0,
                    "host_stalled_samples": {"dispatch": 0, "roundtrip": 0},
                    "missed_fraction": 0.0,
                    "missed_samples": {"combine": 0, "dispatch": 0, "roundtrip": 0},
                })
                for block in (row["components"]["roundtrip"], row["components"]["dispatch"],
                              row["cross_rank_min_us"]["roundtrip"]):
                    self.assertIsNotNone(block["percentiles_us"]["p99"])
                self.assertIsNotNone(row["token_rate_at_latency_percentile"]["p99"])
        self.assertIn("missed=0.00% held=100.00% stalled=0.00% correct=True", swept.stdout)

    def test_missed_replays_withhold_the_graphed_row_tails(self):
        # A replay call that outlasted the spin may have opened its peers' windows before this
        # rank's dispatch was submitted; graphed tails are gated exactly as eager ones.
        swept = drive(backend_factory=_MissedGraphSweepBackend)
        self.assertEqual(swept.rc, 0)
        for row in swept.rows:
            with self.subTest(tokens=row["tokens_per_rank"]):
                self.assertEqual(row["rank_alignment"]["missed_samples"],
                                 {"combine": 0, "dispatch": 0, "roundtrip": 8})
                for block in (row["components"]["roundtrip"], row["components"]["dispatch"],
                              row["cross_rank_min_us"]["roundtrip"]):
                    tails = block["percentiles_us"]
                    self.assertIsNotNone(tails["p50"])
                    self.assertEqual([tails[k] for k in ("p90", "p95", "p99")], [None] * 3)
                self.assertIsNotNone(row["components"]["pair_period"]["percentiles_us"]["p99"])
        self.assertIn("missed=100.00%", swept.stdout)


class PublicationContract(unittest.TestCase):
    def test_the_graph_value_check_restores_the_inputs_after_its_drained_reference(self):
        swept = drive(backend_factory=_RestoringGraphSweepBackend)
        for T in LADDER:
            with self.subTest(T=T):
                mine = [event for event in swept.events
                        if event[0] in ("graph-value-check", "drained", "graph-restore")
                        and event[-1] == T]
                check = mine.index(("graph-value-check", T))
                self.assertEqual(mine[check:check + 3], [("graph-value-check", T), ("drained", T),
                                                         ("graph-restore", T)])

    def test_a_trial_still_stalled_after_its_restarts_keeps_only_the_period_median(self):
        swept = drive(backend_factory=_ChainStalledSweepBackend)
        self.assertEqual(swept.rc, 0)
        for row in swept.rows:
            with self.subTest(tokens=row["tokens_per_rank"]):
                period = row["components"]["pair_period"]["percentiles_us"]
                self.assertEqual(period["p50"], PAIR_US)
                self.assertEqual([period[k] for k in ("p90", "p95", "p99")], [None] * 3)
                health = row["chain_health"]
                self.assertEqual(health["host_retried_siblings"], CHAIN_TRIALS)
                self.assertEqual(health["host_stalled_trials"], CHAIN_TRIALS)

    def test_isolated_sum_adds_dispatch_and_combine_but_never_stage(self):
        swept = drive(backend_factory=_StagedSweepBackend)
        for row in swept.rows:
            with self.subTest(tokens=row["tokens_per_rank"]):
                self.assertEqual(row["components"]["stage"]["percentiles_us"]["p50"], 7.0)
                self.assertEqual(row["components"]["isolated_sum"]["percentiles_us"]["p50"], 24.0)

    def test_the_adapter_reports_its_realized_library_config(self):
        self.assertIsNone(drive().doc["implementation"]["library_config"])
        self.assertEqual(drive(backend_factory=_ReportingSweepBackend).doc["implementation"]
                         ["library_config"], {"num_sms": 64, "allow_multiple_reduction": False})

    def test_a_clamped_ladder_is_recorded_in_the_artifact_not_only_on_stdout(self):
        # A stdout NOTE alone leaves a document that measured two rungs indistinguishable from
        # one that was asked for three and capped.
        class _Clamped(_SweepBackend):
            def make_inputs(self, args):
                spec = super().make_inputs(args)
                spec.cap, spec.dropped = 8, [16]
                return spec

        swept = drive(backend_factory=_Clamped)
        workload = swept.doc["workload"]
        self.assertEqual(
            (workload["ladder_measured"], workload["ladder_dropped"], workload["ladder_cap"]),
            ([4, 8], [16], 8),
        )
        self.assertIn("NOTE: dropped tokens/rank [16]", swept.stdout)


class CudaGraphPublication(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.swept = drive(backend_factory=_GraphSweepBackend)

    def test_existing_component_fields_carry_graph_measurements(self):
        self.assertEqual(self.swept.rc, 0)
        self.assertIs(self.swept.doc["implementation"]["cuda_graph_replay"], True)
        self.assertEqual(self.swept.doc["implementation"]["rank_alignment"],
                         {"method": "in-graph-spin-then-all-reduce", "spin_us": 2000.0})
        # Graph mode keeps the chained family: the chain is captured, not dropped.
        self.assertIs(self.swept.doc["implementation"]["chained_period"], True)
        self.assertTrue(any(event[0] == "chain" for event in self.swept.events))
        for row in self.swept.rows:
            with self.subTest(tokens=row["tokens_per_rank"]):
                roundtrip = row["components"]["roundtrip"]
                self.assertEqual(roundtrip["percentiles_us"]["p50"], 30.0)
                self.assertEqual(roundtrip["origin"], "cuda-graph-replay")
                self.assertEqual(roundtrip["sample_count"], 8)
                self.assertEqual(row["components"]["dispatch"]["percentiles_us"]["p50"], 11.0)
                self.assertEqual(row["components"]["combine"]["percentiles_us"]["p50"], 13.0)
                self.assertEqual(row["components"]["dispatch"]["origin"], "cuda-graph-replay")
                self.assertEqual(row["components"]["combine"]["origin"], "cuda-graph-replay")
                self.assertEqual(row["components"]["isolated_sum"]["percentiles_us"]["p50"], 24.0)
                self.assertIsNone(row["components"]["stage"]["percentiles_us"])
                self.assertIsNotNone(row["components"]["pair_period"]["percentiles_us"])
                self.assertIs(row["correctness"]["cuda_graph_output_rewritten"], True)
                self.assertIs(row["correctness"]["cuda_graph_last_output_passed"], True)
                self.assertIs(row["correctness"]["post_chain_state_passed"], True)
                # No replay call outlasted the captured spin, so graphed rows count no suspect
                # samples and publish every tail, as the chained period does. Replay has no
                # host gate.
                for name in ("roundtrip", "dispatch", "combine", "isolated_sum"):
                    tails = row["components"][name]["percentiles_us"]
                    self.assertTrue(all(tails[k] is not None for k in ("p50", "p90", "p95", "p99")))
                self.assertIsNotNone(row["cross_rank_min_us"]["roundtrip"]["percentiles_us"]["p99"])
                self.assertIsNotNone(row["token_rate_at_latency_percentile"]["p99"])
                self.assertIsNotNone(row["components"]["pair_period"]["percentiles_us"]["p99"])
                self.assertEqual(row["rank_alignment"], {
                    "host_held_fraction": None,
                    "host_held_samples": None,
                    "host_stalled_fraction": None,
                    "host_stalled_samples": None,
                    "missed_fraction": 0.0,
                    "missed_samples": {"combine": 0, "dispatch": 0, "roundtrip": 0},
                })
        checked = [event[1] for event in self.swept.events if event[0] == "graph-value-check"]
        self.assertEqual(sorted(checked), sorted(LADDER))

    def test_a_replay_that_does_not_rewrite_its_output_reds_the_case(self):
        swept = drive(backend_factory=_BrokenGraphSweepBackend)
        self.assertEqual(swept.rc, 3)
        self.assertEqual(swept.doc["outcome"]["status"], "invalid")
        for row in swept.rows:
            self.assertIs(row["correctness"]["cuda_graph_output_rewritten"], False)
            self.assertIs(row["correctness"]["passed"], False)


class _DriftingBackend(_SweepBackend):
    """A chain whose late half runs DRIFT_US slower -- an unconverged (or down-clocking) run."""

    def pair_series(self, kept):
        half = kept // 2
        return [PAIR_US] * half + [PAIR_US + DRIFT_US] * (kept - half)


class SettleDrift(unittest.TestCase):
    """`chain_drop` assumes the chain settled by the time the kept iterations start, and nothing
    else in the artifact could show that it hadn't -- so a drifting chain publishes its drift."""

    def test_an_unconverged_chain_publishes_its_drift(self):
        run = drive(backend_factory=_DriftingBackend)
        # A health diagnostic, not a gate: the case stays green and the number says how much
        # to distrust the period.
        self.assertEqual(run.rc, 0)
        for row in run.rows:
            with self.subTest(tokens=row["tokens_per_rank"]):
                drift = row["chain_health"]["settle_drift_us"]
                self.assertEqual(drift["percentiles_us"]["p50"], DRIFT_US)
                self.assertEqual(drift["sample_count"], CHAIN_TRIALS)


class _Vec:
    """Just enough elementwise tensor for `_chain_output_matches`: a flat list of floats."""

    def __init__(self, values):
        self.values = [float(value) for value in values]

    @property
    def shape(self):
        return (len(self.values),)

    def numel(self):
        return len(self.values)

    def float(self):
        return _Vec(self.values)

    def abs(self):
        return _Vec(abs(value) for value in self.values)

    def clamp_min(self, floor):
        return _Vec(max(value, floor) for value in self.values)

    def __sub__(self, other):
        return _Vec(a - b for a, b in zip(self.values, other.values))

    def __truediv__(self, other):
        return _Vec(a / b for a, b in zip(self.values, other.values))

    def max(self):
        return SimpleNamespace(item=lambda: max(self.values))


class ChainOutputCheck(unittest.TestCase):
    """The regime A/B judged the way the oracle judges combines: elementwise relative error
    under a magnitude floor, because a combine kernel is not required to be order-deterministic
    across invocations -- bit equality would red healthy backends."""

    def test_the_verdict_and_the_magnitude_together(self):
        # The magnitude matters as much as the verdict: a bare bool cannot separate a transport
        # corruption from a tolerance too tight for an accumulator.
        for label, chained, drained, ok, expected_error in (
            ("identical", [1.0, -3.5, 0.25], [1.0, -3.5, 0.25], True, 0.0),
            # A rank that legitimately combined nothing under this routing.
            ("empty", [], [], True, 0.0),
            # Jitter inside tolerance passes and still reports its size.
            ("jitter", [1.01, -2.02], [1.0, -2.0], True, 0.01),
            ("corruption", [1.0, 2.0], [1.0, 4.0], False, 0.5),
            # Infinity, not a small number, so a cross-rank MAX sees a structural mismatch.
            ("shape", [1.0, 2.0], [1.0, 2.0, 3.0], False, float("inf")),
            # NaN would vanish from the cross-rank MAX and publish "failed, error 0.0".
            ("non-finite", [float("nan"), 1.0], [1.0, 1.0], False, float("inf")),
            # 0.0001 apart, judged under the 0.02 magnitude floor rather than against 1e-6.
            ("near-zero", [0.000101], [0.000001], True, 0.005),
        ):
            with self.subTest(label):
                got, error = ep_oracle.chain_output_matches(_Vec(chained), _Vec(drained))
                self.assertIs(got, ok)
                self.assertAlmostEqual(error, expected_error)


ROUNDTRIP = {"p50": 100.0, "p90": 110.0, "p95": 115.0, "p99": 120.0}
PERIOD = {"p50": 60.0, "p90": 66.0, "p95": 69.0, "p99": 72.0}


def document(with_period):
    components = {
        "roundtrip": {"percentiles_us": dict(ROUNDTRIP)},
    }
    if with_period:
        components["pair_period"] = {
            "percentiles_us": dict(PERIOD), "origin": "chained-median",
        }
    return {
        "version": 1,
        "outcome": {"status": "success"},
        "identity": {
            "case_factors": {
                "sku": "stub-sku",
                "case": {
                    "backend": "stub", "suite": "ep-core", "routing": "uniform",
                    "mode": "low-latency", "phase": "decode", "ep": 8,
                    "precision": "bf16",
                },
            },
        },
        "topology": {"gpus_per_node": 8, "scale_up_domain": 8, "nodes": 1},
        "measurement": {
            "rows": [{
                "tokens_per_rank": 64,
                "components": components,
                "logical_copies": {"wire": "per-assignment"},
                "cross_rank_min_us": {"roundtrip": {"percentiles_us": {"p50": 90.0}}},
                "cross_rank_spread_us": {"percentiles_us": {"p50": 5.0}},
            }],
        },
    }


class Headline(unittest.TestCase):
    def test_the_headline_is_the_pair_period_when_a_row_carries_one(self):
        tokens, p50, p99, _, _, carries = summarize._headline(document(with_period=True))
        self.assertEqual((tokens, p50, p99), (64, PERIOD["p50"], PERIOD["p99"]))
        self.assertTrue(carries)
        self.assertIn("chained pair period", summarize.render([document(with_period=True)]))

    def test_a_row_without_a_period_falls_back_to_the_roundtrip(self):
        _, p50, p99, _, _, carries = summarize._headline(document(with_period=False))
        self.assertEqual((p50, p99), (ROUNDTRIP["p50"], ROUNDTRIP["p99"]))
        self.assertFalse(carries)
        self.assertIn(
            "no row here carries a chained pair period",
            summarize.render([document(with_period=False)]),
        )


class RunnerFilterIsContentBased(unittest.TestCase):
    def test_runner_filters_on_the_declared_sku_not_the_filename(self):
        # Results are named by case_id, which joins its factors with "-", so a filename-prefix
        # filter would match nothing; and a filename claiming one SKU must not admit another's row.
        case = "stub-deepseek-v3-low-latency-decode-ep8-uniform-bf16_TS-c000.json"
        with tempfile.TemporaryDirectory() as directory:
            for name, sku in ((f"stub-sku-{case}", "stub-sku"), (f"other-sku-{case}", "other-sku"),
                              ("stub-sku_anything_TS-c000.json", "other-sku")):
                record = document(with_period=True)
                record["record_type"] = summarize.CASE_RECORD_TYPE  # what load_results keys on
                record["identity"]["case_factors"]["sku"] = sku
                (Path(directory) / name).write_text(json.dumps(record))
            kept = summarize.load_results(directory, "stub-sku")
        self.assertEqual([doc["identity"]["case_factors"]["sku"] for doc in kept], ["stub-sku"])


class GpuClockTelemetry(unittest.TestCase):
    def test_the_document_says_why_clocks_were_not_read(self):
        clocks = drive().doc["runtime"]["gpu_clocks"]
        self.assertIsNone(clocks["source"])
        self.assertIn("AttributeError", clocks["unavailable_reason"])
        self.assertIsNone(clocks["sm_clock_ratio_min"])

    def test_clock_reads_bracket_the_timed_passes_and_follow_every_burst(self):
        class _CountingBackend(_SweepBackend):
            def benchmark_component(self, component, problem, warmup, iters):
                self.events.append(("component", problem.T))
                return super().benchmark_component(component, problem, warmup, iters)

        backends = []

        def factory():
            backends.append(_CountingBackend())
            return backends[-1]

        class _Monitor:
            def __init__(self, torch, device):
                self.reason = None

            def start(self):
                backends[-1].events.append(("clock-start",))

            def sample(self, temperature=False):
                backends[-1].events.append(("clock", "chain" if temperature else "point"))

            def finish(self):
                backends[-1].events.append(("clock-finish",))

            def vector(self):
                return [1.0 if field == "source" else 40.0 if field == "sm_clock_samples"
                        else 1980.0 if field.startswith("sm_clock") else math.nan
                        for field in gpu_clocks.FIELDS]

        with mock.patch.object(gpu_clocks, "ClockMonitor", _Monitor):
            run = drive(backend_factory=factory)
        events = [event[0] if event[0] != "clock" else event for event in run.events]
        timed = [i for i, event in enumerate(events) if event in ("component", "chain")]
        self.assertLess(events.index("clock-start"), timed[0])
        self.assertGreater(events.index("clock-finish"), timed[-1])
        # One read per (trial, point) after its components, one after every chain.
        self.assertEqual(events.count(("clock", "point")), 2 * len(LADDER))
        self.assertEqual(events.count(("clock", "chain")), CHAIN_TRIALS * len(LADDER))
        for i, event in enumerate(events):
            if event == ("clock", "chain"):
                self.assertEqual(events[i - 1], "chain")
            if event == ("clock", "point"):
                self.assertEqual(events[i - 1], "component")
        self.assertEqual(run.doc["runtime"]["gpu_clocks"]["per_rank"][0]["sm_clock_mhz"]["p50"], 1980)
        self.assertEqual(run.doc["runtime"]["gpu_clocks"]["sm_clock_ratio_min"], 1.0)


if __name__ == "__main__":
    unittest.main()
