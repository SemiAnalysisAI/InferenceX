#!/usr/bin/env python3
"""Contract for the chained steady-state pair period.

`roundtrip` drains the GPU around every pair, so it reports the latency of an idle pipeline
and charges inter-rank entry stagger to whichever op the ranks happened to enter unevenly. A
decode loop never stops between layers, so what it actually pays per layer is the free-running
PERIOD of the dispatch->combine chain. Measuring that means issuing pairs back-to-back with no
host sync inside the loop, which is exactly what is delicate about it: these tests pin the call
order, that a hoisted stage stays out of every pair window, that the discarded iterations are
the head (pipeline fill) and not the tail, and that the opt-in re-alignment barrier lands
BETWEEN pairs and never between a dispatch and its own combine.

Torch-free: a stub `torch` supplies CUDA events whose elapsed_time reads a clock the fake
backend advances by a fixed cost per operation, so every reported window has an exact expected
value and a window that brackets the wrong work is a numeric mismatch, not a judgement call.
"""
from __future__ import annotations

import contextlib
import io
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "bench")]

import ep_backend  # noqa: E402
import ep_harness  # noqa: E402

# Per-operation device cost in the stub clock (ms). Distinct primes so that any window
# reports a sum unique to the operations it actually brackets.
DISPATCH_MS = 3.0
STAGE_MS = 7.0
COMBINE_MS = 5.0


class _Clock:
    """Stub device clock; only the fake backend's operations advance it."""

    def __init__(self):
        self.now_ms = 0.0

    def advance(self, ms):
        self.now_ms += ms


class _FakeEvent:
    """torch.cuda.Event stand-in: records the clock, reports ms between two records."""

    def __init__(self, clock):
        self._clock = clock
        self.t = None

    def record(self, *_args, **_kwargs):
        self.t = self._clock.now_ms

    def elapsed_time(self, other):
        if self.t is None or other.t is None:
            raise AssertionError("elapsed_time on an event that was never recorded")
        return other.t - self.t

    def synchronize(self, *_args, **_kwargs):
        pass

    def query(self):
        return True


class _FakeTensor:
    """Absorbs whatever a barrier payload is asked to do."""

    def __getattr__(self, _name):
        return lambda *args, **kwargs: self


@contextlib.contextmanager
def fake_torch(clock, log):
    """Install a stub `torch`/`torch.distributed` that logs the calls this contract is about."""
    tensor = lambda *args, **kwargs: _FakeTensor()  # noqa: E731
    dist = types.SimpleNamespace(
        all_reduce=lambda *args, **kwargs: log.append("all_reduce"),
        barrier=lambda *args, **kwargs: log.append("dist_barrier"),
        is_initialized=lambda: True,
        get_rank=lambda *args, **kwargs: 0,
        get_world_size=lambda *args, **kwargs: 2,
        ReduceOp=types.SimpleNamespace(SUM="sum", MAX="max", MIN="min"),
    )
    torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            Event=lambda *args, **kwargs: _FakeEvent(clock),
            synchronize=lambda *args, **kwargs: log.append("sync"),
            current_stream=lambda *args, **kwargs: types.SimpleNamespace(
                synchronize=lambda: log.append("sync")
            ),
        ),
        distributed=dist,
        zeros=tensor, ones=tensor, empty=tensor, full=tensor, tensor=tensor,
        float32="float32", float64="float64", bfloat16="bfloat16", int32="int32",
    )
    with mock.patch.dict(sys.modules, {"torch": torch, "torch.distributed": dist}):
        yield torch


class _ChainBackend(ep_backend.EPBackend):
    """Records the call order and charges each operation a fixed slice of the stub clock."""

    name = "chain-stub"

    def __init__(self, stage_device_work=True, fp8_consume="native", precision="fp8",
                 dispatch_schedule=None):
        self.calls: list[str] = []
        self.clock = _Clock()
        self.stage_device_work = stage_device_work
        self.fp8_consume = fp8_consume
        self.precision = precision
        self.device = "cpu"
        self.rank = 0
        self.world_size = 2
        # Per-dispatch cost overrides, consumed in order; the constant cost applies after.
        self._dispatch_schedule = list(dispatch_schedule or [])

    def create_buffer(self, spec):  # pragma: no cover - unused
        raise NotImplementedError

    def dispatch(self, problem):
        self.calls.append("dispatch")
        cost = self._dispatch_schedule.pop(0) if self._dispatch_schedule else DISPATCH_MS
        self.clock.advance(cost)
        return types.SimpleNamespace(combine_input=None)

    def stage(self, problem, handle):
        self.calls.append("stage")
        self.clock.advance(STAGE_MS)
        handle.combine_input = "staged-by-stage"

    def combine(self, problem, handle):
        self.calls.append("combine")
        self.clock.advance(COMBINE_MS)
        return handle.combine_input

    def recv_tokens(self, handle):
        return 0

    def inspect_dispatch(self, problem, handle):  # pragma: no cover - unused
        return {}

    def combine_transformed(self, problem, handle, transformed):  # pragma: no cover
        return transformed


def new_problem():
    """A problem the backend can hang cached state on -- `warm` caches recv_tokens there."""
    return types.SimpleNamespace()


def timed_tail(calls, iters, per_pair):
    """The chain loop's own calls: the last `per_pair * iters` entries, trailing syncs removed.

    Everything before them is warm-up and the untimed staged hoist, which have their own
    contracts elsewhere.
    """
    trace = list(calls)
    while trace and trace[-1] in ("sync", "all_reduce", "dist_barrier"):
        trace.pop()
    return trace[-per_pair * iters:]


def all_reduce_splits_a_pair(calls):
    """True if any all_reduce lands between a dispatch and the combine that closes it."""
    inside = False
    for entry in calls:
        if entry == "dispatch":
            inside = True
        elif entry == "combine":
            inside = False
        elif entry == "all_reduce" and inside:
            return True
    return False


class ChainedPairPeriod(unittest.TestCase):
    def test_warms_once_before_the_chain_and_never_inside_it(self):
        backend = _ChainBackend()
        problem = new_problem()
        with fake_torch(backend.clock, backend.calls), mock.patch.object(
            backend, "warm", wraps=backend.warm
        ) as warm:
            backend.benchmark_chain(problem, 4, 6, 2)
        warm.assert_called_once()
        self.assertIs(warm.call_args.args[0], problem)
        self.assertEqual(warm.call_args.args[1], 4)

    def test_the_loop_is_dispatch_combine_pairs_with_no_sync_between_them(self):
        # The whole point of the chain: iterations overlap. A host sync inside the loop drains
        # the GPU and turns the period back into a sequence of drained roundtrips.
        iters = 6
        backend = _ChainBackend()
        with fake_torch(backend.clock, backend.calls):
            backend.benchmark_chain(new_problem(), 0, iters, 2)
        self.assertEqual(
            timed_tail(backend.calls, iters, 2), ["dispatch", "combine"] * iters
        )

    def test_a_hoisted_stage_runs_once_and_stays_out_of_every_pair_window(self):
        # Same hoist benchmark_roundtrip performs: with a real device copy in `stage`, the
        # conversion is materialised once, untimed, so the chained pair is dispatch -> combine.
        for precision in ("bf16", "fp8"):
            with self.subTest(precision=precision):
                iters = 6
                backend = _ChainBackend(
                    stage_device_work=True, fp8_consume="native", precision=precision
                )
                self.assertTrue(backend.stage_excluded_from_roundtrip)
                with fake_torch(backend.clock, backend.calls):
                    pair, dispatch, combine = backend.benchmark_chain(new_problem(), 0, iters, 2)
                self.assertEqual(backend.calls.count("stage"), 1)
                self.assertEqual(
                    timed_tail(backend.calls, iters, 2), ["dispatch", "combine"] * iters
                )
                # The staged cost is absent from the pair window, not merely from the trace.
                for value in pair:
                    self.assertAlmostEqual(value, (DISPATCH_MS + COMBINE_MS) * 1000.0)
                for value in dispatch:
                    self.assertAlmostEqual(value, DISPATCH_MS * 1000.0)
                for value in combine:
                    self.assertAlmostEqual(value, COMBINE_MS * 1000.0)

    def test_the_dequant_hatch_stages_inside_every_pair(self):
        # CX_FP8_CONSUME=dequant models a stack that really does convert between the two
        # collectives, so the chain must carry that conversion on every iteration.
        iters = 5
        backend = _ChainBackend(
            stage_device_work=True, fp8_consume="dequant", precision="fp8"
        )
        self.assertFalse(backend.stage_excluded_from_roundtrip)
        with fake_torch(backend.clock, backend.calls):
            pair, _, _ = backend.benchmark_chain(new_problem(), 0, iters, 1)
        self.assertEqual(backend.calls.count("stage"), iters)
        self.assertEqual(
            timed_tail(backend.calls, iters, 3), ["dispatch", "stage", "combine"] * iters
        )
        for value in pair:
            self.assertAlmostEqual(value, (DISPATCH_MS + STAGE_MS + COMBINE_MS) * 1000.0)

    def test_a_no_op_stage_stays_inline_and_is_never_hoisted(self):
        # deepep-v2 / uccl-ep / nccl-ep at BF16: `stage` is a pointer assignment, and hoisting
        # it would hand a low-latency backend a view into its double-buffered receive.
        iters = 4
        backend = _ChainBackend(
            stage_device_work=False, fp8_consume="native", precision="bf16"
        )
        self.assertFalse(backend.stage_excluded_from_roundtrip)
        with fake_torch(backend.clock, backend.calls):
            backend.benchmark_chain(new_problem(), 0, iters, 1)
        self.assertEqual(backend.calls.count("stage"), iters)
        self.assertEqual(
            timed_tail(backend.calls, iters, 3), ["dispatch", "stage", "combine"] * iters
        )

    def test_the_pair_itself_satisfies_the_backends_that_need_a_paired_call(self):
        # combine_needs_redispatch (a combine consumes its dispatch) and
        # dispatch_needs_combine_cleanup (a dispatch must be drained) are both satisfied by
        # the chain's structure, so neither may inject an extra untimed call into the loop.
        iters = 5
        backend = _ChainBackend(
            stage_device_work=False, fp8_consume="native", precision="bf16"
        )
        backend.combine_needs_redispatch = True
        backend.dispatch_needs_combine_cleanup = True
        with fake_torch(backend.clock, backend.calls):
            backend.benchmark_chain(new_problem(), 0, iters, 1)
        self.assertEqual(
            timed_tail(backend.calls, iters, 3), ["dispatch", "stage", "combine"] * iters
        )

    def test_returns_one_sample_per_kept_iteration(self):
        for iters, drop in ((8, 0), (8, 2), (6, 5)):
            with self.subTest(iters=iters, drop=drop):
                backend = _ChainBackend()
                with fake_torch(backend.clock, backend.calls):
                    series = backend.benchmark_chain(new_problem(), 0, iters, drop)
                self.assertEqual(len(series), 3)
                for samples in series:
                    self.assertEqual(len(samples), iters - drop)

    def test_the_dropped_iterations_are_the_head_of_the_chain(self):
        # `drop` exists to discard pipeline fill, so it must cut the head. A tail cut would
        # keep exactly the unfilled iterations it is meant to remove.
        iters, drop = 6, 2
        backend = _ChainBackend(
            stage_device_work=False, fp8_consume="native", precision="bf16",
            dispatch_schedule=[50.0] * drop + [DISPATCH_MS] * (iters - drop),
        )
        with fake_torch(backend.clock, backend.calls):
            _, dispatch, _ = backend.benchmark_chain(new_problem(), 0, iters, drop)
        self.assertEqual(len(dispatch), iters - drop)
        for value in dispatch:
            self.assertAlmostEqual(value, DISPATCH_MS * 1000.0)


class ChainBarrier(unittest.TestCase):
    """The escape valve for a backend that wedges when pairs are free-running.

    It re-aligns the ranks BETWEEN pairs. Between a dispatch and its own combine it would do
    the opposite of its job: that is the window whose overlap the period is measuring, and a
    collective dropped into it serialises the very thing under test.
    """

    def test_off_by_default_so_no_backend_pays_for_a_barrier_it_did_not_ask_for(self):
        self.assertFalse(ep_backend.EPBackend.chain_barrier)

    def test_no_barrier_is_issued_while_the_flag_is_off(self):
        backend = _ChainBackend()
        with fake_torch(backend.clock, backend.calls):
            backend.benchmark_chain(new_problem(), 0, 5, 1)
        self.assertNotIn("all_reduce", backend.calls)

    def test_the_barrier_runs_once_between_pairs(self):
        iters = 6
        backend = _ChainBackend()
        backend.chain_barrier = True
        with fake_torch(backend.clock, backend.calls):
            backend.benchmark_chain(new_problem(), 0, iters, 2)
        # Between pairs is iters-1 gaps; one per iteration is equally sound.
        self.assertIn(backend.calls.count("all_reduce"), (iters - 1, iters))

    def test_the_barrier_never_splits_a_dispatch_from_its_combine(self):
        backend = _ChainBackend()
        backend.chain_barrier = True
        with fake_torch(backend.clock, backend.calls):
            backend.benchmark_chain(new_problem(), 0, 6, 2)
        self.assertFalse(
            all_reduce_splits_a_pair(backend.calls),
            f"barrier landed inside a pair: {backend.calls}",
        )

    def test_the_barrier_does_not_disturb_the_pair_sequence(self):
        iters = 6
        backend = _ChainBackend()
        backend.chain_barrier = True
        with fake_torch(backend.clock, backend.calls):
            backend.benchmark_chain(new_problem(), 0, iters, 2)
        pairs = [entry for entry in backend.calls if entry in ("dispatch", "combine")]
        self.assertEqual(pairs[-2 * iters:], ["dispatch", "combine"] * iters)


class ChainBudgetGate(unittest.TestCase):
    """A chain budget that could publish nothing must stop the leg before it measures.

    `pair_period` reduced from zero kept pairs serialises as merely "unavailable", which reads
    identically to a backend that could not be chained at all -- so a drop at or past the
    iteration count would spend the whole leg and then emit a row nobody can interpret. The
    gate sits above run_sweep's lazy torch imports, which is what makes it reachable here.
    """

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

    def test_an_unusable_chain_budget_fails_closed(self):
        for label, updates in (
            ("no iterations", dict(chain_iters=0)),
            ("no trials", dict(chain_trials=0)),
            ("drop swallows every pair", dict(chain_iters=8, chain_drop=8)),
            ("drop exceeds the chain", dict(chain_iters=8, chain_drop=9)),
            ("negative drop", dict(chain_drop=-1)),
        ):
            with self.subTest(budget=label):
                rc, output = self._gate(**updates)
                self.assertEqual(rc, 2)
                self.assertIn("chain", output)

    def test_the_chain_gate_did_not_displace_the_fresh_entry_one(self):
        # Both budgets are still checked, and each names its own fields, so an operator reading
        # the failure knows which profile field to fix.
        rc, output = self._gate(iters=0)
        self.assertEqual(rc, 2)
        self.assertIn("iters/trials/warmup", output)


class ChainComponentContract(unittest.TestCase):
    """The two chain contracts a driven sweep does not show.

    Field names, origins and placement are asserted against a real emitted row in
    test_run_sweep_chain.py, which is strictly stronger than reading them out of the source, so
    the AST guards that used to live here are gone. What is left is what that run does not
    exercise: the constants a consumer imports by name, and `_component`'s behaviour on the two
    paths no chain row takes.
    """

    def test_the_origin_constants_carry_the_published_values(self):
        self.assertEqual(ep_harness.CHAIN_PERIOD_ORIGIN, "chained-median")
        self.assertEqual(ep_harness.CHAIN_FLOOR_ORIGIN, "chained-cross-rank-min")

    def test_an_overridden_origin_leaves_the_rest_of_the_component_alone(self):
        # The chain families reach the artifact through `_component`'s `origin` override, which
        # every pre-chain row also flows through. Omitting it must reproduce the old strings
        # exactly, or the chain silently reclassifies rows that have nothing to do with it.
        percentiles = {"p50": 1.0, "p90": 2.0, "p95": 3.0, "p99": 4.0}
        self.assertEqual(ep_harness._component(percentiles, 3)["origin"], "measured")
        self.assertEqual(
            ep_harness._component(percentiles, 0, derived=True)["origin"],
            "derived-percentile-sum",
        )
        self.assertIsNone(ep_harness._component(None, 0)["origin"])

        overridden = ep_harness._component(percentiles, 3, origin="chained-median")
        self.assertEqual(overridden["origin"], "chained-median")
        # Only the origin moves: availability, the percentiles and the count are untouched.
        self.assertEqual(overridden["availability"], "measured")
        self.assertEqual(overridden["percentiles_us"], percentiles)
        self.assertEqual(overridden["sample_count"], 3)


if __name__ == "__main__":
    unittest.main()
