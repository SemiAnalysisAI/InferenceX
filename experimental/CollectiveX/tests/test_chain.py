#!/usr/bin/env python3
"""The chained pair period end to end: the timing primitive, its publication through run_sweep, and the headline it feeds.

Two torch doubles live here on purpose. The trace_* family logs record()/sync()/all_reduce against a controllable clock, so event PLACEMENT is assertable; the value_* family carries real tensor arithmetic, so the published NUMBERS are. Merging them would be more complex than either."""
from __future__ import annotations

import contextlib
import io
import sys
import types
import unittest
from pathlib import Path
from unittest import mock
import copy
import json
import os
import statistics
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "bench")]
sys.path[:0] = [str(ROOT)]

import ep_backend  # noqa: E402
import ep_harness  # noqa: E402
import summarize  # noqa: E402


# ---- from test_chain_period.py ----------------------------------------------------
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

    def synchronize(self, *_args, **_kwargs):
        pass

    def query(self):
        return True


class _TraceTensor:
    """Absorbs whatever a tensor is asked to do."""

    def __getattr__(self, _name):
        return lambda *args, **kwargs: self


@contextlib.contextmanager
def trace_torch(clock, log):
    """Install a stub `torch`/`torch.distributed` that logs the calls this contract is about."""
    tensor = lambda *args, **kwargs: _TraceTensor()  # noqa: E731
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
            Event=lambda *args, **kwargs: _TraceEvent(clock, log),
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


class _Combined:
    """Just enough combined-output tensor for the chain's final-output capture."""

    def __init__(self, value):
        self.value = value
        self.cloned = False

    def clone(self):
        detached = _Combined(self.value)
        detached.cloned = True
        return detached


class _ChainBackend(ep_backend.EPBackend):
    """Records the call order and charges each operation a fixed slice of the stub clock."""

    name = "chain-stub"

    def __init__(self, stage_device_work=True, fp8_consume="native", precision="fp8",
                 dispatch_schedule=None):
        self.calls: list[str] = []
        self.consumed: list = []
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
        self.consumed.append(handle.combine_input)
        self.clock.advance(COMBINE_MS)
        return _Combined(handle.combine_input)

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
        # the very stagger the chain exists to amortise. Neither may appear.
        iters = 6
        backend = _ChainBackend()
        with trace_torch(backend.clock, backend.calls):
            backend.benchmark_chain(new_problem(), 0, iters, 2)
        self.assertEqual(
            timed_tail(backend.calls, iters, 2), ["dispatch", "combine"] * iters
        )
        self.assertNotIn("all_reduce", backend.calls)
        self.assertNotIn("dist_barrier", backend.calls)

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
                with trace_torch(backend.clock, backend.calls):
                    series = backend.benchmark_chain(new_problem(), 0, iters, 2)
                self.assertEqual(backend.calls.count("stage"), 1)
                self.assertEqual(backend.consumed, ["staged-by-stage"] * (2 * iters + 1))
                floors, period = chain_sections(backend.calls)
                self.assertEqual(ops_only(floors), ["dispatch", "combine"] * iters)
                self.assertEqual(ops_only(period), ["dispatch", "combine"] * iters)
                # The staged cost is absent from the pair window, not merely from the trace.
                for value in series["pair"]:
                    self.assertAlmostEqual(value, (DISPATCH_MS + COMBINE_MS) * 1000.0)
                for value in series["start_to_start"]:
                    self.assertAlmostEqual(value, (DISPATCH_MS + COMBINE_MS) * 1000.0)
                for value in series["dispatch"]:
                    self.assertAlmostEqual(value, DISPATCH_MS * 1000.0)
                for value in series["combine"]:
                    self.assertAlmostEqual(value, COMBINE_MS * 1000.0)

    def test_returns_one_sample_per_kept_iteration(self):
        for iters, drop in ((8, 0), (8, 2), (6, 5)):
            with self.subTest(iters=iters, drop=drop):
                backend = _ChainBackend()
                with trace_torch(backend.clock, backend.calls):
                    series = backend.benchmark_chain(new_problem(), 0, iters, drop)
                self.assertEqual(
                    sorted(series),
                    ["combine", "combined", "dispatch", "pair", "start_to_start"],
                )
                for key in ("pair", "dispatch", "combine"):
                    self.assertEqual(len(series[key]), iters - drop)
                # Start-to-start is a difference series: one fewer than the kept pairs.
                self.assertEqual(len(series["start_to_start"]), max(iters - drop - 1, 0))

    def test_the_dropped_iterations_are_the_head_of_each_chain(self):
        # `drop` discards pipeline fill, so it must cut the head of both chains -- the period
        # chain refills after the inter-chain synchronize.
        iters, drop = 6, 2
        slow_head = [50.0] * drop + [DISPATCH_MS] * (iters - drop)
        backend = _ChainBackend(
            stage_device_work=False, fp8_consume="native", precision="bf16",
            dispatch_schedule=slow_head * 2,  # floors chain runs first, then the period chain
        )
        with trace_torch(backend.clock, backend.calls):
            series = backend.benchmark_chain(new_problem(), 0, iters, drop)
        self.assertEqual(len(series["dispatch"]), iters - drop)
        for value in series["dispatch"]:
            self.assertAlmostEqual(value, DISPATCH_MS * 1000.0)
        for value in series["pair"]:
            self.assertAlmostEqual(
                value, (DISPATCH_MS + STAGE_MS + COMBINE_MS) * 1000.0
            )


class EventPlacement(unittest.TestCase):
    """Which events each sibling chain may carry. The stub charges host work nothing, so these
    assert record placement in the trace rather than window values."""

    def _sections(self, **backend_kwargs):
        iters = 4
        backend = _ChainBackend(**backend_kwargs)
        with trace_torch(backend.clock, backend.calls):
            backend.benchmark_chain(new_problem(), 0, iters, 1)
        floors, period = chain_sections(backend.calls)
        return iters, floors, period

    def test_the_period_pairs_carry_only_the_outer_events(self):
        # One record before the dispatch, one after the combine, nothing between: both records'
        # host cost lands in the inter-pair gap, outside the published window.
        iters, _, period = self._sections(
            stage_device_work=False, fp8_consume="native", precision="bf16"
        )
        self.assertEqual(
            period,
            ["record", "dispatch", "stage", "combine", "record"] * iters,
        )

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

    def test_an_unusable_chain_budget_fails_closed(self):
        for label, updates in (
            ("no iterations", dict(chain_iters=0)),
            ("no trials", dict(chain_trials=0)),
            ("drop swallows every pair", dict(chain_iters=8, chain_drop=8)),
            ("drop exceeds the chain", dict(chain_iters=8, chain_drop=9)),
            # A single kept pair has an empty start-to-start series: interpair gap and settle
            # drift would silently publish as "unavailable" health for a measured period.
            ("drop leaves a single pair", dict(chain_iters=8, chain_drop=7)),
            ("negative drop", dict(chain_drop=-1)),
        ):
            with self.subTest(budget=label):
                rc, output = self._gate(**updates)
                self.assertEqual(rc, 2)
                self.assertIn("chain", output)

    def test_the_chain_gate_did_not_displace_the_fresh_entry_one(self):
        # Both budgets are checked and each names its own fields, so the failure says which
        # profile field to fix.
        rc, output = self._gate(iters=0)
        self.assertEqual(rc, 2)
        self.assertIn("iters/trials/warmup", output)


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


# ---- from test_run_sweep_chain.py -------------------------------------------------
LADDER = [4, 8]
CHAIN_ITERS, CHAIN_DROP, CHAIN_TRIALS = 8, 2, 2
KEPT_PER_TRIAL = CHAIN_ITERS - CHAIN_DROP
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

    def __iter__(self):
        return iter(self.data)

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

    @property
    def shape(self):
        return (len(self.data),)


class _ValueEvent:
    clock = [0.0]

    def __init__(self, enable_timing=False):
        self.t = None

    def record(self):
        _ValueEvent.clock[0] += 1.0
        self.t = _ValueEvent.clock[0]

    def elapsed_time(self, other):
        return (other.t - self.t) / 1000.0


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
    torch.float64, torch.int64, torch.bfloat16 = "f64", "i64", "bf16"
    torch.cuda = SimpleNamespace(synchronize=lambda: None, Event=_ValueEvent)
    torch.tensor = lambda values, device=None, dtype=None: _ValueTensor(list(values))
    torch.empty_like = lambda x: _ValueTensor(list(x.data))
    torch.stack = lambda xs: _ValueTensor([x.data for x in xs])
    torch.equal = lambda a, b: a.data == b.data
    torch.zeros = lambda n, device=None: _ValueTensor([0.0] * n)
    torch.distributed = SimpleNamespace(all_reduce=lambda x: None)
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

    def benchmark_chain(self, problem, warmup, iters, drop):
        self.events.append(("chain", problem.T))
        kept = iters - drop
        return {
            "pair": [PAIR_US] * kept,
            "start_to_start": [PAIR_US + GAP_US] * (kept - 1),
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
        return ep_harness._oracle_report(
            passed=passed,
            receive_count=8,
            max_elementwise_relative_error=chain_error if index in error_indices else 0.0,
            checks=dict.fromkeys(ep_harness._ORACLE_CHECKS, passed),
        )

    def fake_output_match(chained, drained):
        output_checks.append((chained, drained))
        return chain_output_ok, 0.0 if chain_output_ok else 1.0

    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory) / "result.json"
        stdout = io.StringIO()
        with mock.patch.dict(sys.modules, {"routing": fake_routing()}), \
                mock.patch.dict(os.environ, {"COLLX_ATTEMPT_ID": "1"}), \
                mock.patch.object(ep_harness, "_run_expert_oracle", fake_oracle), \
                mock.patch.object(ep_harness, "_chain_output_matches", fake_output_match), \
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


class ChainedRegimeOracleGate(unittest.TestCase):
    """The published regime has to be the gated one: Passes 1 and 3 only ever check drained calls,
    so a backend that corrupts only under free-running pairs would present as the suite's fastest."""

    def test_a_chain_output_mismatch_reds_the_case_on_its_own(self):
        # The two chained verdicts are independent: a chain whose own final output is wrong
        # reds the case even when the state it leaves behind still passes a fresh oracle --
        # exactly the shape a stale-parity/aliased-signal defect presents.
        #
        # This check was briefly demoted to reporting-only (2026-08-07) on the theory that its
        # tolerance was too tight for FP8. Probe 31180411148 measured the magnitude and
        # falsified that: bf16 differs by exactly 0.0, FP8 by 1000x-2966x the tolerance. Re-armed
        # on that number. If it is ever demoted again, demote it on a MEASURED magnitude too --
        # the verdict alone is what justified the wrong call both times.
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

    def test_the_output_check_is_skipped_where_staging_is_hoisted(self):
        # Under the hoist the chain captures one warm-up dispatch's staged stand-in and reuses
        # it for every pair, so neither the chain's final combine nor the drained reference
        # consumes an input matching its OWN dispatch. They are two differently mismatched
        # pairs and nothing requires them to agree, so the question is not asked.
        # Measured, h100/deepep-v2/EP8, identical but for the hoist: hoisted gives
        # chain_last_output_error 31..93 (1000x-2966x tolerance), per-pair staging gives 0.0 at
        # every rung, in both normal and low-latency mode. Gating on it under the hoist
        # reddened every FP8 leg fleet-wide for a harness artifact.
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
                self.assertEqual(period["origin"], "chained-median")
                self.assertEqual(period["availability"], "measured")
                self.assertEqual(period["sample_count"], KEPT_PER_TRIAL * CHAIN_TRIALS)
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


class _DriftingBackend(_SweepBackend):
    """A chain whose late half runs DRIFT_US slower -- an unconverged (or down-clocking) run."""

    def benchmark_chain(self, problem, warmup, iters, drop):
        self.events.append(("chain", problem.T))
        kept = iters - drop
        half = kept // 2
        pair = [PAIR_US] * half + [PAIR_US + DRIFT_US] * (kept - half)
        return {
            "pair": pair,
            "start_to_start": [value + GAP_US for value in pair[:-1]],
            "dispatch": [DISPATCH_FLOOR_US] * kept,
            "combine": [COMBINE_FLOOR_US] * kept,
            "combined": f"chained-{problem.T}",
        }


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
        # One table over the whole contract. The magnitude matters as much as the verdict: a
        # bare bool cost two wrong diagnoses of the same FP8 failures, because it cannot
        # separate a transport corruption from a tolerance too tight for an accumulator.
        tol = ep_harness.COMBINE_REL_TOL
        jitter = [value * (1.0 + tol / 2) for value in (1.0, -2.0)]
        for label, chained, drained, ok, check in (
            ("identical", [1.0, -3.5, 0.25], [1.0, -3.5, 0.25], True,
             lambda e: self.assertEqual(e, 0.0)),
            # A rank that legitimately combined nothing under this routing.
            ("empty", [], [], True, lambda e: self.assertEqual(e, 0.0)),
            # Run-to-run jitter inside tolerance passes, and still reports its size -- a
            # magnitude creeping toward the gate is the early warning a bool cannot give.
            ("jitter", jitter, [1.0, -2.0], True,
             lambda e: self.assertAlmostEqual(e, tol / 2)),
            # The defect class this exists for lands orders of magnitude past tolerance.
            ("corruption", [1.0, 2.0], [1.0, 4.0], False,
             lambda e: self.assertGreater(e, 10 * tol)),
            # No elementwise error is defined across shapes; infinity stops a cross-rank MAX
            # reporting a small number for a structural mismatch.
            ("shape", [1.0, 2.0], [1.0, 2.0, 3.0], False,
             lambda e: self.assertEqual(e, float("inf"))),
        ):
            with self.subTest(label):
                got, error = ep_harness._chain_output_matches(_Vec(chained), _Vec(drained))
                self.assertIs(got, ok)
                check(error)

    def test_near_zero_elements_are_judged_against_the_magnitude_floor(self):
        # Relative error against a denominator of 1e-6 would be huge; the floor keeps
        # numerically-tiny elements from redding a healthy chain.
        drained, chained = 1e-6, 1e-6 + 1e-4
        self.assertGreater(
            abs(chained - drained) / abs(drained), ep_harness.COMBINE_REL_TOL
        )
        self.assertTrue(
            ep_harness._chain_output_matches(_Vec([chained]), _Vec([drained]))[0]
        )


# ---- from test_summarize_headline.py ----------------------------------------------
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
    # --runner filters on the SKU the row declares, not on the result filename: results are
    # named by case_id, which joins its factors with "-", so any filename-prefix test would
    # match nothing and render an empty table rather than failing.
    def _directory(self, names):
        directory = tempfile.mkdtemp()
        for name, sku in names:
            record = document(with_period=True)
            # The shared fixture omits record_type (nothing else loads from disk);
            # load_results discriminates on it, so a written record must carry it.
            record["record_type"] = summarize.CASE_RECORD_TYPE
            record["identity"]["case_factors"]["sku"] = sku
            (Path(directory) / name).write_text(json.dumps(record))
        return directory

    def test_case_id_named_files_still_match_their_runner(self):
        directory = self._directory([
            ("stub-sku-stub-deepseek-v3-low-latency-decode-ep8-uniform-bf16_TS-c000.json",
             "stub-sku"),
            ("other-sku-stub-deepseek-v3-low-latency-decode-ep8-uniform-bf16_TS-c000.json",
             "other-sku"),
        ])
        kept = summarize.load_results(directory, "stub-sku", None)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["identity"]["case_factors"]["sku"], "stub-sku")

    def test_a_foreign_sku_is_excluded_even_when_the_filename_would_match(self):
        # The filename claims one SKU, the row declares another; the row wins.
        directory = self._directory([("stub-sku_anything_TS-c000.json", "other-sku")])
        self.assertEqual(summarize.load_results(directory, "stub-sku", None), [])


if __name__ == "__main__":
    unittest.main()
