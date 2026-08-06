#!/usr/bin/env python3
"""Driven contract for the chained regime's oracle gate and its barrier-mode suppression.

These drive `run_sweep` end to end against stub torch/dist/routing modules and a stub backend,
then assert on the artifact it writes. That is the level the two review amendments actually
live at: whether a field name appears somewhere in the module says nothing about whether a real
row carries it with the right origin, the right sample count and the right tri-state.

SCOPE -- wiring, not arithmetic. `_run_expert_oracle` is replaced by a scripted verdict, so what
is covered here is that the chained oracle RUNS once per ladder point after that point's final
chain trial, with Pass 3's invocation shape, and that its verdict reaches
`correctness.chain_regime_passed`, `correctness.passed`, `correctness.max_relative_error` and
the process exit code -- and that barrier mode skips it. The oracle's own expected-combine math
is a different subject with a different fixture (it needs real tensors): nothing here would
catch a wrong expectation, only a verdict that was computed and then dropped on the floor.
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import statistics
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "bench")]

import ep_backend  # noqa: E402
import ep_harness  # noqa: E402

LADDER = [4, 8]
CHAIN_ITERS, CHAIN_DROP, CHAIN_TRIALS = 8, 2, 2
KEPT_PER_TRIAL = CHAIN_ITERS - CHAIN_DROP
# What the stub backend reports for every chained iteration, so a published number is traceable
# to the op it came from rather than to an accidental sum.
PAIR_US, DISPATCH_FLOOR_US, COMBINE_FLOOR_US = 50.0, 20.0, 25.0
UNAVAILABLE = {
    "availability": "unavailable", "origin": None, "percentiles_us": None, "sample_count": 0,
}


class _FakeTensor:
    """Enough tensor for the reductions run_sweep performs: gather, stack, median/max/min, sub."""

    def __init__(self, data):
        self.data = data

    def tolist(self):
        return self.data

    def item(self):
        return self.data[0]

    def clone(self):
        return _FakeTensor(copy.deepcopy(self.data))

    def to(self, *args, **kwargs):
        return self

    def __iter__(self):
        return iter(self.data)

    def __sub__(self, other):
        return _FakeTensor([a - b for a, b in zip(self.data, other.data)])

    def _reduce(self, fn):
        columns = (
            [[row[i] for row in self.data] for i in range(len(self.data[0]))]
            if self.data else []
        )
        return SimpleNamespace(values=_FakeTensor([fn(column) for column in columns]))

    def median(self, dim=0):
        return self._reduce(statistics.median)

    def max(self, dim=0):
        return self._reduce(max)

    def min(self, dim=0):
        return self._reduce(min)

    @property
    def shape(self):
        return (len(self.data),)


class _FakeEvent:
    clock = [0.0]

    def __init__(self, enable_timing=False):
        self.t = None

    def record(self):
        _FakeEvent.clock[0] += 1.0
        self.t = _FakeEvent.clock[0]

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


def fake_torch():
    torch = types.ModuleType("torch")
    torch.float64, torch.int64, torch.bfloat16 = "f64", "i64", "bf16"
    torch.cuda = SimpleNamespace(synchronize=lambda: None, Event=_FakeEvent)
    torch.tensor = lambda values, device=None, dtype=None: _FakeTensor(list(values))
    torch.empty_like = lambda x: _FakeTensor(list(x.data))
    torch.stack = lambda xs: _FakeTensor([x.data for x in xs])
    torch.equal = lambda a, b: a.data == b.data
    torch.zeros = lambda n, device=None: _FakeTensor([0.0] * n)
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


class _StubBackend(ep_backend.EPBackend):
    """Constant-cost backend; every timed call returns a value unique to what it measures."""

    name = "stub"
    maturity = "candidate"

    def __init__(self, chain_barrier=False):
        self.chain_barrier = chain_barrier
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
                tokens_per_rank=tokens, topk_idx=_FakeTensor([0]),
                topk_weights=_FakeTensor([1.0]), activations=_FakeTensor([1.0]),
                global_idx=_FakeTensor([0]), global_weights=_FakeTensor([1.0]),
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
        return ([PAIR_US] * kept, [DISPATCH_FLOOR_US] * kept, [COMBINE_FLOOR_US] * kept)

    def dispatch(self, problem):
        return SimpleNamespace(combine_input=None)

    def stage(self, problem, handle):
        return None

    def combine(self, problem, handle):
        return None

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
    """Which pass each oracle call belongs to, by position.

    Pass 1 opens with one call per ladder point and Pass 3 closes with one per point, so
    anything between them is the chained gate. Deliberately positional rather than inferred
    from neighbouring chain calls: in barrier mode the chained gate does not run, and Pass 3's
    first call then sits immediately after a chain call too, which is exactly where a
    neighbour rule quietly mislabels it. The tests below assert this structure directly on the
    raw event log, so nothing here is assumed -- this only spares the failure cases from
    hardcoding a call index.
    """
    return (
        ["pre"] * points
        + ["chain"] * (oracle_count - 2 * points)
        + ["post"] * points
    )


def _sweep(chain_barrier, fail_indices, error_indices, chain_error):
    """One full run_sweep against the stubs; failures scripted by oracle call index."""
    backend = _StubBackend(chain_barrier=chain_barrier)
    events = backend.events
    oracle_calls = []

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

    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory) / "result.json"
        stdout = io.StringIO()
        with mock.patch.dict(sys.modules, {"routing": fake_routing()}), \
                mock.patch.dict(os.environ, {"COLLX_ATTEMPT_ID": "1"}), \
                mock.patch.object(ep_harness, "_run_expert_oracle", fake_oracle), \
                contextlib.redirect_stdout(stdout):
            rc = ep_harness.run_sweep(
                make_args(out), backend, fake_torch(), _FakeDist(), "cuda:0", 0, 1
            )
        doc = json.loads(out.read_text())
    return SimpleNamespace(
        rc=rc, doc=doc, rows=doc["measurement"]["rows"], events=events,
        oracle_calls=oracle_calls, stdout=stdout.getvalue(), backend=backend,
        phases=phases_by_index(len(oracle_calls), len(LADDER)),
    )


def drive(*, chain_barrier=False, fail_phases=(), chain_error=0.0):
    """Run the sweep, optionally failing every oracle of a given pass.

    A clean probe run first, to learn how many oracle calls the configuration makes; the phase
    labels then follow from position and the real run scripts its failures by index. The probe
    is what keeps the failure cases from hardcoding "call 3 and 4", which would silently start
    failing a different pass the moment the ladder or the trial count changed.
    """
    probe = _sweep(chain_barrier, frozenset(), frozenset(), 0.0)
    if not fail_phases and not chain_error:
        return probe
    selected = lambda wanted: frozenset(  # noqa: E731
        index for index, phase in enumerate(probe.phases) if phase in wanted
    )
    return _sweep(
        chain_barrier, selected(set(fail_phases)),
        selected({"chain"}) if chain_error else frozenset(), chain_error,
    )


class ChainedRegimeOracleGate(unittest.TestCase):
    """The published regime has to be the gated one.

    Passes 1 and 3 only ever check drained calls, so without a chained-regime check the pair
    period would be published from a regime no oracle has ever seen -- and a backend that
    corrupts only under free-running pairs would present as the fastest one in the suite.
    """

    def test_the_chained_oracle_runs_once_per_point_after_its_final_chain_trial(self):
        run = drive()
        points = len(LADDER)
        kinds = [kind for kind, _ in run.events]

        # Pass 1 opens with one oracle per point; Pass 3 closes with one per point, in ladder
        # order. Asserted on the raw log, so the positional phase labels rest on this and not
        # the other way round.
        self.assertEqual(kinds[:points], ["oracle"] * points)
        self.assertEqual(sorted(T for _, T in run.events[:points]), sorted(LADDER))
        self.assertEqual(kinds[-points:], ["oracle"] * points)
        self.assertEqual([T for _, T in run.events[-points:]], list(LADDER))

        middle = run.events[points:-points]
        middle_kinds = [kind for kind, _ in middle]
        self.assertEqual(middle_kinds.count("chain"), CHAIN_TRIALS * points)
        self.assertEqual(middle_kinds.count("oracle"), points)
        self.assertEqual(
            sorted(T for kind, T in middle if kind == "oracle"), sorted(LADDER),
        )

        for index, (kind, T) in enumerate(middle):
            if kind != "oracle":
                continue
            with self.subTest(tokens=T):
                # It gated the state that point's own chain left behind ...
                self.assertEqual(middle[index - 1], ("chain", T))
                # ... on the FINAL trial: no further chain for that point follows it.
                self.assertNotIn(("chain", T), middle[index + 1:])

    def test_the_chained_oracle_is_invoked_with_pass_3s_shape(self):
        # A stale global trace or a mismatched expert count here would gate a different problem
        # than the one the chain just measured, which is worse than not gating at all.
        run = drive()
        calls = {}
        for index, T, call_args in run.oracle_calls:
            phase = run.phases[index]
            if phase in ("chain", "post"):
                calls.setdefault(T, {})[phase] = call_args
        self.assertEqual(sorted(calls), sorted(LADDER))
        for T, per_phase in sorted(calls.items()):
            with self.subTest(tokens=T):
                self.assertEqual(per_phase["chain"], per_phase["post"])

    def test_a_healthy_chain_publishes_a_passing_regime(self):
        run = drive()
        self.assertEqual(run.rc, 0)
        self.assertEqual(run.doc["outcome"]["status"], "success")
        for row in run.rows:
            with self.subTest(tokens=row["tokens_per_rank"]):
                self.assertIs(row["correctness"]["chain_regime_passed"], True)
                self.assertIs(row["correctness"]["passed"], True)

    def test_a_chained_oracle_failure_reds_the_case(self):
        # It must invalidate the leg rather than quietly leave a field out: rc is the only
        # success signal CI reads, and the doc is uploaded either way.
        run = drive(fail_phases=("chain",))
        self.assertEqual(run.rc, 3)
        self.assertEqual(run.doc["outcome"]["status"], "invalid")
        for row in run.rows:
            with self.subTest(tokens=row["tokens_per_rank"]):
                self.assertIs(row["correctness"]["chain_regime_passed"], False)
                self.assertIs(row["correctness"]["passed"], False)

    def test_the_drained_oracles_still_red_the_case_on_their_own(self):
        # The chained gate is an addition, not a replacement.
        for phase in ("pre", "post"):
            with self.subTest(phase=phase):
                run = drive(fail_phases=(phase,))
                self.assertEqual(run.rc, 3)
                self.assertEqual(run.doc["outcome"]["status"], "invalid")
                for row in run.rows:
                    self.assertIs(row["correctness"]["passed"], False)
                    # The chained regime itself was clean, and says so.
                    self.assertIs(row["correctness"]["chain_regime_passed"], True)

    def test_the_chained_error_is_folded_into_max_relative_error(self):
        # Maxed in like the other two oracles', so a chained regime that is within tolerance but
        # noticeably worse than the drained one is visible instead of being averaged away.
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

    def test_the_pair_period_is_published_as_a_chained_median(self):
        for row in self.swept.rows:
            with self.subTest(tokens=row["tokens_per_rank"]):
                period = row["components"]["pair_period"]
                self.assertEqual(period["percentiles_us"]["p50"], PAIR_US)
                self.assertEqual(period["origin"], "chained-median")
                self.assertEqual(period["availability"], "measured")
                self.assertEqual(period["sample_count"], KEPT_PER_TRIAL * CHAIN_TRIALS)

    def test_the_per_op_floors_are_published_as_cross_rank_minima(self):
        for row in self.swept.rows:
            for op, expected in (
                ("dispatch", DISPATCH_FLOOR_US), ("combine", COMBINE_FLOOR_US),
            ):
                with self.subTest(tokens=row["tokens_per_rank"], op=op):
                    floor = row["chain_floor_us"][op]
                    self.assertEqual(floor["percentiles_us"]["p50"], expected)
                    self.assertEqual(floor["origin"], "chained-cross-rank-min")

    def test_the_pair_spread_is_published_as_the_health_proof(self):
        for row in self.swept.rows:
            with self.subTest(tokens=row["tokens_per_rank"]):
                spread = row["chain_health"]["pair_spread_us"]
                # One rank, so the ranks trivially agree; the point is that it is emitted and
                # component-shaped, since a wide spread is what disqualifies a period.
                self.assertEqual(spread["percentiles_us"]["p50"], 0.0)
                self.assertEqual(set(spread), set(UNAVAILABLE))

    def test_the_period_does_not_displace_the_fresh_entry_family(self):
        # The chained family is additive: roundtrip and the isolated components keep their
        # fresh-entry meaning, and old consumers keep reading them unchanged.
        for row in self.swept.rows:
            with self.subTest(tokens=row["tokens_per_rank"]):
                self.assertEqual(row["components"]["roundtrip"]["origin"], "measured")
                self.assertEqual(row["components"]["roundtrip"]["percentiles_us"]["p50"], 10.0)
                self.assertEqual(row["components"]["dispatch"]["percentiles_us"]["p50"], 10.0)

    def test_the_doc_stamps_the_chain_as_free_running(self):
        implementation = self.swept.doc["implementation"]
        self.assertIs(implementation["chained_period"], True)
        self.assertIs(implementation["chain_barrier"], False)

    def test_the_sampling_block_records_the_chain_budget(self):
        sampling = self.swept.doc["measurement"]["sampling"]
        self.assertEqual(sampling["chain_iterations_per_trial"], CHAIN_ITERS)
        self.assertEqual(sampling["chain_trials"], CHAIN_TRIALS)
        self.assertEqual(sampling["chain_drop"], CHAIN_DROP)

    def test_the_per_point_line_reports_the_period(self):
        self.assertIn("period=", self.swept.stdout)
        self.assertNotIn("period=n/a", self.swept.stdout)


class BarrierModeSuppression(unittest.TestCase):
    """A barrier-mode period is a different quantity, so it is measured and then not published.

    The barrier adds its own cost and removes the cross-pair overlap the period is defined by.
    Emitting it into the same fields would let something rank two definitions against each
    other, so the chain still runs as a bring-up diagnostic and the fields stay unavailable.
    """

    @classmethod
    def setUpClass(cls):
        cls.swept = drive(chain_barrier=True)

    def test_the_chain_still_runs(self):
        self.assertEqual(
            [kind for kind, _ in self.swept.events].count("chain"),
            CHAIN_TRIALS * len(LADDER),
        )

    def test_nothing_chained_is_published(self):
        for row in self.swept.rows:
            tokens = row["tokens_per_rank"]
            with self.subTest(tokens=tokens):
                self.assertEqual(row["components"]["pair_period"], UNAVAILABLE)
                self.assertEqual(row["chain_floor_us"]["dispatch"], UNAVAILABLE)
                self.assertEqual(row["chain_floor_us"]["combine"], UNAVAILABLE)
                self.assertEqual(row["chain_health"]["pair_spread_us"], UNAVAILABLE)

    def test_the_chained_oracle_is_skipped(self):
        # Pass 1 and Pass 3 only: with nothing chained published there is nothing to stand
        # behind, and a gate that ran anyway could red a leg over a number no one can read.
        points = len(LADDER)
        kinds = [kind for kind, _ in self.swept.events]
        self.assertEqual(kinds.count("oracle"), 2 * points)
        middle = self.swept.events[points:-points]
        self.assertEqual(
            [kind for kind, _ in middle], ["chain"] * (CHAIN_TRIALS * points),
        )

    def test_the_regime_verdict_is_null_rather_than_failed(self):
        # There was no chained number to gate, which is not the same as a chained number that
        # failed its gate; false here would red a leg for a check that never ran.
        self.assertEqual(self.swept.rc, 0)
        self.assertEqual(self.swept.doc["outcome"]["status"], "success")
        for row in self.swept.rows:
            with self.subTest(tokens=row["tokens_per_rank"]):
                self.assertIsNone(row["correctness"]["chain_regime_passed"])
                self.assertIs(row["correctness"]["passed"], True)

    def test_the_doc_records_why_the_blocks_are_unavailable(self):
        implementation = self.swept.doc["implementation"]
        self.assertIs(implementation["chain_barrier"], True)
        # The chain ran, so the case still measured the chained regime; only publication stopped.
        self.assertIs(implementation["chained_period"], True)

    def test_the_measured_numbers_survive_only_on_stdout_labeled_as_diagnostic(self):
        self.assertIn("period=n/a", self.swept.stdout)
        self.assertIn("not published (chain_barrier=on)", self.swept.stdout)
        self.assertIn(f"period p50={PAIR_US:.1f}us", self.swept.stdout)


if __name__ == "__main__":
    unittest.main()
