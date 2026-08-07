#!/usr/bin/env python3
"""EPBackend contracts: ladder/spec construction, the staging-vs-roundtrip gate, and the NCCL EP handle."""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "bench")]

import ep_backend  # noqa: E402
from ep_backend import EPBackend, RankInputs  # noqa: E402


# ---- from test_ep_backend.py ------------------------------------------------------
def args(**updates):
    values = dict(
        experts=8, phase="decode", tokens_ladder="", routing="uniform", seed=0,
        hidden=16, topk=2, mode="normal", precision="bf16",
    )
    values.update(updates)
    return types.SimpleNamespace(**values)


class FakeBackend(EPBackend):
    name = "fake"

    def __init__(self, options, *, cap=None, world_size=1):
        super().__init__(options, 0, world_size, 0, "cpu")
        self.cap = cap
        self.calls: list[str] = []

    def create_buffer(self, spec):
        return None

    def dispatch(self, problem):
        self.calls.append("dispatch")
        return object()

    def stage(self, problem, handle):
        self.calls.append("stage")

    def combine(self, problem, handle):
        self.calls.append("combine")

    def recv_tokens(self, handle):
        return 0

    def inspect_dispatch(self, problem, handle):
        return None

    def combine_transformed(self, problem, handle, transformed):
        return None

    def buffer_cap(self, options):
        return self.cap

    def _build_rank_inputs(self, options, tokens):
        return RankInputs(
            tokens_per_rank=tokens, topk_idx=None, topk_weights=None,
            activations=None,
        )


class BackendTests(unittest.TestCase):
    def test_input_plan_sizes_for_the_measured_ladder(self):
        backend = FakeBackend(args(tokens_ladder="8 16"), world_size=2)
        spec = backend.make_inputs(backend.args)
        self.assertTrue(spec.ok)
        self.assertEqual(spec.ladder, [8, 16])
        self.assertEqual(spec.max_tokens_per_rank, 16)
        self.assertEqual((spec.ep_size, spec.experts_per_rank), (2, 4))
        self.assertEqual(sorted(spec.points), [8, 16])

    def test_invalid_or_fully_clamped_ladder_fails_before_execution(self):
        for backend, message in (
            (FakeBackend(args(tokens_ladder="0")), "empty token ladder"),
            (FakeBackend(args(tokens_ladder="128"), cap=64), "cap=64"),
        ):
            with self.subTest(message=message):
                spec = backend.make_inputs(backend.args)
                self.assertEqual(spec.rc, 2)
                self.assertIn(message, spec.message)

    def test_timed_components_follow_backend_contract(self):
        backend = FakeBackend(args())
        self.assertEqual(backend.timed_components(), ["roundtrip", "dispatch", "combine"])
        backend.stage_device_work = True
        self.assertEqual(
            backend.timed_components(), ["roundtrip", "dispatch", "combine", "stage"]
        )

    def test_dispatch_cleanup_is_outside_timed_call(self):
        backend = FakeBackend(args())
        backend.requires_fresh_pair = True
        captured = {}

        def fake_time(_torch, operation, _warmup, _iters, **kwargs):
            handle = operation()
            kwargs["post"](handle)
            captured.update(kwargs)
            return [1.0]

        with mock.patch.dict(sys.modules, {"torch": types.SimpleNamespace()}), mock.patch.object(
            ep_backend, "time_us", side_effect=fake_time
        ):
            backend.benchmark_dispatch(object(), 0, 1)
        self.assertIn("post", captured)
        self.assertEqual(backend.calls, ["dispatch", "stage", "combine"])

    def test_stage_cleanup_matches_the_dispatch_contract(self):
        # MoRI-shaped backends (requires_fresh_pair) must not leak an
        # un-combined dispatch out of an isolated-stage iteration.
        for needs_cleanup, calls in (
            (True, ["dispatch", "stage", "combine"]), (False, ["dispatch", "stage"]),
        ):
            backend = FakeBackend(args())
            backend.requires_fresh_pair = needs_cleanup

            def fake_time(_torch, operation, _warmup, _iters, **kwargs):
                result = operation(kwargs["pre"]())
                if kwargs["post"] is not None:
                    kwargs["post"](result)
                return [1.0]

            with mock.patch.dict(sys.modules, {"torch": types.SimpleNamespace()}), mock.patch.object(
                ep_backend, "time_us", side_effect=fake_time
            ):
                backend.benchmark_stage(object(), 0, 1)
            with self.subTest(needs_cleanup=needs_cleanup):
                self.assertEqual(backend.calls, calls)

    def test_mode_is_fail_closed(self):
        with self.assertRaises(ValueError):
            FakeBackend(args(mode="unsupported"))

    def test_low_latency_mode_accepted_only_when_declared(self):
        # The base backend is normal-only, so it must reject low-latency; an adapter that
        # declares it in SUPPORTED_MODES is accepted and can carry the token-expert
        # receive layout the low-latency oracle path keys on, alongside its
        # weighted-kernel combine.
        with self.assertRaises(ValueError):
            FakeBackend(args(mode="low-latency"))

        class LowLatencyBackend(FakeBackend):
            SUPPORTED_MODES = ("normal", "low-latency")

        backend = LowLatencyBackend(args(mode="low-latency"))
        backend.receive_layout = "token-expert"
        backend.combine_weight_semantics = "weighted-kernel-sum"
        self.assertEqual(backend.mode, "low-latency")
        self.assertEqual(backend.receive_layout, "token-expert")
        self.assertEqual(backend.combine_weight_semantics, "weighted-kernel-sum")

    def test_precision_is_fail_closed(self):
        # The base SUPPORTED_PRECISIONS is BF16-only; an adapter that has not opted
        # into a precision must reject it rather than silently run the wrong codec.
        with self.assertRaises(ValueError):
            FakeBackend(args(precision="fp8"))

    def test_base_dispatch_encoding_is_identity(self):
        # BF16 default: semantic_payload is identity, so oracle_x is x itself and the
        # combine oracle compares against the source activations (unchanged behavior).
        backend = FakeBackend(args())
        payload = object()
        self.assertIs(backend.semantic_payload(payload), payload)
        self.assertIsNone(backend._validate_quantizer(payload))
        self.assertEqual(backend.dispatch_dtype, "bf16")
        self.assertEqual(backend.combine_dtype, "bf16")

    def test_make_problem_sends_x_and_points_the_oracle_at_semantic_payload(self):
        # dispatch_x is always x -- adapters quantize inside dispatch(), where production
        # pays it -- and oracle_x is the semantic round-trip, so the two can never drift
        # apart the way two independent encode paths could.
        backend = FakeBackend(args())
        calls = []
        backend.semantic_payload = lambda value: calls.append(value) or "semantic"
        torch = types.ModuleType("torch")
        torch.float32, torch.int64 = "float32", "int64"
        cast = lambda dtype: f"cast:{dtype}"  # noqa: E731
        with mock.patch.dict(sys.modules, {"torch": torch}):
            problem = backend.make_problem(
                4, types.SimpleNamespace(to=cast), types.SimpleNamespace(to=cast), "X"
            )
        self.assertIs(problem.dispatch_x, problem.x)
        self.assertEqual(problem.dispatch_x, "X")
        self.assertEqual(problem.oracle_x, "semantic")
        self.assertEqual(calls, ["X"])


# ---- from test_roundtrip_staging.py -----------------------------------------------
class _StagingBackend(ep_backend.EPBackend):
    """Records the call order; no device work."""

    name = "stub"

    def __init__(self, stage_device_work: bool, fp8_consume: str, precision: str = "fp8"):
        self.calls: list[str] = []
        self.stage_device_work = stage_device_work
        self.fp8_consume = fp8_consume
        self.precision = precision

    def create_buffer(self, spec):  # pragma: no cover - unused
        raise NotImplementedError

    def dispatch(self, problem):
        self.calls.append("dispatch")
        return types.SimpleNamespace(combine_input=None)

    def stage(self, problem, handle):
        self.calls.append("stage")
        handle.combine_input = "staged-by-stage"

    def combine(self, problem, handle):
        self.calls.append(f"combine({handle.combine_input})")
        return handle.combine_input

    def recv_tokens(self, handle):  # pragma: no cover - unused
        return 0

    def inspect_dispatch(self, problem, handle):  # pragma: no cover - unused
        return {}

    def combine_transformed(self, problem, handle, transformed):  # pragma: no cover
        return transformed


class RoundtripStaging(unittest.TestCase):
    def test_staged_input_keeps_the_conversion_out_of_the_chain(self):
        b = _StagingBackend(stage_device_work=True, fp8_consume="native")
        b.run_roundtrip(object(), staged="pre-materialised")
        self.assertEqual(b.calls, ["dispatch", "combine(pre-materialised)"])
        self.assertNotIn("stage", b.calls)

    def test_without_staged_input_the_stage_runs_inline(self):
        # The fp8 `dequant` model takes this path, as does any backend whose `stage` is a bare
        # pointer assignment; mori/flashinfer-ep hoist their real device copy instead.
        b = _StagingBackend(stage_device_work=True, fp8_consume="dequant")
        b.run_roundtrip(object())
        self.assertEqual(b.calls, ["dispatch", "stage", "combine(staged-by-stage)"])

    def test_a_staged_tensor_reaches_combine_through_the_handle(self):
        # The staged expert-output stand-in must be what combine consumes; a break here
        # would silently measure a combine over stale data rather than fail.
        b = _StagingBackend(stage_device_work=True, fp8_consume="native")
        b.run_roundtrip(object(), staged="X")
        self.assertEqual(b.calls[-1], "combine(X)")

    def test_default_models_the_native_path(self):
        # deepseek-v3 block-fp8 hits vLLM's matched branch and SGLang's no-dequant path.
        self.assertEqual(ep_backend.EPBackend.fp8_consume, "native")


class RoundtripStagingGate(unittest.TestCase):
    """`roundtrip` must mean dispatch -> combine in every row, or it is not comparable: the gate
    is `stage_device_work` alone, with `CX_FP8_CONSUME=dequant` as the sole opt-out."""

    def test_the_gate_truth_table(self):
        table = (
            # A real device copy hoists regardless of precision (MoRI BF16 scale-up and
            # FlashInfer BF16 were previously charged here and to no other roundtrip)...
            (True, "native", "bf16", True),
            (True, "native", "fp8", True),
            # ...a pointer-assignment stage never does: hoisting would hand a low-latency
            # backend a view into its double-buffered receive...
            (False, "native", "bf16", False),
            (False, "native", "fp8", False),
            # ...and CX_FP8_CONSUME=dequant restores the inline stage for fp8 only -- a stack
            # that really converts between the collectives -- with nothing to model at BF16.
            (True, "dequant", "fp8", False),
            (True, "dequant", "bf16", True),
        )
        for stage_device_work, consume, precision, hoisted in table:
            with self.subTest(stage=stage_device_work, consume=consume, precision=precision):
                backend = _StagingBackend(stage_device_work, consume, precision)
                self.assertEqual(bool(backend.stage_excluded_from_roundtrip), hoisted)

class WarmStaging(unittest.TestCase):
    """Warm-up must not rehearse work the timed region skips: where staging is excluded from the
    chain it was the leg's largest single cost (~247us x 32 iters x every component x trial)."""

    @staticmethod
    def _warm(backend, count, **kwargs):
        # `warm` imports torch for one synchronize; a stub keeps this runnable without a GPU.
        fake = types.ModuleType("torch")
        fake.cuda = types.SimpleNamespace(synchronize=lambda: None)
        saved = sys.modules.get("torch")
        sys.modules["torch"] = fake
        try:
            backend.warm(types.SimpleNamespace(), count, **kwargs)
        finally:
            if saved is None:
                del sys.modules["torch"]
            else:
                sys.modules["torch"] = saved

    def test_stages_once_when_the_chain_excludes_staging(self):
        b = _StagingBackend(stage_device_work=True, fp8_consume="native")
        self._warm(b, 5)
        self.assertEqual(b.calls.count("dispatch"), 5)
        self.assertEqual(b.calls.count("stage"), 1)
        # Every later iteration still hands combine the staged payload, not a stale None.
        self.assertEqual(b.calls.count("combine(staged-by-stage)"), 5)

    def test_stage_every_rehearses_it_on_every_iteration(self):
        b = _StagingBackend(stage_device_work=True, fp8_consume="native")
        self._warm(b, 5, stage_every=True)
        self.assertEqual(b.calls.count("stage"), 5)

    def test_a_chain_that_includes_staging_keeps_warming_it(self):
        # The `dequant` hatch puts the conversion back in the timed chain, so warm-up must match.
        b = _StagingBackend(stage_device_work=True, fp8_consume="dequant")
        self._warm(b, 5)
        self.assertEqual(b.calls.count("stage"), 5)

# The chained-period staging contract lives in tests/test_chain_period.py, which asserts it per
# sibling chain with window values.


# ---- from test_ep_nccl_handle.py --------------------------------------------------
def _stub_modules():
    """Fake torch / nccl modules so `import ep_nccl` succeeds without the benchmark image."""
    torch = types.ModuleType("torch")
    torch.bfloat16 = "bfloat16"
    torch.int32 = "int32"
    torch.empty = lambda *a, **k: types.SimpleNamespace(shape=a[0] if a else ())
    torch.zeros = lambda *a, **k: types.SimpleNamespace(item=lambda: 7)
    torch.cuda = types.SimpleNamespace(synchronize=lambda: None)
    dist = types.ModuleType("torch.distributed")
    torch.distributed = dist

    ep = types.ModuleType("nccl.ep")
    for name in (
        "Algorithm", "CombineConfig", "CombineInputs", "CombineOutputs", "DispatchConfig",
        "DispatchInputs", "DispatchOutputs", "GroupConfig", "HandleConfig", "Layout",
        "LayoutInfo", "Tensor",
    ):
        setattr(ep, name, type(name, (), {"__init__": lambda self, *a, **k: None}))
    ep.Algorithm = types.SimpleNamespace(LOW_LATENCY="LL", HIGH_THROUGHPUT="HT")
    ep.Layout = types.SimpleNamespace(EXPERT_MAJOR="EM", FLAT="FLAT")
    core = types.ModuleType("nccl.core")
    pkg = types.ModuleType("nccl")
    pkg.ep, pkg.core = ep, core
    return {
        "torch": torch, "torch.distributed": dist,
        "nccl": pkg, "nccl.ep": ep, "nccl.core": core,
    }


sys.path[:0] = [str(ROOT), str(ROOT / "bench")]

# Import ep_nccl against the stubs, then withdraw them: a fake torch left in sys.modules makes
# genuinely torch-dependent modules (test_runtime, test_ll_oracle) error instead of skipping.
with mock.patch.dict(sys.modules, _stub_modules()):
    import ep_nccl  # noqa: E402

    sys.modules.pop("ep_nccl", None)


class FakeHandle:
    """Records every rebind so the tests can assert on the collective call pattern."""

    def __init__(self):
        self.updates = []
        self.destroyed = False

    def update(self, topk_idx, *, layout_info=None, stream=None):
        self.updates.append((topk_idx, layout_info))

    def destroy(self):
        self.destroyed = True


class FakeGroup:
    def __init__(self):
        self.created = 0
        self.handle = FakeHandle()

    def create_handle(self, layout, topk_idx, *, layout_info=None, config=None, stream=None):
        self.created += 1
        return self.handle


def backend(ll=True):
    """An NCCLEPBackend with just the fields _ensure_handle touches (no __init__, no GPU)."""
    b = object.__new__(ep_nccl.NCCLEPBackend)
    b._ll = ll
    b._layout = "EM" if ll else "FLAT"
    b._handle = None
    b._bound = None
    b._ep_group = FakeGroup()
    b.device = "cuda:0"
    b.num_local_experts = 4
    b.args = types.SimpleNamespace(hidden=16)
    b._t = lambda x: x
    b._stream = lambda: 0
    # create_buffer always runs before the first _ensure_handle, so the HT receive plane exists
    # by then; a list stands in for the tensor because `_t` is identity here.
    b._recv_x = list(range(64))
    return b


def problem(T):
    return types.SimpleNamespace(
        T=T, dispatch_x=f"x{T}", topk_idx=f"idx{T}", topk_weights=f"w{T}"
    )


class TestSingleHandle(unittest.TestCase):
    def test_one_handle_across_many_shapes(self):
        """Nine ladder rungs must still produce exactly one create_handle."""
        b = backend()
        for T in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            b._ensure_handle(problem(T))
        self.assertEqual(b._ep_group.created, 1)

    def test_shape_change_rebinds_and_repeat_does_not(self):
        """update() on a shape switch; no collective when the bound shape is re-entered."""
        b = backend()
        pa, pb = problem(1), problem(2)
        b._ensure_handle(pa)
        self.assertEqual(len(b._ep_group.handle.updates), 0)  # first bind is the create

        b._ensure_handle(pb)
        self.assertEqual(len(b._ep_group.handle.updates), 1)

        # Re-entering the bound problem must not enter a collective, or every iteration of the
        # timed loop gains a rank-synchronising step.
        for _ in range(8):
            b._ensure_handle(pb)
        self.assertEqual(len(b._ep_group.handle.updates), 1)

        # Returning to an earlier shape rebinds again (its cached namespace is reused).
        b._ensure_handle(pa)
        self.assertEqual(len(b._ep_group.handle.updates), 2)

    def test_every_problem_shares_the_one_handle(self):
        b = backend()
        handles = {id(b._ensure_handle(problem(T)).handle) for T in (1, 2, 4)}
        self.assertEqual(len(handles), 1)
        self.assertIs(b._ensure_handle(problem(1)).handle, b._handle)

    def test_ll_never_passes_layout_info_on_rebind(self):
        """The API forbids layout_info on create/update in LL mode."""
        b = backend(ll=True)
        b._ensure_handle(problem(1))
        b._ensure_handle(problem(2))
        self.assertEqual([info for _, info in b._ep_group.handle.updates], [None])

    def test_ll_gate_wrapper_is_built_once_per_handle(self):
        """LL applies the gate in combine, so its weights wrapper must be cached: building one
        per timed combine puts a torch resolve and an np.asarray inside `time_us`."""
        ll = backend(ll=True)
        pa = problem(1)
        h = ll._ensure_handle(pa)
        self.assertTrue(hasattr(h, "combine_weights_t"))
        self.assertEqual(h.combine_weights_t, "w1")
        # Re-entering the same problem reuses the handle and therefore the wrapper.
        self.assertIs(ll._ensure_handle(pa).combine_weights_t, h.combine_weights_t)

        ht = backend(ll=False)
        self.assertFalse(hasattr(ht._ensure_handle(problem(1)), "combine_weights_t"))

    def test_ht_combine_input_is_sliced_to_the_received_count(self):
        """HT combine's staging copy is sized by the tensor it is handed: the whole ladder-max
        receive plane put a rung-independent floor under it. LL keeps the full padded plane."""
        b = backend(ll=False)
        h = b._ensure_handle(problem(1))
        # 7 is what the stubbed `torch.zeros(...).item()` reports as the received count.
        self.assertEqual(h.count, 7)
        self.assertEqual(h.combine_in_t, list(range(7)))
        self.assertLess(len(h.combine_in_t), len(b._recv_x))

        ll = backend(ll=True)
        ll_h = ll._ensure_handle(problem(1))
        self.assertFalse(hasattr(ll_h, "combine_in_t"))

    def test_ht_rebind_carries_that_problems_counters(self):
        """HT re-runs the metadata exchange into the rebound problem's own counter tensors."""
        b = backend(ll=False)
        ha = b._ensure_handle(problem(1))
        hb = b._ensure_handle(problem(2))
        self.assertEqual(len(b._ep_group.handle.updates), 1)
        self.assertIs(b._ep_group.handle.updates[0][1], hb.layout_info)
        self.assertIsNot(ha.layout_info, hb.layout_info)
        self.assertEqual(hb.count, 7)  # re-read after the exchange

    def test_destroy_releases_the_handle_once(self):
        b = backend()
        b._ensure_handle(problem(1))
        handle = b._handle
        b._destroy_handles()
        self.assertTrue(handle.destroyed)
        self.assertIsNone(b._handle)
        self.assertIsNone(b._bound)
        b._destroy_handles()  # idempotent


if __name__ == "__main__":
    unittest.main()
