#!/usr/bin/env python3
"""EPBackend contracts: ladder/spec construction, the staging-vs-roundtrip gate, and the NCCL EP handle."""
from __future__ import annotations

import os
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
    def test_invalid_or_fully_clamped_ladder_fails_before_execution(self):
        for backend, message in (
            (FakeBackend(args(tokens_ladder="0")), "empty token ladder"),
            (FakeBackend(args(tokens_ladder="128"), cap=64), "cap=64"),
        ):
            with self.subTest(message=message):
                spec = backend.make_inputs(backend.args)
                self.assertEqual(spec.rc, 2)
                self.assertIn(message, spec.message)

    def test_mode_is_fail_closed(self):
        with self.assertRaises(ValueError):
            FakeBackend(args(mode="unsupported"))

    def test_precision_is_fail_closed(self):
        # The base SUPPORTED_PRECISIONS is BF16-only; an adapter that has not opted
        # into a precision must reject it rather than silently run the wrong codec.
        with self.assertRaises(ValueError):
            FakeBackend(args(precision="fp8"))

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

    def test_an_unrecognised_consume_mode_fails_instead_of_silently_meaning_native(self):
        # The value is read at class-body evaluation, so a typo raises at import -- before
        # any measurement -- rather than quietly running the default model and tagging the
        # artifact with whatever the typo said.
        import importlib

        with mock.patch.dict(os.environ, {"CX_FP8_CONSUME": "dequantize"}):
            with self.assertRaises(ValueError):
                importlib.reload(ep_backend)
        # Restore the module other tests hold references into.
        importlib.reload(ep_backend)
        self.assertEqual(ep_backend.EPBackend.fp8_consume, "native")

    def test_a_backend_is_deployed_unless_it_says_otherwise(self):
        # path_status is per-CASE: flashinfer-ep is a production BACKEND whose fp8 precision
        # no engine can select, which per-backend maturity cannot express.
        self.assertEqual(ep_backend.EPBackend.path_status, "deployed")


class NcclLowLatencyLadderClamp(unittest.TestCase):
    """The measured ladder is clamped below the receive buffer around an unfixed upstream race."""

    def _module(self):
        with mock.patch.dict(sys.modules, _stub_modules()):
            import importlib
            import ep_nccl
            return importlib.reload(ep_nccl)

    def test_the_ladder_is_clamped_below_the_buffer(self):
        # Two separate numbers on purpose: clamping the ladder must not shrink the transport
        # footprint, or the rungs that remain quietly measure a smaller receive plane.
        m = self._module()
        self.assertLess(m._LL_LADDER_CAP, m._LL_BUFFER_CAP)
        self.assertLessEqual(m._LL_BUFFER_CAP, 511)

    def test_buffer_cap_bounds_the_ladder_not_the_allocation(self):
        # buffer_cap() is what the harness clamps the ladder against, so it must report the
        # LADDER cap; the allocation reads _LL_BUFFER_CAP directly.
        m = self._module()
        source = (ROOT / "bench" / "ep_nccl.py").read_text()
        self.assertIn("return _LL_LADDER_CAP", source)
        self.assertIn("_LL_BUFFER_CAP if self._ll else", source)


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

if __name__ == "__main__":
    unittest.main()
