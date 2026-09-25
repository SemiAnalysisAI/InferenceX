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


class FlashInferCombineModelSwitch(unittest.TestCase):
    """Which arithmetic the oracle holds FlashInfer's combine to, selected by wheel version.

    An inverted comparison or a typo'd `_COMBINE_FP32_SINCE` silently swaps the expected
    combine for every FlashInfer row -- a wrong-model failure runs 30-90x COMBINE_REL_TOL,
    so it reds correct runs rather than passing bad ones, but nothing else in CI sees it.
    """

    def _module(self):
        with mock.patch.dict(sys.modules, _stub_modules()):
            import importlib
            import ep_flashinfer
            return importlib.reload(ep_flashinfer)

    def test_the_fp32_boundary_is_exact_and_ordered_by_version_not_by_text(self):
        gate = self._module()._wheel_has_fp32_combine
        self.assertFalse(gate("0.6.15"), "the wheel below the boundary rounds per level")
        self.assertTrue(gate("0.6.16"), "the boundary wheel itself accumulates in FP32")
        self.assertTrue(gate("0.6.17"))
        # Real version ordering, not a digit scrape: 0.10.0 sorts BELOW 0.6.16 as text.
        self.assertTrue(gate("0.10.0"))

    def test_an_unreadable_version_falls_back_to_the_rounding_model(self):
        # Asymmetric costs: modelling FP32 against a per-level-rounding kernel can exceed
        # COMBINE_REL_TOL and red a correct run, while the opposite error costs a few ulps.
        gate = self._module()._wheel_has_fp32_combine
        for version in ("0.6.16rc1", "not-a-version", ""):
            self.assertFalse(gate(version), f"{version!r} must fall back to the safe model")


class NcclLowLatencyLadderSizing(unittest.TestCase):
    """The measured ladder and the receive buffer are separate knobs; under nccl-ep v0.2 (combine
    fence shipped) the ladder is restored to the full buffer, and must never exceed it."""

    def _module(self):
        with mock.patch.dict(sys.modules, _stub_modules()):
            import importlib
            import ep_nccl
            return importlib.reload(ep_nccl)

    def _backend(self, module, low_latency):
        """A backend far enough along to run create_buffer against the stubs."""
        backend = module.NCCLEPBackend.__new__(module.NCCLEPBackend)
        backend._ll = low_latency
        backend.zero_copy = True
        backend.world_size, backend.num_local_experts, backend.device = 8, 4, "cuda:0"
        backend.args = types.SimpleNamespace(hidden=7168, experts=256, topk=8)
        backend._algorithm = "LL" if low_latency else "HT"
        backend._layout = module.Layout.RANK_MAJOR if low_latency else module.Layout.FLAT
        backend._bootstrap_comm = lambda: None
        backend._recv_window_for_test = types.SimpleNamespace(
            is_valid=True, close=lambda: None
        )
        backend._comm = types.SimpleNamespace(
            register_window=mock.Mock(return_value=backend._recv_window_for_test)
        )
        backend.created_configs = []

        def group_config(**kwargs):
            config = types.SimpleNamespace(**kwargs)
            backend.created_configs.append(config)
            return config

        module.GroupConfig = group_config
        module.nccl_ep.Group = types.SimpleNamespace(create=lambda *a, **k: object())
        return backend

    def test_only_ll_rank_major_selects_the_nccl_fp32_reduction(self):
        module = self._module()
        module.dist.group = types.SimpleNamespace(WORLD=object())

        def base_init(instance, options, rank, world_size, local_rank, device):
            instance.args = options
            instance.mode = options.mode

        common = dict(experts=384, hidden=7168, topk=6, scale_up_domain=8)
        with mock.patch.object(module.EPBackend, "__init__", base_init):
            ll = module.NCCLEPBackend(
                types.SimpleNamespace(mode="low-latency", **common), 0, 16, 0, "cuda:0"
            )
            ht = module.NCCLEPBackend(
                types.SimpleNamespace(mode="normal", **common), 0, 16, 0, "cuda:0"
            )

        self.assertEqual(ll.combine_reduction, "rank-fp32")
        self.assertEqual(getattr(ht, "combine_reduction", "domain-fp32"), "domain-fp32")

    def test_ladder_cap_drops_only_oversized_measurement_points(self):
        module = self._module()
        backend = self._backend(module, low_latency=True)
        backend.args.tokens_ladder = "32 64 128"
        backend._build_rank_inputs = mock.Mock(return_value=None)
        with mock.patch.object(module, "_LL_LADDER_CAP", 64):
            spec = backend.make_inputs(backend.args)
        self.assertEqual(spec.ladder, [32, 64])
        self.assertEqual(spec.dropped, [128])

    def test_the_receive_is_sized_from_the_buffer_cap_not_the_ladder(self):
        # The regression this guards would silently re-baseline every low-latency row: clamping
        # the MEASURED ladder must not shrink the receive the remaining rungs are measured
        # against. Driven through create_buffer, so it fails on the allocation the kernel gets
        # rather than on the shape of the source line that computes it.
        module = self._module()
        spec = types.SimpleNamespace(max_tokens_per_rank=99)
        with mock.patch.object(module, "_LL_BUFFER_CAP", 512), \
                mock.patch.object(module, "_LL_LADDER_CAP", 64):
            sized = self._backend(module, low_latency=True)
            sized.create_buffer(spec)
            self.assertEqual(sized.max_dispatch, 512)
        # Throughput mode is unclamped and keeps taking its size from the ladder spec.
        throughput = self._backend(module, low_latency=False)
        throughput.create_buffer(spec)
        self.assertEqual(throughput.max_dispatch, 99)

    def test_ht_registers_a_symmetric_zero_copy_receive_window(self):
        module = self._module()
        backend = self._backend(module, low_latency=False)
        backend.create_buffer(types.SimpleNamespace(max_tokens_per_rank=99))

        self.assertEqual(backend.created_configs[0].zero_copy, module.ZeroCopyMode.ON)
        backend._comm.register_window.assert_called_once_with(
            backend._recv_x, flags=module.nccl_core.WindowFlag.COLL_SYMMETRIC
        )
        self.assertIs(backend._recv_x_t.window, backend._recv_window_for_test)
        self.assertEqual(backend._recv_x_t.window_offset, 0)

    def test_ll_rank_major_registers_a_direct_receive_window(self):
        module = self._module()
        backend = self._backend(module, low_latency=True)
        backend.create_buffer(types.SimpleNamespace(ladder=[1, 64], max_tokens_per_rank=64))

        self.assertTrue(backend.zero_copy)
        self.assertEqual(backend.created_configs[0].zero_copy, module.ZeroCopyMode.ON)
        backend._comm.register_window.assert_called_once_with(
            backend._recv_x, flags=module.nccl_core.WindowFlag.COLL_SYMMETRIC
        )
        self.assertEqual(backend._recv_x.shape, (8, 256, 7168))
        self.assertEqual(backend._recv_count.shape, (8,))
        self.assertIs(backend._recv_x_t.window, backend._recv_window)

    def test_ll_rank_major_scale_out_uses_the_staged_receive_plane(self):
        module = self._module()
        backend = self._backend(module, low_latency=True)
        backend.zero_copy = False
        backend.create_buffer(types.SimpleNamespace(max_tokens_per_rank=64))

        self.assertEqual(backend.created_configs[0].zero_copy, module.ZeroCopyMode.OFF)
        backend._comm.register_window.assert_not_called()
        self.assertIsNone(backend._recv_window)
        self.assertIsNone(backend._recv_x_t.window)

    def test_ll_dispatch_uses_rank_major_metadata_and_returns_expert_inputs(self):
        module = self._module()
        backend = self._backend(module, low_latency=True)
        backend.create_buffer(types.SimpleNamespace(ladder=[1, 64], max_tokens_per_rank=64))
        captured = {}

        class DispatchInputs:
            def __init__(self, **kwargs):
                captured["inputs"] = kwargs

        class DispatchOutputs:
            def __init__(self, **kwargs):
                captured["outputs"] = kwargs

        class LayoutInfo:
            def __init__(self, **kwargs):
                captured["layout"] = kwargs

        handle = types.SimpleNamespace(dispatch=mock.Mock())
        h = types.SimpleNamespace(in_tokens_t="tokens", in_weights_t="weights", handle=handle)
        module.DispatchInputs = DispatchInputs
        module.DispatchOutputs = DispatchOutputs
        module.LayoutInfo = LayoutInfo
        backend._ensure_handle = lambda _: h
        backend._dispatch_cfg = object()
        backend._stream = lambda: 0
        backend._finish = lambda *_: None

        backend.dispatch(object())

        self.assertIs(backend._layout, module.Layout.RANK_MAJOR)
        self.assertEqual(captured["inputs"], {"tokens": "tokens", "topk_weights": "weights"})
        self.assertEqual(captured["outputs"]["tokens"].window, backend._recv_window)
        self.assertIs(captured["outputs"]["topk_weights"], backend._recv_w_t)
        self.assertIs(captured["outputs"]["topk_idx"], backend._recv_idx_t)
        self.assertIs(captured["layout"]["src_rank_counters"], backend._recv_count_t)
        handle.dispatch.assert_called_once()


class RoundtripStagingGate(unittest.TestCase):
    """`roundtrip` must mean dispatch -> combine in every row, or it is not comparable: the gate
    is `stage_device_work` alone, with `CX_FP8_CONSUME=dequant` as the sole opt-out."""

    def test_the_gate_truth_table(self):
        table = (
            # A real device copy hoists regardless of precision, so MoRI BF16 scale-up and
            # FlashInfer BF16 are excluded on the same terms as every fp8 stage...
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
    class StubTorchTensor:
        def __init__(self, shape=()):
            self.shape = shape

        def data_ptr(self):
            return 0x1000

    class StubEpTensor:
        def __init__(self, buffer, *, window=None, window_offset=0):
            self.buffer = buffer
            self.window = window
            self.window_offset = window_offset

    torch = types.ModuleType("torch")
    torch.bfloat16 = "bfloat16"
    torch.int32 = "int32"
    torch.float32 = "float32"
    torch.int64 = "int64"
    torch.empty = lambda *a, **k: StubTorchTensor(a[0] if a else ())
    torch.empty_like = lambda x, *a, **k: StubTorchTensor(getattr(x, "shape", ()))
    torch.zeros = lambda *a, **k: types.SimpleNamespace(item=lambda: 7)
    torch.cuda = types.SimpleNamespace(synchronize=lambda: None)
    dist = types.ModuleType("torch.distributed")
    torch.distributed = dist

    ep = types.ModuleType("nccl.ep")
    for name in (
        "Algorithm", "CombineConfig", "CombineInputs", "CombineOutputs", "DispatchConfig",
        "DispatchInputs", "DispatchOutputs", "GroupConfig", "HandleConfig", "Layout",
        "LayoutInfo", "Tensor", "ZeroCopyMode",
    ):
        setattr(ep, name, type(name, (), {"__init__": lambda self, *a, **k: None}))
    ep.Algorithm = types.SimpleNamespace(LOW_LATENCY="LL", HIGH_THROUGHPUT="HT")
    ep.Layout = types.SimpleNamespace(EXPERT_MAJOR="EM", RANK_MAJOR="RM", FLAT="FLAT")
    ep.Tensor = StubEpTensor
    ep.ZeroCopyMode = types.SimpleNamespace(ON="ON", OFF="OFF")
    core = types.ModuleType("nccl.core")
    core.torch = types.SimpleNamespace(empty=torch.empty)
    core.WindowFlag = types.SimpleNamespace(COLL_SYMMETRIC="symmetric")
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
    b._window_t = lambda x: x
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

    def test_ll_rank_major_weight_wrapper_is_built_once_per_handle(self):
        """LL caches its dispatch-weight wrapper with the handle."""
        ll = backend(ll=True)
        ll._t = mock.Mock(side_effect=lambda value: types.SimpleNamespace(value=value))
        pa = problem(1)
        h = ll._ensure_handle(pa)
        first_weights = h.in_weights_t
        self.assertEqual(first_weights.value, "w1")
        # Re-entering the same problem reuses the handle and therefore the wrapper.
        ll._t.reset_mock()
        self.assertIs(ll._ensure_handle(pa).in_weights_t, first_weights)
        ll._t.assert_not_called()

    def test_ht_combine_input_is_sliced_to_the_received_count(self):
        """HT combine's staging copy is sized by the tensor it is handed: the whole ladder-max
        receive plane put a rung-independent floor under it. LL keeps the full padded plane."""
        b = backend(ll=False)
        h = b._ensure_handle(problem(1))
        # 7 is what the stubbed `torch.zeros(...).item()` reports as the received count.
        self.assertEqual(h.count, 7)
        self.assertEqual(h.combine_in_t, list(range(7)))
        self.assertLess(len(h.combine_in_t), len(b._recv_x))

class OracleCombineAdapters(unittest.TestCase):
    def test_legacy_oracle_preserves_each_normal_transport_shape(self):
        import importlib
        import torch

        _legacy_module()
        vendor = types.ModuleType("deep_ep")
        vendor.Buffer = vendor.Config = vendor.ElasticBuffer = type("Buffer", (), {})
        transformed = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
        for name, class_name, expected in (
            ("ep_deepep_v2", "DeepEPV2Backend", [[1, 2], [3, 4], [0, 0]]),
            ("ep_uccl", "UCCLEPBackend", [[1, 2], [3, 4]]),
        ):
            with self.subTest(backend=name), mock.patch.dict(sys.modules, {"deep_ep": vendor}):
                module = importlib.import_module(name)
                adapter = getattr(module, class_name).__new__(getattr(module, class_name))
                adapter.mode, adapter._fp8 = "normal", False
                adapter.num_sms, adapter.num_qps, adapter.combine_config = 8, 3, object()

                def combine(*args, **kwargs):
                    values = args[0] if args else kwargs["x"]
                    return values.clone(), None, None

                adapter.buffer = types.SimpleNamespace(combine=combine)
                handle = types.SimpleNamespace(
                    recv_x=torch.full((3, 2), 99, dtype=torch.bfloat16), handle=object(),
                )
                with mock.patch.object(module, "torch", torch):
                    output = adapter.combine_transformed(object(), handle, transformed)
                self.assertEqual(output.tolist(), expected)
                self.assertEqual(output.dtype, torch.bfloat16)

    def test_nccl_oracle_keeps_the_workspace_and_completion_contract(self):
        import torch

        for low_latency in (False, True):
            with self.subTest(low_latency=low_latency):
                adapter = ep_nccl.NCCLEPBackend.__new__(ep_nccl.NCCLEPBackend)
                adapter._ll, adapter._combine_cfg = low_latency, object()
                adapter._stream, adapter._t = lambda: 0, lambda tensor: tensor
                adapter._window_t, adapter.zero_copy = adapter._t, True
                shape = (2, 2, 2) if low_latency else (3, 2)
                adapter._recv_x = torch.full(shape, 99, dtype=torch.bfloat16)
                output = torch.empty((2, 2), dtype=torch.bfloat16)
                completed = []

                def combine(inputs, outputs, **kwargs):
                    values = inputs.tokens.sum(dim=0) if low_latency else inputs.tokens
                    outputs.tokens.copy_(values)

                handle = types.SimpleNamespace(
                    source_rank=torch.tensor([0, 1]), source_slot=torch.tensor([0, 1]),
                    combine_in_t=adapter._recv_x[:2],
                    out=output, out_t=output,
                    handle=types.SimpleNamespace(
                        combine=combine, complete=lambda **kwargs: completed.append(True),
                    ),
                )
                with mock.patch.object(ep_nccl, "torch", torch), \
                        mock.patch.object(ep_nccl, "CombineInputs", types.SimpleNamespace), \
                        mock.patch.object(ep_nccl, "CombineOutputs", types.SimpleNamespace):
                    result = adapter.combine_transformed(
                        types.SimpleNamespace(T=2), handle, torch.tensor([[1, 2], [3, 4]]),
                    )
                self.assertEqual(result.tolist(), [[1, 2], [3, 4]])
                self.assertEqual(completed, [])  # SEND|RECV self-completes in the merged NCCL API.
                unused = adapter._recv_x[0, 1] if low_latency else adapter._recv_x[2]
                self.assertEqual(unused.tolist(), [0, 0])


def _legacy_module():
    """BF16 adapter tests need tensor arithmetic, but do not compile a GPU dequantizer."""
    import torch

    with mock.patch.object(torch, "compile", return_value=lambda function: function):
        import ep_legacy
    return ep_legacy


class LegacyBufferContract(unittest.TestCase):
    def test_padded_receive_is_compacted_and_scattered_to_its_original_slots(self):
        import torch
        ep_legacy = _legacy_module()

        backend = ep_legacy.LegacyBufferOperations()
        backend._fp8 = False
        backend.rank, backend.num_local_experts = 2, 2
        backend.max_tokens = 3
        backend.args = types.SimpleNamespace(experts=8)
        received = torch.tensor(
            [[[1, 2], [99, 99], [99, 99]], [[3, 4], [5, 6], [99, 99]]],
            dtype=torch.bfloat16,
        )
        counts = torch.tensor([1, 2])
        combined_inputs = []

        def combine(values, indices, weights, handle):
            combined_inputs.append(values.clone())
            return values.sum(dim=0), None, None

        backend.buffer = types.SimpleNamespace(
            low_latency_dispatch=lambda *args, **kwargs: (received, counts, object(), None, None),
            low_latency_combine=combine,
        )
        point = types.SimpleNamespace(T=2, dispatch_x=None, topk_idx=None, topk_weights=None)
        with mock.patch.object(ep_legacy, "torch", torch):
            handle = backend._ll_dispatch(point)
            view = backend._ll_inspect_dispatch(point, handle)
            self.assertEqual(view.payload.tolist(), [[1, 2], [3, 4], [5, 6]])
            self.assertEqual(view.expert_ids.tolist(), [4, 5, 5])
            transformed = torch.tensor([[10, 20], [30, 40], [50, 60]])
            output = backend._ll_combine_transformed(point, handle, transformed)
        self.assertEqual(output.tolist(), [[40, 60], [50, 60]])
        self.assertEqual(
            combined_inputs[0].tolist(),
            [[[10, 20], [0, 0], [0, 0]], [[30, 40], [50, 60], [0, 0]]],
        )

    def test_teardown_reports_a_destroy_failure(self):
        ep_legacy = _legacy_module()

        backend = ep_legacy.LegacyBufferOperations()
        backend.buffer = types.SimpleNamespace(destroy=mock.Mock(side_effect=OSError("busy")))
        distributed = types.SimpleNamespace(barrier=lambda: None)
        with mock.patch.object(ep_legacy, "dist", distributed):
            self.assertEqual(backend.finalize(0), 1)


if __name__ == "__main__":
    unittest.main()
