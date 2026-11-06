"""nccl-ep runs TensorRT-LLM's NcclEP configuration where an engine selects it and labels every other
row as the library's: what reaches nccl.ep, and what the row says it ran. CPU-only fakes."""
from __future__ import annotations

import json
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "tests")]

from test_backends import _import_stubbed, args  # noqa: E402

VERSION = "0.2.0.dev20261013"
DEVICE_SMS = 132


class FakeHandle:
    """Records the order of the handle calls a dispatch makes and the outputs it was handed."""

    def __init__(self):
        self.calls = []
        self.outputs = None

    def update(self, topk_idx, *, layout_info=None, stream=None):
        self.calls.append("update")

    def dispatch(self, inputs, outputs, *, layout_info=None, config=None, stream=None):
        self.calls.append("dispatch")
        self.outputs = outputs


def _module():
    """A fresh adapter import whose vendor surface records what the adapter passes it."""
    module = _import_stubbed("ep_nccl")
    module.metadata = types.SimpleNamespace(version=lambda name: VERSION)
    module.torch.float8_e4m3fn = "float8_e4m3fn"
    module.torch.cuda = types.SimpleNamespace(
        synchronize=lambda: None,
        get_device_properties=lambda device: types.SimpleNamespace(multi_processor_count=DEVICE_SMS),
    )
    module.nccl_ep.DispatchQuantizationRecipe = types.SimpleNamespace(DS_FP8E3M4="DS_FP8E3M4")
    module.recorded = {"group": [], "dispatch": []}
    module.GroupConfig = lambda **fields: module.recorded["group"].append(fields) or fields
    module.DispatchConfig = lambda **fields: module.recorded["dispatch"].append(fields) or fields
    module.DispatchOutputs = lambda **fields: types.SimpleNamespace(**fields)
    return module


def _backend(mode="low-latency", world_size=8, phase="decode", precision="bf16", capacity=512, env=None,
             scale_up_domain=8):
    """(module, backend, handle, details) for one case built and sized as the harness drives it, under
    `env` alone (graphs on unless it sets COLLX_CUDA_GRAPH=0). `scale_up_domain=72` is a GB200/GB300
    MNNVL rack, where EP16 spans four 4-GPU nodes inside one scale-up domain."""
    module = _module()
    options = types.SimpleNamespace(
        mode=mode, precision=precision, phase=phase, experts=384, hidden=7168, topk=6,
        scale_up_domain=scale_up_domain,
    )
    handle = FakeHandle()
    group = types.SimpleNamespace(create_handle=lambda *a, **k: handle, destroy=lambda: None)
    module.nccl_ep.Group = types.SimpleNamespace(create=lambda comm, config: group)
    with mock.patch.dict(os.environ, env or {}, clear=True):
        backend = module.NCCLEPBackend(options, 0, world_size, 0, "cuda:0")
        backend._bootstrap_comm = lambda: None
        window = types.SimpleNamespace(is_valid=True, close=lambda: None)
        backend._comm = types.SimpleNamespace(register_window=mock.Mock(return_value=window))
        backend.create_buffer(types.SimpleNamespace(max_tokens_per_rank=capacity))
        details = backend.implementation_details()
    backend._stream = lambda: 0
    return module, backend, handle, details


def _dispatch(backend, steps=1):
    """`steps` timed dispatches of one ladder point, as the harness replays a bound problem."""
    problem = types.SimpleNamespace(T=4, dispatch_x="tokens", topk_idx="indices", topk_weights="weights")
    for _ in range(steps):
        backend.dispatch(problem)


class LowLatencyZeroCopies(unittest.TestCase):
    """Inside one NVLink domain LL rank-major dispatches into the registered window, graphed or
    eager, and `-zc` names what ran."""

    def test_rank_major_dispatches_into_the_zero_copy_window_in_both_regimes(self):
        for graph in ("1", "0"):
            with self.subTest(graph=graph):
                module, backend, handle, details = _backend(env={"COLLX_CUDA_GRAPH": graph})
                _dispatch(backend)

                self.assertTrue(backend.zero_copy)
                self.assertEqual(module.recorded["group"][0]["zero_copy"], module.ZeroCopyMode.ON)
                backend._comm.register_window.assert_called_once()
                self.assertIs(handle.outputs.tokens.window, backend._recv_window)
                self.assertTrue(details["zero_copy"])
                self.assertEqual(backend.kernel_generation, f"nccl-ep-v02-ll-rm-zc-cap512-{VERSION}")

    def test_ep16_inside_an_mnnvl_domain_is_not_scale_out(self):
        # GB200/GB300 EP16 spans four nodes but one 72-GPU MNNVL domain: the window path applies.
        for graph in ("1", "0"):
            with self.subTest(graph=graph):
                module, backend, _, details = _backend(
                    world_size=16, scale_up_domain=72, env={"COLLX_CUDA_GRAPH": graph},
                )
                self.assertFalse(backend._internode)
                self.assertTrue(backend.zero_copy)
                self.assertEqual(module.recorded["group"][0]["zero_copy"], module.ZeroCopyMode.ON)
                self.assertEqual(backend.kernel_generation, f"nccl-ep-v02-ll-rm-zc-cap512-{VERSION}")

    def test_policy_keeps_excluded_paths_staged_in_both_regimes(self):
        cases = {
            "scale-out": {"world_size": 16},
            "expert-major": {"env": {"COLLX_NCCL_LL_LAYOUT": "expert-major"}},
            "opt-out": {"env": {"COLLX_NCCL_LL_ZERO_COPY": "0"}},
            "fp8-scale-out": {"world_size": 16, "precision": "fp8"},
            "fp8-expert-major": {"precision": "fp8",
                                 "env": {"COLLX_NCCL_LL_LAYOUT": "expert-major"}},
        }
        for name, case in cases.items():
            for graph in ("1", "0"):
                with self.subTest(case=name, graph=graph):
                    env = {**case.get("env", {}), "COLLX_CUDA_GRAPH": graph}
                    module, backend, _, details = _backend(
                        world_size=case.get("world_size", 8), precision=case.get("precision", "bf16"), env=env,
                    )
                    self.assertFalse(backend.zero_copy)
                    self.assertEqual(details["ll_zero_copy_opt_out"], name == "opt-out")
                    self.assertFalse(details["zero_copy"])
                    self.assertEqual(module.recorded["group"][0]["zero_copy"], module.ZeroCopyMode.OFF)
                    self.assertNotIn("-zc", backend.kernel_generation)

    def test_fp8_dispatch_uses_windows_in_both_phases_and_regimes(self):
        for phase in ("prefill", "decode"):
            for graph in ("0", "1"):
                with self.subTest(phase=phase, graph=graph):
                    module, backend, handle, details = _backend(
                        phase=phase, precision="fp8", env={"COLLX_CUDA_GRAPH": graph},
                        capacity=8192 if phase == "prefill" else 512,
                    )
                    _dispatch(backend)
                    expected = True
                    self.assertEqual(details["zero_copy"], expected)
                    self.assertEqual(module.recorded["group"][0]["zero_copy"],
                                     module.ZeroCopyMode.ON if expected else module.ZeroCopyMode.OFF)
                    self.assertEqual(backend._comm.register_window.call_count, 2 if expected else 0)
                    self.assertIs(handle.outputs.tokens.window,
                                  backend._recv_window if expected else None)
                    self.assertIs(handle.outputs.scales.window,
                                  backend._recv_scales_window if expected else None)
                    self.assertIs(backend._combine_x_t.buffer, backend._combine_scratch)
                    self.assertIsNone(backend._combine_x_t.window)
                    self.assertEqual("-zc" in backend.kernel_generation, expected)

    def test_group_is_sized_and_configured_like_trtllm(self):
        # TensorRT-LLM: GroupConfig(LOW_LATENCY, num_experts, max_tokens_per_rank,
        # ep * max_tokens_per_rank, max_token_bytes) with every SM budget left to the library
        # (nccl_ep_utils.py:251-257), and DispatchConfig(round_scales=0, recipe NONE) (365-374).
        for capacity in (64, 512):
            with self.subTest(capacity=capacity):
                module, backend, _, details = _backend(capacity=capacity)
                fields = module.recorded["group"][0]
                self.assertEqual(
                    {key: value for key, value in fields.items() if key != "zero_copy"},
                    {
                        "algorithm": module.Algorithm.LOW_LATENCY,
                        "num_experts": 384,
                        "max_dispatch_tokens_per_rank": capacity,
                        "max_recv_tokens_per_rank": 8 * capacity,
                        "max_token_bytes": 7168 * 2,
                    },
                )
                self.assertEqual(module.recorded["dispatch"], [{"send_only": 0, "round_scales": 0}])
                self.assertEqual(details["capacity_tokens_per_rank"], capacity)
                self.assertEqual(details["num_sms"], {"dispatch": DEVICE_SMS, "combine": DEVICE_SMS})
                self.assertIn(f"-cap{capacity}-", backend.kernel_generation)


class HighThroughputIsALibraryRow(unittest.TestCase):
    """No engine runs HT, so its capacity-sized routing update stays in dispatch and the capacity
    that sizes it is part of every row's identity."""

    def test_capacity_is_recorded_for_each_phase(self):
        expected = {
            ("decode", "1"): f"nccl-ep-v02-ht-routed-zc-static-gum1-cap512-{VERSION}",
            ("decode", "0"): f"nccl-ep-v02-ht-routed-zc-static-cap512-{VERSION}",
            ("prefill", "1"): f"nccl-ep-v02-ht-routed-zc-static-cap8192-{VERSION}",
        }
        for (phase, graph), generation in expected.items():
            with self.subTest(phase=phase, graph=graph):
                capacity = 512 if phase == "decode" else 8192
                module, backend, _, details = _backend(
                    mode="normal", phase=phase, capacity=capacity, env={"COLLX_CUDA_GRAPH": graph},
                )
                self.assertEqual(backend.kernel_generation, generation)
                self.assertEqual(module.recorded["group"][0]["max_dispatch_tokens_per_rank"], capacity)
                self.assertEqual(module.recorded["group"][0]["zero_copy"], module.ZeroCopyMode.ON)
                self.assertEqual(details["capacity_tokens_per_rank"], capacity)
                self.assertTrue(details["zero_copy"])
                self.assertEqual(details["num_sms"], {"dispatch": 16, "combine": 16})

    def test_every_dispatch_carries_the_routing_update(self):
        _, backend, handle, _ = _backend(mode="normal")
        _dispatch(backend, steps=2)
        self.assertEqual(handle.calls, ["update", "dispatch", "update", "dispatch"])

    def test_low_latency_dispatch_carries_no_update(self):
        # ncclEpUpdateHandle returns before any device work in LL (nccl_ep.cc:4014-4020).
        _, backend, handle, _ = _backend()
        _dispatch(backend, steps=2)
        self.assertEqual(handle.calls, ["dispatch", "dispatch"])


class RowLabels(unittest.TestCase):
    def test_only_the_trtllm_configuration_is_production(self):
        eager = {"COLLX_CUDA_GRAPH": "0"}
        staged = {"COLLX_NCCL_LL_ZERO_COPY": "0"}
        cases = {
            # With TensorRT-LLM#20631 it dispatches into a registered window in both regimes: an
            # in-domain row is its configuration when it zero-copies, a scale-out row stages.
            "ll-rank-major-graphed-zc": ({}, "production", "tensorrt-llm"),
            "ll-rank-major-eager-zc": ({"env": eager}, "production", "tensorrt-llm"),
            "ll-mnnvl-ep16-graphed-zc": (
                {"world_size": 16, "scale_up_domain": 72}, "production", "tensorrt-llm"),
            "ll-mnnvl-ep16-eager-zc": (
                {"world_size": 16, "scale_up_domain": 72, "env": eager}, "production", "tensorrt-llm"),
            "ll-scale-out-graphed": ({"world_size": 16}, "production", "tensorrt-llm"),
            "ll-scale-out-eager": ({"world_size": 16, "env": eager}, "production", "tensorrt-llm"),
            "ll-opt-out-graphed": ({"env": staged}, "candidate", None),
            "ll-opt-out-eager": ({"env": {**staged, **eager}}, "candidate", None),
            "ll-opt-out-scale-out": ({"world_size": 16, "env": staged}, "production", "tensorrt-llm"),
            "ll-expert-major": ({"env": {"COLLX_NCCL_LL_LAYOUT": "expert-major"}}, "candidate", None),
            "ll-ds-fp8": ({"precision": "fp8"}, "candidate", None),
            "ht": ({"mode": "normal"}, "candidate", None),
            "ht-opt-out-ignored": ({"mode": "normal", "env": staged}, "candidate", None),
        }
        for name, (case, maturity, engine) in cases.items():
            with self.subTest(case=name):
                _, backend, _, details = _backend(**case)
                self.assertEqual(backend.maturity, maturity)
                self.assertEqual(details["engine_contract"], engine)

    def test_the_opt_out_stages_and_is_a_library_row(self):
        module, backend, handle, details = _backend(env={"COLLX_NCCL_LL_ZERO_COPY": "0"})
        _dispatch(backend)
        self.assertEqual(module.recorded["group"][0]["zero_copy"], module.ZeroCopyMode.OFF)
        backend._comm.register_window.assert_not_called()
        self.assertEqual(backend.kernel_generation, f"nccl-ep-v02-ll-rm-cap512-{VERSION}")
        self.assertTrue(details["ll_zero_copy_opt_out"])
        self.assertIsNone(details["engine_contract"])

    def test_the_opt_out_rejects_other_values(self):
        with self.assertRaisesRegex(ValueError, "COLLX_NCCL_LL_ZERO_COPY"):
            _backend(env={"COLLX_NCCL_LL_ZERO_COPY": "off"})

    def test_generation_and_library_version_name_the_installed_wheel(self):
        _, backend, _, _ = _backend(precision="fp8")
        self.assertEqual(backend.library_version, VERSION)
        self.assertEqual(backend.kernel_generation, f"nccl-ep-v02-ll-rm-zc-fp8-ds-e3m4-cap512-{VERSION}")

    def test_library_config_is_one_json_object(self):
        # Published as `implementation.library_config`; the wheel stays in `library_version`.
        _, _, _, details = _backend(env={"COLLX_CUDA_GRAPH": "0"})
        self.assertEqual(json.loads(json.dumps(details)), {
            "capacity_tokens_per_rank": 512,
            "zero_copy": True,
            "num_sms": {"dispatch": DEVICE_SMS, "combine": DEVICE_SMS},
            "ll_zero_copy_opt_out": False,
            "engine_contract": "tensorrt-llm",
        })


class SmBudgets(unittest.TestCase):
    """The budgets ncclEpCreateGroup resolves (nccl_ep.cc:2219-2276), so an override is visible."""

    def test_overrides_apply_in_the_library_order(self):
        resolve = _module()._resolve_num_sms
        table = (
            ({}, {"dispatch": 16, "combine": 16}),
            ({"NCCL_EP_COMM_SMS": "20"}, {"dispatch": 20, "combine": 20}),
            ({"NCCL_EP_COMM_SMS": "20", "NCCL_EP_DISPATCH_SMS": "24"}, {"dispatch": 24, "combine": 20}),
            # Out of [1, device SMs], empty or non-numeric: the library warns and keeps the value.
            ({"NCCL_EP_COMBINE_SMS": "999"}, {"dispatch": 16, "combine": 16}),
            ({"NCCL_EP_COMM_SMS": "", "NCCL_EP_DISPATCH_SMS": "many"}, {"dispatch": 16, "combine": 16}),
            ({"NCCL_EP_COMBINE_SMS": "0"}, {"dispatch": 16, "combine": 16}),
        )
        for env, expected in table:
            with self.subTest(env=env):
                self.assertEqual(resolve(16, DEVICE_SMS, env), expected)

    def test_the_artifact_reads_the_environment_the_group_was_created_under(self):
        _, _, _, details = _backend(env={"NCCL_EP_DISPATCH_SMS": "64"})
        self.assertEqual(details["num_sms"], {"dispatch": 64, "combine": DEVICE_SMS})


class NcclLowLatencyLadderSizing(unittest.TestCase):
    """LL keeps every requested point and allocates receive space for the full ladder."""

    def _module(self):
        return _import_stubbed("ep_nccl")

    @staticmethod
    def _construct(module, world_size, mode="low-latency", **env):
        """NCCLEPBackend.__init__ for a BF16 case under `env`."""
        options = types.SimpleNamespace(
            mode=mode, precision="bf16", experts=384, hidden=7168, topk=6, scale_up_domain=8
        )
        with mock.patch.dict(os.environ, env):
            return module.NCCLEPBackend(options, 0, world_size, 0, "cuda:0")

    def _backend(self, module, low_latency):
        """A backend far enough along to run create_buffer against the stubs."""
        backend = module.NCCLEPBackend.__new__(module.NCCLEPBackend)
        backend._ll = low_latency
        backend.zero_copy = True
        backend.world_size, backend.num_local_experts, backend.device = 8, 4, "cuda:0"
        backend.args = types.SimpleNamespace(hidden=7168, experts=256, topk=8, phase="decode")
        backend._algorithm = "LL" if low_latency else "HT"
        backend._layout = module.Layout.RANK_MAJOR if low_latency else module.Layout.FLAT
        backend._bootstrap_comm = lambda: None
        # create_buffer stamps the installed wheel into kernel_generation.
        module.metadata = types.SimpleNamespace(version=lambda name: "0.2.0.dev20261013")
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

    def test_the_communicator_runs_graph_usage_mode_one(self):
        module = self._module()
        with mock.patch.object(module.nccl_core, "NCCLConfig", lambda **kw: kw, create=True):
            self.assertEqual(module.NCCLEPBackend._comm_config(), {"graph_usage_mode": 1})

    def test_ll_layout_selector_restores_the_expert_major_contract(self):
        module = self._module()
        em = self._construct(module, 8, COLLX_NCCL_LL_LAYOUT="expert-major")
        rm = self._construct(module, 8)
        ht = self._construct(module, 16, mode="normal")
        with self.assertRaisesRegex(ValueError, "COLLX_NCCL_LL_LAYOUT"):
            self._construct(module, 8, COLLX_NCCL_LL_LAYOUT="flat")

        self.assertEqual(em._layout, module.Layout.EXPERT_MAJOR)
        self.assertEqual(
            (em.kernel_generation, em.receive_layout, em.combine_weight_semantics),
            ("nccl-ep-v02-ll-em", "token-expert", "weighted-kernel-sum"),
        )
        self.assertFalse(em.zero_copy)
        self.assertEqual(rm._layout, module.Layout.RANK_MAJOR)
        # Only LL rank-major selects NCCL's FP32 reduction.
        self.assertEqual((rm.combine_reduction, em.combine_reduction, ht.combine_reduction),
                         ("rank-fp32", "domain-fp32", "domain-fp32"))

    def test_ll_ladder_keeps_512_and_larger_points(self):
        module = self._module()
        backend = self._backend(module, low_latency=True)
        backend.mode = "low-latency"
        backend.args.tokens_ladder = "32 256 512 1024"
        backend._build_rank_inputs = mock.Mock(return_value=None)
        spec = backend.make_inputs(backend.args)
        self.assertEqual(spec.ladder, [32, 256, 512, 1024])
        self.assertEqual(spec.dropped, [])

    def test_fp8_ll_accepts_both_phases_and_keeps_ht_disabled(self):
        module = self._module()
        module.nccl_ep.DispatchQuantizationRecipe = types.SimpleNamespace(DS_FP8E3M4="ds-fp8")
        module.DispatchConfig = lambda **kw: types.SimpleNamespace(**kw)
        with mock.patch.dict(os.environ, {"COLLX_NCCL_LL_LAYOUT": "rank-major"}):
            for phase in ("decode", "prefill"):
                with self.subTest(phase=phase):
                    options = args(phase=phase, precision="fp8", mode="low-latency",
                                   hidden=7168, experts=384, topk=6, scale_up_domain=72)
                    backend = module.NCCLEPBackend(options, 0, 8, 0, "cuda:0")
                    self.assertEqual(backend._dispatch_cfg.quantization_recipe, "ds-fp8")
                    self.assertEqual(backend.dispatch_dtype, "fp8-e4m3fn")
                    self.assertTrue(backend.zero_copy)
                    self.assertEqual(backend.cuda_graph_supported, phase == "decode")
                    options.mode = "normal"
                    with self.assertRaisesRegex(ValueError, "low-latency mode only"):
                        module.NCCLEPBackend(options, 0, 8, 0, "cuda:0")

    def test_ll_prefill_uses_the_full_ladder_receive_without_graphs(self):
        module = self._module()
        backend = self._backend(module, low_latency=True)
        backend.mode = "low-latency"
        backend.args.phase = "prefill"
        self.assertIsNone(backend.buffer_cap(backend.args))
        backend.create_buffer(types.SimpleNamespace(max_tokens_per_rank=8192))
        self.assertEqual(backend._recv_x.shape, (8, 8192, 7168))
        self.assertEqual(backend.created_configs[0].max_dispatch_tokens_per_rank, 8192)
        self.assertFalse(backend.cuda_graph_supported)

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
        self.assertEqual(backend._recv_x.shape, (8, 64, 7168))
        self.assertEqual(backend._recv_count.shape, (8,))
        self.assertIs(backend._recv_x_t.window, backend._recv_window)

    def test_fp8_buffer_binding_follows_selected_policy_in_both_phases(self):
        for phase in ("decode", "prefill"):
            for zero_copy in (False, True):
                with self.subTest(phase=phase, zero_copy=zero_copy):
                    module = self._module()
                    backend = self._backend(module, low_latency=True)
                    backend.args.phase = phase
                    backend._fp8 = True
                    backend.zero_copy = zero_copy
                    windows = [types.SimpleNamespace(is_valid=True, close=mock.Mock())
                               for _ in range(2)]
                    backend._comm.register_window.side_effect = windows
                    backend.create_buffer(types.SimpleNamespace(max_tokens_per_rank=1024))
                    self.assertEqual(backend._recv_x.shape, (8, 1024, 7168))
                    self.assertEqual(backend._recv_scales.shape, (8, 1024, 56))
                    self.assertIs(backend._recv_x_t.window, windows[0] if zero_copy else None)
                    self.assertIs(backend._recv_scales_t.window,
                                  windows[1] if zero_copy else None)
                    self.assertIs(backend._recv_scales_t.buffer, backend._recv_scales)
                    self.assertIs(backend._combine_x_t.buffer, backend._combine_scratch)
                    self.assertIsNone(backend._combine_x_t.window)
                    backend._handle = None
                    backend._bound = None
                    backend._ep_group = types.SimpleNamespace(destroy=lambda: None)
                    backend._comm.destroy = lambda: None
                    module.dist.barrier = lambda: None
                    module.dist.destroy_process_group = lambda: None
                    self.assertEqual(backend.finalize(0), 0)
                    for window in windows:
                        self.assertEqual(window.close.call_count, int(zero_copy))

    def test_fp8_scale_registration_failure_releases_token_window_and_group(self):
        module = self._module()
        backend = self._backend(module, low_latency=True)
        backend._fp8 = True
        windows = [types.SimpleNamespace(is_valid=valid, close=mock.Mock())
                   for valid in (True, False)]
        backend._comm.register_window.side_effect = windows
        group = types.SimpleNamespace(destroy=mock.Mock())
        module.nccl_ep.Group.create = lambda *a, **k: group
        with self.assertRaisesRegex(RuntimeError, "scale window registration failed"):
            backend.create_buffer(types.SimpleNamespace(max_tokens_per_rank=1024))
        for window in windows:
            window.close.assert_called_once()
        group.destroy.assert_called_once()
        self.assertIsNone(backend._recv_window)
        self.assertIsNone(backend._recv_scales_window)
        self.assertIsNone(backend._ep_group)

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
        module.DispatchInputs = DispatchInputs
        module.DispatchOutputs = DispatchOutputs
        module.LayoutInfo = LayoutInfo
        backend._handle = None
        backend._bound = None
        backend._ep_group = types.SimpleNamespace(create_handle=mock.Mock(return_value=handle))
        backend._dispatch_cfg = object()
        backend._stream = lambda: 0

        p = types.SimpleNamespace(T=1, dispatch_x="tokens", topk_idx="indices", topk_weights="weights")
        backend.dispatch(p)

        self.assertIs(backend._layout, module.Layout.RANK_MAJOR)
        self.assertEqual(captured["inputs"]["tokens"].buffer, "tokens")
        self.assertEqual(captured["inputs"]["topk_weights"].buffer, "weights")
        self.assertEqual(captured["outputs"]["tokens"].window, backend._recv_window)
        self.assertIs(captured["outputs"]["topk_weights"], backend._recv_w_t)
        self.assertIs(captured["outputs"]["topk_idx"], backend._recv_idx_t)
        self.assertIs(captured["layout"]["src_rank_counters"], backend._recv_count_t)
        handle.dispatch.assert_called_once()


_ep_nccl = _import_stubbed("ep_nccl")


class RebindRecordingHandle:
    """Records every rebind so the tests can assert on the collective call pattern."""

    def __init__(self):
        self.updates = []

    def update(self, topk_idx, *, layout_info=None, stream=None):
        self.updates.append((topk_idx, layout_info))


class RebindRecordingGroup:
    def __init__(self):
        self.created = 0
        self.handle = RebindRecordingHandle()

    def create_handle(self, layout, topk_idx, *, layout_info=None, config=None, stream=None):
        self.created += 1
        return self.handle


def _handle_backend(ll=True):
    """An NCCLEPBackend with just the fields _ensure_handle touches (no __init__, no GPU)."""
    b = object.__new__(_ep_nccl.NCCLEPBackend)
    b._ll = ll
    b._layout = "EM" if ll else "FLAT"
    b._handle = None
    b._bound = None
    b._ep_group = RebindRecordingGroup()
    b.device = "cuda:0"
    b.num_local_experts = 4
    b.args = types.SimpleNamespace(hidden=16)
    b._t = lambda x: x
    b._window_t = lambda x: x
    b._stream = lambda: 0
    # create_buffer always runs before the first _ensure_handle, so the HT receive plane exists
    # by then; a list stands in for the tensor because `_t` is identity here.
    b._recv_x = list(range(64))
    b._recv_x_t = ("window", "full-plane")
    return b


def _ladder_problem(T):
    return types.SimpleNamespace(
        T=T, dispatch_x=f"x{T}", topk_idx=f"idx{T}", topk_weights=f"w{T}"
    )


class TestSingleHandle(unittest.TestCase):
    def test_one_handle_across_many_shapes(self):
        """Nine ladder rungs produce exactly one create_handle, rebound to each later routing."""
        b = _handle_backend()
        for T in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            b._ensure_handle(_ladder_problem(T))
        self.assertEqual(b._ep_group.created, 1)
        self.assertEqual([idx for idx, _ in b._ep_group.handle.updates],
                         [f"idx{T}" for T in (2, 4, 8, 16, 32, 64, 128, 256)])

    def test_ll_rank_major_weight_wrapper_is_built_once_per_handle(self):
        """LL caches its dispatch-weight wrapper with the handle."""
        ll = _handle_backend(ll=True)
        ll._t = mock.Mock(side_effect=lambda value: types.SimpleNamespace(value=value))
        pa = _ladder_problem(1)
        h = ll._ensure_handle(pa)
        first_weights = h.in_weights_t
        self.assertEqual(first_weights.value, "w1")
        # Re-entering the same problem reuses the handle and therefore the wrapper.
        ll._t.reset_mock()
        self.assertIs(ll._ensure_handle(pa).in_weights_t, first_weights)
        ll._t.assert_not_called()

    def test_ht_combine_input_is_the_full_static_receive_plane(self):
        """The FLAT contract gives combine the dispatch output's static [num_recv_slots, hidden]
        shape (required under graph capture); a count-sized slice needs an AUTO-sized group."""
        b = _handle_backend(ll=False)
        h = b._ensure_handle(_ladder_problem(1))
        # 7 is what the stubbed `torch.zeros(...).item()` reports as the received count.
        self.assertEqual(h.count, 7)
        self.assertIs(h.combine_in_t, b._recv_x_t)


if __name__ == "__main__":
    unittest.main()
