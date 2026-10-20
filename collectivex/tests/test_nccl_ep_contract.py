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

from test_backends import _import_stubbed  # noqa: E402

VERSION = "0.2.0.dev20261005"
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


def _backend(mode="low-latency", world_size=8, phase="decode", precision="bf16", capacity=512, env=None):
    """(module, backend, handle, details) for one case built and sized as the harness drives it, under
    `env` alone (graphs on unless it sets COLLX_CUDA_GRAPH=0)."""
    module = _module()
    options = types.SimpleNamespace(
        mode=mode, precision=precision, phase=phase, experts=384, hidden=7168, topk=6, scale_up_domain=8,
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


class LowLatencyFollowsTensorRTLLM(unittest.TestCase):
    """TensorRT-LLM dispatches into a registered window only when it is not capturing
    (nccl_ep.py:305-327): graphed LL stages, eager LL zero-copies, and `-zc` names what ran."""

    def test_graphed_rank_major_dispatches_into_a_plain_tensor(self):
        module, backend, handle, details = _backend()
        _dispatch(backend)

        self.assertFalse(backend.zero_copy)
        self.assertEqual(module.recorded["group"][0]["zero_copy"], module.ZeroCopyMode.OFF)
        backend._comm.register_window.assert_not_called()
        self.assertIsNone(handle.outputs.tokens.window)
        self.assertFalse(details["zero_copy"])
        self.assertEqual(backend.kernel_generation, f"nccl-ep-v02-ll-rm-cap512-{VERSION}")

    def test_eager_rank_major_keeps_the_zero_copy_window(self):
        module, backend, handle, details = _backend(env={"COLLX_CUDA_GRAPH": "0"})
        _dispatch(backend)

        self.assertTrue(backend.zero_copy)
        self.assertEqual(module.recorded["group"][0]["zero_copy"], module.ZeroCopyMode.ON)
        backend._comm.register_window.assert_called_once()
        self.assertIs(handle.outputs.tokens.window, backend._recv_window)
        self.assertTrue(details["zero_copy"])
        self.assertEqual(backend.kernel_generation, f"nccl-ep-v02-ll-rm-zc-cap512-{VERSION}")

    def test_paths_the_library_cannot_zero_copy_stage_in_both_regimes(self):
        cases = {
            "scale-out": {"world_size": 16},
            "expert-major": {"env": {"COLLX_NCCL_LL_LAYOUT": "expert-major"}},
            "ds-fp8": {"precision": "fp8"},
        }
        for name, case in cases.items():
            for graph in ("1", "0"):
                with self.subTest(case=name, graph=graph):
                    env = {**case.get("env", {}), "COLLX_CUDA_GRAPH": graph}
                    module, backend, _, details = _backend(
                        world_size=case.get("world_size", 8), precision=case.get("precision", "bf16"), env=env,
                    )
                    self.assertFalse(backend.zero_copy)
                    self.assertFalse(details["zero_copy"])
                    self.assertEqual(module.recorded["group"][0]["zero_copy"], module.ZeroCopyMode.OFF)
                    self.assertNotIn("-zc", backend.kernel_generation)

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
        cases = {
            "ll-rank-major": ({}, "production", "tensorrt-llm"),
            "ll-rank-major-eager": ({"env": {"COLLX_CUDA_GRAPH": "0"}}, "production", "tensorrt-llm"),
            "ll-scale-out": ({"world_size": 16}, "production", "tensorrt-llm"),
            "ll-expert-major": ({"env": {"COLLX_NCCL_LL_LAYOUT": "expert-major"}}, "candidate", None),
            "ll-ds-fp8": ({"precision": "fp8"}, "candidate", None),
            "ht": ({"mode": "normal"}, "candidate", None),
        }
        for name, (case, maturity, engine) in cases.items():
            with self.subTest(case=name):
                _, backend, _, details = _backend(**case)
                self.assertEqual(backend.maturity, maturity)
                self.assertEqual(details["engine_contract"], engine)

    def test_generation_and_library_version_name_the_installed_wheel(self):
        _, backend, _, _ = _backend(precision="fp8")
        self.assertEqual(backend.library_version, VERSION)
        self.assertEqual(backend.kernel_generation, f"nccl-ep-v02-ll-rm-fp8-ds-e3m4-cap512-{VERSION}")

    def test_library_config_is_one_json_object(self):
        # Published as `implementation.library_config`; the wheel stays in `library_version`.
        _, _, _, details = _backend()
        self.assertEqual(json.loads(json.dumps(details)), {
            "capacity_tokens_per_rank": 512,
            "zero_copy": False,
            "num_sms": {"dispatch": DEVICE_SMS, "combine": DEVICE_SMS},
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


if __name__ == "__main__":
    unittest.main()
