#!/usr/bin/env python3
"""flashinfer-ep runs vLLM's FlashInfer one-sided integration (`FlashInferNVLinkOneSidedManager`
and its prepare/finalize), checked against CPU fakes: what the adapter hands the library."""
from __future__ import annotations

import dataclasses
import enum
import importlib
import os
import sys
import types
import unittest
from pathlib import Path
from typing import Any, Optional
from unittest import mock

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "bench")]

import ep_harness  # noqa: E402

HIDDEN, TOPK, EP, T = 256, 2, 4, 3


class FakeTensor:
    """Records the calls the adapter makes on it; nothing is computed."""

    def __init__(self, shape, dtype="bfloat16", name="tensor", device="cuda:0"):
        self.shape, self.dtype, self.name, self.device = tuple(shape), dtype, name, device
        self.writes, self.copied_from = [], []

    def view(self, *shape):
        return FakeView(self, shape)

    def to(self, dtype):
        return self if dtype == self.dtype else FakeTensor(self.shape, dtype, f"{self.name}.to({dtype})")

    def copy_(self, source):
        self.copied_from.append(source)
        return self


class FakeView:
    def __init__(self, base, shape):
        self.base, self.shape = base, shape

    def __getitem__(self, index):
        return FakeTensor((), self.base.dtype, f"{self.base.name}.view{self.shape}[{index}]")

    def __setitem__(self, index, value):
        self.base.writes.append((self.shape, index, value))


def fake_torch(allocations):
    torch = types.ModuleType("torch")
    for dtype in ("bfloat16", "float32", "int32", "int64", "float8_e4m3fn"):
        setattr(torch, dtype, dtype)

    def allocate(kind):
        def make(shape, dtype=None, device=None):
            tensor = FakeTensor(shape, dtype, kind, device)
            allocations.append(tensor)
            return tensor
        return make

    torch.empty, torch.zeros = allocate("empty"), allocate("zeros")
    torch.compile = lambda fn, **_: fn
    torch._dynamo = types.SimpleNamespace(config=types.SimpleNamespace(cache_size_limit=8))
    torch.cuda = types.SimpleNamespace(current_device=lambda: 1)
    dist = types.ModuleType("torch.distributed")
    dist.group = types.SimpleNamespace(WORLD="ep-group")
    dist.barrier = lambda group=None: None
    torch.distributed = dist
    return {"torch": torch, "torch.distributed": dist}


class QuantFormat(enum.Enum):
    BF16 = 0
    DeepSeekFp8 = 3


@dataclasses.dataclass(frozen=True)
class BootstrapConfig:
    world_size: int
    rank: int
    device: Optional[int] = None
    process_group: Any = None


@dataclasses.dataclass(frozen=True)
class MoEEpCommParams:
    num_experts: int
    top_k: int
    max_tokens_per_rank: int
    hidden_size: int
    dtype: Any = None
    dispatch_format: Any = None
    invalid_expert_id: int = -1


@dataclasses.dataclass
class NVLinkOneSidedConfig:
    """0.7.3's fields and defaults."""

    backend_name: str = "nvlink_one_sided"
    extra_payload_bytes_per_token: int = 0
    eplb_stats_num_experts: int = 0
    enable_rank_mask: bool = False
    use_low_precision_combine: bool = False
    comm_backend: Any = None
    cft: Optional[bool] = None
    cft_max_tokens_for_dispatch: int = 128
    cft_max_tokens_for_combine: int = 128
    lamport: Optional[bool] = None
    lamport_max_tokens: Optional[int] = None
    timeout_sec: int = 300
    enable_pdl: Optional[bool] = None


# Every fake NVLinkOneSidedAlltoAll built, newest last.
BUILT_A2A = []


class NVLinkOneSidedAlltoAll:
    """The 0.7.3 one-sided communicator's dispatch and combine, plus the zero-copy workspace API
    vLLM's finalize does not use here."""

    def __init__(self, bootstrap, params, config):
        self.constructed = {"bootstrap": bootstrap, "params": params, "config": config}
        self.metainfo = [0, 4096]
        self._index = types.SimpleNamespace(WORKSPACE_SIZE_INDEX=1)
        self.dispatches, self.combines = [], []
        BUILT_A2A.append(self)

    def dispatch(self, hidden_states, topk_ids, topk_weights=None, *, hidden_states_scale=None,
                 runtime_max_tokens_per_rank=None, eplb_local_stats=None, active_rank_mask=None):
        tokens = runtime_max_tokens_per_rank
        self.dispatches.append({"hidden": hidden_states, "ids": topk_ids, "weights": topk_weights,
                                "scale": hidden_states_scale, "tokens": tokens})
        scale = hidden_states_scale
        return types.SimpleNamespace(
            hidden_states=FakeTensor((EP * tokens, hidden_states.shape[1]), hidden_states.dtype, "recv0"),
            hidden_states_scale=None if scale is None else FakeTensor((EP * tokens, scale.shape[1]), scale.dtype,
                                                                      "recv1"),
            topk_ids=FakeTensor((EP * tokens, TOPK), "int32", "recv-ids"),
            topk_weights=FakeTensor((EP * tokens, TOPK), "float32", "recv-weights"),
            tokens_per_rank=tokens,
        )

    def get_combine_input_buffer(self, dtype):
        raise AssertionError("combine is handed a caller-owned payload, not the workspace region")

    def combine(self, expert_output, *, output=None, active_rank_mask=None):
        self.combines.append({"payload": expert_output, "output": output})
        return output


def fake_flashinfer(version, constructions):
    package = types.ModuleType("flashinfer")
    package.__version__ = version
    fused_moe = types.ModuleType("flashinfer.fused_moe")
    fused_moe.QuantFormat = QuantFormat
    moe_ep = types.ModuleType("flashinfer.moe_ep")
    moe_ep.BootstrapConfig, moe_ep.MoEEpCommParams = BootstrapConfig, MoEEpCommParams
    moe_ep.NVLinkOneSidedConfig = NVLinkOneSidedConfig

    def create_communication(bootstrap, params, backend, **options):
        constructions.append({"options": options})
        return NVLinkOneSidedAlltoAll(bootstrap, params, backend)

    moe_ep.create_communication = create_communication
    comm = types.ModuleType("flashinfer.comm")
    mnnvl = types.ModuleType("flashinfer.comm.mnnvl")
    mnnvl.CommBackend = type("CommBackend", (), {})
    package.comm, package.fused_moe, package.moe_ep, comm.mnnvl = comm, fused_moe, moe_ep, mnnvl
    return {"flashinfer": package, "flashinfer.fused_moe": fused_moe, "flashinfer.moe_ep": moe_ep,
            "flashinfer.comm": comm, "flashinfer.comm.mnnvl": mnnvl}


class Harness:
    """One adapter built and buffered against the fakes, with the fakes kept installed."""

    def __init__(self, test, precision="bf16", version="0.7.3", phase="decode"):
        self.allocations, self.constructions = [], []
        modules = {**fake_torch(self.allocations), **fake_flashinfer(version, self.constructions)}
        patcher = mock.patch.dict(sys.modules, modules)
        patcher.start()
        test.addCleanup(patcher.stop)
        sys.modules.pop("ep_flashinfer", None)
        self.module = importlib.import_module("ep_flashinfer")
        test.addCleanup(sys.modules.pop, "ep_flashinfer", None)
        args = types.SimpleNamespace(
            mode="normal", precision=precision, experts=16, hidden=HIDDEN, topk=TOPK, gpus_per_node=4,
            phase=phase, runner="gb200", scale_up_domain=72,
        )
        self.backend = self.module.FlashInferEPBackend(args, 1, EP, 1, "cuda:1")
        BUILT_A2A.clear()
        self.backend.create_buffer(types.SimpleNamespace(max_tokens_per_rank=64))
        self.a2a = BUILT_A2A[-1]

    def problem(self):
        backend = self.backend
        if backend._fp8:
            quantize = lambda x: (FakeTensor(x.shape, "float8_e4m3fn", "values"),  # noqa: E731
                                  FakeTensor((x.shape[0], HIDDEN // 128), "float32", "scales"))
            backend._fp8_codec = (quantize, lambda values, scales: "semantic")
            backend._quant = quantize
        idx = FakeTensor((T, TOPK), "int64", "idx")
        weights = FakeTensor((T, TOPK), "float32", "weights")
        return backend.make_problem(T, idx, weights, FakeTensor((T, HIDDEN), "bfloat16", "x"))


class Bf16Contract(unittest.TestCase):
    def test_stage_hands_off_the_receive_and_combine_stages_it(self):
        rig = Harness(self)
        backend, a2a = rig.backend, rig.a2a
        problem = rig.problem()
        handle = backend.dispatch(problem)
        allocated = len(rig.allocations)
        backend.stage(problem, handle)
        # A pointer: no tensor allocated, nothing written, so nothing to hoist or time.
        self.assertIs(handle.combine_input, handle.recv_x)
        self.assertEqual(len(rig.allocations), allocated)
        self.assertFalse(backend.stage_device_work)
        self.assertFalse(backend.stage_excluded_from_roundtrip)
        out = backend.combine(problem, handle)
        (call,) = a2a.combines
        # vLLM's finalize: the caller-owned payload, combined into the caller's output.
        self.assertIs(call["payload"], handle.recv_x)
        self.assertIs(call["output"], problem.combine_output)
        self.assertIs(out, problem.combine_output)
        self.assertEqual(problem.combine_output.copied_from, [])

    def test_dispatch_payloads_follow_vllm_prepare(self):
        rig = Harness(self)
        problem = rig.problem()
        rig.backend.dispatch(problem)
        (call,) = rig.a2a.dispatches
        self.assertEqual((call["hidden"].name, call["ids"].name, call["weights"].name, call["scale"]),
                         ("x", "idx.to(int32)", "weights", None))
        self.assertEqual(call["tokens"], T)

    def test_output_is_allocated_once_per_rung_outside_any_window(self):
        rig = Harness(self)
        problem = rig.problem()
        self.assertEqual((problem.combine_output.shape, problem.combine_output.dtype), ((T, HIDDEN), "bfloat16"))
        for _ in range(2):
            handle = rig.backend.dispatch(problem)
            rig.backend.stage(problem, handle)
            rig.backend.combine(problem, handle)
        self.assertEqual({id(call["output"]) for call in rig.a2a.combines}, {id(problem.combine_output)})


class Fp8Contract(unittest.TestCase):
    def test_fp8_is_on_the_engine_path_with_vllms_codec_and_order(self):
        rig = Harness(self, precision="fp8")
        backend = rig.backend
        self.assertEqual(backend.dispatch_dtype, "fp8-e4m3fn")
        # per_token_group_quant_fp8(x, 128): 1 byte/value + one FP32 scale per 128 channels, the
        # x_sf_bytes_per_token vLLM's dispatch layout budgets.
        self.assertEqual((backend.dispatch_value_bytes, backend.dispatch_scale_bytes_per_copy),
                         (1, HIDDEN // 128 * 4))
        self.assertIs(rig.a2a.constructed["params"].dispatch_format, QuantFormat.DeepSeekFp8)
        problem = rig.problem()
        backend.dispatch(problem)
        (call,) = rig.a2a.dispatches
        self.assertEqual((call["hidden"].name, call["scale"].name, call["ids"].name, call["weights"].name),
                         ("values", "scales", "idx.to(int32)", "weights"))

    def test_stage_dequantizes_filled_slots_into_a_caller_owned_bf16_payload(self):
        rig = Harness(self, precision="fp8")
        backend = rig.backend
        self.assertTrue(backend.stage_device_work)
        self.assertTrue(backend.stage_excluded_from_roundtrip)  # hoisted like every FP8 stage
        problem = rig.problem()
        problem.flashinfer_filled_slots = "filled"
        dequantized = []
        cast_back = lambda values, scales: dequantized.append((values.name, scales.name)) or "bf16-rows"  # noqa: E731
        handle = backend.dispatch(problem)
        with mock.patch.object(rig.module, "_blockwise_cast_back", cast_back):
            backend.stage(problem, handle)
        payload = handle.combine_input
        self.assertIsNot(payload, handle.recv_x)
        self.assertEqual((payload.name, payload.shape, payload.dtype), ("empty", handle.recv_x.shape, "bfloat16"))
        self.assertEqual(payload.writes, [((-1, HIDDEN), "filled", "bf16-rows")])
        self.assertEqual(dequantized, [(f"recv0.view{(-1, HIDDEN)}[filled]",
                                        f"recv1.view{(-1, HIDDEN // 128)}[filled]")])
        backend.combine(problem, handle)
        (call,) = rig.a2a.combines
        self.assertIs(call["payload"], payload)


class OracleContract(unittest.TestCase):
    def test_transformed_payload_takes_the_timed_call(self):
        rig = Harness(self)
        problem = rig.problem()
        handle = rig.backend.dispatch(problem)
        rig.backend._valid_rows = lambda h: "valid"
        out = rig.backend.combine_transformed(problem, handle, FakeTensor((), "float32", "transformed"))
        payload = rig.allocations[-1]
        self.assertEqual((payload.name, payload.shape, payload.dtype),
                         ("zeros", handle.recv_x.shape, "bfloat16"))
        self.assertEqual([(index, value.name) for _, index, value in payload.writes],
                         [("valid", "transformed.to(bfloat16)")])
        (call,) = rig.a2a.combines
        self.assertIs(call["payload"], payload)
        self.assertIs(out, problem.combine_output)


class Provenance(unittest.TestCase):
    def test_kernel_generation_and_library_version_carry_the_wheel(self):
        for version, slug in (("0.7.3", "0.7.3"), ("0.7.3+cu130", "0.7.3-cu130")):
            with self.subTest(version=version):
                backend = Harness(self, version=version).backend
                self.assertEqual(backend.library_version, version)
                self.assertEqual(backend.kernel_generation, f"flashinfer-mnnvl-one-sided-staged-{slug}")
                self.assertEqual(backend.combine_reduction, "domain-fp32")
                with mock.patch.dict(os.environ, {"COLLX_CUDA_GRAPH": "1"}):
                    self.assertEqual(ep_harness.kernel_generation(backend),
                                     f"flashinfer-mnnvl-one-sided-staged-{slug}-cudagraph-spin-ar")

    def test_the_communicator_is_built_as_vllm_initialize_builds_it(self):
        for precision, dispatch_format in (("bf16", QuantFormat.BF16), ("fp8", QuantFormat.DeepSeekFp8)):
            with self.subTest(precision=precision):
                rig = Harness(self, precision=precision)
                built = rig.a2a.constructed
                self.assertEqual(built["bootstrap"], BootstrapConfig(world_size=EP, rank=1, device=1,
                                                                     process_group="ep-group"))
                self.assertEqual(built["params"], MoEEpCommParams(
                    num_experts=16, top_k=TOPK, max_tokens_per_rank=64, hidden_size=HIDDEN,
                    dtype="bfloat16", dispatch_format=dispatch_format, invalid_expert_id=-1))
                self.assertEqual(built["config"].comm_backend._group, "ep-group")
                # Every transport choice, Lamport and CFT included, is left to FlashInfer.
                self.assertEqual(dataclasses.replace(built["config"], comm_backend=None), NVLinkOneSidedConfig())
                self.assertEqual(rig.constructions, [{"options": {}}])

    def test_implementation_details_record_the_contract(self):
        with mock.patch.dict(os.environ, {"TLLM_MOE_A2A_COMBINE_BLOCK_SIZE": "512", "OTHER": "1"}):
            new = Harness(self).backend.implementation_details()
            fp8 = Harness(self, precision="fp8").backend.implementation_details()
        self.assertEqual(new, {
            "flashinfer_version": "0.7.3",
            "combine_payload": "caller-owned",
            "payload_in_workspace": False,
            "zero_copy": False,
            "combine_output": "output-kw",
            "low_precision_combine": False,
            "dispatch_scale_format": None,
            "pdl_available": True,
            "num_sms": None,
            "num_qps": None,
            "launch_env": {"TLLM_MOE_A2A_COMBINE_BLOCK_SIZE": "512"},
            "capacity_tokens_per_rank": 64,
            "dispatch_bytes_per_token": HIDDEN * 2 + TOPK * 8,
            "combine_bytes_per_token": HIDDEN * 2,
            "workspace_bytes_per_rank": 4096,
        })
        self.assertEqual(
            {key: fp8[key] for key in ("dispatch_scale_format", "dispatch_bytes_per_token")},
            {"dispatch_scale_format": "fp32-per-128-block-row-major",
             "dispatch_bytes_per_token": HIDDEN + HIDDEN // 128 * 4 + TOPK * 8},
        )


if __name__ == "__main__":
    unittest.main()
