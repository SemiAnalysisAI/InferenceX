#!/usr/bin/env python3
"""flashinfer-ep runs vLLM's FlashInfer one-sided integration (`FlashInferNVLinkOneSidedManager`
and its prepare/finalize), checked against CPU fakes: what the adapter hands the library."""
from __future__ import annotations

import importlib
import os
import sys
import types
import unittest
from pathlib import Path
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
    dist = types.ModuleType("torch.distributed")
    dist.group = types.SimpleNamespace(WORLD="ep-group")
    dist.barrier = lambda group=None: None
    torch.distributed = dist
    return {"torch": torch, "torch.distributed": dist}


# Every fake MoeAlltoAll built, newest last.
BUILT_A2A = []


class _A2ABase:
    """MoeAlltoAll's dispatch, plus the zero-copy workspace API vLLM never calls."""

    def __init__(self, mapping, max_num_tokens, top_k, num_experts, workspace_size_per_rank, mnnvl_config):
        self.constructed = {"mapping": mapping, "max_num_tokens": max_num_tokens, "top_k": top_k,
                            "num_experts": num_experts, "workspace_size_per_rank": workspace_size_per_rank}
        self.dispatches, self.combines = [], []
        BUILT_A2A.append(self)

    def dispatch(self, token_selected_experts, input_payloads, runtime_max_tokens_per_rank,
                 invalid_token_expert_id=None, expert_id_payload_index=None):
        self.dispatches.append({
            "experts": token_selected_experts, "payloads": list(input_payloads),
            "tokens": runtime_max_tokens_per_rank, "invalid": invalid_token_expert_id,
            "id_index": expert_id_payload_index,
        })
        return [FakeTensor((EP, runtime_max_tokens_per_rank, payload.shape[1]), payload.dtype, f"recv{i}")
                for i, payload in enumerate(input_payloads)]

    def get_combine_payload_tensor_in_workspace(self, *args, **kwargs):
        raise AssertionError("vLLM never writes the combine payload into the workspace")


class A2AWithOutput(_A2ABase):
    """MoeAlltoAll.combine as of FlashInfer 0.6.16."""

    def combine(self, payload, runtime_max_tokens_per_rank, payload_in_workspace=False, output_dtype=None,
                output_scales=None, output_scalar_scale=1.0, sf_layout=None, output=None, *,
                use_low_precision=False, active_rank_mask=None):
        self.combines.append({"payload": payload, "tokens": runtime_max_tokens_per_rank,
                              "in_workspace": payload_in_workspace, "output": output,
                              "low_precision": use_low_precision})
        return output


class A2ABeforeOutput(_A2ABase):
    """MoeAlltoAll.combine through FlashInfer 0.6.15 (the image's 0.6.8.post1 included)."""

    def combine(self, payload, runtime_max_tokens_per_rank, payload_in_workspace=False):
        result = FakeTensor((T, HIDDEN), "bfloat16", "library-output")
        self.combines.append({"payload": payload, "tokens": runtime_max_tokens_per_rank,
                              "in_workspace": payload_in_workspace, "result": result})
        return result


def fake_flashinfer(version, a2a_class, sizes, pdl=True):
    package = types.ModuleType("flashinfer")
    package.__version__ = version
    comm = types.ModuleType("flashinfer.comm")
    comm.Mapping = lambda *a, **k: ("mapping", a, k)
    mnnvl = types.ModuleType("flashinfer.comm.mnnvl")
    mnnvl.MnnvlConfig = lambda comm_backend: ("mnnvl-config", comm_backend)
    mnnvl.CommBackend = type("CommBackend", (), {})
    a2a = types.ModuleType("flashinfer.comm.trtllm_moe_alltoall")
    a2a.MoeAlltoAll = a2a_class

    def workspace_size(**kwargs):
        sizes.append(kwargs)
        return 4096

    a2a.moe_a2a_get_workspace_size_per_rank = workspace_size
    if pdl:
        a2a.moe_a2a_combine = lambda payload, *a, enable_pdl=None, **k: None
    else:
        a2a.moe_a2a_combine = lambda payload, local_num_tokens, workspace, payload_in_workspace=False: None
    package.comm, comm.mnnvl, comm.trtllm_moe_alltoall = comm, mnnvl, a2a
    return {"flashinfer": package, "flashinfer.comm": comm, "flashinfer.comm.mnnvl": mnnvl,
            "flashinfer.comm.trtllm_moe_alltoall": a2a}


class Harness:
    """One adapter built and buffered against the fakes, with the fakes kept installed."""

    def __init__(self, test, precision="bf16", version="0.6.8.post1", a2a_class=A2ABeforeOutput, pdl=False,
                 phase="decode"):
        self.allocations, self.sizes = [], []
        modules = {**fake_torch(self.allocations),
                   **fake_flashinfer(version, a2a_class, self.sizes, pdl=pdl)}
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
        rig = Harness(self, a2a_class=A2AWithOutput, version="0.7.0.post1")
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
        # vLLM combine_into: caller-owned payload, payload_in_workspace at its False default, the
        # caller's output, use_low_precision at its False default.
        self.assertIs(call["payload"], handle.recv_x)
        self.assertEqual(call["tokens"], T)
        self.assertIs(call["in_workspace"], False)
        self.assertIs(call["output"], problem.combine_output)
        self.assertIs(call["low_precision"], False)
        self.assertIs(out, problem.combine_output)
        self.assertEqual(problem.combine_output.copied_from, [])

    def test_a_wheel_without_output_copies_the_result_into_the_callers_tensor(self):
        rig = Harness(self, a2a_class=A2ABeforeOutput)
        problem = rig.problem()
        handle = rig.backend.dispatch(problem)
        rig.backend.stage(problem, handle)
        out = rig.backend.combine(problem, handle)
        (call,) = rig.a2a.combines
        self.assertIs(call["payload"], handle.recv_x)
        self.assertIs(call["in_workspace"], False)
        self.assertEqual(problem.combine_output.copied_from, [call["result"]])
        self.assertIs(out, problem.combine_output)

    def test_dispatch_payloads_follow_vllm_prepare(self):
        rig = Harness(self)
        problem = rig.problem()
        rig.backend.dispatch(problem)
        (call,) = rig.a2a.dispatches
        self.assertEqual([p.name for p in call["payloads"]], ["x", "idx.to(int32)", "weights"])
        self.assertIs(call["experts"], problem.topk_idx)
        self.assertEqual((call["invalid"], call["id_index"], call["tokens"]), (-1, 1, T))

    def test_output_is_allocated_once_per_rung_outside_any_window(self):
        rig = Harness(self, a2a_class=A2AWithOutput)
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
        problem = rig.problem()
        backend.dispatch(problem)
        (call,) = rig.a2a.dispatches
        self.assertEqual([p.name for p in call["payloads"]], ["values", "scales", "idx.to(int32)", "weights"])
        self.assertEqual((call["invalid"], call["id_index"]), (-1, 2))

    def test_stage_dequantizes_filled_slots_into_a_caller_owned_bf16_payload(self):
        rig = Harness(self, precision="fp8", a2a_class=A2AWithOutput, version="0.7.0.post1")
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
        self.assertIs(call["in_workspace"], False)


class OracleContract(unittest.TestCase):
    def test_transformed_payload_takes_the_timed_call(self):
        for a2a_class in (A2AWithOutput, A2ABeforeOutput):
            with self.subTest(wheel=a2a_class.__name__):
                rig = Harness(self, a2a_class=a2a_class)
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
                self.assertIs(call["in_workspace"], False)
                self.assertIs(out, problem.combine_output)


class Provenance(unittest.TestCase):
    def test_kernel_generation_and_library_version_carry_the_wheel(self):
        for version, slug, reduction in (("0.6.8.post1", "0.6.8.post1", "topk-slot-tree"),
                                         ("0.7.1+cu130", "0.7.1-cu130", "domain-fp32")):
            with self.subTest(version=version):
                backend = Harness(self, version=version).backend
                self.assertEqual(backend.library_version, version)
                self.assertEqual(backend.kernel_generation, f"flashinfer-mnnvl-one-sided-staged-{slug}")
                self.assertEqual(backend.combine_reduction, reduction)
                with mock.patch.dict(os.environ, {"COLLX_CUDA_GRAPH": "1"}):
                    self.assertEqual(ep_harness.kernel_generation(backend),
                                     f"flashinfer-mnnvl-one-sided-staged-{slug}-cudagraph-spin-ar")

    def test_workspace_and_mapping_are_sized_as_vllm_initialize_sizes_them(self):
        for precision, x_bytes, sf_bytes in (("bf16", HIDDEN * 2, 0), ("fp8", HIDDEN, HIDDEN // 128 * 4)):
            with self.subTest(precision=precision):
                rig = Harness(self, precision=precision)
                (sizes,) = rig.sizes
                self.assertEqual(sizes, {
                    "ep_size": EP, "max_num_tokens": 64,
                    "total_dispatch_payload_size_per_token": x_bytes + sf_bytes + TOPK * 4 + TOPK * 4,
                    "combine_payload_size_per_token": HIDDEN * 2,
                })
                self.assertEqual(rig.a2a.constructed["mapping"],
                                 ("mapping", (EP, 1, 4), {"tp_size": EP, "moe_ep_size": EP}))

    def test_implementation_details_record_the_contract(self):
        with mock.patch.dict(os.environ, {"TLLM_MOE_A2A_COMBINE_BLOCK_SIZE": "512", "OTHER": "1"}):
            # Each read runs while its own fake wheel is the one installed.
            new = Harness(self, a2a_class=A2AWithOutput, version="0.7.0.post1", pdl=True).backend
            new = new.implementation_details()
            old = Harness(self, a2a_class=A2ABeforeOutput, precision="fp8", pdl=False).backend
            old = old.implementation_details()
        self.assertEqual(new, {
            "flashinfer_version": "0.7.0.post1",
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
            {key: old[key] for key in ("flashinfer_version", "combine_output", "pdl_available",
                                       "dispatch_scale_format", "dispatch_bytes_per_token")},
            {"flashinfer_version": "0.6.8.post1", "combine_output": "copy", "pdl_available": False,
             "dispatch_scale_format": "fp32-per-128-block-row-major",
             "dispatch_bytes_per_token": HIDDEN + HIDDEN // 128 * 4 + TOPK * 8},
        )

    def test_keyword_probe_matches_vllm_supports_kw(self):
        module = Harness(self).module

        def explicit(payload, output=None):
            pass

        def keyword_only(payload, *, output=None):
            pass

        def var_kwargs(payload, **output):
            pass

        def absent(payload):
            pass

        self.assertTrue(module._accepts_keyword(explicit, "output"))
        self.assertTrue(module._accepts_keyword(keyword_only, "output"))
        # allow_var_kwargs=False, as vLLM calls it: **kwargs does not promise the keyword.
        self.assertFalse(module._accepts_keyword(var_kwargs, "output"))
        self.assertFalse(module._accepts_keyword(absent, "output"))


if __name__ == "__main__":
    unittest.main()
