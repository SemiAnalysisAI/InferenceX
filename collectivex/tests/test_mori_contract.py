#!/usr/bin/env python3
"""MoRI adapter contract: what it hands MoRI, the labels it publishes and its FP8 oracle model."""
from __future__ import annotations

import importlib
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "bench"), str(ROOT / "tests")]

# Imported before any stubbed load so the adapter binds the real shared base, which stays loaded.
import ep_backend  # noqa: E402,F401

try:
    import torch as _torch
except Exception:  # torch is absent in the CPU test image; the codec check runs where it exists
    _torch = None


class _Dtype:
    def __init__(self, name, itemsize):
        self.name, self.itemsize = name, itemsize

    def __repr__(self):
        return self.name


class _Compiled:
    """torch.compile's wrapper, so a test can tell the compiled quantize from the eager one."""

    def __init__(self, fn):
        self.fn = fn


class _Rows:
    """A received buffer with `rows` rows that records how it was sliced."""

    def __init__(self, name, rows=16):
        self.name, self.rows, self.ndim = name, rows, 2

    def size(self, dim):
        return self.rows

    def __getitem__(self, index):
        return f"{self.name}[:{index.stop}]"


class _FakeOp:
    """Records the calls the adapter makes and reports back the config it was built from, as MoRI's
    get_handle_info does, except for any `realized` field the test overrides."""

    def __init__(self, config, **realized):
        self.config = config
        self.launch_config_mode = os.environ.get("MORI_EP_LAUNCH_CONFIG_MODE", "MANUAL")
        self._handle_info = {
            "block_num": config.block_num,
            "rdma_block_num": config.rdma_block_num,
            "warp_num_per_block": config.warp_num_per_block,
            "gpu_per_node": config.gpu_per_node,
            "scale_dim": config.scale_dim,
            "scale_type_size": config.scale_type_size,
            "use_external_inp_buffer": config.use_external_inp_buf,
            **realized,
        }
        self.calls = []

    def dispatch(self, *args, **kwargs):
        self.calls.append(("dispatch", args, kwargs))
        return _Rows("recv_x"), _Rows("recv_w"), _Rows("recv_scales"), _Rows("recv_idx"), ["count"]

    def combine(self, *args, **kwargs):
        self.calls.append(("combine", args, kwargs))
        return list(range(8)), None


def _fake_torch():
    torch = types.ModuleType("torch")
    for name, itemsize in (
        ("bfloat16", 2), ("float32", 4), ("float8_e4m3fn", 1), ("float8_e4m3fnuz", 1),
        ("int32", 4), ("int64", 8),
    ):
        setattr(torch, name, _Dtype(name, itemsize))
    torch.arch = "gfx950:sramecc+:xnack-"
    torch.cuda = types.SimpleNamespace(
        get_device_properties=lambda device: types.SimpleNamespace(gcnArchName=torch.arch)
    )
    torch._C = types.SimpleNamespace(
        _distributed_c10d=types.SimpleNamespace(_register_process_group=lambda name, group: None)
    )
    torch._dynamo = types.SimpleNamespace(config=types.SimpleNamespace(cache_size_limit=8))
    torch.compile = lambda fn, dynamic=None: _Compiled(fn)
    torch.distributed = types.ModuleType("torch.distributed")
    torch.distributed.group = types.SimpleNamespace(WORLD="world")
    return torch


def _fake_mori():
    mori = types.ModuleType("mori")
    mori.sdma_at_shmem_init = []
    mori.__version__ = "1.2.3.dev45+g12d1bc32d"
    mori.ops = types.SimpleNamespace(
        EpDispatchCombineKernelType=types.SimpleNamespace(
            IntraNode="IntraNode", InterNodeV1="InterNodeV1", InterNodeV1LL="InterNodeV1LL"
        ),
        EpDispatchCombineConfig=lambda **kwargs: types.SimpleNamespace(**kwargs),
        EpDispatchCombineOp=_FakeOp,
    )
    mori.shmem = types.SimpleNamespace(
        shmem_torch_process_group_init=lambda name: mori.sdma_at_shmem_init.append(
            os.environ.get("MORI_ENABLE_SDMA")
        ),
        shmem_num_qp_per_pe=lambda: 4,
    )
    return mori


def _stubs(torch, mori):
    return {"torch": torch, "torch.distributed": torch.distributed, "mori": mori}


def _load(torch, mori):
    """A fresh ep_mori bound to `torch` and `mori`; sys.modules and the environment are restored."""
    with mock.patch.dict(sys.modules, _stubs(torch, mori)), mock.patch.dict(os.environ):
        sys.modules.pop("ep_mori", None)
        return importlib.import_module("ep_mori")


def _args(**updates):
    values = dict(
        mode="normal", precision="bf16", phase="decode", scope="scale-up", experts=384,
        hidden=7168, topk=6, gpus_per_node=8,
    )
    values.update(updates)
    return types.SimpleNamespace(**values)


def _problem():
    return types.SimpleNamespace(T=2, dispatch_x="x", weights="w", indices="i", recv_tokens=3)


class MoriContract(unittest.TestCase):
    def setUp(self):
        self.torch, self.mori = _fake_torch(), _fake_mori()
        self.module = _load(self.torch, self.mori)
        for patcher in (
            mock.patch.dict(sys.modules, _stubs(self.torch, self.mori)),
            mock.patch.dict(os.environ, {"COLLX_CUDA_GRAPH": "1"}),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        os.environ.pop("MORI_ENABLE_SDMA", None)

    def backend(self, arch="gfx950", world_size=8, max_tokens=512, **updates):
        self.torch.arch = f"{arch}:sramecc+:xnack-"
        backend = self.module.MoRIBackend(_args(**updates), 0, world_size, 0, "cuda:0")
        backend.create_buffer(types.SimpleNamespace(max_tokens_per_rank=max_tokens))
        return backend

    @staticmethod
    def roundtrip(backend):
        problem = _problem()
        handle = backend.dispatch(problem)
        handle.combine_input = "staged"
        return backend.combine(problem, handle)

    def test_one_node_runs_vllms_intranode_launch_from_the_config_alone(self):
        backend = self.backend()
        config = backend.config
        self.assertEqual(
            (config.kernel_type, config.block_num, config.rdma_block_num, config.warp_num_per_block,
             config.num_qp_per_pe, config.gpu_per_node, config.use_external_inp_buf, config.quant_type),
            ("IntraNode", 80, 0, 16, 1, 8, True, "none"),
        )
        self.assertEqual(
            (config.data_type, config.scale_dim, config.scale_type_size), (self.torch.bfloat16, 0, 0)
        )
        self.assertEqual(self.roundtrip(backend), [0, 1])
        # vLLM's prepare/finalize pass no per-call launch override and no scales for BF16.
        self.assertEqual(backend.op.calls, [
            ("dispatch", ("x", "w", None, "i"), {}),
            ("combine", ("staged", None, "i"), {}),
        ])
        self.assertEqual(backend.kernel_generation, "intranode-vllm")
        realized = backend.implementation_details()
        self.assertEqual(
            (realized["contract"], realized["kernel"], realized["block_num"], realized["num_qp_per_pe"],
             realized["realized_from"], realized["sdma"]),
            ("vllm", "IntraNode", 80, 1, "handle", False),
        )

    def test_two_nodes_take_vllms_internode_tuple_for_the_arch(self):
        for arch, launch in (("gfx950", (64, 32, 8)), ("gfx942", (32, 16, 16))):
            with self.subTest(arch=arch):
                backend = self.backend(arch=arch, world_size=16, scope="scale-out")
                config = backend.config
                self.assertEqual(
                    (config.kernel_type, config.block_num, config.rdma_block_num,
                     config.warp_num_per_block, config.num_qp_per_pe, config.gpu_per_node),
                    ("InterNodeV1", *launch, 1, 8),
                )
                self.assertEqual(backend.kernel_generation, "inter-node-v1-vllm")
        with self.assertRaisesRegex(RuntimeError, "gfx942 and gfx950"):
            self.backend(arch="gfx90a")

    def test_low_latency_runs_vllms_internode_v1ll_tuple_across_nodes_only(self):
        # vLLM's mori_low_latency: InterNodeV1LL with the arch's tuple across nodes, called as
        # dispatch()/combine() like every vLLM MoRI path; one node runs the normal row's IntraNode.
        for arch, launch in (("gfx950", (64, 32, 8)), ("gfx942", (32, 16, 16))):
            with self.subTest(arch=arch):
                backend = self.backend(arch=arch, world_size=16, scope="scale-out", mode="low-latency",
                                       max_tokens=256)
                config = backend.config
                self.assertEqual(
                    (config.kernel_type, config.block_num, config.rdma_block_num,
                     config.warp_num_per_block, config.num_qp_per_pe),
                    ("InterNodeV1LL", *launch, 1),
                )
                self.roundtrip(backend)
                self.assertEqual([call[0] for call in backend.op.calls], ["dispatch", "combine"])
                self.assertEqual(
                    (backend.kernel_generation, backend.implementation_details()["contract"]),
                    ("inter-node-v1ll-vllm", "vllm"),
                )
        with self.assertRaisesRegex(RuntimeError, "scale-out only"):
            self.backend(mode="low-latency", max_tokens=256)

    def test_fp8_carries_per_128_scales_in_the_arch_format(self):
        for arch, dtype, label in (
            ("gfx942", "float8_e4m3fnuz", "fp8-e4m3fnuz"), ("gfx950", "float8_e4m3fn", "fp8-e4m3fn"),
        ):
            with self.subTest(arch=arch):
                backend = self.backend(arch=arch, precision="fp8")
                config = backend.config
                self.assertEqual(
                    (config.data_type, config.scale_dim, config.scale_type_size),
                    (getattr(self.torch, dtype), 56, 4),
                )
                # One byte per value plus 7168 / 128 FP32 scales on every dispatched copy.
                self.assertEqual(
                    (backend.dispatch_dtype, backend.dispatch_value_bytes,
                     backend.dispatch_scale_bytes_per_copy),
                    (label, 1, 224),
                )

    def test_fp8_quantizes_in_dispatch_and_dequantizes_received_rows_with_their_scales(self):
        backend = self.backend(precision="fp8")
        dequantized = []
        backend._quant = lambda x: (f"q({x})", f"s({x})")
        backend._fp8_codec = (None, lambda values, scales: dequantized.append((values, scales)) or "bf16")
        problem = _problem()
        handle = backend.dispatch(problem)
        self.assertEqual(backend.op.calls, [("dispatch", ("q(x)", "w", "s(x)", "i"), {})])
        backend.stage(problem, handle)
        self.assertEqual(handle.combine_input, "bf16")
        # The oracle's expected payload is the same codec's round trip of the sent activations.
        self.assertEqual(backend.semantic_payload("x"), "bf16")
        self.assertEqual(dequantized, [("recv_x[:3]", "recv_scales[:3]"), ("q(x)", "s(x)")])

    def test_low_latency_compiles_the_quantize_because_v1ll_is_caller_quantized(self):
        backend = self.backend(world_size=16, scope="scale-out", mode="low-latency", precision="fp8",
                               max_tokens=256)
        self.assertIsInstance(backend._quant, _Compiled)
        self.assertIs(backend._quant.fn, backend._fp8_codec[0])

    def test_only_decode_replays_a_graph(self):
        self.assertTrue(self.backend(phase="decode").cuda_graph_supported)
        self.assertFalse(self.backend(phase="prefill", max_tokens=8192).cuda_graph_supported)

    def test_no_path_turns_sdma_on_as_vllm_never_sets_it(self):
        # MoRI snapshots MORI_ENABLE_SDMA at shmem init; vLLM leaves it unset on every path.
        for mode, world_size, scope, generation in (
            ("normal", 8, "scale-up", "intranode-vllm"),
            ("normal", 16, "scale-out", "inter-node-v1-vllm"),
            ("low-latency", 16, "scale-out", "inter-node-v1ll-vllm"),
        ):
            with self.subTest(mode=mode, world_size=world_size):
                backend = self.backend(world_size=world_size, scope=scope, mode=mode, max_tokens=256)
                self.assertIsNone(self.mori.sdma_at_shmem_init[-1])
                self.assertEqual(backend.kernel_generation, generation)
                self.assertIs(backend.implementation_details()["sdma"], False)

    def test_sdma_is_read_as_mori_reads_it_and_tags_the_rows(self):
        for value, enabled in (
            ("", False), ("0", False), ("false", False), ("OFF", False), ("no", False),
            ("1", True), ("true", True), ("Yes", True), ("2", True),
        ):
            with self.subTest(value=value), mock.patch.dict(os.environ, {"MORI_ENABLE_SDMA": value}):
                backend = self.backend()
                self.assertEqual(backend.kernel_generation.endswith("-sdma"), enabled)
                self.assertIs(backend.implementation_details()["sdma"], enabled)

    def test_a_handle_that_realizes_another_launch_fails_closed(self):
        self.mori.ops.EpDispatchCombineOp = lambda config: _FakeOp(config, block_num=96)
        with self.assertRaisesRegex(RuntimeError, "block_num"):
            self.backend()

    def test_library_version_is_the_build_stamp_then_the_wheel(self):
        backend = self.backend()
        self.assertEqual(backend.library_version, "1.2.3.dev45+g12d1bc32d")
        self.mori.__version__ = "unknown"
        # Patch the metadata module ep_mori bound: another suite's sys.modules restore can leave
        # this file's `metadata` a different object from the one the adapter imported.
        used = self.module.metadata
        with mock.patch.object(used, "version", return_value="1.2.3.post1"):
            self.assertEqual(backend.library_version, "1.2.3.post1")
        with mock.patch.object(used, "version", side_effect=used.PackageNotFoundError):
            self.assertEqual(backend.library_version, "unknown")


@unittest.skipUnless(_torch is not None, "the FP8 codec check requires torch")
class BlockwiseCodec(unittest.TestCase):
    def test_scales_are_block_amax_over_the_format_max_and_block_peaks_round_trip(self):
        torch = _torch
        module = _load(torch, _fake_mori())
        x = torch.zeros(2, 256, dtype=torch.bfloat16)
        x[0, 5], x[0, 200], x[1] = 3.0, -0.5, 1.0
        for dtype, fp8_max in ((torch.float8_e4m3fn, 448.0), (torch.float8_e4m3fnuz, 240.0)):
            with self.subTest(dtype=dtype):
                quantize, dequantize = module._blockwise_codec(dtype, fp8_max)
                values, scales = quantize(x)
                self.assertEqual((values.dtype, tuple(scales.shape)), (dtype, (2, 2)))
                expected = torch.tensor([[3.0, 0.5], [1.0, 1.0]]) / fp8_max
                self.assertTrue(torch.allclose(scales, expected))
                # Every nonzero is its block's peak, so it lands on +-fp8_max and comes back exactly.
                self.assertTrue(torch.equal(dequantize(values, scales), x))


if __name__ == "__main__":
    unittest.main()
