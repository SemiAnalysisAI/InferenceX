#!/usr/bin/env python3
"""deepep-v2 runs vLLM's DeepEP configuration: what reaches ElasticBuffer and the legacy LL Buffer,
the labels that keep the new rows apart, and the oracle models that follow the arithmetic."""
from __future__ import annotations

import importlib
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "bench")]

# Imported before any stubbed load so the adapter binds the real shared base, which stays loaded.
import ep_backend  # noqa: E402,F401
import ep_oracle  # noqa: E402

try:
    import torch as _torch
except Exception:  # torch is absent in the plain CPU test image; the math checks run on CI
    _torch = None


class _ElasticBuffer:
    """Records the ElasticBuffer surface the adapter touches. DeepEP's SM model is stubbed to the
    64 SMs vLLM's flags yield; the QP model is upstream's (elastic.py get_theoretical_num_qps)."""

    lsa_ranks = 8  # the NVLink (LSA) team NCCL realized: one x86 node

    def __init__(self, group, **kwargs):
        self.kwargs = kwargs
        self.allow_hybrid_mode = kwargs["allow_hybrid_mode"]
        self.num_allocated_qps = kwargs["num_allocated_qps"] or (65 if self.allow_hybrid_mode else 17)
        self.num_nvlink_ranks = _ElasticBuffer.lsa_ranks
        self.sm_calls, self.dispatches, self.combines = [], [], []

    def get_theoretical_num_sms(self, *args, **kwargs):
        self.sm_calls.append((args, kwargs))
        return 64

    def get_theoretical_num_qps(self, num_sms):
        wanted = num_sms * 16 + 1 if self.allow_hybrid_mode else min(num_sms, 9)
        return min(wanted, self.num_allocated_qps)

    def dispatch(self, x, **kwargs):
        self.dispatches.append(kwargs)
        return "recv_x", "recv_idx", "recv_w", "handle", None

    def combine(self, x, **kwargs):
        self.combines.append(kwargs)
        return "combined", None, None


_ElasticBuffer.__name__ = "ElasticBuffer"


class _LegacyBuffer:
    """The legacy low-latency Buffer surface: constructor kwargs, and each collective's kwargs plus
    how often its receive hook ran."""

    def __init__(self, group, num_rdma_bytes=0, low_latency_mode=False, num_qps_per_rank=24,
                 allow_nvlink_for_low_latency_mode=True, allow_mnnvl=False, explicitly_destroy=False):
        self.init = dict(
            num_rdma_bytes=num_rdma_bytes, low_latency_mode=low_latency_mode,
            num_qps_per_rank=num_qps_per_rank, allow_mnnvl=allow_mnnvl,
            allow_nvlink_for_low_latency_mode=allow_nvlink_for_low_latency_mode,
        )
        self.calls, self.hooks = [], []

    @staticmethod
    def get_low_latency_rdma_size_hint(*args):
        return 4096

    def _hook(self, name):
        return lambda: self.hooks.append(name)

    def low_latency_dispatch(self, x, topk_idx, cap, experts, **kwargs):
        self.calls.append(("dispatch", kwargs))
        return "recv_x", "recv_count", "ll_handle", None, self._hook("dispatch")

    def low_latency_combine(self, x, topk_idx, weights, handle, **kwargs):
        self.calls.append(("combine", kwargs))
        return list(range(8)), None, self._hook("combine")


def _stub_torch():
    torch = types.ModuleType("torch")
    torch.bfloat16, torch.float32, torch.int32, torch.int64 = "bfloat16", "float32", "int32", "int64"
    torch.compile = lambda fn=None, **_: fn if fn is not None else (lambda inner: inner)
    torch._dynamo = types.SimpleNamespace(config=types.SimpleNamespace(cache_size_limit=8))
    dist = types.ModuleType("torch.distributed")
    dist.group = types.SimpleNamespace(WORLD="world")
    torch.distributed = dist
    return {"torch": torch, "torch.distributed": dist}


def _adapter(ue8m0=False):
    """A fresh ep_deepep_v2 against the stubs, with the Blackwell/UE8M0 decision fixed."""
    deep_ep = types.ModuleType("deep_ep")
    deep_ep.ElasticBuffer, deep_ep.Buffer = _ElasticBuffer, _LegacyBuffer
    deep_ep.topk_idx_t, deep_ep.__version__ = "int64", "2.1.0"
    with mock.patch.dict(sys.modules, {**_stub_torch(), "deep_ep": deep_ep}):
        for name in ("ep_legacy_ll", "ep_deepep_v2"):
            sys.modules.pop(name, None)
        module = importlib.import_module("ep_deepep_v2")
        legacy = sys.modules["ep_legacy_ll"]
    legacy._packed_ue8m0_scales = lambda device: ue8m0
    module._fp8_cast_helpers = lambda: ("quantize", "dequantize")
    return module


def _args(**updates):
    values = dict(
        mode="normal", phase="decode", precision="bf16", experts=384, topk=6, hidden=7168,
        scale_up_domain=8, scale_up_transport="nvlink", runner="h200-dgxc", tokens_ladder="",
    )
    values.update(updates)
    return types.SimpleNamespace(**values)


def _built(module, world_size=8, env=None, max_tokens=512, **updates):
    backend = module.DeepEPV2Backend(_args(**updates), 0, world_size, 0, "cpu")
    with mock.patch.dict(os.environ, {"EP_JIT_CACHE_DIR": "/tmp/jit", **(env or {})}, clear=True):
        backend.create_buffer(types.SimpleNamespace(max_tokens_per_rank=max_tokens))
    return backend


def _problem(tokens):
    return types.SimpleNamespace(T=tokens, dispatch_x="x", topk_idx="i", topk_weights="w")


class ElasticBufferContract(unittest.TestCase):
    """vLLM's DeepEPV2All2AllManager flags reach the buffer at every scope."""

    def test_buffer_gets_vllms_flags_and_the_ep16_qp_budget(self):
        module = _adapter()
        for world_size, hybrid, allocated in ((8, False, 0), (16, True, 33)):
            with self.subTest(world_size=world_size):
                kwargs = _built(module, world_size).buffer.kwargs
                self.assertFalse(kwargs["prefer_overlap_with_compute"])
                self.assertFalse(kwargs["allow_multiple_reduction"])
                self.assertFalse(kwargs["deterministic"])
                self.assertEqual(kwargs["allow_hybrid_mode"], hybrid)
                self.assertEqual(kwargs["num_allocated_qps"], allocated)

    def test_hybrid_follows_the_realized_lsa_team_or_fails_closed(self):
        # Keying hybrid on the registry matches vLLM's autoselect (hybrid iff LSA < world) only
        # while the registry's scale-up domain is the team NCCL built.
        module = _adapter()
        self.assertEqual(_built(module, world_size=16).implementation_details()["lsa_team_ranks"], 8)
        with mock.patch.object(_ElasticBuffer, "lsa_ranks", 16):
            # GB EP16: the MNNVL team covers the world, so DeepEP would pick the direct kernels too.
            rack = _built(module, world_size=16, scale_up_domain=72, scale_up_transport="mnnvl")
            self.assertFalse(rack.buffer.kwargs["allow_hybrid_mode"])
            with self.assertRaisesRegex(RuntimeError, "LSA team has 16 ranks"):
                _built(module, world_size=16)
        with mock.patch.object(_ElasticBuffer, "lsa_ranks", 4):
            with self.assertRaisesRegex(RuntimeError, "expects 8"):
                _built(module, world_size=8)

    def test_sm_and_qp_counts_are_left_to_deepep_as_vllm_leaves_them(self):
        module = _adapter()
        backend = _built(module, world_size=16)
        # The model is called exactly as dispatch(num_sms=0) calls it, so the cached count is the
        # one the kernels run with, and dispatch and combine pass nothing of their own.
        self.assertEqual(backend.buffer.sm_calls, [((384, 6), {})])
        backend.dispatch(_problem(3))
        backend.combine(_problem(3), types.SimpleNamespace(combine_input="staged", handle="h"))
        dispatched, combined = backend.buffer.dispatches[-1], backend.buffer.combines[-1]
        self.assertEqual(dispatched["num_sms"], 0)
        self.assertNotIn("num_qps", dispatched)
        self.assertFalse({"num_sms", "num_qps"} & set(combined))
        # Realized: 64 SMs; hybrid EP16 wants 64 x 16 + 1 QPs and gets the 33 allocated.
        details = backend.implementation_details()
        self.assertEqual(
            (details["num_sms"], details["num_qps"], details["num_allocated_qps"]), (64, 33, 33))

    def test_efa_supplies_the_rate_the_ibstat_probe_cannot_see(self):
        module = _adapter()
        backend = _built(module, world_size=16, env={"COLLX_RDMA_GBS": "100"})
        self.assertEqual(backend.buffer.sm_calls, [((384, 6), {"rdma_gbs": 100.0})])
        backend.dispatch(_problem(3))
        self.assertEqual(backend.buffer.dispatches[-1]["num_sms"], 64)
        self.assertEqual(backend.implementation_details()["sm_model_rdma_gbs"], 100.0)

    def test_prefill_runs_the_decode_path_but_stays_eager(self):
        module = _adapter()
        for phase, graphed in (("decode", True), ("prefill", False)):
            with self.subTest(phase=phase):
                backend = _built(module, phase=phase, max_tokens=8192)
                # vLLM (graphs on) passes the next power of two of the batch, with no host sync.
                for tokens, capacity in ((1, 1), (3, 4), (65, 128), (3000, 4096)):
                    backend.dispatch(_problem(tokens))
                    dispatched = backend.buffer.dispatches[-1]
                    self.assertEqual(
                        (dispatched["do_cpu_sync"], dispatched["do_expand"],
                         dispatched["num_max_tokens_per_rank"]), (False, False, capacity))
                self.assertFalse(backend.host_synced_dispatch)
                self.assertIs(backend.cuda_graph_supported, graphed)
                self.assertEqual(backend.kernel_generation, "v2-elastic-buffer-vllm-nosync")

    def test_the_single_reduction_selects_the_single_level_oracle(self):
        module = _adapter()
        for world_size in (8, 16):
            backend = _built(module, world_size)
            self.assertEqual(backend.combine_reduction, "rank-fp32")
            self.assertIsInstance(backend.combine_model, ep_oracle.RankFP32)
            self.assertFalse(backend.implementation_details()["allow_multiple_reduction"])

    def test_library_version_prefers_the_installed_wheel_then_the_module(self):
        module = _adapter()
        backend = module.DeepEPV2Backend(_args(), 0, 8, 0, "cpu")
        with mock.patch.object(module.metadata, "version", return_value="2.1.0+local"):
            self.assertEqual(backend.library_version.split("@")[0], "2.1.0+local")
        with mock.patch.object(module.metadata, "version",
                               side_effect=module.metadata.PackageNotFoundError):
            self.assertEqual(backend.library_version.split("@")[0], "2.1.0")


class LowLatencyContract(unittest.TestCase):
    """vLLM's DeepEPLLAll2AllManager kwargs and deepep_ll.py calls."""

    def _ll(self, ue8m0=False, **updates):
        module = _adapter(ue8m0)
        return _built(module, mode="low-latency", max_tokens=256, **updates)

    def test_buffer_kwargs_follow_vllm_and_its_documented_gb200_switch(self):
        for transport, mnnvl in (("nvlink", False), ("mnnvl", True)):
            with self.subTest(transport=transport):
                backend = self._ll(scale_up_transport=transport)
                init = backend.buffer.init
                self.assertEqual(init["num_qps_per_rank"], 384 // 8)
                self.assertTrue(init["allow_nvlink_for_low_latency_mode"])
                self.assertTrue(init["low_latency_mode"])
                self.assertIs(init["allow_mnnvl"], mnnvl)
                self.assertIs(backend.implementation_details()["allow_mnnvl"], mnnvl)

    def test_both_collectives_split_into_send_and_receive_hook(self):
        backend = self._ll()
        handle = backend.dispatch(_problem(4))
        backend.combine(_problem(4), types.SimpleNamespace(combine_input="plane", ll_handle="h"))
        (_, dispatched), (_, combined) = backend.buffer.calls
        self.assertEqual(
            {k: dispatched[k] for k in ("return_recv_hook", "async_finish", "use_fp8")},
            {"return_recv_hook": True, "async_finish": False, "use_fp8": False},
        )
        self.assertEqual(
            {k: combined[k] for k in ("return_recv_hook", "async_finish", "zero_copy")},
            {"return_recv_hook": True, "async_finish": False, "zero_copy": False},
        )
        self.assertEqual(backend.buffer.hooks, ["dispatch", "combine"])
        self.assertEqual(handle.ll_handle, "ll_handle")
        self.assertEqual(backend.kernel_generation, "legacy-buffer-ll-hook")
        self.assertIs(backend.cuda_graph_supported, True)

    def test_fp8_scales_are_ue8m0_on_blackwell_and_fp32_on_hopper(self):
        for blackwell, generation in ((True, "legacy-buffer-ll-hook-ue8m0"), (False, "legacy-buffer-ll-hook")):
            with self.subTest(blackwell=blackwell):
                backend = self._ll(ue8m0=blackwell, precision="fp8")
                backend.dispatch(_problem(4))
                dispatched = backend.buffer.calls[-1][1]
                self.assertEqual((dispatched["round_scale"], dispatched["use_ue8m0"]), (blackwell, blackwell))
                self.assertEqual(backend.kernel_generation, generation)
                self.assertEqual(
                    backend.implementation_details()["fp8_scales"], "ue8m0-packed" if blackwell else "fp32")
                # The wire still carries one FP32 scale per 128-value block either way.
                self.assertEqual(backend.dispatch_scale_bytes_per_copy, 7168 // 128 * 4)
                quantize = backend._fp8_codec[0].__name__ if blackwell else backend._fp8_codec[0]
                self.assertEqual(quantize, "ue8m0_cast_to_fp8" if blackwell else "quantize")


@unittest.skipUnless(_torch is not None, "UE8M0 cast checks require torch")
class Ue8m0CastModel(unittest.TestCase):
    """The oracle's payload compare is bit-exact, so the cast must be the kernel's."""

    def setUp(self):
        # Not imported under patch.dict: restoring sys.modules would drop torch's lazy submodules.
        self.legacy = importlib.import_module("ep_legacy_ll")

    def _blocks(self):
        torch = _torch
        x = torch.zeros(2, 128 * 8, dtype=torch.bfloat16)
        x[:, 0] = 1.75                       # amax * fl(1/448) lands exactly on 2^-8
        x[:, 128] = 1.7578125                # one bf16 step above: rounds up to 2^-7
        x[:, 384] = -448.0                   # negative amax, product exactly 1
        x[:, 512:640] = (torch.arange(128) % 257 - 128).to(torch.bfloat16) / 64
        x[:, 640:] = torch.randn(2, 384).to(torch.bfloat16) * 37
        return x                             # block 2 (columns 256-383) stays zero: the margin

    def test_scales_match_the_kernels_power_of_two_rounding(self):
        torch = _torch
        x = self._blocks()
        values, scales = self.legacy.ue8m0_cast_to_fp8(x)
        self.assertEqual(scales[0, :5].tolist(), [2.0 ** -8, 2.0 ** -7, 2.0 ** -22, 1.0, 2.0 ** -7])
        self.assertEqual(values.dtype, torch.float8_e4m3fn)
        self.assertLessEqual(float(values.float().abs().max()), 448.0)
        # Exactly representable after scaling, so the round trip is exact.
        back = (values.float().view(2, 8, 128) * scales.unsqueeze(2)).view(2, -1)
        self.assertEqual(back[:, 0].tolist(), [1.75, 1.75])
        self.assertEqual(back[:, 384].tolist(), [-448.0, -448.0])

    def test_packed_receive_scales_decode_to_the_same_powers_of_two(self):
        torch = _torch
        x = self._blocks().repeat(3, 1)
        _values, scales = self.legacy.ue8m0_cast_to_fp8(x)
        experts, slots, groups = 1, x.shape[0], scales.shape[1]
        exponent_bytes = ((scales.view(torch.int32) >> 23) & 0xFF).to(torch.uint8)
        packed = exponent_bytes.view(experts, slots, groups).view(torch.int32)
        # The kernel's layout: column-major in the last two dims (TMA), as dispatch returns it.
        column_major = packed.transpose(1, 2).contiguous().transpose(1, 2)
        decoded = self.legacy._ue8m0_to_fp32(column_major)
        self.assertTrue(torch.equal(decoded, scales.view(experts, slots, groups)))

    def test_the_receive_hands_the_dequant_its_decoded_scales(self):
        torch = _torch
        values, scales = self.legacy.ue8m0_cast_to_fp8(self._blocks())
        packed = ((scales.view(torch.int32) >> 23) & 0xFF).to(torch.uint8).view(1, 2, -1).view(torch.int32)
        backend = types.SimpleNamespace(_fp8=True, _ll_ue8m0=True)
        seen = []
        with mock.patch.object(self.legacy, "_ll_dequant_static",
                               lambda fp8, s: seen.append(s) or "bf16"):
            received = self.legacy.LegacyBufferLL._ll_recv_bf16(backend, (values.view(1, 2, -1), packed))
        self.assertEqual(received, "bf16")
        self.assertTrue(torch.equal(seen[0], scales.view(1, 2, -1)))


@unittest.skipUnless(_torch is not None, "combine-oracle math checks require torch")
class SingleReductionOracle(unittest.TestCase):
    """allow_multiple_reduction=False: x86 EP16 forwarders relay each rank's BF16 row and the source
    sums the first top-k slot of each unique rank in FP32 -- no per-domain BF16 partial."""

    def _reduce(self, model, destination, rows, scale_up_domain):
        torch = _torch
        x = torch.zeros(1, 1, dtype=torch.bfloat16)
        valid = destination >= 0
        present = sorted(destination[valid].unique().tolist())
        return model.reduce(torch, x, destination, valid, present,
                            lambda rank: torch.full((1, 1), rows[rank]), scale_up_domain).item()

    def test_remote_ranks_in_one_domain_are_not_pre_rounded(self):
        torch = _torch
        # Ranks 8 and 9 share the remote domain; 1 + 2^-8 is a BF16 tie that rounds to 1. The
        # single reduction sums each unique rank's first top-k slot in FP32 and never rounds it.
        rows = {8: 1.0, 9: 2.0 ** -8, 0: 2.0 ** -8}
        destination = torch.tensor([[8, 9, 9, 0]])
        self.assertEqual(self._reduce(ep_oracle.RankFP32(), destination, rows, 8), 1.0 + 2.0 ** -7)
        self.assertEqual(self._reduce(ep_oracle.DomainFP32(), destination, rows, 8), 1.0 + 2.0 ** -8)

    def test_one_domain_models_agree_up_to_the_final_cast(self):
        torch = _torch
        # EP8 and MNNVL: no forwarder, so either model describes DeepEP's single FP32 sum.
        rows = {0: 1.0, 1: 2.0 ** -9, 2: 2.0 ** -9}
        destination = torch.tensor([[0, 1, 2]])
        single = self._reduce(ep_oracle.RankFP32(), destination, rows, 8)
        domain = self._reduce(ep_oracle.DomainFP32(), destination, rows, 8)
        self.assertEqual(torch.tensor(single).to(torch.bfloat16).item(), domain)


if __name__ == "__main__":
    unittest.main()
