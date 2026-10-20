#!/usr/bin/env python3
"""uccl-ep runs vLLM's DeepEP configuration: what the adapter hands UCCL's Buffer, how it labels
the rows, and the prepare step's build record and import probe. CPU-only fakes throughout."""
from __future__ import annotations

import importlib
import json
import os
import re
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import ClassVar
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench")]

PREPARE = ROOT / "runtime" / "prepare_backend.sh"
PIN = "df000fd3001da04a397253fd6a2204bde62d32aa"
GIB = 1024 * 1024 * 1024


class FakeConfig:
    """UCCL's Config: the constructor rounds the RDMA receive chunk up to a multiple of the send
    chunk (ep/include/ep_config.hpp), so realized values can differ from the literals."""

    def __init__(self, num_sms, nvl_send, nvl_recv, rdma_send, rdma_recv):
        self.num_sms = num_sms
        self.num_max_nvl_chunked_send_tokens = nvl_send
        self.num_max_nvl_chunked_recv_tokens = nvl_recv
        self.num_max_rdma_chunked_send_tokens = rdma_send
        self.num_max_rdma_chunked_recv_tokens = -(-rdma_recv // rdma_send) * rdma_send


class FakeBuffer:
    """UCCL's Buffer at the pin, reduced to the recommended-config tables (non-EFA rows of
    ep/bench/buffer.py get_dispatch_config/get_combine_config) and call recording."""

    num_sms = 20
    built: ClassVar[list] = []
    hints: ClassVar[list] = []

    def __init__(self, *args, **kwargs):
        self.args, self.kwargs, self.calls = args, kwargs, []
        self.proxies = [object()] * 4  # constructed always, started only across nodes
        FakeBuffer.built.append(self)

    @staticmethod
    def set_num_sms(new_num_sms):
        raise AssertionError("vLLM's default (non-DBO) path never calls set_num_sms")

    @staticmethod
    def get_dispatch_config(num_ranks):
        return {8: FakeConfig(FakeBuffer.num_sms, 6, 256, 6, 128),
                16: FakeConfig(FakeBuffer.num_sms, 36, 288, 20, 128)}[num_ranks]

    @staticmethod
    def get_combine_config(num_ranks):
        return {8: FakeConfig(FakeBuffer.num_sms, 4, 256, 6, 128),
                16: FakeConfig(FakeBuffer.num_sms, 4, 288, 12, 128)}[num_ranks]

    @staticmethod
    def get_low_latency_rdma_size_hint(*args):
        FakeBuffer.hints.append(args)
        return 4096

    def _hooked(self, name, kwargs, result):
        self.calls.append((name, kwargs))
        return (*result, lambda: self.calls.append((f"{name}-hook", {})))

    def low_latency_dispatch(self, *args, **kwargs):
        return self._hooked("ll-dispatch", kwargs, ("recv", "count", "ll-handle", None))

    def low_latency_combine(self, *args, **kwargs):
        return self._hooked("ll-combine", kwargs, ("ll-combined", None))

    def get_dispatch_layout(self, topk_idx, num_experts):
        return "per-rank", None, "per-expert", "in-rank", None

    def dispatch(self, **kwargs):
        self.calls.append(("dispatch", kwargs))
        return "recv", "recv-idx", "recv-weights", [], "handle", None

    def combine(self, **kwargs):
        self.calls.append(("combine", kwargs))
        return "combined", None, None


def _stubs(*, hip=None, foreign_config=False):
    torch = types.ModuleType("torch")
    torch.version = types.SimpleNamespace(cuda=None if hip else "13.0", hip=hip)
    # Hopper: the shared low-latency path keys its Blackwell-only FP8 scale format on this.
    torch.cuda = types.SimpleNamespace(get_device_capability=lambda device=None: (9, 0))
    torch.float8_e4m3fn, torch.float8_e4m3fnuz = "e4m3fn", "e4m3fnuz"
    torch.bfloat16, torch.float32, torch.int64, torch.int = "bfloat16", "float32", "int64", "int"
    # `@torch.compile(dynamic=False)` and `torch.compile(fn, dynamic=False)` both pass fn through.
    torch.compile = lambda fn=None, **kwargs: fn if fn is not None else (lambda inner: inner)
    torch._dynamo = types.SimpleNamespace(config=types.SimpleNamespace(cache_size_limit=8))
    dist = types.ModuleType("torch.distributed")
    dist.group = types.SimpleNamespace(WORLD="world")
    torch.distributed = dist
    uccl, uccl_ep = types.ModuleType("uccl"), types.ModuleType("uccl.ep")
    uccl_ep.Config = FakeConfig
    uccl.ep = uccl_ep
    deep_ep = types.ModuleType("deep_ep")
    # UCCL's wrapper re-exports uccl.ep's Config; DeepSeek's DeepEP brings its own.
    deep_ep.Buffer = FakeBuffer
    deep_ep.Config = type("Config", (), {}) if foreign_config else FakeConfig
    return {"torch": torch, "torch.distributed": dist, "uccl": uccl, "uccl.ep": uccl_ep,
            "deep_ep": deep_ep}


def _load(**stub_options):
    with mock.patch.dict(sys.modules, _stubs(**stub_options)):
        for name in ("ep_uccl", "ep_legacy_ll"):
            sys.modules.pop(name, None)
        return importlib.import_module("ep_uccl")


def _args(**updates):
    values = {"mode": "normal", "precision": "bf16", "phase": "decode", "experts": 384, "hidden": 7168,
              "topk": 6, "scale_up_domain": 8, "runner": "h200-dgxc"}
    values.update(updates)
    return types.SimpleNamespace(**values)


def _backend(module, world_size=8, **updates):
    # EPBackend imports torch lazily (the FP8 quantizer's compile), so the stub stays importable.
    with mock.patch.dict(sys.modules, {"torch": module.torch}):
        return module.UCCLEPBackend(_args(**updates), 0, world_size, 0, "cpu")


def _built(module, world_size=8, max_tokens=64, **updates):
    FakeBuffer.built.clear()
    FakeBuffer.hints.clear()
    backend = _backend(module, world_size, **updates)
    backend.create_buffer(types.SimpleNamespace(max_tokens_per_rank=max_tokens))
    return backend, FakeBuffer.built[-1]


def _prepare_snippet(*functions):
    """The UCCL build constants and the named functions from prepare_backend.sh, which cannot be
    sourced whole (it changes directory and runs its main)."""
    text = PREPARE.read_text()
    parts = re.findall(r"^readonly (?:-a )?UCCL_BUILD_\w+=.*$", text, re.MULTILINE)
    for name in functions:
        parts.append(re.search(rf"^{name}\(\) \{{$.*?^\}}$", text, re.MULTILINE | re.DOTALL).group(0))
    return "\n".join(parts)


def _prepare(functions, call, *args, env=None):
    return subprocess.run(["bash", "-c", f"{_prepare_snippet(*functions)}\n{call}", "uccl", *args],
                          capture_output=True, text=True, check=False, env={**os.environ, **(env or {})})


class ImportIdentity(unittest.TestCase):
    def test_a_deep_ep_that_is_not_uccls_wrapper_fails_at_import(self):
        # DeepSeek's DeepEP answers the same legacy Buffer API, so only identity separates them.
        with self.assertRaisesRegex(ImportError, "not UCCL's deep_ep_wrapper"):
            _load(foreign_config=True)


class NormalModeLaunchConfig(unittest.TestCase):
    """deepep_ht passes Buffer.get_dispatch_config / get_combine_config(EP size) on UCCL's default
    Buffer.num_sms (only dual-batch overlap calls set_num_sms, clamped to that same 20);
    DeepEPHTAll2AllManager sizes the Buffer from VLLM_DEEPEP_BUFFER_SIZE_MB."""

    def setUp(self):
        self.module = _load()

    def test_ep8_takes_the_recommended_configs_and_vllms_intranode_buffer(self):
        backend, buffer = _built(self.module)
        self.assertEqual(buffer.args, ("world", GIB, 0))
        self.assertEqual(buffer.kwargs, {"low_latency_mode": False, "num_qps_per_rank": 1,
                                         "explicitly_destroy": True, "is_intranode": True})
        details = backend.implementation_details()
        self.assertEqual(json.loads(json.dumps(details)), details)
        self.assertEqual((details["deep_ep"], details["num_sms"]), ("uccl deep_ep_wrapper", 20))
        # Config(20, 6, 256, 6, 128) and Config(20, 4, 256, 6, 128), with the (unused intranode)
        # RDMA receive chunk rounded up to a multiple of 6.
        self.assertEqual(list(details["dispatch_config"].values()), [20, 6, 256, 6, 132])
        self.assertEqual(list(details["combine_config"].values()), [20, 4, 256, 6, 132])
        self.assertEqual((details["num_nvl_bytes"], details["num_rdma_bytes"]), (GIB, 0))
        self.assertEqual((details["is_intranode"], details["cpu_proxy_threads"]), (True, 0))

    def test_ep16_adds_vllms_rdma_buffer_and_qps_and_publishes_the_realized_rounding(self):
        backend, buffer = _built(self.module, world_size=16)
        self.assertEqual(buffer.args, ("world", GIB, GIB))
        self.assertEqual((buffer.kwargs["num_qps_per_rank"], buffer.kwargs["is_intranode"]), (10, False))
        details = backend.implementation_details()
        # Config(20, 36, 288, 20, 128) realizes a 140-token RDMA receive chunk.
        self.assertEqual(list(details["dispatch_config"].values()), [20, 36, 288, 20, 140])
        self.assertEqual(list(details["combine_config"].values()), [20, 4, 288, 12, 132])
        self.assertEqual((details["is_intranode"], details["cpu_proxy_threads"]), (False, 4))

    def test_one_sm_budget_at_ep8_and_ep16_on_cuda_and_hip(self):
        budgets = set()
        for module in (self.module, _load(hip="7.2")):
            for world_size in (8, 16):
                realized = _built(module, world_size=world_size)[0].implementation_details()
                budgets |= {realized["num_sms"], realized["dispatch_config"]["num_sms"],
                            realized["combine_config"]["num_sms"]}
        self.assertEqual(budgets, {20})

    def test_dispatch_and_combine_pass_the_resolved_configs(self):
        backend, buffer = _built(self.module)
        handle = backend.dispatch(types.SimpleNamespace(
            T=4, dispatch_x="x", topk_idx="idx", topk_weights="weights"))
        backend.stage(None, handle)
        backend.combine(None, handle)
        (_, dispatched), (_, combined) = buffer.calls
        self.assertIs(dispatched["config"], backend.dispatch_config)
        self.assertEqual((dispatched["x"], dispatched["async_finish"]), ("x", False))
        self.assertIs(combined["config"], backend.combine_config)
        self.assertNotIn("topk_weights", combined)  # unweighted rank-sum, as vLLM's finalize passes None

    def test_fp8_scales_go_out_row_major_as_quantized(self):
        class RowMajorScales:
            def __getattr__(self, name):
                raise AssertionError(f"dispatch touched the scales layout: .{name}")

        backend, buffer = _built(self.module, precision="fp8")
        values, scales = object(), RowMajorScales()
        backend._quant = lambda x: (values, scales)
        backend.dispatch(types.SimpleNamespace(T=4, dispatch_x="x", topk_idx="i", topk_weights="w"))
        sent_values, sent_scales = buffer.calls[0][1]["x"]
        self.assertIs(sent_values, values)
        self.assertIs(sent_scales, scales)


class LowLatencyContract(unittest.TestCase):
    def setUp(self):
        self.module = _load()

    def test_receive_is_sized_to_the_fixed_budget_not_the_ladder(self):
        # vLLM passes max_num_batched_tokens (256 for batched-DP low latency) to the size hint and to
        # every dispatch, so a partial ladder runs the same receive plane as the full one.
        backend, buffer = _built(self.module, mode="low-latency", max_tokens=4)
        self.assertEqual(backend.max_tokens, 256)
        self.assertEqual(FakeBuffer.hints, [(256, 7168, 8, 384)])
        self.assertEqual(buffer.args, ("world", GIB, 4096))
        self.assertEqual(buffer.kwargs, {
            "low_latency_mode": True, "num_qps_per_rank": 48, "allow_nvlink_for_low_latency_mode": True,
            "allow_mnnvl": False, "explicitly_destroy": True, "is_intranode": True})
        self.assertEqual(backend.implementation_details()["receive_capacity"], 256)
        with self.assertRaisesRegex(RuntimeError, "exceeds the LL buffer cap"):
            _built(self.module, mode="low-latency", max_tokens=512)

    def test_both_collectives_split_into_send_and_receive_kernels_as_vllm_runs_them(self):
        backend, buffer = _built(self.module, mode="low-latency")
        problem = types.SimpleNamespace(T=4, dispatch_x="x", topk_idx="i", topk_weights="w")
        handle = backend.dispatch(problem)
        handle.combine_input = "staged"
        backend.combine(problem, handle)
        self.assertEqual([name for name, _ in buffer.calls],
                         ["ll-dispatch", "ll-dispatch-hook", "ll-combine", "ll-combine-hook"])
        for name, kwargs in buffer.calls[::2]:
            with self.subTest(name):
                self.assertEqual((kwargs["return_recv_hook"], kwargs["async_finish"]), (True, False))
        self.assertIs(buffer.calls[2][1]["zero_copy"], False)
        self.assertIs(backend.implementation_details()["return_recv_hook"], True)

    def test_only_intranode_low_latency_replays(self):
        # No per-SKU eager exception (b200 FP8 once had one); scale-out LL runs through the CPU
        # proxy, unvalidated under capture; normal host-syncs.
        intranode = _backend(self.module, mode="low-latency", runner="b200-nscale", precision="fp8")
        self.assertTrue(intranode.cuda_graph_supported)
        self.assertFalse(_backend(self.module, world_size=16, mode="low-latency").cuda_graph_supported)
        self.assertFalse(_backend(self.module).cuda_graph_supported)


class Provenance(unittest.TestCase):
    def setUp(self):
        self.module = _load()

    def test_nothing_is_published_before_the_buffer_exists(self):
        self.assertEqual(_backend(self.module).implementation_details(), {})

    def test_kernel_generations_split_the_vllm_normal_rows_from_the_test_script_rows(self):
        self.assertEqual(_backend(self.module).kernel_generation, "uccl-legacy-buffer-vllm-ht")
        self.assertEqual(_backend(self.module, mode="low-latency").kernel_generation, "uccl-legacy-buffer-ll-hook")

    def test_library_version_round_trips_the_record_the_prepare_step_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            record_path = Path(tmp) / "site" / "collx_uccl_build.json"
            (Path(tmp) / "site" / "deep_ep").mkdir(parents=True)
            written = _prepare(["uccl_record_build"], 'uccl_record_build "$1" sm90', tmp,
                               env={"COLLX_UCCL_COMMIT": PIN})
            self.assertEqual(written.returncode, 0, written.stderr)
            record = json.loads(record_path.read_text())
            self.assertEqual((record["commit"], record["arch"]), (PIN, "sm90"))
            deep_ep = types.SimpleNamespace(__file__=str(Path(tmp) / "site" / "deep_ep" / "__init__.py"))
            with mock.patch.object(self.module, "deep_ep", deep_ep), \
                    mock.patch.object(self.module.metadata, "version", return_value="0.1.0"):
                # The flags the build recorded are the flags every row publishes.
                self.assertEqual(self.module._library_version(),
                                 f"uccl@df000fd3001d deep_ep==0.1.0 {record['build_flags']}")
                record_path.unlink()
                self.assertEqual(self.module._library_version(), "uccl@unrecorded deep_ep==0.1.0")

    def test_the_build_keys_the_shared_cache(self):
        rooted = _prepare(["backend_cache_root", "uccl_cache_root"], 'uccl_cache_root "$1"', "9.0", env={
            "COLLX_BACKEND_CACHE_ROOT": "/cache", "COLLECTIVEX_IMAGE": "img", "COLLX_UCCL_COMMIT": PIN})
        self.assertEqual(rooted.returncode, 0, rooted.stderr)
        # cache root, backend, cpu, arch, image, commit prefix, then the build generation.
        self.assertRegex(rooted.stdout, r"^/cache/uccl-ep-[^/]+-9\.0-img-df000fd3001d-[^/-]+$")


class ImportProbe(unittest.TestCase):
    """prepare_backend.sh's uccl_probe must reject DeepSeek's DeepEP, a deep_ep loaded from outside
    the prepared cache, and a cache recorded for another commit."""

    UCCL_WRAPPER = ("from uccl.ep import Config\n"
                    "class Buffer:\n    low_latency_dispatch = get_dispatch_layout = None\n")
    DEEPSEEK = "class Config: pass\n" + UCCL_WRAPPER.split("\n", 1)[1]

    def _probe(self, wrapper=UCCL_WRAPPER, commit=PIN, shadow=False):
        with tempfile.TemporaryDirectory() as tmp:
            site, fakes = Path(tmp) / "root" / "site", Path(tmp) / "fakes"
            for directory in (site / "deep_ep", site / "uccl", fakes):
                directory.mkdir(parents=True)
            (site / "deep_ep" / "__init__.py").write_text(wrapper)
            (site / "uccl" / "__init__.py").write_text("")
            (site / "uccl" / "ep.py").write_text("class Config: pass\n")
            (fakes / "torch.py").write_text("")
            if shadow:  # a UCCL-shaped deep_ep ahead of the cache on the path
                (fakes / "deep_ep").mkdir()
                (fakes / "deep_ep" / "__init__.py").write_text(self.UCCL_WRAPPER)
            recorded = _prepare(["uccl_record_build"], 'uccl_record_build "$1" sm90', str(site.parent),
                                env={"COLLX_UCCL_COMMIT": commit})
            self.assertEqual(recorded.returncode, 0, recorded.stderr)
            return _prepare(["uccl_probe"], 'uccl_probe "$1"', str(site.parent), env={
                "COLLX_UCCL_COMMIT": PIN, "PYTHONPATH": os.pathsep.join((str(fakes), str(site)))})

    def test_uccls_wrapper_from_the_prepared_cache_passes(self):
        probed = self._probe()
        self.assertEqual(probed.returncode, 0, probed.stderr)

    def test_deepseeks_deep_ep_fails(self):
        self.assertIn("not UCCL's deep_ep_wrapper", self._probe(wrapper=self.DEEPSEEK).stderr)

    def test_a_deep_ep_shadowing_the_cache_fails(self):
        self.assertIn("deep_ep is not from", self._probe(shadow=True).stderr)

    def test_a_record_for_another_commit_fails(self):
        self.assertIn("build record is stale", self._probe(commit="0" * 40).stderr)


if __name__ == "__main__":
    unittest.main()
