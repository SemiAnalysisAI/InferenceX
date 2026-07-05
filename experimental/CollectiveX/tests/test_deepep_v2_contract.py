#!/usr/bin/env python3
"""CPU-only structural and registry tests for the pinned DeepEP V2 path."""
from __future__ import annotations

import ast
import argparse
import copy
import ctypes
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import shutil
import stat
import subprocess
import sys
import tempfile
import types
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import capability  # noqa: E402
import contracts  # noqa: E402
import ep_harness  # noqa: E402
import identity  # noqa: E402
import run_ep  # noqa: E402


COMMIT = "fa8a9b16898204afd347c663b89e65ef87dc6ce6"
TREE = "29809e75c5874e6609dac4804e7b651d5226959f"
FMT_COMMIT = "a4c7e17133ee9cb6a2f45545f6e974dd3c393efa"
NCCL_CHECK_COMMIT = "93d0564188f7a0a6288c6e316484861b0efa042e"


def deepep_v2_jit_provenance() -> list[dict[str, str]]:
    return [
        {
            "cache_key": f"kernel.{name}.{index:032x}",
            "cubin_sha256": f"{index + 1:x}" * 64,
            "sass_sha256": f"{index + 2:x}" * 64,
            "source_sha256": f"{index + 3:x}" * 64,
        }
        for index, name in enumerate(sorted(contracts.DEEPEP_V2_JIT_KERNELS))
    ]


def hybrid_realized_config() -> dict[str, object]:
    config = {field: 1 for field in contracts.HYBRID_REALIZED_CONFIG_FIELDS}
    for field in contracts.HYBRID_REALIZED_BOOL_FIELDS:
        config[field] = True
    config["token_data_type"] = "UINT16"
    return config


def hybrid_jit_provenance(ranks: int = 2) -> tuple[list[str], list[dict[str, object]]]:
    keys = ["combine-key", "dispatch-key", "preprocess-key"]
    artifacts = [
        {
            "kernel_key": key,
            "rank_artifacts": [
                {"bytes": 10 + index, "rank": rank, "sha256": f"{index + 1:x}" * 64}
                for rank in range(ranks)
            ],
        }
        for index, key in enumerate(keys)
    ]
    return keys, artifacts


def load_uccl_function(name: str, namespace: dict[str, object]):
    path = HERE / "ep_uccl.py"
    function = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def operator_config(root: Path) -> dict[str, object]:
    path = str(root)
    network = {"socket_ifname": "eth0", "rdma_devices": "mlx5_0:1"}
    runners = {
        "h100-dgxc": {
            "partition": "test", "account": "test", "squash_dir": path,
            "stage_dir": path, **network,
        },
        "h200-dgxc": {
            "partition": "test", "squash_dir": path, "stage_dir": path, **network,
        },
        "b200-dgxc": {
            "partition": "test", "account": "test", "squash_dir": path,
            "stage_dir": path, **network,
        },
        "b300": {
            "partition": "test", "account": "test", "squash_dir": path, "stage_dir": path,
            **network,
        },
        "gb200": {"partition": "test", "account": "test", "storage_roots": [path]},
        "gb300": {
            "partition": "test", "account": "test", "squash_dir": path,
            "stage_dir": path, "enroot_cache_path": path,
        },
        "mi325x": {
            "partition": "test", "squash_dir": path, "stage_dir": path, **network,
        },
        "mi355x": {
            "partition": "test", "squash_dir": path, "stage_dir": path, **network,
        },
    }
    return {"schema_version": 1, "audit_salt": "a" * 64, "runners": runners}


class DeepEPV2ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path = HERE / "ep_deepep_v2.py"
        cls.tree = ast.parse(cls.path.read_text(), str(cls.path))

    def test_capability_is_explicit_for_every_sku(self) -> None:
        backend = capability.BACKENDS["deepep-v2"]
        self.assertEqual(
            (backend["implementation"], backend["commit"], backend["torch"], backend["nccl"]),
            ("deep_ep.ElasticBuffer", COMMIT, "2.10.0+cu130", "2.30.4"),
        )
        self.assertEqual(backend["source"], "deepseek-ai/DeepEP#605+#630+#640")
        self.assertEqual(backend["communication_backend"], "nccl-device-lsa")
        self.assertEqual(set(backend["sku_capabilities"]), set(capability.PLATFORMS))
        for sku, platform in capability.PLATFORMS.items():
            ok, _ = capability.resolve(sku, "deepep-v2")
            self.assertEqual(ok, platform["vendor"] == "nvidia" and sku != "h100-dgxc")
            self.assertEqual(
                set(backend["sku_capabilities"][sku]), {"basis", "schedulable"}
            )
        self.assertEqual(
            backend["sku_capabilities"]["h100-dgxc"],
            {
                "schedulable": False,
                "basis": "current-runner-nccl-device-api-symmetric-memory-unavailable",
            },
        )

    def test_adapter_ast_pins_elastic_api_and_weight_semantics(self) -> None:
        imports = {
            alias.name
            for node in ast.walk(self.tree)
            if isinstance(node, ast.ImportFrom) and node.module == "deep_ep"
            for alias in node.names
        }
        self.assertEqual(imports, {"ElasticBuffer"})
        constants = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in self.tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
        }
        self.assertEqual(constants["DEEPEP_V2_COMMIT"], COMMIT)
        self.assertEqual(constants["DEEPEP_V2_TREE"], TREE)
        self.assertEqual(constants["DEEPEP_V2_FMT_COMMIT"], FMT_COMMIT)
        self.assertEqual(constants["DEEPEP_V2_PR"], 605)
        self.assertEqual(constants["DEEPEP_V2_FIX_PR"], 630)
        self.assertEqual(constants["DEEPEP_V2_NCCL_CHECK_FIX_PR"], 640)
        self.assertEqual(constants["DEEPEP_V2_NCCL_CHECK_COMMIT"], NCCL_CHECK_COMMIT)
        self.assertEqual(
            constants["DEEPEP_V2_JIT_RANDOM_SEED"],
            "collectivex-deepep-v2-fa8a9b1",
        )
        self.assertEqual(constants["NCCL_VERSION"], "2.30.4")
        self.assertEqual(constants["NVSHMEM_VERSION"], "3.3.9")
        backend = next(
            node for node in self.tree.body
            if isinstance(node, ast.ClassDef) and node.name == "DeepEPV2Backend"
        )
        assignments = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in backend.body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
        }
        self.assertEqual(assignments["combine_weight_semantics"], "unweighted-rank-sum")
        methods = {node.name for node in backend.body if isinstance(node, ast.FunctionDef)}
        self.assertTrue({
            "dispatch", "inspect_dispatch", "combine_transformed", "capture_deferred_provenance",
            "finalize",
        } <= methods)
        self.assertNotIn("expected", methods)
        constructor = next(
            node for node in ast.walk(backend)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "ElasticBuffer"
        )
        deterministic = next(
            keyword for keyword in constructor.keywords if keyword.arg == "deterministic"
        )
        self.assertIs(ast.literal_eval(deterministic.value), False)
        self.assertIn("deterministic", contracts.REQUIRED_BACKEND_PROVENANCE["deepep-v2"])
        self.assertIn("num_experts", contracts.REQUIRED_BACKEND_PROVENANCE["deepep-v2"])
        self.assertIn("tuning_num_experts", contracts.REQUIRED_BACKEND_PROVENANCE["deepep-v2"])
        self.assertIn("jit_random_seed", contracts.REQUIRED_BACKEND_PROVENANCE["deepep-v2"])
        self.assertIn("gin_enabled", contracts.REQUIRED_BACKEND_PROVENANCE["deepep-v2"])
        self.assertIn("communication_backend", contracts.REQUIRED_BACKEND_PROVENANCE["deepep-v2"])
        self.assertIn("deepep_pr", contracts.REQUIRED_BACKEND_PROVENANCE["deepep-v2"])
        self.assertIn("deepep_fix_pr", contracts.REQUIRED_BACKEND_PROVENANCE["deepep-v2"])
        source = self.path.read_text()
        self.assertIn('getattr(args, "num_logical_experts", args.experts)', source)
        self.assertIn('"use_expanded_layout": False', source)
        self.assertIn("allow_hybrid_mode = _configure_gin_mode(args, world_size)", source)
        self.assertIn("get_theoretical_num_sms(tuning_num_experts, args.topk)", source)

        jit_function = next(
            node for node in self.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_jit_cache_key"
        )
        namespace = {"hashlib": __import__("hashlib"), "json": json}
        exec(compile(ast.Module(body=[jit_function], type_ignores=[]), str(self.path), "exec"), namespace)
        key = namespace["_jit_cache_key"]
        baseline = types.SimpleNamespace(
            runner="h100-dgxc", hidden=7168, topk=8, experts=256,
            routing="uniform", eplb=False, case_id="uniform",
        )
        zipf = types.SimpleNamespace(**{**vars(baseline), "routing": "zipf", "case_id": "zipf"})
        eplb = types.SimpleNamespace(
            **{**vars(zipf), "experts": 288, "num_logical_experts": 256, "eplb": True}
        )
        realized = {
            "num_sms": 24,
            "num_qps": 9,
            "allocated_qps": 17,
            "logical_scaleout_ranks": 1,
            "logical_scaleup_ranks": 8,
            "physical_rdma_ranks": 2,
            "physical_nvlink_ranks": 4,
            "is_scaleup_nvlink": False,
            "device_arch_major": 9,
            "device_arch_minor": 0,
            "device_sms": 132,
            "device_smem_bytes": 232448,
            "gpu_timeout_cycles": 198000000000,
        }
        direct = key(baseline, 8, 128, False, realized)
        self.assertTrue(direct.startswith("jitcfg-v3-"))
        self.assertEqual(direct, key(zipf, 8, 128, False, realized))
        self.assertNotEqual(direct, key(zipf, 8, 128, True, realized))
        self.assertNotEqual(direct, key(eplb, 8, 128, False, realized))
        for field, value in realized.items():
            changed = not value if type(value) is bool else value + 1
            self.assertNotEqual(
                direct,
                key(baseline, 8, 128, False, {**realized, field: changed}),
                field,
            )
        init = next(
            node for node in backend.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        buffer_call = next(
            node for node in ast.walk(init)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "ElasticBuffer"
        )
        jit_config_check = next(
            node for node in ast.walk(init)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_require_cross_rank_equal"
            and ast.literal_eval(node.args[1]) == "JIT configuration"
        )
        cache_assignment = next(
            node for node in ast.walk(init)
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Subscript)
            and ast.unparse(node.targets[0].value) == "os.environ"
            and ast.literal_eval(node.targets[0].slice) == "EP_JIT_CACHE_DIR"
        )
        self.assertLess(buffer_call.lineno, jit_config_check.lineno)
        self.assertLess(jit_config_check.lineno, cache_assignment.lineno)
        capture = next(
            node for node in backend.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "capture_deferred_provenance"
        )
        calls = [node for node in ast.walk(capture) if isinstance(node, ast.Call)]
        barrier = next(
            node for node in calls
            if isinstance(node.func, ast.Attribute) and node.func.attr == "barrier"
        )
        self.assertEqual(
            {keyword.arg: ast.literal_eval(keyword.value) for keyword in barrier.keywords},
            {"use_comm_stream": True, "with_cpu_sync": True},
        )
        scan = next(
            node for node in calls
            if isinstance(node.func, ast.Name) and node.func.id == "_jit_artifact_evidence"
        )
        self.assertLess(barrier.lineno, scan.lineno)
        realized_check = next(
            node for node in ast.walk(backend)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_require_cross_rank_equal"
            and len(node.args) > 1
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == "realized tuning/topology"
        )
        self.assertIsInstance(realized_check, ast.Call)
        self.assertEqual(
            (ROOT / "tests" / "ep_harness.py").read_text().count(
                "capture_deferred_provenance()"
            ),
            2,
        )
        schema = json.loads((ROOT / "schemas" / "raw-case-v1.schema.json").read_text())
        provenance = schema["properties"]["implementation"]["properties"]["provenance"]
        self.assertEqual(provenance["properties"]["deterministic"], {"type": "boolean"})
        self.assertEqual(
            provenance["properties"]["num_experts"],
            {"minimum": 1, "type": "integer"},
        )
        self.assertEqual(
            provenance["properties"]["tuning_num_experts"],
            {"minimum": 1, "type": "integer"},
        )
        self.assertEqual(
            provenance["properties"]["jit_cubins"]["items"],
            {"$ref": "#/$defs/deepep_v2_jit_cubin"},
        )
        self.assertEqual(
            (
                provenance["properties"]["jit_cubins"]["minItems"],
                provenance["properties"]["jit_cubins"]["maxItems"],
            ),
            (5, 5),
        )
        self.assertEqual(
            provenance["properties"]["jit_random_seed"],
            {"const": "collectivex-deepep-v2-fa8a9b1"},
        )
        self.assertEqual(provenance["properties"]["allow_hybrid_mode"], {"type": "boolean"})
        self.assertEqual(provenance["properties"]["gin_enabled"], {"type": "boolean"})
        self.assertEqual(provenance["properties"]["deepep_pr"], {"const": 605})
        self.assertEqual(provenance["properties"]["deepep_fix_pr"], {"const": 630})
        self.assertEqual(
            provenance["properties"]["deepep_nccl_check_fix_pr"], {"const": 640}
        )
        self.assertEqual(
            provenance["properties"]["deepep_nccl_check_commit"],
            {"const": NCCL_CHECK_COMMIT},
        )
        self.assertEqual(
            provenance["properties"]["communication_backend"],
            {"enum": ["nccl-device-lsa", "nccl-gin"]},
        )
        self.assertEqual(
            provenance["properties"]["num_rdma_bytes"],
            {"minimum": 0, "type": "integer"},
        )
        self.assertEqual(
            provenance["properties"]["num_qps_per_rank"],
            {"minimum": 1, "type": "integer"},
        )
        for field, value in (
            ("num_experts", "288"),
            ("tuning_num_experts", "not-an-integer"),
            ("tuning_num_experts", 0),
        ):
            with self.subTest(provenance_field=field, value=value):
                self.assertIn(
                    field,
                    contracts.backend_provenance_issues(
                        "deepep-v2", {field: value}
                    ),
                )

    def test_v2_gin_mode_uses_the_scale_up_domain_and_safe_fallbacks(self) -> None:
        functions = {
            node.name: node for node in self.tree.body if isinstance(node, ast.FunctionDef)
        }
        namespace = {"os": os}
        exec(
            compile(
                ast.Module(
                    body=[
                        functions["_configure_gin_mode"],
                        functions["_lsa_topology_is_valid"],
                    ],
                    type_ignores=[],
                ),
                str(self.path),
                "exec",
            ),
            namespace,
        )
        configure = namespace["_configure_gin_mode"]
        topology_is_valid = namespace["_lsa_topology_is_valid"]
        original = os.environ.get("EP_DISABLE_GIN")
        try:
            args = types.SimpleNamespace(scale_up_domain=72, gpus_per_node=4)
            self.assertFalse(configure(args, 8))
            self.assertEqual(os.environ.get("EP_DISABLE_GIN"), "1")

            os.environ["EP_DISABLE_GIN"] = "stale"
            args = types.SimpleNamespace(scale_up_domain=8, gpus_per_node=4)
            self.assertTrue(configure(args, 16))
            self.assertNotIn("EP_DISABLE_GIN", os.environ)

            args = types.SimpleNamespace(gpus_per_node=4)
            self.assertTrue(configure(args, 8))
            self.assertNotIn("EP_DISABLE_GIN", os.environ)

            self.assertFalse(configure(types.SimpleNamespace(), 8))
            self.assertEqual(os.environ.get("EP_DISABLE_GIN"), "1")

            topology = {
                "physical_rdma_ranks": 1,
                "physical_nvlink_ranks": 8,
                "logical_scaleout_ranks": 1,
                "logical_scaleup_ranks": 8,
                "is_scaleup_nvlink": True,
            }
            self.assertTrue(topology_is_valid(False, 8, 8, topology))
            topology["physical_rdma_ranks"] = 2
            topology["logical_scaleout_ranks"] = 2
            self.assertTrue(topology_is_valid(True, 16, 8, topology))
            topology["physical_nvlink_ranks"] = 4
            self.assertFalse(topology_is_valid(False, 8, 8, topology))
        finally:
            if original is None:
                os.environ.pop("EP_DISABLE_GIN", None)
            else:
                os.environ["EP_DISABLE_GIN"] = original

    def test_ep_adapters_declare_unweighted_rank_sum(self) -> None:
        adapters = {
            "ep_deepep.py": "DeepEPBackend",
            "ep_deepep_v2.py": "DeepEPV2Backend",
            "ep_deepep_hybrid.py": "DeepEPHybridBackend",
            "ep_mori.py": "MoRIBackend",
            "ep_nccl.py": "NCCLBackend",
            "ep_uccl.py": "UCCLBackend",
        }
        for filename, class_name in adapters.items():
            with self.subTest(adapter=filename):
                tree = ast.parse((HERE / filename).read_text())
                backend = next(
                    node for node in tree.body
                    if isinstance(node, ast.ClassDef) and node.name == class_name
                )
                assignment = next(
                    node for node in backend.body
                    if isinstance(node, ast.Assign)
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "combine_weight_semantics"
                )
                self.assertEqual(ast.literal_eval(assignment.value), "unweighted-rank-sum")
                combine_methods = [
                    item for item in backend.body
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name in {"combine", "combine_transformed"}
                ]
                self.assertEqual(len(combine_methods), 2)
                for method in combine_methods:
                    source = ast.unparse(method)
                    if filename in {"ep_deepep.py", "ep_uccl.py"}:
                        self.assertIn("self.mode == 'low-latency'", source)
                    else:
                        self.assertNotIn("topk_weights", source)
                        self.assertNotIn("combine_topk_weights", source)

    def test_low_latency_mode_parser_and_profile_are_explicit(self) -> None:
        parser = argparse.ArgumentParser()
        ep_harness.add_common_args(parser)
        required = [
            "--runner", "test", "--topology-class", "test",
            "--scope", "scale-up", "--scale-up-transport", "nvlink",
            "--out", "test.json",
        ]
        self.assertEqual(parser.parse_args(required).mode, "normal")
        self.assertEqual(
            parser.parse_args([*required, "--mode", "low-latency"]).mode,
            "low-latency",
        )
        profile = identity.case_profile("low-latency")
        self.assertEqual(profile["contract"], "expert-packed-weighted-combine-v1")
        self.assertEqual(
            profile["component_order_contract"],
            "qualification-hash-rotated-components-v1",
        )
        self.assertEqual(
            profile["correctness_scope"],
            "expert-assignment-and-weighted-combine",
        )
        self.assertEqual(profile["payload_unit"], "token-expert")

    def test_expert_packed_slot_map_reconstructs_exact_sources(self) -> None:
        pack = lambda begin, count: (begin << 32) | count
        slots = ep_harness.expert_packed_slot_map(
            [2, 1],
            [[1, 0, 0, 0], [1, 0, 0, 0]],
            [[pack(0, 1), pack(1, 1)], [pack(0, 0), pack(0, 1)]],
            tokens_per_rank=2,
            experts_per_rank=2,
            world_size=2,
        )
        self.assertEqual(slots, [(0, 0, 1), (0, 1, 2), (1, 0, 3)])

        invalid = (
            ([1], [[0]], [[pack(1, 1), pack(0, 0)]]),
            ([1], [[2]], [[pack(0, 1), pack(1, 0)]]),
            ([2], [[1, 1]], [[pack(0, 2), pack(2, 0)]]),
        )
        for counts, source, layout in invalid:
            with self.subTest(counts=counts, source=source, layout=layout):
                with self.assertRaises(ValueError):
                    ep_harness.expert_packed_slot_map(
                        counts,
                        source,
                        layout,
                        tokens_per_rank=2,
                        experts_per_rank=1,
                        world_size=2,
                    )

    def test_deepep_and_uccl_expose_genuine_low_latency_calls(self) -> None:
        required_fragments = (
            "Buffer.get_low_latency_rdma_size_hint(",
            "low_latency_mode=True",
            "num_qps_per_rank=num_qps_per_rank",
            "self.buffer.clean_low_latency_buffer(",
            "self.buffer.low_latency_dispatch(",
            "use_fp8=False",
            "self.buffer.low_latency_combine(",
            "p.topk_weights",
            'self.combine_weight_semantics = "gate-weighted-sum"',
            "self.combine_needs_redispatch = True",
            "def inspect_expert_dispatch(",
        )
        for filename in ("ep_deepep.py", "ep_uccl.py"):
            source = (HERE / filename).read_text()
            with self.subTest(adapter=filename):
                for fragment in required_fragments:
                    self.assertIn(fragment, source)
                self.assertIn("self.max_tokens_per_rank = 128", source)
                self.assertIn("async_finish=False", source)
                self.assertIn("return_recv_hook=False", source)

        run_ep_source = (HERE / "run_ep.py").read_text()
        self.assertIn('args.backend not in {"deepep", "uccl"}', run_ep_source)
        self.assertIn('args.phase != "decode"', run_ep_source)

    def test_deepep_v2_jit_evidence_is_strict_and_stable(self) -> None:
        valid = deepep_v2_jit_provenance()
        self.assertTrue(contracts._deepep_v2_jit_cubins_are_valid(valid))
        for invalid in (
            [],
            [{**valid[0], "path": "/private/kernel.cubin"}],
            [{**item, "cache_key": "dispatch"} for item in valid],
            [{**item, "cubin_sha256": "invalid"} for item in valid],
            valid[:-1],
            [*valid, valid[0]],
            [
                *valid,
                {
                    **valid[0],
                    "cache_key": valid[0]["cache_key"][:-32] + "f" * 32,
                },
            ],
        ):
            with self.subTest(invalid=invalid):
                self.assertFalse(contracts._deepep_v2_jit_cubins_are_valid(invalid))

        backend = next(
            node for node in self.tree.body
            if isinstance(node, ast.ClassDef) and node.name == "DeepEPV2Backend"
        )
        capture = next(
            node for node in backend.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "capture_deferred_provenance"
        )
        artifacts = copy.deepcopy(valid)

        class FakeBuffer:
            @staticmethod
            def barrier(*, use_comm_stream: bool, with_cpu_sync: bool) -> None:
                self.assertTrue(use_comm_stream)
                self.assertTrue(with_cpu_sync)

        namespace = {
            "torch": types.SimpleNamespace(
                cuda=types.SimpleNamespace(synchronize=lambda: None)
            ),
            "_jit_artifact_evidence": lambda: copy.deepcopy(artifacts),
            "_require_cross_rank_equal": lambda _value, _label: None,
        }
        exec(
            compile(ast.Module(body=[capture], type_ignores=[]), str(self.path), "exec"),
            namespace,
        )
        state = types.SimpleNamespace(
            buffer=FakeBuffer(),
            _deferred_jit_snapshot=None,
            backend_provenance={"jit_cubins": []},
        )
        namespace["capture_deferred_provenance"](state)
        namespace["capture_deferred_provenance"](state)
        artifacts[0]["cubin_sha256"] = "f" * 64
        with self.assertRaisesRegex(RuntimeError, "changed after measurement"):
            namespace["capture_deferred_provenance"](state)

    def test_deepep_v2_jit_files_are_complete_regular_and_content_bound(self) -> None:
        functions = [
            node for node in self.tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"_sha256", "_jit_artifact_evidence"}
        ]
        namespace = {
            "hashlib": hashlib,
            "os": os,
            "Path": Path,
            "re": __import__("re"),
            "DEEPEP_V2_JIT_KERNELS": contracts.DEEPEP_V2_JIT_KERNELS,
        }
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(self.path), "exec"), namespace)
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "cache"
            cache.mkdir()
            for index, name in enumerate(sorted(contracts.DEEPEP_V2_JIT_KERNELS)):
                kernel = cache / f"kernel.{name}.{index:032x}"
                kernel.mkdir()
                for suffix in ("cu", "cubin", "sass"):
                    (kernel / f"kernel.{suffix}").write_bytes(f"{name}-{suffix}".encode())
            old_cache = os.environ.get("EP_JIT_CACHE_DIR")
            os.environ["EP_JIT_CACHE_DIR"] = temporary
            try:
                evidence = namespace["_jit_artifact_evidence"]()
                self.assertEqual(len(evidence), len(contracts.DEEPEP_V2_JIT_KERNELS))
                self.assertEqual(
                    set(evidence[0]),
                    {"cache_key", "cubin_sha256", "sass_sha256", "source_sha256"},
                )
                first = cache / evidence[0]["cache_key"]
                duplicate = cache / (evidence[0]["cache_key"][:-32] + "f" * 32)
                duplicate.mkdir()
                for suffix in ("cu", "cubin", "sass"):
                    (duplicate / f"kernel.{suffix}").write_bytes(b"duplicate")
                with self.assertRaisesRegex(RuntimeError, "kernel set"):
                    namespace["_jit_artifact_evidence"]()
                shutil.rmtree(duplicate)
                (first / "kernel.sass").unlink()
                with self.assertRaisesRegex(RuntimeError, "incomplete"):
                    namespace["_jit_artifact_evidence"]()
                (first / "kernel.sass").symlink_to(first / "kernel.cubin")
                with self.assertRaisesRegex(RuntimeError, "regular file"):
                    namespace["_jit_artifact_evidence"]()
            finally:
                if old_cache is None:
                    os.environ.pop("EP_JIT_CACHE_DIR", None)
                else:
                    os.environ["EP_JIT_CACHE_DIR"] = old_cache

    def test_runtime_and_shared_version_formatter_are_valid(self) -> None:
        subprocess.run(
            ["bash", "-n", str(ROOT / "runtime" / "run_in_container.sh")],
            check=True,
        )
        self.assertEqual(ep_harness.format_collective_version(23004), "2.30.4")
        self.assertEqual(ep_harness.format_collective_version((2, 30, 4)), "2.30.4")
        source = self.path.read_text()
        version_function = next(
            node for node in self.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_loaded_nccl_version"
        )

        class FakeNccl:
            @staticmethod
            def ncclGetVersion(pointer) -> int:
                pointer._obj.value = 23004
                return 0

        namespace = {
            "ctypes": types.SimpleNamespace(
                CDLL=lambda _path: FakeNccl(), byref=ctypes.byref, c_int=ctypes.c_int,
            ),
            "ep_harness": ep_harness,
            "os": os,
            "_loaded_library_paths": lambda: {"/safe/libnccl.so.2"},
        }
        exec(
            compile(ast.Module(body=[version_function], type_ignores=[]), str(self.path), "exec"),
            namespace,
        )
        self.assertEqual(namespace["_loaded_nccl_version"](), "2.30.4")
        for paths in (set(), {"/safe/libnccl.so.2", "/other/libnccl.so.2"}):
            namespace["_loaded_library_paths"] = lambda paths=paths: paths
            with self.assertRaisesRegex(RuntimeError, "exactly one"):
                namespace["_loaded_nccl_version"]()
        evidence_function = next(
            node for node in self.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_loaded_library_evidence"
        )
        paths = {
            "/safe/_C.cpython-310-x86_64-linux-gnu.so",
            "/safe/libnccl.so.2",
            "/safe/libnvshmem_host.so.3",
        }
        namespace.update(
            _loaded_library_paths=lambda: paths,
            _sha256=lambda _path: "a" * 64,
        )
        exec(
            compile(ast.Module(body=[evidence_function], type_ignores=[]), str(self.path), "exec"),
            namespace,
        )
        evidence = namespace["_loaded_library_evidence"]()
        self.assertIn(
            {"name": "deep_ep._C", "role": "deepep-extension", "sha256": "a" * 64},
            evidence,
        )
        self.assertTrue(
            contracts._content_evidence_is_valid(
                evidence, {"deepep-extension", "nccl", "nvshmem"}
            )
        )
        self.assertNotIn("torch.cuda.nccl.version()", source)
        fingerprint = {"runtime": "cuda", "version": "13.0"}
        self.assertIs(
            run_ep._common_runtime_fingerprint([fingerprint, dict(fingerprint)]),
            fingerprint,
        )
        with self.assertRaises(ValueError):
            run_ep._common_runtime_fingerprint([fingerprint, {"runtime": "cuda", "version": "12.8"}])

    def test_conditioning_contract_is_exact_for_each_phase(self) -> None:
        expected = {
            "decode": [1, 2, 4, 8, 16, 32, 64, 128],
            "prefill": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
        }
        for phase, ladder in expected.items():
            valid = {
                "contract": "fixed-phase-ramp-8-roundtrips-v1",
                "ladder": ladder,
                "roundtrips_per_shape": 8,
            }
            self.assertIs(contracts.validate_conditioning_contract(valid, phase), valid)
            for mutate in (
                lambda item: item["ladder"].reverse(),
                lambda item: item["ladder"].pop(),
                lambda item: item.update(ladder=[1.0, *item["ladder"][1:]]),
                lambda item: item.update(roundtrips_per_shape=7),
                lambda item: item.update(roundtrips_per_shape=8.0),
            ):
                changed = copy.deepcopy(valid)
                mutate(changed)
                with self.assertRaises(contracts.ContractError):
                    contracts.validate_conditioning_contract(changed, phase)
            other = "prefill" if phase == "decode" else "decode"
            with self.assertRaises(contracts.ContractError):
                contracts.validate_conditioning_contract(valid, other)

    def test_content_manifest_evidence_is_stable_and_content_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second = root / "first", root / "second"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            files = [("pkg/first", first), ("pkg/second", second)]
            evidence = contracts.content_manifest_evidence(
                role="test-content", name="test-build", files=files,
            )
            self.assertNotIn(temporary, json.dumps(evidence))
            self.assertEqual(
                evidence,
                contracts.content_manifest_evidence(
                    role="test-content", name="test-build", files=reversed(files),
                ),
            )
            self.assertRegex(evidence["sha256"], r"^[0-9a-f]{64}$")
            second.write_bytes(b"changed")
            self.assertNotEqual(
                evidence,
                contracts.content_manifest_evidence(
                    role="test-content", name="test-build", files=files,
                ),
            )
            for invalid in (
                [("../first", first)],
                [("same", first), ("same", second)],
                [("missing", root / "missing")],
            ):
                with self.assertRaises(contracts.ContractError):
                    contracts.content_manifest_evidence(
                        role="test-content", name="test-build", files=invalid,
                    )

    def test_hybrid_realized_config_and_jit_evidence_are_path_free(self) -> None:
        path = HERE / "ep_deepep_hybrid.py"
        tree = ast.parse(path.read_text(), str(path))
        selected = [
            node for node in tree.body
            if (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "HYBRID_CONFIG_FIELDS"
                    for target in node.targets
                )
            )
            or isinstance(node, ast.FunctionDef)
            and node.name in {
                "_hybrid_realized_config", "_sha256_with_size", "_hybrid_jit_evidence",
            }
        ]
        namespace = {"Path": Path, "hashlib": hashlib, "re": __import__("re")}
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
        fields = namespace["HYBRID_CONFIG_FIELDS"]
        self.assertEqual(set(fields), contracts.HYBRID_REALIZED_CONFIG_FIELDS)

        class TokenType:
            def __init__(self, label: str, name: str | None = None) -> None:
                self.label = label
                if name is not None:
                    self.name = name

            def __str__(self) -> str:
                return self.label

        values = {field: 1 for field in fields}
        values.update({field: True for field in contracts.HYBRID_REALIZED_BOOL_FIELDS})
        for raw, expected in (("uint16_t", "UINT16"), ("uint8_t", "UINT8")):
            values["token_data_type"] = TokenType(raw)
            config = types.SimpleNamespace(**values)
            realized = namespace["_hybrid_realized_config"](config)
            self.assertEqual(realized["token_data_type"], expected)
            self.assertEqual(set(realized), contracts.HYBRID_REALIZED_CONFIG_FIELDS)
        values["token_data_type"] = TokenType("opaque-enum", "UINT16")
        self.assertEqual(
            namespace["_hybrid_realized_config"](types.SimpleNamespace(**values))[
                "token_data_type"
            ],
            "UINT16",
        )
        values["token_data_type"] = TokenType("UINT16")
        with self.assertRaisesRegex(RuntimeError, "token_data_type is invalid"):
            namespace["_hybrid_realized_config"](types.SimpleNamespace(**values))
        values["token_data_type"] = TokenType("uint16_t")
        config = types.SimpleNamespace(**values)
        delattr(config, "hidden_dim")
        with self.assertRaisesRegex(RuntimeError, "omits hidden_dim"):
            namespace["_hybrid_realized_config"](config)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for key, payload in (
                ("preprocess-key", b"pre"),
                ("combine-key", b"combine"),
                ("dispatch-key", b"dispatch"),
            ):
                (root / f"{key}.so").write_bytes(payload)
            evidence = namespace["_hybrid_jit_evidence"](root)
            self.assertEqual(
                [item["kernel_key"] for item in evidence],
                ["combine-key", "dispatch-key", "preprocess-key"],
            )
            self.assertNotIn(temporary, json.dumps(evidence))
            (root / "dispatch-key.so").write_bytes(b"changed")
            self.assertNotEqual(evidence, namespace["_hybrid_jit_evidence"](root))
            (root / "extra-key.so").write_bytes(b"extra")
            with self.assertRaisesRegex(RuntimeError, "expected 3"):
                namespace["_hybrid_jit_evidence"](root)
            (root / "extra-key.so").unlink()
            (root / "bad key.so").write_bytes(b"bad")
            with self.assertRaisesRegex(RuntimeError, "kernel key"):
                namespace["_hybrid_jit_evidence"](root)
            (root / "bad key.so").unlink()
            (root / "combine-key.so").unlink()
            (root / "combine-key.so").symlink_to(root / "dispatch-key.so")
            with self.assertRaisesRegex(RuntimeError, "regular file"):
                namespace["_hybrid_jit_evidence"](root)
            empty = root / "empty"
            empty.mkdir()
            with self.assertRaisesRegex(RuntimeError, "expected 3"):
                namespace["_hybrid_jit_evidence"](empty)

    def test_hybrid_uses_communication_domains_not_physical_hosts(self) -> None:
        path = HERE / "ep_deepep_hybrid.py"
        function = next(
            node for node in ast.parse(path.read_text(), str(path)).body
            if isinstance(node, ast.FunctionDef) and node.name == "_hybrid_topology"
        )
        namespace: dict[str, object] = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        resolve = namespace["_hybrid_topology"]
        cases = (
            (8, 8, 8, "scale-up", "nvlink", "", 8, 1, 1),
            (16, 8, 8, "scale-out", "nvlink", "rdma", 8, 2, 2),
            (8, 4, 72, "scale-up", "mnnvl", "", 8, 1, 2),
            (16, 4, 72, "scale-up", "mnnvl", "", 16, 1, 4),
        )
        for world, gpn, domain, scope, up, out, ranks, domains, hosts in cases:
            with self.subTest(world=world, gpus_per_node=gpn, transport=up):
                topology = resolve(types.SimpleNamespace(
                    gpus_per_node=gpn,
                    scale_up_domain=domain,
                    scope=scope,
                    scale_up_transport=up,
                    scale_out_transport=out,
                    transport=up if not out else f"{up}-{out}",
                ), world)
                self.assertEqual(
                    (topology["domain_ranks"], topology["communication_domains"],
                     topology["physical_nodes"]),
                    (ranks, domains, hosts),
                )
        with self.assertRaisesRegex(RuntimeError, "outside the fixed v1 matrix"):
            resolve(types.SimpleNamespace(
                gpus_per_node=8, scale_up_domain=8, scope="scale-up",
                scale_up_transport="nvlink", scale_out_transport="", transport="nvlink",
            ), 16)

    def test_mori_ep16_pins_upstream_internode_v1_resources(self) -> None:
        source = (HERE / "ep_mori.py").read_text()
        for fragment in (
            'kernel_enum.InterNodeV1',
            'self.block_num = self._block_target = 96',
            'self.rdma_block_num = 64',
            'self.dispatch_warps = self.combine_warps = 8',
            'self.num_qps = 1',
            '"gpu_per_node": gpus_per_node',
            '"rdma_block_num": self.rdma_block_num',
            '"num_qp_per_pe": self.num_qps',
            '"use_external_inp_buf": self._external_input',
            'os.environ["MORI_EP_LAUNCH_CONFIG_MODE"] = "MANUAL"',
            'rdma_block_num=self.rdma_block_num',
        ):
            self.assertIn(fragment, source)
        self.assertGreaterEqual(source.count("rdma_block_num=self.rdma_block_num"), 2)

    def test_hybrid_deferred_provenance_wraps_before_conditioning_and_recaptures(self) -> None:
        path = HERE / "ep_deepep_hybrid.py"
        source = path.read_text()
        tree = ast.parse(source, str(path))
        backend = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "DeepEPHybridBackend"
        )
        methods = {node.name for node in backend.body if isinstance(node, ast.FunctionDef)}
        self.assertIn("capture_deferred_provenance", methods)
        constructor = next(node for node in backend.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
        buffer_call = next(
            node for node in ast.walk(constructor)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "HybridEPBuffer"
        )
        wrapper_install = next(
            node for node in ast.walk(constructor)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Attribute)
                and target.attr == "update_template_config"
                for target in node.targets
            )
        )
        cache_line = source[:source.index('os.environ["HYBRID_EP_CACHE_DIR"]')].count("\n") + 1
        self.assertLess(cache_line, buffer_call.lineno)
        self.assertLess(buffer_call.lineno, wrapper_install.lineno)

        capture = next(
            node for node in backend.body
            if isinstance(node, ast.FunctionDef) and node.name == "capture_deferred_provenance"
        )
        called = {
            node.func.id if isinstance(node.func, ast.Name) else node.func.attr
            for node in ast.walk(capture) if isinstance(node, ast.Call)
            and isinstance(node.func, (ast.Name, ast.Attribute))
        }
        self.assertTrue({"_hybrid_jit_evidence", "_require_cross_rank_equal", "all_gather_object"} <= called)
        self.assertIn("changed after measurement", ast.get_source_segment(source, capture))

        artifacts = [[
            {"bytes": 1, "kernel_key": key, "sha256": digit * 64}
            for key, digit in (("a", "1"), ("b", "2"), ("c", "3"))
        ]]

        class FakeCuda:
            @staticmethod
            def synchronize() -> None:
                return None

        class FakeDist:
            @staticmethod
            def barrier() -> None:
                return None

            @staticmethod
            def get_world_size() -> int:
                return 2

            @staticmethod
            def all_gather_object(output, value) -> None:
                output[:] = [copy.deepcopy(value), copy.deepcopy(value)]

        namespace = {
            "torch": types.SimpleNamespace(cuda=FakeCuda),
            "dist": FakeDist,
            "_hybrid_jit_evidence": lambda _root: copy.deepcopy(artifacts[0]),
            "_require_cross_rank_equal": lambda _value, _label: None,
        }
        exec(compile(ast.Module(body=[capture], type_ignores=[]), str(path), "exec"), namespace)
        state = types.SimpleNamespace(
            _deferred_jit_diagnostics=None,
            _deferred_semantic_snapshot=None,
            _jit_root=Path("private-cache"),
            _realized_config=hybrid_realized_config(),
            backend_provenance={},
        )
        namespace["capture_deferred_provenance"](state)
        artifacts[0][0]["kernel_key"] = "changed"
        with self.assertRaisesRegex(RuntimeError, "kernel set changed"):
            namespace["capture_deferred_provenance"](state)
        artifacts[0][0]["kernel_key"] = "a"
        artifacts[0][0]["sha256"] = "f" * 64
        with self.assertRaisesRegex(RuntimeError, "artifacts changed"):
            namespace["capture_deferred_provenance"](state)

        harness = (HERE / "ep_harness.py").read_text()
        captures = [
            index for index in range(len(harness))
            if harness.startswith("capture_deferred_provenance()", index)
        ]
        self.assertEqual(len(captures), 2)
        self.assertLess(harness.index("for wt in conditioning_ladder:"), captures[0])
        self.assertLess(captures[0], harness.index("oracle = _run_expert_oracle("))
        self.assertLess(harness.index("trace_sig = hashlib.sha256"), captures[1])

    def test_hybrid_diagnostic_hashes_do_not_split_series_identity(self) -> None:
        keys, artifacts = hybrid_jit_provenance()
        provenance = {
            "deepep_tree": "b" * 40,
            "jit_kernel_keys": keys,
            "jit_shared_objects": artifacts,
            "loaded_libraries": [{
                "name": "hybrid_ep_cpp", "role": "deepep-hybrid-extension",
                "sha256": "a" * 64,
            }],
            "realized_config": hybrid_realized_config(),
        }
        baseline = ep_harness._series_provenance(provenance)
        changed = copy.deepcopy(provenance)
        changed["jit_shared_objects"][0]["rank_artifacts"][0]["sha256"] = "f" * 64
        self.assertEqual(ep_harness._series_provenance(changed), baseline)
        changed = copy.deepcopy(provenance)
        changed["loaded_libraries"][0]["sha256"] = "f" * 64
        self.assertEqual(ep_harness._series_provenance(changed), baseline)
        changed = copy.deepcopy(provenance)
        changed["jit_kernel_keys"][0] = "changed-key"
        self.assertNotEqual(ep_harness._series_provenance(changed), baseline)
        changed = copy.deepcopy(provenance)
        changed["realized_config"]["num_of_blocks_dispatch_api"] += 1
        self.assertNotEqual(ep_harness._series_provenance(changed), baseline)
        changed = copy.deepcopy(provenance)
        changed["deepep_tree"] = "c" * 40
        self.assertNotEqual(ep_harness._series_provenance(changed), baseline)

    def test_v2_series_identity_uses_source_and_sass_not_container_metadata(self) -> None:
        provenance = {
            "deepep_tree": "a" * 40,
            "loaded_libraries": [
                {"name": "deep_ep._C.so", "role": "deepep-extension", "sha256": "1" * 64},
                {"name": "libnccl.so.2", "role": "nccl", "sha256": "2" * 64},
            ],
            "jit_cubins": deepep_v2_jit_provenance(),
            "jit_random_seed": "collectivex-deepep-v2-fa8a9b1",
        }
        baseline = contracts.series_provenance(provenance)
        changed = copy.deepcopy(provenance)
        changed["loaded_libraries"][0]["sha256"] = "f" * 64
        changed["jit_cubins"][0]["cubin_sha256"] = "e" * 64
        self.assertEqual(contracts.series_provenance(changed), baseline)
        for mutate in (
            lambda item: item["loaded_libraries"][1].update(sha256="f" * 64),
            lambda item: item["jit_cubins"][0].update(source_sha256="f" * 64),
            lambda item: item["jit_cubins"][0].update(sass_sha256="f" * 64),
            lambda item: item.update(deepep_tree="f" * 40),
        ):
            changed = copy.deepcopy(provenance)
            mutate(changed)
            self.assertNotEqual(contracts.series_provenance(changed), baseline)

    def test_mnnvl_resolution_has_no_ambiguous_signature_fallback(self) -> None:
        self.assertEqual(
            contracts.resolve_deepep_mnnvl(
                requested=False, signature_parameters=(), deepep_commit=None,
            ),
            ({}, "not-requested"),
        )
        self.assertEqual(
            contracts.resolve_deepep_mnnvl(
                requested=True, signature_parameters=("allow_mnnvl",),
                deepep_commit="a" * 40,
            ),
            ({"allow_mnnvl": True}, "explicit-allow-mnnvl"),
        )
        with self.assertRaises(contracts.ContractError):
            contracts.resolve_deepep_mnnvl(
                requested=True, signature_parameters=(),
                deepep_commit="814e508537c6ffc775d59f6f1b9ba43f3a65968c",
            )

    def test_backend_provenance_requires_lineage_and_content_hashes(self) -> None:
        def record(role: str, name: str, digit: str) -> dict[str, str]:
            return {"role": role, "name": name, "sha256": digit * 64}

        hybrid_keys, hybrid_artifacts = hybrid_jit_provenance()
        v2 = {
            **contracts.DEEPEP_V2_V1_PROVENANCE,
            "api_signature_sha256": "c" * 64,
            "loaded_libraries": [
                record("deepep-extension", "deep_ep._C", "1"),
                record("nccl", "libnccl.so.2", "2"),
                record("nvshmem", "libnvshmem_host.so.3", "3"),
            ],
            "jit_cubins": deepep_v2_jit_provenance(),
            "jit_random_seed": "collectivex-deepep-v2-fa8a9b1",
            "deterministic": False,
            "num_experts": 256,
            "tuning_num_experts": 256,
            "allow_hybrid_mode": False,
            "gin_enabled": False,
            "communication_backend": "nccl-device-lsa",
        }
        deepep = {
            "deepep_version": "1.1.0", "deepep_commit": "a" * 40,
            "backend_lineage": "deepep-v1", "allow_mnnvl": False,
            "mnnvl_comm": "not-requested", "mode": "normal",
            "num_nvl_bytes": 1024, "num_rdma_bytes": 0,
        }
        hybrid = {
            "deepep_commit": "a" * 40, "deepep_tree": "b" * 40,
            "branch": "hybrid-ep", "backend_lineage": "deepep-hybrid",
            "loaded_libraries": [
                record("deepep-extension", "deep_ep_cpp", "1"),
                record("deepep-hybrid-extension", "hybrid_ep_cpp", "2"),
            ],
            "jit_kernel_keys": hybrid_keys,
            "jit_shared_objects": hybrid_artifacts,
            "realized_config": hybrid_realized_config(),
        }
        uccl = {
            "uccl_version": "0.1.1", "uccl_commit": "pkg-0.1.1",
            "uccl_wrapper_commit": "c" * 40, "backend_lineage": "uccl",
            "uccl_dependency_versions": dict(contracts.UCCL_DEPENDENCY_VERSIONS),
            "loaded_libraries": [
                record("uccl-distribution", "uccl-0.1.1", "3"),
                record("uccl-wrapper", "uccl-deepep-wrapper", "4"),
                record("intervaltree-distribution", "intervaltree-3.1.0", "5"),
                record("sortedcontainers-distribution", "sortedcontainers-2.4.0", "6"),
                record("cuda-runtime", "nvidia-cuda-runtime-cu12-12.9.79", "7"),
            ],
            "mode": "normal", "num_nvl_bytes": 1024, "num_rdma_bytes": 0,
        }
        reference = {
            "nccl_version": "2.30.4", "collective_library": "nccl",
            "backend_lineage": "nccl",
        }
        for backend, provenance in (
            ("deepep", deepep), ("deepep-v2", v2), ("deepep-hybrid", hybrid),
            ("uccl", uccl), ("nccl-ep", reference),
        ):
            self.assertEqual(contracts.backend_provenance_issues(backend, provenance), [])
            changed = copy.deepcopy(provenance)
            if "loaded_libraries" in changed:
                changed["loaded_libraries"][0]["sha256"] = "invalid"
                expected = "loaded_libraries"
            else:
                changed["backend_lineage"] = "wrong"
                expected = "backend_lineage"
            self.assertIn(expected, contracts.backend_provenance_issues(backend, changed))

        changed = copy.deepcopy(uccl)
        changed["uccl_dependency_versions"]["intervaltree"] = "3.2.0"
        self.assertIn(
            "uccl_dependency_versions",
            contracts.backend_provenance_issues("uccl", changed),
        )
        changed = copy.deepcopy(uccl)
        changed["loaded_libraries"] = [
            item
            for item in changed["loaded_libraries"]
            if item["role"] != "sortedcontainers-distribution"
        ]
        self.assertIn(
            "loaded_libraries", contracts.backend_provenance_issues("uccl", changed)
        )

        for field, mutate in (
            ("realized_config", lambda item: item["realized_config"].pop("hidden_dim")),
            ("jit_kernel_keys", lambda item: item["jit_kernel_keys"].reverse()),
            (
                "jit_shared_objects",
                lambda item: item["jit_shared_objects"][0]["rank_artifacts"][0].update(
                    sha256="invalid"
                ),
            ),
        ):
            with self.subTest(hybrid_field=field):
                changed = copy.deepcopy(hybrid)
                mutate(changed)
                self.assertIn(
                    field,
                    contracts.backend_provenance_issues("deepep-hybrid", changed),
                )

        for field, value in (
            ("jit_cubins", [{"cache_key": "invalid", "cubin_sha256": "4" * 64}]),
            ("jit_random_seed", "different-seed"),
        ):
            with self.subTest(v2_field=field):
                changed = copy.deepcopy(v2)
                changed[field] = value
                self.assertIn(
                    field,
                    contracts.backend_provenance_issues("deepep-v2", changed),
                )

        changed = copy.deepcopy(v2)
        changed["gin_enabled"] = True
        self.assertIn("gin_enabled", contracts.backend_provenance_issues("deepep-v2", changed))
        changed = copy.deepcopy(v2)
        changed["communication_backend"] = "nccl-gin"
        self.assertIn(
            "communication_backend", contracts.backend_provenance_issues("deepep-v2", changed)
        )
        changed = copy.deepcopy(v2)
        changed.update(
            allow_hybrid_mode=True,
            gin_enabled=True,
            communication_backend="nccl-gin",
        )
        self.assertEqual(
            contracts.backend_provenance_issues("deepep-v2", changed),
            [],
        )
        changed["allow_hybrid_mode"] = False
        self.assertEqual(
            contracts.backend_provenance_issues("deepep-v2", changed),
            ["allow_hybrid_mode", "communication_backend", "gin_enabled"],
        )
        for field, expected in contracts.DEEPEP_V2_V1_PROVENANCE.items():
            with self.subTest(v2_pin_field=field):
                changed = copy.deepcopy(v2)
                changed[field] = not expected if type(expected) is bool else "wrong"
                self.assertIn(
                    field,
                    contracts.backend_provenance_issues("deepep-v2", changed),
                )

        schema = json.loads((ROOT / "schemas" / "raw-case-v1.schema.json").read_text())
        provenance_schema = schema["properties"]["implementation"]["properties"]["provenance"]
        self.assertEqual(
            provenance_schema["properties"]["realized_config"],
            {"$ref": "#/$defs/hybrid_realized_config"},
        )
        self.assertFalse(schema["$defs"]["hybrid_realized_config"]["additionalProperties"])
        self.assertEqual(provenance_schema["properties"]["jit_kernel_keys"]["minItems"], 3)
        self.assertEqual(provenance_schema["properties"]["jit_shared_objects"]["minItems"], 3)

        self.assertEqual(contracts.collective_kernel_generation("nccl"), "nccl")
        self.assertEqual(contracts.collective_kernel_generation("rccl"), "rccl")
        with self.assertRaises(contracts.ContractError):
            contracts.collective_kernel_generation("unknown")

    def test_transport_resource_provenance_is_exact(self) -> None:
        self.assertEqual(contracts.hybrid_communication_domains(8, 8), (8, 1))
        self.assertEqual(contracts.hybrid_communication_domains(16, 8), (8, 2))
        self.assertEqual(contracts.hybrid_communication_domains(8, 72), (8, 1))
        self.assertEqual(contracts.hybrid_communication_domains(16, 72), (16, 1))

        profile = contracts.project_resource_profile({
            "num_nvl_bytes": 1024, "num_rdma_bytes": 2048,
            "num_qps_per_rank": 32, "heap_size": "6G",
        })
        self.assertEqual(profile["persistent_bytes"], 3072)
        self.assertEqual(profile["qps_per_rank"], 32)
        self.assertEqual(
            contracts.project_resource_profile({
                "num_nvl_bytes": 0, "num_rdma_bytes": 0, "heap_size": "6G",
            })["persistent_bytes"],
            0,
        )
        self.assertEqual(
            contracts.project_resource_profile({"heap_size": "6G"})[
                "persistent_bytes"
            ],
            "6G",
        )

        mori = {
            "mori_commit": "a" * 40, "kernel_type": "InterNodeV1",
            "block_num": 96, "rdma_block_num": 64,
            "dispatch_warps": 8, "combine_warps": 8, "num_qps": 1,
            "use_external_inp_buf": True, "gpus_per_node": 8,
        }
        self.assertEqual(contracts.backend_provenance_issues("mori", mori), [])
        for field in (
            "block_num", "rdma_block_num", "dispatch_warps", "combine_warps",
            "num_qps", "use_external_inp_buf", "gpus_per_node",
        ):
            changed = copy.deepcopy(mori)
            changed[field] = False if field == "use_external_inp_buf" else 0
            with self.subTest(mori_field=field):
                self.assertIn(
                    field, contracts.backend_provenance_issues("mori", changed)
                )

    def test_routing_control_binds_binary_but_allows_treatment_configuration(self) -> None:
        hybrid_keys, hybrid_artifacts = hybrid_jit_provenance()
        implementation = {
            "kernel_generation": "hybrid",
            "name": "deepep-hybrid",
            "provenance": {
                "deepep_tree": "a" * 40,
                "loaded_libraries": [{
                    "role": "deepep-extension", "name": "deep_ep_cpp", "sha256": "1" * 64,
                }],
                "local_experts": 32,
                "num_experts": 256,
                "num_sms": 24,
                "jit_cache_key": "case-one",
                "jit_cubins": [{"cache_key": "one", "cubin_sha256": "2" * 64}],
                "jit_kernel_keys": hybrid_keys,
                "jit_shared_objects": hybrid_artifacts,
                "realized_config": hybrid_realized_config(),
            },
            "resource_profile": {"configured_units": 24},
        }
        baseline = contracts.routing_implementation_control_sha256(implementation)
        treatment = copy.deepcopy(implementation)
        treatment["provenance"].update({
            "local_experts": 36,
            "num_experts": 288,
            "jit_cache_key": "case-two",
            "jit_cubins": [{"cache_key": "two", "cubin_sha256": "3" * 64}],
            "jit_kernel_keys": ["changed-a", "changed-b", "changed-c"],
            "jit_shared_objects": hybrid_jit_provenance(3)[1],
            "realized_config": {
                **hybrid_realized_config(),
                "num_of_experts_per_rank": 36,
            },
        })
        self.assertEqual(
            contracts.routing_implementation_control_sha256(treatment), baseline
        )
        changed = copy.deepcopy(implementation)
        changed["provenance"]["loaded_libraries"][0]["sha256"] = "4" * 64
        self.assertEqual(
            contracts.routing_implementation_control_sha256(changed), baseline
        )
        changed = copy.deepcopy(implementation)
        changed["provenance"]["deepep_tree"] = "b" * 40
        self.assertNotEqual(
            contracts.routing_implementation_control_sha256(changed), baseline
        )
        changed = copy.deepcopy(implementation)
        changed["provenance"]["num_sms"] = 20
        self.assertNotEqual(
            contracts.routing_implementation_control_sha256(changed), baseline
        )

    def test_runtime_pins_uccl_wheel_and_hybrid_source_tree(self) -> None:
        runtime = (ROOT / "runtime" / "run_in_container.sh").read_text()
        common = (ROOT / "runtime" / "common.sh").read_text()
        self.assertIn("cd /ix/experimental/CollectiveX", runtime)
        for launcher_name in ("launch_single-slurm.sh", "launch_gb-nv.sh"):
            launcher = (ROOT / "launchers" / launcher_name).read_text()
            self.assertIn("$MOUNT_SRC:/ix", launcher)
            self.assertIn("cx_prepare_backend_cache", launcher)
            self.assertNotIn('$(cx_prepare_backend_cache', launcher)
            self.assertIn("$CX_PREPARED_BACKEND_CACHE:/cx-cache", launcher)
            self.assertIn("CX_BACKEND_CACHE_ROOT=/cx-cache", launcher)
            self.assertIn("CX_BACKEND_SOURCE_ROOT=/ix/experimental/CollectiveX/.cx_sources", launcher)
            self.assertIn('|| [ "$CX_BENCH" = deepep-hybrid ]', launcher)
            self.assertIn("cx_prepare_backend_source", launcher)
            cache_block = launcher[launcher.index('if [ "$CX_BENCH" = deepep-v2 ]'):]
            self.assertLess(
                cache_block.index("cx_set_failure_stage backend-setup"),
                cache_block.index("cx_prepare_backend_cache"),
            )
            self.assertLess(
                cache_block.index("cx_prepare_backend_source"),
                cache_block.index("cx_set_failure_stage scheduler-allocation"),
            )
        self.assertIn("--frandom-seed=$seed", runtime)
        self.assertIn("DEEPEP_V2_JIT_RANDOM_SEED", runtime)
        persisted = runtime[runtime.index("cx_persist_backend_env()") :]
        self.assertIn("CUDA_HOME CPATH NVCC_PREPEND_FLAGS", persisted)
        self.assertIn(
            "390c1320918972206546e44d79b132988f2818ec07e23afcd0595f7183916cec",
            runtime,
        )
        self.assertIn("--require-hashes", runtime)
        self.assertIn('$PWD/.cx_backend/uccl-wrapper-node-$node_id', runtime)
        self.assertNotIn("/tmp/uccl_deepep_pkg", runtime)
        self.assertIn('export PYTHONPATH="$wrapper_root:${PYTHONPATH:-}"', runtime)
        self.assertIn("d77aeab7f1bb52b615666fe178d26ced41fae08e", common)
        self.assertIn("HEAD^{tree}", runtime)
        self.assertIn("$PWD/.cx_backend/deepep-hybrid-", runtime)
        self.assertIn("cx_materialize_backend_source deepep-hybrid", runtime)
        self.assertIn("cx_materialize_backend_source deepep-v2", runtime)
        self.assertIn("cx_deepep_hybrid_marker_content_sha256", runtime)
        self.assertIn("cx_deepep_hybrid_cache_is_valid", runtime)
        self.assertIn("cx_extension_pair_sha256", runtime)
        self.assertIn(".collectivex-complete.tmp.", runtime)
        self.assertNotIn("cx_fetch_revision", runtime)
        self.assertIn("cx_fetch_revision", common)
        self.assertIn("third-party/fmt", common)
        hybrid = runtime[
            runtime.index("cx_build_deepep_hybrid()"):
            runtime.index("# UCCL EP")
        ]
        configure = runtime[
            runtime.index("cx_configure_deepep_hybrid_build()"):
            runtime.index("cx_deepep_hybrid_marker_content_sha256()")
        ]
        self.assertIn("cx_prepare_cuda_cccl", hybrid)
        self.assertIn("unset NVSHMEM_DIR", hybrid)
        self.assertIn(
            "unset HYBRID_EP_MULTINODE USE_NIXL RDMA_CORE_HOME", configure
        )
        self.assertIn("cx_configure_deepep_hybrid_build || return 1", hybrid)
        self.assertIn('[ "$(uname -m)" = x86_64 ]', configure)
        self.assertIn('[ -n "${GLOO_SOCKET_IFNAME:-}" ]', configure)
        self.assertIn('[ -d "/sys/class/infiniband/$rdma_name" ]', configure)
        self.assertIn("command -v make", configure)
        self.assertIn("/usr/include/infiniband/verbs.h", configure)
        self.assertIn("export HYBRID_EP_MULTINODE=1 USE_NIXL=0", configure)
        self.assertNotIn("cx_prepare_deepep_toolchain", hybrid)
        toolchain = runtime[
            runtime.index("cx_prepare_deepep_toolchain()"):
            runtime.index("cx_probe_deepep()")
        ]
        self.assertIn('overlay="$root/nvshmem-overlay"', toolchain)
        self.assertIn("flock 8 || exit 1", toolchain)
        self.assertIn('mv "$temporary" "$overlay" || exit 1', toolchain)
        self.assertNotIn("/tmp/collectivex-nvshmem", toolchain)
        jit = runtime[
            runtime.index("cx_enable_deepep_v2_jit_reproducibility()"):
            runtime.index("cx_probe_deepep_v2()")
        ]
        self.assertIn('cccl="${CX_CUDA_CCCL:-}"', jit)
        self.assertNotIn("/usr/local/cuda*", jit)
        self.assertIn("deepep-v2-cache-v3|$cpu|sm${arch/./}", runtime)
        self.assertNotIn("deepep-v2-cache-v1|", runtime)
        self.assertIn('base="${CX_BACKEND_CACHE_ROOT:-}"', runtime)
        self.assertNotIn("${CX_BACKEND_CACHE_ROOT:-$PWD/.cx_backend}", runtime)
        self.assertIn(
            "recipe=aot-persistent-nvshmem-active-cuda-maxjobs16-v3", runtime
        )
        self.assertNotIn("recipe=aot-source-date-epoch-arch-maxjobs16-v1", runtime)
        self.assertNotIn("recipe=$source_sha", runtime)
        self.assertIn("pip=26.1.2|setuptools=82.0.1|wheel=0.47.0|ninja=1.13.0", runtime)
        self.assertIn("manual-unverified", runtime)
        self.assertIn("cx_deepep_v2_content_sha256", runtime)
        self.assertIn("DeepEP V2 cache validation failed", runtime)
        probe = runtime[
            runtime.index("cx_probe_deepep_v2()"):
            runtime.index("cx_deepep_v2_content_sha256()")
        ]
        self.assertNotIn("torch.cuda.nccl.version", probe)
        self.assertIn("ncclGetVersion", probe)
        self.assertIn("runtime_version.value == 23004", probe)
        self.assertIn("cx_nvidia_package_root nvidia-nccl-cu13 nccl", runtime)
        self.assertIn("cx_nvidia_package_root nvidia-nvshmem-cu12 nvshmem", runtime)
        self.assertNotIn("import os,nvidia.nccl", runtime)
        self.assertNotIn("import os,nvidia.nvshmem", runtime)
        self.assertIn(
            'export EP_JIT_CACHE_DIR="$stage_root/.cx_backend/deepep-v2-jit"', runtime
        )
        self.assertIn('stage_root="${CX_BACKEND_SOURCE_ROOT%/.cx_sources}"', runtime)
        self.assertNotIn('export EP_JIT_CACHE_DIR="$root/jit"', runtime)
        self.assertIn('EP_NVSHMEM_ROOT_DIR="$NVSHMEM_DIR"', runtime)
        reference = (HERE / "ep_nccl.py").read_text()
        self.assertIn("self.kernel_generation = contracts.collective_kernel_generation", reference)
        self.assertIn("scale-out collective network mode is not IB", reference)
        self.assertIn("scale-out collective HCA allowlist is invalid", reference)
        self.assertNotIn("scale-out collective transport is not pinned to RDMA", reference)

    def test_deepep_v2_cache_recovers_from_an_unpublished_partial_build(self) -> None:
        runtime = ROOT / "runtime" / "run_in_container.sh"
        with tempfile.TemporaryDirectory() as temporary:
            cache_key = "a" * 64
            content_hash = "b" * 64
            root = Path(temporary) / f"deepep-v2-{cache_key}"
            root.mkdir(mode=0o700)
            marker = root / ".collectivex-complete"
            stale = root / "stale-partial-build"
            stale.write_text("partial\n")
            command = r'''
              set -euo pipefail
              eval "$(sed -n '/^cx_build_deepep_v2()/,/^}/p' "$1")"
              cache_root="$2"; expected_revision="$3"; expected_tree="$4"; expected_fmt="$5"
              expected_content="$6"
              cx_log() { :; }
              cx_verify_backend_cache_mount() { return 0; }
              cx_cuda_arch() { printf '9.0'; }
              cx_deepep_v2_root() { printf '%s' "$cache_root"; }
              cx_activate_deepep_v2() { export DEEPEP_V2_COMMIT="$expected_revision"; }
              cx_prepare_deepep_toolchain() { export NVSHMEM_DIR=/tmp/cx-test-nvshmem; }
              cx_probe_deepep_v2() { return 0; }
              cx_deepep_v2_content_sha256() { printf '%s' "$expected_content"; }
              cx_deepep_v2_cache_is_valid() {
                test -f "$2" && test "$(wc -l < "$2" | tr -d ' ')" = 5
              }
              cx_enable_deepep_v2_jit_reproducibility() { return 0; }
              cx_materialize_backend_source() { mkdir -p "$2/third-party/fmt"; }
              flock() { return 0; }
              python3() {
                if [ "${1:-}" = -m ] && [ "${2:-}" = venv ]; then
                  mkdir -p "$3/bin"
                  printf '#!/bin/sh\nexit 0\n' > "$3/bin/python"
                  chmod 700 "$3/bin/python"
                fi
                return 0
              }
              git() {
                case " $* " in
                  *' third-party/fmt rev-parse HEAD '*) printf '%s\n' "$expected_fmt" ;;
                  *' rev-parse HEAD^{tree} '*) printf '%s\n' "$expected_tree" ;;
                  *' show -s --format=%ct HEAD '*) printf '1\n' ;;
                  *) return 0 ;;
                esac
              }
              cx_git_in_tree() { shift; git "$@"; }
              cx_build_deepep_v2
            '''
            subprocess.run(
                [
                    "bash", "-c", command, "_", str(runtime), str(root),
                    COMMIT, TREE, FMT_COMMIT, content_hash,
                ],
                check=True,
            )
            self.assertFalse(stale.exists())
            self.assertEqual(
                marker.read_text(),
                f"{COMMIT}\n{TREE}\n{FMT_COMMIT}\n{cache_key}\n{content_hash}\n",
            )
            self.assertEqual(list(root.glob(".collectivex-complete.tmp.*")), [])

    def test_deepep_v2_published_cache_is_never_deleted_after_probe_failure(self) -> None:
        runtime = ROOT / "runtime" / "run_in_container.sh"
        with tempfile.TemporaryDirectory() as temporary:
            cache_key = "a" * 64
            root = Path(temporary) / f"deepep-v2-{cache_key}"
            root.mkdir(mode=0o700)
            marker = root / ".collectivex-complete"
            marker.write_text("published\n")
            sentinel = root / "active-reader"
            sentinel.write_text("active\n")
            command = r'''
              set -euo pipefail
              eval "$(sed -n '/^cx_build_deepep_v2()/,/^}/p' "$1")"
              cache_root="$2"
              cx_log() { :; }
              cx_verify_backend_cache_mount() { return 0; }
              cx_cuda_arch() { printf '9.0'; }
              cx_deepep_v2_root() { printf '%s' "$cache_root"; }
              cx_deepep_v2_cache_is_valid() { return 0; }
              cx_activate_deepep_v2() { return 0; }
              cx_prepare_deepep_toolchain() { return 0; }
              cx_enable_deepep_v2_jit_reproducibility() { return 0; }
              cx_probe_deepep_v2() { return 1; }
              ! cx_build_deepep_v2
            '''
            subprocess.run(
                ["bash", "-c", command, "_", str(runtime), str(root)],
                check=True,
            )
            self.assertEqual(sentinel.read_text(), "active\n")
            self.assertEqual(marker.read_text(), "published\n")

    def test_deepep_v2_corrupt_published_cache_fails_without_reset(self) -> None:
        runtime = ROOT / "runtime" / "run_in_container.sh"
        with tempfile.TemporaryDirectory() as temporary:
            cache_key = "a" * 64
            root = Path(temporary) / f"deepep-v2-{cache_key}"
            root.mkdir(mode=0o700)
            marker = root / ".collectivex-complete"
            marker.write_text("corrupt\n")
            sentinel = root / "active-reader"
            sentinel.write_text("active\n")
            command = r'''
              set -euo pipefail
              eval "$(sed -n '/^cx_build_deepep_v2()/,/^}/p' "$1")"
              cache_root="$2"
              cx_log() { :; }
              cx_verify_backend_cache_mount() { return 0; }
              cx_cuda_arch() { printf '9.0'; }
              cx_deepep_v2_root() { printf '%s' "$cache_root"; }
              cx_deepep_v2_cache_is_valid() { return 1; }
              flock() { return 0; }
              ! cx_build_deepep_v2
            '''
            subprocess.run(
                ["bash", "-c", command, "_", str(runtime), str(root)],
                check=True,
            )
            self.assertEqual(sentinel.read_text(), "active\n")
            self.assertEqual(marker.read_text(), "corrupt\n")

    def test_deepep_v2_marker_requires_private_owned_cache_objects(self) -> None:
        runtime = ROOT / "runtime" / "run_in_container.sh"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cache"
            root.mkdir(mode=0o700)
            (root / "source").mkdir(mode=0o700)
            (root / "venv").mkdir(mode=0o700)
            marker = root / ".collectivex-complete"
            cache_key = "a" * 64
            content_hash = "b" * 64
            marker.write_text(
                f"{COMMIT}\n{TREE}\n{FMT_COMMIT}\n{cache_key}\n{content_hash}\n"
            )
            root.chmod(0o2700)
            marker.chmod(0o600)
            command = r'''
              set -euo pipefail
              eval "$(sed -n '/^cx_deepep_v2_marker_content_sha256()/,/^}/p' "$1")"
              cx_deepep_v2_marker_content_sha256 "$2" "$3" "$4" "$5" "$6" "$7"
            '''
            args = [
                "bash", "-c", command, "_", str(runtime), str(root), str(marker),
                COMMIT, TREE, FMT_COMMIT, cache_key,
            ]
            valid = subprocess.run(args, text=True, capture_output=True, check=True)
            self.assertEqual(valid.stdout, content_hash)
            marker.chmod(0o644)
            self.assertNotEqual(subprocess.run(args).returncode, 0)

    def test_deepep_hybrid_marker_requires_a_private_regular_file(self) -> None:
        runtime = ROOT / "runtime" / "run_in_container.sh"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cache"
            root.mkdir(mode=0o700)
            marker = root / ".collectivex-complete"
            content_hash = "b" * 64
            marker.write_text(f"{COMMIT}\n{TREE}\n{content_hash}\n")
            root.chmod(0o2700)
            marker.chmod(0o600)
            command = r'''
              set -euo pipefail
              eval "$(sed -n '/^cx_deepep_hybrid_marker_content_sha256()/,/^}/p' "$1")"
              cx_deepep_hybrid_marker_content_sha256 "$2" "$3" "$4" "$5"
            '''
            args = [
                "bash", "-c", command, "_", str(runtime), str(root), str(marker),
                COMMIT, TREE,
            ]
            valid = subprocess.run(args, text=True, capture_output=True, check=True)
            self.assertEqual(valid.stdout, content_hash)
            marker_contract = runtime.read_text()
            marker_contract = marker_contract[
                marker_contract.index("cx_deepep_hybrid_marker_content_sha256()"):
                marker_contract.index("cx_deepep_hybrid_cache_is_valid()")
            ]
            self.assertIn("marker_item.st_uid != root_item.st_uid", marker_contract)
            self.assertNotIn("st_uid != os.getuid()", marker_contract)
            marker.chmod(0o644)
            self.assertNotEqual(subprocess.run(args).returncode, 0)

    def test_deepep_v2_installed_content_digest_binds_every_distribution_file(self) -> None:
        runtime = ROOT / "runtime" / "run_in_container.sh"
        with tempfile.TemporaryDirectory() as temporary:
            site = Path(temporary) / "venv" / "lib" / "python3.11" / "site-packages"
            package = site / "deep_ep"
            info = site / "deep_ep-2.0.0.dist-info"
            package.mkdir(parents=True)
            info.mkdir()
            (package / "__init__.py").write_text("__version__ = '2.0.0'\n")
            extension = package / "_C.so"
            extension.write_bytes(b"extension-one")
            (info / "METADATA").write_text(
                "Metadata-Version: 2.1\nName: deep_ep\nVersion: 2.0.0\n"
            )
            (info / "RECORD").write_text(
                "deep_ep/__init__.py,,\n"
                "deep_ep/_C.so,,\n"
                "deep_ep-2.0.0.dist-info/METADATA,,\n"
                "deep_ep-2.0.0.dist-info/RECORD,,\n"
            )
            command = r'''
              set -euo pipefail
              eval "$(sed -n '/^cx_deepep_v2_content_sha256()/,/^}/p' "$1")"
              cx_deepep_v2_content_sha256
            '''
            env = {
                **os.environ,
                "PYTHONPATH": str(site),
                "VIRTUAL_ENV": str(Path(temporary) / "venv"),
            }
            first = subprocess.run(
                ["bash", "-c", command, "_", str(runtime)],
                text=True, capture_output=True, check=True, env=env,
            ).stdout
            extension.write_bytes(b"extension-two")
            second = subprocess.run(
                ["bash", "-c", command, "_", str(runtime)],
                text=True, capture_output=True, check=True, env=env,
            ).stdout
            self.assertRegex(first, r"^[0-9a-f]{64}$")
            self.assertRegex(second, r"^[0-9a-f]{64}$")
            self.assertNotEqual(first, second)
            extension.unlink()
            outside = Path(temporary) / "outside.so"
            outside.write_bytes(b"outside")
            extension.symlink_to(outside)
            self.assertNotEqual(
                subprocess.run(
                    ["bash", "-c", command, "_", str(runtime)], env=env,
                ).returncode,
                0,
            )

    def test_uccl_content_identity_excludes_install_generated_files(self) -> None:
        keep = load_uccl_function(
            "_is_uccl_runtime_payload", {"PurePosixPath": PurePosixPath}
        )
        self.assertTrue(keep("uccl/ep.abi3.so"))
        self.assertTrue(keep("uccl.libs/libnuma.so"))
        self.assertFalse(keep("uccl/__pycache__/collective.cpython-312.pyc"))
        self.assertFalse(keep("uccl-0.1.1.dist-info/RECORD"))

    def test_uccl_dependency_versions_are_exact(self) -> None:
        installed = dict(contracts.UCCL_DEPENDENCY_VERSIONS)
        dependency_versions = load_uccl_function(
            "_uccl_dependency_versions",
            {
                "contracts": contracts,
                "metadata": types.SimpleNamespace(
                    version=lambda package: installed[package]
                ),
            },
        )
        self.assertEqual(dependency_versions(), contracts.UCCL_DEPENDENCY_VERSIONS)
        installed["intervaltree"] = "3.2.0"
        with self.assertRaisesRegex(RuntimeError, "differ from the v1 contract"):
            dependency_versions()

        schema = json.loads((ROOT / "schemas" / "raw-case-v1.schema.json").read_text())
        dependency_schema = schema["properties"]["implementation"]["properties"][
            "provenance"
        ]["properties"]["uccl_dependency_versions"]
        self.assertFalse(dependency_schema["additionalProperties"])
        self.assertEqual(
            {
                package: definition["const"]
                for package, definition in dependency_schema["properties"].items()
            },
            contracts.UCCL_DEPENDENCY_VERSIONS,
        )

    def test_uccl_support_dependency_content_is_path_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_entry = PurePosixPath("intervaltree/__init__.py")
            cache_entry = PurePosixPath("intervaltree/__pycache__/__init__.pyc")
            metadata_entry = PurePosixPath("intervaltree-3.1.0.dist-info/RECORD")
            for entry in (source_entry, cache_entry, metadata_entry):
                path = root / entry
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(entry.as_posix().encode())
            distribution = types.SimpleNamespace(
                files=[source_entry, cache_entry, metadata_entry],
                locate_file=lambda item: root / item,
            )
            evidence_for = load_uccl_function(
                "_python_dependency_evidence",
                {
                    "Path": Path,
                    "PurePosixPath": PurePosixPath,
                    "contracts": contracts,
                    "metadata": types.SimpleNamespace(
                        distribution=lambda package: distribution
                    ),
                },
            )
            evidence = evidence_for("intervaltree", "3.1.0")
            self.assertEqual(
                evidence,
                contracts.content_manifest_evidence(
                    role="intervaltree-distribution",
                    name="intervaltree-3.1.0",
                    files=[(source_entry.as_posix(), root / source_entry)],
                ),
            )
            self.assertNotIn(str(root), json.dumps(evidence))

    def test_uccl_hashes_the_mapped_pinned_libcudart_without_exposing_paths(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = PurePosixPath("nvidia/cuda_runtime/lib/libcudart.so.12")
            library = root / entry
            library.parent.mkdir(parents=True)
            library.write_bytes(b"pinned CUDA 12 runtime")
            distribution = types.SimpleNamespace(
                files=[entry],
                locate_file=lambda item: root / item,
            )
            evidence_for = load_uccl_function(
                "_loaded_libcudart_evidence",
                {
                    "Path": Path,
                    "PurePosixPath": PurePosixPath,
                    "contracts": contracts,
                    "metadata": types.SimpleNamespace(
                        distribution=lambda package: distribution
                    ),
                },
            )
            maps = root / "maps"
            maps.write_text(f"7f00-7f10 r-xp 00000000 00:00 0 {library}\n")
            evidence = evidence_for("12.9.79", maps)
            self.assertEqual(
                evidence,
                contracts.content_manifest_evidence(
                    role="cuda-runtime",
                    name="nvidia-cuda-runtime-cu12-12.9.79",
                    files=[("libcudart.so", library)],
                ),
            )
            self.assertNotIn(str(root), json.dumps(evidence))

            unowned = root / "unowned" / library.name
            unowned.parent.mkdir()
            unowned.write_bytes(library.read_bytes())
            maps.write_text(f"7f00-7f10 r-xp 00000000 00:00 0 {unowned}\n")
            with self.assertRaisesRegex(RuntimeError, "not owned") as raised:
                evidence_for("12.9.79", maps)
            self.assertNotIn(str(root), str(raised.exception))

    def test_private_runtime_logs_are_not_public_artifacts(self) -> None:
        path = subprocess.check_output(
            [
                "bash", "-c", 'source "$1"; cx_private_log_path test', "_",
                str(ROOT / "runtime" / "common.sh"),
            ],
            text=True,
            env={**os.environ, "COLLECTIVEX_EXECUTION_ID": "contract-test"},
        ).strip()
        try:
            log = Path(path)
            self.assertEqual(stat.S_IMODE(log.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(log.parent.stat().st_mode), 0o700)
            self.assertFalse(log.is_relative_to(ROOT))
        finally:
            shutil.rmtree(Path(path).parent, ignore_errors=True)

    def test_private_runtime_logs_reject_traversal_and_symlinks(self) -> None:
        common = str(ROOT / "runtime" / "common.sh")
        for variable, value in (
            ("COLLECTIVEX_EXECUTION_ID", ".."),
            ("CX_TEST_LABEL", ".."),
        ):
            environment = {
                **os.environ,
                "COLLECTIVEX_EXECUTION_ID": "contract-adversarial",
                "CX_TEST_LABEL": "test",
                variable: value,
            }
            result = subprocess.run(
                ["bash", "-c", 'source "$1"; cx_private_log_path "$CX_TEST_LABEL"', "_", common],
                text=True,
                capture_output=True,
                env=environment,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn(value, result.stderr)

        private_root = Path(f"/tmp/inferencex-collectivex-{os.getuid()}")
        private_root.mkdir(mode=0o700, exist_ok=True)
        self.assertFalse(private_root.is_symlink())
        os.chmod(private_root, 0o700)
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)
            tag = f"contract-symlink-{os.getpid()}"
            link = private_root / tag
            link.symlink_to(target, target_is_directory=True)
            try:
                result = subprocess.run(
                    ["bash", "-c", 'source "$1"; cx_private_log_path test', "_", common],
                    text=True,
                    capture_output=True,
                    env={**os.environ, "COLLECTIVEX_EXECUTION_ID": tag},
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(list(target.iterdir()), [])
            finally:
                link.unlink(missing_ok=True)

            tag = f"contract-log-symlink-{os.getpid()}"
            directory = private_root / tag
            directory.mkdir(mode=0o700)
            target_file = target / "target"
            target_file.write_text("unchanged")
            log_link = directory / "test.log"
            log_link.symlink_to(target_file)
            try:
                result = subprocess.run(
                    ["bash", "-c", 'source "$1"; cx_private_log_path test', "_", common],
                    text=True,
                    capture_output=True,
                    env={**os.environ, "COLLECTIVEX_EXECUTION_ID": tag},
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(target_file.read_text(), "unchanged")
            finally:
                log_link.unlink(missing_ok=True)
                directory.rmdir()

    def test_operator_config_failure_is_value_free(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "operator.env"
            config.write_text("printf 'private-config-token\\n' >&2\nfalse\n")
            config.chmod(0o600)
            result = subprocess.run(
                ["bash", "-c",
                 'export COLLECTIVEX_EXECUTION_ID="operator-failure-$$"; '
                 "trap 'cx_cleanup_private_logs 0' EXIT; source \"$1\"; "
                 "cx_load_operator_config", "_",
                 str(ROOT / "runtime" / "common.sh")],
                text=True,
                capture_output=True,
                env={
                    **os.environ,
                    "CX_RUNNER": "h100-dgxc",
                    "COLLECTIVEX_OPERATOR_CONFIG": str(config),
                },
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("runner-local configuration failed", result.stderr)
            self.assertNotIn("private-config-token", result.stderr)

    def test_ephemeral_operator_config_is_removed_after_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "operator.env"
            decoy = Path(temporary) / "decoy"
            decoy.write_text("keep")
            config.write_text(json.dumps(operator_config(Path(temporary) / "storage")))
            config.chmod(0o600)
            result = subprocess.run(
                [
                    "bash", "-c",
                    'export COLLECTIVEX_EXECUTION_ID="operator-ephemeral-$$"; '
                    "trap 'cx_cleanup_private_logs 0' EXIT; "
                    'config="$COLLECTIVEX_OPERATOR_CONFIG"; source "$1"; '
                    'cx_load_operator_config; test ! -e "$config"; '
                    'test "$CX_PARTITION" = test; '
                    'test -z "${COLLECTIVEX_OPERATOR_CONFIG+x}"',
                    "_", str(ROOT / "runtime" / "common.sh"),
                ],
                text=True,
                capture_output=True,
                env={
                    **os.environ,
                    "CX_RUNNER": "h100-dgxc",
                    "COLLECTIVEX_OPERATOR_CONFIG": str(config),
                    "COLLECTIVEX_OPERATOR_CONFIG_EPHEMERAL": "1",
                },
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(config.exists())
            self.assertEqual(decoy.read_text(), "keep")

    def test_operator_config_is_strict_per_runner_json(self) -> None:
        command = (
            'source "$1"; export COLLECTIVEX_EXECUTION_ID="operator-config-$$"; '
            "trap 'cx_cleanup_private_logs 0' EXIT; cx_load_operator_config; "
            'test "$CX_PARTITION" = test; '
            'test -z "${COLLECTIVEX_OPERATOR_CONFIG_CONTENT+x}"; '
            'test -z "${ENROOT_CACHE_PATH+x}"'
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            document = operator_config(root / "storage")
            config = root / "operator.json"
            config.write_text(json.dumps(document))
            config.chmod(0o600)
            for runner in capability.PLATFORMS:
                with self.subTest(runner=runner):
                    result = subprocess.run(
                        ["bash", "-c", command, "_", str(ROOT / "runtime" / "common.sh")],
                        text=True,
                        capture_output=True,
                        env={
                            **os.environ,
                            "CX_RUNNER": runner,
                            "ENROOT_CACHE_PATH": "/private/stale-enroot-cache",
                            "COLLECTIVEX_OPERATOR_CONFIG": str(config),
                        },
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)

            lock_dir = root / "amd-locks"
            document["runners"]["mi355x"]["lock_dir"] = str(lock_dir)
            config.write_text(json.dumps(document))
            config.chmod(0o600)
            canonical = subprocess.run(
                [
                    "bash",
                    "-c",
                    'source "$1"; export COLLECTIVEX_EXECUTION_ID="canonical-lock-$$"; '
                    "trap 'cx_cleanup_private_logs 0' EXIT; cx_load_operator_config; "
                    'cx_lock_canonical_gha_env mi355x; test "$CX_LOCK_DIR" = "$2"',
                    "_",
                    str(ROOT / "runtime" / "common.sh"),
                    str(lock_dir),
                ],
                text=True,
                capture_output=True,
                env={
                    **os.environ,
                    "CX_RUNNER": "mi355x",
                    "CX_SHARD_FILE": ".shards/test.json",
                    "CX_SHARD_SKU": "mi355x",
                    "CX_NODES": "1",
                    "CX_GPUS_PER_NODE": "8",
                    "COLLECTIVEX_CANONICAL_GHA": "1",
                    "COLLECTIVEX_OPERATOR_CONFIG": str(config),
                    "COLLECTIVEX_SOURCE_SHA": "a" * 40,
                    "GITHUB_ACTIONS": "true",
                    "GITHUB_RUN_ATTEMPT": "1",
                    "GITHUB_RUN_ID": "1",
                    "GITHUB_WORKSPACE": str(root.resolve()),
                },
            )
            self.assertEqual(canonical.returncode, 0, canonical.stderr)

            selected_only = {
                "schema_version": 1,
                "runners": {"h100-dgxc": document["runners"]["h100-dgxc"]},
            }
            result = subprocess.run(
                [
                    "bash", "-c", command + '; test "$CX_STAGE_DIR" = "$2"', "_",
                    str(ROOT / "runtime" / "common.sh"), str(root / "storage"),
                ],
                text=True,
                capture_output=True,
                env={
                    **os.environ,
                    "CX_RUNNER": "h100-dgxc",
                    "CX_STAGE_DIR": "/private/stale-stage",
                    "ENROOT_CACHE_PATH": "/private/stale-enroot-cache",
                    "COLLECTIVEX_OPERATOR_CONFIG_LOADED": "1",
                    "COLLECTIVEX_OPERATOR_CONFIG_CONTENT": json.dumps(selected_only),
                    "COLLECTIVEX_OPERATOR_CONFIG_REQUIRED": "1",
                },
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            rejected = json.loads(json.dumps(document))
            rejected["runners"]["h100-dgxc"]["shell"] = "private-command"
            boolean_version = {**document, "schema_version": True}
            missing_socket = json.loads(json.dumps(document))
            del missing_socket["runners"]["h100-dgxc"]["socket_ifname"]
            missing_rdma = json.loads(json.dumps(document))
            del missing_rdma["runners"]["mi355x"]["rdma_devices"]
            missing_amd_stage = json.loads(json.dumps(document))
            del missing_amd_stage["runners"]["mi325x"]["stage_dir"]
            missing_h100_stage = json.loads(json.dumps(document))
            del missing_h100_stage["runners"]["h100-dgxc"]["stage_dir"]
            missing_b300_stage = json.loads(json.dumps(document))
            del missing_b300_stage["runners"]["b300"]["stage_dir"]
            missing_gb300_stage = json.loads(json.dumps(document))
            del missing_gb300_stage["runners"]["gb300"]["stage_dir"]
            missing_nvidia_account = json.loads(json.dumps(document))
            del missing_nvidia_account["runners"]["h100-dgxc"]["account"]
            for invalid in (rejected, boolean_version, missing_nvidia_account):
                config.write_text(json.dumps(invalid))
                config.chmod(0o600)
                result = subprocess.run(
                    ["bash", "-c", command, "_", str(ROOT / "runtime" / "common.sh")],
                    text=True,
                    capture_output=True,
                    env={
                        **os.environ,
                        "CX_RUNNER": "h100-dgxc",
                        "COLLECTIVEX_OPERATOR_CONFIG": str(config),
                    },
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("private-command", result.stderr)

            for valid, runner in (
                (missing_socket, "h100-dgxc"),
                (missing_rdma, "h100-dgxc"),
                (missing_amd_stage, "h100-dgxc"),
                (missing_h100_stage, "h100-dgxc"),
                (missing_b300_stage, "b300"),
                (missing_gb300_stage, "gb300"),
                (missing_amd_stage, "mi325x"),
            ):
                config.write_text(json.dumps(valid))
                config.chmod(0o600)
                result = subprocess.run(
                    [
                        "bash", "-c", command + "; cx_apply_network_profile 1 nvlink",
                        "_", str(ROOT / "runtime" / "common.sh"),
                    ],
                    text=True,
                    capture_output=True,
                    env={
                        **os.environ,
                        "CX_RUNNER": runner,
                        "COLLECTIVEX_OPERATOR_CONFIG": str(config),
                    },
                )
                self.assertEqual(result.returncode, 0, result.stderr)

            config.write_text(json.dumps(missing_socket))
            config.chmod(0o600)
            scaleout = subprocess.run(
                [
                    "bash", "-c", command + "; cx_apply_network_profile 2 nvlink-rdma",
                    "_", str(ROOT / "runtime" / "common.sh"),
                ],
                text=True,
                capture_output=True,
                env={
                    **os.environ,
                    "CX_RUNNER": "h100-dgxc",
                    "COLLECTIVEX_OPERATOR_CONFIG": str(config),
                },
            )
            self.assertNotEqual(scaleout.returncode, 0)

            config.write_text(json.dumps(missing_amd_stage))
            config.chmod(0o600)
            selected_missing_stage = subprocess.run(
                ["bash", "-c", command, "_", str(ROOT / "runtime" / "common.sh")],
                text=True,
                capture_output=True,
                env={
                    **os.environ,
                    "CX_RUNNER": "mi325x",
                    "COLLECTIVEX_OPERATOR_CONFIG": str(config),
                },
            )
            self.assertEqual(
                selected_missing_stage.returncode, 0, selected_missing_stage.stderr
            )

            config.write_text(json.dumps(document))
            config.chmod(0o644)
            result = subprocess.run(
                ["bash", "-c", command, "_", str(ROOT / "runtime" / "common.sh")],
                text=True,
                capture_output=True,
                env={
                    **os.environ,
                    "CX_RUNNER": "h100-dgxc",
                    "COLLECTIVEX_OPERATOR_CONFIG": str(config),
                },
            )
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
