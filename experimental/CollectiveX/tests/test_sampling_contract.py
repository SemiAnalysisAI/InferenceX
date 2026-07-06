#!/usr/bin/env python3
"""CPU-only behavioral tests for the CollectiveX v1 execution contract."""
from __future__ import annotations

import argparse
import ast
import copy
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]

import artifact_safety  # noqa: E402
import capability  # noqa: E402
import contracts  # noqa: E402
import eplb  # noqa: E402
import ep_harness  # noqa: E402
import identity  # noqa: E402
import run_ep  # noqa: E402
import source_archive  # noqa: E402
import summarize  # noqa: E402
import sweep_matrix  # noqa: E402
import workload  # noqa: E402


class SamplingContractTest(unittest.TestCase):
    def test_identity_and_fixed_sampling_profile(self) -> None:
        identity.verify_test_vector()
        self.assertTrue(identity.is_typed_id(identity.IDENTITY_TEST_VECTOR["series_id"], "series"))
        self.assertEqual(ep_harness.SAMPLING_CONTRACT, "fixed-512-v1")
        self.assertEqual(
            (
                ep_harness.TIMED_ITERS_PER_TRIAL,
                ep_harness.TRIALS_PER_POINT,
                ep_harness.TIMED_SAMPLES_PER_POINT,
                ep_harness.WARMUP_ITERS_PER_TRIAL,
            ),
            (8, 64, 512, 32),
        )
        self.assertEqual(identity.V1_CASE_PROFILE["activation_profile"], "canonical-counter-source-v4")
        self.assertEqual(
            identity.V1_CASE_PROFILE["activation_generator"],
            "collectivex-activation-counter-v4",
        )
        self.assertEqual(identity.V1_CASE_PROFILE["sampling_contract"], "fixed-512-v1")
        self.assertEqual(identity.V1_CASE_PROFILE["percentile_method"], "nearest-rank")
        self.assertEqual(
            identity.V1_CASE_PROFILE["rank_reduction"],
            "cross-rank-max-per-iteration",
        )
        self.assertEqual(
            identity.V1_CASE_PROFILE["oracle_contract"],
            "expert-specific-transform-v1",
        )
        self.assertEqual(
            set(identity.V1_CASE_PROFILES), {"normal", "low-latency"}
        )
        self.assertEqual(
            identity.V1_LOW_LATENCY_CASE_PROFILE["payload_unit"], "token-expert"
        )
        self.assertNotEqual(
            identity.digest("case", identity.V1_NORMAL_CASE_PROFILE),
            identity.digest("case", identity.V1_LOW_LATENCY_CASE_PROFILE),
        )
        parser = argparse.ArgumentParser()
        ep_harness.add_common_args(parser)
        args = parser.parse_args(
            [
                "--runner", "test", "--topology-class", "test",
                "--scope", "scale-up", "--scale-up-transport", "nvlink",
                "--out", "result.json",
            ]
        )
        self.assertEqual((args.iters, args.trials, args.warmup), (8, 64, 32))
        self.assertEqual(args.qualification_index, 1)
        with mock.patch.dict(os.environ, {"CX_QUALIFICATION_INDEX": "3"}):
            parser = argparse.ArgumentParser()
            ep_harness.add_common_args(parser)
            env_args = parser.parse_args(
                [
                    "--runner", "test", "--topology-class", "test",
                    "--scope", "scale-up", "--scale-up-transport", "nvlink",
                    "--out", "result.json",
                ]
            )
        self.assertEqual(env_args.qualification_index, 3)
        for profile in ((8, 64, 32), (128, 4, 32), (8, 1, 4), (0, 64, 32)):
            with self.subTest(profile=profile):
                self.assertEqual(
                    ep_harness.sampling_contract_error(*profile) is None,
                    profile == (8, 64, 32),
                )

    def test_nearest_rank_percentiles_use_all_512_samples(self) -> None:
        samples = list(range(1, 513))
        self.assertEqual(ep_harness.percentile(samples, 50), 256)
        self.assertEqual(ep_harness.percentile(samples, 99), 507)

    def test_qualification_order_is_deterministic_and_position_balanced(self) -> None:
        values = [1, 2, 4, 8, 16, 32, 64, 128]
        for qualification_index in range(1, ep_harness.QUALIFICATION_RUNS + 1):
            orders = [
                ep_harness.qualification_order(values, qualification_index, trial)
                for trial in range(64)
            ]
            self.assertEqual(
                orders,
                [
                    ep_harness.qualification_order(values, qualification_index, trial)
                    for trial in range(64)
                ],
            )
            self.assertTrue(all(sorted(order) == values for order in orders))
            for position in range(len(values)):
                self.assertEqual(
                    {value: sum(order[position] == value for order in orders) for value in values},
                    {value: 8 for value in values},
                )
        with self.assertRaises(ValueError):
            ep_harness.qualification_order(values, 0, 0)
        with self.assertRaises(ValueError):
            ep_harness.qualification_order([1, 1], 1, 0)

    def test_sample_evidence_preserves_exact_trial_blocks(self) -> None:
        trials = [
            [float(trial * 8 + sample) for sample in range(8)]
            for trial in range(64)
        ]
        evidence = ep_harness.sampled_component_evidence(trials)
        self.assertEqual(evidence["availability"], "measured")
        self.assertEqual(evidence["sample_count"], 512)
        self.assertEqual(evidence["trials"], trials)
        self.assertIsNot(evidence["trials"], trials)
        self.assertEqual(
            ep_harness.sampled_component_evidence([]),
            {"availability": "unavailable", "sample_count": 0, "trials": None},
        )
        for malformed in (trials[:-1], [*trials[:-1], trials[-1][:-1]]):
            with self.assertRaises(ValueError):
                ep_harness.sampled_component_evidence(malformed)
        invalid = copy.deepcopy(trials)
        invalid[0][0] = float("nan")
        with self.assertRaises(ValueError):
            ep_harness.sampled_component_evidence(invalid)

    def test_terminal_summary_uses_bound_sku_and_route(self) -> None:
        terminal = {
            "format": contracts.TERMINAL_FORMAT,
            "case": {
                "backend": "deepep", "phase": "prefill", "ep": 8,
                "suite": "ep-routing-v1", "routing": "zipf", "eplb": True,
                "required_publication": "comparable-experimental",
            },
            "identity": {"case_factors": {"sku": "h100-dgxc"}},
        }
        self.assertEqual(
            summarize._identity(terminal),
            (
                "h100-dgxc", "ep-routing-v1", "zipf", "prefill", True,
                "comparable-experimental", 8,
            ),
        )

    def test_matrix_cases_and_shards_are_identity_bound(self) -> None:
        matrix = sweep_matrix.validate_matrix_document(
            sweep_matrix.resolve_matrix(backends="all")
        )
        requested = {item["case"]["case_id"]: item for item in matrix["requested_cases"]}
        assigned = [case_id for shard in matrix["include"] for case_id in shard["case_ids"]]
        runnable = {
            case_id for case_id, item in requested.items()
            if item["disposition"] == "runnable"
        }
        runnable_cases = [
            item for item in matrix["requested_cases"]
            if item["disposition"] == "runnable"
        ]
        unsupported_cases = [
            item for item in matrix["requested_cases"]
            if item["disposition"] == "unsupported"
        ]
        self.assertEqual(
            (
                len(matrix["include"]),
                len(matrix["requested_cases"]),
                len(runnable_cases),
                len(unsupported_cases),
                sum(
                    len(item["case"]["ladder"].split())
                    for item in matrix["requested_cases"]
                ),
                sum(len(item["case"]["ladder"].split()) for item in runnable_cases),
                sum(len(item["case"]["ladder"].split()) for item in unsupported_cases),
            ),
            (49, 748, 387, 361, 1740, 877, 863),
        )
        b300_ep16 = [
            item for item in unsupported_cases
            if item["sku"] == "b300" and item["case"]["ep"] == 16
            and item["case"]["backend"] in {
                "deepep", "deepep-v2", "deepep-hybrid", "nccl-ep",
            }
        ]
        self.assertTrue(b300_ep16)
        self.assertTrue(all(
            item["detail"] == "v1 publication fabric unavailable for B300 EP16"
            for item in b300_ep16
        ))
        expected_topologies = {}
        for sku, product in (
            ("h100-dgxc", "h100"), ("h200-dgxc", "h200"),
            ("b200-dgxc", "b200"), ("b300", "b300"),
        ):
            expected_topologies[sku, 8] = (
                1, 8, 8, "scale-up", "nvlink", None, "nvlink",
                f"{product}-nvlink-island",
            )
            expected_topologies[sku, 16] = (
                2, 8, 8, "scale-out", "nvlink", "rdma", "nvlink-rdma",
                f"{product}-nvlink-rdma",
            )
        for sku in ("gb200", "gb300"):
            topology_class = f"{sku}-nvl72-mnnvl"
            expected_topologies[sku, 8] = (
                2, 4, 72, "scale-up", "mnnvl", None, "mnnvl", topology_class,
            )
            expected_topologies[sku, 16] = (
                4, 4, 72, "scale-up", "mnnvl", None, "mnnvl", topology_class,
            )
        for sku in ("mi300x", "mi355x"):
            expected_topologies[sku, 8] = (
                1, 8, 8, "scale-up", "xgmi", None, "xgmi", f"{sku}-xgmi",
            )
            expected_topologies[sku, 16] = (
                2, 8, 8, "scale-out", "xgmi", "rdma", "xgmi-rdma",
                f"{sku}-xgmi-rdma",
            )
        topology_fields = sweep_matrix.TOPOLOGY_FIELDS
        observed_topologies: dict[tuple[str, int], set[tuple[object, ...]]] = {}
        for item in matrix["requested_cases"]:
            case = item["case"]
            observed_topologies.setdefault((item["sku"], case["ep"]), set()).add(
                tuple(case[field] for field in topology_fields)
            )
        self.assertEqual(
            {key: next(iter(values)) for key, values in observed_topologies.items()},
            expected_topologies,
        )
        self.assertTrue(all(len(values) == 1 for values in observed_topologies.values()))
        self.assertEqual(
            {
                (sku, ep): tuple(topology[field] for field in topology_fields)
                for sku, platform in capability.PLATFORMS.items()
                if sku != "mi325x"
                for ep, topology in platform["topologies"].items()
            },
            expected_topologies,
        )
        self.assertIsNotNone(capability.topology_for("mi325x", 8))
        self.assertEqual(
            Counter(shard["n"] for shard in matrix["include"]),
            Counter({6: 29, 8: 6, 10: 1, 11: 1, 12: 12}),
        )
        ll_cases = [
            item for item in matrix["requested_cases"]
            if item["case"]["mode"] == "low-latency"
        ]
        self.assertEqual(len(ll_cases), 80)
        self.assertTrue(all(
            item["case"]["backend"] in {"deepep", "uccl"}
            and item["case"]["phase"] == "decode"
            and item["case"]["routing"] == "uniform"
            and not item["case"]["eplb"]
            and (
                (
                    item["case"]["suite"] == "ep-low-latency-v1"
                    and item["case"]["ladder"] == "1 2 4 8 16 32 64 128"
                )
                or (
                    item["case"]["suite"] == "ep-precision-low-latency-v1"
                    and item["case"]["ladder"] == "128"
                )
            )
            for item in ll_cases
        ))
        for shard in matrix["include"]:
            ep = next(
                requested[case_id]["case"]["ep"] for case_id in shard["case_ids"]
            )
            self.assertEqual(
                tuple(shard[field] for field in topology_fields),
                expected_topologies[shard["sku"], ep],
            )
        routing_points = {
            phase: {
                int(point)
                for item in matrix["requested_cases"]
                if item["case"]["suite"] == "ep-routing-v1"
                and item["case"]["phase"] == phase
                for point in item["case"]["ladder"].split()
            }
            for phase in ("decode", "prefill")
        }
        self.assertEqual(routing_points, {"decode": {128}, "prefill": {512}})
        skus = sorted({shard["sku"] for shard in matrix["include"]})
        self.assertEqual(
            [shard["sku"] for shard in matrix["include"][:len(skus)]],
            skus,
        )
        self.assertEqual(set(assigned), runnable)
        self.assertEqual(len(assigned), len(set(assigned)))
        self.assertEqual({item["case"]["ep"] for item in matrix["requested_cases"]}, {8, 16})
        self.assertFalse(capability.resolve("gb200", "deepep", ep=8, nodes=1)[0])
        excluded = {
            "uccl": {"b200-dgxc", "b300"},
        }
        for backend, skus in excluded.items():
            for sku in skus:
                with self.subTest(backend=backend, sku=sku):
                    self.assertFalse(capability.resolve(sku, backend)[0])
        for case_id, item in requested.items():
            case = {key: value for key, value in item["case"].items() if key != "case_id"}
            self.assertEqual(
                case_id,
                identity.case_id(
                    sku=item["sku"], profile=identity.profile_for_case(case), case=case
                ),
            )
            self.assertEqual(case["timing"], "8:64:32")
            self.assertEqual(case["samples_per_point"], 512)

        bad_matrix = copy.deepcopy(matrix)
        bad_matrix["schema_version"] = True
        with self.assertRaises(sweep_matrix.MatrixError):
            sweep_matrix.validate_matrix_document(bad_matrix)

        bad_catalog = copy.deepcopy(matrix)
        wrapper = next(
            item for item in bad_catalog["requested_cases"]
            if item["disposition"] == "runnable"
        )
        old_id = wrapper["case"]["case_id"]
        wrapper["case"]["hidden"] = 1
        factors = {key: value for key, value in wrapper["case"].items() if key != "case_id"}
        new_id = identity.case_id(
            sku=wrapper["sku"], profile=identity.V1_CASE_PROFILE, case=factors
        )
        wrapper["case"]["case_id"] = new_id
        for shard in bad_catalog["include"]:
            shard["case_ids"] = [new_id if value == old_id else value for value in shard["case_ids"]]
        with self.assertRaisesRegex(sweep_matrix.MatrixError, "frozen v1"):
            sweep_matrix.validate_matrix_document(bad_catalog)

        bad_topology = copy.deepcopy(matrix)
        wrapper = next(
            item for item in bad_topology["requested_cases"]
            if item["disposition"] == "runnable"
        )
        old_id = wrapper["case"]["case_id"]
        wrapper["case"]["transport"] = "incorrect-transport"
        factors = {key: value for key, value in wrapper["case"].items() if key != "case_id"}
        new_id = identity.case_id(
            sku=wrapper["sku"], profile=identity.V1_CASE_PROFILE, case=factors
        )
        wrapper["case"]["case_id"] = new_id
        for shard in bad_topology["include"]:
            shard["case_ids"] = [new_id if value == old_id else value for value in shard["case_ids"]]
        with self.assertRaisesRegex(sweep_matrix.MatrixError, "platform registry"):
            sweep_matrix.validate_matrix_document(bad_topology)

        shard_meta = matrix["include"][0]
        requested_cases = {item["case"]["case_id"]: item["case"] for item in matrix["requested_cases"]}
        shard = {
            "schema_version": True,
            "id": shard_meta["id"],
            "sku": shard_meta["sku"],
            "backend": shard_meta["backend"],
            "nodes": shard_meta["nodes"],
            "n": shard_meta["n"],
            "cases": [requested_cases[value] for value in shard_meta["case_ids"]],
        }
        with self.assertRaises(sweep_matrix.MatrixError):
            sweep_matrix.validate_shard_control(
                shard, sku=shard_meta["sku"], backend=shard_meta["backend"],
                nodes=shard_meta["nodes"],
            )

    def test_matrix_yaml_and_config_validation_are_strict(self) -> None:
        suites = sweep_matrix._load("suites.yaml")
        workloads = sweep_matrix._load("workloads.yaml")
        self.assertEqual(
            {tuple(suite["ep_degrees"]) for suite in suites["suites"].values()},
            {(8, 16)},
        )
        invalid = (
            ("unknown top", lambda s, _w: s.update({"typo": True})),
            (
                "unknown suite field",
                lambda s, _w: s["suites"]["ep-core-v1"].update({"modes": ["normal"]}),
            ),
            (
                "unknown workload field",
                lambda _s, w: w["model_derived"]["deepseek-v3-v1"].update({"unused": 1}),
            ),
            (
                "string phases",
                lambda s, _w: s["suites"]["ep-core-v1"].update({"phases": "decode"}),
            ),
            (
                "unknown routing",
                lambda s, _w: s["suites"]["ep-core-v1"].update({"routings": ["random"]}),
            ),
            (
                "integer EPLB",
                lambda s, _w: s["suites"]["ep-routing-v1"].update({"eplb": [0, 1]}),
            ),
            (
                "duplicate platform",
                lambda s, _w: s["suites"]["ep-core-v1"]["platforms"].append("h100-dgxc"),
            ),
            (
                "missing EP degrees",
                lambda s, _w: s["suites"]["ep-core-v1"].pop("ep_degrees"),
            ),
            (
                "non-v1 EP degrees",
                lambda s, _w: s["suites"]["ep-core-v1"].update({"ep_degrees": [8]}),
            ),
            ("missing top field", lambda s, _w: s.pop("schema_version")),
            (
                "string dimension",
                lambda _s, w: w["model_derived"]["deepseek-v3-v1"].update({"hidden": "7168"}),
            ),
            (
                "unreachable phase ladder",
                lambda s, _w: s["suites"]["ep-routing-v1"].update({"phases": ["prefill"]}),
            ),
        )
        for label, mutate in invalid:
            with self.subTest(label=label), self.assertRaises(SystemExit):
                bad_suites, bad_workloads = copy.deepcopy(suites), copy.deepcopy(workloads)
                mutate(bad_suites, bad_workloads)
                sweep_matrix.validate_config_documents(bad_suites, bad_workloads)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "configs").mkdir()
            (root / "configs" / "duplicate.yaml").write_text(
                "schema_version: 1\nsuites:\n  same: 1\n  same: 2\n"
            )
            with mock.patch.object(sweep_matrix, "HERE", root), self.assertRaisesRegex(
                SystemExit, "duplicate YAML key"
            ):
                sweep_matrix._load("duplicate.yaml")

    def test_semantically_duplicate_suite_points_are_rejected(self) -> None:
        matrix = sweep_matrix.resolve_matrix()
        with mock.patch.object(
            sweep_matrix, "_semantic_points", return_value=["duplicate"]
        ), self.assertRaisesRegex(
            sweep_matrix.MatrixError, "duplicates a semantic token point"
        ):
            sweep_matrix.validate_matrix_document(matrix)

    def test_only_three_shared_launchers_are_registered(self) -> None:
        expected = {
            "launch_single-slurm.sh",
            "launch_gb-nv.sh",
            "launch_mi-amds.sh",
        }
        self.assertEqual({path.name for path in (ROOT / "launchers").glob("launch_*.sh")}, expected)
        self.assertEqual(
            {platform["launcher"] for platform in capability.PLATFORMS.values()},
            {"single-slurm", "gb-nv", "mi-amds"},
        )
        for platform in capability.PLATFORMS.values():
            launcher = ROOT / "launchers" / f"launch_{platform['launcher']}.sh"
            self.assertTrue(launcher.is_file())
            source = launcher.read_text()
            self.assertNotIn("RUNNER_NAME", source)
            self.assertIn("cx_preflight_allocation", source)
            lock_environment = 'cx_lock_canonical_gha_env "$RUNNER"'
            self.assertIn(lock_environment, source)
            self.assertLess(
                source.index("cx_load_operator_config"),
                source.index(lock_environment),
            )
            validate = 'cx_validate_shard_control "$CX_DIR"'
            stage = 'MOUNT_SRC="$(cx_stage_path '
            self.assertIn(validate, source)
            self.assertLess(source.index(validate), source.index(stage))
            self.assertLess(source.index(stage), source.index('cx_stage_repo "$REPO_ROOT"'))
            self.assertLess(source.index(validate), source.index("cx_require_vars"))
            if platform["launcher"] in {"single-slurm", "mi-amds"}:
                network = "cx_validate_network_profile_on_job"
                self.assertIn(network, source)
                self.assertLess(source.index("cx_salloc_jobid"), source.index(network))
                self.assertLess(source.index(network), source.index("cx_preflight_allocation"))
                if platform["launcher"] == "single-slurm":
                    self.assertLess(
                        source.index(network),
                        source.index("CX_ENROOT_LOCAL_IMPORT=1 cx_ensure_squash"),
                    )

        common = (ROOT / "runtime" / "common.sh").read_text()
        workflow = (ROOT.parent.parent / ".github" / "workflows" / "collectivex-sweep.yml").read_text()
        self.assertNotIn("RUNNER_NAME", common)
        self.assertNotIn("RUNNER_NAME:", workflow)
        self.assertNotIn("flashinfer", capability.BACKENDS)
        self.assertFalse((HERE / "ep_flashinfer.py").exists())

    def test_canonical_operator_config_requires_a_private_audit_salt(self) -> None:
        salt = "a" * 64
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            config = root / "operator.json"
            document = {
                "schema_version": 1,
                "audit_salt": salt,
                "runners": {
                    "h100-dgxc": {
                        "partition": "test", "account": "test",
                        "squash_dir": str(root), "stage_dir": str(root),
                        "ib_gid_index": "3", "rdma_service_level": "2",
                        "rdma_traffic_class": "104",
                    },
                },
            }
            command = (
                'source "$1"; export COLLECTIVEX_EXECUTION_ID="audit-config-$$"; '
                "trap 'cx_cleanup_private_logs 0' EXIT; cx_load_operator_config; "
                'test "$CX_AUDIT_SALT" = "$EXPECTED_AUDIT_SALT"; '
                'test "$CX_IB_GID_INDEX:$CX_RDMA_SERVICE_LEVEL:$CX_RDMA_TRAFFIC_CLASS" = 3:2:104'
            )

            def invoke(
                value: dict, *, canonical: bool, expect_salt: bool = True,
                audit_override: str | None = None,
            ) -> subprocess.CompletedProcess[str]:
                config.write_text(json.dumps(value))
                config.chmod(0o600)
                environment = {
                    **os.environ,
                    "CX_RUNNER": "h100-dgxc",
                    "COLLECTIVEX_OPERATOR_CONFIG": str(config),
                    "EXPECTED_AUDIT_SALT": salt,
                }
                if canonical:
                    environment["COLLECTIVEX_CANONICAL_GHA"] = "1"
                if audit_override is not None:
                    environment["COLLECTIVEX_OPERATOR_AUDIT_SALT"] = audit_override
                invocation = command if expect_salt else (
                    'source "$1"; export COLLECTIVEX_EXECUTION_ID="audit-config-$$"; '
                    "trap 'cx_cleanup_private_logs 0' EXIT; cx_load_operator_config; "
                    'test -z "${CX_AUDIT_SALT+x}"'
                )
                return subprocess.run(
                    ["bash", "-c", invocation, "_", str(ROOT / "runtime" / "common.sh")],
                    text=True,
                    capture_output=True,
                    env=environment,
                )

            accepted = invoke(document, canonical=True)
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            self.assertNotIn(salt, accepted.stdout + accepted.stderr)

            missing = copy.deepcopy(document)
            del missing["audit_salt"]
            rejected = invoke(missing, canonical=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertRegex(rejected.stderr, r"validation-line-[0-9]+")
            self.assertNotIn(salt, rejected.stdout + rejected.stderr)

            separate = invoke(missing, canonical=True, audit_override=salt)
            self.assertEqual(separate.returncode, 0, separate.stderr)
            self.assertNotIn(salt, separate.stdout + separate.stderr)

            malformed_override = invoke(missing, canonical=True, audit_override="A" * 64)
            self.assertNotEqual(malformed_override.returncode, 0)
            self.assertNotIn("A" * 64, malformed_override.stdout + malformed_override.stderr)

            manual = invoke(missing, canonical=False, expect_salt=False)
            self.assertEqual(manual.returncode, 0, manual.stderr)
            self.assertNotIn(salt, manual.stdout + manual.stderr)

            malformed = copy.deepcopy(document)
            malformed["audit_salt"] = "A" * 64
            rejected = invoke(malformed, canonical=False)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertNotIn("A" * 64, rejected.stdout + rejected.stderr)

            missing_field = copy.deepcopy(document)
            del missing_field["runners"]["h100-dgxc"]["account"]
            rejected = invoke(missing_field, canonical=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("validation-missing-required-account", rejected.stderr)

            for field, value in (
                ("ib_gid_index", "03"),
                ("rdma_service_level", "16"),
                ("rdma_traffic_class", "256"),
            ):
                invalid_selector = copy.deepcopy(document)
                invalid_selector["runners"]["h100-dgxc"][field] = value
                rejected = invoke(invalid_selector, canonical=True)
                self.assertNotEqual(rejected.returncode, 0)

    def test_scaleout_network_profile_is_explicit_and_allowlisted(self) -> None:
        command = r'''
          set -euo pipefail
          source "$1"
          test "$(cx_nccl_hca_device_name '=mlx5_0:1')" = mlx5_0
          test "$(cx_nccl_hca_device_name 'mlx5_1')" = mlx5_1
          ! (unset CX_SOCKET_IFNAME CX_RDMA_DEVICES; cx_apply_network_profile 2 nvlink-rdma)
          ! (export CX_SOCKET_IFNAME=eth0; unset CX_RDMA_DEVICES; cx_apply_network_profile 2 nvlink-rdma)
          export CX_SOCKET_IFNAME=ib0 CX_RDMA_DEVICES=mlx5_0:1,mlx5_1:1
          export NCCL_NET=Socket NCCL_IB_HCA=stale NVSHMEM_HCA_LIST=stale
          export NVSHMEM_HCA_PE_MAPPING=stale NVSHMEM_REMOTE_TRANSPORT=stale
          export CX_SHARD_SKU=b300
          cx_apply_network_profile 1 nvlink
          test "$NVSHMEM_DISABLE_IB" = 1
          test -z "${NVSHMEM_HCA_PE_MAPPING+x}${NVSHMEM_REMOTE_TRANSPORT+x}"
          export CX_BENCH=deepep CX_MODE=low-latency CX_IB_GID_INDEX=3
          cx_apply_network_profile 1 nvlink
          test -z "${NVSHMEM_DISABLE_IB+x}${NCCL_NET+x}${NCCL_IB_HCA+x}"
          test -z "${NVSHMEM_IB_GID_INDEX+x}"
          cx_export_gid_index_for_link_layer roce 0
          test "$NVSHMEM_HCA_LIST:$NVSHMEM_IB_GID_INDEX" = mlx5_0:1,mlx5_1:1:3
          test "$NVSHMEM_ENABLE_NIC_PE_MAPPING" = 1
          test "$NVSHMEM_IB_ENABLE_IBGDA:$NVSHMEM_IBGDA_NIC_HANDLER" = 1:gpu
          unset CX_BENCH CX_MODE CX_IB_GID_INDEX
          unset CX_SHARD_SKU
          cx_apply_network_profile 1 nvlink
          test -z "${NCCL_NET+x}${NCCL_IB_HCA+x}${NVSHMEM_DISABLE_IB+x}${NVSHMEM_HCA_LIST+x}"
          cx_apply_network_profile 4 mnnvl
          test -z "${NCCL_NET+x}${NCCL_IB_HCA+x}${NVSHMEM_HCA_LIST+x}"
          export CX_IB_GID_INDEX=3 CX_RDMA_SERVICE_LEVEL=2 CX_RDMA_TRAFFIC_CLASS=104
          cx_apply_network_profile 2 nvlink-rdma
          test "$NCCL_SOCKET_IFNAME:$GLOO_SOCKET_IFNAME:$UCCL_SOCKET_IFNAME" = ib0:ib0:ib0
          test "$NCCL_NET:$NCCL_IB_HCA" = 'IB:=mlx5_0:1,mlx5_1:1'
          NCCL_IB_HCA=mlx5_0:1,mlx5_1:1
          export NCCL_IB_HCA CX_NODES=2 CX_TRANSPORT=nvlink-rdma
          cx_restore_exact_hca_selector
          test "$NCCL_IB_HCA" = '=mlx5_0:1,mlx5_1:1'
          wrapper="$(cx_slurm_rank_wrapper)"
          bash -n <<< "$wrapper"
          grep -Fq '. /ix/experimental/CollectiveX/runtime/common.sh' <<< "$wrapper"
          grep -Fq 'cx_apply_network_profile "$CX_NODES" "$CX_TRANSPORT" || exit 68' <<< "$wrapper"
          grep -Fq 'cx_write_runtime_stage execution || exit 68' <<< "$wrapper"
          test "$NVSHMEM_HCA_LIST" = mlx5_0:1,mlx5_1:1
          test "$NVSHMEM_ENABLE_NIC_PE_MAPPING" = 1
          test "$MORI_RDMA_DEVICES:$EP_NIC_NAME" = mlx5_0,mlx5_1:mlx5_0
          test "$MORI_RDMA_TC:$MORI_IO_TC:$MORI_RDMA_SL:$MORI_IO_SL" = 104:104:2:2
          test -z "${NCCL_IB_GID_INDEX+x}${NVSHMEM_IB_GID_INDEX+x}${UCCL_IB_GID_INDEX+x}"
          cx_export_gid_index_for_link_layer roce 1
          test "$NCCL_IB_GID_INDEX:$NCCL_IB_SL" = 3:2
          test "$NVSHMEM_IB_ENABLE_IBGDA:$NVSHMEM_IBGDA_NIC_HANDLER" = 1:gpu
          export CX_SHARD_SKU=b200-dgxc CX_BENCH=deepep
          cx_apply_network_profile 2 nvlink-rdma
          test "$NVSHMEM_IB_ENABLE_IBGDA:$NVSHMEM_IBGDA_NIC_HANDLER" = 1:cpu
          unset CX_SHARD_SKU CX_BENCH
          cx_apply_network_profile 2 nvlink-rdma
          cx_export_gid_index_for_link_layer infiniband 1
          test -z "${NVSHMEM_IB_GID_INDEX+x}${NCCL_IB_GID_INDEX+x}${UCCL_IB_GID_INDEX+x}"
          test "$NVSHMEM_IB_SL:$NCCL_IB_SL:$UCCL_IB_SL" = 2:2:2
          export CX_RDMA_LINK_LAYER=roce
          cx_apply_network_profile 2 nvlink-rdma
          test "$NVSHMEM_IB_GID_INDEX:$NCCL_IB_GID_INDEX:$UCCL_IB_GID_INDEX" = 3:3:3
        '''
        subprocess.run(
            ["bash", "-c", command, "_", str(ROOT / "runtime" / "common.sh")],
            check=True,
            env={**os.environ, "COLLECTIVEX_OPERATOR_CONFIG": "/dev/null"},
        )

        v2_adapter = (ROOT / "tests" / "ep_deepep_v2.py").read_text()
        container_runtime = (ROOT / "runtime" / "run_in_container.sh").read_text()
        for version in ("2.0.0+fa8a9b1", "2.0.0+local"):
            self.assertIn(version, v2_adapter)
            self.assertIn(version, container_runtime)

    def test_network_mode_is_loaded_from_validated_control(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mixed = root / "mixed.json"
            mixed.write_text(json.dumps({"cases": [
                {"mode": "normal"}, {"mode": "low-latency"},
            ]}))
            normal = root / "normal.json"
            normal.write_text(json.dumps({"cases": [{"mode": "normal"}]}))
            probe = root / "probe.json"
            probe.write_text(json.dumps({
                "id": "probe-id",
                "target": {
                    "backend": "deepep", "sku": "b300", "ep": 8,
                    "mode": "low-latency", "precision_profile": "profile",
                },
            }))
            command = r'''
              set -euo pipefail
              source "$1"
              export CX_SHARD_FILE="$2" CX_PRECISION_PROBE="$3"
              cx_load_network_control_mode "$4"
              printf '%s' "$CX_MODE"
            '''
            for path, precision, expected in (
                (mixed, "0", "low-latency"),
                (normal, "0", "normal"),
                (probe, "1", "low-latency"),
            ):
                result = subprocess.run(
                    ["bash", "-c", command, "_", str(ROOT / "runtime" / "common.sh"),
                     str(path), precision, str(ROOT)],
                    text=True, capture_output=True, check=True,
                    env={**os.environ, "COLLECTIVEX_OPERATOR_CONFIG": "/dev/null"},
                )
                self.assertEqual(result.stdout, expected)

    def test_network_profile_validation_is_private_and_all_node(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            binary = root / "srun"
            arguments = root / "arguments"
            script = root / "script"
            binary.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\\n' \"$@\" > \"$CAPTURE_ARGS\"\n"
                "cat > \"$CAPTURE_SCRIPT\"\n"
                "tasks=1\n"
                "for arg in \"$@\"; do\n"
                "  case \"$arg\" in --ntasks=*) tasks=\"${arg#--ntasks=}\" ;; esac\n"
                "done\n"
                "if [ \"${SRUN_RC:-0}\" = 0 ]; then\n"
                "  for ((task=0; task<tasks; task++)); do\n"
                "    printf '[collectivex-private] socket-interface-selected=privateif0\\n'\n"
                "    printf '[collectivex-private] rdma-link-layer=roce\\n'\n"
                "  done\n"
                "fi\n"
                "exit \"${SRUN_RC:-0}\"\n"
            )
            binary.chmod(0o700)
            command = (
                'source "$1"; export COLLECTIVEX_EXECUTION_ID="network-test-$$"; '
                "trap 'cx_cleanup_private_logs 0' EXIT; "
                'cx_validate_network_profile_on_job 42 2 nvlink-rdma'
            )
            environment = {
                **os.environ,
                "PATH": f"{root}:{os.environ['PATH']}",
                "COLLECTIVEX_OPERATOR_CONFIG": "/dev/null",
                "CAPTURE_ARGS": str(arguments),
                "CAPTURE_SCRIPT": str(script),
                "CX_SOCKET_IFNAME": "privateif0",
                "CX_RDMA_DEVICES": "privatehca0:1",
                "CX_IB_GID_INDEX": "3",
            }
            result = subprocess.run(
                ["bash", "-c", command, "_", str(ROOT / "runtime" / "common.sh")],
                text=True,
                capture_output=True,
                env=environment,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            invoked = arguments.read_text()
            self.assertIn("--nodes=2", invoked)
            self.assertIn("--ntasks=2", invoked)
            self.assertIn("--input=all", invoked)
            self.assertIn("CX_SOCKET_IFNAME,CX_RDMA_DEVICES,CX_IB_GID_INDEX", invoked)
            validation_script = script.read_text()
            self.assertIn('/sys/class/infiniband/$device/ports', validation_script)
            self.assertIn('rdma-link-layer=%s', validation_script)
            self.assertNotIn("privateif0", result.stdout + result.stderr)
            self.assertNotIn("privatehca0", result.stdout + result.stderr)

            failed = subprocess.run(
                ["bash", "-c", command, "_", str(ROOT / "runtime" / "common.sh")],
                text=True,
                capture_output=True,
                env={**environment, "SRUN_RC": "9"},
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertNotIn("privateif0", failed.stdout + failed.stderr)
            self.assertNotIn("privatehca0", failed.stdout + failed.stderr)

            quiet_failed = subprocess.run(
                [
                    "bash", "-c",
                    command + " 0",
                    "_", str(ROOT / "runtime" / "common.sh"),
                ],
                text=True,
                capture_output=True,
                env={**environment, "SRUN_RC": "9"},
            )
            self.assertNotEqual(quiet_failed.returncode, 0)
            self.assertNotIn("failure-class=", quiet_failed.stderr)

            arguments.unlink()
            subprocess.run(
                [
                    "bash", "-c",
                    'source "$1"; cx_validate_network_profile_on_job 42 1 nvlink',
                    "_", str(ROOT / "runtime" / "common.sh"),
                ],
                check=True,
                env=environment,
            )
            self.assertFalse(arguments.exists())

            single_node_command = (
                'source "$1"; export COLLECTIVEX_EXECUTION_ID="network-test-$$"; '
                "trap 'cx_cleanup_private_logs 0' EXIT; "
                "export CX_SHARD_SKU=b300 CX_BENCH=deepep CX_MODE=low-latency; "
                'cx_validate_network_profile_on_job 42 1 nvlink'
            )
            single_node = subprocess.run(
                ["bash", "-c", single_node_command, "_", str(ROOT / "runtime" / "common.sh")],
                text=True,
                capture_output=True,
                env={**environment, "CX_SOCKET_IFNAME": ""},
            )
            self.assertEqual(single_node.returncode, 0, single_node.stderr)
            self.assertIn("--nodes=1", arguments.read_text())
            self.assertIn("--ntasks=1", arguments.read_text())

    def test_rejected_allocation_nodes_are_expanded_safely(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            for name, payload in (
                ("squeue", "#!/bin/sh\nprintf '%s\\n' 'rack-[1,3]'\n"),
                ("scontrol", "#!/bin/sh\nprintf '%s\\n' rack-1 rack-3\n"),
            ):
                binary = root / name
                binary.write_text(payload)
                binary.chmod(0o700)
            result = subprocess.run(
                [
                    "bash", "-c",
                    'source "$1"; cx_allocation_nodes_csv 42',
                    "_", str(ROOT / "runtime" / "common.sh"),
                ],
                text=True,
                capture_output=True,
                env={
                    **os.environ,
                    "PATH": f"{root}:{os.environ['PATH']}",
                    "COLLECTIVEX_OPERATOR_CONFIG": "/dev/null",
                },
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "rack-1,rack-3")

            (root / "squeue").write_text("#!/bin/sh\nprintf 'rack;unsafe\\n'\n")
            (root / "squeue").chmod(0o700)
            rejected = subprocess.run(
                [
                    "bash", "-c",
                    'source "$1"; cx_allocation_nodes_csv 42',
                    "_", str(ROOT / "runtime" / "common.sh"),
                ],
                env={
                    **os.environ,
                    "PATH": f"{root}:{os.environ['PATH']}",
                    "COLLECTIVEX_OPERATOR_CONFIG": "/dev/null",
                },
            )
            self.assertNotEqual(rejected.returncode, 0)

    def test_allocation_preflight_proves_shared_write_visibility(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            mount = root / "mount"
            runtime = mount / "experimental" / "CollectiveX" / "runtime"
            runtime.mkdir(parents=True)
            (runtime / "run_in_container.sh").write_text("#!/bin/sh\n")
            squash = root / "image.sqsh"
            squash.write_bytes(b"squash")
            binary = root / "bin"
            binary.mkdir()
            (binary / "unsquashfs").write_text("#!/bin/sh\nexit 0\n")
            (binary / "unsquashfs").chmod(0o700)
            (binary / "srun").write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "case \" $* \" in *' --input=all '*) ;; *) exit 97 ;; esac\n"
                "worker=\"$FAKE_ROOT/worker.sh\"\n"
                "cat > \"$worker\"\n"
                "args=(\"$@\")\n"
                "start=0\n"
                "for ((i=0; i<${#args[@]}; i++)); do\n"
                "  [ \"${args[$i]}\" != -- ] || start=$((i + 1))\n"
                "done\n"
                "[ \"$start\" -gt 0 ]\n"
                "worker_args=(\"${args[@]:$start}\")\n"
                "probe=\"${worker_args[4]}\"\n"
                "case \"${FAKE_MODE:-success}\" in\n"
                "  missing-source) rm -f -- \"$probe/source\" ;;\n"
                "  readonly) chmod 500 \"$probe\" ;;\n"
                "esac\n"
                "for ((node=0; node<FAKE_TASKS; node++)); do\n"
                "  SLURM_NODEID=\"$node\" bash \"$worker\" \"${worker_args[@]}\"\n"
                "done\n"
            )
            (binary / "srun").chmod(0o700)
            command = r'''
              set -euo pipefail
              source "$1"
              case "$(uname -m)" in
                arm64|aarch64) export CX_IMAGE_PLATFORM=linux/arm64 ;;
                *) export CX_IMAGE_PLATFORM=linux/amd64 ;;
              esac
              export COLLECTIVEX_EXECUTION_ID="preflight-success-$$"
              cx_preflight_allocation 42 2 "$2" "$3" ""
              test ! -e "$2/.collectivex-preflight"
              cx_cleanup_private_logs 0
              export COLLECTIVEX_EXECUTION_ID="preflight-node-$$" FAKE_TASKS=1
              ! cx_preflight_allocation 42 2 "$2" "$3" ""
              test ! -e "$2/.collectivex-preflight"
              cx_cleanup_private_logs 0
              export COLLECTIVEX_EXECUTION_ID="preflight-missing-$$"
              export FAKE_TASKS=2 FAKE_MODE=missing-source
              ! cx_preflight_allocation 42 2 "$2" "$3" ""
              test ! -e "$2/.collectivex-preflight"
              cx_cleanup_private_logs 0
              export COLLECTIVEX_EXECUTION_ID="preflight-readonly-$$" FAKE_MODE=readonly
              ! cx_preflight_allocation 42 2 "$2" "$3" ""
              test ! -e "$2/.collectivex-preflight"
              cx_cleanup_private_logs 0
            '''
            subprocess.run(
                [
                    "bash", "-c", command, "_",
                    str(ROOT / "runtime" / "common.sh"), str(mount), str(squash),
                ],
                check=True,
                env={
                    **os.environ,
                    "PATH": f"{binary}:{os.environ['PATH']}",
                    "FAKE_ROOT": str(root),
                    "FAKE_TASKS": "2",
                    "COLLECTIVEX_OPERATOR_CONFIG": "/dev/null",
                },
            )

    def test_image_pinned_deepep_and_input_integrity_order_are_explicit(self) -> None:
        runtime = (ROOT / "runtime" / "run_in_container.sh").read_text()
        probe = runtime[runtime.index("cx_probe_deepep()"):
                        runtime.index("cx_activate_deepep_v2()")]
        self.assertIn('expected_version="1.2.1"', probe)
        self.assertIn('expected_version="1.1.0+814e508"', probe)
        self.assertNotIn("pip install", probe)
        self.assertNotIn("cx_fetch_revision", probe)
        self.assertIn("Path(deep_ep.__file__).resolve() in recorded_files", probe)
        self.assertIn("Path(buffer_module.__file__).resolve() in recorded_files", probe)

        harness = (HERE / "ep_harness.py").read_text()
        pass_one = harness[harness.index("# ---- Pass 1"):
                           harness.index("# ---- Pass 2")]
        self.assertLess(
            pass_one.index("input_snapshots[T] ="),
            pass_one.index("oracle = _run_expert_oracle"),
        )
        self.assertIn("pre_input_unchanged", pass_one)
        self.assertIn(
            "hh = prep_combine()\n                        torch.cuda.synchronize()",
            harness,
        )

    def test_squash_imports_are_reproducible_and_use_a_fresh_cache_key(self) -> None:
        common = (ROOT / "runtime" / "common.sh").read_text()
        amd = (ROOT / "launchers" / "launch_mi-amds.sh").read_text()
        self.assertIn('CX_SQUASH_FORMAT_VERSION="repro-v1"', common)
        self.assertIn("SOURCE_DATE_EPOCH=\"$CX_SQUASH_SOURCE_DATE_EPOCH\"", common)
        self.assertIn("${COLLECTIVEX_IMAGE_DIGEST#sha256:}", common)
        self.assertIn("cx_ensure_squash_on_job", amd)
        self.assertIn('"${CX_LOCK_DIR:-}"', amd)
        self.assertNotIn('"${CX_LOCK_DIR:-/tmp}"', amd)
        self.assertIn('[ -n "$lock_dir" ] || lock_dir="$squash_dir/.locks"', common)
        self.assertGreaterEqual(common.count("--chdir=/tmp"), 2)
        self.assertGreaterEqual(amd.count("--chdir=/tmp"), 2)
        self.assertIn('ENROOT_CACHE_PATH="$compute_home/enroot-cache"', common)
        self.assertIn('ENROOT_RUNTIME_PATH="$compute_home/enroot-run"', common)
        self.assertEqual(common.count('cx_reverify_registry_image "$image"'), 2)
        result = subprocess.run(
            [
                "bash",
                "-c",
                f'source "{ROOT / "runtime" / "common.sh"}"; '
                'COLLECTIVEX_IMAGE_DIGEST="sha256:$(printf b%.0s {1..64})"; '
                'CX_IMAGE_PLATFORM=linux/amd64; cx_squash_path /cache repo/image:tag; '
                'printf "\\n"; CX_IMAGE_PLATFORM=linux/arm64; '
                'cx_squash_path /cache repo/image:tag',
            ],
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        digest = "b" * 64
        self.assertEqual(
            result.stdout.splitlines(),
            [
                f"/cache/repro-v1_{digest}_repo_image_tag.sqsh",
                f"/cache/repro-v1_linux_arm64_{digest}_repo_image_tag.sqsh",
            ],
        )

    def test_launchers_preserve_platform_specific_runtime_requirements(self) -> None:
        single = (ROOT / "launchers" / "launch_single-slurm.sh").read_text()
        gb = (ROOT / "launchers" / "launch_gb-nv.sh").read_text()
        amd = (ROOT / "launchers" / "launch_mi-amds.sh").read_text()
        common = (ROOT / "runtime" / "common.sh").read_text()
        self.assertIn("ALLOC_EXTRA=(--mem=0)", single)
        self.assertIn("ALLOC_EXTRA=(-N 1 --mem=0)", single)
        self.assertIn("SRUN_EXTRA=(--mpi=none --container-remap-root)", single)
        self.assertIn("CX_ENROOT_LOCAL_IMPORT=1", single)
        self.assertIn('PRODUCT="${CX_SHARD_SKU:-${CX_GB_PRODUCT:-', gb)
        self.assertIn("cx_ensure_squash_on_job", gb)
        self.assertIn("--mem=0 --cpus-per-task=35", gb)
        self.assertIn("--container-writable", gb)
        self.assertIn("--container-remap-root", gb)
        workload_stage = common[
            common.index("workload_args=("):
            common.index("workload_log=", common.index("workload_args=("))
        ]
        self.assertNotIn("--workload", workload_stage)
        self.assertIn('cd -P -- "$(dirname "${BASH_SOURCE[0]}")"', amd)
        self.assertIn("mi325x) CPUS_PER_TASK=256", amd)
        self.assertIn("/dev/kfd:/dev/kfd,/dev/dri:/dev/dri", amd)
        self.assertIn("--container-writable --container-remap-root", amd)
        self.assertIn(
            "CX_DISTRIBUTED_CONTAINER_ARGS=(--container-writable --container-remap-root)",
            amd,
        )
        collect = common[common.index("cx_collect_results()"):
                         common.index("cx_cleanup_stage()")]
        cleanup = common[common.index("cx_launcher_cleanup()"):
                         common.index("cx_install_launcher_fail_safe()")]
        self.assertNotIn("cx_cleanup_stage", collect)
        self.assertLess(cleanup.index("cx_cancel_job"), cleanup.index("cx_cleanup_stage"))
        runtime = (ROOT / "runtime" / "run_in_container.sh").read_text()
        self.assertIn('distribution.read_text("direct_url.json")', runtime)
        self.assertIn("6548e9c504a12b2471af4b7f4d9546321210a57a456b5dc55bd4a8dad0f932ac", runtime)
        self.assertIn("2671cff7baf8c2c214ff4bac721af875d513130670bec57601998bd1aae82882", runtime)
        stage_cleanup = common[
            common.index("cx_cleanup_stage()"):
            common.index("cx_has_result_doc()")
        ]
        self.assertIn('allow_uid_mapped_root = sys.argv[3] == "1"', stage_cleanup)
        self.assertIn("base_metadata.st_uid == 0", stage_cleanup)
        self.assertIn("stat.S_IMODE(base_metadata.st_mode) == 0o700", stage_cleanup)
        self.assertIn("parent_metadata.st_uid != 0", stage_cleanup)
        self.assertIn("stat.S_IWGRP | stat.S_IWOTH", stage_cleanup)

    def test_deferred_backend_provenance_resolves_before_measurement(self) -> None:
        harness = (ROOT / "tests" / "ep_harness.py").read_text()
        conditioning = harness.index("for wt in conditioning_ladder")
        provenance = harness.index("# Setup may materialize deferred provenance")
        measurement = harness.index("# ---- Pass 1: build each deterministic problem")
        self.assertLess(conditioning, provenance)
        self.assertLess(provenance, measurement)

    def test_backend_specific_routing_contracts_are_explicit(self) -> None:
        hybrid = (ROOT / "tests" / "ep_deepep_hybrid.py").read_text()
        self.assertIn("self.domain_rank = int(self.buffer.local_rank)", hybrid)
        self.assertIn(
            "probability_columns = self.domain_rank * self.local_experts + local_expert_ids",
            hybrid,
        )
        self.assertIn("h.recv_probs[:count][rows, probability_columns]", hybrid)

        mori = (ROOT / "tests" / "ep_mori.py").read_text()
        self.assertIn("topk_idx=indices", mori)
        self.assertIn("indices=indices", mori)
        self.assertIn(
            "combine_indices = p.indices if self._async_ll else h.dispatch_indices",
            mori,
        )
        self.assertIn("h.combine_input,\n            None,\n            combine_indices", mori)
        self.assertIn('"use_external_inp_buf": self._external_input', mori)
        self.assertIn("self.block_num = self._block_target = 64", mori)
        self.assertIn('config_kwargs["block_num"] = self.block_num', mori)
        self.assertIn(
            'config_kwargs["warp_num_per_block"] = self.dispatch_warps', mori
        )
        self.assertIn("count > tensor.size(0)", mori)
        self.assertIn("return combined[:p.T]", mori)
        self.assertNotIn("return combined\n", mori)
        self.assertIn(
            "raw_expert_ids < local_start + experts_per_rank",
            mori,
        )
        self.assertNotIn("MoRI returned a non-local expert", mori)
        harness = (ROOT / "tests" / "ep_harness.py").read_text()
        self.assertIn("problem.recv_tokens = backend.recv_tokens(handle)", harness)

    def test_mori_masks_global_topk_metadata_to_the_local_rank(self) -> None:
        path = HERE / "ep_mori.py"
        tree = ast.parse(path.read_text(), str(path))
        helper = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_project_local_metadata"
        )
        namespace: dict[str, object] = {}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), str(path), "exec"), namespace)
        raw_ids = np.array([[0, 32, 63, -1], [64, 95, 7, 96]], dtype=np.int64)
        raw_weights = np.arange(8, dtype=np.float32).reshape(2, 4)
        torch_module = types.SimpleNamespace(
            where=np.where,
            full_like=np.full_like,
            zeros_like=np.zeros_like,
        )
        ids, weights, local_ids = namespace["_project_local_metadata"](
            torch_module, raw_ids, raw_weights, 1, 32
        )
        np.testing.assert_array_equal(
            ids,
            np.array([[-1, 32, 63, -1], [-1, -1, -1, -1]], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            weights,
            np.array([[0, 1, 2, 0], [0, 0, 0, 0]], dtype=np.float32),
        )
        counts = np.bincount(local_ids, minlength=32)
        self.assertEqual((counts[0], counts[31], int(counts.sum())), (1, 1, 2))
        commit_helper = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_mori_source_commit"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            module = root / "python" / "mori" / "__init__.py"
            module.parent.mkdir(parents=True)
            module.touch()
            git = root / ".git"
            git.mkdir()
            (git / "HEAD").write_text("a" * 40 + "\n")
            commit_namespace = {
                "Path": Path,
                "re": re,
                "mori": types.SimpleNamespace(__file__=str(module)),
            }
            exec(
                compile(ast.Module(body=[commit_helper], type_ignores=[]), str(path), "exec"),
                commit_namespace,
            )
            self.assertEqual(commit_namespace["_mori_source_commit"](), "a" * 40)
            (git / "HEAD").write_text("ref: refs/heads/main\n")
            with self.assertRaisesRegex(RuntimeError, "detached commit"):
                commit_namespace["_mori_source_commit"]()

        profile = contracts.project_resource_profile(
            {
                "block_num": 64,
                "device_cus": 304,
                "kernel_type": "AsyncLL",
                "tuned_source": "upstream-asyncll-64x8-external-input",
            }
        )
        self.assertIsNone(profile["comm_units_kind"])
        self.assertIsNone(profile["configured_units"])

    def test_mori_zipf_capacity_covers_worst_rank_skew(self) -> None:
        path = HERE / "ep_mori.py"
        tree = ast.parse(path.read_text(), str(path))
        helper = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_mori_input_capacity"
        )
        namespace: dict[str, object] = {}
        exec(
            compile(ast.Module(body=[helper], type_ignores=[]), str(path), "exec"),
            namespace,
        )
        capacity = namespace["_mori_input_capacity"]
        self.assertEqual(capacity(types.SimpleNamespace(routing="uniform"), 8), 512)
        self.assertEqual(
            capacity(
                types.SimpleNamespace(
                    routing="zipf", tokens_ladder="512", phase="prefill"
                ),
                8,
            ),
            4096,
        )
        self.assertEqual(
            capacity(
                types.SimpleNamespace(
                    routing="zipf", tokens_ladder="128", phase="decode"
                ),
                8,
            ),
            1024,
        )

    def test_squash_identity_rehashes_instead_of_trusting_a_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "image.sqsh"
            image.write_bytes(b"current squash bytes")
            sidecar = Path(f"{image}.sha256")
            sidecar.write_text("a" * 64)
            os.utime(sidecar, (image.stat().st_mtime + 10, image.stat().st_mtime + 10))
            result = subprocess.run(
                [
                    "bash", "-c",
                    'source "$1"; COLLECTIVEX_EXECUTION_ID="squash-hash-$$"; '
                    'cx_export_squash_identity "$2"; cx_cleanup_private_logs 0; '
                    'printf "%s" "$COLLECTIVEX_SQUASH_SHA256"',
                    "_", str(ROOT / "runtime" / "common.sh"), str(image),
                ],
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, hashlib.sha256(image.read_bytes()).hexdigest())

    def _run_salloc_scenario(
        self, salloc_body: str, squeue_body: str, *, cleanup: bool
    ) -> dict[str, object]:
        prefix = f"inferencex-collectivex-{os.getpid()}-1-"
        with tempfile.TemporaryDirectory(prefix=prefix, dir="/tmp") as temporary:
            root = Path(temporary)
            command_dir = root / "bin"
            repo = root / "repo"
            command_dir.mkdir()
            repo.mkdir()
            paths = {
                name: root / name
                for name in ("arguments", "squeue-calls", "sleep-calls", "scancel-calls")
            }
            scripts = {
                "salloc": (
                    "printf '%s\\n' \"$@\" > \"$CX_TEST_SALLOC_ARGUMENTS\"\n"
                    + salloc_body
                ),
                "squeue": (
                    "printf '%s\\n' \"$*\" >> \"$CX_TEST_SQUEUE_CALLS\"\n"
                    + squeue_body
                ),
                "sleep": "printf '%s\\n' \"$1\" >> \"$CX_TEST_SLEEP_CALLS\"\n",
                "scancel": (
                    "printf '%s\\n' \"$*\" >> \"$CX_TEST_SCANCEL_CALLS\"\n"
                ),
            }
            for name, body in scripts.items():
                path = command_dir / name
                path.write_text(f"#!/usr/bin/env bash\n{body}\n")
                path.chmod(0o700)
            execution_id = f"scheduler-{root.name}"
            expected_name = "cx-" + hashlib.sha256(
                execution_id.encode()
            ).hexdigest()[:24]
            command = r'''
              source "$1"
              JOB_ID=""
              set +e
              cx_salloc_jobid --partition=compute
              run_rc=$?
              set -e
              printf '%s:%s:%s\n' \
                "$run_rc" "$JOB_ID" "$CX_ALLOCATION_UNCERTAIN"
              cx_cleanup_private_logs 0
              if [ "$3" = cleanup ]; then
                export CX_JOB_ROOT="$2" REPO_ROOT="$2/repo" MOUNT_SRC="$2/repo"
                export COLLECTIVEX_CANONICAL_GHA=1
                cx_clear_allocation_jobid() { rm -f -- "$CX_JOB_ROOT/allocation-job-id"; }
                cx_write_cleanup_guard() {
                  rm -f -- "$CX_JOB_ROOT/cleanup-safe" "$CX_JOB_ROOT/cleanup-unsafe"
                  : > "$CX_JOB_ROOT/cleanup-$1"
                }
                unset CX_BENCH
                cx_launcher_cleanup "$run_rc"
              fi
              exit "$run_rc"
            '''
            result = subprocess.run(
                [
                    "bash", "-c", command, "_", str(ROOT / "runtime" / "common.sh"),
                    str(root), "cleanup" if cleanup else "no-cleanup",
                ],
                text=True,
                capture_output=True,
                env={
                    **os.environ,
                    "PATH": f"{command_dir}:{os.environ['PATH']}",
                    "COLLECTIVEX_EXECUTION_ID": execution_id,
                    "CX_TEST_SALLOC_ARGUMENTS": str(paths["arguments"]),
                    "CX_TEST_SQUEUE_CALLS": str(paths["squeue-calls"]),
                    "CX_TEST_SLEEP_CALLS": str(paths["sleep-calls"]),
                    "CX_TEST_SCANCEL_CALLS": str(paths["scancel-calls"]),
                },
            )
            return {
                "result": result,
                "job_name": expected_name,
                "arguments": paths["arguments"].read_text().splitlines(),
                "squeue_calls": (
                    paths["squeue-calls"].read_text().splitlines()
                    if paths["squeue-calls"].exists() else []
                ),
                "sleep_calls": (
                    paths["sleep-calls"].read_text().splitlines()
                    if paths["sleep-calls"].exists() else []
                ),
                "scancel_calls": (
                    paths["scancel-calls"].read_text().splitlines()
                    if paths["scancel-calls"].exists() else []
                ),
                "cleanup_safe": (root / "cleanup-safe").is_file(),
                "cleanup_unsafe": (root / "cleanup-unsafe").is_file(),
            }

    def test_salloc_job_id_parser_uses_the_portable_grant_message(self) -> None:
        scenario = self._run_salloc_scenario(
            "printf 'salloc: Granted job allocation 4242\\n' >&2",
            "exit 2",
            cleanup=False,
        )
        result = scenario["result"]
        self.assertIsInstance(result, subprocess.CompletedProcess)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout, "0:4242:0\n"
        )
        self.assertEqual(
            scenario["arguments"],
            [
                "--partition=compute",
                f"--job-name={scenario['job_name']}",
                "--no-shell",
            ],
        )
        self.assertIn("scheduler-request=submit", result.stderr)
        self.assertEqual(scenario["squeue_calls"], [])

    def test_salloc_job_id_parser_ignores_duplicate_wrapper_output(self) -> None:
        scenario = self._run_salloc_scenario(
            "for _ in {1..10000}; do "
            "printf 'salloc: Granted job allocation 4242\\n' >&2; done",
            "exit 2",
            cleanup=False,
        )
        result = scenario["result"]
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "0:4242:0\n")
        self.assertNotIn("grant-parse", result.stderr)

    def test_salloc_contains_a_shell_wrapper_that_calls_exit(self) -> None:
        common = ROOT / "runtime" / "common.sh"
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "scheduler.log"
            command = r'''
              set -euo pipefail
              source "$1"
              cx_private_log_path() { : > "$CX_TEST_LOG"; printf '%s' "$CX_TEST_LOG"; }
              salloc() { printf 'salloc: Granted job allocation 4242\n' >&2; exit 0; }
              export COLLECTIVEX_EXECUTION_ID=wrapper-exit
              JOB_ID=""
              cx_salloc_jobid --partition=compute
              test "$JOB_ID" = 4242
            '''
            result = subprocess.run(
                ["bash", "-c", command, "_", str(common), str(log)],
                text=True,
                capture_output=True,
                env={**os.environ, "CX_TEST_LOG": str(log)},
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_salloc_verified_rejection_is_cleanup_safe(self) -> None:
        scenario = self._run_salloc_scenario("exit 1", "exit 0", cleanup=True)
        result = scenario["result"]
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "1::0\n")
        self.assertIn("scheduler-request=rejected", result.stderr)
        self.assertIn(
            "failure-class=scheduler-allocation diagnostic=empty-log",
            result.stderr,
        )
        self.assertEqual(len(scenario["squeue_calls"]), 3)
        scheduler_user = subprocess.check_output(["id", "-un"], text=True).strip()
        self.assertTrue(all(
            f"--name={scenario['job_name']}" in call
            and f"--user={scheduler_user}" in call
            for call in scenario["squeue_calls"]
        ))
        self.assertEqual(scenario["sleep_calls"], ["1", "2"])
        self.assertTrue(scenario["cleanup_safe"])
        self.assertFalse(scenario["cleanup_unsafe"])

    def test_salloc_recovers_and_cancels_one_matching_allocation(self) -> None:
        scenario = self._run_salloc_scenario(
            "exit 1",
            r'''
              case " $* " in
                *" --name="*) printf '5151\n' ;;
                *" -j 5151 "*) exit 0 ;;
                *) exit 2 ;;
              esac
            ''',
            cleanup=True,
        )
        result = scenario["result"]
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "1:5151:0\n")
        self.assertEqual(scenario["scancel_calls"], ["5151"])
        self.assertTrue(scenario["cleanup_safe"])
        self.assertFalse(scenario["cleanup_unsafe"])

    def test_salloc_ambiguous_lookup_remains_cleanup_unsafe(self) -> None:
        scenario = self._run_salloc_scenario(
            "exit 1", "printf '5151\\n5152\\n'", cleanup=True
        )
        result = scenario["result"]
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "1::1\n")
        self.assertEqual(scenario["scancel_calls"], [])
        self.assertFalse(scenario["cleanup_safe"])
        self.assertTrue(scenario["cleanup_unsafe"])

    def test_salloc_query_failure_and_interruption_remain_cleanup_unsafe(self) -> None:
        query_failure = self._run_salloc_scenario("exit 1", "exit 2", cleanup=True)
        self.assertEqual(query_failure["result"].returncode, 1)
        self.assertEqual(len(query_failure["squeue_calls"]), 1)
        self.assertFalse(query_failure["cleanup_safe"])
        self.assertTrue(query_failure["cleanup_unsafe"])

        interrupted = self._run_salloc_scenario("exit 130", "exit 0", cleanup=True)
        self.assertEqual(interrupted["result"].returncode, 1)
        self.assertEqual(interrupted["squeue_calls"], [])
        self.assertFalse(interrupted["cleanup_safe"])
        self.assertTrue(interrupted["cleanup_unsafe"])

    def test_allocation_cleanup_fails_closed_when_scheduler_queries_fail(self) -> None:
        common = (ROOT / "runtime" / "common.sh").read_text()
        self.assertIn("for delay in 1 2 4 8 16 32 64; do", common)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name, body in {
                "scancel": "exit 0",
                "squeue": "exit 2",
                "sleep": "exit 0",
            }.items():
                command = directory / name
                command.write_text(f"#!/usr/bin/env bash\n{body}\n")
                command.chmod(0o700)
            result = subprocess.run(
                [
                    "bash", "-c",
                    'source "$1"; cx_cancel_job 4242',
                    "_", str(ROOT / "runtime" / "common.sh"),
                ],
                text=True,
                capture_output=True,
                env={**os.environ, "PATH": f"{directory}:{os.environ['PATH']}"},
            )
            self.assertNotEqual(result.returncode, 0)
        self.assertIn("did not terminate", result.stderr)

    def test_launcher_signal_traps_cancel_the_allocation(self) -> None:
        prefix = f"inferencex-collectivex-{os.getpid()}-1-"
        with tempfile.TemporaryDirectory(prefix=prefix, dir="/tmp") as temporary:
            root = Path(temporary)
            binary = root / "bin"
            repo = root / "repo"
            calls = root / "scancel-calls"
            binary.mkdir()
            repo.mkdir()
            for name, body in {
                "scancel": 'printf "%s\\n" "$*" >> "$CX_TEST_SCANCEL_CALLS"',
                "squeue": "exit 0",
            }.items():
                path = binary / name
                path.write_text(f"#!/usr/bin/env bash\n{body}\n")
                path.chmod(0o700)
            command = r'''
              set -euo pipefail
              source "$1"
              export JOB_ID=5151 CX_JOB_ROOT="$2" REPO_ROOT="$2/repo" MOUNT_SRC="$2/repo"
              export COLLECTIVEX_CANONICAL_GHA=1
              cx_clear_allocation_jobid() { rm -f -- "$CX_JOB_ROOT/allocation-job-id"; }
              cx_write_cleanup_guard() {
                rm -f -- "$CX_JOB_ROOT/cleanup-safe" "$CX_JOB_ROOT/cleanup-unsafe"
                : > "$CX_JOB_ROOT/cleanup-$1"
              }
              cx_install_launcher_fail_safe
              kill -TERM $$
              exit 99
            '''
            result = subprocess.run(
                ["bash", "-c", command, "_", str(ROOT / "runtime" / "common.sh"), str(root)],
                text=True,
                capture_output=True,
                env={
                    **os.environ,
                    "PATH": f"{binary}:{os.environ['PATH']}",
                    "CX_TEST_SCANCEL_CALLS": str(calls),
                },
            )
            self.assertEqual(result.returncode, 143, result.stderr)
            self.assertEqual(calls.read_text().splitlines(), ["5151"])
            self.assertTrue((root / "cleanup-safe").is_file())
            self.assertFalse((root / "cleanup-unsafe").exists())

        workflow = (ROOT.parent.parent / ".github" / "workflows" / "collectivex-sweep.yml").read_text()
        self.assertIn("cleanup-unsafe", workflow)
        self.assertIn("cleanup-safe", workflow)
        self.assertIn("Reconcile allocation cleanup", workflow)
        self.assertIn("Confirm allocation cleanup", workflow)
        self.assertLess(
            workflow.index("Reconcile allocation cleanup"),
            workflow.index("Confirm allocation cleanup"),
        )
        self.assertIn("Prepare pinned backend source archive", workflow)
        self.assertIn("Install pinned backend source seed", workflow)
        self.assertIn("CX_BACKEND_SOURCE_SEED_ROOT", workflow)
        self.assertIn("steps.gen.outputs.source_backends", workflow)
        self.assertIn("secrets.COLLECTIVEX_NETWORK_CONFIG_V1", workflow)
        self.assertIn("secrets.COLLECTIVEX_B200_CONFIG_V1", workflow)
        self.assertIn("secrets.COLLECTIVEX_MI300_CONFIG_V1", workflow)
        self.assertIn('set(overlay["runners"]) != {"b200-dgxc"}', workflow)
        self.assertIn('set(overlay["runners"]) != {"mi300x"}', workflow)
        self.assertIn('overlay_name != "COLLECTIVEX_MI300_CONFIG_CONTENT"', workflow)
        self.assertIn('"partition", "squash_dir", "stage_dir", "exclude_nodes"', workflow)
        self.assertIn("set(fields) - private_fields", workflow)
        self.assertIn('private_fields.add("exclude_nodes")', workflow)
        self.assertIn('invalid H100 overlay runner', workflow)
        self.assertIn('private_fields.update({"nodelist", "qos"})', workflow)
        self.assertIn('[ -z "${CX_QOS:-}" ] || allocation+=(--qos="$CX_QOS")', (
            ROOT / "launchers" / "launch_single-slurm.sh"
        ).read_text())
        self.assertIn('[ -z "${CX_NODELIST:-}" ] || allocation+=(--nodelist="$CX_NODELIST")', (
            ROOT / "launchers" / "launch_single-slurm.sh"
        ).read_text())
        self.assertIn("COLLECTIVEX_OPERATOR_CONFIG_EPHEMERAL=1", workflow)
        self.assertLess(
            workflow.index("unset COLLECTIVEX_OPERATOR_CONFIG_REQUIRED"),
            workflow.index('export COLLECTIVEX_OPERATOR_CONFIG="$operator_config"'),
        )

    def test_workflow_reconciles_a_recorded_allocation_after_launcher_loss(self) -> None:
        workflow = (
            ROOT.parent.parent / ".github" / "workflows" / "collectivex-sweep.yml"
        ).read_text()
        prefix = f"inferencex-collectivex-{os.getpid()}-1-"
        with tempfile.TemporaryDirectory(prefix=prefix, dir="/tmp") as temporary:
            root = Path(temporary)
            binary = root / "bin"
            calls = root / "scancel-calls"
            binary.mkdir()
            for name, body in {
                "scancel": 'printf "%s\\n" "$*" >> "$CX_TEST_SCANCEL_CALLS"',
                "squeue": "exit 0",
                "stat": (
                    'case "$3" in */allocation-job-id) mode=600 ;; *) mode=700 ;; esac\n'
                    'printf "%s:%s\\n" "$CX_TEST_UID" "$mode"'
                ),
            }.items():
                path = binary / name
                path.write_text(f"#!/usr/bin/env bash\n{body}\n")
                path.chmod(0o700)
            command = r'''
              set -euo pipefail
              source "$1"
              export CX_JOB_ROOT="$2"
              : > "$CX_JOB_ROOT/cleanup-unsafe"
              cx_record_allocation_jobid 6262
              cx_reconcile_recorded_allocation "$CX_JOB_ROOT"
            '''
            result = subprocess.run(
                ["bash", "-c", command, "_", str(ROOT / "runtime" / "common.sh"), str(root)],
                text=True,
                capture_output=True,
                env={
                    **os.environ,
                    "PATH": f"{binary}:{os.environ['PATH']}",
                    "CX_TEST_SCANCEL_CALLS": str(calls),
                    "CX_TEST_UID": str(os.getuid()),
                },
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(calls.read_text().splitlines(), ["6262"])
            self.assertTrue((root / "cleanup-safe").is_file())
            self.assertFalse((root / "cleanup-unsafe").exists())
            self.assertFalse((root / "allocation-job-id").exists())
        prepare_start = workflow.index("- name: Prepare pinned backend source archive")
        source_archive_step = workflow[
            prepare_start:workflow.index("- uses: actions/upload-artifact", prepare_start)
        ]
        self.assertIn('--no-recursion -C "$work/experimental/CollectiveX"', source_archive_step)
        self.assertIn('members+=(".cx_sources/$source_basename")', source_archive_step)
        self.assertIn('-rf "$archive" "$member"', source_archive_step)
        self.assertIn('python3 "$destination/source_archive.py"', workflow)
        install_source = workflow[workflow.index("- name: Install pinned backend source seed"):]
        self.assertIn('chmod 600 -- "$archive"', install_source)
        self.assertLess(
            install_source.index('chmod 600 -- "$archive"'),
            install_source.index('python3 "$destination/source_archive.py"'),
        )
        artifact_validation = workflow[workflow.index("- name: Validate shard artifact safety"):]
        self.assertIn("steps.allocation_cleanup.outcome == 'success'", artifact_validation)
        self.assertIn(
            "inputs.operation != 'probe-precision' || steps.sweep_shard.outcome == 'success'",
            artifact_validation,
        )
        cleanup_function = (ROOT / "runtime" / "common.sh").read_text()
        self.assertIn('[ "${CX_PRECISION_PROBE:-0}" != 1 ]', cleanup_function)
        self.assertIn("precision-probe-failure-class=", cleanup_function)
        sweep_workflow = workflow[workflow.index("  sweep:"):]
        self.assertNotIn("GITHUB_WORKSPACE", sweep_workflow)
        self.assertNotIn("RUNNER_WORKSPACE", sweep_workflow)
        self.assertNotIn("github.workspace", sweep_workflow)
        self.assertIn("matrix.launcher == 'mi-amds'", sweep_workflow)
        self.assertIn("CX_JOB_PARENT:", sweep_workflow)
        self.assertIn("/tmp/inferencex-collectivex-parent-", sweep_workflow)
        self.assertIn("|| '/tmp'", sweep_workflow)
        source_step = sweep_workflow[:sweep_workflow.index("- uses: actions/download-artifact")]
        self.assertNotIn("unsafe_guards=", source_step)
        self.assertIn('shared_parent = os.path.join(os.getcwd(), ".collectivex-jobs")', source_step)
        self.assertIn("os.lstat(shared_parent)", source_step)
        self.assertIn("os.symlink(shared_parent, parent)", source_step)
        self.assertIn("for entry in os.scandir(scan_parent)", source_step)
        self.assertIn("cutoff = time.time() - 86400", source_step)
        self.assertIn("stat.S_IMODE(metadata.st_mode) != 0o700", source_step)
        self.assertIn('for marker_name in ("cleanup-safe", "cleanup-unsafe")', source_step)
        self.assertIn("stat.S_IMODE(marker.st_mode) == 0o600", source_step)
        self.assertIn("shutil.rmtree(entry.path)", source_step)
        self.assertLess(
            source_step.index('rev-parse HEAD'),
            source_step.index("echo 'prepared=true'"),
        )
        upload = workflow[workflow.index("- name: Stage shard artifact"):]
        self.assertIn("id: stage_artifact", upload)
        self.assertIn("id: upload_artifact", upload)
        self.assertIn("steps.stage_artifact.outcome == 'success'", upload)
        cleanup = workflow[workflow.index("- name: Cleanup isolated workspace"):]
        self.assertIn("CollectiveX cleanup parent is invalid", cleanup)
        self.assertIn('[ "${CX_JOB_ROOT%/*}" = "$CX_JOB_PARENT" ]', cleanup)
        self.assertIn('^inferencex-collectivex-', cleanup)
        self.assertIn('rm -f -- "$CX_JOB_PARENT"', cleanup)
        self.assertIn("steps.sweep_shard.outcome }}' != success", cleanup)
        self.assertIn('[ "$CX_JOB_PARENT" != /tmp ]', cleanup)
        for step in (
            "sweep_shard", "allocation_cleanup", "artifact_safety",
            "delivery_contracts", "stage_artifact", "upload_artifact",
        ):
            self.assertIn(f"steps.{step}.outcome", cleanup)
        self.assertLess(
            cleanup.index('cleanup-safe" ]'),
            cleanup.index('rm -rf -- "$CX_JOB_ROOT"'),
        )

    def test_v1_publication_requires_explicit_release_markers(self) -> None:
        workflows = ROOT.parent.parent / ".github" / "workflows"
        sweep = (workflows / "collectivex-sweep.yml").read_text()
        self.assertFalse((workflows / "collectivex-publish.yml").exists())

        self.assertIn("options: [sweep, probe-precision, publish-v1, refresh-v1]", sweep)
        self.assertIn("${{ inputs.only_sku }}-${{ inputs.probe_id }}", sweep)
        self.assertIn("cancel-in-progress: false", sweep)
        self.assertIn("COLLECTIVEX_MI355_CONFIG_V1", sweep)
        self.assertIn("COLLECTIVEX_MI300_CONFIG_V1", sweep)
        self.assertIn('set(overlay["runners"]) != {"mi300x"}', sweep)
        self.assertIn('set(overlay["runners"]) != {"mi355x"}', sweep)
        self.assertIn("collectivex.precision-probe-plan.v1", (ROOT / "tests" / "probe_precision.py").read_text())
        self.assertIn("cxprecision-probes-${{ github.run_id }}-${{ github.run_attempt }}", sweep)
        self.assertIn("--validate-bundle", sweep)
        self.assertIn("release_tag:", sweep)
        self.assertIn("default: unversioned", sweep)
        self.assertIn("options: [unversioned, v1]", sweep)
        self.assertIn("qualification_index:", sweep)
        self.assertIn("inputs.release_tag == 'v1'", sweep)
        self.assertIn("collectivex.release-tag.v1", sweep)
        self.assertIn("V1 release tag requires the locked full matrix", sweep)
        self.assertIn("EXPECTED_MATRIX_SHA256", sweep)
        self.assertIn("cxrelease-v1-${{ github.run_id }}-${{ github.run_attempt }}", sweep)

        self.assertIn("publish_run_ids must contain exactly three IDs", sweep)
        self.assertIn("source runs do not share one source SHA", sweep)
        self.assertIn("cxrelease-v1-$run_id-$attempt/release.json", sweep)
        self.assertIn("run $run_id is not tagged for V1 publication", sweep)
        self.assertIn("ref: ${{ steps.runs.outputs.source_sha }}", sweep)
        self.assertIn("[ \"$attempt\" = 1 ]", sweep)
        self.assertIn("cxpublication-v1-${{ github.run_id }}-${{ github.run_attempt }}", sweep)
        self.assertIn("refresh source bytes differ from their requested digest", sweep)
        self.assertIn("retention-days: 90", sweep)
        self.assertNotIn("workflow_run:", sweep)

    def test_source_archive_preserves_only_contained_leaf_symlinks(self) -> None:
        selected = "deepep-hybrid-pinned"
        other = "deepep-v2-pinned"

        def directory(name: str) -> tarfile.TarInfo:
            member = tarfile.TarInfo(name)
            member.type = tarfile.DIRTYPE
            member.mode = 0o755
            return member

        def regular(
            name: str, payload: bytes, mode: int = 0o644
        ) -> tuple[tarfile.TarInfo, io.BytesIO]:
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            member.mode = mode
            return member, io.BytesIO(payload)

        def symbolic(name: str, target: str) -> tarfile.TarInfo:
            member = tarfile.TarInfo(name)
            member.type = tarfile.SYMTYPE
            member.linkname = target
            member.mode = 0o777
            return member

        def write_archive(path: Path, extras: list[tarfile.TarInfo] | None = None) -> None:
            root = f".cx_sources/{selected}"
            with tarfile.open(path, "w") as archive:
                for name in (
                    ".cx_sources", root, f"{root}/third-party",
                    f"{root}/third-party/nccl", f"{root}/third-party/nccl/pkg",
                    f"{root}/third-party/nccl/pkg/debian",
                    f".cx_sources/{other}",
                ):
                    archive.addfile(directory(name))
                member, stream = regular(
                    f"{root}/third-party/nccl/LICENSE.txt", b"license\n"
                )
                archive.addfile(member, stream)
                member, stream = regular(f".cx_sources/{other}/sentinel", b"other\n")
                archive.addfile(member, stream)
                member, stream = regular(f"{root}/group-executable", b"exec\n", 0o010)
                archive.addfile(member, stream)
                archive.addfile(symbolic(
                    f"{root}/third-party/nccl/pkg/debian/copyright",
                    "../../LICENSE.txt",
                ))
                for member in extras or []:
                    archive.addfile(member)
            path.chmod(0o600)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            archive = root / "source.tar"
            destination = root / "destination"
            destination.mkdir(mode=0o700)
            write_archive(archive)
            source_archive.extract_source_archive(archive, destination, selected)
            link = (
                destination / ".cx_sources" / selected / "third-party" / "nccl"
                / "pkg" / "debian" / "copyright"
            )
            self.assertTrue(link.is_symlink())
            self.assertEqual(os.readlink(link), "../../LICENSE.txt")
            self.assertEqual(link.read_text(), "license\n")
            self.assertFalse((destination / ".cx_sources" / other).exists())
            extracted = destination / ".cx_sources" / selected
            self.assertEqual(
                stat.S_IMODE((extracted / "group-executable").stat().st_mode), 0o700
            )
            self.assertEqual(
                stat.S_IMODE(
                    (extracted / "third-party" / "nccl" / "LICENSE.txt").stat().st_mode
                ),
                0o600,
            )

        invalid: dict[str, list[tarfile.TarInfo]] = {
            "absolute member": [directory("/outside")],
            "traversal member": [directory(".cx_sources/../outside")],
            "duplicate member": [directory(f".cx_sources/{selected}")],
            "absolute link": [symbolic(f".cx_sources/{selected}/absolute", "/tmp/x")],
            "escaping link": [symbolic(f".cx_sources/{selected}/escape", "../x")],
            "cross-root link": [
                symbolic(f".cx_sources/{selected}/cross", f"../{other}/sentinel")
            ],
            "missing target": [symbolic(f".cx_sources/{selected}/missing", "none")],
        }
        hardlink = tarfile.TarInfo(f".cx_sources/{selected}/hard")
        hardlink.type = tarfile.LNKTYPE
        hardlink.linkname = f".cx_sources/{selected}/third-party/nccl/LICENSE.txt"
        invalid["hardlink"] = [hardlink]
        fifo = tarfile.TarInfo(f".cx_sources/{selected}/fifo")
        fifo.type = tarfile.FIFOTYPE
        invalid["fifo"] = [fifo]
        character = tarfile.TarInfo(f".cx_sources/{selected}/character")
        character.type = tarfile.CHRTYPE
        invalid["character device"] = [character]
        block = tarfile.TarInfo(f".cx_sources/{selected}/block")
        block.type = tarfile.BLKTYPE
        invalid["block device"] = [block]
        unknown = tarfile.TarInfo(f".cx_sources/{selected}/unknown")
        unknown.type = b"Z"
        invalid["unknown type"] = [unknown]
        invalid["unsafe unselected root"] = [
            symbolic(f".cx_sources/{other}/escape", f"../{selected}/group-executable")
        ]
        chain_target = symbolic(
            f".cx_sources/{selected}/chain-target", "third-party/nccl/LICENSE.txt"
        )
        invalid["symlink chain"] = [
            chain_target, symbolic(f".cx_sources/{selected}/chain", "chain-target")
        ]
        linked_child = tarfile.TarInfo(f".cx_sources/{selected}/linked-file/child")
        invalid["symlink parent"] = [
            symbolic(
                f".cx_sources/{selected}/linked-file",
                "third-party/nccl/LICENSE.txt",
            ),
            linked_child,
        ]
        for label, extras in invalid.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                archive = root / "source.tar"
                destination = root / "destination"
                destination.mkdir(mode=0o700)
                write_archive(archive, extras)
                with self.assertRaises(source_archive.SourceArchiveError):
                    source_archive.extract_source_archive(archive, destination, selected)
                self.assertFalse((destination / ".cx_sources").exists())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            archive = root / "source.tar"
            destination = root / "destination"
            destination.mkdir(mode=0o700)
            existing = destination / ".cx_sources"
            existing.mkdir(mode=0o700)
            marker = existing / "marker"
            marker.write_text("existing\n")
            write_archive(archive)
            with self.assertRaises(source_archive.SourceArchiveError):
                source_archive.extract_source_archive(archive, destination, selected)
            self.assertEqual(marker.read_text(), "existing\n")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            archive = root / "source.tar"
            write_archive(archive)
            real_destination = root / "real-destination"
            real_destination.mkdir(mode=0o700)
            linked_destination = root / "linked-destination"
            linked_destination.symlink_to(real_destination, target_is_directory=True)
            with self.assertRaises((OSError, source_archive.SourceArchiveError)):
                source_archive.extract_source_archive(archive, linked_destination, selected)
            self.assertFalse((real_destination / ".cx_sources").exists())

            unsafe_destination = root / "unsafe-destination"
            unsafe_destination.mkdir(mode=0o700)
            unsafe_destination.chmod(0o755)
            with self.assertRaises(source_archive.SourceArchiveError):
                source_archive.extract_source_archive(archive, unsafe_destination, selected)
            self.assertFalse((unsafe_destination / ".cx_sources").exists())

        for limit, value in (
            ("MAX_ARCHIVE_MEMBERS", 1),
            ("MAX_MEMBER_BYTES", 1),
            ("MAX_EXPANDED_BYTES", 1),
            ("MAX_ARCHIVE_BYTES", 1),
            ("MAX_ARCHIVE_HEADERS", 1),
        ):
            with self.subTest(limit=limit), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                archive = root / "source.tar"
                destination = root / "destination"
                destination.mkdir(mode=0o700)
                write_archive(archive)
                with mock.patch.object(source_archive, limit, value):
                    with self.assertRaises(source_archive.SourceArchiveError):
                        source_archive.extract_source_archive(archive, destination, selected)
                self.assertFalse((destination / ".cx_sources").exists())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            archive = root / "source.tar"
            destination = root / "destination"
            destination.mkdir(mode=0o700)
            write_archive(archive)
            long_name = f".cx_sources/{selected}/long-name-result\0".encode()
            with tarfile.open(archive, "a") as handle:
                for _ in range(3):
                    extension = tarfile.TarInfo("././@LongLink")
                    extension.type = tarfile.GNUTYPE_LONGNAME
                    extension.size = len(long_name)
                    handle.addfile(extension, io.BytesIO(long_name))
                member, stream = regular("placeholder", b"payload\n")
                handle.addfile(member, stream)
            archive.chmod(0o600)
            for limit, value in (
                ("MAX_EXTENSION_CHAIN", 1),
                ("MAX_EXTENSION_MEMBER_BYTES", 1),
                ("MAX_EXTENSION_BYTES", len(long_name) * 2),
            ):
                with self.subTest(limit=limit), mock.patch.object(
                    source_archive, limit, value
                ):
                    with self.assertRaises(source_archive.SourceArchiveError):
                        source_archive.extract_source_archive(
                            archive, destination, selected
                        )
                    self.assertFalse((destination / ".cx_sources").exists())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            archive = root / "source.tar"
            destination = root / "destination"
            destination.mkdir(mode=0o700)
            write_archive(archive)
            with tarfile.open(archive, "a", format=tarfile.PAX_FORMAT) as handle:
                member, stream = regular(
                    f".cx_sources/{selected}/sparse-v1", b"1\n0\n1\n"
                )
                member.pax_headers = {
                    "GNU.sparse.major": "1",
                    "GNU.sparse.minor": "0",
                    "GNU.sparse.realsize": "1",
                }
                handle.addfile(member, stream)
            archive.chmod(0o600)
            with self.assertRaises(source_archive.SourceArchiveError):
                source_archive.extract_source_archive(archive, destination, selected)
            self.assertFalse((destination / ".cx_sources").exists())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            archive = root / "source.tar"
            destination = root / "destination"
            destination.mkdir(mode=0o700)
            write_archive(archive)
            original_next = tarfile.TarFile.next

            def sparse_next(handle: tarfile.TarFile) -> tarfile.TarInfo | None:
                member = original_next(handle)
                if member is not None and member.isfile():
                    member.sparse = [(0, 1)]
                return member

            with mock.patch.object(tarfile.TarFile, "next", sparse_next):
                with self.assertRaises(source_archive.SourceArchiveError):
                    source_archive.extract_source_archive(archive, destination, selected)
            self.assertFalse((destination / ".cx_sources").exists())

    def test_runtime_identity_and_realized_placement_are_behavioral(self) -> None:
        self.assertFalse(capability.runtime_identity_issues(
            "mi325x", vendor="amd", arch="gfx942", machine="amd64",
            device_name="AMD Instinct MI325X", device_count=8, world_size=8,
        ))
        self.assertTrue(capability.runtime_identity_issues(
            "mi355x", vendor="amd", arch="gfx942", machine="amd64",
            device_name="AMD Instinct MI325X", device_count=8, world_size=8,
        ))
        records = [("private-a", rank) for rank in range(4)] + [
            ("private-b", rank) for rank in range(4)
        ]
        self.assertEqual(
            run_ep._summarize_realized_placement(
                records, expected_nodes=2, expected_gpus_per_node=4, expected_world_size=8
            ),
            {
                "gpus_per_node": 4,
                "nodes": 2,
                "ranks_per_node": 4,
                "unique_local_ranks": True,
                "valid": True,
            },
        )
        with self.assertRaises(ValueError):
            run_ep._summarize_realized_placement(
                records[:-1] + [("private-b", 2)],
                expected_nodes=2,
                expected_gpus_per_node=4,
                expected_world_size=8,
            )

    def test_private_allocation_stratum_is_salted_ordered_and_rank_consistent(self) -> None:
        salt = "a" * 64
        hosts = ["private-node-b", "private-node-a", "private-node-a"]
        selectors = {
            "ib_gid_index": "3",
            "rdma_devices": "private-hca0:1,private-hca1:1",
            "rdma_service_level": "2",
            "rdma_traffic_class": "104",
            "socket_ifname": "private-if0",
        }
        digest = run_ep._allocation_stratum_sha256(
            hosts, audit_salt=salt, fabric_selectors=selectors, required=True
        )
        self.assertRegex(digest or "", r"^[0-9a-f]{64}$")
        self.assertEqual(
            digest,
            run_ep._allocation_stratum_sha256(
                list(reversed(hosts)),
                audit_salt=salt,
                fabric_selectors=selectors,
                required=True,
            ),
        )
        for changed_hosts, changed_salt, changed_selectors in (
            (hosts + ["private-node-c"], salt, selectors),
            (hosts, "b" * 64, selectors),
            (hosts, salt, {**selectors, "ib_gid_index": "4"}),
        ):
            self.assertNotEqual(
                digest,
                run_ep._allocation_stratum_sha256(
                    changed_hosts,
                    audit_salt=changed_salt,
                    fabric_selectors=changed_selectors,
                    required=True,
                ),
            )
        serialized = json.dumps({"allocation_stratum_sha256": digest})
        private_literals = [
            salt,
            *hosts,
            selectors["rdma_devices"],
            selectors["socket_ifname"],
        ]
        self.assertFalse(any(value in serialized for value in private_literals))
        self.assertNotIn("physical_hosts", serialized)
        self.assertNotIn("fabric_selectors", serialized)
        self.assertEqual(
            run_ep._common_allocation_stratum([digest, digest, digest], required=True),
            digest,
        )
        with self.assertRaisesRegex(ValueError, "differs across distributed ranks"):
            run_ep._common_allocation_stratum([digest, "b" * 64], required=True)
        with self.assertRaisesRegex(ValueError, "requires"):
            run_ep._allocation_stratum_sha256(
                hosts, audit_salt=None, fabric_selectors=selectors, required=True
            )
        self.assertIsNone(run_ep._allocation_stratum_sha256(
            hosts, audit_salt=None, fabric_selectors=selectors, required=False
        ))

    def test_collective_version_and_rccl_fingerprint_are_normalized(self) -> None:
        self.assertEqual(ep_harness.format_collective_version(23004), "2.30.4")
        self.assertEqual(ep_harness.format_collective_version(21805), "2.18.5")
        self.assertEqual(ep_harness.format_collective_version((2, 21, 5)), "2.21.5")

        properties = types.SimpleNamespace(
            multi_processor_count=304, total_memory=1024, warp_size=64
        )
        fake = types.SimpleNamespace(
            __version__="2.9.0",
            version=types.SimpleNamespace(cuda=None, hip="7.2"),
            cuda=types.SimpleNamespace(
                get_device_properties=lambda _device: properties,
                get_device_name=lambda _device: "AMD Instinct MI325X",
                nccl=types.SimpleNamespace(version=lambda: 21805),
            ),
        )
        with mock.patch.object(
            run_ep, "_loaded_collective_version", return_value="2.18.5"
        ):
            fingerprint = run_ep._runtime_fingerprint(
                fake, "device", machine="amd64", vendor="amd", arch="gfx942"
            )
        self.assertEqual(fingerprint["collective_library"], {"kind": "rccl", "version": "2.18.5"})
        self.assertEqual(fingerprint["accelerator_runtime"], {"kind": "hip", "version": "7.2"})

        class FakeCollective:
            @staticmethod
            def ncclGetVersion(pointer) -> int:
                pointer._obj.value = 23004
                return 0

        maps = "0-1 r-xp 0 00:00 0 /runtime/libnccl.so.2\n"
        with (
            mock.patch("builtins.open", return_value=io.StringIO(maps)),
            mock.patch.object(run_ep.os.path, "isfile", return_value=True),
            mock.patch.object(
                run_ep.os.path, "realpath", return_value="/runtime/libnccl.so.2"
            ),
            mock.patch.object(run_ep.ctypes, "CDLL", return_value=FakeCollective()),
        ):
            self.assertEqual(run_ep._loaded_collective_version(), "2.30.4")

        path = HERE / "ep_nccl.py"
        tree = ast.parse(path.read_text(), str(path))
        helper = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_runtime_collective"
        )
        namespace = {"re": re}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), str(path), "exec"), namespace)
        args = types.SimpleNamespace(
            runtime_fingerprint={
                "collective_library": {"kind": "nccl", "version": "2.30.4"}
            }
        )
        cuda = types.SimpleNamespace(version=types.SimpleNamespace(hip=None))
        self.assertEqual(namespace["_runtime_collective"](args, cuda), ("nccl", "2.30.4"))
        args.runtime_fingerprint["collective_library"]["version"] = None
        with self.assertRaisesRegex(RuntimeError, "runtime identity is unavailable"):
            namespace["_runtime_collective"](args, cuda)
        self.assertNotIn("torch.cuda.nccl.version", path.read_text())

    def test_workloads_bind_generator_activation_and_trace(self) -> None:
        args = ("uniform", 7168, 8, 256, 8, 64, 67)
        first = workload.compute_workload_id(*args)
        self.assertTrue(identity.is_typed_id(first, "workload"))
        self.assertEqual(first, workload.compute_workload_id(*args))
        self.assertNotEqual(first, workload.compute_workload_id(*args[:-1], 68))
        self.assertNotEqual(
            first,
            workload.compute_workload_id(*args, trace_checksum="a" * 64),
        )
        _, _, manifest = workload.build_workload(8, 2, 4, "uniform", 4, 67, 2)
        member, checksums, _, _ = workload.canonical_member(
            "uniform", 8, 2, 4, 2, 2, 67
        )
        self.assertEqual(member, manifest["workload_id"])
        self.assertEqual(checksums, manifest["checksums"])

    def test_eplb_calibration_window_is_disjoint_and_identity_bound(self) -> None:
        evaluation = workload.canonical_member("zipf", 8, 2, 8, 2, 4, 67)
        calibration = workload.canonical_eplb_calibration_member(
            "zipf", 8, 2, 8, 2, 4, 67
        )
        self.assertNotEqual(evaluation[0], calibration[0])
        self.assertNotEqual(evaluation[1]["trace"], calibration[1]["trace"])
        self.assertGreater(
            workload.EPLB_CALIBRATION_TOKEN_OFFSET,
            2 * 4,
        )
        repeated = workload.canonical_eplb_calibration_member(
            "zipf", 8, 2, 8, 2, 4, 67
        )
        self.assertEqual(calibration, repeated)
        with self.assertRaises(ValueError):
            workload.canonical_routing_rows(
                8, 8, 2, "zipf", 67, token_offset=-1
            )

    def test_canonical_members_are_bound_to_each_scheduled_row(self) -> None:
        case = {
            "routing": "uniform", "hidden": 8, "topk": 2, "experts": 4, "ep": 2,
            "mode": "normal",
        }
        eplb_record = {
            "enabled": False, "mapping_hash": None, "num_physical_experts": 4,
        }

        def expected(
            *, tokens: int = 1, hidden: int = 8
        ) -> tuple[str, dict[str, str], str]:
            member, checksums, row_hash, _, _ = contracts._expected_canonical_trace(
                "uniform", hidden, 2, 4, 4, 2, tokens, 67, False, 2048
            )
            return member, checksums, row_hash

        member, checksums, row_hash = expected()
        rows = [{"tokens_per_rank": 1, "routing": {"hash": row_hash}}]
        proof = {
            "manifest_checksums": {member: checksums},
            "members": [member],
            "workload_id": identity.workload_id({
                "members": [{"checksums": checksums, "workload_id": member}]
            }),
        }
        contracts._validate_canonical_workload(proof, case, rows, eplb_record)

        def replace_member(document: dict, replacement: tuple[str, dict[str, str], str]) -> None:
            replacement_id, replacement_checksums, _ = replacement
            document["members"] = [replacement_id]
            document["manifest_checksums"] = {replacement_id: replacement_checksums}
            document["workload_id"] = identity.workload_id({
                "members": [{
                    "checksums": replacement_checksums,
                    "workload_id": replacement_id,
                }]
            })

        mutations = {
            "wrong member token": lambda document, mutated_rows: replace_member(
                document, expected(tokens=2)
            ),
            "wrong member dimensions": lambda document, mutated_rows: replace_member(
                document, expected(hidden=16)
            ),
            "wrong member checksum": lambda document, mutated_rows: replace_member(
                document,
                (
                    member,
                    {**checksums, "topk_idx": "0" * 64},
                    row_hash,
                ),
            ),
            "row hash unrelated to member": lambda document, mutated_rows: mutated_rows[0][
                "routing"
            ].update({"hash": "f" * 64}),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), self.assertRaises(contracts.ContractError):
                bad_proof, bad_rows = copy.deepcopy(proof), copy.deepcopy(rows)
                mutate(bad_proof, bad_rows)
                contracts._validate_canonical_workload(
                    bad_proof, case, bad_rows, eplb_record
                )

    def test_eplb_row_hash_is_bound_to_the_frozen_remap(self) -> None:
        case = {
            "routing": "zipf", "hidden": 8, "topk": 2, "experts": 4, "ep": 2,
            "mode": "normal",
        }
        physical = eplb.physical_count(4, 32, 2)
        plan = contracts._expected_eplb_plan("zipf", 2, 4, physical, 2, 67, 2048)
        eplb_record = {
            "enabled": True,
            "mapping_hash": eplb.mapping_hash(plan),
            "num_physical_experts": physical,
        }
        member, checksums, row_hash, _, _ = contracts._expected_canonical_trace(
            "zipf", 8, 2, 4, physical, 2, 1, 67, True, 2048
        )
        self.assertNotEqual(row_hash, checksums["trace"])
        workload_proof = {
            "manifest_checksums": {member: checksums},
            "members": [member],
            "workload_id": identity.workload_id({
                "members": [{"checksums": checksums, "workload_id": member}]
            }),
        }
        rows = [{"tokens_per_rank": 1, "routing": {"hash": row_hash}}]
        contracts._validate_canonical_workload(workload_proof, case, rows, eplb_record)
        with self.assertRaisesRegex(contracts.ContractError, "EPLB mapping"):
            contracts._validate_canonical_workload(
                workload_proof, case, rows, {**eplb_record, "mapping_hash": "0" * 64}
            )

    def test_oracle_pass_cannot_ignore_combined_value_failure(self) -> None:
        oracle = {
            "atol": ep_harness.ORACLE_ATOL,
            "checks": {
                "combine_values": True,
                "counts": True,
                "metadata": True,
                "multiplicity": True,
                "payload": True,
                "source_set": True,
                "weights": True,
            },
            "combine_weight_semantics": "unweighted-rank-sum",
            "contract": ep_harness.ORACLE_CONTRACT,
            "dispatch_sha256": "a" * 64,
            "max_absolute_error": 0.0,
            "max_elementwise_relative_error": 0.0,
            "max_relative_error": 0.0,
            "max_weight_error": 0.0,
            "order_sha256": "b" * 64,
            "ordering_contract": "stable-v1",
            "passed": True,
            "receive_count": 1,
            "rtol": ep_harness.ORACLE_RTOL,
        }
        contracts._validate_oracle(oracle, "oracle")
        weighted = copy.deepcopy(oracle)
        weighted["combine_weight_semantics"] = "native-gate-weighted"
        with self.assertRaisesRegex(contracts.ContractError, "differs from v1"):
            contracts._validate_oracle(weighted, "oracle")
        tampered = copy.deepcopy(oracle)
        tampered["checks"]["combine_values"] = False
        with self.assertRaises(contracts.ContractError):
            contracts._validate_oracle(tampered, "oracle")

        for profile_id, expected in (
            ("d-bf16.c-logfmt10-dynamic64", {"atol": 3e-2, "rtol": 6e-2}),
            ("d-bf16.c-fp8-e4m3fn-direct-cast-noscale", {"atol": 4e-2, "rtol": 8e-2}),
        ):
            with self.subTest(profile_id=profile_id):
                precision = identity.precision_profile(profile_id)
                codec_oracle = copy.deepcopy(oracle)
                codec_oracle.update(expected)
                contracts._validate_oracle(
                    codec_oracle, "oracle", communication_precision=precision
                )
                codec_oracle["rtol"] = 5e-2
                with self.assertRaisesRegex(contracts.ContractError, "rtol"):
                    contracts._validate_oracle(
                        codec_oracle, "oracle", communication_precision=precision
                    )

    def test_precision_evidence_rejects_direct_cast_saturation(self) -> None:
        profile_id = "d-bf16.c-fp8-e4m3fn-direct-cast-noscale"
        communication_precision = identity.precision_profile(profile_id)
        axis = {
            "dequantized_semantics": True,
            "encoded_payload_valid": True,
            "max_abs_error": 0.0,
            "max_rel_error": 0.0,
            "passed": True,
            "saturation_count": 0,
            "saturation_rate": 0.0,
            "scales_finite": None,
            "scales_positive": None,
        }
        evidence = {
            "combine": copy.deepcopy(axis),
            "dispatch": copy.deepcopy(axis),
            "passed": True,
            "profile_id": profile_id,
        }
        contracts._validate_precision_evidence(
            evidence, profile_id, communication_precision, "precision"
        )
        saturated = copy.deepcopy(evidence)
        saturated["combine"].update({"saturation_count": 1, "saturation_rate": 0.01})
        with self.assertRaisesRegex(contracts.ContractError, "passed differs"):
            contracts._validate_precision_evidence(
                saturated, profile_id, communication_precision, "precision"
            )
        saturated["combine"]["passed"] = False
        saturated["passed"] = False
        contracts._validate_precision_evidence(
            saturated, profile_id, communication_precision, "precision"
        )

        logfmt_profile_id = "d-bf16.c-logfmt10-dynamic64"
        logfmt_precision = identity.precision_profile(logfmt_profile_id)
        logfmt_evidence = copy.deepcopy(evidence)
        logfmt_evidence["profile_id"] = logfmt_profile_id
        contracts._validate_precision_evidence(
            logfmt_evidence, logfmt_profile_id, logfmt_precision, "precision"
        )
        invented_scales = copy.deepcopy(logfmt_evidence)
        invented_scales["combine"].update({
            "scales_finite": True,
            "scales_positive": True,
        })
        with self.assertRaisesRegex(contracts.ContractError, "must be null"):
            contracts._validate_precision_evidence(
                invented_scales, logfmt_profile_id, logfmt_precision, "precision"
            )

    def test_oracle_stability_canonicalizes_native_receive_order(self) -> None:
        source = (HERE / "ep_harness.py").read_text()
        begin = source.index("canonical_order = torch.argsort")
        canonical = source[begin:source.index("problem.recv_tokens = receive_count", begin)]
        self.assertIn("canonical_sources", canonical)
        self.assertIn("canonical_ids", canonical)
        self.assertIn("canonical_weights", canonical)
        self.assertNotIn("_tensor_sha256(source_ids", canonical)
        mori = (HERE / "ep_mori.py").read_text()
        self.assertIn('"inter-node-v1" if self._inter_node', mori)
        self.assertIn('else "async-ll" if self._async_ll', mori)
        backend = types.SimpleNamespace(name="mori", kernel_generation="async-ll")
        self.assertEqual(ep_harness.kernel_generation(backend), "async-ll")
        backend.kernel_generation = "inter-node-v1"
        self.assertEqual(ep_harness.kernel_generation(backend), "inter-node-v1")

    def test_terminal_fail_safe_fills_only_missing_shard_cases(self) -> None:
        matrix = sweep_matrix.resolve_matrix(backends="all", max_cases=128)
        shard = next(item for item in matrix["include"] if item["n"] >= 2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            matrix_path = root / "matrix.json"
            control_path = root / "control.json"
            out_dir = root / "results"
            matrix_path.write_text(json.dumps(matrix))
            control = sweep_matrix.extract_shard(
                matrix_path, shard["id"], control_path,
                sku=shard["sku"], backend=shard["backend"], nodes=shard["nodes"],
            )
            control["cases"] = control["cases"][:2]
            control["n"] = 2
            control_path.write_text(json.dumps(control))
            first = {key: value for key, value in control["cases"][0].items() if key != "case_id"}
            git_run = {
                "artifact": "artifact", "job": "job", "ref": "collectivex",
                "repo": "SemiAnalysisAI/InferenceX", "run_attempt": "1",
                "run_id": "123", "source_sha": "a" * 40,
                "qualification_index": 1,
            }
            allocation = {
                "artifact": "artifact", "execution_id": "execution", "job": "job",
                "repo": "SemiAnalysisAI/InferenceX", "run_attempt": "1", "run_id": "123",
                "runner": shard["sku"], "source_sha": "a" * 40,
                "qualification_index": 1,
            }
            out_dir.mkdir()
            existing = contracts.make_terminal_document(
                allocation_factors=allocation, attempt_ordinal=1, case=first,
                case_factors={
                    "case": first,
                    "profile": identity.profile_for_case(first),
                    "sku": shard["sku"],
                },
                control_sha256=hashlib.sha256(control_path.read_bytes()).hexdigest(),
                failure_mode="setup", generated_at="2026-07-04T00:00:00Z", git_run=git_run,
                reason="launcher-setup-failed", return_code=7, source="runtime-emitter",
                status="failed",
                expected_case_id=control["cases"][0]["case_id"],
            )
            (out_dir / "existing.json").write_text(json.dumps(existing))
            (out_dir / "partial.json").write_text(json.dumps({
                "format": contracts.RAW_FORMAT,
                "identity": {"case_id": control["cases"][1]["case_id"]},
                "sample_artifact": {"path": "partial.samples.json"},
            }))
            (out_dir / "partial.samples.json").write_text("{broken")
            environment = {
                **os.environ,
                "COLLECTIVEX_OPERATOR_CONFIG": "/dev/null",
                "CX_SHARD_FILE": str(control_path),
                "CX_SHARD_SKU": shard["sku"],
                "CX_RUNNER": shard["sku"],
                "CX_BENCH": shard["backend"],
                "CX_NODES": str(shard["nodes"]),
                "COLLECTIVEX_EXECUTION_ID": "execution",
                "COLLECTIVEX_ARTIFACT_NAME": "artifact",
                "GITHUB_JOB": "job", "GITHUB_REF_NAME": "collectivex",
                "GITHUB_REPOSITORY": "SemiAnalysisAI/InferenceX",
                "GITHUB_RUN_ATTEMPT": "1", "GITHUB_RUN_ID": "123",
                "GITHUB_SHA": "a" * 40,
                "CX_QUALIFICATION_INDEX": "1",
            }
            subprocess.run(
                [
                    "bash", "-c",
                    'source "$1"; cx_emit_setup_failures "$2" "$3" "$4" 7',
                    "_", str(ROOT / "runtime" / "common.sh"), str(ROOT),
                    str(out_dir), shard["backend"],
                ],
                check=True,
                env=environment,
            )
            attempts = [contracts.strict_load(path) for path in out_dir.glob("*.json")]
            self.assertEqual(len(attempts), 2)
            self.assertEqual(
                contracts.validate_attempt_paths([str(path) for path in out_dir.glob("*.json")]),
                2,
            )
            delivery = [str(path) for path in out_dir.glob("*.json")]
            self.assertEqual(contracts.validate_delivery(delivery, str(control_path)), 2)
            with self.assertRaises(contracts.ContractError):
                contracts.validate_delivery(delivery[:1], str(control_path))
            self.assertEqual(
                {attempt["identity"]["case_id"] for attempt in attempts},
                {case["case_id"] for case in control["cases"]},
            )
            self.assertTrue((out_dir / "partial.json.quarantine").is_file())
            self.assertTrue((out_dir / "partial.samples.json.quarantine").is_file())

            preallocation = root / "preallocation"
            preallocation_results = preallocation / "experimental" / "CollectiveX" / "results"
            preallocation_results.mkdir(parents=True)
            failed = subprocess.run(
                [
                    "bash", "-c",
                    'source "$1"; REPO_ROOT="$2"; export REPO_ROOT; '
                    'cx_install_launcher_fail_safe; cx_load_operator_config',
                    "_", str(ROOT / "runtime" / "common.sh"), str(preallocation),
                ],
                env={**environment, "COLLECTIVEX_OPERATOR_CONFIG_REQUIRED": "1"},
            )
            self.assertNotEqual(failed.returncode, 0)
            preallocation_attempts = [
                contracts.validate_terminal_document(contracts.strict_load(path))
                for path in preallocation_results.glob("*.json")
            ]
            self.assertEqual(
                {attempt["identity"]["case_id"] for attempt in preallocation_attempts},
                {case["case_id"] for case in control["cases"]},
            )

    def test_runtime_identity_mismatch_is_failed_not_unsupported(self) -> None:
        wrapper = next(
            item for item in sweep_matrix.resolve_matrix()["requested_cases"]
            if item["disposition"] == "runnable"
        )
        case = wrapper["case"]
        environment = {
            "CX_RUNNER": wrapper["sku"], "CX_CASE_ID": case["case_id"],
            "CX_SUITE": case["suite"], "CX_WORKLOAD_NAME": case["workload"],
            "CX_REQUIRED_PUBLICATION": case["required_publication"],
            "CX_ROUTING": case["routing"], "CX_EPLB": "1" if case["eplb"] else "",
            "CX_EP": str(case["ep"]), "CX_NGPUS": str(case["ep"]),
            "CX_HIDDEN": str(case["hidden"]), "CX_TOPK": str(case["topk"]),
            "CX_EXPERTS": str(case["experts"]), "CX_NODES": str(case["nodes"]),
            "CX_GPUS_PER_NODE": str(case["gpus_per_node"]),
            "CX_SCALE_UP_DOMAIN": str(case["scale_up_domain"]),
            "CX_MODE": case["mode"], "CX_SCOPE": case["scope"],
            "CX_TOPO": case["topology_class"], "CX_TRANSPORT": case["transport"],
            "CX_SCALE_UP_TRANSPORT": case["scale_up_transport"],
            "CX_SCALE_OUT_TRANSPORT": case["scale_out_transport"] or "",
            "CX_TOKENS_LADDER": case["ladder"], "CX_CANONICAL": "1",
            "CX_ITERS": "8", "CX_TRIALS": "64", "CX_WARMUP": "32",
            "CX_SAMPLES_PER_POINT": "512", "GITHUB_RUN_ID": "123",
            "GITHUB_RUN_ATTEMPT": "1", "GITHUB_REF_NAME": "collectivex",
            "GITHUB_SHA": "a" * 40, "GITHUB_REPOSITORY": "SemiAnalysisAI/InferenceX",
            "GITHUB_JOB": "sweep", "COLLECTIVEX_ARTIFACT_NAME": "artifact",
            "COLLECTIVEX_EXECUTION_ID": "execution",
        }
        with mock.patch.dict(os.environ, environment, clear=False):
            terminal = contracts.make_terminal_from_environment(
                backend=case["backend"], phase=case["phase"], return_code=5
            )
        self.assertEqual(terminal["identity"]["case_id"], case["case_id"])
        self.assertEqual(
            terminal["outcome"],
            {
                "failure_mode": "runtime-identity",
                "reason": "runtime-identity-mismatch",
                "return_code": 5,
                "status": "failed",
            },
        )
        for mode, reason in contracts.RUNTIME_FAILURE_REASONS.items():
            with self.subTest(mode=mode), mock.patch.dict(os.environ, environment, clear=False):
                staged = contracts.make_terminal_from_environment(
                    backend=case["backend"], phase=case["phase"], return_code=1,
                    failure_mode=mode,
                )
                self.assertEqual(staged["outcome"]["reason"], reason)
                mismatched = copy.deepcopy(staged)
                mismatched["outcome"]["reason"] = "distributed-command-failed"
                if reason == "distributed-command-failed":
                    mismatched["outcome"]["reason"] = "backend-setup-failed"
                with self.assertRaisesRegex(
                    contracts.ContractError, "source and outcome are not registered"
                ):
                    contracts.validate_terminal_document(mismatched)
        with mock.patch.dict(os.environ, environment, clear=False):
            with self.assertRaisesRegex(
                contracts.ContractError, "runtime failure mode is not registered"
            ) as raised:
                contracts.make_terminal_from_environment(
                    backend=case["backend"], phase=case["phase"], return_code=1,
                    failure_mode="raw-private-error",
                )
        self.assertNotIn("raw-private-error", str(raised.exception))
        with mock.patch.dict(os.environ, environment, clear=False):
            generic = contracts.make_terminal_from_environment(
                backend=case["backend"], phase=case["phase"], return_code=6,
            )
        self.assertEqual(
            generic["outcome"],
            {
                "failure_mode": "execution",
                "reason": "distributed-command-failed",
                "return_code": 6,
                "status": "failed",
            },
        )
        manual_environment = {
            "CX_RUNNER": "manual-runner",
            "COLLECTIVEX_EXECUTION_ID": "manual-execution",
        }
        with mock.patch.dict(os.environ, manual_environment, clear=True):
            manual = contracts.make_terminal_from_environment(
                backend="nccl-ep", phase="decode", return_code=6,
            )
        self.assertIsNone(manual["provenance"]["git_run"])
        self.assertEqual(
            {
                field: manual["case"][field]
                for field in ("suite", "workload", "canonical", "required_publication")
            },
            {
                "suite": "manual", "workload": "manual", "canonical": False,
                "required_publication": "diagnostic",
            },
        )
        self.assertEqual(
            manual["identity"]["allocation_factors"],
            {
                "artifact": None, "execution_id": "manual-execution", "job": None,
                "qualification_index": 1, "repo": None,
                "run_attempt": None, "run_id": None,
                "runner": "manual-runner", "source_sha": None,
            },
        )
        broken = copy.deepcopy(manual)
        broken["identity"]["allocation_factors"]["artifact"] = "forged-artifact"
        allocation_id = identity.allocation_id(
            broken["identity"]["allocation_factors"]
        )
        broken["identity"]["allocation_id"] = allocation_id
        broken["identity"]["attempt_id"] = identity.attempt_id(
            allocation=allocation_id,
            case=broken["identity"]["case_id"],
            ordinal=broken["identity"]["attempt_ordinal"],
        )
        with self.assertRaisesRegex(
            contracts.ContractError, "allocation factors differ"
        ):
            contracts.validate_terminal_document(broken)

    def test_launchers_use_private_logs_and_allowlisted_failure_stages(self) -> None:
        expected = {
            "launch_single-slurm.sh": {
                "setup", "registry-verification", "container-import", "container-hash",
                "repository-stage", "scheduler-allocation", "container-launch",
                "artifact-collection",
            },
            "launch_gb-nv.sh": {
                "setup", "registry-verification", "container-import", "container-hash",
                "repository-stage", "scheduler-allocation", "container-launch", "backend-setup",
                "execution", "artifact-collection",
            },
            "launch_mi-amds.sh": {
                "setup", "repository-stage", "registry-verification", "scheduler-allocation",
                "container-import", "container-hash", "container-launch", "artifact-collection",
            },
        }
        common = (ROOT / "runtime" / "common.sh").read_text()
        for name, stages in expected.items():
            launcher = (ROOT / "launchers" / name).read_text()
            stage_source = launcher + common if name == "launch_gb-nv.sh" else launcher
            self.assertNotIn("--export=ALL", launcher)
            if name == "launch_gb-nv.sh":
                self.assertIn("cx_run_distributed_shard", launcher)
            else:
                self.assertIn("cx_container_exports", launcher)
            self.assertIn("collect_rc=0", launcher)
            for stage in stages:
                with self.subTest(launcher=name, stage=stage):
                    self.assertIn(f"cx_set_failure_stage {stage}", stage_source)
        amd = (ROOT / "launchers" / "launch_mi-amds.sh").read_text()
        self.assertIn("cx_ensure_squash_on_job", amd)
        self.assertIn("for allocation_attempt in 1 2 3", amd)
        self.assertIn("allocated nodes failed container import; retrying elsewhere", amd)
        self.assertIn('rejected_nodes="$(cx_allocation_nodes_csv "$JOB_ID")"', amd)
        self.assertIn("cx_fail_stage container-hash", amd)
        self.assertNotIn('cat "$import_log"', amd)
        self.assertIn('bash -s -- "$sq" "$lock" "$image"', common)
        self.assertIn("> \"$log\" 2>&1 <<'BASH'", common)
        self.assertIn("cx_fail_stage container-import", common)
        run_ep = (ROOT / "tests" / "run_ep.py").read_text()
        self.assertIn("def _all_gather_json(", run_ep)
        self.assertIn("dist_module.all_gather(gathered, encoded)", run_ep)
        self.assertNotIn("dist.all_gather_object", run_ep)
        self.assertIn('args.backend == "deepep-v2"', run_ep)
        self.assertNotIn('args.backend in {"deepep-v2", "nccl-ep"}', run_ep)
        self.assertIn('dist.init_process_group("nccl", device_id=device)', run_ep)
        nccl = (ROOT / "tests" / "ep_nccl.py").read_text()
        self.assertIn('if _library == "nccl" and network_selection != "IB"', nccl)
        self.assertIn('if _library == "rccl" and network_selection:', nccl)
        runtime = (ROOT / "runtime" / "run_in_container.sh").read_text()
        export_start = common.index("\ncx_container_exports() {")
        exports = common[export_start:common.index("\n}", export_start)]
        export_names = {
            name
            for payload in re.findall(r"printf '%s' '([^']*)'", exports)
            for name in payload.split(",") if name
        }
        for private_name in (
            "COLLECTIVEX_OPERATOR_CONFIG", "GITHUB_TOKEN", "GITHUB_WORKSPACE", "HOME",
            "CX_PARTITION", "CX_ACCOUNT", "CX_SQUASH_DIR", "CX_STAGE_DIR",
        ):
            self.assertNotIn(private_name, export_names)
        self.assertIn("CX_BACKEND_CACHE_ROOT", export_names)
        self.assertIn("CX_BACKEND_CACHE_SENTINEL_SHA256", export_names)
        self.assertNotIn("CX_PREPARED_BACKEND_CACHE", export_names)
        self.assertIn("MORI_COMMIT", export_names)
        self.assertIn("cx_write_runtime_stage backend-setup", runtime)
        self.assertIn("cx_write_runtime_stage execution", runtime)
        distributed = common[common.index("cx_run_distributed_shard()") :]
        self.assertIn("cx_private_log_path shard-summary", distributed)
        self.assertIn("cx_fail_stage execution", distributed)
        self.assertIn('cx_fail_stage execution "$runtime_log"', distributed)
        self.assertIn("precision probe timed out rc=%s limit=%ss", distributed)
        self.assertIn('export CX_MODE="$mode" CX_PHASE="$ph"', distributed)
        rank_wrapper = common[
            common.index("cx_slurm_rank_wrapper()") : common.index(
                "cx_validate_shard_control()"
            )
        ]
        self.assertLess(
            rank_wrapper.index(". /ix/experimental/CollectiveX/runtime/common.sh"),
            rank_wrapper.index('if [ "${CX_NODES:-1}" -gt 1 ]'),
        )
        self.assertEqual(
            distributed.count(
                '--container-name="$container_name" --container-image="$SQUASH_FILE"'
            ),
            5,
        )
        shard_runtime = runtime[runtime.index('elif [ -n "${CX_SHARD_FILE:-}" ]') :]
        self.assertIn('"CX_PRECISION_PROFILE": g("precision_profile")', shard_runtime)
        self.assertIn("rdma-port-%s=inactive", common)
        self.assertIn("rdma-device-%s=missing", common)
        single_slurm = (ROOT / "launchers" / "launch_single-slurm.sh").read_text()
        self.assertIn("for allocation_attempt in 1 2 3", single_slurm)
        self.assertIn('RUNNER:$validation_failure', single_slurm)
        self.assertIn('h100-dgxc:network', single_slurm)
        self.assertIn('b300:cuda-context', single_slurm)
        self.assertIn('cx_validate_cuda_context_on_job "$JOB_ID" "$NODES" "$GPN"', single_slurm)
        self.assertIn('rejected_nodes="$(cx_allocation_nodes_csv "$JOB_ID")"', single_slurm)
        self.assertIn('export CX_SALLOC_ATTEMPT="$allocation_attempt"', single_slurm)
        self.assertIn('export CX_NETWORK_VALIDATION_ATTEMPT="$allocation_attempt"', single_slurm)
        self.assertIn('test -c /dev/gdrdrv', single_slurm)
        self.assertIn('/dev/gdrdrv:/dev/gdrdrv', single_slurm)
        self.assertIn(
            '[ "$RUNNER" = b200-dgxc ] && [ "$CX_BENCH" = deepep ] && [ "$NODES" -gt 1 ]',
            single_slurm,
        )
        self.assertIn('log_label+="-a${CX_SALLOC_ATTEMPT}"', common)
        self.assertIn('log_label+="-a${CX_NETWORK_VALIDATION_ATTEMPT}"', common)
        self.assertIn('cx_validate_cuda_context_on_job()', common)
        self.assertIn('cuDevicePrimaryCtxRetain', common)
        self.assertIn('diagnostic="accelerator-unavailable"', common)

    def test_case_failure_diagnostic_precedes_normal_srun_footer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "runtime.log"
            log.write_text(
                "WARN: deepep decode run failed rc=1 (CX_RUN_TIMEOUT=900s)\n"
                "SHARD done: 6/6 case(s) failed\n"
                "srun: error: task exited 1\n"
            )
            result = subprocess.run(
                [
                    "bash", "-c",
                    'source "$1"; cx_fail_stage execution "$2"',
                    "_", str(ROOT / "runtime" / "common.sh"), str(log),
                ],
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("diagnostic=benchmark-case-failure", result.stderr)

    def test_non_timeout_failure_warning_is_classified_as_case_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "runtime.log"
            log.write_text("WARN: deepep decode run failed rc=1\nsrun: task exited 1\n")
            result = subprocess.run(
                [
                    "bash", "-c",
                    'source "$1"; cx_fail_stage execution "$2"',
                    "_", str(ROOT / "runtime" / "common.sh"), str(log),
                ],
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 1)
            self.assertNotIn("diagnostic=network-or-timeout", result.stderr)
            self.assertIn("diagnostic=benchmark-case-failure", result.stderr)

    def test_backend_import_failure_precedes_container_cleanup_noise(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "runtime.log"
            log.write_text(
                "ERROR: MoRI backend import failed\n"
                "pyxis: failed to mount container cleanup path: invalid argument\n"
            )
            result = subprocess.run(
                [
                    "bash", "-c",
                    'source "$1"; cx_fail_stage backend-setup "$2"',
                    "_", str(ROOT / "runtime" / "common.sh"), str(log),
                ],
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("diagnostic=backend-build", result.stderr)
            self.assertNotIn("diagnostic=container-runtime", result.stderr)

    def test_all_ranks_validate_raw_evidence_before_rank_zero_writes(self) -> None:
        harness = (ROOT / "tests" / "ep_harness.py").read_text()
        validation = harness.index("contracts.validate_raw_document(doc, samples_document)")
        rank_zero_write = harness.index("if rank == 0:", validation)
        self.assertLess(validation, rank_zero_write)
        self.assertNotIn("if rank != 0:", harness[harness.index("def run_sweep("):validation])

    def test_precision_probe_timeout_reports_only_the_last_closed_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "runtime.log"
            log.write_text(
                "[collectivex] precision-probe-stage=distributed-init\n"
                "[collectivex] precision-probe-stage=backend-construction\n"
                "[collectivex] precision-probe-stage=native-operation\n"
                "[collectivex] precision probe timed out rc=124 limit=900s\n"
            )
            result = subprocess.run(
                [
                    "bash", "-c",
                    'source "$1"; cx_fail_stage execution "$2"',
                    "_", str(ROOT / "runtime" / "common.sh"), str(log),
                ],
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("diagnostic=native-operation-timeout", result.stderr)
            self.assertNotIn("distributed-init", result.stderr)

    def test_precision_probe_memory_failure_reports_the_last_closed_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "runtime.log"
            log.write_text(
                "[collectivex] precision-probe-stage=backend-construction\n"
                "CUDA out of memory\n"
            )
            result = subprocess.run(
                [
                    "bash", "-c",
                    'source "$1"; cx_fail_stage execution "$2"',
                    "_", str(ROOT / "runtime" / "common.sh"), str(log),
                ],
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn(
                "diagnostic=backend-construction-accelerator-memory", result.stderr
            )

    def test_precision_probe_unknown_failure_reports_only_the_last_closed_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "runtime.log"
            log.write_text(
                "[collectivex] precision-probe-stage=native-operation\n"
                "private backend failure details\n"
            )
            result = subprocess.run(
                [
                    "bash", "-c",
                    'source "$1"; cx_fail_stage execution "$2"',
                    "_", str(ROOT / "runtime" / "common.sh"), str(log),
                ],
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("diagnostic=native-operation-failed", result.stderr)
            self.assertNotIn("private backend", result.stderr)

    def test_precision_probe_traceback_prefers_the_last_closed_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "runtime.log"
            log.write_text(
                "[collectivex] precision-probe-stage=native-operation\n"
                "Traceback (most recent call last):\nRuntimeError: private details\n"
            )
            result = subprocess.run(
                [
                    "bash", "-c",
                    'source "$1"; cx_fail_stage execution "$2"',
                    "_", str(ROOT / "runtime" / "common.sh"), str(log),
                ],
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("diagnostic=native-operation-failed", result.stderr)
            self.assertNotIn("python-exception", result.stderr)

    def test_private_runtime_failure_signatures_override_case_footer(self) -> None:
        signatures = {
            "DeepEP V2 no-GIN run is outside one realized LSA domain":
                "accelerator-topology",
            "NCCL exception (/src/nccl.cu:111): 3": "accelerator-topology",
            "NCCL exception (/src/nccl.cu:112): 3": "accelerator-topology",
            "CUDA error: call requires newer driver": "accelerator-driver",
            "NCCL failure in ncclCommWindowRegister": "nccl-device-api",
            "Communicator does not support symmetric memory": "nccl-device-api",
            "NCCL exception (/src/nccl.cu:106): 5": "nccl-device-api",
            "NCCL exception (/src/nccl.cu:127): 5": "nccl-device-api",
            "NCCL exception (/src/nccl.cu:128): 5": "nccl-device-api",
            "NCCL exception (/src/nccl.cu:129): 5": "nccl-device-api",
            "NCCL exception (/src/nccl.cu:135): 5": "nccl-device-api",
            "NVCC compilation failed": "jit-toolchain",
            "CUDA out of memory": "accelerator-memory",
            "torch rendezvous timed out": "network-or-timeout",
            "torch.distributed.DistBackendError: ncclRemoteError": "collective-remote",
            "torch.distributed.DistBackendError: ncclSystemError": "collective-system",
            "torch.distributed.DistBackendError: ncclInternalError": "collective-internal",
            "torch.distributed.DistBackendError: ncclInvalidUsage":
                "collective-invalid-usage",
            "dist.init_process_group('nccl')\nDistBackendError: ncclInvalidUsage":
                "collective-init-invalid-usage",
            "dist.all_gather_object(records, local)\nDistBackendError: ncclInvalidUsage":
                "collective-consensus-invalid-usage",
            "dist.all_to_all_single(output, input)\nDistBackendError: ncclInvalidUsage":
                "collective-alltoall-invalid-usage",
            "torch.distributed.DistBackendError: unknown collective failure":
                "collective-backend",
            "ModuleNotFoundError: missing module": "python-import",
            "AttributeError: backend has no attribute 'probe'": "backend-api",
            "PrecisionError: FP8 dispatch payload is missing block-128 scales":
                "precision-dispatch-scales-missing",
            "PrecisionError: native FP8 dispatch payload has an invalid dtype or shape":
                "precision-dispatch-payload-shape",
            "PrecisionError: native FP8 dispatch scales have an invalid dtype or shape":
                "precision-dispatch-scale-shape",
            "PrecisionError: expert-packed FP8 receive count exceeds capacity":
                "precision-receive-capacity",
            "PrecisionError: expert-packed FP8 receive counts has an invalid shape":
                "precision-receive-shape",
            "PrecisionError: active torch build does not expose torch.float8_e4m3fnuz":
                "precision-runtime-dtype",
            "PrecisionError: pinned MoRI API omits EpDispatchCombineQuantType.Fp8DirectCast":
                "precision-combine-api",
            "PrecisionError: MoRI dispatch FP8 format differs from the pinned GPU architecture":
                "precision-architecture-format",
            "PrecisionError: MoRI native FP8 dispatch requires hidden divisible by 128":
                "precision-hidden-alignment",
            "PrecisionError: unsupported precision profile": "precision-contract",
            "AssertionError: probe invariant": "python-assertion",
            "RuntimeError: probe execution failed": "python-runtime",
            "ValueError: probe fields differ from collectivex.precision-probe.v1":
                "probe-schema-value",
            "ValueError: probe target is not a provisional native adapter cell":
                "probe-target-value",
            "ValueError: probe result reason is invalid": "probe-result-value",
            "ValueError: probe privacy contract differs": "probe-privacy-value",
            "ValueError: probe API calls are empty": "probe-api-value",
            "ValueError: probe completion contract differs": "probe-completion-value",
            "ValueError: probe image digest is invalid": "probe-identity-value",
            "ValueError: probe precision correctness did not pass":
                "probe-correctness-failed",
            "ValueError: probe scale shapes are invalid": "probe-scale-shape",
            "ValueError: probe precision dispatch input is not finite": "probe-nonfinite",
            "ValueError: probe precision combine output shape is invalid":
                "probe-tensor-shape",
            "ValueError: probe transport fallback is present": "probe-transport-value",
            "tests/probe_precision.py\nValueError: probe argument": "probe-manifest-value",
            "tests/ep_harness.py\nValueError: harness argument": "harness-value",
            "tests/workload.py\nValueError: workload argument": "workload-value",
            "tests/run_ep.py\nValueError: runner argument": "runner-value",
            "tests/ep_deepep.py\nValueError: adapter argument": "deepep-adapter-value",
            "site-packages/torch/module.py\nValueError: dependency argument": "dependency-value",
            "ValueError: probe argument": "python-value",
            "KeyError: 'probe-key'": "python-key",
            "OSError: probe path": "python-os",
            "NotImplementedError: probe API": "python-system",
            "CalledProcessError: probe command": "python-subprocess",
            "Traceback (most recent call last):": "python-exception",
        }
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "runtime.log"
            for signature, diagnostic in signatures.items():
                log.write_text(f"{signature}\nSHARD done: 6/6 case(s) failed\n")
                result = subprocess.run(
                    [
                        "bash", "-c",
                        'source "$1"; cx_fail_stage execution "$2"',
                        "_", str(ROOT / "runtime" / "common.sh"), str(log),
                    ],
                    text=True,
                    capture_output=True,
                    env={**os.environ, "CX_BENCH": "deepep-v2"},
                )
                self.assertEqual(result.returncode, 1)
                self.assertIn(f"diagnostic={diagnostic}", result.stderr)

            log.write_text(
                "NCCL exception (/src/nccl.cu:106): 5\n"
                "SHARD done: 6/6 case(s) failed\n"
            )
            result = subprocess.run(
                [
                    "bash", "-c", 'source "$1"; cx_fail_stage execution "$2"',
                    "_", str(ROOT / "runtime" / "common.sh"), str(log),
                ],
                text=True, capture_output=True,
                env={**os.environ, "CX_BENCH": "deepep"},
            )
            self.assertIn("diagnostic=benchmark-case-failure", result.stderr)

    def test_runtime_stage_marker_distinguishes_launch_from_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            mount = Path(temporary)
            root = mount / "experimental" / "CollectiveX"
            root.mkdir(parents=True)
            command = r'''
              set -euo pipefail
              source "$1"
              export COLLECTIVEX_EXECUTION_ID=test_1_shard CX_TS=test
              cx_set_failure_stage container-launch
              cx_prepare_runtime_marker "$2"
              (cd "$2/experimental/CollectiveX"; cx_write_runtime_stage backend-setup)
              cx_adopt_runtime_stage "$2"
              test "$CX_FAILSAFE_MODE" = backend-setup
              (cd "$2/experimental/CollectiveX"; cx_write_runtime_stage execution)
              cx_adopt_runtime_stage "$2"
              test "$CX_FAILSAFE_MODE" = execution
            '''
            subprocess.run(
                ["bash", "-c", command, "_", str(ROOT / "runtime" / "common.sh"),
                 str(mount)],
                check=True,
            )

    def test_canonical_gha_environment_is_locked_but_manual_overrides_survive(self) -> None:
        common = ROOT / "runtime" / "common.sh"
        command = r'''
          set -euo pipefail
          source "$1"
          export COLLECTIVEX_CANONICAL_GHA=1 GITHUB_ACTIONS=true
          export GITHUB_RUN_ID=123 GITHUB_RUN_ATTEMPT=1
          export COLLECTIVEX_SOURCE_SHA=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
          export CX_SHARD_FILE=.shards/test.json CX_SHARD_SKU=mi325x
          export CX_NODES=1 CX_GPUS_PER_NODE=8
          export CX_IMAGE=untrusted CX_IMAGE_DIGEST=untrusted CX_NGPUS=99
          export CX_NCCL_HOME=/untrusted CX_LOCK_DIR=/tmp CX_SQUASH_DIR=/shared/containers
          export CX_STAGE_DIR=/private/stale-stage
          export CX_MORI_KERNEL_TYPE=intranode MORI_ENABLE_SDMA=0
          export NCCL_MNNVL_ENABLE=1 MC_FORCE_MNNVL=1 CX_DRYRUN=1
          export CX_BACKEND_CACHE_ROOT=/untrusted CX_BACKEND_CACHE_SENTINEL_SHA256=bad
          export CX_PREPARED_BACKEND_CACHE=/untrusted CX_BACKEND_SOURCE_ROOT=/untrusted
          ! (cx_lock_canonical_gha_env mi325x)
          export COLLECTIVEX_OPERATOR_CONFIG_LOADED=$$
          export CX_STAGE_DIR="$GITHUB_WORKSPACE" CX_AUDIT_SALT="$(printf 'a%.0s' {1..64})"
          unset CX_LOCK_DIR
          cx_lock_canonical_gha_env mi325x
          test "$CX_IMAGE" = "$CX_IMAGE_AMD_MORI_MI325"
          test "$CX_IMAGE_DIGEST" = "$CX_IMAGE_AMD_MORI_MI325_DIGEST"
          test "$CX_NGPUS:$CX_SEED:$CX_RUN_TIMEOUT" = 8:67:1800
          test "$CX_MORI_KERNEL_TYPE:$MORI_DISABLE_AUTO_XGMI:$MORI_ENABLE_SDMA" = asyncll:0:1
          test "$MORI_COMMIT" = "$CX_MORI_COMMIT_MI325"
          test "$MORI_APP_LOG_LEVEL:$MORI_SHMEM_LOG_LEVEL:$MORI_IO_LOG_LEVEL" = info:info:info
          test "$CX_STAGE_DIR" = "$GITHUB_WORKSPACE"
          test -z "${CX_NCCL_HOME+x}${CX_LOCK_DIR+x}${NCCL_MNNVL_ENABLE+x}${MC_FORCE_MNNVL+x}"
          test -z "${CX_BACKEND_CACHE_ROOT+x}${CX_BACKEND_CACHE_SENTINEL_SHA256+x}"
          test -z "${CX_PREPARED_BACKEND_CACHE+x}${CX_BACKEND_SOURCE_ROOT+x}"
          test -z "${CX_DRYRUN+x}"

          export COLLECTIVEX_OPERATOR_CONFIG_LOADED=$$
          export CX_SHARD_SKU=mi300x CX_NODES=2 CX_GPUS_PER_NODE=8
          export CX_STAGE_DIR="$TEST_STAGE_ALIAS" CX_AUDIT_SALT="$(printf 'a%.0s' {1..64})"
          cx_lock_canonical_gha_env mi300x
          test "$CX_STAGE_DIR" = "$TEST_STAGE_TARGET"

          cx_prepare_implicit_stage_base() {
            if [ -n "${1:-}" ]; then test "$1" = "$TEST_SHARED_PARENT"; fi
            if [ -n "${2:-}" ]; then test "$2" = 123; fi
            printf '%s' "$TEST_IMPLICIT_STAGE"
          }
          export COLLECTIVEX_OPERATOR_CONFIG_LOADED=$$
          export CX_SHARD_SKU=h100-dgxc CX_NODES=1 CX_GPUS_PER_NODE=8
          export CX_SQUASH_DIR="$TEST_SHARED_PARENT/containers"
          unset CX_STAGE_DIR
          cx_lock_canonical_gha_env h100-dgxc
          test "$CX_STAGE_DIR" = "$TEST_IMPLICIT_STAGE"

          export COLLECTIVEX_OPERATOR_CONFIG_LOADED=$$
          export CX_SHARD_SKU=b200-dgxc CX_NODES=1 CX_GPUS_PER_NODE=8
          unset CX_STAGE_DIR
          cx_lock_canonical_gha_env b200-dgxc
          test "$CX_STAGE_DIR" = "$TEST_IMPLICIT_STAGE"

          export COLLECTIVEX_OPERATOR_CONFIG_LOADED=$$
          export CX_SHARD_SKU=b300 CX_NODES=1 CX_GPUS_PER_NODE=8
          export CX_STAGE_DIR=/legacy/root-owned-stage
          cx_lock_canonical_gha_env b300
          test "$CX_STAGE_DIR" = "$TEST_IMPLICIT_STAGE"
          test "$CX_STAGE_PARENT_OWNER_OK" = 1
          test -z "${NVSHMEM_DISABLE_IB+x}"

          export COLLECTIVEX_OPERATOR_CONFIG_LOADED=$$
          export CX_STAGE_DIR=/legacy/group-writable-stage
          export CX_SHARD_SKU=gb300 CX_NODES=2 CX_GPUS_PER_NODE=4
          export CX_IMAGE=untrusted CX_NGPUS=1 CX_MORI_KERNEL_TYPE=untrusted
          export MORI_ENABLE_SDMA=0 CX_NCCL_HOME=/untrusted CX_MASTER_PORT=1
          cx_lock_canonical_gha_env gb300
          test "$CX_IMAGE" = "$CX_IMAGE_MULTIARCH"
          test "$CX_IMAGE_DIGEST" = "$CX_IMAGE_MULTIARCH_DIGEST"
          test "$CX_NGPUS:$CX_SEED:$CX_RUN_TIMEOUT" = 8:67:900
          test "$CX_NCCL_HOME:$CX_MASTER_PORT" = /usr:29551
          test "$CX_STAGE_DIR" = "$TEST_IMPLICIT_STAGE"
          test -z "${CX_STAGE_PARENT_OWNER_OK+x}"
          test -z "${NVSHMEM_DISABLE_IB+x}"
          test -z "${CX_MORI_KERNEL_TYPE+x}${MORI_ENABLE_SDMA+x}"

          export COLLECTIVEX_OPERATOR_CONFIG_LOADED=$$
          export CX_SHARD_SKU=mi355x CX_NODES=1 CX_GPUS_PER_NODE=8
          export CX_LOCK_DIR=/validated/amd-locks CX_STAGE_DIR=/validated/amd-stage
          export CX_AUDIT_SALT="$(printf 'a%.0s' {1..64})"
          cx_lock_canonical_gha_env mi355x
          test "$CX_LOCK_DIR" = /validated/amd-locks
          test "$CX_STAGE_DIR" = /validated/amd-stage
          test "$MORI_COMMIT" = "$CX_MORI_COMMIT_MI355"

          export COLLECTIVEX_OPERATOR_CONFIG_LOADED=$$
          export CX_SHARD_SKU=mi355x CX_NODES=1 CX_GPUS_PER_NODE=8
          export CX_SQUASH_DIR=/validated/amd-shared
          export RUNNER_TEMP="$TEST_SHARED_PARENT/_work/_temp"
          export CX_AUDIT_SALT="$(printf 'a%.0s' {1..64})"
          unset CX_STAGE_DIR
          cx_lock_canonical_gha_env mi355x
          test "$CX_STAGE_DIR" = "$TEST_IMPLICIT_STAGE"

          unset COLLECTIVEX_CANONICAL_GHA
          unset COLLECTIVEX_OPERATOR_CONFIG_LOADED
          CX_IMAGE=manual CX_IMAGE_DIGEST=manual CX_NGPUS=3
          CX_MORI_KERNEL_TYPE=manual
          cx_lock_canonical_gha_env mi355x
          test "$CX_IMAGE:$CX_IMAGE_DIGEST:$CX_NGPUS:$CX_MORI_KERNEL_TYPE" = manual:manual:3:manual
        '''
        with tempfile.TemporaryDirectory(dir=Path.home()) as temporary:
            root = Path(temporary)
            home = root / "home"
            workspace = root / "workspace"
            implicit_stage = root / "implicit-stage"
            stage_target = root / "stage-target"
            stage_alias = root / "stage-alias"
            home.mkdir(mode=0o700)
            workspace.mkdir(mode=0o720)
            implicit_stage.mkdir(mode=0o700)
            stage_target.mkdir(mode=0o700)
            stage_alias.symlink_to(stage_target, target_is_directory=True)
            subprocess.run(
                ["bash", "-c", command, "_", str(common)],
                check=True,
                env={
                    **os.environ,
                    "HOME": str(home),
                    "TEST_IMPLICIT_STAGE": str(implicit_stage),
                    "TEST_STAGE_ALIAS": str(stage_alias),
                    "TEST_STAGE_TARGET": str(stage_target),
                    "TEST_SHARED_PARENT": str(root),
                    "COLLECTIVEX_OPERATOR_CONFIG": "/dev/null",
                    "GITHUB_WORKSPACE": str(workspace),
                },
            )
            self.assertEqual(list(workspace.iterdir()), [])

    def test_slurm_rendezvous_uses_relative_node_zero_and_validated_interface(self) -> None:
        common = ROOT / "runtime" / "common.sh"
        command = r'''
          set -euo pipefail
          source "$1"
          cx_host_exports() { printf '%s' HOME,PATH,USER; }
          squeue() { return 91; }
          scontrol() { return 92; }
          srun() {
            case "$*" in
              *" bash -s -- eth0") printf '%s\n' 192.0.2.7 ;;
              *" hostname -s") printf '%s\n' rank-zero-node ;;
              *) return 93 ;;
            esac
          }
          CX_SOCKET_IFNAME=eth0
          cx_resolve_slurm_rendezvous 123
          test "$MASTER_ADDR:$MASTER_PORT" = 192.0.2.7:29551
          unset CX_SOCKET_IFNAME
          CX_MASTER_PORT=30444
          cx_resolve_slurm_rendezvous 123
          test "$MASTER_ADDR:$MASTER_PORT" = rank-zero-node:30444
        '''
        subprocess.run(
            ["bash", "-c", command, "_", str(common)],
            check=True,
        )

    def test_implicit_stage_base_is_private_and_non_symlinked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            safe_home = root / "safe-home"
            unsafe_home = root / "unsafe-home"
            linked_home = root / "linked-home"
            linked_child_home = root / "linked-child-home"
            safe_home.mkdir(mode=0o700)
            unsafe_home.mkdir(mode=0o770)
            unsafe_home.chmod(0o770)
            linked_home.symlink_to(safe_home, target_is_directory=True)
            linked_child_home.mkdir(mode=0o700)
            (linked_child_home / ".inferencex-collectivex-stage").symlink_to(
                safe_home, target_is_directory=True
            )
            command = r'''
              set -euo pipefail
              source "$1"
              base="$(cx_prepare_implicit_stage_base "$2")"
              test "$base" = "$2/.inferencex-collectivex-stage"
              test "$(stat -c '%a' "$base" 2>/dev/null || stat -f '%Lp' "$base")" = 700
              ! cx_prepare_implicit_stage_base "$3"
              test "$(cx_prepare_implicit_stage_base "$4")" = "$2/.inferencex-collectivex-stage"
              isolated="$(cx_prepare_implicit_stage_base "$2" runner-a)"
              test "$isolated" != "$2/.inferencex-collectivex-stage"
              case "$isolated" in "$2"/.inferencex-collectivex-stage-*) ;; *) exit 1 ;; esac
              ! cx_prepare_implicit_stage_base "$5"
            '''
            subprocess.run(
                [
                    "bash", "-c", command, "_",
                    str(ROOT / "runtime" / "common.sh"), str(safe_home),
                    str(unsafe_home), str(linked_home), str(linked_child_home),
                ],
                check=True,
                env={**os.environ, "COLLECTIVEX_OPERATOR_CONFIG": "/dev/null"},
            )

    def test_canonical_amd_stage_uses_config_not_world_writable_workspace(self) -> None:
        common = ROOT / "runtime" / "common.sh"
        command = r'''
          source "$1"
          export COLLECTIVEX_CANONICAL_GHA=1 GITHUB_ACTIONS=true
          export GITHUB_RUN_ID=123 GITHUB_RUN_ATTEMPT=1
          export COLLECTIVEX_SOURCE_SHA=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
          export CX_SHARD_FILE=.shards/test.json CX_SHARD_SKU=mi325x
          export CX_NODES=1 CX_GPUS_PER_NODE=8 CX_SQUASH_DIR=/shared/containers
          export COLLECTIVEX_OPERATOR_CONFIG_LOADED=$$ CX_STAGE_DIR=/shared/amd-stage
          export CX_AUDIT_SALT="$(printf 'a%.0s' {1..64})"
          cx_lock_canonical_gha_env mi325x
          printf '%s' "$CX_STAGE_DIR"
        '''
        with tempfile.TemporaryDirectory(dir=Path.home()) as workspace:
            Path(workspace).chmod(0o702)
            result = subprocess.run(
                ["bash", "-c", command, "_", str(common)],
                text=True,
                capture_output=True,
                env={
                    **os.environ,
                    "COLLECTIVEX_OPERATOR_CONFIG": "/dev/null",
                    "GITHUB_WORKSPACE": workspace,
                },
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "/shared/amd-stage")
            self.assertNotIn(workspace, result.stderr)

    def test_canonical_amd_stage_uses_config_not_symlinked_workspace(self) -> None:
        common = ROOT / "runtime" / "common.sh"
        command = r'''
          source "$1"
          export COLLECTIVEX_CANONICAL_GHA=1 GITHUB_ACTIONS=true
          export GITHUB_RUN_ID=123 GITHUB_RUN_ATTEMPT=1
          export COLLECTIVEX_SOURCE_SHA=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
          export CX_SHARD_FILE=.shards/test.json CX_SHARD_SKU=mi325x
          export CX_NODES=1 CX_GPUS_PER_NODE=8 CX_SQUASH_DIR=/shared/containers
          export COLLECTIVEX_OPERATOR_CONFIG_LOADED=$$ CX_STAGE_DIR=/shared/amd-stage
          export CX_AUDIT_SALT="$(printf 'a%.0s' {1..64})"
          cx_lock_canonical_gha_env mi325x
          printf '%s' "$CX_STAGE_DIR"
        '''
        with tempfile.TemporaryDirectory(dir=Path.home()) as temporary:
            root = Path(temporary)
            real = root / "real"
            real.mkdir()
            link = root / "workspace"
            link.symlink_to(real, target_is_directory=True)
            result = subprocess.run(
                ["bash", "-c", command, "_", str(common)],
                text=True,
                capture_output=True,
                env={
                    **os.environ,
                    "COLLECTIVEX_OPERATOR_CONFIG": "/dev/null",
                    "GITHUB_WORKSPACE": str(link),
                },
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "/shared/amd-stage")
            self.assertNotIn(str(root), result.stderr)

    def test_image_selection_and_registry_verification_are_fail_closed(self) -> None:
        common = ROOT / "runtime" / "common.sh"
        command = r'''
          source "$1"
          test "$(cx_default_image mi325x)" = "$CX_IMAGE_AMD_MORI_MI325"
          test "$(cx_default_image mi355x)" = "$CX_IMAGE_AMD_MORI"
          pinned="sha256:$(printf 'a%.0s' {1..64})"
          curl() {
            case "$*" in
              *auth.docker.io*) printf '{"token":"test"}' ;;
              *) printf 'Docker-Content-Digest: %s\r\n' "$pinned" ;;
            esac
          }
          test "$(cx_resolve_registry_digest ubuntu:latest)" = "$pinned"
          test "$(cx_resolve_registry_digest docker.io/library/ubuntu:latest)" = "$pinned"
          ! (cx_resolve_registry_digest "ubuntu@$pinned")
          ! (cx_resolve_registry_digest ghcr.io/example/image:tag)
          ! (cx_resolve_registry_digest 'ubuntu@sha256:bad')
          curl() {
            case "$*" in *auth.docker.io*) printf '{"token":"test"}';; esac
          }
          ! (cx_resolve_registry_digest ubuntu:latest)
          cx_resolve_registry_digest() { printf '%s' "$CX_IMAGE_MULTIARCH_DIGEST"; }
          cx_verify_registry_image "$CX_IMAGE_MULTIARCH"
          test "$COLLECTIVEX_IMAGE_DIGEST_VERIFIED" = 1
          test "$COLLECTIVEX_IMAGE_DIGEST" = "$CX_IMAGE_MULTIARCH_DIGEST"
          cx_reverify_registry_image "$CX_IMAGE_MULTIARCH"
          cx_resolve_registry_digest() { printf 'sha256:%064d' 0; }
          ! (cx_reverify_registry_image "$CX_IMAGE_MULTIARCH")
          ! (cx_verify_registry_image "$CX_IMAGE_MULTIARCH")
        '''
        subprocess.run(
            ["bash", "-c", command, "_", str(common)],
            check=True,
            env={**os.environ, "COLLECTIVEX_OPERATOR_CONFIG": "/dev/null"},
        )

    def test_canonical_gha_requires_compute_visible_staging(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            repo = root / "repo"
            squash = root / "squash"
            source = repo / "experimental" / "CollectiveX"
            source.mkdir(parents=True)
            squash.mkdir(mode=0o700)
            (source / "public.py").write_text("public\n")
            (source / "private-infra.md").write_text("private\n")
            command = r'''
              set -euo pipefail
              source "$1"
              unset CX_SHARD_FILE CX_STAGE_DIR
              staged="$(COLLECTIVEX_CANONICAL_GHA=1; cx_stage_path "$2" "")"
              cx_stage_repo "$2" "$staged"
              test "$staged" != "$2"
              test -f "$staged/experimental/CollectiveX/public.py"
              test ! -e "$staged/experimental/CollectiveX/private-infra.md"
              cx_cleanup_stage "$staged" "$2"
              test ! -e "$staged"
            '''
            subprocess.run(
                ["bash", "-c", command, "_", str(ROOT / "runtime" / "common.sh"),
                 str(repo)],
                check=True,
                env={
                    **os.environ,
                    "COLLECTIVEX_OPERATOR_CONFIG": "/dev/null",
                    "CX_SQUASH_DIR": str(squash),
                },
            )
            self.assertEqual(list(squash.iterdir()), [])

    def test_manual_stage_does_not_write_to_checkout_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve() / "readonly-parent"
            repo = parent / "repo"
            squash = parent / "squash"
            source = repo / "experimental" / "CollectiveX"
            source.mkdir(parents=True)
            squash.mkdir(mode=0o700)
            (source / "public.py").write_text("public\n")
            original_mode = parent.stat().st_mode & 0o777
            parent.chmod(0o555)
            try:
                command = r'''
                  set -euo pipefail
                  source "$1"
                  unset CX_STAGE_DIR
                  staged="$(cx_stage_path "$2" "")"
                  cx_stage_repo "$2" "$staged"
                  case "$staged" in "$3"/.collectivex-stage-*) ;; *) exit 1 ;; esac
                  test -f "$staged/experimental/CollectiveX/public.py"
                  test ! -e "$4/.collectivex-stage"
                  cx_cleanup_stage "$staged" "$2"
                  test ! -e "$staged"
                '''
                subprocess.run(
                    [
                        "bash", "-c", command, "_",
                        str(ROOT / "runtime" / "common.sh"), str(repo),
                        str(squash), str(parent),
                    ],
                    check=True,
                    env={
                        **os.environ,
                        "COLLECTIVEX_OPERATOR_CONFIG": "/dev/null",
                        "CX_SQUASH_DIR": str(squash),
                    },
                )
            finally:
                parent.chmod(original_mode)
            self.assertEqual(
                sorted(path.name for path in parent.iterdir()),
                ["repo", "squash"],
            )
            self.assertEqual(list(squash.iterdir()), [])

    def test_stage_refuses_to_reuse_an_execution_child(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            repo = root / "repo"
            source = repo / "experimental" / "CollectiveX"
            source.mkdir(parents=True)
            (source / "public.py").write_text("public\n")
            base = root / "stage"
            child = base / "job_collision"
            child.mkdir(parents=True, mode=0o700)
            sentinel = child / "keep"
            sentinel.write_text("keep")
            command = r'''
              source "$1"
              ! (cx_stage_repo "$2" "$3/job_collision")
            '''
            subprocess.run(
                [
                    "bash", "-c", command, "_",
                    str(ROOT / "runtime" / "common.sh"), str(repo), str(base),
                ],
                check=True,
                env={
                    **os.environ,
                    "COLLECTIVEX_OPERATOR_CONFIG": "/dev/null",
                    "COLLECTIVEX_CANONICAL_GHA": "1",
                    "COLLECTIVEX_EXECUTION_ID": "collision",
                    "CX_STAGE_DIR": str(base),
                },
            )
            self.assertEqual(sentinel.read_text(), "keep")

    def test_stage_removes_its_execution_child_when_copy_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            repo = root / "repo"
            source = repo / "experimental" / "CollectiveX"
            source.mkdir(parents=True)
            (source / "public.py").write_text("public\n")
            base = root / "stage"
            base.mkdir(mode=0o700)
            sentinel = root / "python-called"
            command = r'''
              source "$1"
              python3() {
                if [[ "${2:-}" == */experimental/CollectiveX ]]; then
                  : > "$PYTHON_CALLED"
                  return 1
                fi
                command python3 "$@"
              }
              staged="$(cx_stage_path "$2" "$3")"
              ! cx_stage_repo "$2" "$staged"
            '''
            subprocess.run(
                [
                    "bash", "-c", command, "_",
                    str(ROOT / "runtime" / "common.sh"), str(repo), str(base),
                ],
                check=True,
                env={
                    **os.environ,
                    "COLLECTIVEX_OPERATOR_CONFIG": "/dev/null",
                    "COLLECTIVEX_CANONICAL_GHA": "1",
                    "COLLECTIVEX_EXECUTION_ID": f"test-{root.name}",
                    "CX_STAGE_DIR": str(base),
                    "PYTHON_CALLED": str(sentinel),
                },
            )
            self.assertTrue(sentinel.is_file())
            self.assertEqual(list(base.iterdir()), [])

    def test_interrupted_stage_is_cleanup_capable_before_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            repo = root / "repo"
            source = repo / "experimental" / "CollectiveX"
            source.mkdir(parents=True)
            (source / "public.py").write_text("public\n")
            base = root / "stage"
            base.mkdir(mode=0o700)
            sibling = base / "keep"
            sibling.write_text("keep\n")
            command = r'''
              set -euo pipefail
              source "$1"
              export REPO_ROOT="$2" CX_BENCH=nccl-ep
              MOUNT_SRC="$(cx_stage_path "$REPO_ROOT" "$3")"
              cx_install_launcher_fail_safe
              rsync() { kill -TERM $$; return 143; }
              cx_stage_repo "$REPO_ROOT" "$MOUNT_SRC"
            '''
            result = subprocess.run(
                [
                    "bash", "-c", command, "_",
                    str(ROOT / "runtime" / "common.sh"), str(repo), str(base),
                ],
                text=True,
                capture_output=True,
                env={
                    **os.environ,
                    "COLLECTIVEX_OPERATOR_CONFIG": "/dev/null",
                    "COLLECTIVEX_CANONICAL_GHA": "1",
                    "COLLECTIVEX_EXECUTION_ID": "interrupted",
                    "CX_STAGE_DIR": str(base),
                },
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((base / "job_interrupted").exists())
            self.assertEqual(sibling.read_text(), "keep\n")

    def test_stage_base_and_early_cleanup_are_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            repo = root / "repo"
            source = repo / "experimental" / "CollectiveX"
            source.mkdir(parents=True)
            (source / "public.py").write_text("public\n")
            nested = repo / "stage"
            nested.mkdir(mode=0o700)
            group_writable = root / "group-stage"
            group_writable.mkdir(mode=0o770)
            group_writable.chmod(0o770)
            setgid = root / "setgid-stage"
            setgid.mkdir(mode=0o750)
            setgid.chmod(0o2750)
            command = r'''
              set -euo pipefail
              source "$1"
              ! (CX_STAGE_DIR="$3"; cx_stage_path "$2" "$3")
              ! (CX_STAGE_DIR="$4"; cx_stage_path "$2" "$4")
              export CX_STAGE_DIR="$5" COLLECTIVEX_EXECUTION_ID="setgid-$$"
              trap 'cx_cleanup_private_logs 0' EXIT
              staged="$(cx_stage_path "$2" "$CX_STAGE_DIR")"
              cx_stage_repo "$2" "$staged"
              chmod 2700 "$staged"
              cx_cleanup_stage "$staged" "$2"
              test ! -e "$staged"
            '''
            subprocess.run(
                [
                    "bash", "-c", command, "_",
                    str(ROOT / "runtime" / "common.sh"), str(repo), str(nested),
                    str(group_writable), str(setgid),
                ],
                check=True,
                env={**os.environ, "COLLECTIVEX_OPERATOR_CONFIG": "/dev/null"},
            )

            early = r'''
              set -euo pipefail
              source "$1"
              export REPO_ROOT="$2" CX_STAGE_DIR="$3" CX_BENCH=nccl-ep
              export COLLECTIVEX_EXECUTION_ID="pre-marker-$$"
              MOUNT_SRC="$(cx_stage_path "$REPO_ROOT" "$CX_STAGE_DIR")"
              cx_install_launcher_fail_safe
              mkdir -m 700 "$MOUNT_SRC"
              exit 17
            '''
            result = subprocess.run(
                [
                    "bash", "-c", early, "_",
                    str(ROOT / "runtime" / "common.sh"), str(repo), str(setgid),
                ],
                text=True,
                capture_output=True,
                env={**os.environ, "COLLECTIVEX_OPERATOR_CONFIG": "/dev/null"},
            )
            self.assertEqual(result.returncode, 17, result.stderr)
            self.assertEqual(list(setgid.iterdir()), [])

    def test_backend_cache_reuses_v3_and_falls_back_once_without_repair(self) -> None:
        common = ROOT / "runtime" / "common.sh"
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary) / "stage"
            parent.mkdir(mode=0o700)
            concurrent = Path(temporary) / "concurrent"
            concurrent.mkdir(mode=0o700)
            command = r'''
              set -euo pipefail
              source "$1"
              for worker in 1 2 3; do
                (
                  cx_prepare_backend_cache "$2"
                  printf '%s %s\n' "$CX_BACKEND_CACHE_SENTINEL_SHA256" \
                    "$CX_PREPARED_BACKEND_CACHE" > "$3/$worker"
                ) &
              done
              wait
              cmp "$3/1" "$3/2"
              cmp "$3/1" "$3/3"
              cx_prepare_backend_cache "$2"
              first="$CX_PREPARED_BACKEND_CACHE"
              first_digest="$CX_BACKEND_CACHE_SENTINEL_SHA256"
              chmod 2700 "$first"
              cx_prepare_backend_cache "$2"
              second="$CX_PREPARED_BACKEND_CACHE"
              test "$first" = "$second"
              test "$first_digest" = "$CX_BACKEND_CACHE_SENTINEL_SHA256"
              test "$first" = "$(cd "$2" && pwd -P)/.collectivex-backend-cache-v3-$(id -u)"
              export CX_BACKEND_CACHE_ROOT="$first"
              cx_verify_backend_cache_mount
              export CX_BACKEND_CACHE_SENTINEL_SHA256="$(printf '0%.0s' {1..64})"
              ! cx_verify_backend_cache_mount
            '''
            subprocess.run(
                [
                    "bash", "-c", command, "_", str(common), str(parent),
                    str(concurrent),
                ],
                check=True,
            )
            cache = parent / f".collectivex-backend-cache-v3-{os.getuid()}"
            self.assertTrue(cache.is_dir())
            self.assertEqual(cache.stat().st_mode & 0o777, 0o700)
            self.assertEqual(
                list(cache.glob(".collectivex-mount-sentinel-v1.tmp.*")), []
            )
            alias = Path(temporary) / "stage-alias"
            alias.symlink_to(parent, target_is_directory=True)
            canonical = subprocess.run(
                [
                    "bash", "-c",
                    'source "$1"; cx_prepare_backend_cache "$2"; '
                    'printf "%s\\n%s\\n" "$CX_PREPARED_BACKEND_CACHE" '
                    '"$CX_BACKEND_CACHE_SENTINEL_SHA256"',
                    "_", str(common), str(alias),
                ],
                text=True,
                capture_output=True,
                check=True,
            )
            cache_path, digest = canonical.stdout.splitlines()
            self.assertEqual(cache_path, str(cache.resolve()))
            self.assertRegex(digest, r"^[0-9a-f]{64}$")
            saved = parent / "saved-cache"
            cache.rename(saved)
            cache.mkdir(mode=0o700)
            replacement = cache / ".collectivex-mount-sentinel-v1"
            replacement.write_bytes(b"replacement".ljust(32, b"!"))
            replacement.chmod(0o600)
            replaced = subprocess.run(
                [
                    "bash", "-c",
                    'source "$1"; export CX_BACKEND_CACHE_ROOT="$2" '
                    'CX_BACKEND_CACHE_SENTINEL_SHA256="$3"; '
                    'cx_verify_backend_cache_mount',
                    "_", str(common), str(cache), digest,
                ]
            )
            self.assertNotEqual(replaced.returncode, 0)
            replacement.unlink()
            cache.rmdir()
            saved.rename(cache)
            (cache / ".collectivex-mount-sentinel-v1").unlink()
            cache.rmdir()
            target = Path(temporary) / "target"
            target.mkdir(mode=0o700)
            cache.symlink_to(target, target_is_directory=True)
            fallback = subprocess.run(
                [
                    "bash", "-c",
                    'source "$1"; cx_prepare_backend_cache "$2"; '
                    'printf "%s\\n" "$CX_PREPARED_BACKEND_CACHE"',
                    "_", str(common), str(parent),
                ],
                text=True,
                capture_output=True,
                check=True,
            )
            v4 = parent / f".collectivex-backend-cache-v4-{os.getuid()}"
            self.assertEqual(fallback.stdout.strip(), str(v4.resolve()))
            self.assertTrue(cache.is_symlink())
            self.assertTrue(v4.is_dir())
            (v4 / ".collectivex-mount-sentinel-v1").unlink()
            v4.rmdir()
            v4.symlink_to(target, target_is_directory=True)
            result = subprocess.run(
                [
                    "bash", "-c", 'source "$1"; cx_prepare_backend_cache "$2"',
                    "_", str(common), str(parent),
                ],
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn(str(parent), result.stderr)
            self.assertTrue(cache.is_symlink())
            self.assertTrue(v4.is_symlink())

        source = common.read_text().split("cx_prepare_backend_cache() {", 1)[1]
        program = source.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary) / "stage"
            parent.mkdir(mode=0o700)
            fake_os = types.ModuleType("os")
            fake_os.__dict__.update(os.__dict__)
            fake_os.fsync = mock.Mock(side_effect=OSError("forced fsync failure"))
            with (
                mock.patch.dict(sys.modules, {"os": fake_os}),
                mock.patch.object(sys, "argv", ["-", str(parent)]),
                mock.patch.object(sys, "stdout", io.StringIO()),
                self.assertRaises(SystemExit) as failure,
            ):
                exec(compile(program, "<cache-preparation>", "exec"), {})
            self.assertEqual(failure.exception.code, 1)
            self.assertEqual(
                list(parent.rglob(".collectivex-mount-sentinel-v1.tmp.*")), []
            )

    def test_nvidia_namespace_package_roots_come_from_distribution_files(self) -> None:
        runtime = ROOT / "runtime" / "run_in_container.sh"
        with tempfile.TemporaryDirectory() as temporary:
            site = Path(temporary) / "site"
            package = site / "nvidia" / "nccl"
            (package / "include").mkdir(parents=True)
            (package / "lib").mkdir()
            (package / "include" / "nccl.h").write_text("header\n")
            (package / "lib" / "libnccl.so.2").write_text("library\n")
            info = site / "nvidia_nccl_cu13-2.30.4.dist-info"
            info.mkdir()
            (info / "METADATA").write_text(
                "Metadata-Version: 2.1\nName: nvidia-nccl-cu13\nVersion: 2.30.4\n"
            )
            (info / "RECORD").write_text(
                "nvidia/nccl/include/nccl.h,,\n"
                "nvidia/nccl/lib/libnccl.so.2,,\n"
                "nvidia_nccl_cu13-2.30.4.dist-info/METADATA,,\n"
                "nvidia_nccl_cu13-2.30.4.dist-info/RECORD,,\n"
            )
            command = r'''
              set -euo pipefail
              eval "$(sed -n '/^cx_nvidia_package_root()/,/^}/p' "$1")"
              root="$(cx_nvidia_package_root nvidia-nccl-cu13 nccl)"
              test "$root" = "$2/nvidia/nccl"
              ! cx_nvidia_package_root nvidia-nccl-cu13 nvshmem
            '''
            subprocess.run(
                ["bash", "-c", command, "_", str(runtime), str(site.resolve())],
                check=True,
                env={**os.environ, "PYTHONPATH": str(site)},
            )

    def test_cuda_cccl_exports_the_resolved_jit_toolchain_root(self) -> None:
        runtime = ROOT / "runtime" / "run_in_container.sh"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            toolkit = root / "cuda-13.0"
            (toolkit / "bin").mkdir(parents=True)
            (toolkit / "include").mkdir()
            (toolkit / "lib64").mkdir()
            cccl = toolkit / "targets" / "x86_64-linux" / "include" / "cccl"
            cccl.mkdir(parents=True)
            nvcc = toolkit / "bin" / "nvcc"
            nvcc.write_text("#!/bin/sh\nexit 0\n")
            nvcc.chmod(0o755)
            alias = root / "cuda"
            alias.symlink_to(toolkit, target_is_directory=True)
            command = r'''
              set -euo pipefail
              eval "$(sed -n '/^cx_prepare_cuda_cccl()/,/^}/p' "$1")"
              cx_prepare_cuda_cccl
              test "$CUDA_HOME" = "$2"
              test "$CX_CUDA_CCCL" = "$2/targets/x86_64-linux/include/cccl"
              test "$CPATH" = "$2/targets/x86_64-linux/include/cccl:"
              test "$NVCC_PREPEND_FLAGS" = "-I$2/targets/x86_64-linux/include/cccl "
            '''
            subprocess.run(
                ["bash", "-c", command, "_", str(runtime), str(toolkit.resolve())],
                check=True,
                env={
                    **os.environ,
                    "PATH": f"{alias / 'bin'}:{os.environ['PATH']}",
                    "CPATH": "",
                    "NVCC_PREPEND_FLAGS": "",
                },
            )

    def test_deepep_v2_toolchain_rejects_overlay_lock_failure(self) -> None:
        runtime = ROOT / "runtime" / "run_in_container.sh"
        with tempfile.TemporaryDirectory() as temporary:
            command = r'''
              set -euo pipefail
              eval "$(sed -n '/^cx_prepare_deepep_toolchain()/,/^}/p' "$1")"
              cache_root="$2"
              cx_nvidia_package_root() { printf '%s' /unused; }
              cx_deepep_v2_root() { printf '%s' "$cache_root"; }
              cx_log() { :; }
              flock() { return 1; }
              ! cx_prepare_deepep_toolchain
            '''
            subprocess.run(
                ["bash", "-c", command, "_", str(runtime), temporary],
                check=True,
            )

    def test_pinned_source_fetch_retries_transient_failures(self) -> None:
        common = ROOT / "runtime" / "common.sh"
        with tempfile.TemporaryDirectory() as temporary:
            command = r'''
              set -euo pipefail
              eval "$(sed -n '/^cx_git()/,/^}/p' "$1")"
              eval "$(sed -n '/^cx_git_in_tree()/,/^}/p' "$1")"
              eval "$(sed -n '/^cx_fetch_revision()/,/^}/p' "$1")"
              attempts=0
              expected_directory="$(cd -P -- "$3" && pwd -P)"
              sleep() { :; }
              git() {
                local argument has_directory=0 has_trust=0
                if [ "$1" = '-c' ] && [ "$3" = init ]; then
                  mkdir -p "${@: -1}"
                  return 0
                fi
                for argument in "$@"; do
                  [ "$argument" != '-C' ] || has_directory=1
                  [ "$argument" != "safe.directory=$expected_directory" ] || has_trust=1
                  [ "$argument" != 'safe.directory=*' ] || return 1
                done
                [ "$has_directory" = 0 ] || [ "$has_trust" = 1 ] || return 1
                case " $* " in
                  *' fetch '*)
                    attempts=$((attempts + 1))
                    [ "$attempts" = 3 ]
                    ;;
                  *' rev-parse HEAD '*) printf '%s\n' "$revision" ;;
                  *) return 0 ;;
                esac
              }
              cx_fetch_revision https://example.invalid/repo "$2" "$3"
              test "$attempts" = 3
            '''
            revision = "a" * 40
            subprocess.run(
                ["bash", "-c", command, "_", str(common), revision, temporary],
                check=True,
            )

    def test_git_tree_trust_is_exact_and_command_scoped(self) -> None:
        common = ROOT / "runtime" / "common.sh"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = root / "repo"
            repository.mkdir()
            alias = root / "alias"
            alias.symlink_to(repository, target_is_directory=True)
            wildcard = root / "*"
            wildcard.mkdir()
            arguments = root / "arguments"
            command = r'''
              set -euo pipefail
              eval "$(sed -n '/^cx_git()/,/^}/p' "$1")"
              eval "$(sed -n '/^cx_git_in_tree()/,/^}/p' "$1")"
              arguments="$4"
              git() { printf '%s\n' "$@" > "$arguments"; }
              cx_git_in_tree "$2" status --porcelain
              ! cx_git_in_tree relative status
              ! cx_git_in_tree "$3" status
              ! cx_git_in_tree "$5" status
            '''
            subprocess.run(
                [
                    "bash",
                    "-c",
                    command,
                    "_",
                    str(common),
                    str(repository),
                    str(alias),
                    str(arguments),
                    str(wildcard),
                ],
                check=True,
            )
            self.assertEqual(
                arguments.read_text().splitlines(),
                [
                    "-c",
                    "credential.helper=",
                    "-c",
                    f"safe.directory={repository.resolve()}",
                    "-C",
                    str(repository.resolve()),
                    "status",
                    "--porcelain",
                ],
            )
            self.assertNotIn("safe.directory=*", arguments.read_text())

    def test_runtime_materializes_the_verified_host_source_without_network(self) -> None:
        common = ROOT / "runtime" / "common.sh"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            seed = root / "seed"
            seed.mkdir()
            (seed / "pinned").write_text("source\n")
            destination = root / "build"
            fetched = root / "network-fetch"
            command = r'''
              set -euo pipefail
              source "$1"
              export CX_BACKEND_SOURCE_ROOT="$2/source"
              SEED="$3" FETCHED="$5"
              copy_mode=
              cx_backend_source_path() { printf '%s' "$SEED"; }
              cx_backend_source_is_valid() { test -f "$2/pinned"; }
              cx_fetch_revision() { : > "$FETCHED"; return 1; }
              cp() {
                test "$1" = -R
                copy_mode=recursive
                command cp "$@"
              }
              cx_materialize_backend_source deepep-hybrid "$4"
              test -f "$4/pinned"
              test "$copy_mode" = recursive
              python3 - "$4" <<'PY'
import os
import stat
import sys
assert stat.S_IMODE(os.stat(sys.argv[1]).st_mode) == 0o700
PY
              test ! -e "$FETCHED"
            '''
            subprocess.run(
                [
                    "bash", "-c", command, "_", str(common), str(root),
                    str(seed), str(destination), str(fetched),
                ],
                check=True,
            )

    def test_backend_source_validation_rejects_status_errors_and_ignored_files(self) -> None:
        common = ROOT / "runtime" / "common.sh"
        with tempfile.TemporaryDirectory() as temporary:
            command = r'''
              set -euo pipefail
              source "$1"
              cx_backend_source_pin() { printf '%s|%s|' revision tree; }
              git() {
                case " $* " in
                  *' rev-parse HEAD '*) printf '%s\n' revision ;;
                  *' rev-parse HEAD^{tree} '*) printf '%s\n' tree ;;
                  *' status --porcelain '*)
                    [ "${GIT_OPTIONAL_LOCKS:-}" = 0 ] && [ "$mode" != status-error ]
                    ;;
                  *' ls-files --others --ignored '*)
                    [ "$mode" != ignored ] || printf '%s\n' ignored.bin
                    ;;
                  *) return 1 ;;
                esac
              }
              mode=status-error
              ! cx_backend_source_is_valid backend "$2"
              mode=ignored
              ! cx_backend_source_is_valid backend "$2"
              mode=clean
              cx_backend_source_is_valid backend "$2"
            '''
            subprocess.run(
                ["bash", "-c", command, "_", str(common), temporary],
                check=True,
            )

    def test_deepep_v2_applies_only_the_pinned_upstream_nccl_check_fix(self) -> None:
        common = ROOT / "runtime" / "common.sh"
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            init = source / "deep_ep" / "__init__.py"
            init.parent.mkdir()
            init.write_text(
                "for so in [line.strip().split(' ')[-1] for line in f if 'nccl' in line]:\n"
            )
            command = r'''
              set -euo pipefail
              source "$1"
              expected_file="$2/expected"
              sed "s/if 'nccl' in line/if 'libnccl' in line/" \
                "$2/deep_ep/__init__.py" > "$expected_file"
              CX_DEEPEP_V2_INIT_SHA256="$(sha256sum "$expected_file" | awk '{print $1}')"
              rm -f "$expected_file"
              cx_apply_deepep_v2_nccl_check_fix "$2"
              grep -Fq "if 'libnccl' in line" "$2/deep_ep/__init__.py"
              ! cx_apply_deepep_v2_nccl_check_fix "$2"
            '''
            subprocess.run(
                ["bash", "-c", command, "_", str(common), temporary],
                check=True,
            )

    def test_backend_source_root_normalizes_inherited_special_mode(self) -> None:
        common = ROOT / "runtime" / "common.sh"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root = root / "experimental" / "CollectiveX" / ".cx_sources"
            source = source_root / "backend-revision"
            source.mkdir(parents=True)
            command = r'''
              set -euo pipefail
              source "$1"
              export COLLECTIVEX_EXECUTION_ID="source-mode-$$"
              trap 'cx_cleanup_private_logs 0' EXIT
              expected_mount="$2"
              expected_source="$3"
              expected_root="${expected_source%/*}"
              observed_mode=2700
              mock_stage_owner=4200
              mock_root_owner=4200
              chmod_calls=0
              chmod() {
                test "$1" = 700 && test "$2" = "$expected_root"
                chmod_calls=$((chmod_calls + 1))
                [ "$chmod_calls" = 2 ] || return 1
                observed_mode=700
              }
              stat() {
                case "$2" in
                  %u)
                    case "$3" in
                      "$expected_mount") printf '%s\n' "$mock_stage_owner" ;;
                      "$expected_root") printf '%s\n' "$mock_root_owner" ;;
                      *) return 1 ;;
                    esac
                    ;;
                  %a)
                    case "$3" in
                      "$expected_mount") printf '2700\n' ;;
                      "$expected_root") printf '%s\n' "$observed_mode" ;;
                      *) return 1 ;;
                    esac
                    ;;
                  *) return 1 ;;
                esac
              }
              cx_backend_source_path() { printf '%s' "$expected_source"; }
              cx_backend_source_is_valid() {
                test "$1" = backend && test "$2" = "$expected_source"
              }
              cx_prepare_backend_source "$2" backend
              test "$observed_mode" = 2700
              test "$chmod_calls" = 0
              observed_mode=2750
              ! _cx_prepare_backend_source "$2" backend
              test "$chmod_calls" = 1
              _cx_prepare_backend_source "$2" backend
              test "$observed_mode" = 700
              mock_root_owner=4300
              ! _cx_prepare_backend_source "$2" backend
            '''
            subprocess.run(
                ["bash", "-c", command, "_", str(common), str(root), str(source)],
                check=True,
            )

    def test_canonical_backend_sources_use_verified_seed_without_network(self) -> None:
        common = ROOT / "runtime" / "common.sh"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mount = root / "mount"
            source_root = mount / "experimental" / "CollectiveX" / ".cx_sources"
            seed_root = root / "seed"
            seeds = [
                seed_root / f"{backend}-revision"
                for backend in ("backend-one", "backend-two")
            ]
            mount.mkdir(mode=0o700)
            source_root.parent.mkdir(parents=True, mode=0o700)
            for seed in seeds:
                seed.mkdir(parents=True, mode=0o700)
                (seed / "pinned").write_text("source\n")
            network = root / "network"
            command = r'''
              set -euo pipefail
              source "$1"
              export COLLECTIVEX_CANONICAL_GHA=1
              export CX_BACKEND_SOURCE_SEED_ROOT="$4"
              export COLLECTIVEX_EXECUTION_ID="source-seed-$$"
              trap 'cx_cleanup_private_logs 0' EXIT
              NETWORK="$5"
              stat() {
                case "$2" in
                  %u) printf '4200\n' ;;
                  %a) printf '700\n' ;;
                  *) return 1 ;;
                esac
              }
              cx_backend_source_path() { printf '%s/%s-revision' "$1" "$2"; }
              cx_backend_source_is_valid() { test -f "$2/pinned"; }
              cx_fetch_revision() { : > "$NETWORK"; return 1; }
              cx_prepare_backend_source "$2" backend-one
              cx_prepare_backend_source "$2" backend-two
              test -f "$3/backend-one-revision/pinned"
              test -f "$3/backend-two-revision/pinned"
              test ! -e "$NETWORK"
              rm -rf -- "$3/backend-one-revision" "$3/backend-two-revision"
              unset CX_BACKEND_SOURCE_SEED_ROOT
              ! _cx_prepare_backend_source "$2" backend-one
              test ! -e "$NETWORK"
            '''
            subprocess.run(
                [
                    "bash", "-c", command, "_", str(common), str(mount),
                    str(source_root), str(seed_root), str(network),
                ],
                check=True,
            )

    def test_deepep_hybrid_cache_reuse_revalidates_extensions(self) -> None:
        common = ROOT / "runtime" / "common.sh"
        runtime = ROOT / "runtime" / "run_in_container.sh"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "deep_ep_cpp.so").write_bytes(b"deep")
            (root / "hybrid_ep_cpp.so").write_bytes(b"hybrid")
            command = r'''
              set -euo pipefail
              chmod 700 "$3"
              source "$1"
              eval "$(sed -n '/^cx_deepep_hybrid_marker_content_sha256()/,/^}/p' "$2")"
              eval "$(sed -n '/^cx_deepep_hybrid_cache_is_valid()/,/^}/p' "$2")"
              revision=revision tree=tree
              cx_git() {
                case " $* " in
                  *' rev-parse HEAD '*) printf '%s\n' "$revision" ;;
                  *' rev-parse HEAD^{tree} '*) printf '%s\n' "$tree" ;;
                  *' status --porcelain '*|*' ls-files --others '*) return 0 ;;
                  *) return 1 ;;
                esac
              }
              cx_git_in_tree() { shift; cx_git "$@"; }
              marker="$3/.collectivex-complete"
              digest="$(cx_extension_pair_sha256 "$3" 'deep_ep_cpp*.so' 'hybrid_ep_cpp*.so')"
              (umask 077; printf '%s\n%s\n%s\n' "$revision" "$tree" "$digest" > "$marker")
              cx_deepep_hybrid_cache_is_valid "$3" "$marker" "$revision" "$tree"
              printf changed > "$3/hybrid_ep_cpp.so"
              ! cx_deepep_hybrid_cache_is_valid "$3" "$marker" "$revision" "$tree"
              printf hybrid > "$3/hybrid_ep_cpp.so"
              cp "$3/deep_ep_cpp.so" "$3/deep_ep_cpp-extra.so"
              ! cx_deepep_hybrid_cache_is_valid "$3" "$marker" "$revision" "$tree"
            '''
            subprocess.run(
                ["bash", "-c", command, "_", str(common), str(runtime), temporary],
                check=True,
            )

    def test_rack_backend_environment_is_shared_per_node_and_required(self) -> None:
        runtime = ROOT / "runtime" / "run_in_container.sh"
        launcher = (ROOT / "launchers" / "launch_gb-nv.sh").read_text()
        assignment = next(
            line for line in launcher.splitlines()
            if line.startswith("SOURCE_BACKEND_ENV=")
        )
        self.assertNotIn("/tmp/.cx_backend_env", launcher)
        self.assertIn('[ -f "$env_file" ] && [ -r "$env_file" ]', launcher)
        self.assertIn('[ ! -L "$env_file" ]', launcher)
        self.assertIn('$(stat -c "%u" "$env_root"):600', launcher)
        self.assertIn('case "$(stat -c "%a" "$env_root")" in 700|[1-7]700)', launcher)
        self.assertIn("node-${SLURM_NODEID}.sh", launcher)
        self.assertIn("HybridEPBuffer", launcher)
        self.assertIn('. "$env_file" || exit 66', launcher)
        with tempfile.TemporaryDirectory() as temporary:
            consumer = r'''
              eval "$1"
              env_root="$2/env"
              SOURCE_BACKEND_ENV="${SOURCE_BACKEND_ENV//\/ix\/experimental\/CollectiveX\/.cx_backend\/env/$env_root}"
              mkdir -p "$env_root"
              env_file="$env_root/node-1.sh"
              printf 'printf sourced > "$CX_SENTINEL"\n' > "$env_file"
              chmod 600 "$env_file"
              export CX_SENTINEL="$2/sentinel"
              stat() {
                [ "${STAT_FAIL:-0}" = 0 ] || return 1
                case "$2" in
                  %a) printf '%s\n' "$ROOT_MODE" ;;
                  %u) printf '1000\n' ;;
                  %u:%a) printf '%s\n' "$FILE_OWNER_MODE" ;;
                  *) return 2 ;;
                esac
              }
              run_case() {
                rm -f "$CX_SENTINEL"
                ROOT_MODE="$1" FILE_OWNER_MODE="$2" STAT_FAIL="$3" SLURM_NODEID="$4"
                ( eval "$SOURCE_BACKEND_ENV" )
                rc=$?
                [ "$rc" = "$5" ] || return 1
                if [ "$5" = 0 ]; then
                  [ -f "$CX_SENTINEL" ]
                else
                  [ ! -e "$CX_SENTINEL" ]
                fi
              }
              run_case 700 1000:600 0 1 0
              run_case 2700 1000:600 0 1 0
              run_case 755 1000:600 0 1 66
              run_case 700 1000:600 1 1 66
              run_case 700 2000:600 0 1 66
              mv "$env_file" "$env_file.real"
              ln -s "$env_file.real" "$env_file"
              run_case 700 1000:600 0 1 66
              rm "$env_file"
              mv "$env_file.real" "$env_file"
              run_case 700 1000:600 0 invalid 66
            '''
            subprocess.run(
                ["bash", "-c", consumer, "_", assignment, temporary],
                check=True,
            )
            command = r'''
              set -euo pipefail
              cd "$2"
              eval "$(sed -n '/^cx_persist_backend_env()/,/^}/p' "$1")"
              export SLURM_NODEID=1 PYTHONPATH=/ix/pinned DEEPEP_COMMIT=abc
              cx_persist_backend_env
              env_file="$PWD/.cx_backend/env/node-1.sh"
              test -f "$env_file"
              test "$(stat -f %Lp "$env_file" 2>/dev/null || stat -c %a "$env_file")" = 600
              unset PYTHONPATH DEEPEP_COMMIT
              . "$env_file"
              test "$PYTHONPATH" = /ix/pinned
              test "$DEEPEP_COMMIT" = abc
              SLURM_NODEID=invalid && ! cx_persist_backend_env
            '''
            subprocess.run(
                ["bash", "-c", command, "_", str(runtime), temporary],
                check=True,
            )

    def test_stage_cleanup_failure_fails_job_but_marks_allocation_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "repo").mkdir()
            (root / "stage").mkdir()
            command = r'''
              source "$1"
              cx_write_cleanup_guard() {
                rm -f -- "$CX_JOB_ROOT/cleanup-safe" "$CX_JOB_ROOT/cleanup-unsafe"
                : > "$CX_JOB_ROOT/cleanup-$1"
              }
              cx_cleanup_stage() { return 1; }
              cx_cleanup_private_logs() { : > "$CX_JOB_ROOT/logs-deleted"; }
              export CX_JOB_ROOT="$2" REPO_ROOT="$2/repo" MOUNT_SRC="$2/stage"
              export COLLECTIVEX_CANONICAL_GHA=1 CX_ALLOCATION_UNCERTAIN=0
              unset CX_BENCH JOB_ID
              cx_launcher_cleanup 0
            '''
            result = subprocess.run(
                ["bash", "-c", command, "_", str(ROOT / "runtime" / "common.sh"),
                 str(root)],
                text=True,
                capture_output=True,
                env={**os.environ, "COLLECTIVEX_OPERATOR_CONFIG": "/dev/null"},
            )
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertTrue((root / "cleanup-safe").is_file())
            self.assertFalse((root / "cleanup-unsafe").exists())
            self.assertFalse((root / "logs-deleted").exists())

    def test_generated_stage_cleanup_never_removes_configured_base(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "stage"
            repo = root / "repo"
            generated = base / "job_execution"
            generated.mkdir(parents=True, mode=0o700)
            repo.mkdir()
            marker = generated / ".collectivex-stage-v1"
            marker.write_text("collectivex-stage-v1\nexecution\n")
            marker.chmod(0o600)
            (generated / "payload").write_text("temporary")
            subprocess.run(
                [
                    "bash", "-c",
                    'source "$1"; cx_cleanup_stage "$2" "$3"; '
                    '! cx_cleanup_stage "$4" "$3"',
                    "_", str(ROOT / "runtime" / "common.sh"), str(generated),
                    str(repo), str(base),
                ],
                check=True,
                env={
                    **os.environ,
                    "COLLECTIVEX_OPERATOR_CONFIG": "/dev/null",
                    "COLLECTIVEX_EXECUTION_ID": "execution",
                    "CX_STAGE_DIR": str(base),
                },
            )
            self.assertFalse(generated.exists())
            self.assertTrue(base.is_dir())
            self.assertTrue(repo.is_dir())

    def test_adapters_do_not_retain_dead_expected_methods(self) -> None:
        for path in HERE.glob("ep_*.py"):
            tree = ast.parse(path.read_text(), str(path))
            methods = {
                node.name for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            self.assertNotIn("expected", methods, path.name)

    def test_artifact_safety_rejects_sensitive_material(self) -> None:
        private_address = ".".join(str(octet) for octet in (10, 0, 0, 1))
        secret = "github_pat_" + "A" * 24
        sensitive = {
            "ipv4": ({"note": private_address}, private_address),
            "ipv6": ({"note": "[2001:db8::1]:29500"}, "2001:db8::1"),
            "user-at-host": ({"note": "ssh admin@private-host"}, "admin@private-host"),
            "hostname": ({"note": "host=compute-17"}, "compute-17"),
            "private-dns": ({"note": "worker-7.cluster.local"}, "worker-7.cluster.local"),
            "suffixed-host": ({"worker_hostname": "relative"}, "worker_hostname"),
            "suffixed-address": ({"control_address": "relative"}, "control_address"),
            "suffixed-path": ({"scheduler_path": "relative"}, "scheduler_path"),
            "exact-address": ({"address": "relative"}, "address"),
            "exact-ip": ({"ip": "relative"}, "ip"),
            "camel-host": ({"workerHost": "relative"}, "workerHost"),
            "camel-path": ({"schedulerPath": "relative"}, "schedulerPath"),
            "acronym-gpu-uuid": ({"gpuUUID": "relative"}, "gpuUUID"),
            "acronym-device-uuid": ({"deviceUUID": "relative"}, "deviceUUID"),
            "acronym-pci-bus": ({"pciBusID": "relative"}, "pciBusID"),
            "mac-address": ({"note": "00:11:22:33:44:55"}, "00:11:22:33:44:55"),
            "ib-guid": ({"note": "00:11:22:33:44:55:66:77"}, "00:11:22:33:44:55:66:77"),
            "dgx-host": ({"note": "dgx-b300-001"}, "dgx-b300-001"),
            "cloud-host": ({"note": "ip-10-20-30-40"}, "ip-10-20-30-40"),
            "credential-field": ({"service_token": "short"}, "service_token"),
            "prefixed-token": ({"note": secret}, secret),
            "hf-token": ({"note": "hf_" + "A" * 24}, "hf_" + "A" * 24),
            "payment-token": ({"note": "sk_live_" + "A" * 24}, "sk_live_" + "A" * 24),
            "generic-secret": ({"note": "password=not-a-real-secret"}, "not-a-real-secret"),
        }
        for root in ("data", "it-share", "lustre", "raid", "nvme_home", "scratch", "gpfs", "fsx"):
            value = f"/{root}/collectivex/run"
            sensitive[f"private-root-{root}"] = ({"note": value}, value)
        for name, (document, offending) in sensitive.items():
            with self.subTest(name=name), self.assertRaises(
                artifact_safety.ArtifactSafetyError
            ) as caught:
                artifact_safety.assert_publication_safe([document])
            self.assertNotIn(offending, str(caught.exception))

        artifact_safety.assert_publication_safe([{
            "runner": "b300",
            "redaction": "sanitized-v1",
            "path": "datasets/" + "a" * 64 + "/dataset.json",
            "timing": "8:64:32",
            "image_digest": "sha256:" + "b" * 64,
            "source": "github.com",
        }])
        for ref in ("release@candidate", "worker1-feature", "sk-refactor-long-component-name"):
            artifact_safety.assert_publication_safe([{"ref": ref}])

    def test_artifact_safety_cli_does_not_echo_sensitive_values(self) -> None:
        private_value = ".".join(str(octet) for octet in (10, 24, 68, 12))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact.json"
            path.write_text(json.dumps({"note": private_value}))
            result = subprocess.run(
                [sys.executable, str(ROOT / "artifact_safety.py"), str(path)],
                text=True,
                capture_output=True,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("forbidden ipv4-address value", result.stderr)
        self.assertNotIn(private_value, result.stderr)

    def test_artifact_safety_rejects_linked_and_special_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            source.write_text("{}")
            linked = root / "linked.json"
            linked.symlink_to(source)
            fifo = root / "fifo.json"
            os.mkfifo(fifo)
            for path in (linked, fifo):
                with self.subTest(path=path.name), self.assertRaises(
                    artifact_safety.ArtifactSafetyError
                ):
                    artifact_safety.load_documents([str(path)])


if __name__ == "__main__":
    unittest.main()
