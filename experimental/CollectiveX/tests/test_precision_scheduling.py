#!/usr/bin/env python3
"""CPU-only tests for CollectiveX communication-precision scheduling."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]

import capability  # noqa: E402
import identity  # noqa: E402
import probe_precision  # noqa: E402
import sweep_matrix  # noqa: E402


class PrecisionSchedulingTest(unittest.TestCase):
    @staticmethod
    def _synthetic_provisional_targets() -> list[dict[str, object]]:
        targets = []
        for ep in (8, 16):
            target = next(
                item for item in capability.precision_targets()
                if item["sku"] == "h100-dgxc" and item["backend"] == "deepep"
                and item["ep"] == ep and item["mode"] == "normal"
            ).copy()
            target.update(disposition="provisional", basis="test-provisional-cell")
            targets.append(target)
        return targets

    def test_precision_probe_inventory_is_exact_and_non_mutating(self) -> None:
        before = copy.deepcopy(capability.PRECISION_CAPABILITIES)
        targets = probe_precision.provisional_targets()
        key = lambda item: (
            item["sku"], item["backend"], item["ep"], item["mode"],
            item["precision_profile"],
        )
        self.assertEqual(targets, sorted(capability.provisional_precision_targets(), key=key))
        self.assertEqual(
            [
                (item["sku"], item["backend"], item["ep"], item["precision_profile"])
                for item in targets
            ],
            [
                ("mi300x", "mori", 16, "d-fp8-e4m3fnuz-b128-f32-prequantized.c-bf16"),
            ],
        )
        self.assertEqual(capability.PRECISION_CAPABILITIES, before)
        self.assertEqual(
            len({
                (item["backend"], item["sku"], item["ep"], item["mode"],
                 item["precision_profile"])
                for item in targets
            }),
            len(targets),
        )

    def test_precision_probe_selects_only_one_exact_provisional_cell(self) -> None:
        target = self._synthetic_provisional_targets()[0]
        with mock.patch.object(probe_precision, "provisional_targets", return_value=[target]):
            selected = probe_precision.select_target(
                backend=target["backend"], sku=target["sku"], ep=target["ep"],
                mode=target["mode"], precision_profile=target["precision_profile"],
            )
        self.assertEqual(selected, target)
        with self.assertRaisesRegex(probe_precision.ProbeError, "target-not-provisional"):
            probe_precision.select_target(
                backend=target["backend"], sku=target["sku"], ep=target["ep"],
                mode=target["mode"], precision_profile="d-bf16.c-bf16",
            )

    def test_precision_probe_workflow_plan_binds_exact_controls(self) -> None:
        targets = self._synthetic_provisional_targets()
        with mock.patch.object(probe_precision, "provisional_targets", return_value=targets):
            plan = probe_precision.workflow_plan(backend="deepep", only_sku="h100-dgxc")
        self.assertTrue(plan["include"])
        self.assertTrue(all(
            row["backend"] == "deepep" and row["sku"] == "h100-dgxc"
            for row in plan["include"]
        ))
        row = plan["include"][0]
        with mock.patch.object(probe_precision, "provisional_targets", return_value=targets):
            exact = probe_precision.workflow_plan(probe_id=row["id"])
        self.assertEqual(exact["include"], [row])
        with self.assertRaisesRegex(ValueError, "ID is invalid"):
            probe_precision.workflow_plan(probe_id="probe-invalid")
        with self.assertRaisesRegex(ValueError, "select no provisional"):
            probe_precision.workflow_plan(probe_id="probe-00000000000000000000")
        with mock.patch.object(probe_precision, "provisional_targets", return_value=targets):
            control = probe_precision.extract_control(
                plan, probe_id=row["id"], sku=row["sku"], backend=row["backend"],
                nodes=row["nodes"],
            )
            self.assertEqual(
                probe_precision.validate_control(
                    control, sku=row["sku"], backend=row["backend"], nodes=row["nodes"],
                ),
                control,
            )
        with mock.patch.object(probe_precision, "provisional_targets", return_value=targets), \
                self.assertRaisesRegex(ValueError, "workflow matrix"):
            probe_precision.extract_control(
                plan, probe_id=row["id"], sku=row["sku"], backend=row["backend"],
                nodes=row["nodes"] + 1,
            )
        with self.assertRaisesRegex(ValueError, "select no provisional"):
            probe_precision.workflow_plan(backend="mori", only_sku="b200-dgxc")
        with mock.patch.object(probe_precision, "provisional_targets", return_value=targets):
            ep16 = probe_precision.workflow_plan(
                backend="deepep", only_sku="h100-dgxc", min_nodes=2,
            )
        self.assertTrue(ep16["include"])
        self.assertEqual({row["ep"] for row in ep16["include"]}, {16})
        with self.assertRaisesRegex(ValueError, "node filters"):
            probe_precision.workflow_plan(min_nodes=2, max_nodes=1)

    def test_precision_probe_manifest_is_sanitized_and_runtime_evidence_is_required(self) -> None:
        target = self._synthetic_provisional_targets()[0]
        topology = capability.topology_for(target["sku"], target["ep"])
        self.assertIsNotNone(topology)
        topology_record = probe_precision._topology_record(topology, False)
        document = probe_precision.build_manifest(
            target=target, topology=topology_record, disposition="unsupported",
            reason="unsupported-native-api", runtime_executed=True, evidence=None,
        )
        self.assertEqual(document["result"], {
            "disposition": "unsupported",
            "reason": "unsupported-native-api",
            "registry_mutation": False,
            "runtime_executed": True,
            "static_inspection_sufficient": False,
        })
        with self.assertRaises((TypeError, ValueError)):
            probe_precision.build_manifest(
                target=target, topology=topology_record, disposition="supported",
                reason=probe_precision.SUPPORTED_REASON, runtime_executed=True,
                evidence=None,
            )

    def test_precision_probe_aggregates_shapes_as_manifest_arrays(self) -> None:
        tensor = probe_precision._aggregate_tensor_summaries([
            {"finite": True, "rank": 2, "shape": [8, 7168], "storage_dtype": "bfloat16"},
            {"finite": True, "rank": 2, "shape": [16, 7168], "storage_dtype": "bfloat16"},
        ])
        self.assertEqual(tensor["shapes"], [[8, 7168], [16, 7168]])
        self.assertTrue(all(type(shape) is list for shape in tensor["shapes"]))

        scale = probe_precision._aggregate_scale_contracts([
            {
                "alignment": "hidden-block-128", "dtype": "f32", "finite": True,
                "group_size": 128, "layout": "per-token-hidden-block",
                "padding": "right-zero-pad-hidden-to-128", "positive": True,
                "runtime_shape": [8, 56], "runtime_storage_dtype": "float32",
            },
            {
                "alignment": "hidden-block-128", "dtype": "f32", "finite": True,
                "group_size": 128, "layout": "per-token-hidden-block",
                "padding": "right-zero-pad-hidden-to-128", "positive": True,
                "runtime_shape": [16, 56], "runtime_storage_dtype": "float32",
            },
        ])
        self.assertEqual(scale["runtime_shapes"], [[8, 56], [16, 56]])
        self.assertTrue(all(type(shape) is list for shape in scale["runtime_shapes"]))

    def test_precision_profiles_bind_exact_formats_and_timing_boundaries(self) -> None:
        scheduled = set(identity.V1_NORMAL_PRECISION_PROFILE_IDS) | set(
            identity.V1_LOW_LATENCY_PRECISION_PROFILE_IDS
        )
        self.assertEqual(
            set(identity.V1_PRECISION_PROFILES),
            scheduled | {identity.V1_CONTROL_PRECISION_PROFILE},
        )
        self.assertNotIn(identity.V1_CONTROL_PRECISION_PROFILE, scheduled)
        self.assertNotIn("fp4", repr(identity.V1_PRECISION_PROFILES).lower())

        required_axis_fields = {
            "api_input_dtype",
            "api_output_dtype",
            "communication_format",
            "scale_dtype",
            "scale_layout",
            "scale_group_size",
            "padding_contract",
            "alignment_contract",
            "quantization_origin",
            "conversion_boundary",
        }
        for name in identity.V1_PRECISION_PROFILES:
            with self.subTest(profile=name):
                profile = identity.precision_profile(name)
                self.assertEqual(profile["profile_id"], name)
                self.assertEqual(set(profile["dispatch"]), required_axis_fields)
                self.assertEqual(set(profile["combine"]), required_axis_fields)
                self.assertRegex(name, r"^[a-z0-9][a-z0-9.-]*$")

        prequantized = identity.precision_profile(
            "d-fp8-e4m3fn-b128-f32-prequantized.c-bf16"
        )["dispatch"]
        fused = identity.precision_profile(
            "d-fp8-e4m3fn-b128-f32-fused.c-bf16"
        )["dispatch"]
        self.assertEqual(prequantized["conversion_boundary"], "before-dispatch-timing")
        self.assertEqual(fused["conversion_boundary"], "inside-dispatch-timing")
        self.assertEqual(prequantized["scale_group_size"], 128)

        logfmt = identity.precision_profile("d-bf16.c-logfmt10-dynamic64")["combine"]
        self.assertEqual(
            (logfmt["communication_format"], logfmt["scale_group_size"]),
            ("logfmt10", 64),
        )
        self.assertEqual(
            identity.combine_oracle_tolerances(
                identity.precision_profile(identity.V1_CONTROL_PRECISION_PROFILE)
            ),
            {"atol": 2e-2, "rtol": 5e-2},
        )
        self.assertEqual(
            identity.combine_oracle_tolerances(
                identity.precision_profile("d-bf16.c-logfmt10-dynamic64")
            ),
            {"atol": 3e-2, "rtol": 6e-2},
        )
        self.assertEqual(
            identity.combine_oracle_tolerances(identity.precision_profile(
                "d-bf16.c-fp8-e4m3fn-direct-cast-noscale"
            )),
            {"atol": 4e-2, "rtol": 8e-2},
        )

        base = {"mode": "normal"}
        precision_case = {
            **base,
            "precision_profile": "d-fp8-e4m3fn-b128-f32-prequantized.c-bf16",
        }
        self.assertIs(identity.profile_for_case(base), identity.V1_NORMAL_CASE_PROFILE)
        self.assertIn("communication_precision", identity.profile_for_case(precision_case))
        self.assertNotEqual(
            identity.digest("case", identity.profile_for_case(base)),
            identity.digest("case", identity.profile_for_case(precision_case)),
        )

    def test_capability_registry_uses_exact_native_targets(self) -> None:
        targets = capability.precision_targets()
        self.assertTrue(targets)
        self.assertEqual(
            {item["disposition"] for item in targets},
            {"supported", "unsupported", "provisional"},
        )
        self.assertEqual(
            len(targets), 94
        )
        self.assertEqual(
            sum(item["disposition"] == "supported" for item in targets), 66
        )
        self.assertEqual(
            sum(item["disposition"] == "unsupported" for item in targets), 27
        )
        self.assertEqual(len(capability.provisional_precision_targets()), 1)
        keys = {
            (
                item["precision_profile"],
                item["backend"],
                item["sku"],
                item["ep"],
                item["mode"],
            )
            for item in targets
        }
        self.assertEqual(len(keys), len(targets))

        normal = "d-fp8-e4m3fn-b128-f32-prequantized.c-bf16"
        direct = "d-bf16.c-fp8-e4m3fn-direct-cast-noscale"
        low_latency = "d-bf16.c-logfmt10-dynamic64"
        cases = (
            (("h200-dgxc", "deepep-v2", 8, "normal", normal), "unsupported"),
            (("h100-dgxc", "deepep-v2", 8, "normal", normal), "not-applicable"),
            (("h200-dgxc", "nccl-ep", 8, "normal", normal), "not-applicable"),
            (("mi355x", "mori", 8, "normal", direct), "unsupported"),
            (("mi355x", "mori", 16, "normal", direct), "not-applicable"),
            (("h200-dgxc", "deepep", 8, "low-latency", low_latency), "supported"),
            (("h100-dgxc", "deepep", 16, "low-latency", low_latency), "supported"),
            (("h200-dgxc", "deepep-hybrid", 8, "low-latency", low_latency),
             "not-applicable"),
        )
        for (sku, backend, ep, mode, profile), expected in cases:
            with self.subTest(sku=sku, backend=backend, profile=profile):
                topology = capability.topology_for(sku, ep)
                self.assertIsNotNone(topology)
                disposition, _ = capability.resolve_disposition(
                    sku,
                    backend,
                    ep=ep,
                    nodes=topology["nodes"],  # type: ignore[index]
                    mode=mode,
                    precision_profile=profile,
                )
                self.assertEqual(disposition, expected)

        control, _ = capability.resolve_disposition(
            "h200-dgxc",
            "deepep",
            ep=8,
            nodes=1,
            precision_profile=identity.V1_CONTROL_PRECISION_PROFILE,
        )
        self.assertEqual(control, "supported")
        h100_ep16, reason = capability.resolve_disposition(
            "h100-dgxc", "nccl-ep", ep=16, nodes=2,
            precision_profile=identity.V1_CONTROL_PRECISION_PROFILE,
        )
        self.assertEqual(h100_ep16, "supported")
        self.assertEqual(reason, "ok")

        for backend in ("deepep", "deepep-v2", "deepep-hybrid", "nccl-ep"):
            disposition, reason = capability.resolve_disposition(
                "b300", backend, ep=16, nodes=2,
                precision_profile=identity.V1_CONTROL_PRECISION_PROFILE,
            )
            self.assertEqual(disposition, "unsupported")
            self.assertEqual(
                reason, "v1 publication fabric unavailable for B300 EP16"
            )
        b300_ep8, reason = capability.resolve_disposition(
            "b300", "nccl-ep", ep=8, nodes=1,
            precision_profile=identity.V1_CONTROL_PRECISION_PROFILE,
        )
        self.assertEqual(b300_ep8, "supported")
        self.assertEqual(reason, "ok")

        disposition, reason = capability.resolve_disposition(
            "b200-dgxc", "deepep", ep=16, nodes=2,
            precision_profile=identity.V1_CONTROL_PRECISION_PROFILE,
        )
        self.assertEqual(disposition, "supported")
        self.assertEqual(reason, "ok")
        self.assertEqual(
            capability.DEEPEP_V1_IBGDA_NIC_HANDLERS["b200-dgxc"],
            "cpu_host_memory",
        )

        disposition, reason = capability.resolve_disposition(
            "b200-dgxc", "deepep-hybrid", ep=16, nodes=2,
            precision_profile=identity.V1_CONTROL_PRECISION_PROFILE,
        )
        self.assertEqual(disposition, "unsupported")
        self.assertEqual(
            reason,
            "DeepEP Hybrid EP16 requires unavailable GPU-to-NIC doorbell/UAR mappings on B200",
        )
        b200_reference, reason = capability.resolve_disposition(
            "b200-dgxc", "nccl-ep", ep=16, nodes=2,
            precision_profile=identity.V1_CONTROL_PRECISION_PROFILE,
        )
        self.assertEqual(b200_reference, "supported")
        self.assertEqual(reason, "ok")

        for ep in (8, 16):
            disposition, reason = capability.resolve_disposition(
                "mi355x", "mori", ep=ep, nodes=ep // 8,
                precision_profile=identity.V1_CONTROL_PRECISION_PROFILE,
            )
            self.assertEqual(disposition, "unsupported")
            self.assertEqual(
                reason,
                f"Pinned MoRI backend construction does not complete on MI355X EP{ep}",
            )

        for sku, product in (("b200-dgxc", "B200"), ("h200-dgxc", "H200")):
            disposition, reason = capability.resolve_disposition(
                sku, "deepep-v2", ep=16, nodes=2,
                precision_profile=identity.V1_CONTROL_PRECISION_PROFILE,
            )
            self.assertEqual(disposition, "unsupported")
            self.assertEqual(
                reason,
                f"DeepEP V2 EP16 exhausts pinned NCCL Gin QP resources on the {product} fabric",
            )

    def test_split_suites_track_provisional_state_and_do_not_duplicate_bf16(self) -> None:
        suites = sweep_matrix._load("suites.yaml")
        workloads = sweep_matrix._load("workloads.yaml")
        sweep_matrix.validate_config_documents(suites, workloads)
        normal = suites["suites"]["ep-precision-normal-v1"]
        low_latency = suites["suites"]["ep-precision-low-latency-v1"]
        self.assertEqual(
            (
                normal["mode"],
                normal["phases"],
                normal["token_points_decode"],
                normal["token_points_prefill"],
            ),
            ("normal", ["decode", "prefill"], [128], [512]),
        )
        self.assertEqual(
            (
                low_latency["mode"],
                low_latency["phases"],
                low_latency["token_points_decode"],
            ),
            ("low-latency", ["decode"], [128]),
        )
        self.assertTrue(normal["provisional"])
        self.assertFalse(low_latency["provisional"])
        self.assertEqual(
            normal["required_publication"], "comparable-experimental"
        )
        self.assertEqual(
            low_latency["required_publication"], "comparable-experimental"
        )
        listed = normal["precision_profiles"] + low_latency["precision_profiles"]
        self.assertNotIn(identity.V1_CONTROL_PRECISION_PROFILE, listed)
        self.assertEqual(len(listed), len(set(listed)))

        matrix = sweep_matrix.validate_matrix_document(
            sweep_matrix.resolve_matrix(backends="all")
        )
        scheduled_profiles = {
            item["case"].get("precision_profile")
            for item in matrix["requested_cases"]
            if "precision_profile" in item["case"]
        }
        self.assertEqual(scheduled_profiles, set(low_latency["precision_profiles"]))
        self.assertEqual(
            {item["precision_profile"] for item in capability.provisional_precision_targets()},
            {"d-fp8-e4m3fnuz-b128-f32-prequantized.c-bf16"},
        )

        stale = copy.deepcopy(suites)
        stale["suites"]["ep-precision-normal-v1"]["provisional"] = False
        with self.assertRaisesRegex(SystemExit, "must track unresolved"):
            sweep_matrix.validate_config_documents(stale, workloads)

    def test_resolved_profiles_schedule_without_cartesian_fill(self) -> None:
        suites = sweep_matrix._load("suites.yaml")
        workloads = sweep_matrix._load("workloads.yaml")
        promoted = copy.deepcopy(capability.PRECISION_CAPABILITIES)
        for rules in promoted.values():
            for rule in rules:
                rule["disposition"] = "supported"
        normal_profile = "d-fp8-e4m3fn-b128-f32-prequantized.c-bf16"
        promoted[normal_profile][0]["disposition"] = "unsupported"
        resolved_suites = copy.deepcopy(suites)
        for name in ("ep-precision-normal-v1", "ep-precision-low-latency-v1"):
            resolved_suites["suites"][name]["provisional"] = False

        def load_config(name: str) -> dict[str, object]:
            if name == "suites.yaml":
                return resolved_suites
            if name == "workloads.yaml":
                return workloads
            raise AssertionError(name)

        suite_names = "ep-precision-normal-v1,ep-precision-low-latency-v1"
        expected_cases = sum(
            len(resolved_suites["suites"][
                "ep-precision-normal-v1"
                if target["mode"] == "normal"
                else "ep-precision-low-latency-v1"
            ]["phases"])
            for target in capability.precision_targets()
        )
        with mock.patch.object(capability, "PRECISION_CAPABILITIES", promoted):
            sweep_matrix.validate_config_documents(resolved_suites, workloads)
            with mock.patch.object(sweep_matrix, "_load", side_effect=load_config):
                matrix = sweep_matrix.validate_matrix_document(
                    sweep_matrix.resolve_matrix(suites=suite_names, backends="all")
                )
            expected_unsupported_cases = sum(
                len(resolved_suites["suites"][
                    "ep-precision-normal-v1"
                    if target["mode"] == "normal"
                    else "ep-precision-low-latency-v1"
                ]["phases"])
                for target in capability.precision_targets()
                if capability.resolve_disposition(
                    target["sku"], target["backend"], ep=target["ep"],
                    nodes=capability.topology_for(target["sku"], target["ep"])["nodes"],
                    mode=target["mode"], precision_profile=target["precision_profile"],
                )[0] != "supported"
            )

        unsupported = [
            item for item in matrix["requested_cases"]
            if item["disposition"] == "unsupported"
        ]
        self.assertEqual(
            len(unsupported),
            expected_unsupported_cases,
        )
        self.assertTrue(all(
            item["reason"] == "precision-profile-unsupported" for item in unsupported
        ))
        self.assertTrue(any(
            item["disposition"] == "runnable" for item in matrix["requested_cases"]
        ))

        self.assertEqual(len(matrix["requested_cases"]), expected_cases)
        self.assertEqual(
            {item["case"]["precision_profile"] for item in matrix["requested_cases"]},
            set(identity.V1_NORMAL_PRECISION_PROFILE_IDS)
            | set(identity.V1_LOW_LATENCY_PRECISION_PROFILE_IDS),
        )
        self.assertFalse(any(
            item["case"]["precision_profile"] == identity.V1_CONTROL_PRECISION_PROFILE
            for item in matrix["requested_cases"]
        ))
        direct_cases = [
            item for item in matrix["requested_cases"]
            if "direct-cast" in item["case"]["precision_profile"]
        ]
        self.assertTrue(direct_cases)
        self.assertEqual({item["case"]["ep"] for item in direct_cases}, {8})

    def test_probe_runtime_diagnostics_are_closed_stage_codes(self) -> None:
        probe = (ROOT / "tests" / "probe_precision.py").read_text()
        run_ep = (ROOT / "tests" / "run_ep.py").read_text()
        for stage in (
            "distributed-init", "runtime-context", "construction-consensus",
            "operation-consensus", "evidence-aggregation",
        ):
            self.assertIn(f'"{stage}"', probe)
        self.assertIn("precision-probe-diagnostic={diagnostic_stage}-exception", probe)
        self.assertNotIn("precision-probe-diagnostic={exc}", probe)
        self.assertNotIn("cpu:gloo,cuda:nccl", probe + run_ep)
        self.assertIn('backend="nccl", rank=rank, world_size=world_size, device_id=device', probe)
        self.assertIn('backend="nccl",', run_ep)


if __name__ == "__main__":
    unittest.main()
