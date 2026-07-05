#!/usr/bin/env python3
"""Focused structural tests for the fail-closed CollectiveX V1 schemas."""
from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import identity  # noqa: E402


def _load(name: str) -> dict:
    return json.loads((ROOT / "schemas" / name).read_text())


def _definition_validator(schema: dict, name: str) -> jsonschema.Validator:
    return jsonschema.Draft202012Validator(
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$defs": schema["$defs"],
            "$ref": f"#/$defs/{name}",
        }
    )


class CollectiveXV1SchemaContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = _load("raw-case-v1.schema.json")
        cls.samples = _load("samples-v1.schema.json")
        cls.public = _load("public-dataset-v1.schema.json")
        cls.bundle = _load("private-bundle-v1.schema.json")
        cls.terminal = _load("terminal-outcome-v1.schema.json")

    def test_all_checked_in_schemas_are_draft_2020_12_valid(self) -> None:
        for path in sorted((ROOT / "schemas").glob("*.schema.json")):
            with self.subTest(path=path.name):
                schema = json.loads(path.read_text())
                jsonschema.Draft202012Validator.check_schema(schema)
                self.assertFalse(schema["additionalProperties"])

    def test_precision_catalog_and_axes_are_exact_and_strict(self) -> None:
        expected = set(identity.V1_PRECISION_PROFILES)
        self.assertEqual(set(self.raw["$defs"]["precision_profile"]["enum"]), expected)
        self.assertEqual(set(self.public["$defs"]["precisionProfile"]["enum"]), expected)
        self.assertEqual(set(self.terminal["$defs"]["precisionProfile"]["enum"]), expected)

        axis_keys = {
            "alignment_contract",
            "api_input_dtype",
            "api_output_dtype",
            "communication_format",
            "conversion_boundary",
            "padding_contract",
            "quantization_origin",
            "scale_dtype",
            "scale_group_size",
            "scale_layout",
        }
        for schema, name in (
            (self.raw, "communication_axis"),
            (self.public, "communicationAxis"),
            (self.terminal, "communicationAxis"),
        ):
            with self.subTest(schema=schema["$id"]):
                axis = schema["$defs"][name]
                self.assertFalse(axis["additionalProperties"])
                self.assertEqual(set(axis["required"]), axis_keys)
                self.assertEqual(set(axis["properties"]), axis_keys)

        axis_validator = _definition_validator(self.raw, "communication_axis")
        profile_validator = _definition_validator(self.raw, "communication_precision")
        for name in sorted(expected):
            profile = identity.precision_profile(name)
            with self.subTest(profile=name):
                axis_validator.validate(profile["dispatch"])
                axis_validator.validate(profile["combine"])
                profile_validator.validate(profile)
        raw_case_profile = _definition_validator(self.raw, "case_profile")
        terminal_case_profile = _definition_validator(self.terminal, "caseProfile")
        for case in (
            {"mode": "normal"},
            {
                "mode": "normal",
                "precision_profile": "d-fp8-e4m3fn-b128-f32-prequantized.c-bf16",
            },
            {
                "mode": "low-latency",
                "precision_profile": "d-fp8-e4m3fn-b128-f32-fused.c-bf16",
            },
        ):
            resolved = identity.profile_for_case(case)
            raw_case_profile.validate(resolved)
            terminal_case_profile.validate(resolved)

        shape = self.raw["properties"]["case"]["properties"]["shape"]
        self.assertIn("precision_profile", shape["required"])
        self.assertIn("dispatch_precision", shape["required"])
        self.assertIn("combine_precision", shape["required"])
        self.assertNotIn("dispatch_dtype", shape["properties"])
        self.assertNotIn("quant", shape["properties"])

        workload = self.public["$defs"]["series"]["properties"]["workload"]
        self.assertIn("precision_profile", workload["required"])
        self.assertIn("dispatch_precision", workload["required"])
        self.assertIn("combine_precision", workload["required"])
        self.assertNotIn("dispatch_dtype", workload["properties"])
        self.assertNotIn("combine_dtype", workload["properties"])

        profile = self.raw["$defs"]["case_profile"]
        self.assertEqual(
            profile["properties"]["activation_generator"]["const"],
            identity.V1_NORMAL_CASE_PROFILE["activation_generator"],
        )
        self.assertEqual(
            profile["properties"]["activation_profile"]["const"],
            identity.V1_NORMAL_CASE_PROFILE["activation_profile"],
        )
        self.assertEqual(
            profile["properties"]["source_identity_contract"]["const"],
            identity.V1_NORMAL_CASE_PROFILE["source_identity_contract"],
        )

    def test_qualification_index_is_bound_across_private_and_public_records(self) -> None:
        paths = (
            self.raw["properties"]["measurement"]["properties"]["qualification_index"],
            self.raw["properties"]["identity"]["properties"]["allocation_factors"]["properties"]["qualification_index"],
            self.raw["$defs"]["git_run"]["properties"]["qualification_index"],
            self.samples["properties"]["qualification_index"],
            self.bundle["properties"]["run"]["properties"]["qualification_index"],
            self.terminal["$defs"]["allocationFactors"]["properties"]["qualification_index"],
            self.terminal["$defs"]["gitRun"]["properties"]["qualification_index"],
            self.public["$defs"]["attempt"]["properties"]["qualification_index"],
        )
        for value in paths:
            self.assertEqual((value["minimum"], value["maximum"]), (1, 3))
        promotion_indices = self.public["properties"]["promotion"]["properties"]["qualification_indices"]
        series_indices = self.public["$defs"]["series"]["properties"]["measurement"]["properties"]["qualification_indices"]
        for schema, valid_values in (
            (promotion_indices, ([], [1], [1, 2, 3])),
            (series_indices, ([1], [2, 3], [1, 2, 3])),
        ):
            validator = jsonschema.Draft202012Validator(schema)
            for value in valid_values:
                validator.validate(value)
            for value in ([0], [4], [1, 1], [1, 2, 3, 1]):
                with self.assertRaises(jsonschema.ValidationError):
                    validator.validate(value)
        measurement = self.raw["properties"]["measurement"]
        self.assertIn("execution_order_sha256", measurement["required"])
        self.assertEqual(
            measurement["properties"]["execution_order_sha256"]["pattern"],
            "^[0-9a-f]{64}$",
        )

    def test_private_allocation_stratum_is_required_only_in_raw_canonical_evidence(self) -> None:
        provenance = self.raw["properties"]["provenance"]
        self.assertIn("allocation_stratum_sha256", provenance["required"])
        stratum = provenance["properties"]["allocation_stratum_sha256"]
        jsonschema.Draft202012Validator(stratum).validate(None)
        jsonschema.Draft202012Validator(stratum).validate("a" * 64)
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.Draft202012Validator(stratum).validate("A" * 64)
        conditional = self.raw["allOf"][0]
        self.assertEqual(
            conditional["if"]["properties"]["workload"]["properties"]["source"],
            {"const": "canonical-serialized"},
        )
        canonical_stratum = conditional["then"]["properties"]["provenance"][
            "properties"
        ]["allocation_stratum_sha256"]
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.Draft202012Validator(canonical_stratum).validate(None)
        self.assertNotIn("allocation_stratum_sha256", json.dumps(self.public))

    def test_stage_samples_are_absent_or_exactly_512(self) -> None:
        validator = _definition_validator(self.samples, "component")
        measured = {
            "availability": "measured",
            "sample_count": 512,
            "trials": [[1.0] * 8 for _ in range(64)],
        }
        validator.validate(measured)
        unavailable = {
            "availability": "unavailable",
            "sample_count": 0,
            "trials": None,
        }
        validator.validate(unavailable)
        for mutate in (
            lambda value: value.update(sample_count=511),
            lambda value: value["trials"].pop(),
            lambda value: value["trials"][0].pop(),
        ):
            broken = copy.deepcopy(measured)
            mutate(broken)
            with self.assertRaises(jsonschema.ValidationError):
                validator.validate(broken)

        raw_components = self.raw["properties"]["measurement"]["properties"]["rows"]["items"]["properties"]["components"]
        public_components = self.public["$defs"]["point"]["properties"]["components"]
        self.assertIn("stage", raw_components["required"])
        self.assertIn("stage", public_components["required"])

    def test_byte_provenance_supports_precision_aware_rates(self) -> None:
        expected = {
            "accounting_contract",
            "activation_data_bytes",
            "scale_bytes",
            "total_logical_bytes",
        }
        for schema, name in (
            (self.raw, "byte_accounting"),
            (self.public, "byteAccounting"),
        ):
            definition = schema["$defs"][name]
            self.assertFalse(definition["additionalProperties"])
            self.assertEqual(set(definition["required"]), expected)
            self.assertEqual(set(definition["properties"]), expected)
        component = self.public["$defs"]["component"]
        self.assertIn("byte_provenance", component["required"])
        self.assertIn("activation_data_rate_gbps_at_latency_percentile", component["required"])
        self.assertIn("total_logical_data_rate_gbps_at_latency_percentile", component["required"])

    def test_raw_correctness_carries_directional_precision_evidence(self) -> None:
        correctness = self.raw["properties"]["measurement"]["properties"]["rows"]["items"]["properties"]["correctness"]
        self.assertIn("precision", correctness["required"])
        evidence = self.raw["$defs"]["precision_evidence"]
        self.assertFalse(evidence["additionalProperties"])
        self.assertEqual(
            set(evidence["required"]),
            {"profile_id", "dispatch", "combine", "passed"},
        )
        axis = self.raw["$defs"]["precision_axis_evidence"]
        self.assertFalse(axis["additionalProperties"])
        self.assertEqual(
            set(axis["required"]),
            {
                "encoded_payload_valid",
                "scales_finite",
                "scales_positive",
                "dequantized_semantics",
                "saturation_count",
                "saturation_rate",
                "max_abs_error",
                "max_rel_error",
                "passed",
            },
        )

    def test_eplb_calibration_provenance_is_explicit(self) -> None:
        fields = {
            "calibration_workload_id",
            "calibration_trace_sha256",
            "calibration_window",
            "calibration_token_offset",
        }
        raw = self.raw["properties"]["case"]["properties"]["eplb"]
        public = self.public["$defs"]["series"]["properties"]["eplb"]
        for descriptor in (raw, public):
            self.assertTrue(fields <= set(descriptor["required"]))
            self.assertTrue(fields <= set(descriptor["properties"]))
            self.assertFalse(descriptor["additionalProperties"])

    def test_public_coverage_is_a_full_case_and_point_inventory(self) -> None:
        coverage = self.public["$defs"]["coverage"]
        dimensions = {
            "sku",
            "suite",
            "workload",
            "publication_tier",
            "backend",
            "backend_generation",
            "resource",
            "topology",
            "phase",
            "mode",
            "routing",
            "eplb",
            "precision_profile",
            "dispatch_precision",
            "combine_precision",
            "points",
        }
        self.assertTrue(dimensions <= set(coverage["required"]))
        self.assertFalse(coverage["additionalProperties"])
        point = self.public["$defs"]["coveragePoint"]
        self.assertEqual(
            set(point["properties"]["terminal_status"]["enum"]),
            {"measured", "unsupported", "failed", "invalid", "diagnostic"},
        )
        self.assertIn("tokens_per_rank", point["required"])
        self.assertIn("global_tokens", point["required"])
        self.assertIn("reason", point["required"])

        promotion = self.public["properties"]["promotion"]
        counts = {
            "requested_cases",
            "terminal_cases",
            "measured_cases",
            "unsupported_cases",
            "requested_points",
            "terminal_points",
            "measured_points",
            "unsupported_points",
        }
        self.assertTrue(counts <= set(promotion["required"]))
        self.assertFalse(promotion["additionalProperties"])

    def test_public_coverage_point_reason_tracks_terminal_status(self) -> None:
        validator = _definition_validator(self.public, "coveragePoint")
        point_id = "cxpoint-v1-" + "1" * 64
        series_id = "cxseries-v1-" + "2" * 64
        measured = {
            "point_id": point_id,
            "series_id": series_id,
            "tokens_per_rank": 8,
            "global_tokens": 64,
            "terminal_status": "measured",
            "reason": None,
        }
        validator.validate(measured)
        unsupported = {
            **measured,
            "point_id": None,
            "series_id": None,
            "terminal_status": "unsupported",
            "reason": "backend-platform-unsupported",
        }
        validator.validate(unsupported)
        failed = {**unsupported, "terminal_status": "failed", "reason": "execution-failed"}
        validator.validate(failed)

        for broken in (
            {**measured, "reason": "unexpected-reason"},
            {**unsupported, "reason": None},
            {**failed, "reason": None},
            {**unsupported, "reason": "contains spaces"},
        ):
            with self.assertRaises(jsonschema.ValidationError):
                validator.validate(broken)

    def test_public_measured_point_has_bounded_detail_and_three_run_stability(self) -> None:
        point = self.public["$defs"]["point"]
        self.assertFalse(point["additionalProperties"])
        self.assertTrue({"anomalies", "correctness", "stability"} <= set(point["required"]))
        self.assertNotIn("correct", point["properties"])

        axis = {
            "encoded_payload_valid": True,
            "scales_finite": None,
            "scales_positive": None,
            "dequantized_semantics": True,
            "saturation_count": 0,
            "saturation_rate": 0.0,
            "max_abs_error": 0.0,
            "max_rel_error": 0.0,
            "passed": True,
        }
        correctness = {
            "semantic_pass": True,
            "precision": {
                "profile_id": identity.V1_CONTROL_PRECISION_PROFILE,
                "dispatch": axis,
                "combine": copy.deepcopy(axis),
                "passed": True,
            },
        }
        correctness_validator = _definition_validator(self.public, "pointCorrectness")
        correctness_validator.validate(correctness)
        broken_correctness = copy.deepcopy(correctness)
        broken_correctness["precision"]["dispatch"]["unexpected"] = True
        with self.assertRaises(jsonschema.ValidationError):
            correctness_validator.validate(broken_correctness)

        stability_validator = _definition_validator(self.public, "pointStability")
        stability_validator.validate({
            "complete": True,
            "qualification_indices": [1, 2, 3],
            "p50_max_min_ratio": 1.02,
            "p99_max_min_ratio": 1.04,
            "stable_p50": True,
            "stable_p99": True,
        })
        stability_validator.validate({
            "complete": False,
            "qualification_indices": [2],
            "p50_max_min_ratio": None,
            "p99_max_min_ratio": None,
            "stable_p50": False,
            "stable_p99": False,
        })
        for broken in (
            {
                "complete": True,
                "qualification_indices": [1, 2],
                "p50_max_min_ratio": 1.0,
                "p99_max_min_ratio": 1.0,
                "stable_p50": True,
                "stable_p99": True,
            },
            {
                "complete": False,
                "qualification_indices": [1],
                "p50_max_min_ratio": 1.0,
                "p99_max_min_ratio": None,
                "stable_p50": False,
                "stable_p99": False,
            },
        ):
            with self.assertRaises(jsonschema.ValidationError):
                stability_validator.validate(broken)

        anomalies = point["properties"]["anomalies"]
        self.assertEqual(anomalies["maxItems"], 16)
        self.assertTrue(anomalies["uniqueItems"])
        anomaly_validator = jsonschema.Draft202012Validator({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$defs": self.public["$defs"],
            "$ref": "#/$defs/point/properties/anomalies",
        })
        anomaly_validator.validate(["roundtrip-gt-isolated-sum"])
        with self.assertRaises(jsonschema.ValidationError):
            anomaly_validator.validate([f"anomaly-{index}" for index in range(17)])


if __name__ == "__main__":
    unittest.main()
