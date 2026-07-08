#!/usr/bin/env python3
"""Focused structural tests for the fail-closed CollectiveX neutral schemas."""
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


# The scheduled case is the neutral contract every artifact agrees on. Both the
# raw evidence schema and the terminal-outcome schema must enforce this exact set
# of 24 fields with additionalProperties:false so a producer cannot invent,
# rename, or drop a scheduling field on either side of a run.
SCHEDULED_CASE_FIELDS = frozenset({
    "backend", "canonical", "ep", "eplb", "experts", "gpus_per_node", "hidden",
    "ladder", "mode", "nodes", "phase", "routing", "samples_per_point",
    "scale_out_transport", "scale_up_domain", "scale_up_transport", "scope",
    "suite", "timing", "topk", "topology_class", "transport", "warmup_semantics",
    "workload",
})


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


class CollectiveXSchemaContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = _load("raw-case-v1.schema.json")
        cls.samples = _load("samples-v1.schema.json")
        cls.terminal = _load("terminal-outcome-v1.schema.json")

    def test_all_checked_in_schemas_are_draft_2020_12_valid(self) -> None:
        schemas = sorted((ROOT / "schemas").glob("*.schema.json"))
        # The publication schemas are gone; only the three neutral ones remain.
        self.assertEqual(
            {path.name for path in schemas},
            {
                "raw-case-v1.schema.json",
                "samples-v1.schema.json",
                "terminal-outcome-v1.schema.json",
            },
        )
        for path in schemas:
            with self.subTest(path=path.name):
                schema = json.loads(path.read_text())
                jsonschema.Draft202012Validator.check_schema(schema)
                self.assertFalse(schema["additionalProperties"])

    def test_scheduled_case_is_identical_across_raw_and_terminal(self) -> None:
        # Format consts pin the record type of each neutral artifact.
        self.assertEqual(
            self.raw["properties"]["format"]["const"], "collectivex.ep.v1"
        )
        self.assertEqual(
            self.terminal["properties"]["format"]["const"], "collectivex.terminal.v1"
        )
        self.assertEqual(
            self.samples["properties"]["format"]["const"], "collectivex.samples.v1"
        )

        raw_case = self.raw["$defs"]["scheduled_case"]
        terminal_case = self.terminal["$defs"]["case"]
        for descriptor in (raw_case, terminal_case):
            self.assertFalse(descriptor["additionalProperties"])
            self.assertEqual(set(descriptor["properties"]), SCHEDULED_CASE_FIELDS)
            self.assertEqual(set(descriptor["required"]), SCHEDULED_CASE_FIELDS)
        # No precision dimension survives on the scheduled case.
        self.assertNotIn("precision_profile", raw_case["properties"])
        self.assertNotIn("precision_profile", terminal_case["properties"])

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

        raw_components = self.raw["properties"]["measurement"]["properties"]["rows"][
            "items"
        ]["properties"]["components"]
        self.assertIn("stage", raw_components["required"])

    def test_byte_accounting_is_exact_and_strict(self) -> None:
        expected = {
            "accounting_contract",
            "activation_data_bytes",
            "scale_bytes",
            "total_logical_bytes",
        }
        definition = self.raw["$defs"]["byte_accounting"]
        self.assertFalse(definition["additionalProperties"])
        self.assertEqual(set(definition["required"]), expected)
        self.assertEqual(set(definition["properties"]), expected)

    def test_eplb_calibration_provenance_is_explicit(self) -> None:
        fields = {
            "calibration_workload_id",
            "calibration_trace_sha256",
            "calibration_window",
            "calibration_token_offset",
        }
        descriptor = self.raw["properties"]["case"]["properties"]["eplb"]
        self.assertTrue(fields <= set(descriptor["required"]))
        self.assertTrue(fields <= set(descriptor["properties"]))
        self.assertFalse(descriptor["additionalProperties"])


if __name__ == "__main__":
    unittest.main()
