#!/usr/bin/env python3
"""Focused checks for the neutral JSON schemas and strict JSON loader."""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import contracts  # noqa: E402


def schema(name):
    return json.loads((ROOT / "schemas" / name).read_text())


class ContractTests(unittest.TestCase):
    def test_only_neutral_schemas_exist_and_are_valid(self):
        paths = sorted((ROOT / "schemas").glob("*.schema.json"))
        self.assertEqual(
            {path.name for path in paths},
            {"raw-case-v1.schema.json", "samples-v1.schema.json", "terminal-outcome-v1.schema.json"},
        )
        for path in paths:
            document = json.loads(path.read_text())
            jsonschema.Draft202012Validator.check_schema(document)
            self.assertFalse(document["additionalProperties"])

    def test_raw_and_terminal_share_the_scheduled_case_contract(self):
        raw = schema("raw-case-v1.schema.json")["$defs"]["scheduled_case"]
        terminal = schema("terminal-outcome-v1.schema.json")["$defs"]["case"]
        self.assertEqual(set(raw["properties"]), set(terminal["properties"]))
        self.assertEqual(set(raw["required"]), set(terminal["required"]))
        self.assertFalse(raw["additionalProperties"])
        self.assertFalse(terminal["additionalProperties"])
        self.assertNotIn("precision_profile", raw["properties"])

    def test_sample_component_is_exactly_512_or_unavailable(self):
        samples = schema("samples-v1.schema.json")
        validator = jsonschema.Draft202012Validator({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$defs": samples["$defs"], "$ref": "#/$defs/component",
        })
        measured = {
            "availability": "measured", "sample_count": 512,
            "trials": [[1.0] * 8 for _ in range(64)],
        }
        validator.validate(measured)
        for broken in (
            {**measured, "sample_count": 511},
            {**measured, "trials": measured["trials"][:-1]},
        ):
            with self.assertRaises(jsonschema.ValidationError):
                validator.validate(broken)
        validator.validate({"availability": "unavailable", "sample_count": 0, "trials": None})

    def test_strict_loader_rejects_duplicate_keys_and_nonfinite_numbers(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.json"
            for payload in ('{"x":1,"x":2}', '{"x":NaN}'):
                path.write_text(payload)
                with self.assertRaises(contracts.ContractError):
                    contracts.strict_json_load(path)


if __name__ == "__main__":
    unittest.main()
