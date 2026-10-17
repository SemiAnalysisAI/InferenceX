"""Validate the additive source-side projection, not benchmark qualification."""

import json
from copy import deepcopy
from pathlib import Path

import jsonschema
import pytest

SCHEMA = json.loads(
    (Path(__file__).parents[1] / "dashboard-observations.schema.json").read_text()
)
VALIDATOR = jsonschema.Draft202012Validator(
    SCHEMA, format_checker=jsonschema.FormatChecker()
)


def sidecar():
    return {
        "schemaVersion": 1,
        "sourceRunId": "123",
        "sourceSha": "a" * 40,
        "cells": {
            "c1": {
                "runSha256": "b" * 64,
                "specSha256": "c" * 64,
                "deployment": {
                    "ring": 1,
                    "cfg": 1,
                    "maxBatchSize": 1,
                    "batchSize": None,
                    "scheduling": "dynamic",
                    "offload": {"ditCpu": False},
                },
                "hardwareHealth": {
                    "status": "unknown",
                    "reason": "Missing clocks",
                    "evidence": None,
                },
                "quality": {
                    "scale": "ordinal_0_to_4",
                    "contractId": None,
                    "contractSha256": None,
                    "rubricVersion": "fixture-v0",
                    "rubricSha256": None,
                    "metrics": {
                        "prompt_adherence": {
                            "status": "unjudged",
                            "direction": "higher",
                            "value": None,
                            "samples": 0,
                            "total": 20,
                            "calibration": None,
                        }
                    },
                },
            }
        },
    }


def test_nullable_unjudged_projection_and_explicit_hardware_failure_are_valid_shapes():
    record = sidecar()
    VALIDATOR.validate(record)
    failed = deepcopy(record)
    failed["cells"]["c1"]["hardwareHealth"] = {
        "status": "fail",
        "reason": "Thermal throttle",
        "evidence": "health.json",
    }
    VALIDATOR.validate(failed)
    minimal = deepcopy(record)
    minimal["cells"]["c1"] = {"runSha256": "b" * 64, "specSha256": "c" * 64}
    VALIDATOR.validate(minimal)


@pytest.mark.parametrize(
    "defect",
    [
        "unknown_root",
        "unknown_cell",
        "hash",
        "empty_cells",
        "batch_zero",
        "batch_fraction",
        "offload_string",
        "unknown_health",
        "universal_quality",
        "score_range",
        "legacy_scale",
        "missing_scale",
        "legacy_dimension",
        "fractional_samples",
        "calibration_date",
    ],
)
def test_malformed_or_invented_projection_fields_are_rejected(defect):
    record = sidecar()
    cell = record["cells"]["c1"]
    metric = cell["quality"]["metrics"]["prompt_adherence"]
    if defect == "unknown_root":
        record["releaseQualified"] = True
    elif defect == "unknown_cell":
        cell["universalQuality"] = 100
    elif defect == "hash":
        cell["runSha256"] = "not-a-source-checksum"
    elif defect == "empty_cells":
        record["cells"] = {}
    elif defect == "batch_zero":
        cell["deployment"]["maxBatchSize"] = 0
    elif defect == "batch_fraction":
        cell["deployment"]["batchSize"] = 1.5
    elif defect == "offload_string":
        cell["deployment"]["offload"]["ditCpu"] = "false"
    elif defect == "unknown_health":
        cell["hardwareHealth"]["status"] = "probably-good"
    elif defect == "universal_quality":
        cell["quality"]["metrics"]["overall"] = metric
    elif defect == "score_range":
        metric["value"] = 6
    elif defect == "legacy_scale":
        cell["quality"]["scale"] = "ordinal_1_to_5"
    elif defect == "missing_scale":
        del cell["quality"]["scale"]
    elif defect == "legacy_dimension":
        cell["quality"]["metrics"] = {"subject_consistency": metric}
    elif defect == "fractional_samples":
        metric["samples"] = 1.5
    else:
        metric["calibration"] = {"status": "calibrated", "frozenAt": "not-a-date"}
    with pytest.raises(jsonschema.ValidationError):
        VALIDATOR.validate(record)


def test_original_seven_dimension_projection_keeps_zero_scores():
    record = sidecar()
    quality = record["cells"]["c1"]["quality"]
    measurement = quality["metrics"]["prompt_adherence"]
    measurement.update(value=0, status="judged_unqualified", samples=20)
    quality["metrics"] = {
        name: deepcopy(measurement)
        for name in (
            "prompt_adherence",
            "visual_fidelity",
            "temporal_consistency",
            "motion_plausibility",
            "audio_quality",
            "audio_content",
            "av_sync",
        )
    }
    VALIDATOR.validate(record)
