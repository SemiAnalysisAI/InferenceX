# SPDX-License-Identifier: GPL-3.0-only
import json

import pytest
from pydantic import ValidationError

from power_model.tables import TableSet


def test_unresolved_references_fail_at_load():
    named = {"version": "1", "kind": "assumed", "source": "s", "input-boundary": "b"}
    labeled = {"name": "GPU", "provenance": "known"}
    raw = {
        "provenance": {"known": {"profile-id": "known", **named}},
        "parts": {
            "drive": {
                "name": "NVMeDrive",
                "category": "storage",
                "provenance": "unknown",
                "power-w": 5.0,
                "details": {"per_drive_power_w": "$drive_w"},
            }
        },
        "network": {
            "switch": {
                "name": "Switch",
                "provenance": "known",
                "parts": [{"part": "drive", "count": 1}],
            }
        },
        "curves": {
            "psu-efficiency": {},
            "dc-converters": {},
            "hgx-fans": {},
            "rack-fans": {},
            "lpddr5x": {},
            "grace-workload": {},
        },
        "assemblies": {"trays": {}, "power-shelves": {}},
        "facility": {
            "pue": {"provenance": "known", "air": 1.3, "liquid": 1.1},
            "example": {"non-gpu-ratio": 0.5, "gpu": labeled, "non-gpu": labeled},
        },
        "systems": {},
    }

    with pytest.raises(ValidationError, match="provenance unknown.*drive detail variable drive_w"):
        TableSet.model_validate_json(json.dumps(raw))
