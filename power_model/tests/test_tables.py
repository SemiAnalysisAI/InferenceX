# SPDX-License-Identifier: GPL-3.0-only
import json

import pytest
from pydantic import ValidationError

from power_model.tables import TableSet, read_table


def test_unresolved_references_fail_at_load():
    named = {"version": "1", "kind": "assumed", "source": "s", "input_boundary": "b"}
    labeled = {"name": "GPU", "provenance": "known"}
    raw = {
        "provenance": {"known": {"profile_id": "known", **named}},
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
        "inputs": {
            key: labeled
            for key in ("gpus", "x86-cpu", "x86-cpu-and-dram", "grace-socket", "grace-cpu")
        },
        "systems": {},
    }

    with pytest.raises(ValidationError, match="provenance unknown.*drive detail variable drive_w"):
        TableSet.model_validate_json(json.dumps(raw))


def test_duplicate_yaml_keys_fail_but_merge_overrides_load(tmp_path):
    table = tmp_path / "parts.yaml"
    table.write_text("base: &base {power-w: 5.0}\nnvme-drive:\n  <<: *base\n  power-w: 6.0\n")
    assert read_table(table)["nvme-drive"] == {"power-w": 6.0}

    table.write_text("nvme-drive:\n  power-w: 5.0\nnvme-drive:\n  power-w: 6.0\n")
    with pytest.raises(ValueError, match="Duplicate key nvme-drive"):
        read_table(table)
