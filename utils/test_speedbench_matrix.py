"""Unit tests for infx.workflows.speedbench_matrix."""

import json

import pytest

from infx.workflows.speedbench_matrix import aggregate_cells


def test_aggregate_cells_reads_per_cell_jsons(tmp_path):
    """Happy-path: all cells present produce the expected YAML matrix."""
    for mode in ("off", "on"):
        for mtp in (1, 2):
            al = 2.50 if mode == "off" else 3.10
            path = tmp_path / f"speedbench_{mode}_mtp{mtp}.json"
            path.write_text(json.dumps({"al": al + mtp * 0.1}))
    result = aggregate_cells(tmp_path, "dsv4", ["off", "on"], [1, 2], ["Test header"])
    assert "# Test header\n" in result
    assert "dsv4:\n" in result
    assert "  thinking_off:\n" in result
    assert "    1: 2.60\n" in result
    assert "    2: 2.70\n" in result
    assert "  thinking_on:\n" in result
    assert "    1: 3.20\n" in result
    assert "    2: 3.30\n" in result


def test_aggregate_cells_missing_cell_produces_na(tmp_path):
    """Missing or unreadable result JSONs produce N/A."""
    (tmp_path / "speedbench_on_mtp1.json").write_text(json.dumps({"al": 3.14}))
    result = aggregate_cells(tmp_path, "dsr1", ["on"], [1, 2], [])
    assert "    1: 3.14\n" in result
    assert "    2: N/A\n" in result


def test_aggregate_cells_na_value_passthrough(tmp_path):
    """A cell whose AL is already 'N/A' stays N/A."""
    (tmp_path / "speedbench_off_mtp1.json").write_text(json.dumps({"al": "N/A"}))
    result = aggregate_cells(tmp_path, "glm5", ["off"], [1], [])
    assert "    1: N/A\n" in result


def test_aggregate_cells_malformed_json_produces_na(tmp_path):
    """Corrupt JSON produces N/A without crashing."""
    (tmp_path / "speedbench_off_mtp1.json").write_text("not json")
    result = aggregate_cells(tmp_path, "glm52", ["off"], [1], [])
    assert "    1: N/A\n" in result
