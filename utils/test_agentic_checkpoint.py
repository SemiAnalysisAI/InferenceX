"""Tests for per-concurrency AgentX checkpoint staging and restore."""

from __future__ import annotations

import json
from pathlib import Path

from agentic_checkpoint import (
    checkpoint_paths,
    restore_checkpoint,
    stage_checkpoint,
)


BASE = "dsv4_p2x8_d1x8_conc2x4x8"


def write_result(path: Path, *, concurrency: int = 4, successful: int = 12) -> None:
    path.write_text(
        json.dumps({"conc": concurrency, "num_requests_successful": successful})
    )


def test_stage_and_restore_valid_checkpoint(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    checkpoint_dir = tmp_path / "checkpoints"
    output_dir = tmp_path / "output"
    write_result(source)

    stage_checkpoint(source, checkpoint_dir, BASE, 4)

    assert restore_checkpoint(checkpoint_dir, output_dir, BASE, 4)
    restored = json.loads((output_dir / f"{BASE}_conc4.json").read_text())
    assert restored["num_requests_successful"] == 12


def test_restore_requires_success_marker(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    result, _ = checkpoint_paths(checkpoint_dir, BASE, 4)
    result.parent.mkdir()
    write_result(result)

    assert not restore_checkpoint(checkpoint_dir, tmp_path / "output", BASE, 4)


def test_restore_rejects_mismatched_marker(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    checkpoint_dir = tmp_path / "checkpoints"
    write_result(source)
    stage_checkpoint(source, checkpoint_dir, BASE, 4)
    _, marker = checkpoint_paths(checkpoint_dir, BASE, 4)
    marker.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "base_result_filename": BASE,
                "result_filename": f"{BASE}_conc4.json",
                "concurrency": 8,
            }
        )
    )

    assert not restore_checkpoint(checkpoint_dir, tmp_path / "output", BASE, 4)


def test_stage_rejects_unsuccessful_result(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    write_result(source, successful=0)

    try:
        stage_checkpoint(source, tmp_path / "checkpoints", BASE, 4)
    except ValueError as error:
        assert "no successful requests" in str(error)
    else:
        raise AssertionError("unsuccessful result was checkpointed")
