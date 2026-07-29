"""Tests for draft-aware golden acceptance-length validation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = REPO_ROOT / "utils" / "validate_golden_al.py"


def _write_curve(
    directory: Path,
    filename: str,
    *,
    draft_model: str | None,
    level_3_al: float,
) -> None:
    metadata = (
        f"_metadata:\n  draft-model: {draft_model}\n" if draft_model is not None else ""
    )
    (directory / filename).write_text(
        (
            f"{metadata}"
            "minimax-m3:\n"
            "  thinking_on:\n"
            f"    3: {level_3_al}\n"
            "  thinking_off:\n"
            "    3: 2.0\n"
        ),
        encoding="utf-8",
    )


def _run_validator(directory: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--directory",
            str(directory),
            "--model",
            "minimax-m3",
            "--thinking-mode",
            "thinking_on",
            "--num-speculative-tokens",
            "3",
            *args,
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    ("draft_model", "expected_al"),
    [
        ("Inferact/MiniMax-M3-EAGLE3", "2.83"),
        ("Inferact/MiniMax-M3-EAGLE3-GQA", "2.78"),
    ],
)
def test_resolves_exact_draft_model(
    tmp_path: Path, draft_model: str, expected_al: str
) -> None:
    _write_curve(
        tmp_path,
        "minimaxm3_eagle3.yaml",
        draft_model="Inferact/MiniMax-M3-EAGLE3",
        level_3_al=2.83,
    )
    _write_curve(
        tmp_path,
        "minimaxm3_eagle3_gqa.yaml",
        draft_model="Inferact/MiniMax-M3-EAGLE3-GQA",
        level_3_al=2.78,
    )

    result = _run_validator(tmp_path, "--draft-model", draft_model)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected_al


def test_rejects_ambiguous_model_without_draft_model(tmp_path: Path) -> None:
    _write_curve(
        tmp_path,
        "minimaxm3_eagle3.yaml",
        draft_model="Inferact/MiniMax-M3-EAGLE3",
        level_3_al=2.83,
    )
    _write_curve(
        tmp_path,
        "minimaxm3_eagle3_gqa.yaml",
        draft_model="Inferact/MiniMax-M3-EAGLE3-GQA",
        level_3_al=2.78,
    )

    result = _run_validator(tmp_path)

    assert result.returncode != 0
    assert "ambiguous" in result.stderr.lower()
    assert "Inferact/MiniMax-M3-EAGLE3-GQA" in result.stderr


def test_rejects_unknown_draft_model_and_lists_available_curves(tmp_path: Path) -> None:
    _write_curve(
        tmp_path,
        "minimaxm3_eagle3.yaml",
        draft_model="Inferact/MiniMax-M3-EAGLE3",
        level_3_al=2.83,
    )
    _write_curve(
        tmp_path,
        "minimaxm3_eagle3_gqa.yaml",
        draft_model="Inferact/MiniMax-M3-EAGLE3-GQA",
        level_3_al=2.78,
    )

    result = _run_validator(
        tmp_path, "--draft-model", "Inferact/MiniMax-M3-EAGLE3-UNKNOWN"
    )

    assert result.returncode != 0
    assert "no golden AL curve" in result.stderr
    assert "Inferact/MiniMax-M3-EAGLE3" in result.stderr
    assert "Inferact/MiniMax-M3-EAGLE3-GQA" in result.stderr


def test_rejects_duplicate_draft_model_metadata(tmp_path: Path) -> None:
    draft_model = "Inferact/MiniMax-M3-EAGLE3"
    _write_curve(
        tmp_path,
        "first.yaml",
        draft_model=draft_model,
        level_3_al=2.83,
    )
    _write_curve(
        tmp_path,
        "second.yaml",
        draft_model=draft_model,
        level_3_al=2.84,
    )

    result = _run_validator(tmp_path, "--draft-model", draft_model)

    assert result.returncode != 0
    assert "duplicate" in result.stderr.lower()
    assert "first.yaml" in result.stderr
    assert "second.yaml" in result.stderr


def test_rejects_pinned_acceptance_length_mismatch(tmp_path: Path) -> None:
    draft_model = "Inferact/MiniMax-M3-EAGLE3-GQA"
    _write_curve(
        tmp_path,
        "minimaxm3_eagle3_gqa.yaml",
        draft_model=draft_model,
        level_3_al=2.78,
    )

    result = _run_validator(
        tmp_path,
        "--draft-model",
        draft_model,
        "--expected-al",
        "2.83",
    )

    assert result.returncode != 0
    assert "does not match" in result.stderr
    assert "expected 2.83" in result.stderr
    assert "golden 2.78" in result.stderr
