import json
from pathlib import Path

from infx.results.artifacts import agentic_point_files

ROW = {"scenario_type": "agentic-coding", "hw": "gb200", "conc": 48, "recipe_fingerprint": "a" * 64}


def write(root: Path, name: str, row: dict) -> Path:
    path = root / f"bmk_agentic_{name}" / "point_conc48.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(row))
    return path


def test_keeps_only_the_latest_attempt_of_a_rerun_point(tmp_path: Path):
    write(tmp_path, "a1_runner", {**ROW, "run_attempt": 1})
    latest = write(tmp_path, "a2_runner", {**ROW, "run_attempt": 2})
    other = write(tmp_path, "a1_other", {**ROW, "conc": 56, "run_attempt": 1})
    assert agentic_point_files(tmp_path) == sorted([latest, other])


def test_keeps_files_without_run_attempt(tmp_path: Path):
    first = write(tmp_path, "old_runner", ROW)
    second = write(tmp_path, "new_runner", {**ROW, "run_attempt": 2})
    assert agentic_point_files(tmp_path) == sorted([first, second])


def test_rerun_does_not_supersede_a_point_with_other_spec_decoding(tmp_path: Path):
    mtp = write(tmp_path, "mtp_a1_runner", {**ROW, "spec_decoding": "mtp", "run_attempt": 1})
    plain = write(tmp_path, "none_a2_runner", {**ROW, "spec_decoding": "none", "run_attempt": 2})
    assert agentic_point_files(tmp_path) == sorted([mtp, plain])
