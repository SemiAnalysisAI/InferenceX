import json
import subprocess
import sys
from pathlib import Path

from infx.results.collect_results import keep_latest_attempts

REPO_ROOT = Path(__file__).resolve().parents[3]


def row(conc: int, attempt: int | None, value: float, fingerprint: str = "a" * 64) -> dict:
    result = {
        "hw": "gb200",
        "framework": "dynamo-vllm",
        "model": "kimik3",
        "precision": "fp4",
        "scenario_type": "agentic-coding",
        "conc": conc,
        "recipe_fingerprint": fingerprint,
        "avg_power_w": value,
    }
    if attempt is not None:
        result["run_attempt"] = attempt
    return result


def test_keeps_the_latest_attempt_of_each_point():
    rows = [row(48, 1, 1.0), row(48, 3, 3.0), row(48, 2, 2.0), row(56, 1, 5.0)]
    assert keep_latest_attempts(rows) == [row(48, 3, 3.0), row(56, 1, 5.0)]


def test_keeps_rows_without_attempt_or_fingerprint():
    rows = [row(48, None, 1.0), row(48, None, 2.0), row(48, 2, 3.0, fingerprint="")]
    assert keep_latest_attempts(rows) == rows


def test_cli_writes_one_row_per_point(tmp_path: Path):
    results = tmp_path / "results"
    for name, result in {"a": row(48, 1, 1.0), "b": row(48, 2, 2.0)}.items():
        (results / name).mkdir(parents=True)
        (results / name / "r.json").write_text(json.dumps(result))
    proc = subprocess.run(
        [sys.executable, "-m", "infx.results.collect_results", str(results), "bmk"],
        cwd=tmp_path,
        env={"PYTHONPATH": str(REPO_ROOT)},
        capture_output=True,
        text=True,
        check=True,
    )
    assert "superseded by a later run attempt" in proc.stderr
    assert json.loads((tmp_path / "agg_bmk.json").read_text()) == [row(48, 2, 2.0)]
