from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

E2E_ROOT = Path(__file__).resolve().parents[4]


def _record(isl: int, osl: int, *, phase: str = "profiling", error: dict | None = None) -> dict:
    return {
        "metadata": {"benchmark_phase": phase},
        "metrics": {
            "input_sequence_length": {"value": isl, "unit": "tokens"},
            "output_sequence_length": {"value": osl, "unit": "tokens"},
        },
        "error": error,
    }


def test_summary_excludes_warmup_and_failed_requests(tmp_path: Path) -> None:
    artifacts = tmp_path / "aiperf_artifacts"
    artifacts.mkdir()
    rows = [
        _record(9000, 900, phase="warmup"),
        _record(100, 10),
        _record(5000, 500, error={"type": "HTTPStatusError", "message": "500"}),
        _record(300, 30),
    ]
    (artifacts / "profile_export.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    subprocess.run(
        [sys.executable, "-m", "infx.results.agentic.analyze_benchmark_distributions",
         str(artifacts), "-o", str(tmp_path / "out")],
        cwd=E2E_ROOT, check=True, capture_output=True, text=True, timeout=30,
    )

    summary = (tmp_path / "out" / "workload_distribution_summary.txt").read_text()
    assert "Total requests: 2\n" in summary
    assert "ALL REQUESTS ISL:\n  n=2  mean=200  " in summary
    assert "ALL REQUESTS OSL:\n  n=2  mean=20  " in summary
