"""Shell-contract tests for deriving AIPerf worker metrics URLs from srt-slurm endpoints."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
BENCHMARK_LIB = REPO_ROOT / "benchmarks" / "benchmark_lib.sh"

PD_ENDPOINTS = {
    "SRT_PREFILL_ENDPOINTS": "10.0.0.1:8081,10.0.0.2:8081",
    "SRT_DECODE_ENDPOINTS": "10.0.0.3:8082",
}


def _resolve(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    script = (
        f"source {str(BENCHMARK_LIB)!r} && resolve_srt_worker_server_metrics_urls && "
        'printf "%s" "${AIPERF_SERVER_METRICS_URLS-<unset>}"'
    )
    clean = {
        k: v for k, v in os.environ.items() if not k.startswith(("SRT_", "AIPERF_", "SRTCTL_"))
    }
    return subprocess.run(
        ["/bin/bash", "-c", script],
        env={**clean, **env},
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        (
            {
                "SRTCTL_FRONTEND_TYPE": "dynamo",
                "AIPERF_SCRAPE_WORKER_METRICS": "true",
                **PD_ENDPOINTS,
            },
            (
                "http://10.0.0.1:8081/metrics,http://10.0.0.2:8081/metrics,"
                "http://10.0.0.3:8082/metrics"
            ),
        ),
        ({"SRTCTL_FRONTEND_TYPE": "dynamo", **PD_ENDPOINTS}, "<unset>"),
        (
            {"SRTCTL_FRONTEND_TYPE": "sglang", "SRT_AGG_ENDPOINTS": "10.0.0.9:30000"},
            "http://10.0.0.9:30000/metrics",
        ),
        (
            {
                "SRTCTL_FRONTEND_TYPE": "dynamo",
                "AIPERF_SCRAPE_WORKER_METRICS": "true",
                "AIPERF_SERVER_METRICS_URLS": "http://curated:9/metrics",
                **PD_ENDPOINTS,
            },
            "http://curated:9/metrics",
        ),
    ],
    ids=["dynamo-opt-in", "dynamo-default", "non-dynamo-agg", "explicit-wins"],
)
def test_resolves_worker_metrics_urls(env: dict[str, str], expected: str) -> None:
    result = _resolve(env)

    assert result.returncode == 0, result.stderr
    assert result.stdout == expected


def test_rejects_invalid_opt_in_value() -> None:
    result = _resolve(
        {"SRTCTL_FRONTEND_TYPE": "dynamo", "AIPERF_SCRAPE_WORKER_METRICS": "1", **PD_ENDPOINTS}
    )

    assert result.returncode == 1
    assert "AIPERF_SCRAPE_WORKER_METRICS must be true or false" in result.stderr
