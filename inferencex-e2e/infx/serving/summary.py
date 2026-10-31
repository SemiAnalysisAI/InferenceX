"""Markdown table of the points of one gn-benchmark run, for the run's summary page.

    python -P -m infx.serving.summary <dir>  >> "$GITHUB_STEP_SUMMARY"

``<dir>`` holds the aggregate result JSONs the points uploaded (any depth). One row per point; when a
(scenario, concurrency) has a speculative arm and a base arm (spec_decoding none), the speculative
row also carries the speedup over the base. Only aggregate numbers are printed: nothing of a request.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

HEADERS = (
    "scenario",
    "spec",
    "conc",
    "requests ok/total",
    "output tok/s",
    "TTFT p50 / p95 (ms)",
    "TPOT mean (ms)",
    "acceptance length",
    "accept rate",
    "speedup",
)
DASH = "-"
OUTPUT_TPS = ("request_metrics", "throughput", "output", "tokens_per_second")


def dig(doc: Any, *path: str) -> Any:
    for key in path:
        if not isinstance(doc, dict) or key not in doc:
            return None
        doc = doc[key]
    return doc


def load(directory: Path) -> list[dict[str, Any]]:
    """The aggregate results under ``directory``, ordered for display."""
    results = []
    for path in sorted(directory.rglob("*.json")):
        try:
            doc = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict) and "request_metrics" in doc and "scenario" in doc:
            results.append(doc)
    return sorted(
        results,
        key=lambda r: (str(r["scenario"]), int(r.get("conc") or 0), str(r.get("spec_decoding"))),
    )


def number(value: Any, digits: int = 1, scale: float = 1.0) -> str:
    return (
        f"{value * scale:,.{digits}f}"
        if isinstance(value, int | float) and not isinstance(value, bool)
        else DASH
    )


def speedups(results: list[dict[str, Any]]) -> dict[int, float]:
    """Output-throughput ratio of each speculative result over the base result of its scenario and concurrency."""
    base = {
        (r["scenario"], r.get("conc")): dig(r, *OUTPUT_TPS)
        for r in results
        if r.get("spec_decoding", "none") == "none"
    }
    out = {}
    for index, result in enumerate(results):
        denominator = base.get((result["scenario"], result.get("conc")))
        numerator = dig(result, *OUTPUT_TPS)
        if result.get("spec_decoding", "none") != "none" and denominator and numerator:
            out[index] = numerator / denominator
    return out


def row(result: dict[str, Any], speedup: float | None) -> list[str]:
    metrics = result["request_metrics"]
    spec = result.get("spec_decode") or {}
    ok, total = result.get("num_requests_successful"), result.get("num_requests_total")
    return [
        str(result["scenario"]),
        str(result.get("spec_decoding", "none"))
        + (f"x{spec['draft_depth']}" if spec.get("draft_depth") else ""),
        str(result.get("conc", DASH)),
        f"{ok}/{total}" if ok is not None and total is not None else DASH,
        number(dig(result, *OUTPUT_TPS)),
        f"{number(dig(metrics, 'latency', 'ttft', 'p50'), 0, 1000)} / {number(dig(metrics, 'latency', 'ttft', 'p95'), 0, 1000)}",
        number(dig(metrics, "latency", "tpot", "mean"), 1, 1000),
        number(spec.get("acceptance_length"), 2),
        number(spec.get("acceptance_rate"), 1, 100)
        + ("%" if spec.get("acceptance_rate") is not None else ""),
        f"{speedup:.2f}x" if speedup else DASH,
    ]


def table(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No aggregate result was uploaded by this run.\n"
    ratios = speedups(results)
    lines = ["| " + " | ".join(HEADERS) + " |", "|" + "|".join(["---"] * len(HEADERS)) + "|"]
    lines += ["| " + " | ".join(row(r, ratios.get(i))) + " |" for i, r in enumerate(results)]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m infx.serving.summary <dir>", file=sys.stderr)
        return 2
    print("### Results\n")
    print(table(load(Path(argv[0]))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
