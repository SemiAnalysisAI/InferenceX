import argparse
import json
from pathlib import Path
from typing import Any, Callable


Row = dict[str, Any]
Criterion = tuple[str, Callable[[Row], bool]]

EXPECTED_131K_COVERAGE = {
    ("b200", "vllm"),
    ("b200", "sglang"),
    ("h100", "vllm"),
    ("h100", "sglang"),
    ("h200", "vllm"),
    ("h200", "sglang"),
}
EXPECTED_1M_COVERAGE = {
    ("b200", "vllm"),
    ("b200", "sglang"),
}


def normalize_hw_label(hw: str | None) -> str:
    """Normalize runner labels like h200-cw-1 to coverage labels like h200."""
    if not hw:
        return ""
    return hw.split("-", 1)[0]


def load_rows(report_path: Path) -> list[Row]:
    """Load aggregated ISB1 rows from JSON."""
    payload = json.loads(report_path.read_text())
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        return [payload]
    raise ValueError(f"Unsupported ISB1 payload type: {type(payload)!r}")


def build_row_reference(row: Row, failed_criteria: list[str] | None = None) -> Row:
    """Build a concise row reference for gate reports."""
    reference: Row = {
        "result_filename": row.get("result_filename"),
        "artifact_stems": row.get("artifact_stems") or {},
        "hw": row.get("hw"),
        "framework": row.get("framework"),
        "infmax_model_prefix": row.get("infmax_model_prefix"),
        "support_status": row.get("support_status"),
        "context_pressure_status": (row.get("context_pressure_signal") or {}).get("status"),
    }
    if failed_criteria:
        reference["failed_criteria"] = failed_criteria
    return reference


def completed_sessions_match(row: Row) -> bool:
    return row.get("completed_sessions") == row.get("total_sessions")


def throughput_positive(row: Row) -> bool:
    return float(row.get("session_throughput_sps") or 0.0) > 0.0


def certification_verified(row: Row) -> bool:
    return row.get("benchmark_certification_status") == "dataset_replay_verified"


def context_not_suspicious(row: Row) -> bool:
    return not bool(row.get("context_pressure_suspicious"))


def vllm_context_ok(row: Row) -> bool:
    if row.get("framework") != "vllm":
        return True
    signal = row.get("context_pressure_signal") or {}
    return signal.get("status") == "ok" and not bool(row.get("context_pressure_suspicious"))


def get_present_coverage(rows: list[Row]) -> set[tuple[str, str]]:
    return {
        (normalize_hw_label(row.get("hw")), row.get("framework", ""))
        for row in rows
    }


def evaluate_gate(
    gate_id: str,
    label: str,
    rows: list[Row],
    criteria: list[Criterion],
    *,
    expected_coverage: set[tuple[str, str]] | None = None,
    exact_coverage: bool = False,
) -> Row:
    """Evaluate a gate definition over matching rows."""
    if not rows:
        return {
            "id": gate_id,
            "label": label,
            "status": "no_rows",
            "matched_rows": 0,
            "failing_rows": [],
            "review_required_rows": [],
            "missing_coverage": [],
            "unexpected_coverage": [],
        }

    failing_rows = []
    review_required_rows = []
    for row in rows:
        failed_criteria = [description for description, checker in criteria if not checker(row)]
        if failed_criteria:
            failing_rows.append(build_row_reference(row, failed_criteria))
        signal = row.get("context_pressure_signal") or {}
        if signal.get("requires_log_review"):
            review_required_rows.append(build_row_reference(row))

    missing_coverage: list[list[str]] = []
    unexpected_coverage: list[list[str]] = []
    if expected_coverage is not None:
        present_coverage = get_present_coverage(rows)
        missing_coverage = [list(item) for item in sorted(expected_coverage - present_coverage)]
        if exact_coverage:
            unexpected_coverage = [list(item) for item in sorted(present_coverage - expected_coverage)]

    status = "pass"
    if failing_rows or missing_coverage or unexpected_coverage:
        status = "fail"

    return {
        "id": gate_id,
        "label": label,
        "status": status,
        "matched_rows": len(rows),
        "failing_rows": failing_rows,
        "review_required_rows": review_required_rows,
        "missing_coverage": missing_coverage,
        "unexpected_coverage": unexpected_coverage,
    }


def build_gate_report(rows: list[Row], advisory: bool = True) -> Row:
    """Build the full advisory gate report for an aggregated ISB1 result set."""
    gates = [
        evaluate_gate(
            "control_lanes",
            "DSR1/GPT-OSS control lanes",
            [
                row
                for row in rows
                if row.get("infmax_model_prefix") in {"dsr1", "gptoss"}
                and row.get("support_status") == "supported"
            ],
            [
                ("completed_sessions == total_sessions", completed_sessions_match),
                ("session_throughput_sps > 0", throughput_positive),
            ],
        ),
        evaluate_gate(
            "qwen_131k",
            "Qwen 131k preview lanes",
            [
                row
                for row in rows
                if row.get("infmax_model_prefix") == "qwen3.5"
                and row.get("support_status") == "reviewed_preview"
                and (row.get("effective_max_context_depth") or 0) < 200000
            ],
            [
                ("completed_sessions == total_sessions", completed_sessions_match),
                ("session_throughput_sps > 0", throughput_positive),
            ],
            expected_coverage=EXPECTED_131K_COVERAGE,
        ),
        evaluate_gate(
            "qwen_500k",
            "Qwen 500k preview lanes",
            [
                row
                for row in rows
                if row.get("infmax_model_prefix") == "qwen3.5"
                and row.get("effective_max_context_depth") == 524288
                and row.get("context_pressure_class") == "extended_500k"
            ],
            [
                ("completed_sessions == total_sessions", completed_sessions_match),
                (
                    "benchmark_certification_status == dataset_replay_verified",
                    certification_verified,
                ),
                ("context_pressure_suspicious == false", context_not_suspicious),
                ("vllm context_pressure_signal.status == ok", vllm_context_ok),
            ],
        ),
        evaluate_gate(
            "qwen_1m",
            "Qwen 1M preview lanes",
            [
                row
                for row in rows
                if row.get("infmax_model_prefix") == "qwen3.5"
                and row.get("effective_max_context_depth") == 1048576
                and row.get("context_pressure_class") == "extended_1m"
            ],
            [
                ("completed_sessions == total_sessions", completed_sessions_match),
                ("context_pressure_suspicious == false", context_not_suspicious),
                ("vllm context_pressure_signal.status == ok", vllm_context_ok),
            ],
            expected_coverage=EXPECTED_1M_COVERAGE,
            exact_coverage=True,
        ),
    ]

    statuses = {gate["status"] for gate in gates}
    if "fail" in statuses:
        overall = "fail"
    elif statuses == {"pass"}:
        overall = "pass"
    else:
        overall = "partial"

    return {
        "gates": gates,
        "overall": overall,
        "advisory": advisory,
    }


def render_markdown(report: Row) -> str:
    """Render a concise markdown advisory summary for workflow step summaries."""
    lines = [
        "## ISB1 Advisory Gates",
        "",
        f"Overall: **{report['overall'].upper()}** ({'advisory' if report['advisory'] else 'strict'})",
        "",
    ]

    for gate in report["gates"]:
        lines.append(f"### {gate['label']} — {gate['status'].upper()}")
        lines.append("")
        lines.append(f"- Matched rows: {gate['matched_rows']}")
        if gate["missing_coverage"]:
            formatted = ", ".join(f"{hw}/{framework}" for hw, framework in gate["missing_coverage"])
            lines.append(f"- Missing coverage: {formatted}")
        if gate["unexpected_coverage"]:
            formatted = ", ".join(
                f"{hw}/{framework}" for hw, framework in gate["unexpected_coverage"]
            )
            lines.append(f"- Unexpected coverage: {formatted}")
        if gate["failing_rows"]:
            lines.append("- Failing rows:")
            for row in gate["failing_rows"]:
                failed_criteria = ", ".join(row.get("failed_criteria", [])) or "unknown"
                lines.append(
                    f"  - `{row.get('result_filename', 'unknown')}` ({row.get('hw', '-')}/"
                    f"{row.get('framework', '-')}) failed: {failed_criteria}"
                )
        elif gate["matched_rows"]:
            lines.append("- No failing rows.")
        if gate["review_required_rows"]:
            review_rows = ", ".join(
                f"`{row.get('result_filename', 'unknown')}`" for row in gate["review_required_rows"]
            )
            lines.append(
                "- Manual log review still required for: "
                f"{review_rows}"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate advisory ISB1 gates.")
    parser.add_argument("report_path", type=Path)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--format", choices=["json", "markdown"], default="json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_gate_report(load_rows(args.report_path), advisory=not args.strict)

    if args.format == "markdown":
        print(render_markdown(report))
    else:
        print(json.dumps(report, indent=2))

    if args.strict and report["overall"] == "fail":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
