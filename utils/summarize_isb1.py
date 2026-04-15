import argparse
import json
from pathlib import Path
from typing import Any

try:
    from tabulate import tabulate as _tabulate
except ImportError:  # pragma: no cover - fallback for minimal local environments
    _tabulate = None


SUPPORT_STATUS_ORDER = {
    "supported": 0,
    "reviewed_preview": 1,
    "gated": 2,
    "artifact_only": 3,
    "unsupported": 4,
    None: 5,
}


def load_isb1_rows(results_dir: Path) -> list[dict[str, Any]]:
    """Load processed ISB1 rows from a results directory."""
    rows: list[dict[str, Any]] = []
    for result_path in results_dir.rglob("*.json"):
        try:
            payload = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue

        candidates = payload if isinstance(payload, list) else [payload]
        for candidate in candidates:
            if isinstance(candidate, dict) and candidate.get("benchmark_type") == "isb1_replay":
                rows.append(candidate)
    return rows


def sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort rows in an operator-friendly order."""
    return sorted(
        rows,
        key=lambda row: (
            SUPPORT_STATUS_ORDER.get(row.get("support_status"), 99),
            row.get("infmax_model_prefix", ""),
            row.get("hw", ""),
            row.get("framework", ""),
            row.get("effective_max_context_depth", 0) or 0,
            row.get("result_filename", ""),
        ),
    )


def format_float(value: Any, precision: int = 2) -> str:
    """Format a numeric value for markdown tables."""
    if value is None:
        return "-"
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return str(value)


def format_bool(value: Any) -> str:
    """Format a truthy value as yes/no for operators."""
    return "yes" if bool(value) else "no"


def render_table(headers: list[str], rows: list[list[Any]], tablefmt: str) -> str:
    """Render a markdown/plain table with a lightweight fallback if tabulate is absent."""
    normalized_rows = [[str(cell) for cell in row] for row in rows]
    if _tabulate is not None:
        return _tabulate(normalized_rows, headers=headers, tablefmt=tablefmt)

    widths = [len(header) for header in headers]
    for row in normalized_rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def render_row(row: list[str]) -> str:
        cells = [cell.ljust(widths[index]) for index, cell in enumerate(row)]
        return f"| {' | '.join(cells)} |"

    divider = f"| {' | '.join('-' * width for width in widths)} |"
    lines = [render_row(headers), divider]
    lines.extend(render_row(row) for row in normalized_rows)
    return "\n".join(lines)


def build_lane_summary_table(rows: list[dict[str, Any]], tablefmt: str) -> str:
    """Render the main operator lane summary table."""
    headers = [
        "Lane",
        "Model",
        "HW",
        "Framework",
        "Support",
        "Cert",
        "Max Ctx",
        "Context Class",
        "Sessions",
        "Session Tput",
        "TTFT Median (s)",
        "Ctx Pressure",
        "Log Review",
        "KV Offload",
        "GPU Cache Peak",
        "CPU Cache Peak",
    ]
    table_rows = [
        [
            row.get("result_filename", "-"),
            row.get("infmax_model_prefix", "-"),
            row.get("hw", "-"),
            row.get("framework", "-"),
            row.get("support_status", "-"),
            row.get("benchmark_certification_status", "-"),
            row.get("effective_max_context_depth", "-"),
            row.get("context_pressure_class", "-"),
            f"{row.get('completed_sessions', 0)}/{row.get('total_sessions', 0)}",
            format_float(row.get("session_throughput_sps"), 2),
            format_float(row.get("median_ttft"), 3),
            (row.get("context_pressure_signal") or {}).get("status", "-"),
            format_bool((row.get("context_pressure_signal") or {}).get("requires_log_review")),
            format_bool(row.get("kv_offload_observed")),
            format_float(row.get("peak_gpu_cache_usage"), 2),
            format_float(row.get("peak_cpu_cache_usage"), 2),
        ]
        for row in rows
    ]
    return render_table(headers, table_rows, tablefmt)


def build_runtime_override_table(rows: list[dict[str, Any]], tablefmt: str) -> str | None:
    """Render the runtime override table when any override is present."""
    override_rows = []
    for row in rows:
        runtime_overrides = row.get("runtime_overrides") or {}
        if not any(value not in (None, "") for value in runtime_overrides.values()):
            continue
        override_rows.append(
            [
                row.get("result_filename", "-"),
                row.get("infmax_model_prefix", "-"),
                row.get("hw", "-"),
                row.get("framework", "-"),
                runtime_overrides.get("vllm_cpu_offload_gb") or "-",
                runtime_overrides.get("vllm_swap_space_gb") or "-",
                runtime_overrides.get("sglang_mem_fraction_override") or "-",
                runtime_overrides.get("sglang_chunked_prefill_override") or "-",
                row.get("dispatch_ref") or "-",
            ]
        )

    if not override_rows:
        return None

    headers = [
        "Lane",
        "Model",
        "HW",
        "Framework",
        "VLLM CPU Offload GB",
        "VLLM Swap GB",
        "SGLang Mem Fraction",
        "SGLang Chunked Prefill",
        "Dispatch Ref",
    ]
    return render_table(headers, override_rows, tablefmt)


def build_action_items(rows: list[dict[str, Any]]) -> list[str]:
    """Build operator action items for suspicious or manual-review rows."""
    items: list[str] = []
    for row in rows:
        signal = row.get("context_pressure_signal") or {}
        if not row.get("context_pressure_suspicious") and not signal.get("requires_log_review"):
            continue

        artifact_stems = row.get("artifact_stems") or {}
        items.append(
            "- "
            f"`{row.get('result_filename', 'unknown')}` ({row.get('infmax_model_prefix', '-')}/"
            f"{row.get('hw', '-')}/{row.get('framework', '-')}) "
            f"requires follow-up: context pressure `{signal.get('status', 'unknown')}`; "
            f"review replay `{artifact_stems.get('raw_replay', '-')}`, "
            f"logs `{artifact_stems.get('server_logs', '-')}`, "
            f"GPU metrics `{artifact_stems.get('gpu_metrics', '-')}`"
            + (
                f", dispatch `{row.get('dispatch_ref')}`"
                if row.get("dispatch_ref")
                else ""
            )
            + "."
        )
    return items


def generate_summary(results_dir: Path, tablefmt: str = "github") -> str:
    """Generate an ISB1-specific operator summary in markdown/plain text."""
    rows = sort_rows(load_isb1_rows(results_dir))
    sections = ["## ISB1 Operator Summary", ""]

    if not rows:
        sections.append("No ISB1 replay rows found.")
        return "\n".join(sections).rstrip() + "\n"

    sections.extend(["### Lane Summary", "", build_lane_summary_table(rows, tablefmt), ""])

    runtime_override_table = build_runtime_override_table(rows, tablefmt)
    if runtime_override_table:
        sections.extend(["### Runtime Overrides", "", runtime_override_table, ""])

    action_items = build_action_items(rows)
    sections.append("### Action Items")
    sections.append("")
    if action_items:
        sections.extend(action_items)
    else:
        sections.append("- None. No suspicious or manual-log-review rows were detected.")

    return "\n".join(sections).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an ISB1-specific operator summary.")
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--format", choices=["github", "plain"], default="github")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(generate_summary(args.results_dir, tablefmt=args.format))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
