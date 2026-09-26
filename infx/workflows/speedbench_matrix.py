"""Aggregate per-cell SPEED-Bench AL result JSONs into the golden YAML matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def aggregate_cells(
    result_dir: Path,
    model_key: str,
    thinking_modes: list[str],
    mtp_list: list[int],
    header_lines: list[str],
) -> str:
    """Read per-cell JSONs and produce the YAML matrix string.

    Each cell is expected at ``result_dir/speedbench_{mode}_mtp{mtp}.json`` with
    an ``al`` field (float or ``"N/A"``).  Missing or unreadable cells produce
    ``N/A``.  The output format matches the legacy collector scripts exactly:
    same key ordering, 2-decimal ALs, ``N/A`` for failed/missing cells.
    """
    cells: dict[str, dict[int, str]] = {}
    for mode in thinking_modes:
        cells[mode] = {}
        for mtp in mtp_list:
            path = result_dir / f"speedbench_{mode}_mtp{mtp}.json"
            al = "N/A"
            try:
                data = json.loads(path.read_text())
                raw = data["al"]
                if raw != "N/A":
                    al = f"{float(raw):.2f}"
            except (OSError, KeyError, ValueError, TypeError):
                pass
            cells[mode][mtp] = al

    lines: list[str] = []
    for header in header_lines:
        lines.append(f"# {header}")
    lines.append(f"{model_key}:")
    for mode in thinking_modes:
        lines.append(f"  thinking_{mode}:")
        for mtp in mtp_list:
            lines.append(f"    {mtp}: {cells[mode][mtp]}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--thinking-modes", required=True, help="Space-separated modes")
    parser.add_argument("--mtp-list", required=True, help="Space-separated MTP levels")
    parser.add_argument("--header", action="append", default=[], help="Header comment line")
    parser.add_argument("--output", type=Path, help="Write to file instead of stdout")
    args = parser.parse_args()

    modes = args.thinking_modes.split()
    mtps = [int(m) for m in args.mtp_list.split()]
    result = aggregate_cells(args.result_dir, args.model_key, modes, mtps, args.header)

    if args.output:
        args.output.write_text(result)
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
