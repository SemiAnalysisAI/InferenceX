"""Query the committed golden AL curves.

python -m infx.golden_al_distribution list
python -m infx.golden_al_distribution show qwen3.5_mtp
python -m infx.golden_al_distribution lookup qwen3.5 mtp 3 --thinking thinking_on
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from .curves import (
    GOLDEN_DIR,
    THINKING_MODES,
    Curve,
    curve_name,
    golden_length,
    list_curves,
    load_curve,
)


def token_range(tokens: list[int]) -> str:
    if not tokens:
        return "-"
    if tokens == list(range(tokens[0], tokens[-1] + 1)) and len(tokens) > 1:
        return f"{tokens[0]}-{tokens[-1]}"
    return ",".join(str(token) for token in tokens)


def curve_json(curve: Curve) -> dict:
    return {
        "curve": curve.name,
        "model": curve.model,
        "modes": {
            mode: {str(tokens): float(curve.modes[mode][tokens]) for tokens in curve.tokens(mode)}
            for mode in curve.modes
        },
    }


def list_command(args: argparse.Namespace) -> None:
    curves = list_curves(args.golden_dir)
    if args.json:
        print(json.dumps([curve_json(curve) for curve in curves], indent=2))
        return
    rows = [("curve", "model", *THINKING_MODES)]
    rows += [
        (curve.name, curve.model, *(token_range(curve.tokens(mode)) for mode in THINKING_MODES))
        for curve in curves
    ]
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    for row in rows:
        print(
            "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip()
        )


def show_command(args: argparse.Namespace) -> None:
    curve = load_curve(args.curve, args.golden_dir)
    if args.json:
        print(json.dumps(curve_json(curve), indent=2))
        return
    modes = [mode for mode in THINKING_MODES if mode in curve.modes]
    print(f"{curve.name} ({curve.model})")
    print("  ".join(["tokens", *modes]))
    for tokens in sorted({token for mode in modes for token in curve.tokens(mode)}):
        cells = [
            f"{float(curve.modes[mode][tokens]):.2f}" if tokens in curve.modes[mode] else "-"
            for mode in modes
        ]
        print(
            "  ".join(
                [
                    str(tokens).ljust(6),
                    *(cell.ljust(len(mode)) for cell, mode in zip(cells, modes, strict=True)),
                ]
            ).rstrip()
        )


def lookup_command(args: argparse.Namespace) -> None:
    spec = {"method": args.method, "num_speculative_tokens": args.tokens}
    if args.draft_model:
        spec["model"] = args.draft_model
    if args.draft_sample_method:
        spec["draft_sample_method"] = args.draft_sample_method
    value = golden_length(args.model, spec, args.thinking, args.golden_dir)
    if args.json:
        print(
            json.dumps(
                {
                    "curve": curve_name(args.model, spec),
                    "thinking": args.thinking,
                    "tokens": args.tokens,
                    "acceptance_length": value,
                }
            )
        )
        return
    print(f"{value:g}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m infx.golden_al_distribution",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--golden-dir", type=Path, default=GOLDEN_DIR, help=argparse.SUPPRESS)
    commands = parser.add_subparsers(dest="command", required=True)

    list_parser = commands.add_parser("list", help="every curve with its measured draft lengths")
    list_parser.add_argument("--json", action="store_true")
    list_parser.set_defaults(handler=list_command)

    show_parser = commands.add_parser("show", help="every AL in one curve")
    show_parser.add_argument("curve", help="curve file stem, e.g. qwen3.5_mtp")
    show_parser.add_argument("--json", action="store_true")
    show_parser.set_defaults(handler=show_command)

    lookup_parser = commands.add_parser(
        "lookup", help="the AL synthetic acceptance applies for a model prefix and method"
    )
    lookup_parser.add_argument("model", help="InferenceX model prefix, e.g. qwen3.5, dsv4, kimik3")
    lookup_parser.add_argument("method", help="spec method: mtp, eagle, nextn, eagle3, dspark")
    lookup_parser.add_argument("tokens", type=int, help="num_speculative_tokens")
    lookup_parser.add_argument("--thinking", choices=THINKING_MODES, default="thinking_on")
    lookup_parser.add_argument("--draft-model", help="draft checkpoint (selects MiniMax GQA)")
    lookup_parser.add_argument(
        "--draft-sample-method",
        choices=("greedy", "probabilistic"),
        help="Kimi DSpark draft sampler",
    )
    lookup_parser.add_argument("--json", action="store_true")
    lookup_parser.set_defaults(handler=lookup_command)

    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
