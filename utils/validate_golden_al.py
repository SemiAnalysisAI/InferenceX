"""Resolve and validate a golden acceptance length for a draft model."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class GoldenALError(ValueError):
    """Raised when a golden AL curve cannot be selected or validated."""


@dataclass(frozen=True)
class GoldenALCurve:
    """One model curve loaded from a golden AL YAML file."""

    path: Path
    draft_model: str | None
    values: dict[str, Any]


def _load_curves(directory: Path, model: str) -> list[GoldenALCurve]:
    curves: list[GoldenALCurve] = []
    for path in sorted(directory.glob("*.yaml")):
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise GoldenALError(f"failed to read {path}: {exc}") from exc

        if not isinstance(document, dict) or model not in document:
            continue

        values = document[model]
        if not isinstance(values, dict):
            raise GoldenALError(f"{path} has a non-mapping curve for model {model!r}")

        metadata = document.get("_metadata", {})
        if not isinstance(metadata, dict):
            raise GoldenALError(f"{path} has non-mapping _metadata")
        draft_model = metadata.get("draft-model")
        if draft_model is not None and not isinstance(draft_model, str):
            raise GoldenALError(f"{path} has a non-string _metadata.draft-model")

        curves.append(
            GoldenALCurve(path=path, draft_model=draft_model, values=values)
        )
    return curves


def _available_drafts(curves: list[GoldenALCurve]) -> str:
    return ", ".join(
        curve.draft_model or f"<unspecified in {curve.path.name}>" for curve in curves
    )


def _select_curve(
    curves: list[GoldenALCurve], model: str, draft_model: str | None
) -> GoldenALCurve:
    if not curves:
        raise GoldenALError(f"no golden AL curve found for model {model!r}")

    if draft_model is None:
        if len(curves) != 1:
            raise GoldenALError(
                f"golden AL curve for model {model!r} is ambiguous; pass "
                f"--draft-model. Available drafts: {_available_drafts(curves)}"
            )
        return curves[0]

    matches = [curve for curve in curves if curve.draft_model == draft_model]
    if not matches:
        raise GoldenALError(
            f"no golden AL curve found for model {model!r} and draft model "
            f"{draft_model!r}. Available drafts: {_available_drafts(curves)}"
        )
    if len(matches) > 1:
        files = ", ".join(curve.path.name for curve in matches)
        raise GoldenALError(
            f"duplicate golden AL curves for model {model!r} and draft model "
            f"{draft_model!r}: {files}"
        )
    return matches[0]


def resolve_golden_al(
    directory: Path,
    model: str,
    draft_model: str | None,
    thinking_mode: str,
    num_speculative_tokens: int,
) -> float:
    """Return the exact golden AL selected by model, draft, mode, and level."""
    curve = _select_curve(_load_curves(directory, model), model, draft_model)

    mode_values = curve.values.get(thinking_mode)
    if not isinstance(mode_values, dict):
        raise GoldenALError(
            f"{curve.path} has no golden AL values for thinking mode "
            f"{thinking_mode!r}"
        )

    value = mode_values.get(num_speculative_tokens)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise GoldenALError(
            f"{curve.path} has no numeric golden AL for {num_speculative_tokens} "
            "speculative tokens"
        )
    return float(value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve a golden acceptance length by target model, draft model, "
            "thinking mode, and speculative-token count."
        )
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path("golden_al_distribution"),
        help="Directory containing golden AL YAML files.",
    )
    parser.add_argument("--model", required=True, help="Target model key in the YAML.")
    parser.add_argument(
        "--draft-model",
        help="Exact _metadata.draft-model identifier; required when curves overlap.",
    )
    parser.add_argument(
        "--thinking-mode",
        required=True,
        choices=("thinking_on", "thinking_off"),
    )
    parser.add_argument("--num-speculative-tokens", required=True, type=int)
    parser.add_argument(
        "--expected-al",
        type=float,
        help="Fail unless the pinned numeric AL matches the selected golden value.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        golden_al = resolve_golden_al(
            directory=args.directory,
            model=args.model,
            draft_model=args.draft_model,
            thinking_mode=args.thinking_mode,
            num_speculative_tokens=args.num_speculative_tokens,
        )
        if args.expected_al is not None and not math.isclose(
            args.expected_al, golden_al, rel_tol=0.0, abs_tol=1e-9
        ):
            raise GoldenALError(
                f"pinned acceptance length does not match: expected "
                f"{args.expected_al:g}, golden {golden_al:g}"
            )
    except GoldenALError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"{golden_al:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
