#!/usr/bin/env python3
"""SpeedBench acceptance-length reference and result helpers."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


MODEL_PREFIX_ALIASES = {
    "dsv4": "deepseek-v4-pro",
}


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"", "null", "None", "~"}:
        return None
    if value in {"N/A", "NA", "n/a", "na"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        if re.match(r"^-?\d+$", value):
            return int(value)
        return float(value)
    except ValueError:
        return value


def _load_simple_reference_yaml(path: Path) -> dict[str, Any]:
    """Parse the simple nested mapping emitted by the SpeedBench AL workflow."""
    data: dict[str, Any] = {}
    current_model: str | None = None
    current_mode: str | None = None

    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if ":" not in line:
            continue
        key, value = line.strip().split(":", 1)
        key = key.strip().strip("'\"")
        value = value.strip()

        if indent == 0:
            current_model = key
            data.setdefault(current_model, {})
            current_mode = None
        elif indent == 2 and current_model is not None:
            current_mode = key
            data[current_model].setdefault(current_mode, {})
        elif indent == 4 and current_model is not None and current_mode is not None:
            data[current_model][current_mode][key] = _parse_scalar(value)

    return data


def load_reference(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError:
        return _load_simple_reference_yaml(path)

    with path.open() as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a mapping at the top level")
    return loaded


def normalize_key(value: str) -> str:
    value = value.strip().split("/")[-1].lower()
    value = value.replace("_", "-")
    value = re.sub(r"[^a-z0-9.+-]+", "-", value)
    return value.strip("-")


def model_candidates(model: str, model_prefix: str | None = None) -> list[str]:
    candidates: list[str] = []
    if model_prefix:
        prefix = normalize_key(model_prefix)
        candidates.append(MODEL_PREFIX_ALIASES.get(prefix, prefix))
    if model:
        normalized = normalize_key(model)
        candidates.append(MODEL_PREFIX_ALIASES.get(normalized, normalized))
    seen = set()
    out = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            out.append(candidate)
            seen.add(candidate)
    return out


def normalize_mode(thinking_mode: str) -> str:
    mode = thinking_mode.strip().lower().replace("-", "_")
    if mode == "on":
        return "thinking_on"
    if mode == "off":
        return "thinking_off"
    raise ValueError("SpeedBench thinking mode must be 'on' or 'off'")


def lookup_reference(
    reference: dict[str, Any],
    model: str,
    model_prefix: str | None,
    thinking_mode: str,
    num_speculative_tokens: int,
) -> tuple[str, str, float]:
    normalized_reference = {normalize_key(str(k)): v for k, v in reference.items()}
    mode_key = normalize_mode(thinking_mode)
    token_key = str(num_speculative_tokens)

    for candidate in model_candidates(model, model_prefix):
        model_block = normalized_reference.get(candidate)
        if not isinstance(model_block, dict):
            continue
        mode_block = model_block.get(mode_key)
        if not isinstance(mode_block, dict):
            continue
        value = mode_block.get(num_speculative_tokens, mode_block.get(token_key))
        if value is None:
            continue
        try:
            return candidate, mode_key, float(value)
        except (TypeError, ValueError):
            continue

    candidates = ", ".join(model_candidates(model, model_prefix)) or "<none>"
    raise KeyError(
        "No SpeedBench AL reference for "
        f"model candidates [{candidates}], mode {mode_key}, MTP {num_speculative_tokens}"
    )


def _optional_float(value: str | None) -> float | None:
    if value in {None, "", "None", "null", "N/A"}:
        return None
    return float(value)


def _optional_int(value: str | None) -> int | None:
    if value in {None, "", "None", "null", "N/A"}:
        return None
    return int(float(value))


def build_result(args: argparse.Namespace) -> dict[str, Any]:
    reference_al: float | None = None
    min_acceptance_length: float | None = None
    model_key: str | None = None
    mode_key = normalize_mode(args.thinking_mode)
    error: str | None = args.error

    if args.reference_yaml:
        reference_path = Path(args.reference_yaml)
        if reference_path.exists():
            try:
                model_key, mode_key, reference_al = lookup_reference(
                    load_reference(reference_path),
                    args.model,
                    args.model_prefix,
                    args.thinking_mode,
                    args.num_speculative_tokens,
                )
                min_acceptance_length = reference_al * args.threshold_ratio
            except Exception as exc:  # noqa: BLE001 - recorded for CI artifacts
                error = error or str(exc)
        else:
            error = error or f"Reference YAML not found: {reference_path}"

    acceptance_length = _optional_float(args.acceptance_length)
    accepted_tokens = _optional_int(args.accepted_tokens)
    draft_tokens = _optional_int(args.draft_tokens)
    verify_steps = _optional_int(getattr(args, "verify_steps", None))
    proposed_draft_tokens = _optional_int(getattr(args, "proposed_draft_tokens", None))
    if verify_steps is None:
        verify_steps = draft_tokens
    passed = (
        error is None
        and acceptance_length is not None
        and min_acceptance_length is not None
        and acceptance_length >= min_acceptance_length
    )

    result = {
        "speedbench_al_eval_version": 1,
        "task": "speedbench_al",
        "model": args.model,
        "model_key": model_key,
        "model_prefix": args.model_prefix,
        "thinking_mode": mode_key,
        "num_speculative_tokens": args.num_speculative_tokens,
        "category": args.category,
        "output_len": args.output_len,
        "temperature": args.temperature,
        "framework": getattr(args, "framework", ""),
        "metric_source": getattr(args, "metric_source", ""),
        "acceptance_length": acceptance_length,
        "accepted_tokens": accepted_tokens,
        "verify_steps": verify_steps,
        "proposed_draft_tokens": proposed_draft_tokens,
        "draft_tokens": draft_tokens,
        "reference_acceptance_length": reference_al,
        "threshold_ratio": args.threshold_ratio,
        "min_acceptance_length": min_acceptance_length,
        "passed": passed,
    }
    if error:
        result["error"] = error
    return result


def cmd_resolve(args: argparse.Namespace) -> int:
    model_key, mode_key, reference_al = lookup_reference(
        load_reference(Path(args.reference_yaml)),
        args.model,
        args.model_prefix,
        args.thinking_mode,
        args.num_speculative_tokens,
    )
    payload = {
        "model_key": model_key,
        "thinking_mode": mode_key,
        "num_speculative_tokens": args.num_speculative_tokens,
        "reference_acceptance_length": reference_al,
        "threshold_ratio": args.threshold_ratio,
        "min_acceptance_length": reference_al * args.threshold_ratio,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


def cmd_record(args: argparse.Namespace) -> int:
    result = build_result(args)
    output = Path(args.output)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    status = "PASS" if result["passed"] else "FAIL"
    actual = result.get("acceptance_length")
    minimum = result.get("min_acceptance_length")
    print(
        f"{status}: SpeedBench AL {actual} "
        f"(min {minimum}, mode {result['thinking_mode']}, "
        f"mtp {result['num_speculative_tokens']})"
    )
    if args.exit_status and not result["passed"]:
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    resolve = subparsers.add_parser("resolve", help="Resolve a reference AL cell")
    resolve.add_argument("--reference-yaml", required=True)
    resolve.add_argument("--model", required=True)
    resolve.add_argument("--model-prefix", default="")
    resolve.add_argument("--thinking-mode", required=True)
    resolve.add_argument("--num-speculative-tokens", type=int, required=True)
    resolve.add_argument("--threshold-ratio", type=float, default=0.90)
    resolve.set_defaults(func=cmd_resolve)

    record = subparsers.add_parser("record", help="Write a compact AL eval result")
    record.add_argument("--output", required=True)
    record.add_argument("--reference-yaml", default="")
    record.add_argument("--model", required=True)
    record.add_argument("--model-prefix", default="")
    record.add_argument("--thinking-mode", required=True)
    record.add_argument("--num-speculative-tokens", type=int, required=True)
    record.add_argument("--category", default="coding")
    record.add_argument("--output-len", type=int, default=4096)
    record.add_argument("--temperature", type=float, default=1.0)
    record.add_argument("--threshold-ratio", type=float, default=0.90)
    record.add_argument("--framework", default="")
    record.add_argument("--metric-source", default="")
    record.add_argument("--acceptance-length", default=None)
    record.add_argument("--accepted-tokens", default=None)
    record.add_argument("--draft-tokens", default=None)
    record.add_argument("--verify-steps", default=None)
    record.add_argument("--proposed-draft-tokens", default=None)
    record.add_argument("--error", default=None)
    record.add_argument("--exit-status", action="store_true")
    record.set_defaults(func=cmd_record)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except Exception as exc:  # noqa: BLE001 - CLI should print concise failures
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
