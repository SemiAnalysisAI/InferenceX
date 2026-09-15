"""Pass speculative-acceptance settings to srtctl through native overrides.

The source recipe is read-only. srtctl owns selector expansion, application of
``--set`` / ``--unset``, validation, and the saved runtime recipe. Eval-only
runs restore real verification; ordinary RUN_EVAL behavior is unchanged.
"""

from __future__ import annotations

import argparse
import copy
import fnmatch
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from infx.recipes.overrides import RecipeOverride, override_argv

REFERENCE_PATH = (
    Path(__file__).resolve().parents[2] / "benchmarks/speedbench-reference-al.yaml"
)
MODEL_KEYS = {
    "dsv4": "deepseek-v4-pro",
    "dsr1": "deepseek-r1",
    "dsv4dspark": "deepseek-v4-pro-0813",
    "dsv4dsparkprob": "deepseek-v4-pro-0813",
}
ENGINES = {
    "vllm": "vllm",
    "dynamo-vllm": "vllm",
    "dynamo-sglang": "sglang",
    "trt": "trtllm",
    "dynamo-trt": "trtllm",
}
SGLANG_VARIABLES = (
    "SGLANG_SIMULATE_ACC_LEN",
    "SGLANG_SIMULATE_ACC_METHOD",
    "SGLANG_SIMULATE_ACC_TOKEN_MODE",
)
TRT_VARIABLE = "TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS"


def enabled(environment: Mapping[str, str], name: str) -> bool:
    return environment.get(name, "false").strip().lower() == "true"


def _spec_config(value: Any) -> dict[str, Any]:
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, dict):
        raise ValueError("speculative-config must be a JSON object")
    return dict(parsed)


def _spec_tokens(roles: Mapping[str, Any], engine: str) -> int:
    for role in roles.values():
        args = role.get("args", {})
        if engine == "vllm" and "speculative-config" in args:
            value = _spec_config(args["speculative-config"]).get(
                "num_speculative_tokens"
            )
        elif engine == "sglang":
            value = args.get("speculative-num-steps")
        else:
            spec = args.get("speculative_config", {})
            value = spec.get("max_draft_len", spec.get("num_nextn_predict_layers"))
        if value:
            return int(value)
    return 2


def resolve_acceptance_length(
    roles: Mapping[str, Any],
    engine: str,
    environment: Mapping[str, str],
    reference: Mapping[str, Any] | None,
) -> float:
    explicit = environment.get("SYNTHETIC_ACCEPTANCE_LENGTH", "").strip()
    if explicit:
        value = float(explicit)
    else:
        if reference is None:
            raise ValueError(
                "SYNTHETIC_ACCEPTANCE_LENGTH is unset and reference AL data is unavailable"
            )
        prefix = environment.get("MODEL_PREFIX", "")
        key = MODEL_KEYS.get(prefix, prefix)
        if key not in reference:
            raise ValueError(f'model key "{key}" not found in reference AL data')
        block = reference[key]
        nst = environment.get("NUM_SPEC_TOKENS", "").strip()
        tokens = int(nst) if nst else _spec_tokens(roles, engine)
        if isinstance(block, list):
            block = {level: al for item in block for level, al in item.items()}
        if isinstance(block, dict) and any(
            str(k).startswith("thinking") for k in block
        ):
            mode = (
                environment.get("THINKING_MODE", "thinking_on").strip() or "thinking_on"
            )
            if mode not in block:
                raise ValueError(
                    f"THINKING_MODE='{mode}' not found in reference AL data"
                )
            block = block[mode]
        if not isinstance(block, dict) or tokens not in block:
            raise ValueError(
                f"num_spec_tokens={tokens} not found for {key} in reference AL data"
            )
        value = float(block[tokens])
    if not math.isfinite(value) or value < 1:
        raise ValueError("synthetic acceptance length must be finite and at least 1")
    return value


def build_overrides(
    recipe: Mapping[str, Any],
    framework: str,
    environment: Mapping[str, str],
    *,
    reference: Mapping[str, Any] | None = None,
    enable_throughput: bool = True,
    require_match: bool = True,
) -> list[RecipeOverride]:
    """Build overrides for one resolved recipe without mutating its mappings."""
    real = enabled(environment, "EVAL_ONLY")
    synthetic = enable_throughput and enabled(environment, "SYNTHETIC_ACCEPTANCE")
    if not real and not synthetic:
        return []
    engine = ENGINES.get(framework)
    if engine is None:
        if real:
            return []
        raise ValueError(f"no synthetic-acceptance backend for FRAMEWORK='{framework}'")
    if "backend" in recipe and "roles" not in recipe:
        raise ValueError("native acceptance overrides require a schema-2 roles recipe")
    roles = recipe.get("roles", {})
    if not isinstance(roles, dict) or any(
        not isinstance(r, dict) for r in roles.values()
    ):
        raise ValueError("recipe roles must contain worker mappings")
    roles = {
        name: role
        for name, role in roles.items()
        if name in ("agg", "prefill", "decode")
    }
    al = (
        None
        if real
        else resolve_acceptance_length(roles, engine, environment, reference)
    )
    overrides: list[RecipeOverride] = []
    for name, role in roles.items():
        prefix = f"roles.{name}"
        if engine == "vllm":
            raw = role.get("args", {}).get("speculative-config")
            if raw is None:
                continue
            spec = _spec_config(raw)
            if real:
                if (
                    spec.get("rejection_sample_method") != "synthetic"
                    and "synthetic_acceptance_length" not in spec
                ):
                    continue
                spec["rejection_sample_method"] = "block"
                spec.pop("synthetic_acceptance_length", None)
            else:
                spec.update(
                    rejection_sample_method="synthetic", synthetic_acceptance_length=al
                )
            overrides.append(
                RecipeOverride(
                    f"{prefix}.args.speculative-config",
                    json.dumps(spec, separators=(",", ":")),
                )
            )
        else:
            variables = SGLANG_VARIABLES if engine == "sglang" else (TRT_VARIABLE,)
            env = role.get("env") or {}
            if real:
                overrides.extend(
                    RecipeOverride(f"{prefix}.env.{key}", unset=True)
                    for key in variables
                    if key in env
                )
            else:
                if engine == "sglang" and SGLANG_VARIABLES[0] in env:
                    raise ValueError(
                        "recipe already contains SGLANG_SIMULATE_ACC_* variables"
                    )
                values = (
                    (f"{al:g}", "match-expected", "real-draft-token")
                    if engine == "sglang"
                    else (f"{al - 1:g}",)
                )
                overrides.extend(
                    RecipeOverride(f"{prefix}.env.{key}", value)
                    for key, value in zip(variables, values)
                )
    if not real and not overrides and require_match:
        raise ValueError(
            "SYNTHETIC_ACCEPTANCE=true but no speculative-config entries or worker roles were found"
        )
    return overrides


def selected_recipes(
    raw: dict[str, Any], selector: str | None
) -> list[tuple[str | None, dict[str, Any]]]:
    """Use upstream expansion, retaining a native selector for each variant."""
    if "base" not in raw:
        if selector is not None:
            raise ValueError("recipe selector requires an override-format recipe")
        return [(None, raw)]
    from srtctl.core.config import generate_override_configs

    # Validate the original selection with upstream, including empty globs and
    # bad indexes. Only selector routing lives here; merging/zipping stays there.
    selected = generate_override_configs(raw, selector=selector)
    if selector == "base" or (
        selector
        and (selector.startswith("override_") and not any(c in selector for c in "*?"))
    ):
        return [(selector, selected[0][1])]
    if selector and re.fullmatch(r"zip_override_[\w-]+\[\d+\]", selector):
        return [(selector, selected[0][1])]
    regular = sorted(k for k in raw if k.startswith("override_"))
    zipped = sorted(k for k in raw if k.startswith("zip_override_"))
    keys = (
        regular + zipped
        if selector is None
        else [k for k in sorted(regular + zipped) if fnmatch.fnmatch(k, selector)]
    )
    result = []
    for key in keys:
        variants = generate_override_configs(raw, selector=key)
        for index, (_, recipe) in enumerate(variants):
            native_selector = (
                f"{key}[{index}]" if key.startswith("zip_override_") else key
            )
            result.append((native_selector, recipe))
    return result


def plan_commands(
    config: str,
    framework: str,
    arguments: list[str],
    environment: Mapping[str, str],
    *,
    enable_throughput: bool,
    reference_path: Path = REFERENCE_PATH,
) -> list[list[str]]:
    """Plan all submissions before executing any; caller options stay intact."""
    command = ["srtctl", "apply", *arguments]
    real = enabled(environment, "EVAL_ONLY")
    if not real and not (
        enable_throughput and enabled(environment, "SYNTHETIC_ACCEPTANCE")
    ):
        return [command]
    if real and framework not in ENGINES:
        return [command]
    path, separator, selector = config.partition(":")
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("recipe must be a mapping")
    # Existing native overrides can set a variant's speculative JSON. Resolve
    # them first so adding acceptance settings preserves the caller's fields.
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--set", action="append")
    parser.add_argument("--unset", action="append")
    existing, _ = parser.parse_known_args(arguments)
    from srtctl.core.overrides import apply_overrides_to_recipe, parse_overrides

    apply_overrides_to_recipe(raw, parse_overrides(existing.set, existing.unset))
    reference = None
    if not real and not environment.get("SYNTHETIC_ACCEPTANCE_LENGTH", "").strip():
        reference = yaml.safe_load(reference_path.read_text(encoding="utf-8"))
    variants = selected_recipes(raw, selector if separator else None)
    commands = []
    any_overrides = False
    for variant, recipe in variants:
        overrides = build_overrides(
            recipe,
            framework,
            environment,
            reference=reference,
            enable_throughput=enable_throughput,
            require_match=len(variants) == 1,
        )
        any_overrides = any_overrides or bool(overrides)
        # Native srtctl applies all --unset options after all --set options,
        # regardless of argv order. Reject a caller removal that would silently
        # erase an acceptance setting we are about to add.
        for removal in existing.unset or []:
            if any(
                not item.unset
                and (item.path == removal or item.path.startswith(f"{removal}."))
                for item in overrides
            ):
                raise ValueError(
                    f"caller --unset {removal} conflicts with acceptance overrides"
                )
        # Native overrides replace zipped leaves with a one-element broadcast.
        # Check every resulting selection before submitting anything: changing
        # the only zip dimension must not submit an earlier variant and then
        # fail halfway through the group with an out-of-range index.
        overridden = copy.deepcopy(raw)
        apply_overrides_to_recipe(
            overridden,
            parse_overrides(
                [item.argv()[1] for item in overrides if not item.unset],
                [item.argv()[1] for item in overrides if item.unset],
            ),
        )
        selected_recipes(overridden, variant)
        selected_file = f"{path}:{variant}" if variant is not None else path
        # argparse's final --file wins over the caller's original group/glob.
        commands.append([*command, "--file", selected_file, *override_argv(overrides)])
    if not real and not any_overrides:
        raise ValueError(
            "SYNTHETIC_ACCEPTANCE=true but no speculative-config entries or worker roles were found"
        )
    return commands


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("framework")
    parser.add_argument("mode", choices=("throughput", "eval-only"))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    try:
        commands = plan_commands(
            args.config,
            args.framework,
            arguments,
            os.environ,
            enable_throughput=args.mode == "throughput",
        )
    except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
        print(f"ERROR: acceptance overrides: {exc}", file=sys.stderr)
        return 1
    for command in commands:
        result = subprocess.run(command, check=False)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
