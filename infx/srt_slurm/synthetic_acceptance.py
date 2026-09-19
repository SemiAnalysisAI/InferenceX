"""Select AgentX golden acceptance automatically and pass native srtctl overrides."""

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

GOLDEN_DIR = Path(__file__).resolve().parents[2] / "golden_al_distribution"
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


def spec_parameters(role: Mapping[str, Any], engine: str) -> dict[str, Any]:
    args = role.get("args", {})
    if engine == "vllm":
        raw = args.get("speculative-config")
        if raw is None:
            return {}
        spec = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(spec, dict):
            raise ValueError("speculative-config must be a JSON object")
        return dict(spec)
    if engine == "sglang":
        algorithms = {
            str(args[key]).lower()
            for key in ("speculative-algorithm", "speculative-algo")
            if key in args
        }
        if len(algorithms) > 1:
            raise ValueError("Conflicting speculative-algorithm and speculative-algo values")
        algorithm = next(iter(algorithms), "")
        if not algorithm:
            return {}
        return {
            "method": "dspark" if algorithm == "dspark" else algorithm,
            "num_speculative_tokens": args.get(
                "speculative-dspark-block-size"
                if algorithm == "dspark"
                else "speculative-num-steps"
            ),
            "model": args.get("speculative-draft-model-path", ""),
        }
    spec = args.get("speculative_config") or {}
    if not spec:
        return {}
    return {
        "method": str(spec.get("decoding_type", "")).lower(),
        "num_speculative_tokens": spec.get("max_draft_len"),
        "model": spec.get("speculative_model", ""),
    }


def golden_length(model: str, spec: Mapping[str, Any], thinking: str, golden_dir: Path) -> float:
    method = str(spec.get("method", "")).lower()
    # SGLang calls native model MTP EAGLE/NEXTN; the curve describes the model's head.
    if method in ("eagle", "nextn"):
        method = "eagle3" if model in ("kimik2.5", "minimaxm3") else "mtp"
    curve = f"{model}_{method}"
    if model in ("dsv4", "dsv4dspark", "dsv4dsparkprob") and method == "dspark":
        curve = "dsv4-pro-0813-dspark"
    elif model == "minimaxm3" and method == "eagle3":
        if "gqa" in str(spec.get("model", "")).lower():
            curve += "_gqa"
    elif model == "kimik3" and method == "dspark":
        # Kimi has distinct measured curves; require the recipe to choose its sampler.
        sampling = spec.get("draft_sample_method")
        if sampling == "probabilistic":
            curve += "_probabilistic_sample_method_block_rejection_sample_method"
        elif sampling != "greedy":
            raise ValueError(f"No Kimi DSpark golden curve for draft sampling {sampling!r}")
    if not re.fullmatch(r"[a-z0-9_.-]+", curve):
        raise ValueError(f"Invalid golden curve identity: {curve!r}")
    tokens = spec.get("num_speculative_tokens")
    if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens <= 0:
        raise ValueError("Speculative decoding requires a positive integer draft length")
    path = golden_dir / f"{curve}.yaml"
    if not path.is_file():
        raise ValueError(f"No committed golden curve for {model}/{method}: {path.name}")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or len(data) != 1:
        raise ValueError(f"Expected one model in golden curve {path.name}")
    modes = next(iter(data.values()))
    try:
        value = float(modes[thinking][tokens])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"No golden acceptance for {curve}/{thinking}/{tokens} draft tokens"
        ) from error
    if not math.isfinite(value) or not 1 <= value <= tokens + 1:
        raise ValueError(f"Invalid golden acceptance {value} in {path.name}")
    return value


def build_overrides(
    recipe: Mapping[str, Any],
    framework: str,
    environment: Mapping[str, str],
    *,
    golden_dir: Path = GOLDEN_DIR,
) -> list[str]:
    """Infer acceptance from generation parameters; never accept a caller AL."""
    engine = ENGINES.get(framework)
    if engine is None:
        return []
    roles = recipe.get("roles", {})
    synthetic = (
        environment["EVAL_ONLY"].lower() != "true"
        and environment["IS_AGENTIC"].lower() in ("1", "true")
        and environment["SPEC_DECODING"] != "none"
    )
    # Prefill may have a different MTP depth; generation defines the AL target.
    generation = roles.get("decode", roles.get("agg", {}))
    spec = spec_parameters(generation, engine)
    al = None
    if synthetic and spec:
        al = golden_length(
            environment["MODEL_PREFIX"], spec, environment["THINKING_MODE"], golden_dir
        )
    overrides = []
    variables = {"sglang": SGLANG_VARIABLES, "trtllm": (TRT_VARIABLE,)}.get(engine, ())
    # SRT applies recipe-wide environment after role environment. Keep simulation
    # role-local so global values cannot override the golden AL or leak into evals.
    for key in variables:
        if key in (recipe.get("environment") or {}):
            overrides += ["--unset", f"environment.{key}"]
    for name, role in roles.items():
        if name not in ("agg", "prefill", "decode"):
            continue
        prefix = f"roles.{name}"
        worker_spec = spec_parameters(role, engine)
        if engine == "vllm":
            if not worker_spec:
                continue
            if al is not None:
                worker_spec.update(
                    rejection_sample_method="synthetic", synthetic_acceptance_length=al
                )
            elif (
                worker_spec.get("rejection_sample_method") == "synthetic"
                or "synthetic_acceptance_length" in worker_spec
            ):
                worker_spec["rejection_sample_method"] = "block"
                worker_spec.pop("synthetic_acceptance_length", None)
            else:
                continue
            overrides += [
                "--set",
                f"{prefix}.args.speculative-config={json.dumps(worker_spec)}",
            ]
        elif al is not None and worker_spec:
            values = (
                (f"{al:g}", "match-expected", "real-draft-token")
                if engine == "sglang"
                else (f"{al - 1:g}",)
            )
            for key, value in zip(variables, values, strict=True):
                overrides += ["--set", f"{prefix}.env.{key}={json.dumps(value)}"]
        else:
            for key in variables:
                if key in (role.get("env") or {}):
                    overrides += ["--unset", f"{prefix}.env.{key}"]
    return overrides


def selected_recipes(
    raw: dict[str, Any], selector: str | None
) -> list[tuple[str | None, dict[str, Any]]]:
    """Delegate expansion to SRT, retaining its selector for each submission."""
    if "base" not in raw:
        if selector is not None:
            raise ValueError("recipe selector requires an override-format recipe")
        return [(None, raw)]
    from srtctl.core.config import generate_override_configs

    selected = generate_override_configs(raw, selector=selector)
    if (
        selector == "base"
        or (selector and selector.startswith("override_") and not any(c in selector for c in "*?"))
        or (selector and re.fullmatch(r"zip_override_[\w-]+\[\d+\]", selector))
    ):
        return [(selector, selected[0][1])]
    keys = sorted(k for k in raw if k.startswith("override_")) + sorted(
        k for k in raw if k.startswith("zip_override_")
    )
    result = []
    for key in keys:
        if selector is not None and not fnmatch.fnmatch(key, selector):
            continue
        for index, (_, recipe) in enumerate(generate_override_configs(raw, selector=key)):
            name = f"{key}[{index}]" if key.startswith("zip_override_") else key
            result.append((name, recipe))
    return result


def plan_commands(
    config: str,
    framework: str,
    arguments: list[str],
    environment: Mapping[str, str],
    *,
    golden_dir: Path = GOLDEN_DIR,
) -> list[list[str]]:
    """Build native arguments for every selected variant before submitting jobs."""
    command = ["srtctl", "apply", *arguments]
    if framework not in ENGINES:
        return [command]
    from srtctl.core.overrides import apply_overrides_to_recipe, parse_overrides

    path, _, selector = config.partition(":")
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("Recipe must be a mapping")
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--set", action="append")
    parser.add_argument("--unset", action="append")
    existing, _ = parser.parse_known_args(arguments)
    caller_overrides = parse_overrides(existing.set, existing.unset)
    apply_overrides_to_recipe(raw, caller_overrides)
    commands = []
    for variant, recipe in selected_recipes(raw, selector or None):
        arguments_to_add = build_overrides(recipe, framework, environment, golden_dir=golden_dir)
        parsed, _ = parser.parse_known_args(arguments_to_add)
        generated_overrides = parse_overrides(parsed.set, parsed.unset)
        for removal in caller_overrides:
            if removal.unset and any(
                not item.unset and item.path[: len(removal.path)] == removal.path
                for item in generated_overrides
            ):
                raise ValueError(f"Caller {removal.render()} conflicts with golden acceptance")
        # SRT broadcasts overrides into zip groups. Reject a collapsed selection
        # before any job is submitted, rather than selecting the wrong variant.
        materialized = copy.deepcopy(raw)
        apply_overrides_to_recipe(materialized, generated_overrides)
        selected_recipes(materialized, variant)
        selected_file = f"{path}:{variant}" if variant is not None else path
        commands.append([*command, "--file", selected_file, *arguments_to_add])
    return commands


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("framework")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    try:
        commands = plan_commands(args.config, args.framework, arguments, os.environ)
    except (OSError, KeyError, ValueError, TypeError, yaml.YAMLError) as error:
        print(f"ERROR: golden acceptance: {error}", file=sys.stderr)
        return 1
    for command in commands:
        result = subprocess.run(command, check=False)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
