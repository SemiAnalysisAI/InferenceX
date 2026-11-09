"""Select AgentX golden acceptance automatically and pass native srtctl overrides."""

from __future__ import annotations

import argparse
import copy
import fnmatch
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

import yaml

from infx import yamlio
from infx.bench.proc import status
from infx.golden_al_distribution import GOLDEN_DIR, golden_length
from infx.golden_al_distribution.identity import Fact
from infx.srt_slurm.variants import ZIP_VARIANT, without_nulls

ENGINES = {
    "sglang": "sglang",
    "sglang-disagg": "sglang",
    "vllm-disagg": "vllm",
    "vllm": "vllm",
    "dynamo-vllm": "vllm",
    "llmd-vllm": "vllm",
    "dynamo-sglang": "sglang",
    "trt": "trtllm",
    "dynamo-trt": "trtllm",
    "atom": "atom",
    "atom-disagg": "atom",
    "tilert": "tilert",
    "tokenspeed": "tokenspeed",
}
SGLANG_VARIABLES = (
    "SGLANG_SIMULATE_ACC_LEN",
    "SGLANG_SIMULATE_ACC_METHOD",
    "SGLANG_SIMULATE_ACC_TOKEN_MODE",
)
TRT_VARIABLE = "TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS"
TILERT_VARIABLES = ("TILERT_SIMULATE_ACC_LEN", "TILERT_SIMULATE_ACC_METHOD")
# Synthetic acceptance streams unverified draft tokens. Kimi's parsers stop streaming on them.
KIMI_VLLM_PARSER_ARGS = ("enable-auto-tool-choice", "tool-call-parser", "reasoning-parser")


def spec_parameters(role: Mapping[str, Any], engine: str) -> dict[str, Any]:
    """Return a role's speculative config for ``engine`` as method, draft length and draft model.

    Empty when the role does not speculate. DFlash's draft length is its verify block minus the
    current token. Raises ValueError on a malformed or conflicting config.
    """
    args = role.get("args", {})
    if engine == "tokenspeed":
        if "speculative-config" in args:
            raise ValueError(
                "TokenSpeed AgentX requires explicit speculative flags, not speculative-config"
            )
        algorithm = str(args.get("speculative-algorithm", "")).lower()
        if not algorithm:
            return {}
        if algorithm == "mtp":
            return {
                "method": "mtp",
                "num_speculative_tokens": args.get("speculative-num-steps"),
            }
        if algorithm == "dflash":
            # DFlash drafts its steps in one block; the verify window adds the current token.
            return {
                "method": "dflash",
                "num_speculative_tokens": args.get("speculative-num-steps"),
                "model": args.get("speculative-draft-model-path", ""),
            }
        raise ValueError("TokenSpeed AgentX supports MTP and DFLASH speculation")
    if engine == "tilert":
        if args.get("with-mtp") is not True:
            return {}
        method = "dspark" if args.get("model") == "dsv41" else "mtp"
        return {"method": method, "num_speculative_tokens": args.get("num-mtp")}
    if engine == "atom":
        method = args.get("method")
        if not method:
            return {}
        spec = {
            "method": str(method).lower(),
            "num_speculative_tokens": args.get("num-speculative-tokens"),
            "model": args.get("draft-model", ""),
        }
        if spec["method"] == "dspark":
            spec["draft_sample_method"] = "probabilistic"
        return spec
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
        if algorithm == "dflash":
            width = args.get("speculative-num-draft-tokens")
            return {
                "method": "dflash",
                "num_speculative_tokens": width - 1
                if isinstance(width, int) and not isinstance(width, bool)
                else None,
                "model": args.get("speculative-draft-model-path", ""),
            }
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


def verifier(role: Mapping[str, Any], engine: str) -> Fact:
    """How a role verifies drafts. Block, relaxed and adaptive verification are opt-in, so an
    unset flag means standard; a synthetic sampler becomes block on real runs. ATOM and TileRT
    do not expose their verifier."""
    args = role.get("args") or {}
    if engine == "vllm":
        method = spec_parameters(role, engine).get("rejection_sample_method")
        if method == "synthetic":
            return Fact("block", "documented")
        return Fact(method, "explicit") if method else Fact("standard", "documented")
    if engine == "sglang" and args.get("speculative-use-block-verification") is True:
        return Fact("block", "explicit")
    config = args.get("speculative_config") or {}
    if engine == "trtllm" and config.get("use_relaxed_acceptance_for_thinking") is True:
        return Fact("relaxed", "explicit")
    if engine in ("sglang", "trtllm"):
        return Fact("standard", "documented")
    return Fact(None, "unknown")


def acceptance_length(
    recipe: Mapping[str, Any],
    framework: str,
    environment: Mapping[str, str],
    *,
    golden_dir: Path = GOLDEN_DIR,
) -> float | None:
    """The golden AL the generation role of a speculative AgentX throughput run simulates,
    else None."""
    roles = recipe.get("roles", {})
    generation = roles.get("decode", roles.get("agg", {}))
    if (
        ENGINES.get(framework) == "vllm"
        and spec_parameters(generation, "vllm").get("enable_adaptive_verification") is True
    ):
        return None
    return _simulated_length(recipe, framework, environment, golden_dir=golden_dir)


def _simulated_length(
    recipe: Mapping[str, Any],
    framework: str,
    environment: Mapping[str, str],
    *,
    golden_dir: Path = GOLDEN_DIR,
) -> float | None:
    """The golden AL ``build_overrides`` sets on non-adaptive speculative roles, else None.

    Raises ValueError unless a Qwen3.5 SGLang MTP generation role's window is its draft steps
    plus one verification token.
    """
    engine = ENGINES.get(framework)
    if engine is None:
        return None
    roles = recipe.get("roles", {})
    synthetic = is_synthetic(environment)
    generation = generation_role(recipe)
    spec = spec_parameters(generation, engine)
    if not (synthetic and spec):
        return None
    if engine == "vllm" and all(
        spec_parameters(role, engine).get("enable_adaptive_verification") is True
        for name, role in roles.items()
        if name in ("agg", "prefill", "decode") and spec_parameters(role, engine)
    ):
        return None
    al = golden_length(
        environment["MODEL_PREFIX"],
        spec,
        environment["THINKING_MODE"],
        golden_dir,
        verifier(generation, engine).value,
    )
    if engine == "tokenspeed":
        args = generation.get("args", {})
        drafts = args.get("speculative-num-draft-tokens")
        if (
            not isinstance(drafts, int)
            or isinstance(drafts, bool)
            or drafts != spec["num_speculative_tokens"] + 1
        ):
            raise ValueError(
                "TokenSpeed speculative-num-draft-tokens must equal speculative-num-steps + 1"
            )
        if args.get("speculative-eagle-topk") != 1:
            raise ValueError("TokenSpeed synthetic acceptance requires speculative-eagle-topk 1")
    if (
        engine == "sglang"
        and environment["MODEL_PREFIX"] == "qwen3.5"
        and spec.get("method") in ("eagle", "nextn", "mtp")
    ):
        drafts = generation.get("args", {}).get("speculative-num-draft-tokens")
        expected = spec["num_speculative_tokens"] + 1
        if not isinstance(drafts, int) or isinstance(drafts, bool) or drafts != expected:
            raise ValueError(
                "Qwen3.5 SGLang generation role speculative-num-draft-tokens "
                f"must be an integer equal to speculative-num-steps + 1 ({expected})"
            )
    return al


def is_synthetic(environment: Mapping[str, str]) -> bool:
    """AgentX speculative throughput forces golden acceptance; evals verify for real."""
    return (
        environment["EVAL_ONLY"].lower() != "true"
        and environment["IS_AGENTIC"].lower() in ("1", "true")
        and environment["SPEC_DECODING"] != "none"
    )


def generation_role(recipe: Mapping[str, Any]) -> Mapping[str, Any]:
    """The worker whose drafts are verified: decode, otherwise the aggregated worker."""
    roles = recipe.get("roles", {})
    return roles.get("decode", roles.get("agg", {}))


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
    al = _simulated_length(recipe, framework, environment, golden_dir=golden_dir)
    overrides = []
    variables = {
        "sglang": SGLANG_VARIABLES,
        "trtllm": (TRT_VARIABLE,),
        "tilert": TILERT_VARIABLES,
    }.get(engine, ())
    for key in variables:
        if key in (recipe.get("environment") or {}):
            overrides += ["--unset", f"environment.{key}"]
    for name, role in roles.items():
        if name not in ("agg", "prefill", "decode"):
            continue
        prefix = f"roles.{name}"
        worker_engine = engine
        worker_al = al
        if engine == "tilert":
            selected = role.get("engine", recipe.get("engine", "tilert"))
            worker_engine = selected.get("type") if isinstance(selected, Mapping) else selected
            if name != "decode" or worker_engine != "tilert":
                worker_al = None
        worker_spec = spec_parameters(role, worker_engine)
        if engine == "tilert" and (worker_al is None or not worker_spec):
            for key in TILERT_VARIABLES:
                if key in (role.get("env") or {}):
                    overrides += ["--unset", f"{prefix}.env.{key}"]
        if worker_engine == "vllm":
            if not worker_spec:
                continue
            if worker_spec.get("enable_adaptive_verification") is True:
                if (
                    worker_spec.get("rejection_sample_method") == "synthetic"
                    or "synthetic_acceptance_length" in worker_spec
                ):
                    worker_spec["rejection_sample_method"] = "block"
                    worker_spec.pop("synthetic_acceptance_length", None)
                else:
                    continue
            elif worker_al is not None:
                worker_spec.update(
                    rejection_sample_method="synthetic", synthetic_acceptance_length=worker_al
                )
                if environment["MODEL_PREFIX"].startswith("kimi"):
                    for key in KIMI_VLLM_PARSER_ARGS:
                        if key in (role.get("args") or {}):
                            overrides += ["--unset", f"{prefix}.args.{key}"]
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
        elif worker_engine == "tokenspeed":
            key = "synthetic-acceptance-length"
            sampled = "enable-speculative-sampling"
            if worker_al is not None and worker_spec:
                overrides += ["--set", f"{prefix}.args.{key}={worker_al:g}"]
                overrides += ["--set", f'{prefix}.args.sampling-backend="greedy"']
                # The greedy sampler verifies by exact match and rejects sampled verification.
                if sampled in (role.get("args") or {}):
                    overrides += ["--unset", f"{prefix}.args.{sampled}"]
            else:
                if key in (role.get("args") or {}):
                    overrides += ["--unset", f"{prefix}.args.{key}"]
                if worker_spec.get("method") == "dflash":
                    # Only the triton sampler reads the DFlash2 draft distribution.
                    overrides += ["--set", f'{prefix}.args.sampling-backend="triton"']
                    overrides += ["--set", f"{prefix}.args.{sampled}=true"]
                elif worker_spec:
                    overrides += ["--set", f'{prefix}.args.sampling-backend="flashinfer"']
        elif worker_engine == "atom":
            key = "spec-decode-acceptance-length"
            if worker_al is not None and worker_spec:
                overrides += ["--set", f"{prefix}.args.{key}={worker_al:g}"]
            elif key in (role.get("args") or {}):
                overrides += ["--unset", f"{prefix}.args.{key}"]
        elif worker_al is not None and worker_spec:
            values = (
                (f"{worker_al:g}", "match-expected", "real-draft-token")
                if engine == "sglang"
                else (f"{worker_al:g}", "match-expected")
                if engine == "tilert"
                else (f"{worker_al - 1:g}",)
            )
            for key, value in zip(variables, values, strict=True):
                overrides += ["--set", f"{prefix}.env.{key}={json.dumps(value)}"]
        elif engine != "tilert":
            for key in variables:
                if key in (role.get("env") or {}):
                    overrides += ["--unset", f"{prefix}.env.{key}"]
    return overrides


def selected_recipes(
    raw: dict[str, Any], selector: str | None
) -> list[tuple[str | None, dict[str, Any]]]:
    """Delegate expansion to SRT, retaining its selector for each submission.

    ``srtctl apply`` merges a zip variant over its base as an override, which drops the
    variant's null-valued keys (``connector: null`` falls back to the engine default);
    the selected zip variants match that.
    """
    if "base" not in raw:
        if selector is not None:
            raise ValueError("recipe selector requires an override-format recipe")
        return [(None, raw)]
    from srtctl.core.config import generate_override_configs

    selected = generate_override_configs(raw, selector=selector)
    zip_variant = selector and ZIP_VARIANT.fullmatch(selector)
    if zip_variant:
        return [(selector, without_nulls(selected[0][1]))]
    if selector == "base" or (
        selector and selector.startswith("override_") and not any(c in selector for c in "*?")
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
            if key.startswith("zip_override_"):
                result.append((f"{key}[{index}]", without_nulls(recipe)))
            else:
                result.append((key, recipe))
    return result


class Planned(NamedTuple):
    """One ``srtctl apply`` command and the golden AL it simulates (None: real acceptance).

    ``al_gate`` is an agentic eval's acceptance gate (``al_gate.plan``), which ``environment``
    adds to the submission's environment to run; None without one.
    """

    command: list[str]
    acceptance_length: float | None
    al_gate: dict[str, Any] | None = None
    environment: tuple[tuple[str, str], ...] = ()


def plan_commands(
    config: str,
    framework: str,
    arguments: list[str],
    environment: Mapping[str, str],
    *,
    golden_dir: Path = GOLDEN_DIR,
) -> list[Planned]:
    """Build native arguments for every selected variant before submitting jobs."""
    from infx.srt_slurm import al_gate

    command = ["srtctl", "apply", *arguments]
    if framework not in ENGINES:
        gate = al_gate.plan({}, framework, environment) if al_gate.requested(environment) else None
        return [Planned([*command, "--file", config], None, gate)]
    from srtctl.core.overrides import apply_overrides_to_recipe, parse_overrides

    path, _, selector = config.partition(":")
    raw = yamlio.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("Recipe must be a mapping")
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--set", action="append")
    parser.add_argument("--unset", action="append")
    existing, _ = parser.parse_known_args(arguments)
    caller_overrides = parse_overrides(existing.set, existing.unset)
    apply_overrides_to_recipe(raw, caller_overrides)
    plans = []
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
        materialized = copy.deepcopy(raw)
        apply_overrides_to_recipe(materialized, generated_overrides)
        selected_recipes(materialized, variant)
        selected_file = f"{path}:{variant}" if variant is not None else path
        length = acceptance_length(recipe, framework, environment, golden_dir=golden_dir)
        gate, srt_environment = None, {}
        if al_gate.requested(environment):
            gate = al_gate.plan(recipe, framework, environment, golden_dir=golden_dir)
            if gate is not None and gate.get("status") != "skipped":
                arguments_to_add = [*arguments_to_add, *al_gate.overrides(gate, environment)]
                srt_environment = al_gate.SRT_ENVIRONMENT
        argv = [*command, "--file", selected_file, *arguments_to_add]
        plans.append(Planned(argv, length, gate, tuple(srt_environment.items())))
    return plans


def main(argv: list[str] | None = None) -> int:
    """Plan srtctl commands with golden acceptance from ``argv``, run them and return the rc."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--record",
        type=Path,
        help="write the plan's golden acceptance length and acceptance gate here as JSON",
    )
    parser.add_argument("config")
    parser.add_argument("framework")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    try:
        plans = plan_commands(args.config, args.framework, arguments, os.environ)
    except (OSError, KeyError, ValueError, TypeError, yaml.YAMLError) as error:
        print(f"ERROR: golden acceptance: {error}", file=sys.stderr)
        return 1
    if args.record is not None:
        lengths = {plan.acceptance_length for plan in plans}
        length = lengths.pop() if len(lengths) == 1 else None
        gate = next((plan.al_gate for plan in plans if plan.al_gate is not None), None)
        record = {"golden_acceptance_length": length, "al_gate": gate}
        args.record.write_text(json.dumps(record) + "\n")
    for plan in plans:
        result = subprocess.run(
            plan.command, env={**os.environ, **dict(plan.environment)}, check=False
        )
        if result.returncode:
            return status(result.returncode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
