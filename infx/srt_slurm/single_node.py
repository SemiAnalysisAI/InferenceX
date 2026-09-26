"""Bind a native single-node SRT recipe to one fixed-sequence, AgentX, or SPEED-Bench point."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from infx.srt_slurm.synthetic_acceptance import ENGINES, selected_recipes, spec_parameters

SINGLE_NODE_ENGINES = {**ENGINES, "atom": "atom"}

SPEEDBENCH_CLIENT = "srt_speedbench.sh"


def _is_speedbench(benchmark: dict[str, Any]) -> bool:
    """Detect a SPEED-Bench AL collector from its benchmark command."""
    return benchmark.get("command", "").endswith(SPEEDBENCH_CLIENT)


def parallelism_constraints(
    engine: str, args: Mapping[str, Any], environment: Mapping[str, str]
) -> dict[str, tuple[Any, Any]]:
    """Read each engine's native topology fields without translating the recipe."""
    tp, ep = int(environment["TP"]), int(environment["EP_SIZE"])
    dp_attention = environment["DP_ATTENTION"] == "true"
    if engine == "sglang":
        return {
            "tensor-parallel-size": (args["tensor-parallel-size"], tp),
            "data-parallel-size": (args.get("data-parallel-size", 1), tp if dp_attention else 1),
            "expert-parallel-size": (args.get("expert-parallel-size", args.get("ep-size", 1)), ep),
            "DP_ATTENTION": (args.get("enable-dp-attention", False), dp_attention),
        }
    if engine == "trtllm":
        return {
            "tensor_parallel_size": (args["tensor_parallel_size"], tp),
            "moe_expert_parallel_size": (args["moe_expert_parallel_size"], ep),
            "pipeline_parallel_size": (args.get("pipeline_parallel_size", 1), 1),
            "DP_ATTENTION": (args.get("enable_attention_dp", False), dp_attention),
        }
    if engine == "vllm":
        # vLLM spreads DP attention across data-parallel ranks of tensor size 1.
        data_parallel = args.get("data-parallel-size", 1)
        return {
            "tensor x data parallel": (args.get("tensor-parallel-size", 1) * data_parallel, tp),
            "DP_ATTENTION": (data_parallel > 1, dp_attention),
            "enable-expert-parallel": (args.get("enable-expert-parallel", False), ep > 1),
        }
    if engine == "atom":
        if ep not in {1, tp}:
            raise ValueError("ATOM expert parallelism must be 1 or TP")
        return {
            "enable-expert-parallel": (args.get("enable-expert-parallel", False), ep > 1),
            "DP_ATTENTION": (args.get("enable-dp-attention", False), dp_attention),
        }
    raise ValueError(f"Unsupported single-node SRT engine: {engine!r}")


def select_recipe(config: str, environment: Mapping[str, str]) -> tuple[str, dict[str, Any]]:
    """Resolve a matrix point to one native variant, never submit an entire sweep."""
    path, _, selector = config.partition(":")
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("Recipe must be a mapping")
    recipes = selected_recipes(raw, selector or None)
    matches = []
    errors = []
    for name, recipe in recipes:
        try:
            validate_recipe(recipe, environment)
        except ValueError as exc:
            errors.append(f"{name}: {exc}")
        else:
            matches.append((f"{path}:{name}" if name else path, recipe))
    if len(matches) != 1:
        detail = "; ".join(errors) if not matches else ", ".join(name for name, _ in matches)
        raise ValueError(f"Expected exactly one matching single-node SRT recipe; {detail}")
    return matches[0]


def validate_recipe(recipe: dict[str, Any], environment: Mapping[str, str]) -> None:
    """Reject metadata mismatches without overwriting recipe-owned server settings."""
    role = recipe["roles"]["agg"]
    args = role["args"]
    benchmark = recipe["benchmark"]
    workload = benchmark["env"]
    engine_config = recipe["engine"]
    engine = engine_config["type"] if isinstance(engine_config, dict) else engine_config
    if environment["FRAMEWORK"] not in {"sglang", "trt", "atom", "vllm"}:
        raise ValueError(f"Unsupported single-node framework: {environment['FRAMEWORK']!r}")
    spec = spec_parameters(role, engine)
    if spec and spec["method"] not in {"eagle", "eagle3", "nextn", "mtp", "dspark"}:
        raise ValueError(
            "Single-node SRT supports only native MTP, EAGLE3, DSpark or no speculation"
        )
    # A point that stops drafting may keep its matrix label.
    speculation = "mtp" if spec else workload.get("SPEC_DECODING", "none")
    agentic = environment["IS_AGENTIC"] == "1"
    collector = _is_speedbench(benchmark)
    expected = {
        "engine": (engine, SINGLE_NODE_ENGINES[environment["FRAMEWORK"]]),
        "model": (recipe["model"]["path"], f"hf:{environment['MODEL']}"),
        "image": (recipe["model"]["container"], environment["IMAGE"]),
        "precision": (recipe["model"]["precision"], environment["PRECISION"]),
        **parallelism_constraints(engine, args, environment),
        "gpus": (role["gpus"], int(environment["GPU_COUNT"])),
        "nodes": (role["nodes"], 1),
        "workers": (role["workers"], 1),
        "roles": (set(recipe["roles"]), {"agg"}),
        "benchmark type": (benchmark["type"], "custom"),
        "benchmark MODEL": (workload["MODEL"], environment["MODEL"]),
        # draft_model names a bundled or separate draft; its recipes speculate natively.
        "SPEC_DECODING": (
            speculation,
            "mtp"
            if environment["SPEC_DECODING"] == "draft_model"
            else environment["SPEC_DECODING"],
        ),
    }
    if not collector:
        expected["AgentX client"] = (
            benchmark.get("command", "").endswith("srt_agentic.sh"),
            agentic,
        )
    if not agentic and not collector:
        expected["USE_CHAT_TEMPLATE"] = (workload["USE_CHAT_TEMPLATE"], "true" if spec else "false")
        for name in ("ISL", "OSL", "RANDOM_RANGE_RATIO"):
            expected[name] = (str(workload[name]), environment[name])
    # A variant that names its point, or the host budget it sizes, must match the matrix.
    for name in ("CONC", "KV_OFFLOADING", "TOTAL_CPU_DRAM_GB"):
        if name in workload:
            expected[name] = (str(workload[name]), environment[name])
    if engine == "atom":
        # Native ATOM derives -tp from the aggregate worker's GPU allocation.
        expected["ATOM TP"] = (role["gpus"], int(environment["TP"]))
    # vLLM shards decode KV across its tensor-parallel ranks.
    dcp = str(args.get("decode-context-parallel-size", 1)) if engine == "vllm" else "1"
    for name, value in {"PP_SIZE": "1", "DCP_SIZE": dcp, "PCP_SIZE": "1"}.items():
        expected[name] = (environment[name], value)
    for name, (actual, wanted) in expected.items():
        if actual != wanted:
            raise ValueError(f"Single-node SRT {name}: recipe/matrix {actual!r} != {wanted!r}")


def runtime_arguments(config: str, environment: Mapping[str, str]) -> list[str]:
    """Bind only runtime-owned values after validating the selected recipe."""
    _, recipe = select_recipe(config, environment)
    for name in ("RUN_EVAL", "EVAL_ONLY", "DP_ATTENTION"):
        if environment[name] not in {"true", "false"}:
            raise ValueError(f"{name} must be true or false")
    # Exclusive nodes include idle GPUs. Restrict each server/client step to
    # the serving GPU count so client-side power collection sees the same set.
    overrides = ["--set", f"srun_options.gpus-per-node={json.dumps(environment['GPU_COUNT'])}"]
    # Match the legacy container working directory using the existing repo mount.
    # PyTorch's generated module imports fail from / with PYTHONPYCACHEPREFIX set.
    overrides += ["--set", 'srun_options.container-workdir="/infmax-workspace"']
    if environment.get("SRT_SRUN_OPTIONS"):
        options = json.loads(environment["SRT_SRUN_OPTIONS"])
        if not isinstance(options, dict) or any(
            not re.fullmatch(r"[a-z][a-z0-9-]*", key) or not isinstance(value, str)
            for key, value in options.items()
        ):
            raise ValueError("SRT_SRUN_OPTIONS must map option names to string values")
        # Native --set preserves whole mappings as JSON strings for engine
        # flags. Runtime option mappings therefore need individual leaf sets.
        for key, value in options.items():
            overrides += ["--set", f"srun_options.{key}={json.dumps(value)}"]
    agentic = environment["IS_AGENTIC"] == "1"
    collector = _is_speedbench(recipe["benchmark"])
    names = [
        "CONC",
        "RESULT_FILENAME",
        "GPU_MONITOR_INTERVAL",
        "RUN_EVAL",
        "EVAL_ONLY",
        "FRAMEWORK",
    ]
    if agentic:
        names += ["MODEL_PREFIX", "PRECISION", "DURATION", "TP", "PP_SIZE", "PCP_SIZE"]
    for name in names:
        value = environment[name]
        if not value:
            raise ValueError(f"Missing runtime input: {name}")
        # Native --set broadcasts into zip groups. CONC already matched above;
        # replacing its list could collapse the selected variant's index.
        if name == "CONC" and name in recipe["benchmark"]["env"]:
            continue
        overrides += ["--set", f"benchmark.env.{name}={json.dumps(value)}"]
    if collector:
        # SPEED-Bench collector tunables: bind workflow-level settings into
        # the per-cell benchmark environment so the client reads them.
        for name in ("CATEGORY", "SPEEDBENCH_OUTPUT_LEN"):
            value = environment.get(name, "")
            if not value:
                raise ValueError(f"Missing SPEED-Bench input: {name}")
            overrides += ["--set", f"benchmark.env.{name}={json.dumps(value)}"]
        # Chat-template kwargs are optional (dsr1 has no off mode).
        for name in ("CHAT_TEMPLATE_KWARGS_ON",):
            value = environment.get(name, "")
            if value:
                overrides += ["--set", f"benchmark.env.{name}={json.dumps(value)}"]
        return [*overrides, "--set", 'benchmark.env.RESULT_DIR="/logs"']
    if agentic:
        # The aggregated result lands where fixed-sequence results do.
        overrides += ["--set", 'benchmark.env.AGENTIC_OUTPUT_DIR="/logs"']
        return [*overrides, "--set", 'benchmark.env.RESULT_DIR="/logs/agentic"']
    if environment["EVAL_ONLY"] == "true":
        context = int(environment["MAX_MODEL_LEN"])
        if context <= 0:
            raise ValueError("MAX_MODEL_LEN must be positive")
        context_keys = {
            "sglang": ("context-length",),
            "trt": ("max_seq_len", "max_num_tokens"),
            "atom": ("max-model-len",),
            "vllm": ("max-model-len",),
        }[environment["FRAMEWORK"]]
        for key in context_keys:
            overrides += ["--set", f"roles.agg.args.{key}={context}"]
    return [*overrides, "--set", 'benchmark.env.RESULT_DIR="/logs"']


def submission_fields(path: Path) -> tuple[str, str]:
    """Accept exactly one successful native JSON submission, never scrape prose."""
    record = json.loads(path.read_text())
    if record.get("status") != "submitted":
        raise ValueError("SRT did not submit a job")
    job_id = str(record["slurm_job_id"])
    output = str(record["output_dir"])
    if not job_id.isascii() or not job_id.isdecimal() or int(job_id) <= 0:
        raise ValueError("Invalid SRT Slurm job ID")
    if not Path(output).is_absolute() or "\n" in output:
        raise ValueError("SRT output directory must be absolute")
    return job_id, output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("recipe")
    prepare.add_argument("output", type=Path)
    submitted = commands.add_parser("submission")
    submitted.add_argument("manifest", type=Path)
    parsed = parser.parse_args()
    try:
        if parsed.command == "prepare":
            config, _ = select_recipe(parsed.recipe, os.environ)
            arguments = runtime_arguments(parsed.recipe, os.environ)
            parsed.output.write_bytes("\0".join([config, *arguments, ""]).encode())
        else:
            print("\n".join(submission_fields(parsed.manifest)))
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
