"""Bind a native single-node SRT recipe to one fixed-sequence matrix point."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path

import yaml

from infx.srt_slurm.synthetic_acceptance import selected_recipes, spec_parameters


def runtime_arguments(config: str, environment: Mapping[str, str]) -> list[str]:
    """Reject mismatched metadata before allocation; retain recipe-owned server settings."""
    path, _, selector = config.partition(":")
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("Recipe must be a mapping")
    recipes = selected_recipes(raw, selector or None)
    if len(recipes) != 1:
        raise ValueError("A single-node matrix point must select exactly one SRT recipe")
    recipe = recipes[0][1]
    role = recipe["roles"]["agg"]
    args = role["args"]
    benchmark = recipe["benchmark"]
    workload = benchmark["env"]
    spec = spec_parameters(role, "sglang")
    if spec and spec["method"] not in {"eagle", "nextn"}:
        raise ValueError("Single-node SRT pilot supports only native MTP or no speculation")
    speculation = "mtp" if spec else "none"
    expected = {
        "engine": (recipe["engine"], environment["FRAMEWORK"]),
        "model": (recipe["model"]["path"], f"hf:{environment['MODEL']}"),
        "image": (recipe["model"]["container"], environment["IMAGE"]),
        "precision": (recipe["model"]["precision"], environment["PRECISION"]),
        "tensor-parallel-size": (args["tensor-parallel-size"], int(environment["TP"])),
        "data-parallel-size": (args["data-parallel-size"], 1),
        "expert-parallel-size": (
            args.get("expert-parallel-size", args.get("ep-size", 1)),
            int(environment["EP_SIZE"]),
        ),
        "gpus": (role["gpus"], int(environment["GPU_COUNT"])),
        "nodes": (role["nodes"], 1),
        "workers": (role["workers"], 1),
        "roles": (set(recipe["roles"]), {"agg"}),
        "benchmark type": (benchmark["type"], "custom"),
        "benchmark MODEL": (workload["MODEL"], environment["MODEL"]),
        "SPEC_DECODING": (speculation, environment["SPEC_DECODING"]),
        "USE_CHAT_TEMPLATE": (workload["USE_CHAT_TEMPLATE"], "true" if spec else "false"),
    }
    if "CONC" in workload:
        expected["CONC"] = (str(workload["CONC"]), environment["CONC"])
    for name in ("ISL", "OSL", "RANDOM_RANGE_RATIO"):
        expected[name] = (str(workload[name]), environment[name])
    # Other topology/eval paths remain on their current launchers until ported.
    for name, value in {
        "FRAMEWORK": "sglang",
        "PP_SIZE": "1",
        "DCP_SIZE": "1",
        "PCP_SIZE": "1",
        "DP_ATTENTION": "false",
        "IS_AGENTIC": "0",
        "RUN_EVAL": "false",
        "EVAL_ONLY": "false",
    }.items():
        expected[name] = (environment[name], value)
    for name, (actual, wanted) in expected.items():
        if actual != wanted:
            raise ValueError(f"Single-node SRT {name}: recipe/matrix {actual!r} != {wanted!r}")
    overrides = []
    for name in ("CONC", "RESULT_FILENAME", "GPU_MONITOR_INTERVAL", "RUN_EVAL", "EVAL_ONLY"):
        value = environment[name]
        if not value:
            raise ValueError(f"Missing runtime input: {name}")
        overrides += ["--set", f"benchmark.env.{name}={json.dumps(value)}"]
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
            arguments = runtime_arguments(parsed.recipe, os.environ)
            parsed.output.write_bytes("\0".join([*arguments, ""]).encode())
        else:
            print("\n".join(submission_fields(parsed.manifest)))
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
