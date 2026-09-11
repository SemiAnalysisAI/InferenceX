"""InferenceX recipe inputs and artifact contract for srt-slurm.

This module does not allocate nodes, launch processes, poll Slurm, or repair
hosts. The cluster launcher supplies paths; srt-slurm owns the job lifecycle.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import tarfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

_FORWARDED_ENV = (
    "AIPERF_EXPERIMENTAL_FAST",
    "CONC",
    "CONC_LIST",
    "DECODE_DP_ATTN",
    "DECODE_EP",
    "DECODE_NUM_WORKERS",
    "DECODE_PCP_SIZE",
    "DECODE_PP_SIZE",
    "DECODE_TP",
    "DURATION",
    "EVAL_CONC",
    "EVAL_FRAMEWORK",
    "EVAL_LIMIT",
    "EVAL_ONLY",
    "EVAL_SUITE",
    "FRAMEWORK",
    "IS_AGENTIC",
    "ISL",
    "KV_OFFLOADING",
    "MAX_MODEL_LEN",
    "MODEL",
    "MODEL_PREFIX",
    "PREFILL_DP_ATTN",
    "PREFILL_EP",
    "PREFILL_NUM_WORKERS",
    "PREFILL_PCP_SIZE",
    "PREFILL_PP_SIZE",
    "PREFILL_TP",
    "PRECISION",
    "RANDOM_RANGE_RATIO",
    "RESULT_FILENAME",
    "RUN_EVAL",
    "RUNNER_TYPE",
    "OSL",
    "SPEC_DECODING",
    "SWEBENCH_GEN_MODE",
    "TOTAL_CPU_DRAM_GB",
)

_EVAL_COMMAND = r"""
set -euo pipefail
eval_root="/results/${SLURM_JOB_ID}/eval"
mkdir -p "${eval_root}"
cd "${eval_root}"
export SRTCTL_LM_EVAL_RESULT_DIR="${eval_root}"
source /infmax-workspace/benchmarks/benchmark_lib.sh
export EVAL_SERVER_HOST="${SRT_FRONTEND_HOST}"
if [[ -n "${EVAL_CONC:-}" ]]; then
  export EVAL_CONCURRENT_REQUESTS="${EVAL_CONC}"
else
  export EVAL_CONCURRENT_REQUESTS="$(printf '%s\n' "${CONC_LIST:-${CONC:-1}}" | tr ' ' '\n' | sort -n | tail -1)"
fi
export CONC="${EVAL_CONCURRENT_REQUESTS}"
bridge_disagg_eval_metadata
run_eval --port "${SRT_FRONTEND_PORT}"
# AgentX eval-only runs stage their artifacts inside run_eval.
if [[ "${EVAL_ONLY:-false}" != "true" ]] || \
   [[ "${IS_AGENTIC:-0}" != "1" && "${SCENARIO_TYPE:-}" != "agentic-coding" ]]; then
  append_lm_eval_summary
fi
""".strip()


def prepare_recipe(
    recipe: dict[str, Any],
    profile: dict[str, Any],
    environment: Mapping[str, str],
    *,
    workspace: Path,
    results_root: Path,
    aiperf_cache: Path,
    image_cache: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Adapt CI metadata without changing the recipe's serving contract."""
    recipe = copy.deepcopy(recipe)
    profile = copy.deepcopy(profile)
    profile.setdefault("default_mounts", {}).update(
        {
            str(workspace): "/infmax-workspace",
            str(results_root): "/results",
            str(aiperf_cache): "/aiperf_mmap_cache",
        }
    )
    image = environment["IMAGE"]
    cached_image = image_cache / (image.replace("/", "_").replace(":", "_") + ".sqsh")
    # Reuse a provisioned image when available. Otherwise Pyxis imports the
    # recipe's image during its normal container lifecycle, not a staging job.
    profile.setdefault("containers", {})[recipe["model"]["container"]] = (
        str(cached_image) if cached_image.is_file() else image
    )
    benchmark_env = recipe.setdefault("benchmark", {}).setdefault("env", {})
    for key in _FORWARDED_ENV:
        value = environment.get(key)
        if value:
            benchmark_env[key] = value

    _configure_evaluation(recipe, environment)
    return recipe, profile


def _configure_evaluation(recipe: dict[str, Any], environment: Mapping[str, str]) -> None:
    benchmark_env = recipe["benchmark"]["env"]
    eval_only = environment.get("EVAL_ONLY", "false").lower() == "true"
    run_eval = environment.get("RUN_EVAL", "false").lower() == "true"
    if run_eval and not eval_only:
        raise ValueError("srt-slurm requires a separate EVAL_ONLY=true job for evaluation")
    if eval_only:
        benchmark_env["SRTCTL_LM_EVAL_RESULT_DIR"] = "/results/{job_id}/eval"
        backend = recipe.get("backend", {})
        for role in ("prefill", "decode", "aggregated"):
            role_env = backend.get(f"{role}_environment", {})
            for key in (
                "SGLANG_SIMULATE_ACC_LEN",
                "SGLANG_SIMULATE_ACC_METHOD",
                "SGLANG_SIMULATE_ACC_TOKEN_MODE",
            ):
                role_env.pop(key, None)
        server_config = recipe.get("backend", {}).get("sglang_config", {})
        for mode in ("prefill", "decode", "aggregated"):
            server_config.get(mode, {}).pop("ep-dispatch-algorithm", None)

        resources = recipe.get("resources", {})
        prefill = server_config.get("prefill", server_config.get("aggregated", {}))
        decode = server_config.get("decode", prefill)

        def topology_value(config: dict[str, Any], *keys: str, default: int = 1) -> int:
            for key in keys:
                if key in config:
                    return int(config[key])
            return default

        topology_defaults = {
            "IS_MULTINODE": "true",
            "MODEL_NAME": environment["MODEL"],
            "EVAL_MAX_MODEL_LEN": str(prefill.get("context-length", environment.get("MAX_MODEL_LEN", "16384"))),
            "PREFILL_TP": str(topology_value(prefill, "tp-size", "tensor-parallel-size")),
            "PREFILL_EP": str(topology_value(prefill, "ep-size", "expert-parallel-size")),
            "PREFILL_NUM_WORKERS": str(resources.get("prefill_workers", resources.get("agg_workers", 1))),
            "DECODE_TP": str(topology_value(decode, "tp-size", "tensor-parallel-size")),
            "DECODE_EP": str(topology_value(decode, "ep-size", "expert-parallel-size")),
            "DECODE_NUM_WORKERS": str(resources.get("decode_workers", resources.get("agg_workers", 1))),
            "PREFILL_DP_ATTN": str(prefill.get("enable-dp-attention", False)).lower(),
            "DECODE_DP_ATTN": str(decode.get("enable-dp-attention", False)).lower(),
        }
        for key, value in topology_defaults.items():
            benchmark_env.setdefault(key, value)

        recipe["benchmark"]["command"] = _EVAL_COMMAND


def collect_results(
    submission: dict[str, Any],
    environment: Mapping[str, str],
    *,
    workspace: Path,
    results_root: Path,
) -> None:
    """Collect only this allocation's artifacts into the workflow workspace."""
    job_id = str(submission["slurm_job_id"])
    if not job_id.isdecimal():
        raise ValueError("Submission must identify one numeric Slurm job")
    log_dir = Path(submission["output_dir"]) / "logs"
    result_dir = results_root / job_id
    if log_dir.is_dir():
        with tarfile.open(workspace / "multinode_server_logs.tar.gz", "w:gz") as archive:
            archive.add(log_dir, arcname=".")
            lockfile = log_dir.parent / "recipe.lock.yaml"
            if lockfile.is_file():
                archive.add(lockfile, arcname="recipe.lock.yaml")
    if result_dir.is_dir():
        shutil.copytree(result_dir, workspace / "LOGS", dirs_exist_ok=True)

    filename = environment["RESULT_FILENAME"]
    eval_only = environment.get("EVAL_ONLY", "false").lower() == "true"
    if not eval_only and environment.get("IS_AGENTIC", "0") == "1":
        for result in result_dir.glob(f"{filename}_conc*.json"):
            shutil.copy2(result, workspace / result.name)
        if not list(workspace.glob(f"{filename}_conc*.json")):
            raise ValueError(f"No AgentX aggregate results found for {filename}")
    elif not eval_only:
        results = sorted((result_dir / "fixed-seq").glob("*.json"))
        if not results:
            raise ValueError(f"No fixed-sequence results found in {result_dir}")
        prefill_gpus = int(environment["PREFILL_NUM_WORKERS"]) * int(environment["PREFILL_TP"])
        if environment.get("DISAGG", "false").lower() == "true":
            decode_gpus = int(environment["DECODE_NUM_WORKERS"]) * int(environment["DECODE_TP"])
            suffix = f"gpus_{prefill_gpus + decode_gpus}_ctx_{prefill_gpus}_gen_{decode_gpus}"
        else:
            total = (
                prefill_gpus
                * int(environment.get("PREFILL_PP_SIZE", "1"))
                * int(environment.get("PREFILL_PCP_SIZE", "1"))
            )
            suffix = f"gpus_{total}"
        for result in results:
            match = re.search(r"-c([0-9]+)\.json$", result.name)
            if not match:
                raise ValueError(f"Cannot parse concurrency from {result}")
            destination = workspace / f"{filename}_srt-{job_id}_conc{match[1]}_{suffix}.json"
            shutil.copy2(result, destination)
            print(f"Collected {destination}")

    if eval_only or environment.get("RUN_EVAL", "false").lower() == "true":
        eval_dir = result_dir / "eval"
        if not (eval_dir / "meta_env.json").is_file():
            raise ValueError(f"No eval metadata found in {eval_dir}")
        for artifact in eval_dir.glob("*"):
            if artifact.is_file():
                shutil.copy2(artifact, workspace / artifact.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--recipe", type=Path, required=True)
    prepare.add_argument("--profile", type=Path, required=True)
    prepare.add_argument("--work-dir", type=Path, required=True)
    prepare.add_argument("--aiperf-cache", type=Path, required=True)
    prepare.add_argument("--image-cache", type=Path, required=True)
    collect = commands.add_parser("collect")
    collect.add_argument("--submission", type=Path, required=True)
    for command in (prepare, collect):
        command.add_argument("--workspace", type=Path, required=True)
        command.add_argument("--results-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        recipe, profile = prepare_recipe(
            yaml.safe_load(args.recipe.read_text()),
            yaml.safe_load(args.profile.read_text()),
            os.environ,
            workspace=args.workspace,
            results_root=args.results_root,
            aiperf_cache=args.aiperf_cache,
            image_cache=args.image_cache,
        )
        profile.setdefault("output_dir", str(args.work_dir / "outputs"))
        (args.work_dir / "recipe.yaml").write_text(yaml.safe_dump(recipe, sort_keys=False))
        (args.work_dir / "srtslurm.yaml").write_text(yaml.safe_dump(profile, sort_keys=False))
    else:
        collect_results(
            json.loads(args.submission.read_text()),
            os.environ,
            workspace=args.workspace,
            results_root=args.results_root,
        )


if __name__ == "__main__":
    main()
