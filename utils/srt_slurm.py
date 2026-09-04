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
append_lm_eval_summary
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
            # Native setup_script paths resolve beneath /configs. Keep recipe
            # assets in InferenceX without copying them into the runtime checkout.
            str(workspace / "benchmarks/multi_node/srt-slurm-recipes"): "/configs/inferencex",
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

    _configure_sglang_contract(recipe, environment)
    _configure_evaluation(recipe, environment)
    return recipe, profile


def _configure_sglang_contract(recipe: dict[str, Any], environment: Mapping[str, str]) -> None:
    # The original SGLang launcher sized DP+EP admission from the largest
    # concurrency exercised by a recipe. It also honored a model-specific MoRI
    # dispatch pin when present; only the inter-kernel switch threshold was
    # derived per topology. Preserve those semantics rather than treating MTP
    # draft tokens as additional independent requests.
    if environment.get("PREFILL_DP_ATTN", "false").lower() == "true" and int(environment.get("PREFILL_EP", "1")) > 1:
        concurrency_text = environment.get("CONC_LIST") or environment.get("CONC")
        if not concurrency_text:
            raise ValueError("DP+EP recipe requires CONC_LIST or CONC")
        concurrency_values = concurrency_text.split()
        concurrency = max(int(value) for value in concurrency_values)
        prefill = recipe["backend"]["sglang_config"]["prefill"]
        decode = recipe["backend"]["sglang_config"]["decode"]
        prefill["max-running-requests"] = concurrency
        decode["max-running-requests"] = concurrency

        decode_tp = int(environment["DECODE_TP"])
        decode_environment = recipe["backend"]["decode_environment"]
        dispatch_tokens = max(1, concurrency // decode_tp)
        decode_environment.setdefault("SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK", str(dispatch_tokens))
        # The retired launcher also exposed its harness-level dispatch budget to
        # the server environment after scaling it by the MTP draft width. Keep that
        # auxiliary value distinct from the model-specific per-rank SGLang pin.
        if "MORI_MAX_DISPATCH_TOKENS_DECODE" in decode_environment:
            mtp_size = int(environment.get("DECODE_MTP_SIZE", "0"))
            decode_environment["MORI_MAX_DISPATCH_TOKENS_DECODE"] = str(dispatch_tokens * (mtp_size + 1))
        decode_environment["SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD"] = str(2 * dispatch_tokens)


def _configure_evaluation(recipe: dict[str, Any], environment: Mapping[str, str]) -> None:
    benchmark_env = recipe["benchmark"]["env"]
    eval_only = environment.get("EVAL_ONLY", "false").lower() == "true"
    run_eval = environment.get("RUN_EVAL", "false").lower() == "true"
    if eval_only or run_eval:
        benchmark_env["SRTCTL_LM_EVAL_RESULT_DIR"] = "/results/{job_id}/eval"
        decode_env = recipe.get("backend", {}).get("decode_environment", {})
        for key in (
            "SGLANG_SIMULATE_ACC_LEN",
            "SGLANG_SIMULATE_ACC_METHOD",
            "SGLANG_SIMULATE_ACC_TOKEN_MODE",
        ):
            decode_env.pop(key, None)
        server_config = recipe.get("backend", {}).get("sglang_config", {})
        for mode in ("prefill", "decode"):
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

        if eval_only:
            recipe["benchmark"]["command"] = _EVAL_COMMAND
        else:
            recipe["benchmark"]["command"] = recipe["benchmark"]["command"].rstrip() + "\n" + _EVAL_COMMAND


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
    if result_dir.is_dir():
        shutil.copytree(result_dir, workspace / "LOGS", dirs_exist_ok=True)

    filename = environment["RESULT_FILENAME"]
    eval_only = environment.get("EVAL_ONLY", "false").lower() == "true"
    if not eval_only and environment.get("IS_AGENTIC", "0") == "1":
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
        if eval_only and not (eval_dir / "meta_env.json").is_file():
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
