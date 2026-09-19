"""InferenceX workflow inputs and artifact contract for srt-slurm.

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

from infx.srt_slurm.cluster_config import render_cluster_config

_FORWARDED_ENV = (
    "AIPERF_EXPERIMENTAL_FAST",
    "AIPERF_DRAIN_TIMEOUT_SECONDS",
    "AIPERF_DRAIN_POLL_SECONDS",
    "CLEAR_CACHE_BETWEEN_CONC",
    "FLUSH_DRAIN_TIMEOUT",
    "CONC",
    "CONC_LIST",
    "DECODE_DP_ATTN",
    "DECODE_EP",
    "DECODE_NUM_WORKERS",
    "DECODE_PCP_SIZE",
    "DECODE_PP_SIZE",
    "DECODE_TP",
    "DECODE_HARDWARE",
    "DISAGG",
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
    "KV_OFFLOAD_BACKEND",
    "KV_OFFLOAD_BACKEND_METADATA",
    "KV_P2P_TRANSFER",
    "MAX_MODEL_LEN",
    "MODEL",
    "MODEL_PREFIX",
    "PREFILL_DP_ATTN",
    "PREFILL_EP",
    "PREFILL_NUM_WORKERS",
    "PREFILL_PCP_SIZE",
    "PREFILL_PP_SIZE",
    "PREFILL_TP",
    "PREFILL_HARDWARE",
    "PRECISION",
    "RANDOM_RANGE_RATIO",
    "REQUIRE_POWER",
    "RESULT_FILENAME",
    "RUN_EVAL",
    "RUNNER_TYPE",
    "RUNNER_NAME",
    "SCENARIO_TYPE",
    "OSL",
    "SPEC_DECODING",
    "SWEBENCH_GEN_MODE",
    "SWEBENCH_USE_MODAL",
    "MODAL_TOKEN_ID",
    "MODAL_TOKEN_SECRET",
    "THINKING_MODE",
    "TOTAL_CPU_DRAM_GB",
    "WEKA_LOADER_OVERRIDE",
)

_EVAL_COMMAND = r"""
set -eo pipefail
source /infmax-workspace/benchmarks/benchmark_lib.sh --validation-only
check_env_vars SLURM_JOB_ID SRT_FRONTEND_HOST SRT_FRONTEND_PORT CONC_LIST
eval_root="/results/${SLURM_JOB_ID}/eval"
mkdir -p "${eval_root}"
cd "${eval_root}"
source /infmax-workspace/benchmarks/benchmark_lib.sh
export EVAL_SERVER_HOST="${SRT_FRONTEND_HOST}"
if [[ -n "${EVAL_CONC:-}" ]]; then
  export EVAL_CONCURRENT_REQUESTS="${EVAL_CONC}"
else
  export EVAL_CONCURRENT_REQUESTS="$(printf '%s\n' "$CONC_LIST" | tr ' ' '\n' | sort -n | tail -1)"
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
    if "base" in recipe:
        from srtctl.core.config import generate_override_configs

        _, separator, selector = environment["CONFIG_FILE"].partition(":")
        if not separator:
            raise ValueError("CONFIG_FILE must select one recipe variant")
        variants = generate_override_configs(recipe, selector=selector)
        if len(variants) != 1:
            raise ValueError("Each CI job must select exactly one recipe variant")
        recipe = variants[0][1]
    profile = render_cluster_config(profile, dict(environment), {})
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
    if environment.get("CLIENT_IMAGE"):
        recipe["benchmark"]["container_image"] = environment["CLIENT_IMAGE"]
    for key in (
        *_FORWARDED_ENV,
        *environment.get("INFERENCEX_RUNTIME_ENV_VARS", "").split(),
    ):
        value = environment.get(key)
        if value:
            benchmark_env[key] = value

    if recipe.get("telemetry", {}).get("enabled"):
        recipe["benchmark"]["concurrencies"] = [
            int(value) for value in environment["CONC_LIST"].split()
        ]
        benchmark_env.update(
            SRT_MEASUREMENT_WINDOW_BENCHMARK_TYPE=recipe["benchmark"]["type"],
            SRT_MEASUREMENT_WINDOW_CONCURRENCIES=environment["CONC_LIST"],
            SRT_MEASUREMENT_WINDOW_RESULT_ROOT="/logs",
        )

    _configure_sglang_contract(recipe, environment)
    _configure_evaluation(recipe, environment)
    return recipe, profile


def _configure_sglang_contract(recipe: dict[str, Any], environment: Mapping[str, str]) -> None:
    if recipe.get("engine") != "sglang":
        return
    roles = recipe.get("roles", {})
    prefill = roles.get("prefill", {}).get("args", {})
    decode = roles.get("decode", {}).get("args", {})
    # The Pro-0813 recipe leaves this range empty for the workflow's actual
    # concurrency, just as models.yaml sizes each legacy allocation at launch.
    if decode.get("cuda-graph-bs-decode") != []:
        if prefill.get("enable-dp-attention") and int(environment.get("PREFILL_EP", "1")) > 1:
            concurrency = max(int(value) for value in environment["CONC_LIST"].split())
            prefill["max-running-requests"] = concurrency
            decode["max-running-requests"] = concurrency
            decode_env = roles["decode"].setdefault("env", {})
            dispatch_tokens = max(1, concurrency // int(environment["DECODE_TP"]))
            decode_env.setdefault(
                "SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK", str(dispatch_tokens)
            )
            if "MORI_MAX_DISPATCH_TOKENS_DECODE" in decode_env:
                mtp_size = int(environment.get("DECODE_MTP_SIZE", "0"))
                decode_env["MORI_MAX_DISPATCH_TOKENS_DECODE"] = str(
                    dispatch_tokens * (mtp_size + 1)
                )
            decode_env["SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD"] = str(
                2 * dispatch_tokens
            )
        return
    concurrency = max(int(value) for value in environment["CONC_LIST"].split())
    if concurrency <= 0:
        raise ValueError("CONC_LIST must contain positive concurrency values")
    for role, config in (("PREFILL", prefill), ("DECODE", decode)):
        if int(environment[f"{role}_TP"]) != config["tp-size"]:
            raise ValueError(f"{role}_TP disagrees with the selected recipe")
        dp = environment[f"{role}_DP_ATTN"].lower() == "true"
        if dp != config.get("enable-dp-attention", False):
            raise ValueError(f"{role}_DP_ATTN disagrees with the selected recipe")
        config["max-running-requests"] = concurrency * 2
        if environment.get("DISABLE_CUSTOM_ALL_REDUCE") == "1":
            config["disable-custom-all-reduce"] = True
    graph_max = concurrency // 4 if decode.get("enable-dp-attention") else concurrency * 2
    if graph_max < 1:
        raise ValueError("Concurrency is too small for the DP decode graph range")
    decode["cuda-graph-bs-decode"] = list(range(1, graph_max + 1))
    if decode.get("enable-dp-attention"):
        decode["max-running-requests"] = min(concurrency * 2, graph_max * decode["tp-size"])
    if prefill.get("enable-hierarchical-cache") and "HICACHE_RATIO" in environment:
        prefill["hicache-ratio"] = float(environment["HICACHE_RATIO"])
    if "PREFILL_ROUTER_POLICY" in environment:
        recipe["frontend"]["args"]["policy"] = environment["PREFILL_ROUTER_POLICY"]


def _configure_evaluation(recipe: dict[str, Any], environment: Mapping[str, str]) -> None:
    benchmark_env = recipe["benchmark"]["env"]
    eval_only = environment.get("EVAL_ONLY", "false").lower() == "true"
    run_eval = environment.get("RUN_EVAL", "false").lower() == "true"
    if eval_only or run_eval:
        roles = recipe.get("roles", {})
        decode_env = roles.get("decode", {}).get("env", {})
        for key in (
            "SGLANG_SIMULATE_ACC_LEN",
            "SGLANG_SIMULATE_ACC_METHOD",
            "SGLANG_SIMULATE_ACC_TOKEN_MODE",
        ):
            decode_env.pop(key, None)
        server_config = {name: role.get("args", {}) for name, role in roles.items()}
        for mode in ("prefill", "decode"):
            server_config.get(mode, {}).pop("ep-dispatch-algorithm", None)

        prefill = server_config.get("prefill", server_config.get("agg", {}))
        decode = server_config.get("decode", prefill)

        def topology_value(config: dict[str, Any], *keys: str, default: int = 1) -> int:
            for key in keys:
                if key in config:
                    return int(config[key])
            return default

        topology_defaults = {
            "IS_MULTINODE": "true",
            "MODEL_NAME": environment["MODEL"],
            "EVAL_MAX_MODEL_LEN": str(
                prefill.get("context-length", environment.get("MAX_MODEL_LEN", "16384"))
            ),
            "PREFILL_TP": str(topology_value(prefill, "tp-size", "tensor-parallel-size")),
            "PREFILL_EP": str(topology_value(prefill, "ep-size", "expert-parallel-size")),
            "PREFILL_NUM_WORKERS": str(
                roles.get("prefill", roles.get("agg", {})).get("workers", 1)
            ),
            "DECODE_TP": str(topology_value(decode, "tp-size", "tensor-parallel-size")),
            "DECODE_EP": str(topology_value(decode, "ep-size", "expert-parallel-size")),
            "DECODE_NUM_WORKERS": str(roles.get("decode", roles.get("agg", {})).get("workers", 1)),
            "PREFILL_DP_ATTN": str(prefill.get("enable-dp-attention", False)).lower(),
            "DECODE_DP_ATTN": str(decode.get("enable-dp-attention", False)).lower(),
        }
        for key, value in topology_defaults.items():
            benchmark_env.setdefault(key, value)

        if eval_only:
            recipe["benchmark"]["command"] = _EVAL_COMMAND
        else:
            recipe["benchmark"]["command"] = (
                recipe["benchmark"]["command"].rstrip() + "\n" + _EVAL_COMMAND
            )


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
    for name in ("agentic", "power"):
        source = log_dir / name
        if source.is_dir():
            shutil.copytree(source, workspace / "LOGS" / name, dirs_exist_ok=True)

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
        eval_dir = log_dir / "eval_results"
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
