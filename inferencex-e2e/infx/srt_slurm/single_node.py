"""Bind a native single-node SRT recipe to one fixed-sequence or AgentX matrix point."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from infx.config import repository_root
from infx.srt_slurm.synthetic_acceptance import ENGINES, selected_recipes, spec_parameters
from infx.srt_slurm.workload import (
    BOUND_RECIPE,
    add_fabric_argument,
    bind_workload,
    check_setup_script,
    compose_recipe,
    resolve_fabric,
)

SINGLE_NODE_ENGINES = {**ENGINES, "atom": "atom"}


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


def select_recipe(
    config: str,
    environment: Mapping[str, str],
    *,
    root: Path | None = None,
    expand: Callable[..., list[tuple[str | None, dict[str, Any]]]] = selected_recipes,
) -> tuple[str, dict[str, Any]]:
    """Resolve a matrix point to one native variant, never submit an entire sweep.

    Variants are composed with the shared block under ``root`` (default: this
    repository) and bound before they are validated. ``expand`` lists the selected
    variants: srtctl at launch, ``variants.expand_variants`` in the planner.
    """
    path, _, selector = config.partition(":")
    agentic = environment["IS_AGENTIC"] == "1"
    raw = compose_recipe(
        Path(path),
        agentic=agentic,
        multinode=False,
        root=root or repository_root(),
        native_power=environment.get("EVAL_ONLY") != "true",
    )
    recipes = expand(raw, selector or None)
    matches = []
    errors = []
    for name, recipe in recipes:
        try:
            recipe = bind_workload(recipe, environment, agentic=agentic, multinode=False)
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
    if spec and spec["method"] not in {"eagle", "eagle3", "nextn", "mtp", "dspark", "dflash"}:
        raise ValueError(
            "Single-node SRT supports only native MTP, EAGLE3, DSpark, DFlash or no speculation"
        )
    speculation = "mtp" if spec else workload.get("SPEC_DECODING", "none")
    agentic = environment["IS_AGENTIC"] == "1"
    expected = {
        "engine": (engine, SINGLE_NODE_ENGINES[environment["FRAMEWORK"]]),
        **parallelism_constraints(engine, args, environment),
        "gpus": (role["gpus"], int(environment["GPU_COUNT"])),
        "nodes": (role["nodes"], 1),
        "workers": (role["workers"], 1),
        "roles": (set(recipe["roles"]), {"agg"}),
        "benchmark type": (benchmark["type"], "custom"),
        "SPEC_DECODING": (
            speculation,
            "mtp"
            if environment["SPEC_DECODING"] == "draft_model"
            else environment["SPEC_DECODING"],
        ),
        "AgentX client": (benchmark.get("command", "").endswith("srt_agentic.sh"), agentic),
    }
    for name in ("CONC", "KV_OFFLOADING", "TOTAL_CPU_DRAM_GB"):
        if name in workload:
            expected[name] = (str(workload[name]), environment[name])
    if engine == "atom":
        expected["ATOM TP"] = (role["gpus"], int(environment["TP"]))
    dcp = str(args.get("decode-context-parallel-size", 1)) if engine in {"vllm", "atom"} else "1"
    for name, value in {"PP_SIZE": "1", "DCP_SIZE": dcp, "PCP_SIZE": "1"}.items():
        expected[name] = (environment[name], value)
    for name, (actual, wanted) in expected.items():
        if actual != wanted:
            raise ValueError(f"Single-node SRT {name}: recipe/matrix {actual!r} != {wanted!r}")


PROFILE_DIR = "/logs/infx_profile"
PROFILE_DEFAULTS: dict[str, Any] = {
    "windows": [["warmup", 0, 32], ["decode", 0, 32]],
    "capture_ranks": None,
    "capture_graphs_per_file": 0,
    "host_headroom_gib": 128,
    "mode": "agentic",
    "isl": 8192,
    "osl": 4096,
    "engine_args": {},
}
PROFILE_MODES = ("agentic", "synthetic")
PROFILE_PHASES = ("warmup", "decode", "profiling")
PROFILE_MEASURED_CAP_SECONDS = 600


def offload_headroom_arguments(role_args: Mapping[str, Any], headroom_gib: float) -> list[str]:
    """Shrink a CPU KV-offload pool so profiling leaves the host headroom_gib free."""
    raw = role_args.get("kv-transfer-config")
    if not raw or headroom_gib <= 0:
        return []
    transfer = json.loads(raw) if isinstance(raw, str) else dict(raw)
    extra = transfer.get("kv_connector_extra_config") or {}
    per_rank = extra.get("cpu_bytes_to_use_per_rank")
    if per_rank is None:
        return []
    ranks = int(role_args.get("data-parallel-size", 1)) * int(
        role_args.get("tensor-parallel-size", 1)
    )
    shrunk = int(per_rank) - int(headroom_gib * 2**30 / ranks)
    if shrunk <= 0:
        raise ValueError("INFX_PROFILE host_headroom_gib exceeds the recipe's CPU offload pool")
    transfer["kv_connector_extra_config"] = {**extra, "cpu_bytes_to_use_per_rank": shrunk}
    return ["--set", f"roles.agg.args.kv-transfer-config={json.dumps(transfer)}"]


def mooncake_headroom_arguments(
    recipe: Mapping[str, Any], role_args: Mapping[str, Any], headroom_gib: float
) -> list[str]:
    """Shrink an embedded Mooncake store's per-rank segment so profiling leaves the host headroom_gib.

    Each TP rank registers global_segment_size of host memory for RDMA; with the
    profiler's buffers on top, B300 Kimi K3 ran out registering it.
    """
    if headroom_gib <= 0:
        return []
    ranks = int(role_args.get("data-parallel-size", 1)) * int(
        role_args.get("tensor-parallel-size", 1)
    )
    services = copy.deepcopy(list(recipe.get("services") or []))
    changed = False
    for service in services:
        if not isinstance(service, Mapping) or service.get("type") != "mooncake-master":
            continue
        store = (service.get("options") or {}).get("store_config") or {}
        match = re.fullmatch(
            r"(\d+(?:\.\d+)?)\s*([KMGT]?B)",
            str(store.get("global_segment_size") or ""),
            re.IGNORECASE,
        )
        if not match:
            continue
        scale = {"KB": 1 / 2**20, "MB": 1 / 2**10, "GB": 1, "TB": 2**10}[match.group(2).upper()]
        shrunk = float(match.group(1)) * scale - headroom_gib / ranks
        if shrunk <= 0:
            raise ValueError("INFX_PROFILE host_headroom_gib exceeds the recipe's Mooncake segment")
        store["global_segment_size"] = f"{int(shrunk)}GB"
        changed = True
    return ["--set", f"services={json.dumps(services)}"] if changed else []


def _recipe_env_int(recipe: Mapping[str, Any] | None, name: str) -> int:
    """An integer the recipe's agg role (or the whole recipe) sets in its environment, else 0."""
    for env in (
        ((recipe or {}).get("roles") or {}).get("agg", {}).get("env") or {},
        (recipe or {}).get("environment") or {},
    ):
        try:
            return int(env[name])
        except (KeyError, TypeError, ValueError):
            continue
    return 0


def _set(path: str, value: Any) -> list[str]:
    return ["--set", f"{path}={json.dumps(value)}"]


@dataclass(frozen=True)
class ProfileRequest:
    """One profiled point: INFX_PROFILE's settings and the selected recipe variant."""

    settings: Mapping[str, Any]
    role_args: Mapping[str, Any]
    recipe: Mapping[str, Any]

    @property
    def headroom_gib(self) -> float:
        return float(self.settings["host_headroom_gib"])


class ProfiledEngine:
    """How one engine is profiled: its hooks, how windows open, and the recipe changes that needs.

    Every engine loads the infx_prof hooks through PYTHONPATH (sitecustomize);
    INFX_PROF_ENGINE picks its adapter. `control` is how the window client
    opens windows: "http" (the engine's own profiler endpoint) or "file"
    (control/windows.jsonl, read by the hooks at step boundaries).
    """

    name = ""
    control = "file"
    attention_dp_arg = ""

    def worker_env(self, profile: ProfileRequest) -> dict[str, str]:
        return {
            "INFX_PROF_DIR": PROFILE_DIR,
            "INFX_PROF_ENGINE": self.name,
            "INFX_PROF_CAPTURE_GRAPHS_PER_FILE": str(
                int(profile.settings["capture_graphs_per_file"])
            ),
            "INFX_PROF_CAPTURE_RANKS": profile.settings["capture_ranks"]
            or self.capture_ranks(profile),
            "PYTHONPATH": "/infmax-workspace/benchmarks/profiling",
            "TORCHINDUCTOR_FX_GRAPH_CACHE": "0",
            "TORCHINDUCTOR_AUTOGRAD_CACHE": "0",
        }

    def role_arguments(self, profile: ProfileRequest) -> list[str]:
        """--set overrides of the engine's arguments (and services) for a profiled run."""
        raise NotImplementedError

    def capture_ranks(self, profile: ProfileRequest) -> str:
        """Workers whose graph capture is profiled when INFX_PROFILE does not say: every rank
        under attention DP (SGLang / TRT-LLM ranks capture different graphs: per-rank
        buffers change copy kernels), else dp0_tp0."""
        dp = self.attention_dp_arg and profile.role_args.get(self.attention_dp_arg)
        return "all" if dp else "dp0_tp0"


class VllmProfiling(ProfiledEngine):
    name = "vllm"
    control = "http"

    def worker_env(self, profile: ProfileRequest) -> dict[str, str]:
        timeout = max(1800, _recipe_env_int(profile.recipe, "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"))
        ready = max(3600, _recipe_env_int(profile.recipe, "VLLM_ENGINE_READY_TIMEOUT_S"))
        return {
            **super().worker_env(profile),
            "VLLM_RPC_TIMEOUT": "1800000",
            "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": str(timeout),
            "VLLM_ENGINE_READY_TIMEOUT_S": str(ready),
            "VLLM_DISABLE_COMPILE_CACHE": "1",
        }

    def role_arguments(self, profile: ProfileRequest) -> list[str]:
        profiler_config = {
            "profiler": "torch",
            "torch_profiler_dir": f"{PROFILE_DIR}/torch",
            "torch_profiler_record_shapes": True,
            "torch_profiler_with_stack": False,
            "torch_profiler_dump_cuda_time_total": False,
            "ignore_frontend": True,
            "max_iterations": int(profile.settings["windows"][0][2]),
        }
        return [
            "--set",
            f"roles.agg.args.profiler-config={json.dumps(profiler_config)}",
            *offload_headroom_arguments(profile.role_args, profile.headroom_gib),
            *mooncake_headroom_arguments(profile.recipe, profile.role_args, profile.headroom_gib),
        ]


class SglangProfiling(ProfiledEngine):
    name = "sglang"

    attention_dp_arg = "enable-dp-attention"

    TIMEOUT_SECONDS = 1800

    def role_arguments(self, profile: ProfileRequest) -> list[str]:
        args = profile.role_args
        overrides = []
        for flag in ("watchdog-timeout", "dist-timeout"):
            overrides += _set(
                f"roles.agg.args.{flag}", max(self.TIMEOUT_SECONDS, int(args.get(flag) or 0))
            )
        size = args.get("hicache-size")
        if args.get("enable-hierarchical-cache") and size and profile.headroom_gib > 0:
            shrunk = int(
                float(size) - profile.headroom_gib / int(args.get("tensor-parallel-size", 1))
            )
            if shrunk <= 0:
                raise ValueError("INFX_PROFILE host_headroom_gib exceeds the recipe's HiCache pool")
            overrides += _set("roles.agg.args.hicache-size", shrunk)
        return overrides


class TrtllmProfiling(ProfiledEngine):
    name = "trtllm"

    attention_dp_arg = "enable_attention_dp"

    def role_arguments(self, profile: ProfileRequest) -> list[str]:
        args = profile.role_args
        size = (args.get("kv_cache_config") or {}).get("host_cache_size")
        if not size or profile.headroom_gib <= 0:
            return []
        ranks = int(args.get("tensor_parallel_size", 1)) * int(
            args.get("pipeline_parallel_size", 1)
        )
        shrunk = int(size) - int(profile.headroom_gib * 2**30 / ranks)
        if shrunk <= 0:
            raise ValueError("INFX_PROFILE host_headroom_gib exceeds the recipe's KV host cache")
        return _set("roles.agg.args.kv_cache_config.host_cache_size", shrunk)


PROFILED_ENGINES = {
    engine.name: engine for engine in (VllmProfiling(), SglangProfiling(), TrtllmProfiling())
}


def profiling_arguments(
    environment: Mapping[str, str],
    role_args: Mapping[str, Any] | None = None,
    recipe: Mapping[str, Any] | None = None,
) -> list[str]:
    """Op-attribution profiling: capture hooks, step log and torch windows.

    INFX_PROFILE is a JSON object overriding PROFILE_DEFAULTS; empty disables.
    Everything lands in PROFILE_DIR, which the launcher uploads as its own artifact.
    """
    raw = environment.get("INFX_PROFILE", "")
    if not raw:
        return []
    engine = PROFILED_ENGINES.get(SINGLE_NODE_ENGINES.get(environment["FRAMEWORK"], ""))
    if engine is None:
        raise ValueError(f"INFX_PROFILE supports {sorted(PROFILED_ENGINES)} recipes")
    settings = {**PROFILE_DEFAULTS, **json.loads(raw)}
    windows = settings["windows"]
    if not windows or any(
        len(window) != 3
        or window[0] not in PROFILE_PHASES
        or int(window[1]) < 0
        or int(window[2]) <= 0
        for window in windows
    ):
        raise ValueError(
            f"INFX_PROFILE windows must be [phase in {PROFILE_PHASES}, delay_seconds, iterations]"
        )
    if [PROFILE_PHASES.index(w[0]) for w in windows] != sorted(
        PROFILE_PHASES.index(w[0]) for w in windows
    ):
        raise ValueError("INFX_PROFILE windows must be ordered by phase")
    profile = ProfileRequest(settings, role_args or {}, recipe or {})
    overrides = engine.role_arguments(profile)
    for flag, value in (settings.get("engine_args") or {}).items():
        overrides += _set(f"roles.agg.args.{flag}", value)
    for name, value in engine.worker_env(profile).items():
        overrides += _set(f"roles.agg.env.{name}", value)
    measured = [int(w[1]) for w in windows if w[0] == "profiling"]
    duration = int(
        settings.get("duration") or max(measured, default=0) + PROFILE_MEASURED_CAP_SECONDS
    )
    for name, value in (
        ("INFX_PROFILE_DURATION", str(duration)),
        ("INFX_PROFILE_WINDOWS", json.dumps(windows)),
        ("INFX_PROF_DIR", PROFILE_DIR),
        ("INFX_PROF_CONTROL", engine.control),
    ):
        overrides += _set(f"benchmark.env.{name}", value)
    if settings["mode"] not in PROFILE_MODES:
        raise ValueError(f"INFX_PROFILE mode must be one of {PROFILE_MODES}")
    if settings["mode"] == "synthetic":
        for name in ("isl", "osl"):
            if not isinstance(settings[name], int) or settings[name] <= 0:
                raise ValueError(f"INFX_PROFILE {name} must be a positive integer")
        for name, value in (
            ("INFX_PROFILE_MODE", "synthetic"),
            ("INFX_SYNTH_ISL", settings["isl"]),
            ("INFX_SYNTH_OSL", settings["osl"]),
        ):
            overrides += _set(f"benchmark.env.{name}", str(value))
    for name in ("AIPERF_FAILED_REQUEST_THRESHOLD", "AIPERF_LIVE_FAILED_REQUEST_THRESHOLD"):
        overrides += _set(f"benchmark.env.{name}", "1.0")
    overrides += _set("benchmark.env.AIPERF_REQUIRED_SERVER_METRIC_PREFIX", "")
    return overrides


def runtime_arguments(config: str, environment: Mapping[str, str]) -> list[str]:
    """Bind only runtime-owned values after validating the selected recipe."""
    _, recipe = select_recipe(config, environment)
    for name in ("RUN_EVAL", "EVAL_ONLY", "DP_ATTENTION"):
        if environment[name] not in {"true", "false"}:
            raise ValueError(f"{name} must be true or false")
    overrides = ["--set", f"srun_options.gpus-per-node={json.dumps(environment['GPU_COUNT'])}"]
    overrides += profiling_arguments(environment, recipe["roles"]["agg"]["args"], recipe)
    if environment.get("SRT_SRUN_OPTIONS"):
        options = json.loads(environment["SRT_SRUN_OPTIONS"])
        if not isinstance(options, dict) or any(
            not re.fullmatch(r"[a-z][a-z0-9-]*", key) or not isinstance(value, str)
            for key, value in options.items()
        ):
            raise ValueError("SRT_SRUN_OPTIONS must map option names to string values")
        for key, value in options.items():
            overrides += ["--set", f"srun_options.{key}={json.dumps(value)}"]
    agentic = environment["IS_AGENTIC"] == "1"
    names = ["RESULT_FILENAME", "RUN_EVAL", "EVAL_ONLY", "FRAMEWORK"]
    if agentic:
        names += ["MODEL_PREFIX", "PRECISION", "DURATION", "TP", "PP_SIZE", "PCP_SIZE"]
    for name in names:
        value = environment[name]
        if not value:
            raise ValueError(f"Missing runtime input: {name}")
        overrides += ["--set", f"benchmark.env.{name}={json.dumps(value)}"]
    if agentic:
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
    add_fabric_argument(prepare)
    submitted = commands.add_parser("submission")
    submitted.add_argument("manifest", type=Path)
    parsed = parser.parse_args()
    try:
        if parsed.command == "prepare":
            _, recipe = select_recipe(parsed.recipe, os.environ)
            recipe = resolve_fabric(recipe, parsed.fabric)
            check_setup_script(recipe, Path(parsed.recipe.partition(":")[0]), repository_root())
            arguments = runtime_arguments(parsed.recipe, os.environ)
            config = parsed.output.with_name(BOUND_RECIPE)
            config.write_text(yaml.safe_dump(recipe, sort_keys=False))
            parsed.output.write_bytes("\0".join([str(config), *arguments, ""]).encode())
        else:
            print("\n".join(submission_fields(parsed.manifest)))
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
