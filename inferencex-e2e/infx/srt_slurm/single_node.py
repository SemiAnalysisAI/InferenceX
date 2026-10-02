"""Bind a native single-node SRT recipe to one fixed-sequence or AgentX matrix point."""

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
        "AgentX client": (benchmark.get("command", "").endswith("srt_agentic.sh"), agentic),
    }
    if not agentic:
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
    # vLLM and ATOM shard decode KV across their tensor-parallel ranks.
    dcp = str(args.get("decode-context-parallel-size", 1)) if engine in {"vllm", "atom"} else "1"
    for name, value in {"PP_SIZE": "1", "DCP_SIZE": dcp, "PCP_SIZE": "1"}.items():
        expected[name] = (environment[name], value)
    for name, (actual, wanted) in expected.items():
        if actual != wanted:
            raise ValueError(f"Single-node SRT {name}: recipe/matrix {actual!r} != {wanted!r}")


PROFILE_DIR = "/logs/infx_profile"
PROFILE_DEFAULTS: dict[str, Any] = {
    # [anchor, seconds after it, engine iterations] per torch window. Anchors are
    # aiperf phases ("warmup", "profiling") or "decode", the first steady decode
    # (FULL CUDA graph replays) after warmup starts. AgentX warmup is the lanes'
    # long first turns, all sent at its start (prefill-heavy; at low concurrency
    # they have drained within a minute); each measured-phase turn re-prefills
    # first, so steady decode's arrival depends on concurrency.
    "windows": [["warmup", 0, 32], ["decode", 0, 32]],
    # workers whose CUDA graph capture is profiled; "all" profiles every rank
    "capture_ranks": "dp0_tp0",
    # Host memory kept free of CPU KV offload for the profiler's trace buffers;
    # offload recipes otherwise size the pool to nearly the whole host.
    "host_headroom_gib": 128,
    # "agentic" replays AgentX; "synthetic" drives fixed shapes instead: CONC
    # random-token prompts of `isl` tokens, one-token outputs for the prefill
    # window, then `osl`-token generations for the decode window. No trace
    # dataset or AgentX warmup, so a point takes minutes after engine start-up;
    # `isl` stands in for the KV context decode steps attend over.
    "mode": "agentic",
    "isl": 8192,
    "osl": 4096,
}
PROFILE_MODES = ("agentic", "synthetic")
PROFILE_PHASES = ("warmup", "decode", "profiling")
# Cap on the measured replay past its last phase-anchored window's start. Steady
# decode can take minutes to arrive under data parallelism; the window client
# ends the replay once its windows close. A profiled run's throughput is not a result.
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
    ranks = int(role_args.get("data-parallel-size", 1)) * int(role_args.get("tensor-parallel-size", 1))
    overrides = []
    for index, service in enumerate(recipe.get("services") or []):
        if not isinstance(service, Mapping) or service.get("type") != "mooncake-master":
            continue
        size = ((service.get("options") or {}).get("store_config") or {}).get("global_segment_size")
        match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([KMGT]?B)", str(size or ""), re.IGNORECASE)
        if not match:
            continue
        scale = {"KB": 1 / 2**20, "MB": 1 / 2**10, "GB": 1, "TB": 2**10}[match.group(2).upper()]
        shrunk = float(match.group(1)) * scale - headroom_gib / ranks
        if shrunk <= 0:
            raise ValueError("INFX_PROFILE host_headroom_gib exceeds the recipe's Mooncake segment")
        overrides += ["--set", f"services[{index}].options.store_config.global_segment_size="
                               f"{json.dumps(f'{int(shrunk)}GB')}"]
    return overrides


def profiling_arguments(
    environment: Mapping[str, str],
    role_args: Mapping[str, Any] | None = None,
    recipe: Mapping[str, Any] | None = None,
) -> list[str]:
    """Op-attribution profiling for vLLM: capture hooks, step log and torch windows.

    INFX_PROFILE is a JSON object overriding PROFILE_DEFAULTS; empty disables.
    Everything lands in PROFILE_DIR, which the launcher uploads as its own artifact.
    """
    raw = environment.get("INFX_PROFILE", "")
    if not raw:
        return []
    if environment["FRAMEWORK"] != "vllm":
        raise ValueError("INFX_PROFILE supports only vLLM recipes")
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
    # vLLM's profiler window length is fixed per engine; every window uses the first's.
    profiler_config = {
        "profiler": "torch",
        "torch_profiler_dir": f"{PROFILE_DIR}/torch",
        "torch_profiler_record_shapes": True,
        # Python stacks make exports of eager steps outlast the RPC timeout;
        # modules mark themselves instead (benchmarks/profiling/vllm).
        "torch_profiler_with_stack": False,
        # The summary table walks every event in Python on all ranks at once;
        # on offload-sized hosts that exhausted memory. The raw trace suffices.
        "torch_profiler_dump_cuda_time_total": False,
        "ignore_frontend": True,
        "max_iterations": int(windows[0][2]),
    }
    worker_env = {
        "INFX_PROF_DIR": PROFILE_DIR,
        "INFX_PROF_CAPTURE_RANKS": settings["capture_ranks"],
        "PYTHONPATH": "/infmax-workspace/benchmarks/profiling/vllm",
        # A window's trace export blocks its worker; keep peers from timing out.
        "VLLM_RPC_TIMEOUT": "1800000",
    }
    overrides = ["--set", f"roles.agg.args.profiler-config={json.dumps(profiler_config)}"]
    overrides += offload_headroom_arguments(role_args or {}, float(settings["host_headroom_gib"]))
    overrides += mooncake_headroom_arguments(recipe or {}, role_args or {},
                                             float(settings["host_headroom_gib"]))
    for name, value in worker_env.items():
        overrides += ["--set", f"roles.agg.env.{name}={json.dumps(value)}"]
    measured = [int(w[1]) for w in windows if w[0] == "profiling"]
    duration = int(
        settings.get("duration") or max(measured, default=0) + PROFILE_MEASURED_CAP_SECONDS
    )
    overrides += [
        "--set",
        f"benchmark.env.INFX_PROFILE_DURATION={json.dumps(str(duration))}",
        "--set",
        f"benchmark.env.INFX_PROFILE_WINDOWS={json.dumps(json.dumps(windows))}",
        "--set",
        f"benchmark.env.INFX_PROF_DIR={json.dumps(PROFILE_DIR)}",
    ]
    if settings["mode"] not in PROFILE_MODES:
        raise ValueError(f"INFX_PROFILE mode must be one of {PROFILE_MODES}")
    if settings["mode"] == "synthetic":
        for name in ("isl", "osl"):
            if not isinstance(settings[name], int) or settings[name] <= 0:
                raise ValueError(f"INFX_PROFILE {name} must be a positive integer")
        for name, value in (("INFX_PROFILE_MODE", "synthetic"), ("INFX_SYNTH_ISL", settings["isl"]),
                            ("INFX_SYNTH_OSL", settings["osl"])):
            overrides += ["--set", f"benchmark.env.{name}={json.dumps(str(value))}"]
    # A window's export pauses the engine; its requests must not abort the replay.
    for name in ("AIPERF_FAILED_REQUEST_THRESHOLD", "AIPERF_LIVE_FAILED_REQUEST_THRESHOLD"):
        overrides += ["--set", f'benchmark.env.{name}="1.0"']
    # The window client ends the replay with a user cancel, after which aiperf
    # exports results but not server metrics; a profiled run charts none.
    overrides += ["--set", 'benchmark.env.AIPERF_REQUIRED_SERVER_METRIC_PREFIX=""']
    return overrides


def runtime_arguments(config: str, environment: Mapping[str, str]) -> list[str]:
    """Bind only runtime-owned values after validating the selected recipe."""
    _, recipe = select_recipe(config, environment)
    for name in ("RUN_EVAL", "EVAL_ONLY", "DP_ATTENTION"):
        if environment[name] not in {"true", "false"}:
            raise ValueError(f"{name} must be true or false")
    # Exclusive nodes include idle GPUs. Restrict each server/client step to
    # the serving GPU count so client-side power collection sees the same set.
    overrides = ["--set", f"srun_options.gpus-per-node={json.dumps(environment['GPU_COUNT'])}"]
    overrides += profiling_arguments(environment, recipe["roles"]["agg"]["args"], recipe)
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
