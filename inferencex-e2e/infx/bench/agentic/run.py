"""``agentic``: replay AgentX traces at one concurrency against a ready server, then score them."""

from __future__ import annotations

import argparse
import datetime
import math
import os
import shlex
import sys
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from infx.bench import env as inputs, proc, server
from infx.bench.agentic.replay import (
    REQUIRED as REPLAY_REQUIRED,
    ReplayConfig,
    replay_argv,
)
from infx.bench.agentic.venv import Runtime, bootstrap
from infx.bench.gpu_monitor import GpuMonitor

REQUIRED = (
    "RESULT_DIR",
    "RESULT_FILENAME",
    "EVAL_ONLY",
    "IS_MULTINODE",
    "PRECISION",
    "ENABLE_AGENTX_POWER",
    "REQUIRE_POWER",
    "KV_OFFLOADING",
    "AIPERF_PYTHON_VERSION",
    "AIPERF_FAILED_REQUEST_THRESHOLD",
)
# Spellings the power switches have always accepted as enabled.
TRUE_VALUES = frozenset({"1", "true", "TRUE", "yes", "YES"})
POWER_SAMPLE_INTERVAL_S = 1

# monitor: sample local GPUs; window: mark srt-slurm's multi-node measurement window;
# missing: a multi-node job without that window, recorded as invalid power.
PowerMode = Literal["off", "monitor", "window", "missing"]


@dataclass(frozen=True)
class Plan:
    """One validated AgentX point."""

    replay: ReplayConfig
    result_dir: Path
    output_dir: Path
    result_filename: str
    python_version: str
    chat_budget: tuple[int, int] | None
    """``(timeout, stabilization)`` seconds when an eval-only job waits for the chat route."""
    power: PowerMode
    require_power: bool
    expected_num_gpus: int | None
    failed_request_threshold: str
    required_metric_prefix: str | None

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> Plan:
        """Validate every input before any setup, so a bad point fails in seconds."""
        values = inputs.require(*REQUIRED, *REPLAY_REQUIRED, env=env)
        multinode = inputs.flag("IS_MULTINODE", env)
        result_dir = Path(values["RESULT_DIR"]).absolute()
        result_filename = values["RESULT_FILENAME"]
        if multinode or env.get("CONC_LIST"):
            # Multi-node collection globs one ${RESULT_FILENAME}_conc*.json per point. Some
            # multi-node recipes pin IS_MULTINODE=false, so the workflow's CONC_LIST counts too.
            result_dir /= f"conc_{values['CONC']}"
            result_filename += f"_conc{values['CONC']}"
        replay = ReplayConfig.from_env(env, result_dir)
        _require_single_point(env, values["CONC"])
        _validate_kv_offload(env)
        eval_only = inputs.flag("EVAL_ONLY", env)
        power = _power_mode(values["ENABLE_AGENTX_POWER"], multinode, env)
        return cls(
            replay=replay,
            result_dir=result_dir,
            output_dir=Path(env.get("AGENTIC_OUTPUT_DIR") or proc.REPO_ROOT).absolute(),
            result_filename=result_filename,
            python_version=values["AIPERF_PYTHON_VERSION"],
            chat_budget=server.chat_route_budget(env) if eval_only else None,
            power=power,
            require_power=values["REQUIRE_POWER"] in TRUE_VALUES,
            expected_num_gpus=_expected_num_gpus(env) if power == "monitor" else None,
            failed_request_threshold=values["AIPERF_FAILED_REQUEST_THRESHOLD"],
            required_metric_prefix=inputs.optional("AIPERF_REQUIRED_SERVER_METRIC_PREFIX", env),
        )


def _require_single_point(env: Mapping[str, str], conc: str) -> None:
    """AgentX measures one concurrency per fresh server deployment."""
    points = env.get("CONC_LIST")
    if points is not None and points != conc:
        raise inputs.InputError(
            "AgentX requires exactly one positive concurrency per server deployment; "
            f"CONC_LIST={points!r} must equal CONC={conc!r}. Launch a fresh server for each "
            "concurrency."
        )


def _validate_kv_offload(env: Mapping[str, str]) -> None:
    """The served KV-offload configuration, as the matrix ``kv-offloading`` field allows."""
    mode = env["KV_OFFLOADING"]
    backend = env.get("KV_OFFLOAD_BACKEND")
    if mode == "none":
        if backend:
            raise inputs.InputError("KV_OFFLOAD_BACKEND must be empty when KV_OFFLOADING=none")
    elif mode == "dram":
        if not backend or backend == "none":
            raise inputs.InputError("KV_OFFLOAD_BACKEND is required when KV_OFFLOADING=dram")
        inputs.parse_positive_int("TOTAL_CPU_DRAM_GB", env.get("TOTAL_CPU_DRAM_GB", ""))
    else:
        raise inputs.InputError(
            f"unsupported KV_OFFLOADING value {mode!r} (expected one of: none, dram)"
        )


def _power_mode(enabled: str, multinode: bool, env: Mapping[str, str]) -> PowerMode:
    if enabled not in TRUE_VALUES:
        return "off"
    if not multinode:
        return "monitor"
    # srt-slurm's telemetry measures multi-node points and exports the window directory.
    return "window" if env.get("SRT_MEASUREMENT_WINDOW_DIR") else "missing"


def _expected_num_gpus(env: Mapping[str, str]) -> int:
    values = inputs.require("TP", "PP_SIZE", "PCP_SIZE", env=env)
    return math.prod(inputs.parse_positive_int(name, value) for name, value in values.items())


def execute(plan: Plan, runtime: Runtime, environ: Mapping[str, str]) -> int:
    """Run ``plan`` with ``runtime``'s tools; ``environ`` seeds every child's environment."""
    cfg = plan.replay
    python = str(runtime.python)
    env = {
        **environ,
        "RESULT_DIR": str(plan.result_dir),
        "RESULT_FILENAME": plan.result_filename,
        "AGENTIC_OUTPUT_DIR": str(plan.output_dir),
        "PYTHONPATH": proc.pythonpath(environ),
    }
    rc = _download_traces(cfg, runtime, env)
    if rc:
        return rc
    print(f"Using server endpoint: {cfg.url}", flush=True)
    if plan.chat_budget is not None:
        server.wait_chat_route(cfg.url.rstrip("/"), cfg.model, plan.chat_budget)

    plan.result_dir.mkdir(parents=True, exist_ok=True)
    print(f"Running agentic concurrency {cfg.concurrency} on this server deployment", flush=True)
    rc = _open_power_window(plan, python, env)
    if rc:
        return rc
    argv = replay_argv(cfg, runtime.aiperf)
    (plan.result_dir / "benchmark_command.txt").write_text(f"{shlex.join(argv)}\n")
    monitor = (
        GpuMonitor(plan.result_dir / "gpu_metrics.csv", POWER_SAMPLE_INTERVAL_S)
        if plan.power == "monitor"
        else nullcontext()
    )
    log = plan.result_dir / "benchmark.log"
    with proc.DeferSignals() as signals, monitor:
        replay_rc = 0 if signals.received else proc.tee(argv, log, env)
    if signals.received:
        return 128 + signals.received
    return _score(plan, python, env, replay_rc)


def _download_traces(cfg: ReplayConfig, runtime: Runtime, env: Mapping[str, str]) -> int:
    print(f"Loading traces via aiperf public-dataset: {cfg.loader} ({cfg.dataset})", flush=True)
    # Into the shared HF_HUB_CACHE: later jobs hit it, and the aggregate reads traces there.
    rc = proc.call([str(runtime.hf), "download", "--repo-type", "dataset", cfg.dataset], env)
    if rc:
        print(f"ERROR: downloading {cfg.dataset} failed with code {rc}", file=sys.stderr)
    return rc


def _open_power_window(plan: Plan, python: str, env: Mapping[str, str]) -> int:
    """Record the replay clock's UTC offset; mark srt-slurm's multi-node window running."""
    if plan.power in {"monitor", "window"}:
        # AIPerf exports naive local datetimes and SMI the same wall clock; the power
        # adapter needs the offset to normalize the profiling window.
        now = datetime.datetime.now(datetime.timezone.utc).astimezone()
        (plan.result_dir / "agentic_power_timezone_offset.txt").write_text(f"{now:%z}\n")
    if plan.power != "window":
        return 0
    rc = _power_adapter(plan, python, env, *_window(plan, "running"))
    if rc:
        print("ERROR: failed to publish the AgentX formal running power window", file=sys.stderr)
    return rc


def _score(plan: Plan, python: str, env: Mapping[str, str], replay_rc: int) -> int:
    """Aggregate, plot, audit power, and validate; return the first failure by precedence."""
    result_dir, artifacts = str(plan.result_dir), str(plan.replay.artifact_dir)
    aggregate_rc = _results(python, env, "agentic.process_agentic_result")
    # Best effort: the aggregate JSON is the success gate.
    _results(python, env, "generate_aiperf_plots", result_dir)
    audit = _power_audit(plan, replay_rc)
    power_rc = _power_adapter(plan, python, env, *audit) if audit else 0
    _results(python, env, "agentic.analyze_benchmark_distributions", artifacts, "-o", result_dir)
    threshold = ["--failed-request-threshold", plan.failed_request_threshold]
    validation_rc = _results(python, env, "agentic.validate_agentic_result", artifacts, *threshold)
    for failed, message in (
        (replay_rc, f"agentic trace replay exited with code {replay_rc} after writing results"),
        (aggregate_rc, f"AgentX aggregation exited with code {aggregate_rc}"),
        (validation_rc, "agentic trace replay produced invalid results"),
        (power_rc, "AgentX power validation failed after writing audit artifacts"),
    ):
        if failed:
            print(f"ERROR: {message}", file=sys.stderr)
            return failed
    _check_server_metrics(plan.replay.artifact_dir, plan.required_metric_prefix)
    return 0


def _power_audit(plan: Plan, replay_rc: int) -> list[str] | None:
    """The power adapter's post-replay arguments; a failed replay leaves the window running."""
    aggregate = ["--agg-result", str(plan.output_dir / f"{plan.result_filename}.json")]
    return {
        "monitor": [*aggregate, "--expected-num-gpus", str(plan.expected_num_gpus)],
        "window": _window(plan, "completed") if replay_rc == 0 else None,
        "missing": [*aggregate, "--multinode-contract-missing"],
    }.get(plan.power)


def _window(plan: Plan, state: str) -> list[str]:
    return ["--concurrency", str(plan.replay.concurrency), "--write-multinode-window", state]


def _power_adapter(plan: Plan, python: str, env: Mapping[str, str], *args: str) -> int:
    strict = ["--require-power"] if plan.require_power else []
    adapter = ["agentic.power_adapter", "--result-dir", str(plan.result_dir), *args, *strict]
    return _results(python, env, *adapter)


def _results(python: str, env: Mapping[str, str], module: str, *args: str) -> int:
    return proc.call([python, "-m", f"infx.results.{module}", *args], env)


def _check_server_metrics(artifact_dir: Path, prefix: str | None) -> None:
    """Opt-in: fail rather than publish trace charts without the engine's metrics."""
    if not prefix:
        return
    exported = artifact_dir / "server_metrics_export.json"
    if not (_nonempty(exported) and _nonempty(artifact_dir / "server_metrics_export.csv")):
        raise inputs.BenchError(
            f"required AIPerf server metrics artifacts are missing or empty in {artifact_dir}"
        )
    # The export can be several GiB. Metric names are object keys, so a quoted prefix
    # anywhere proves the engine's metrics were captured.
    if not _contains(exported, f'"{prefix}'.encode()):
        raise inputs.BenchError(f"{exported} has no metric with required prefix {prefix!r}")
    print(f"Validated required AIPerf server metrics prefix {prefix!r}", flush=True)


def _contains(path: Path, needle: bytes) -> bool:
    """Scan ``path`` in 1 MiB chunks, keeping enough overlap for a match across chunks."""
    overlap = b""
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            window = overlap + chunk
            if needle in window:
                return True
            overlap = window[max(0, len(window) - len(needle) + 1) :]
    return False


def _nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def main(argv: list[str]) -> int:
    """Run the ``agentic`` command."""
    argparse.ArgumentParser(
        prog="python3 -m infx.bench agentic",
        description="Replay AgentX traces at one concurrency against a ready server.",
    ).parse_args(argv)
    plan = Plan.from_env(os.environ)
    runtime = Runtime.for_job(os.environ)
    if not runtime.active():
        print(f"Preparing the AIPerf runtime in {runtime.venv}", flush=True)
        rc = bootstrap(runtime, plan.python_version, proc.REPO_ROOT / "utils" / "aiperf")
        if rc:
            return rc
        # The venv interpreter needs the same safe sys.path as this one had.
        environ = {**os.environ, "PYTHONPATH": proc.pythonpath(), "PYTHONSAFEPATH": "1"}
        runtime.exec_python(["-m", "infx.bench", "agentic", *argv], environ)
    return execute(plan, runtime, os.environ)
