"""``job_event.json``: one record per benchmark job, written once when its launch ends."""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import signal
import sys
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

from infx.bench.agentic.venv import REQUIREMENTS
from infx.config import repository_root
from infx.launch.evidence import error_detail
from infx.launch.request import BATCH_REENTRY_ENV
from infx.results.collect_events import FILENAME, SCHEMA_VERSION

AGENTIC = "agentic-coding"
_SHA = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")
_PINNED = re.compile(r"@(sha256:[0-9a-f]{64})$")
_HARNESS = re.compile(r"^(?:agentx-harness|aiperf)==(\S+)", re.MULTILINE)
_RESULT_GPUS = re.compile(r"_gpus_([0-9]+)")


class Outcome(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Power(_Record):
    dcgm: bool
    agentx: bool
    adapter: bool
    require: bool = False


class JobError(_Record):
    """``detail`` is the evidence log's likeliest cause line followed by its last lines.

    A record synthesized for a job that never wrote one knows no exit code or retriability.
    """

    stage: str
    type: str
    message: str
    exit_code: int | None
    retriable: bool | None
    evidence: str | None = None
    detail: str | None = None


class Versions(_Record):
    """``image_digest`` is known only for an image pinned by digest: imports keep no other."""

    inferencex_sha: str | None = None
    workflow_sha: str | None = None
    srt_slurm_sha: str | None = None
    srtctl_version: str | None = None
    agentx_harness: str | None = None
    image_digest: str | None = None


class RoleTopology(_Record):
    num_workers: int | None = None
    tp: int | None = None
    ep: int | None = None
    dp_attn: bool | None = None


class Topology(_Record):
    """Single-node points fill the parallelism and GPU count; multi-node points the rest."""

    tp: int | None = None
    ep: int | None = None
    pp: int | None = None
    dp_attn: bool | None = None
    gpu_count: int | None = None
    node_count: int | None = None
    prefill: RoleTopology | None = None
    decode: RoleTopology | None = None


class Metrics(_Record):
    """One benchmark point's outcome as its result file ``source`` reports it."""

    source: str
    conc: int | None = None
    requests_completed: int | None = None
    requests_failed: int | None = None
    request_throughput: float | None = None
    output_throughput: float | None = None
    total_throughput_per_gpu: float | None = None
    ttft_p50_ms: float | None = None
    ttft_p99_ms: float | None = None
    tpot_p50_ms: float | None = None

    @field_validator(
        "request_throughput",
        "output_throughput",
        "total_throughput_per_gpu",
        "ttft_p50_ms",
        "ttft_p99_ms",
        "tpot_p50_ms",
    )
    @classmethod
    def _rounded(cls, value: float | None) -> float | None:
        """Round a metric to three decimals, keeping None."""
        return None if value is None else round(value, 3)


class GitHub(_Record):
    """The job on GitHub; the collector adds what only the finished job knows."""

    run_url: str | None = None
    job_id: int | None = None
    job_url: str | None = None
    job_name: str | None = None
    event_name: str | None = None
    pr_number: int | None = None
    test_name: str | None = None
    ref: str | None = None
    conclusion: str | None = None
    failed_step: str | None = None


class JobEvent(_Record):
    schema_version: int = SCHEMA_VERSION
    run_id: str | None = None
    run_attempt: int | None = None
    result_filename: str | None = None
    exp_name: str | None = None
    model: str | None = None
    model_prefix: str | None = None
    framework: str | None = None
    precision: str | None = None
    spec_decoding: str | None = None
    scenario: str | None = None
    multinode: bool = False
    conc: int | None = None
    conc_list: list[int] = Field(default_factory=list)
    isl: int | None = None
    osl: int | None = None
    topology: Topology = Field(default_factory=Topology)
    image: str | None = None
    recipe: str | None = None
    recipe_fingerprint: str | None = None
    versions: Versions = Field(default_factory=Versions)
    github: GitHub = Field(default_factory=GitHub)
    runner: str | None = None
    cluster: str | None = None
    slurm_job_id: str | None = None
    batch_job_id: str | None = None
    nodes: str | None = None
    slurm_submitted_at: datetime | None = None
    slurm_started_at: datetime | None = None
    slurm_ended_at: datetime | None = None
    queue_s: float | None = None
    batch_queue_s: float | None = None
    launch_path: str | None = None
    power: Power | None = None
    golden_acceptance_length: float | None = None
    eval_only: bool = False
    run_eval: bool = False
    kv_offloading: str | None = None
    kv_offload_backend: str | None = None
    started_at: datetime | None = None
    duration_s: float | None = None
    stages: dict[str, float] = Field(default_factory=dict)
    outcome: Outcome | None = None
    error: JobError | None = None
    metrics: list[Metrics] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class _Failure:
    stage: str
    kind: str
    message: str
    retriable: bool | None
    evidence: Path | None
    detail: str | None


class JobEventBuilder:
    """One launch's JobEvent, enriched as the launch learns facts and emitted once at its end.

    The first recorded failure wins: later ones are usually its consequences.
    """

    def __init__(
        self,
        event: JobEvent | None = None,
        workspace: Path | None = None,
        *,
        annotate: bool = False,
    ) -> None:
        """Start timing a launch that records into ``event`` and tracks ``workspace`` files."""
        self.event = event if event is not None else JobEvent()
        self.workspace = workspace
        self._annotate = annotate
        self._started_at = datetime.now(UTC)
        self._started = self._since = time.monotonic()
        self._stages: dict[str, float] = {}
        self._active: list[str] = []
        self._last: str | None = None
        self._evidence: Path | None = None
        self._failure: _Failure | None = None
        self._reentry: JobEvent | None = None
        self._before = _files(workspace)

    @classmethod
    def begin(cls, env: Mapping[str, str]) -> Self:
        """Identify the job from the workflow environment; never raises.

        AgentX replays traces, so agentic runs leave ISL and OSL unset. A batch-wrapped launch
        never annotates on GitHub, because its wrapper is the job's exit boundary.
        """
        scenario = _text(env, "SCENARIO_TYPE")
        multinode = env.get("IS_MULTINODE") == "true"
        eval_only = env.get("EVAL_ONLY") == "true"
        event = JobEvent(
            run_id=_text(env, "GITHUB_RUN_ID"),
            run_attempt=_number(env, "GITHUB_RUN_ATTEMPT"),
            result_filename=_text(env, "RESULT_FILENAME"),
            exp_name=_text(env, "EXP_NAME"),
            model=_text(env, "MODEL"),
            model_prefix=_text(env, "MODEL_PREFIX"),
            framework=_text(env, "FRAMEWORK"),
            precision=_text(env, "PRECISION"),
            spec_decoding=_text(env, "SPEC_DECODING"),
            scenario=scenario,
            multinode=multinode,
            conc=_number(env, "CONC"),
            conc_list=[
                int(word)
                for word in env.get("CONC_LIST", "").split()
                if word.isascii() and word.isdecimal()
            ],
            isl=None if scenario == AGENTIC else _number(env, "ISL"),
            osl=None if scenario == AGENTIC else _number(env, "OSL"),
            topology=_topology(env, multinode=multinode),
            image=_text(env, "IMAGE"),
            recipe=_text(env, "SRT_RECIPE") or _text(env, "CONFIG_FILE"),
            recipe_fingerprint=_text(env, "RECIPE_FINGERPRINT"),
            versions=_versions(env, agentx=scenario == AGENTIC and not eval_only),
            github=_github(env),
            runner=_text(env, "RUNNER_NAME"),
            eval_only=eval_only,
            run_eval=env.get("RUN_EVAL") == "true",
            kv_offloading=_text(env, "KV_OFFLOADING"),
            kv_offload_backend=_text(env, "KV_OFFLOAD_BACKEND"),
        )
        workspace = Path(env["GITHUB_WORKSPACE"]) if env.get("GITHUB_WORKSPACE") else None
        annotate = env.get("GITHUB_ACTIONS") == "true" and env.get(BATCH_REENTRY_ENV) != "1"
        return cls(event, workspace, annotate=annotate)

    def set(self, **fields: object) -> None:
        """Set the given fields on the event."""
        for name, value in fields.items():
            setattr(self.event, name, value)

    def set_versions(self, **versions: str | None) -> None:
        """Set the given component versions on the event."""
        for name, value in versions.items():
            setattr(self.event.versions, name, value)

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time the block as ``name``. A nested stage's time counts only toward itself."""
        self._lap()
        self._active.append(name)
        self._last = name
        try:
            yield
        except Exception as error:
            self.error(error, report=False)
            raise
        finally:
            self._lap()
            self._active.pop()

    @property
    def current_stage(self) -> str:
        """Return the innermost running stage, else the last one, else ``prepare``."""
        return self._active[-1] if self._active else self._last or "prepare"

    def evidence(self, log: Path) -> None:
        """Point failures recorded from now on at ``log``."""
        self._evidence = log

    def fail(
        self,
        kind: str,
        message: str,
        *,
        stage: str | None = None,
        retriable: bool = False,
        report: bool = True,
    ) -> None:
        """Print ``ERROR: message`` (unless ``report`` is off) and record the failure.

        The first failure also records what its evidence log says.
        """
        if report:
            print(f"ERROR: {message}", file=sys.stderr, flush=True)
        if self._failure is None:
            stage = stage or self.current_stage
            evidence = self._evidence
            detail = error_detail(evidence) if evidence is not None else None
            self._failure = _Failure(stage, kind, message, retriable, evidence, detail)

    def error(self, error: BaseException, *, stage: str | None = None, report: bool = True) -> None:
        """``fail`` for an exception; its class may declare ``retriable``."""
        retriable = bool(getattr(error, "retriable", False))
        kind = type(error).__name__.lstrip("_")
        self.fail(kind, str(error), stage=stage, retriable=retriable, report=report)

    def interrupted(self, signum: int) -> None:
        """Record a retriable failure for the received signal ``signum``."""
        message = f"received {signal.Signals(signum).name}"
        self.fail("Interrupted", message, retriable=True, report=False)

    def exited(self, returncode: int) -> None:
        """Record a failure if ``returncode`` is non-zero."""
        if returncode:
            self.fail("ExitStatus", f"exited {returncode}", report=False)

    def adopt_reentry(self) -> None:
        """Continue the record the batch-wrapped launch of this job left in the workspace."""
        if self.workspace is None:
            return
        try:
            inner = JobEvent.model_validate_json((self.workspace / FILENAME).read_bytes())
        except (OSError, ValueError):
            return
        identity = ("run_id", "run_attempt", "result_filename")
        if all(getattr(inner, name) == getattr(self.event, name) for name in identity):
            self._reentry = inner

    def finish(self, returncode: int) -> JobEvent:
        """Return the completed record for a launch that exited ``returncode``.

        After a batch re-entry the wrapped launch's record is the base, since it knows the
        benchmark job, and this launch adds its batch job, queue time and ``batch_`` stages.
        """
        self._lap()
        self.exited(returncode)
        record = self.event.model_copy(deep=True)
        stages = dict(self._stages)
        failure = self._failure
        if (inner := self._reentry) is not None:
            outer = record
            record = inner.model_copy(deep=True)
            record.launch_path = outer.launch_path
            record.batch_job_id = outer.slurm_job_id
            record.batch_queue_s = outer.queue_s
            stages = {f"batch_{name}": seconds for name, seconds in stages.items()} | inner.stages
            if failure is not None and inner.error is not None:
                error = inner.error
                evidence = Path(error.evidence) if error.evidence else None
                failure = _Failure(
                    error.stage, error.type, error.message, error.retriable, evidence, error.detail
                )
            elif failure is not None:
                failure = replace(failure, stage=f"batch_{failure.stage}")
        record.started_at = self._started_at
        record.duration_s = round(time.monotonic() - self._started, 3)
        record.stages = {name: round(seconds, 3) for name, seconds in stages.items()}
        record.outcome = Outcome.FAILURE if returncode else Outcome.SUCCESS
        record.error = None
        if returncode and failure is not None:
            record.error = JobError(
                stage=failure.stage,
                type=failure.kind,
                message=failure.message,
                exit_code=returncode,
                retriable=failure.retriable,
                evidence=self._shown(failure.evidence),
                detail=failure.detail,
            )
        after = _files(self.workspace)
        record.artifacts = sorted(
            name
            for name, seen in after.items()
            if name != FILENAME and self._before.get(name) != seen
        )
        record.metrics = self._metrics(record)
        return record

    def emit(self, returncode: int) -> None:
        """Write the record into the workspace, and on GitHub Actions annotate a failure.

        Never raises: the record must not change the launch's exit.
        """
        try:
            record = self.finish(returncode)
        except Exception as error:  # noqa: BLE001
            print(f"WARNING: no {FILENAME}: {error}", file=sys.stderr)
            return
        if self._annotate and record.error is not None:
            print(annotation(record.error), file=sys.stderr, flush=True)
        if self.workspace is None:
            return
        path = self.workspace / FILENAME
        temporary = path.with_name(f".{FILENAME}.tmp")
        try:
            temporary.write_text(record.model_dump_json(indent=2) + "\n")
            temporary.replace(path)
        except OSError as error:
            print(f"WARNING: could not write {path}: {error}", file=sys.stderr)

    def _lap(self) -> None:
        """Charge the time since the last lap to the innermost active stage."""
        now = time.monotonic()
        if self._active:
            name = self._active[-1]
            self._stages[name] = self._stages.get(name, 0.0) + now - self._since
        self._since = now

    def _shown(self, path: Path | None) -> str | None:
        """Return ``path`` relative to the workspace when inside it, else as given, or None."""
        if path is None:
            return None
        if self.workspace is not None and path.is_relative_to(self.workspace):
            return path.relative_to(self.workspace).as_posix()
        return str(path)

    def _metrics(self, record: JobEvent) -> list[Metrics]:
        """The points of the benchmark results this launch left in the workspace.

        Fixed-sequence and AgentX results are ``<stem>.json`` or, one per point, ``<stem>_*.json``.
        """
        stem = record.result_filename
        if self.workspace is None or not stem:
            return []
        names = [
            name
            for name in record.artifacts
            if name == f"{stem}.json" or (name.startswith(f"{stem}_") and name.endswith(".json"))
        ]
        points = (
            _point(self.workspace / name, record.topology.gpu_count)
            for name in names
            if not name.endswith(".pytorch.json")
        )
        return [point for point in points if point is not None]


def annotation(error: JobError) -> str:
    """The GitHub Actions ``::error`` command for ``error``, with its cause line if known."""
    title = _escape_data(error.stage).replace(":", "%3A").replace(",", "%2C")
    message = f"{error.type}: {error.message}"
    if error.detail:
        message += "\n" + error.detail.partition("\n")[0]
    return f"::error title={title}::{_escape_data(message)}"


def _escape_data(text: str) -> str:
    """Escape ``%``, CR and LF in ``text`` for a GitHub Actions workflow command."""
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _files(directory: Path | None) -> dict[str, tuple[int, int]]:
    """``directory``'s top-level files by name, with their modification time and size."""
    if directory is None:
        return {}
    files: dict[str, tuple[int, int]] = {}
    with contextlib.suppress(OSError), os.scandir(directory) as entries:
        for entry in entries:
            with contextlib.suppress(OSError):
                if entry.is_file(follow_symlinks=False):
                    stat = entry.stat(follow_symlinks=False)
                    files[entry.name] = (stat.st_mtime_ns, stat.st_size)
    return files


def _text(env: Mapping[str, str], name: str) -> str | None:
    """Return the variable ``name`` from ``env``, or None if unset or empty."""
    return env.get(name) or None


def _number(env: Mapping[str, str], name: str) -> int | None:
    """Return the variable ``name`` as a non-negative integer, or None if it is not one."""
    value = env.get(name, "")
    return int(value) if value.isascii() and value.isdecimal() else None


def _flag(env: Mapping[str, str], name: str) -> bool | None:
    """Return the variable ``name`` as a bool when it is ``true`` or ``false``, else None."""
    return {"true": True, "false": False}.get(env.get(name, ""))


def _topology(env: Mapping[str, str], *, multinode: bool) -> Topology:
    """Return single-node parallelism, or per-role topology when ``multinode``."""
    if not multinode:
        return Topology(
            tp=_number(env, "TP"),
            ep=_number(env, "EP_SIZE"),
            pp=_number(env, "PP_SIZE"),
            dp_attn=_flag(env, "DP_ATTENTION"),
            gpu_count=_number(env, "GPU_COUNT"),
        )
    roles = {
        role.lower(): RoleTopology(
            num_workers=_number(env, f"{role}_NUM_WORKERS"),
            tp=_number(env, f"{role}_TP"),
            ep=_number(env, f"{role}_EP"),
            dp_attn=_flag(env, f"{role}_DP_ATTN"),
        )
        for role in ("PREFILL", "DECODE")
    }
    return Topology(node_count=_number(env, "NODE_COUNT"), **roles)


def _versions(env: Mapping[str, str], *, agentx: bool) -> Versions:
    """Return checkout, workflow and image versions, plus the AgentX harness if ``agentx``."""
    head = None
    with contextlib.suppress(OSError):
        head = _git_head(repository_root())
    workflow = _text(env, "GITHUB_SHA")
    pinned = _PINNED.search(env.get("IMAGE", ""))
    harness = None
    if agentx:
        with contextlib.suppress(OSError):
            match = _HARNESS.search(REQUIREMENTS.read_text())
            harness = match.group(1) if match else None
    return Versions(
        inferencex_sha=head,
        workflow_sha=workflow if workflow != head else None,
        agentx_harness=harness,
        image_digest=pinned.group(1) if pinned else None,
    )


def _github(env: Mapping[str, str]) -> GitHub:
    """Return the run and job URLs and the event, PR, test and ref details from ``env``."""
    server, repository, run = (
        _text(env, name) for name in ("GITHUB_SERVER_URL", "GITHUB_REPOSITORY", "GITHUB_RUN_ID")
    )
    run_url = f"{server}/{repository}/actions/runs/{run}" if server and repository and run else None
    job_id = _number(env, "JOB_CHECK_RUN_ID")
    return GitHub(
        run_url=run_url,
        job_id=job_id,
        job_url=f"{run_url}/job/{job_id}" if run_url and job_id else None,
        event_name=_text(env, "GITHUB_EVENT_NAME"),
        pr_number=_number(env, "PR_NUMBER"),
        test_name=_text(env, "TEST_NAME"),
        ref=_text(env, "BENCHMARK_REF"),
    )


def _git_head(path: Path) -> str | None:
    """The commit HEAD names in the checkout holding ``path``, read without running git."""
    with contextlib.suppress(OSError):
        for directory in (path, *path.parents):
            dot_git = directory / ".git"
            if dot_git.is_file():
                gitdir = directory / dot_git.read_text().removeprefix("gitdir:").strip()
            elif dot_git.is_dir():
                gitdir = dot_git
            else:
                continue
            head = (gitdir / "HEAD").read_text().strip()
            if _SHA.fullmatch(head):
                return head
            ref = head.removeprefix("ref: ")
            common = gitdir / "commondir"
            root = gitdir / common.read_text().strip() if common.is_file() else gitdir
            for refs in (gitdir, root):
                if (refs / ref).is_file():
                    commit = (refs / ref).read_text().strip()
                    return commit if _SHA.fullmatch(commit) else None
            packed = root / "packed-refs"
            lines = packed.read_text().splitlines() if packed.is_file() else []
            entries = (line.partition(" ") for line in lines)
            return next(
                (sha for sha, _, name in entries if name == ref and _SHA.fullmatch(sha)), None
            )
    return None


def _point(path: Path, gpu_count: int | None) -> Metrics | None:
    """A fixed-sequence or AgentX result's point; None for another file."""
    try:
        result = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(result, dict):
        return None
    if isinstance(result.get("request_metrics"), dict):
        return _agentx_point(path.name, result)
    if "max_concurrency" in result:
        match = _RESULT_GPUS.search(path.name)
        return _fixed_point(path.name, result, int(match.group(1)) if match else gpu_count)
    return None


def _fixed_point(name: str, result: dict[str, Any], gpus: int | None) -> Metrics:
    """The benchmark client's raw result: latencies in ms, per-GPU throughput left to compute."""
    completed = _count(result, "completed")
    failed = _count(_mapping(result, "benchmark_outcome"), "failed")
    requested = _count(result, "num_prompts")
    if failed is None and completed is not None and requested is not None:
        failed = requested - completed if requested >= completed else None
    total = _value(result, "total_token_throughput")
    return Metrics(
        source=name,
        conc=_count(result, "max_concurrency"),
        requests_completed=completed,
        requests_failed=failed,
        request_throughput=_value(result, "request_throughput"),
        output_throughput=_value(result, "output_throughput"),
        total_throughput_per_gpu=total / gpus if total is not None and gpus else None,
        ttft_p50_ms=_value(result, "median_ttft_ms"),
        ttft_p99_ms=_value(result, "p99_ttft_ms"),
        tpot_p50_ms=_value(result, "median_tpot_ms"),
    )


def _agentx_point(name: str, result: dict[str, Any]) -> Metrics:
    """The AgentX aggregate: latencies in seconds, throughput already per GPU."""
    metrics = _mapping(result, "request_metrics")
    throughput, latency = _mapping(metrics, "throughput"), _mapping(metrics, "latency")
    ttft, tpot = _mapping(latency, "ttft"), _mapping(latency, "tpot")
    completed = _count(result, "num_requests_successful")
    duration = _value(throughput, "duration_seconds")
    return Metrics(
        source=name,
        conc=_count(result, "conc"),
        requests_completed=completed,
        requests_failed=_count(_mapping(result, "request_accounting"), "records_error_dropped"),
        request_throughput=completed / duration if completed is not None and duration else None,
        output_throughput=_value(_mapping(throughput, "output"), "tokens_per_second"),
        total_throughput_per_gpu=_value(_mapping(throughput, "per_gpu"), "total_tput_tps"),
        ttft_p50_ms=_milliseconds(_value(ttft, "p50")),
        ttft_p99_ms=_milliseconds(_value(ttft, "p99")),
        tpot_p50_ms=_milliseconds(_value(tpot, "p50")),
    )


def _mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    """Return ``parent[key]`` if it is a dict, else an empty dict."""
    value = parent.get(key)
    return value if isinstance(value, dict) else {}


def _count(parent: dict[str, Any], key: str) -> int | None:
    """Return ``parent[key]`` if it is a non-bool int, else None."""
    value = parent.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _value(parent: dict[str, Any], key: str) -> float | None:
    """Return ``parent[key]`` as a float if it is a finite non-bool number, else None."""
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return None
    return float(value)


def _milliseconds(seconds: float | None) -> float | None:
    """Convert ``seconds`` to milliseconds, keeping None."""
    return None if seconds is None else seconds * 1000
