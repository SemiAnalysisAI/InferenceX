"""``job_event.json``: one record per benchmark job, written once when its launch ends."""

from __future__ import annotations

import contextlib
import os
import signal
import sys
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field

from infx.launch.request import BATCH_REENTRY_ENV
from infx.results.collect_events import FILENAME

SCHEMA_VERSION = 1


class Outcome(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Power(_Record):
    dcgm: bool
    agentx: bool
    adapter: bool


class JobError(_Record):
    stage: str
    type: str
    message: str
    exit_code: int
    retriable: bool
    evidence: str | None = None


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
    image: str | None = None
    recipe: str | None = None
    recipe_fingerprint: str | None = None
    runner: str | None = None
    cluster: str | None = None
    slurm_job_id: str | None = None
    batch_job_id: str | None = None
    nodes: str | None = None
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
    artifacts: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class _Failure:
    stage: str
    kind: str
    message: str
    retriable: bool
    evidence: Path | None


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
        """Identify the job from the workflow environment; never raises."""

        def text(name: str) -> str | None:
            return env.get(name) or None

        def number(name: str) -> int | None:
            value = env.get(name, "")
            return int(value) if value.isascii() and value.isdecimal() else None

        event = JobEvent(
            run_id=text("GITHUB_RUN_ID"),
            run_attempt=number("GITHUB_RUN_ATTEMPT"),
            result_filename=text("RESULT_FILENAME"),
            exp_name=text("EXP_NAME"),
            model=text("MODEL"),
            model_prefix=text("MODEL_PREFIX"),
            framework=text("FRAMEWORK"),
            precision=text("PRECISION"),
            spec_decoding=text("SPEC_DECODING"),
            scenario=text("SCENARIO_TYPE"),
            multinode=env.get("IS_MULTINODE") == "true",
            conc=number("CONC"),
            conc_list=[
                int(word)
                for word in env.get("CONC_LIST", "").split()
                if word.isascii() and word.isdecimal()
            ],
            isl=number("ISL"),
            osl=number("OSL"),
            image=text("IMAGE"),
            recipe=text("SRT_RECIPE") or text("CONFIG_FILE"),
            recipe_fingerprint=text("RECIPE_FINGERPRINT"),
            runner=text("RUNNER_NAME"),
            eval_only=env.get("EVAL_ONLY") == "true",
            run_eval=env.get("RUN_EVAL") == "true",
            kv_offloading=text("KV_OFFLOADING"),
            kv_offload_backend=text("KV_OFFLOAD_BACKEND"),
        )
        workspace = Path(env["GITHUB_WORKSPACE"]) if env.get("GITHUB_WORKSPACE") else None
        # A batch-wrapped launch is not the job's exit boundary; the wrapper annotates.
        annotate = env.get("GITHUB_ACTIONS") == "true" and env.get(BATCH_REENTRY_ENV) != "1"
        return cls(event, workspace, annotate=annotate)

    def set(self, **fields: object) -> None:
        for name, value in fields.items():
            setattr(self.event, name, value)

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
        """Print ``ERROR: message`` (unless ``report`` is off) and record the failure."""
        if report:
            print(f"ERROR: {message}", file=sys.stderr, flush=True)
        if self._failure is None:
            stage = stage or self.current_stage
            self._failure = _Failure(stage, kind, message, retriable, self._evidence)

    def error(self, error: BaseException, *, stage: str | None = None, report: bool = True) -> None:
        """``fail`` for an exception; its class may declare ``retriable``."""
        retriable = bool(getattr(error, "retriable", False))
        kind = type(error).__name__.lstrip("_")
        self.fail(kind, str(error), stage=stage, retriable=retriable, report=report)

    def interrupted(self, signum: int) -> None:
        message = f"received {signal.Signals(signum).name}"
        self.fail("Interrupted", message, retriable=True, report=False)

    def exited(self, returncode: int) -> None:
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
        self._lap()
        self.exited(returncode)
        record = self.event.model_copy(deep=True)
        stages = dict(self._stages)
        failure = self._failure
        if (inner := self._reentry) is not None:
            # The wrapped launch knows the benchmark job; this one only the batch around it.
            outer = record
            record = inner.model_copy(deep=True)
            record.launch_path = outer.launch_path
            record.batch_job_id = outer.slurm_job_id
            stages = {f"batch_{name}": seconds for name, seconds in stages.items()} | inner.stages
            if failure is not None and inner.error is not None:
                error = inner.error
                evidence = Path(error.evidence) if error.evidence else None
                failure = _Failure(
                    error.stage, error.type, error.message, error.retriable, evidence
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
            )
        after = _files(self.workspace)
        record.artifacts = sorted(
            name
            for name, seen in after.items()
            if name != FILENAME and self._before.get(name) != seen
        )
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
        now = time.monotonic()
        if self._active:
            name = self._active[-1]
            self._stages[name] = self._stages.get(name, 0.0) + now - self._since
        self._since = now

    def _shown(self, path: Path | None) -> str | None:
        if path is None:
            return None
        if self.workspace is not None and path.is_relative_to(self.workspace):
            return path.relative_to(self.workspace).as_posix()
        return str(path)


def annotation(error: JobError) -> str:
    """The GitHub Actions ``::error`` command for ``error``."""
    title = _escape_data(error.stage).replace(":", "%3A").replace(",", "%2C")
    return f"::error title={title}::{_escape_data(f'{error.type}: {error.message}')}"


def _escape_data(text: str) -> str:
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
