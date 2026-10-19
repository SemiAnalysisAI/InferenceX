"""A run's job records: why each failed job failed, and whether rerunning it may pass."""

from __future__ import annotations

import tempfile
from pathlib import Path

from infx.launch.event import JobEvent
from infx.results.collect_events import NO_RECORD, SCHEMA_VERSION

from . import github
from .github import VerificationError
from .models import utc

# collect-job-events.yml uploads each attempt's records as this artifact's one JSON Lines file.
ARTIFACT = "job_events"
RECORDS = "events.jsonl"
FAILED = ("failure", "timed_out")
# The collector's stage for a job that failed before any step ran, and GitHub's runner setup.
RUNNER_STAGES = frozenset({"runner", "Set up job"})
# benchmark-tmpl.yml's and benchmark-multinode-tmpl.yml's launcher steps.
LAUNCH_STEPS = frozenset({"Launch job script", "Launch multi-node job script"})


def attempt_jobs(repository: str, run: dict) -> list[dict]:
    return github.items(
        repository,
        f"actions/runs/{run['id']}/attempts/{run['run_attempt']}/jobs?per_page=100",
        "jobs",
    )


def records(repository: str, run: dict) -> dict[int, JobEvent] | None:
    """``run``'s job records by job id; None when it has no readable job_events artifact.

    Rerun jobs get new ids, so an earlier attempt's records never describe a later attempt's jobs.
    """
    uploads = [
        artifact
        for artifact in github.artifacts(repository, run["id"])
        if artifact["name"] == ARTIFACT and not artifact["expired"]
    ]
    if not uploads:
        return None
    newest = max(uploads, key=lambda artifact: (utc(artifact["created_at"]), artifact["id"]))
    with tempfile.TemporaryDirectory(prefix="klaud-job-events-") as temp:
        try:
            github.download_json(repository, newest, Path(temp), suffix=".jsonl")
            lines = (Path(temp) / RECORDS).read_text().splitlines()
        except (OSError, VerificationError):
            return None
    found = {}
    for line in lines:
        try:
            event = JobEvent.model_validate_json(line)
        except ValueError:
            continue
        if event.schema_version == SCHEMA_VERSION and event.github.job_id is not None:
            found[event.github.job_id] = event
    return found


def retriable(job: dict, record: JobEvent | None) -> bool:
    """Whether the failed ``job``'s record says rerunning it may pass.

    The launcher marks the failures that lose a job to the cluster retriable. A job that left no
    record of its own may pass only if it failed on its runner or before reaching its launcher.
    """
    error = record.error if record else None
    if error is None:
        return False
    if error.type != NO_RECORD:
        return error.retriable is True
    return error.stage in RUNNER_STAGES or any(
        step["name"] in LAUNCH_STEPS and step["conclusion"] == "skipped" for step in job["steps"]
    )
