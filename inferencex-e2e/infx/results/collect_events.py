"""Join a sweep's per-job ``job_event.json`` records into one JSON Lines file.

Given the run, each record also gains what only its finished GitHub job knows, and every
benchmark or eval job that left no record gets one synthesized from the run's job listing.
Records are diagnostics: a listing that cannot be read leaves the records as they are.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infx import github

FILENAME = "job_event.json"
SCHEMA_VERSION = 2
NO_RECORD = "NoLauncherRecord"
_BENCHMARK_JOB = re.compile(
    r"(?:^|/ )(?:klaud \| )?p[0-9.]+ \| (?P<model_prefix>\S+) (?P<precision>\S+) (?P<runner>\S+)"
    r" .* c(?P<conc>[0-9]+(?:x[0-9]+)*)(?: \| (?P<eval>eval|eval-only))?$"
)
_FAILED_STEP = frozenset({"failure", "cancelled", "timed_out"})


def read_events(directory: Path) -> list[dict[str, Any]]:
    """Every readable record under ``directory``, in path order."""
    events = []
    for path in sorted(directory.rglob(FILENAME)):
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError) as error:
            print(f"WARNING: skipping {path}: {error}", file=sys.stderr)
            continue
        if not isinstance(record, dict):
            print(f"WARNING: skipping {path}: not a JSON object", file=sys.stderr)
            continue
        events.append(record)
    return events


def reconcile(
    events: list[dict[str, Any]], jobs: Iterable[dict[str, Any]], *, run_url: str
) -> list[dict[str, Any]]:
    """``events`` with their jobs' outcomes, then a record for each benchmark job without one.

    Benchmark jobs are recognized by the names benchmark-tmpl.yml and benchmark-multinode-tmpl.yml give them.
    """
    events = [dict(event) for event in events]
    starts = [_time(event.get("started_at")) for event in events]
    synthesized = []
    for job in jobs:
        name = job.get("name") or ""
        if job.get("conclusion") in {None, "skipped"} or not _BENCHMARK_JOB.search(name):
            continue
        began, ended = _time(job.get("started_at")), _time(job.get("completed_at"))
        event = next(
            (
                event
                for event, started in zip(events, starts, strict=True)
                if _ran(event, started, job, began, ended)
            ),
            None,
        )
        if event is None:
            synthesized.append(_synthesized(job, run_url))
        elif isinstance(event.get("github"), dict):
            event["github"] = {**event["github"], **_outcome(job)}
    return [*events, *synthesized]


def _ran(
    event: dict[str, Any],
    started: datetime | None,
    job: dict[str, Any],
    began: datetime | None,
    ended: datetime | None,
) -> bool:
    """Whether ``event`` is ``job``'s: by check-run id, else by runner and start time."""
    recorded = event.get("github")
    job_id = recorded.get("job_id") if isinstance(recorded, dict) else None
    if job_id is not None:
        return job_id == job.get("id")
    return (
        event.get("runner") is not None
        and event.get("runner") == job.get("runner_name")
        and None not in (started, began, ended)
        and began <= started <= ended
    )


def _outcome(job: dict[str, Any]) -> dict[str, Any]:
    """The finished job's identity and conclusion, and the step that failed it."""
    steps = job.get("steps") or []
    failed = next(
        (step.get("name") for step in steps if step.get("conclusion") in _FAILED_STEP), None
    )
    return {
        "job_id": job.get("id"),
        "job_url": job.get("html_url"),
        "job_name": job.get("name"),
        "conclusion": job.get("conclusion"),
        "failed_step": failed,
    }


def _synthesized(job: dict[str, Any], run_url: str) -> dict[str, Any]:
    """The record of a benchmark job that never ran the launcher to its end."""
    name = job["name"]
    identity = _BENCHMARK_JOB.search(name)
    outcome = _outcome(job)
    concurrencies = [int(value) for value in identity["conc"].split("x")]
    multinode = name.startswith("multi-node")
    runner = identity["runner"]
    began, ended = _time(job.get("started_at")), _time(job.get("completed_at"))
    error = None
    if job["conclusion"] != "success":
        step = outcome["failed_step"]
        where = f"at step {step!r}" if step else "before any step ran"
        error = {
            "stage": step or "runner",
            "type": NO_RECORD,
            "message": f"job concluded {job['conclusion']} {where} and left no {FILENAME}",
            "exit_code": None,
            "retriable": None,
            "evidence": job.get("html_url"),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": str(job["run_id"]) if "run_id" in job else None,
        "run_attempt": job.get("run_attempt"),
        "model_prefix": identity["model_prefix"],
        "precision": identity["precision"],
        "multinode": multinode,
        "conc": None if multinode else concurrencies[0],
        "conc_list": concurrencies if multinode else [],
        "github": {"run_url": run_url, **outcome},
        "runner": job.get("runner_name"),
        "cluster": runner.removeprefix("cluster:") if runner.startswith("cluster:") else None,
        "eval_only": identity["eval"] == "eval-only",
        "run_eval": identity["eval"] is not None,
        "started_at": job.get("started_at"),
        "duration_s": (ended - began).total_seconds() if began and ended else None,
        "outcome": "success" if job["conclusion"] == "success" else "failure",
        "error": error,
    }


def _time(value: object) -> datetime | None:
    """An ISO 8601 time, in UTC unless it names its zone."""
    if not isinstance(value, str):
        return None
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


def main(argv: list[str] | None = None) -> int:
    """Write the job events under the artifacts dir in ``argv`` as JSON Lines, reconciled with ``--run`` jobs, and return 0."""
    parser = argparse.ArgumentParser(
        prog="python -m infx.results.collect_events", description=__doc__
    )
    parser.add_argument(
        "artifacts", type=Path, help="directory of downloaded job_event_* artifacts"
    )
    parser.add_argument("output", type=Path, help="JSON Lines file to write")
    parser.add_argument(
        "--run",
        nargs=3,
        metavar=("SERVER_URL", "REPOSITORY", "RUN_ID"),
        help="the run to list jobs of, with GH_TOKEN, and reconcile the records with",
    )
    args = parser.parse_args(argv)
    events = read_events(args.artifacts)
    collected = len(events)
    if args.run:
        server, repository, run_id = args.run
        try:
            jobs = github.paginate(
                repository,
                f"/actions/runs/{run_id}/jobs",
                os.environ.get("GH_TOKEN"),
                "jobs",
                {"filter": "latest"},
            )
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
            print(f"WARNING: could not list the run's jobs: {error}", file=sys.stderr)
        else:
            run_url = f"{server}/{repository}/actions/runs/{run_id}"
            events = reconcile(events, jobs, run_url=run_url)
    args.output.write_text(
        "".join(json.dumps(event, separators=(",", ":")) + "\n" for event in events)
    )
    print(
        f"Collected {collected} job events and synthesized {len(events) - collected} "
        f"into {args.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
