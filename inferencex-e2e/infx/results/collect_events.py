"""Join a sweep's per-job ``job_event.json`` records into one JSON Lines file.

Given the run, each record also gains what only its finished GitHub job knows, and fails with
its job if its launcher succeeded or never finished; every benchmark or eval job that left no
record gets one synthesized from the run's job listing.
Given a summary file, the job counts and a table of the failed jobs are appended to it.
Records are diagnostics: a listing that cannot be read leaves the records as they are, and a
summary that cannot be written is skipped.
"""

from __future__ import annotations

import argparse
import html
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
STEP_FAILED = "StepFailed"
INTERRUPTED = "Interrupted"
# The names benchmark-tmpl.yml and benchmark-multinode-tmpl.yml give their jobs: an optional
# klaud marker, the priority, model prefix, precision and runner, then the concurrency last.
_BENCHMARK_JOB = re.compile(
    r"(?:^|/ )(?:klaud \| )?p[0-9.]+ \| (?P<point>(?P<model_prefix>\S+) (?P<precision>\S+)"
    r" (?P<runner>\S+) .* c(?P<conc>[0-9]+(?:x[0-9]+)*)(?: \| (?P<eval>eval|eval-only))?)$"
)
_FAILED = frozenset({"failure", "cancelled", "timed_out"})
_SUMMARY_COLUMNS = ("Cluster", "Job", "Stage", "Error", "Retriable", "Cause", "Link")
_CAUSE_CHARS = 200


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

    A record whose launcher succeeded, or never finished and so has no outcome, fails with its
    job at the job's failed step.
    """
    events = [dict(event) for event in events]
    synthesized = []
    for job in jobs:
        name = job.get("name") or ""
        if job.get("conclusion") in {None, "skipped"} or not _BENCHMARK_JOB.search(name):
            continue
        event = next((event for event in events if _ran(event, job)), None)
        if event is None:
            synthesized.append(_synthesized(job, run_url))
            continue
        outcome = _outcome(job)
        if isinstance(event.get("github"), dict):
            event["github"] = {**event["github"], **outcome}
        launched = event.get("outcome")
        if launched != "failure" and outcome["conclusion"] in _FAILED:
            event["outcome"] = "failure"
            if not event.get("error"):
                event["error"] = _job_error(outcome, launched=launched == "success")
    return [*events, *synthesized]


def _ran(event: dict[str, Any], job: dict[str, Any]) -> bool:
    """Whether ``event`` is ``job``'s: by check-run id, else by runner and start time."""
    recorded = event.get("github")
    job_id = recorded.get("job_id") if isinstance(recorded, dict) else None
    if job_id is not None:
        return job_id == job.get("id")
    started, began, ended = (
        _time(value)
        for value in (event.get("started_at"), job.get("started_at"), job.get("completed_at"))
    )
    return (
        event.get("runner") is not None
        and event.get("runner") == job.get("runner_name")
        and None not in (started, began, ended)
        and began <= started <= ended
    )


def _outcome(job: dict[str, Any]) -> dict[str, Any]:
    """The finished job's identity and conclusion, and the step that failed it."""
    steps = job.get("steps") or []
    failed = next((step.get("name") for step in steps if step.get("conclusion") in _FAILED), None)
    return {
        "job_id": job.get("id"),
        "job_url": job.get("html_url"),
        "job_name": job.get("name"),
        "conclusion": job.get("conclusion"),
        "failed_step": failed,
    }


def _job_error(outcome: dict[str, Any], *, launched: bool) -> dict[str, Any]:
    """The error of a failed job its record does not explain.

    A later step failed a job whose launcher succeeded; a cancel, timeout or kill stopped one
    whose launcher never finished.
    """
    conclusion, step = outcome["conclusion"], outcome["failed_step"]
    where = f" at step {step!r}" if step else ""
    if launched:
        kind, when, retriable = STEP_FAILED, "after the launcher succeeded", False
    else:
        kind, when = INTERRUPTED, "before the launcher finished"
        retriable = True if conclusion in {"cancelled", "timed_out"} else None
    return {
        "stage": step or "runner",
        "type": kind,
        "message": f"job concluded {conclusion}{where} {when}",
        "exit_code": None,
        "retriable": retriable,
        "evidence": outcome["job_url"],
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


def summary(events: list[dict[str, Any]], synthesized: int) -> str:
    """Markdown counting ``events`` by outcome, then a row per failure, retriable ones last."""
    failures = sorted(
        (event for event in events if event.get("outcome") == "failure"),
        key=lambda event: (event.get("error") or {}).get("retriable") is True,
    )
    succeeded = sum(event.get("outcome") == "success" for event in events)
    counts = (
        f"**Job events:** {len(events)} total, {succeeded} succeeded, {len(failures)} failed,"
        f" {synthesized} synthesized"
    )
    if not failures:
        return counts + "\n"
    header = _row(_SUMMARY_COLUMNS), _row(["---"] * len(_SUMMARY_COLUMNS))
    return "\n".join([counts, "", *header, *map(_failure, failures)]) + "\n"


def _failure(event: dict[str, Any]) -> str:
    """``event``'s row: its job by benchmark point, its cause by log cause line, else message."""
    error, job = event.get("error") or {}, event.get("github") or {}
    cause = (error.get("detail") or "").partition("\n")[0] or error.get("message") or ""
    if len(cause) > _CAUSE_CHARS:
        cause = cause[: _CAUSE_CHARS - 1] + "…"
    name = job.get("job_name")
    benchmark = _BENCHMARK_JOB.search(name) if name else None
    retriable = error.get("retriable")
    cells = [
        event.get("cluster"),
        benchmark["point"] if benchmark else name,
        error.get("stage"),
        error.get("type"),
        "yes" if retriable is True else "no" if retriable is False else "?",
        cause,
    ]
    link = f"[job]({job['job_url']})" if job.get("job_url") else ""
    return _row([*map(_cell, cells), link])


def _row(cells: Iterable[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _cell(value: object) -> str:
    """``value`` as table cell text, with its HTML, pipes and line breaks escaped."""
    text = html.escape("" if value is None else str(value), quote=False).replace("|", r"\|")
    return "<br>".join(text.splitlines())


def main(argv: list[str] | None = None) -> int:
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
    parser.add_argument(
        "--summary",
        type=Path,
        metavar="PATH",
        help="Markdown file, such as $GITHUB_STEP_SUMMARY, to append the counts and failures to",
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
    if args.summary is not None:
        try:
            with args.summary.open("a", encoding="utf-8") as handle:
                handle.write(summary(events, len(events) - collected))
        except OSError as error:
            print(f"WARNING: could not write {args.summary}: {error}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
