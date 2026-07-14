#!/usr/bin/env python3
"""Plan or apply priority labels for idle self-hosted GitHub Actions runners."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

import yaml

PRIORITY_LABEL_RE = re.compile(
    r"^ci-priority-p(?P<score>skip|[0-9]+(?:\.[0-9]+)?)$"
)
QUEUE_LABEL_RE = re.compile(r"^ci-queue-(?P<token>[a-zA-Z0-9._-]+)$")


def parse_timestamp(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def priority_from_labels(labels: Iterable[str]) -> tuple[str, Decimal] | None:
    matches = []
    for label in labels:
        match = PRIORITY_LABEL_RE.fullmatch(label)
        if match:
            raw_score = match.group("score")
            score = Decimal("Infinity") if raw_score == "skip" else Decimal(raw_score)
            matches.append((label, score))
    if len(matches) > 1:
        raise ValueError(f"Job has multiple CI priority labels: {[label for label, _ in matches]}")
    return matches[0] if matches else None


def queue_label_from_labels(labels: Iterable[str]) -> str | None:
    matches = [label for label in labels if QUEUE_LABEL_RE.fullmatch(label)]
    if len(matches) > 1:
        raise ValueError(f"Job has multiple CI queue labels: {matches}")
    return matches[0] if matches else None


def without_scheduling_labels(labels: Iterable[str]) -> frozenset[str]:
    return frozenset(
        label
        for label in labels
        if not PRIORITY_LABEL_RE.fullmatch(label) and not QUEUE_LABEL_RE.fullmatch(label)
    )


@dataclass(frozen=True)
class QueuedJob:
    id: int
    run_id: int
    labels: frozenset[str]
    queued_at: datetime
    name: str = ""

    @classmethod
    def from_payload(cls, payload: dict[str, Any], run_id: int | None = None) -> "QueuedJob":
        return cls(
            id=int(payload["id"]),
            run_id=int(run_id if run_id is not None else payload.get("run_id", 0)),
            labels=frozenset(payload.get("labels", [])),
            queued_at=parse_timestamp(payload.get("created_at") or payload.get("queued_at")),
            name=str(payload.get("name", "")),
        )


@dataclass(frozen=True)
class Runner:
    id: int
    name: str
    labels: frozenset[str]
    status: str
    busy: bool

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "Runner":
        raw_labels = payload.get("labels", [])
        labels = [item["name"] if isinstance(item, dict) else item for item in raw_labels]
        return cls(
            id=int(payload["id"]),
            name=str(payload["name"]),
            labels=frozenset(labels),
            status=str(payload.get("status", "offline")),
            busy=bool(payload.get("busy", False)),
        )


@dataclass(frozen=True)
class LabelUpdate:
    runner_id: int
    runner_name: str
    labels: tuple[str, ...]
    assigned_job_id: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "runner_id": self.runner_id,
            "runner_name": self.runner_name,
            "labels": list(self.labels),
            "assigned_job_id": self.assigned_job_id,
        }


def effective_priority(
    job: QueuedJob,
    *,
    now: datetime,
    aging_per_hour: Decimal,
) -> Decimal:
    priority = priority_from_labels(job.labels)
    if priority is None:
        raise ValueError(f"Job {job.id} has no CI priority label")
    _, score = priority
    waited_hours = Decimal(str(max(0.0, (now - job.queued_at).total_seconds()) / 3600))
    return score + waited_hours * aging_per_hour


def plan_label_updates(
    jobs: Iterable[QueuedJob],
    runners: Iterable[Runner],
    *,
    aging_per_hour: Decimal = Decimal("0.25"),
    now: datetime | None = None,
) -> list[LabelUpdate]:
    """Assign each idle runner to the best compatible queued job.

    Busy and offline runners are never relabeled. Priority and one-shot queue
    labels are removed from unassigned idle runners, so a runner cannot take a
    second job before the controller has considered newly queued higher
    priorities.
    """
    now = now or datetime.now(timezone.utc)
    eligible_jobs = [
        job
        for job in jobs
        if priority_from_labels(job.labels) is not None
        and queue_label_from_labels(job.labels) is not None
    ]
    eligible_jobs.sort(
        key=lambda job: (
            -effective_priority(job, now=now, aging_per_hour=aging_per_hour),
            job.queued_at,
            job.id,
        )
    )

    idle_runners = [runner for runner in runners if runner.status == "online" and not runner.busy]
    remaining = {runner.id: runner for runner in idle_runners}
    desired: dict[int, tuple[frozenset[str], int | None]] = {
        runner.id: (without_scheduling_labels(runner.labels), None)
        for runner in idle_runners
    }

    for job in eligible_jobs:
        priority = priority_from_labels(job.labels)
        queue_label = queue_label_from_labels(job.labels)
        if priority is None or queue_label is None:
            continue
        priority_label, _ = priority
        required = without_scheduling_labels(job.labels)
        candidates = [
            runner
            for runner in remaining.values()
            if required.issubset(without_scheduling_labels(runner.labels))
        ]
        if not candidates:
            continue
        runner = min(candidates, key=lambda candidate: (candidate.name, candidate.id))
        desired[runner.id] = (
            without_scheduling_labels(runner.labels) | {priority_label, queue_label},
            job.id,
        )
        del remaining[runner.id]

    updates = []
    for runner in sorted(idle_runners, key=lambda item: (item.name, item.id)):
        labels, job_id = desired[runner.id]
        if labels != runner.labels:
            updates.append(
                LabelUpdate(
                    runner_id=runner.id,
                    runner_name=runner.name,
                    labels=tuple(sorted(labels)),
                    assigned_job_id=job_id,
                )
            )
    return updates


class GitHubClient:
    def __init__(self, repository: str, token: str, api_url: str = "https://api.github.com"):
        if repository.count("/") != 1:
            raise ValueError("Repository must be OWNER/REPO")
        self.repository = repository
        self.token = token
        self.api_url = api_url.rstrip("/")

    def request(self, method: str, path: str, body: Any = None) -> Any:
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            f"{self.api_url}{path}",
            data=data,
            method=method,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")
            raise RuntimeError(f"GitHub API {method} {path} failed: {error.code} {detail}") from error
        return json.loads(raw) if raw else None

    def paged(self, path: str, key: str) -> list[dict[str, Any]]:
        separator = "&" if "?" in path else "?"
        values = []
        page = 1
        while True:
            payload = self.request("GET", f"{path}{separator}per_page=100&page={page}")
            batch = payload[key]
            values.extend(batch)
            if len(batch) < 100:
                return values
            page += 1

    def queued_jobs(self) -> list[QueuedJob]:
        base = f"/repos/{self.repository}"
        runs_by_id = {}
        for status in ("queued", "in_progress"):
            runs = self.paged(
                f"{base}/actions/runs?status={status}",
                "workflow_runs",
            )
            runs_by_id.update({int(run["id"]): run for run in runs})

        jobs = []
        for run_id in sorted(runs_by_id):
            run_jobs = self.paged(
                f"{base}/actions/runs/{run_id}/jobs?filter=latest",
                "jobs",
            )
            jobs.extend(
                QueuedJob.from_payload(job, run_id)
                for job in run_jobs
                if job.get("status") == "queued"
            )
        return jobs

    def runners(self) -> list[Runner]:
        payloads = self.paged(f"/repos/{self.repository}/actions/runners", "runners")
        return [Runner.from_payload(payload) for payload in payloads]

    def replace_runner_labels(self, runner_id: int, labels: Iterable[str]) -> None:
        self.request(
            "PUT",
            f"/repos/{self.repository}/actions/runners/{runner_id}/labels",
            {"labels": sorted(labels)},
        )


def load_aging_rate(policy_path: str | Path) -> Decimal:
    with Path(policy_path).open() as policy_file:
        policy = yaml.safe_load(policy_file)
    return Decimal(str(policy["scheduler"]["aging-per-hour"]))


def reconcile(client: GitHubClient, aging_per_hour: Decimal, apply: bool) -> list[LabelUpdate]:
    updates = plan_label_updates(
        client.queued_jobs(),
        client.runners(),
        aging_per_hour=aging_per_hour,
    )
    if apply:
        for update in updates:
            client.replace_runner_labels(update.runner_id, update.labels)
    return updates


def _load_json(path: Path) -> Any:
    with path.open() as source:
        return json.load(source)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="configs/ci-priority.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="Plan from local API response fixtures")
    plan_parser.add_argument("--jobs", type=Path, required=True)
    plan_parser.add_argument("--runners", type=Path, required=True)
    plan_parser.add_argument("--now", help="ISO-8601 clock for deterministic simulations")

    reconcile_parser = subparsers.add_parser("reconcile", help="Poll GitHub and reconcile runner labels")
    reconcile_parser.add_argument("--repository", required=True)
    reconcile_parser.add_argument("--api-url", default="https://api.github.com")
    reconcile_parser.add_argument("--token-env", default="GITHUB_TOKEN")
    reconcile_parser.add_argument("--apply", action="store_true")
    reconcile_parser.add_argument("--watch", type=float, default=0, metavar="SECONDS")

    args = parser.parse_args()
    aging_per_hour = load_aging_rate(args.policy)

    if args.command == "plan":
        raw_jobs = _load_json(args.jobs)
        raw_runners = _load_json(args.runners)
        jobs = [QueuedJob.from_payload(job) for job in raw_jobs]
        runners = [Runner.from_payload(runner) for runner in raw_runners]
        now = parse_timestamp(args.now) if args.now else None
        updates = plan_label_updates(jobs, runners, aging_per_hour=aging_per_hour, now=now)
        json.dump([update.as_dict() for update in updates], sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    token = os.environ.get(args.token_env)
    if not token:
        parser.error(f"{args.token_env} must contain a GitHub token")
    client = GitHubClient(args.repository, token, args.api_url)
    while True:
        updates = reconcile(client, aging_per_hour, args.apply)
        print(json.dumps([update.as_dict() for update in updates], separators=(",", ":")))
        if args.watch <= 0:
            return 0
        time.sleep(args.watch)


if __name__ == "__main__":
    raise SystemExit(main())
