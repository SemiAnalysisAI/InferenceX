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

from authorize_skip_queue import is_authorized

PRIORITY_LABEL_PREFIX = "ci-priority-"
QUEUE_LABEL_PREFIX = "ci-queue-"
SKIP_QUEUE_LABEL_PREFIX = "ci-skip-queue-pr-"
PRIORITY_LABEL_RE = re.compile(
    r"^ci-priority-p(?P<score>[0-9]+(?:\.[0-9]+)?)$"
)
QUEUE_LABEL_RE = re.compile(r"^ci-queue-(?P<token>[a-zA-Z0-9._-]+)$")
SKIP_QUEUE_LABEL_RE = re.compile(r"^ci-skip-queue-pr-(?P<number>[1-9][0-9]*)$")


def parse_timestamp(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def priority_from_labels(labels: Iterable[str]) -> tuple[str, Decimal] | None:
    candidates = [label for label in labels if label.startswith(PRIORITY_LABEL_PREFIX)]
    invalid = [label for label in candidates if not PRIORITY_LABEL_RE.fullmatch(label)]
    if invalid:
        raise ValueError(f"Job has invalid CI priority labels: {invalid}")
    if len(candidates) > 1:
        raise ValueError(f"Job has multiple CI priority labels: {candidates}")
    if not candidates:
        return None
    label = candidates[0]
    match = PRIORITY_LABEL_RE.fullmatch(label)
    return label, Decimal(match.group("score"))


def queue_label_from_labels(labels: Iterable[str]) -> str | None:
    candidates = [label for label in labels if label.startswith(QUEUE_LABEL_PREFIX)]
    invalid = [label for label in candidates if not QUEUE_LABEL_RE.fullmatch(label)]
    if invalid:
        raise ValueError(f"Job has invalid CI queue labels: {invalid}")
    if len(candidates) > 1:
        raise ValueError(f"Job has multiple CI queue labels: {candidates}")
    return candidates[0] if candidates else None

def skip_queue_from_labels(labels: Iterable[str]) -> tuple[str, int] | None:
    candidates = [label for label in labels if label.startswith(SKIP_QUEUE_LABEL_PREFIX)]
    invalid = [label for label in candidates if not SKIP_QUEUE_LABEL_RE.fullmatch(label)]
    if invalid:
        raise ValueError(f"Job has invalid skip_queue labels: {invalid}")
    if len(candidates) > 1:
        raise ValueError(f"Job has multiple skip_queue labels: {candidates}")
    if not candidates:
        return None
    label = candidates[0]
    match = SKIP_QUEUE_LABEL_RE.fullmatch(label)
    return label, int(match.group("number"))


def without_scheduling_labels(labels: Iterable[str]) -> frozenset[str]:
    return frozenset(
        label
        for label in labels
        if not label.startswith(
            (PRIORITY_LABEL_PREFIX, QUEUE_LABEL_PREFIX, SKIP_QUEUE_LABEL_PREFIX)
        )
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
    authorized_skip_prs: frozenset[int] = frozenset(),
) -> Decimal:
    priority = priority_from_labels(job.labels)
    if priority is None:
        raise ValueError(f"Job {job.id} has no CI priority label")
    skip_request = skip_queue_from_labels(job.labels)
    if skip_request is not None and skip_request[1] in authorized_skip_prs:
        return Decimal("Infinity")
    _, score = priority
    waited_hours = Decimal(str(max(0.0, (now - job.queued_at).total_seconds()) / 3600))
    return score + waited_hours * aging_per_hour


def is_compatible(job: QueuedJob, runner: Runner) -> bool:
    required = without_scheduling_labels(job.labels)
    available = without_scheduling_labels(runner.labels)
    return required.issubset(available)


def plan_label_updates(
    jobs: Iterable[QueuedJob],
    runners: Iterable[Runner],
    *,
    aging_per_hour: Decimal = Decimal("0.25"),
    authorized_skip_prs: frozenset[int] = frozenset(),
    now: datetime | None = None,
) -> list[LabelUpdate]:
    """Assign each idle runner to the best compatible queued job.

    Busy and offline runners are never relabeled. Priority and one-shot queue
    labels are removed from unassigned idle runners, so a runner cannot take a
    second job before the controller has considered newly queued higher
    priorities.
    """
    eligible_jobs = []
    for job in jobs:
        priority = priority_from_labels(job.labels)
        queue_label = queue_label_from_labels(job.labels)
        skip_queue_from_labels(job.labels)
        if priority is not None and queue_label is not None:
            eligible_jobs.append(job)
    eligible_jobs.sort(
        key=lambda job: (
            -effective_priority(
                job,
                now=now,
                aging_per_hour=aging_per_hour,
                authorized_skip_prs=authorized_skip_prs,
            ),
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

    for index, job in enumerate(eligible_jobs):
        priority = priority_from_labels(job.labels)
        queue_label = queue_label_from_labels(job.labels)
        skip_request = skip_queue_from_labels(job.labels)
        if priority is None or queue_label is None:
            continue
        priority_label, _ = priority
        candidates = [
            runner for runner in remaining.values() if is_compatible(job, runner)
        ]
        if not candidates:
            continue
        later_jobs = eligible_jobs[index + 1:]
        runner = min(
            candidates,
            key=lambda candidate: (
                sum(is_compatible(later, candidate) for later in later_jobs),
                candidate.name,
                candidate.id,
            ),
        )
        desired[runner.id] = (
            without_scheduling_labels(runner.labels)
            | {priority_label, queue_label}
            | ({skip_request[0]} if skip_request is not None else set()),
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
        except urllib.error.URLError as error:
            raise RuntimeError(f"GitHub API {method} {path} failed: {error}") from error
        return json.loads(raw) if raw else None

    def paged(
        self,
        path: str,
        key: str | None,
    ) -> list[dict[str, Any]]:
        separator = "&" if "?" in path else "?"
        values = []
        page = 1
        while True:
            payload = self.request("GET", f"{path}{separator}per_page=100&page={page}")
            batch = payload if key is None else payload[key]
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


class AuthorizationApi:
    def __init__(self, client: GitHubClient):
        self.client = client

    def paged(self, path: str) -> list[dict[str, Any]]:
        return self.client.paged(path, None)

    def request(self, path: str) -> Any:
        return self.client.request("GET", path)


def load_policy(policy_path: str | Path) -> dict[str, Any]:
    with Path(policy_path).open() as policy_file:
        return yaml.safe_load(policy_file)


def authorize_skip_requests(
    client: GitHubClient,
    jobs: Iterable[QueuedJob],
    policy: dict[str, Any],
) -> frozenset[int]:
    requests = {
        request[1]
        for job in jobs
        if (request := skip_queue_from_labels(job.labels)) is not None
    }
    skip_policy = policy["labels"]["skip-queue"]
    api = AuthorizationApi(client)
    authorized = set()
    for pr_number in sorted(requests):
        try:
            allowed, actor = is_authorized(
                api,
                repository=client.repository,
                pr_number=pr_number,
                organization=skip_policy["organization"],
                team_slug=skip_policy["team-slug"],
                label_name=skip_policy["name"],
            )
        except RuntimeError as error:
            print(f"::warning::{error}; refusing skip_queue authorization", file=sys.stderr)
            continue
        if allowed:
            print(f"::notice::skip_queue authorized by {actor}", file=sys.stderr)
            authorized.add(pr_number)
    return frozenset(authorized)


def load_aging_rate(policy: dict[str, Any]) -> Decimal:
    return Decimal(str(policy["scheduler"]["aging-per-hour"]))


def reconcile(
    client: GitHubClient,
    policy: dict[str, Any],
    apply: bool,
) -> list[LabelUpdate]:
    jobs = client.queued_jobs()
    updates = plan_label_updates(
        jobs,
        client.runners(),
        aging_per_hour=load_aging_rate(policy),
        authorized_skip_prs=authorize_skip_requests(client, jobs, policy),
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
    plan_parser.add_argument("--authorized-skip-pr", action="append", type=int, default=[])

    reconcile_parser = subparsers.add_parser("reconcile", help="Poll GitHub and reconcile runner labels")
    reconcile_parser.add_argument("--repository", required=True)
    reconcile_parser.add_argument("--api-url", default="https://api.github.com")
    # REPO_PAT needs Actions and Issues read.
    # REPO_PAT needs Administration write and Members read.
    reconcile_parser.add_argument("--token-env", default="REPO_PAT")
    reconcile_parser.add_argument("--apply", action="store_true")
    reconcile_parser.add_argument("--watch", type=float, default=0, metavar="SECONDS")

    args = parser.parse_args()
    policy = load_policy(args.policy)

    if args.command == "plan":
        raw_jobs = _load_json(args.jobs)
        raw_runners = _load_json(args.runners)
        jobs = [QueuedJob.from_payload(job) for job in raw_jobs]
        runners = [Runner.from_payload(runner) for runner in raw_runners]
        now = parse_timestamp(args.now) if args.now else None
        updates = plan_label_updates(
            jobs,
            runners,
            aging_per_hour=load_aging_rate(policy),
            authorized_skip_prs=frozenset(args.authorized_skip_pr),
            now=now,
        )
        json.dump([update.as_dict() for update in updates], sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    token = os.environ.get(args.token_env)
    if not token:
        parser.error(f"{args.token_env} must contain a GitHub token")
    client = GitHubClient(args.repository, token, args.api_url)
    while True:
        try:
            updates = reconcile(client, policy, args.apply)
        except RuntimeError as error:
            if args.watch <= 0:
                raise
            print(f"::warning::{error}; retrying", file=sys.stderr)
            time.sleep(args.watch)
            continue
        print(json.dumps([update.as_dict() for update in updates], separators=(",", ":")))
        if args.watch <= 0:
            return 0
        time.sleep(args.watch)


if __name__ == "__main__":
    raise SystemExit(main())
