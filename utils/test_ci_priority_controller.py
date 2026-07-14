from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from ci_priority_controller import (
    GitHubClient,
    QueuedJob,
    Runner,
    authorize_skip_requests,
    parse_timestamp,
    plan_label_updates,
)


NOW = datetime(2026, 7, 14, 18, 0, tzinfo=timezone.utc)


def job(job_id, priority, hardware, queued_minutes=0):
    return QueuedJob(
        id=job_id,
        run_id=100 + job_id,
        labels=frozenset({
            "self-hosted",
            hardware,
            f"ci-priority-p{priority}",
            f"ci-queue-job-{job_id}",
        }),
        queued_at=NOW - timedelta(minutes=queued_minutes),
        name=f"job-{job_id}",
    )


def runner(runner_id, name, *labels, busy=False, status="online"):
    return Runner(
        id=runner_id,
        name=name,
        labels=frozenset({"self-hosted", "Linux", "X64", name, *labels}),
        status=status,
        busy=busy,
    )


def test_naive_timestamps_default_to_utc():
    now = parse_timestamp("2026-07-14T18:00:00")

    updates = plan_label_updates(
        [job(1, "1.000", "h100")],
        [runner(11, "h100_00", "h100")],
        now=now,
    )

    assert now.tzinfo == timezone.utc
    assert updates[0].assigned_job_id == 1


def test_assigns_best_compatible_job_to_each_idle_runner():
    jobs = [
        job(1, "5.000", "h100", queued_minutes=30),
        job(2, "0.700", "b200", queued_minutes=1),
        job(3, "2.500", "h100", queued_minutes=2),
    ]
    runners = [
        runner(10, "b200_00", "b200"),
        runner(11, "h100_00", "h100"),
    ]

    updates = plan_label_updates(jobs, runners, now=NOW)

    assert [(update.runner_name, update.assigned_job_id) for update in updates] == [
        ("b200_00", 2),
        ("h100_00", 1),
    ]
    assert "ci-priority-p0.700" in updates[0].labels
    assert "ci-priority-p5.000" in updates[1].labels
    assert "ci-queue-job-2" in updates[0].labels
    assert "ci-queue-job-1" in updates[1].labels


def test_aging_breaks_starvation_between_nearby_priorities():
    jobs = [
        job(1, "1.000", "h100", queued_minutes=8 * 60),
        job(2, "2.500", "h100", queued_minutes=1),
    ]
    runners = [runner(11, "h100_00", "h100")]

    updates = plan_label_updates(
        jobs,
        runners,
        now=NOW,
        aging_per_hour=Decimal("0.25"),
    )
    assert updates[0].assigned_job_id == 1

def test_authorized_skip_queue_outranks_numeric_priority():
    skip_job = job(2, "1.000", "h100")
    skip_job = replace(
        skip_job,
        labels=skip_job.labels | {"ci-skip-queue-pr-2124"},
    )
    jobs = [
        job(1, "1000000.000", "h100"),
        skip_job,
    ]
    runners = [runner(11, "h100_00", "h100")]

    updates = plan_label_updates(
        jobs,
        runners,
        authorized_skip_prs=frozenset({2124}),
        now=NOW,
    )

    assert updates[0].assigned_job_id == 2
    assert "ci-priority-p1.000" in updates[0].labels
    assert "ci-skip-queue-pr-2124" in updates[0].labels


def test_unauthorized_skip_queue_remains_numeric():
    skip_job = job(2, "1.000", "h100")
    skip_job = replace(
        skip_job,
        labels=skip_job.labels | {"ci-skip-queue-pr-2124"},
    )

    updates = plan_label_updates(
        [job(1, "2.000", "h100"), skip_job],
        [runner(11, "h100_00", "h100")],
        now=NOW,
    )

    assert updates[0].assigned_job_id == 1



def test_controller_verifies_skip_queue_actor():
    class FixtureClient(GitHubClient):
        def __init__(self):
            super().__init__("owner/repo", "unused")

        def paged(self, path, key):
            assert key is None
            assert path == "/repos/owner/repo/issues/2124/timeline"
            return [{
                "event": "labeled",
                "label": {"name": "skip_queue"},
                "actor": {"login": "alice"},
            }]

        def request(self, method, path, body=None):
            assert method == "GET"
            assert path == "/orgs/SemiAnalysisAI/teams/core/memberships/alice"
            return {"state": "active"}

    skip_job = job(2, "1.000", "h100")
    skip_job = replace(
        skip_job,
        labels=skip_job.labels | {"ci-skip-queue-pr-2124"},
    )
    policy = {
        "labels": {
            "skip-queue": {
                "name": "skip_queue",
                "organization": "SemiAnalysisAI",
                "team-slug": "core",
            }
        }
    }

    assert authorize_skip_requests(
        FixtureClient(),
        [skip_job],
        policy,
    ) == frozenset({2124})


def test_unused_idle_runner_loses_stale_priority_label():
    runners = [
        runner(
            10,
            "b200_00",
            "b200",
            "ci-priority-p5.000",
            "ci-queue-old-job",
        )
    ]

    updates = plan_label_updates([], runners, now=NOW)

    assert len(updates) == 1
    assert updates[0].assigned_job_id is None
    assert all(not label.startswith("ci-priority-p") for label in updates[0].labels)
    assert all(not label.startswith("ci-queue-") for label in updates[0].labels)


def test_busy_and_offline_runners_are_never_relabelled():
    runners = [
        runner(10, "b200_00", "b200", "ci-priority-p5.000", busy=True),
        runner(11, "b200_01", "b200", "ci-priority-p5.000", status="offline"),
    ]

    assert plan_label_updates([job(1, "0.700", "b200")], runners, now=NOW) == []


def test_exact_runner_name_remains_a_compatibility_constraint():
    jobs = [job(1, "1.000", "h100_01")]
    runners = [
        runner(10, "h100_00", "h100"),
        runner(11, "h100_01", "h100"),
    ]

    updates = plan_label_updates(jobs, runners, now=NOW)

    assigned = [update for update in updates if update.assigned_job_id is not None]
    assert len(assigned) == 1
    assert assigned[0].runner_name == "h100_01"


def test_preserves_scarce_runner_for_exact_job():
    jobs = [
        job(1, "5.000", "h100"),
        job(2, "4.000", "h100_00"),
    ]
    runners = [
        runner(10, "h100_00", "h100"),
        runner(11, "h100_01", "h100"),
    ]

    updates = plan_label_updates(jobs, runners, now=NOW)

    assert [(update.runner_name, update.assigned_job_id) for update in updates] == [
        ("h100_00", 2),
        ("h100_01", 1),
    ]


@pytest.mark.parametrize(
    ("valid", "invalid", "message"),
    [
        ("ci-priority-p1.000", "ci-priority-p-1", "invalid CI priority"),
        ("ci-queue-job-1", "ci-queue-", "invalid CI queue"),
        ("ci-skip-queue-pr-2124", "ci-skip-queue-pr-zero", "invalid skip_queue"),
    ],
)
def test_rejects_malformed_scheduling_labels(valid, invalid, message):
    queued_job = job(1, "1.000", "h100")
    queued_job = replace(
        queued_job,
        labels=(queued_job.labels - {valid}) | {invalid},
    )

    with pytest.raises(ValueError, match=message):
        plan_label_updates(
            [queued_job],
            [runner(11, "h100_00", "h100")],
            now=NOW,
        )


def test_discovers_queued_jobs_inside_in_progress_workflow_runs():
    class FixtureClient(GitHubClient):
        def __init__(self):
            super().__init__("owner/repo", "unused")
            self.paths = []

        def paged(self, path, key):
            self.paths.append(path)
            if "status=queued" in path:
                return [{"id": 1}]
            if "status=in_progress" in path:
                return [{"id": 2}]
            if "/runs/1/jobs" in path:
                return [{"id": 11, "status": "completed", "labels": []}]
            if "/runs/2/jobs" in path:
                return [{
                    "id": 22,
                    "status": "queued",
                    "created_at": "2026-07-14T18:00:00Z",
                    "labels": [
                        "self-hosted",
                        "b200",
                        "ci-priority-p1.000",
                        "ci-queue-job-22",
                    ],
                }]
            raise AssertionError(path)

    client = FixtureClient()

    jobs = client.queued_jobs()

    assert [queued_job.id for queued_job in jobs] == [22]
    assert any("status=queued" in path for path in client.paths)
    assert any("status=in_progress" in path for path in client.paths)
