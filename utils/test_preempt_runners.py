"""Tests for the preemption selection logic in preempt_runners.py."""

from preempt_runners import job_matches_targets, runner_labels_by_name, select_runs_to_preempt

RUNNER_INDEX = {
    "b200-dgxc_06": {"self-hosted", "b200", "b200-dsv4", "b200-dgxc", "b200-multinode"},
    "b200-dgxc_07": {"self-hosted", "b200", "b200-dsv4", "b200-dgxc", "b200-disagg"},
    "h100-dgxc-slurm_06": {"self-hosted", "h100", "h100-dgxc"},
}


def job(status: str, labels: list[str], runner_name: str | None = None, name: str = "j") -> dict:
    return {"status": status, "labels": labels, "runner_name": runner_name, "name": name}


def run(run_id: int, name: str = "Run Sweep", branch: str = "some-branch") -> dict:
    return {
        "id": run_id,
        "name": name,
        "head_branch": branch,
        "event": "pull_request",
        "html_url": f"https://example.com/runs/{run_id}",
    }


class TestJobMatchesTargets:
    def test_in_progress_job_matched_via_runner_label_set(self):
        # The job requested b200-multinode but occupies a node that also
        # serves b200 — a b200 priority sweep must preempt it.
        j = job("in_progress", ["b200-multinode"], runner_name="b200-dgxc_06")
        assert job_matches_targets(j, {"b200"}, RUNNER_INDEX)

    def test_in_progress_job_on_unrelated_runner_not_matched(self):
        j = job("in_progress", ["h100"], runner_name="h100-dgxc-slurm_06")
        assert not job_matches_targets(j, {"b200"}, RUNNER_INDEX)

    def test_completed_job_never_matched(self):
        j = job("completed", ["b200"], runner_name="b200-dgxc_06")
        assert not job_matches_targets(j, {"b200"}, RUNNER_INDEX)

    def test_queued_job_matched_when_a_target_runner_could_serve_it(self):
        # Queued b200-multinode job competes for the b200 nodes we free up.
        j = job("queued", ["b200-multinode"])
        assert job_matches_targets(j, {"b200"}, RUNNER_INDEX)

    def test_queued_job_not_matched_when_no_target_runner_serves_it(self):
        j = job("queued", ["h100"])
        assert not job_matches_targets(j, {"b200"}, RUNNER_INDEX)

    def test_fallback_without_runner_index_uses_requested_labels(self):
        assert job_matches_targets(job("in_progress", ["b200"]), {"b200"}, None)
        assert not job_matches_targets(job("in_progress", ["b200-multinode"]), {"b200"}, None)

    def test_ubuntu_jobs_never_matched(self):
        j = job("in_progress", ["ubuntu-latest"])
        assert not job_matches_targets(j, {"b200"}, RUNNER_INDEX)


class TestSelectRunsToPreempt:
    def test_skips_self_run(self):
        runs = [(run(1), [job("in_progress", ["b200"], "b200-dgxc_06")])]
        assert select_runs_to_preempt(runs, {"b200"}, RUNNER_INDEX, self_run_id=1) == []

    def test_selects_run_with_one_colliding_job(self):
        runs = [
            (run(1), [job("in_progress", ["b200"], "b200-dgxc_06", name="bmk-b200")]),
            (run(2), [job("in_progress", ["h100"], "h100-dgxc-slurm_06")]),
        ]
        selected = select_runs_to_preempt(runs, {"b200"}, RUNNER_INDEX, self_run_id=99)
        assert [entry["run_id"] for entry in selected] == [1]
        assert selected[0]["matching_jobs"] == ["bmk-b200"]

    def test_run_with_only_completed_jobs_not_selected(self):
        runs = [(run(1), [job("completed", ["b200"], "b200-dgxc_06")])]
        assert select_runs_to_preempt(runs, {"b200"}, RUNNER_INDEX, self_run_id=99) == []


def test_runner_labels_by_name():
    runners = [
        {"name": "b200-dgxc_06", "labels": [{"name": "b200"}, {"name": "b200-multinode"}]},
        {"labels": [{"name": "ignored-no-name"}]},
    ]
    assert runner_labels_by_name(runners) == {"b200-dgxc_06": {"b200", "b200-multinode"}}
