import json
import os
import shlex
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from infx.klaud import __main__ as klaud
from infx.klaud import api, claims, github, lifecycle, reporting, validation
from infx.klaud.models import CandidateOutcome, Feed, OwnedCandidate, Ownership, identity, stamp
from infx.tests.historical_revision import (
    AGENTX_MASTER,
    FAMILY,
    KEY,
    POINTS,
    commit_history,
    forbid_current_config_parsing,
)
from infx.tests.srt_recipes import shared_blocks, single_node_fragment


@pytest.mark.parametrize("current_head,expected", [("ours", True), ("other", False)])
def test_claim_conflict_checks_the_actual_owner(monkeypatch, current_head, expected):
    def run(args, **kwargs):
        endpoint = next(arg for arg in args if arg.startswith("repos/")).partition("?")[
            0
        ]
        method = args[args.index("--method") + 1]
        if endpoint.endswith("git/commits/base") and method == "GET":
            response = {"tree": {"sha": "tree"}}
        elif endpoint.endswith("git/commits") and method == "POST":
            request = json.loads(kwargs["input"])
            assert request == {
                "message": '{"owner": 42}',
                "tree": "tree",
                "parents": ["base"],
            }
            response = {"sha": "ours"}
        elif endpoint.endswith("git/refs") and method == "POST":
            raise subprocess.CalledProcessError(
                1, args, stderr="reference already exists"
            )
        elif endpoint.endswith("git/matching-refs/heads/claim") and method == "GET":
            response = [[{"ref": "refs/heads/claim", "object": {"sha": current_head}}]]
        else:
            raise AssertionError((method, endpoint))
        return subprocess.CompletedProcess(args, 0, json.dumps(response), "")

    monkeypatch.setattr(github.subprocess, "run", run)
    assert claims.create("example/project", "claim", {"owner": 42}, "base") is expected


def test_delete_accepts_empty_response(monkeypatch):
    def run(args, **kwargs):
        assert args[args.index("--method") + 1] == "DELETE"
        assert json.loads(kwargs["input"]) == {}
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(github.subprocess, "run", run)
    assert github.write("example/project", "git/refs/heads/claim", "DELETE") == {}


def test_session_listings_see_the_state_after_a_write(monkeypatch):
    session = lifecycle.Session(
        "example/project", {"id": 42}, OwnedCandidate(id=CANDIDATE_ID, family=FAMILY, base="a" * 40)
    )
    refs = [{"ref": f"refs/heads/{session.branch}"}]
    monkeypatch.setattr(github, "items", lambda *_args: list(refs))
    monkeypatch.setattr(github.github, "api", lambda *_args, **_kwargs: refs.clear() or {})

    assert session.branch_exists() is True
    github.write("example/project", f"git/refs/heads/{session.branch}", "DELETE")
    assert session.branch_exists() is False


@pytest.mark.parametrize(("contents", "data"), [
    ({"type": "file", "encoding": "base64", "size": 3, "content": "YWJj\n"}, b"abc"),
    ([{"type": "file", "name": "recipe.yaml", "path": "recipes/recipe.yaml"}], None),
], ids=["file", "directory"])  # fmt: skip
def test_source_reads_return_file_bytes_or_a_fixed_failure(monkeypatch, contents, data):
    def run(args, **_kwargs):
        assert args[4] == f"repos/example/project/contents/recipes?ref={'a' * 40}"
        return subprocess.CompletedProcess(args, 0, json.dumps(contents), "")

    monkeypatch.setattr(github.subprocess, "run", run)

    if data is None:
        with pytest.raises(github.VerificationError, match="Source data unavailable or too large"):
            github.file_at("example/project", "a" * 40, "recipes")
    else:
        assert github.file_at("example/project", "a" * 40, "recipes") == data


@pytest.mark.parametrize(
    "failure,reason",
    [
        ("command", "State unavailable or invalid; inspect GitHub before retrying"),
        ("json", "State unavailable or invalid; inspect GitHub before retrying"),
        ("shape", "GitHub listing returned an unexpected shape"),
        ("incomplete", "Incomplete GitHub listing"),
    ],
)
def test_recovery_errors_do_not_publish_raw_api_data(
    tmp_path, monkeypatch, capfd, failure, reason
):
    responses = {
        "json": "private",
        "shape": '[{"artifacts": ["private"], "total_count": 1}]',
        "incomplete": '[{"artifacts": [], "total_count": 1, "detail": "private"}]',
    }
    command = (
        "printf '%s' private >&2\nexit 1"
        if failure == "command"
        else f"printf '%s' {shlex.quote(responses[failure])}"
    )
    executable = tmp_path / "gh"
    executable.write_text(f"#!/bin/sh\n{command}\n")
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("GITHUB_REPOSITORY", "example/project")
    monkeypatch.setattr(sys, "argv", ["klaud", "recover", "--max-sessions", "5"])
    assert klaud.main() == 1
    captured = capfd.readouterr()
    assert captured.out == f"::error::Klaud: {reason}.\n"
    assert captured.err == ""


def test_diagnostics_prefers_verified_outcome_when_action_output_is_invalid(
    tmp_path, monkeypatch
):
    class Session:
        def pulls(self):
            return []

        def runs(self):
            return []

        def report(self, _pull):
            raise AssertionError("No PR should not request a report")

        def verify(self, outcome):
            assert outcome == CandidateOutcome(
                outcome="failed",
                phase="baseline",
                pull_request=None,
                run_ids=[],
                repairs_used=0,
                reason_code="baseline-provenance-unverified",
            )

    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "outcome.json").write_text(
        '{"outcome":"failed","phase":"baseline","pull-request":null,'
        '"run-ids":[],"repairs-used":0,"reason-code":"baseline-provenance-unverified"}\n'
    )
    execution = tmp_path / "execution.json"
    execution.write_text("{}\n")
    structured = tmp_path / "structured.json"
    structured.write_text("not json\n")
    output = tmp_path / "diagnostics.json"
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("KLAUD_EVIDENCE", str(evidence))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setenv("GITHUB_REPOSITORY", "example/project")
    monkeypatch.setattr(lifecycle, "current_session", Session)

    assert klaud.save_diagnostics(execution, structured, "success", output)
    diagnostics = json.loads(output.read_text())
    assert diagnostics["outcome-source"] == "verified-outcome"
    assert diagnostics["outcome-report"] == "available"
    assert diagnostics["candidate-outcome"] == {
        "outcome": "failed",
        "phase": "baseline",
        "pull-request": None,
        "run-ids": [],
        "repairs-used": 0,
        "reason-code": "baseline-provenance-unverified",
    }
    assert "baseline-provenance-unverified" in summary.read_text()


def test_candidate_outcome_rejects_raw_reason_and_preserves_legacy_receipts():
    legacy = CandidateOutcome.model_validate(
        {
            "outcome": "failed",
            "phase": "baseline",
            "pull-request": None,
            "run-ids": [],
            "repairs-used": 0,
        }
    )
    assert "reason-code" not in legacy.model_dump(by_alias=True, exclude_unset=True)
    with pytest.raises(ValueError):
        CandidateOutcome.model_validate(
            {
                **legacy.model_dump(by_alias=True, exclude_unset=True),
                "reason-code": "private error",
            }
        )


@pytest.mark.parametrize(("outcome", "phase", "reason", "valid"), [
    ("readiness-blocked", "targeted", "weights-missing", True),
    ("failed", "targeted", "weights-missing", False),
    ("failed", "baseline", "weights-missing", False),
    ("readiness-blocked", "targeted", "baseline-other", False),
    ("failed", "baseline", "baseline-other", True),
])  # fmt: skip
def test_reason_codes_pair_only_with_their_outcome(outcome, phase, reason, valid):
    record = {"outcome": outcome, "phase": phase, "pull-request": None, "run-ids": [],
              "repairs-used": 0, "reason-code": reason}
    if valid:
        assert CandidateOutcome.model_validate(record).reason_code == reason
    else:
        with pytest.raises(ValueError):
            CandidateOutcome.model_validate(record)


def test_finish_requires_specific_baseline_reason_before_touching_session(
    tmp_path, monkeypatch, capfd
):
    requested = tmp_path / "requested-outcome.json"
    requested.write_text(
        json.dumps(
            {
                "outcome": "failed",
                "phase": "baseline",
                "pull-request": None,
                "run-ids": [],
                "repairs-used": 0,
            }
        )
    )
    monkeypatch.setattr(
        sys, "argv", ["klaud", "finish", "--outcome-file", str(requested)]
    )
    monkeypatch.setattr(
        lifecycle,
        "current_session",
        lambda: pytest.fail(
            "An unclassified baseline failure must not reach lifecycle cleanup"
        ),
    )
    assert klaud.main() == 1
    assert capfd.readouterr().out == (
        "::error::Klaud: Baseline failure requires a fixed reason-code.\n"
    )


@pytest.mark.parametrize("command", ["report", "check-final"])
def test_pr_commands_fail_cleanly_before_the_candidate_pr_exists(
    tmp_path, monkeypatch, capfd, command
):
    record = tmp_path / "record.json"
    record.write_text(reporting.Baseline(
        family=FAMILY, date="2026-09-19", image="example/image:1",
        goal=reporting.Prose(en="Update the image.", zh="更新镜像。"), sources=[], points=[],
    ).model_dump_json(by_alias=True))  # fmt: skip
    options = (["--kind", "baseline", "--file"] if command == "report" else ["--matrix-file"])
    monkeypatch.setattr(sys, "argv", ["klaud", command, *options, str(record)])
    monkeypatch.setattr(lifecycle, "current_session", lambda: SimpleNamespace(pulls=lambda: []))

    assert klaud.main() == 1
    assert capfd.readouterr().out == "::error::Klaud: Candidate PR not found.\n"


def test_cooldown_skips_recent_blocks_across_bases_and_retries_capacity_deferrals(
    monkeypatch,
):
    blocked, deferred, unreadable, superseded, untried, not_started, weightless = (
        f"{digit * 16}-{digit * 16}" for digit in "1234567"
    )
    outcomes = {
        1: ("readiness-blocked", "targeted", None),
        2: ("capacity-deferred", "targeted", None),
        4: ("failed", "targeted", None),
        5: ("capacity-deferred", "targeted", None),
        7: ("unexpected-error", "unknown", None),
        8: ("readiness-blocked", "targeted", "weights-missing"),
    }
    sources = {
        blocked: {"model": "dsr1", "precision": "fp8", "hardware": "b200"},
        deferred: {"model": "glm5.2", "precision": "fp4", "hardware": "b200"},
        weightless: {"model": "dsr1", "precision": "fp4", "hardware": "b200"},
    }

    def items(_repository, path, key):
        if path.startswith("actions/workflows/klaud-plan.yml/runs?"):
            return [
                {"id": 41, "created_at": "9999-01-01T00:00:00Z", "head_sha": "a" * 40},
                {"id": 42, "created_at": "9999-01-02T00:00:00Z", "head_sha": "b" * 40},
                {"id": 40, "created_at": "2000-01-01T00:00:00Z", "head_sha": "c" * 40},
            ]
        run_artifacts = {
            41: [(1, blocked), (3, unreadable), (4, superseded), (91, "plan")],
            42: [(2, deferred), (5, superseded), (7, not_started), (8, weightless), (92, "plan")],
            40: [(6, untried)],
        }
        run_id = int(path.split("/")[2])
        return [
            {"id": artifact, "expired": False, "created_at": "9999-01-01T00:00:00Z",
             "name": "klaud-plan" if candidate == "plan" else f"klaud-candidate-{candidate}"}
            for artifact, candidate in run_artifacts[run_id]
        ]

    def download(_repository, artifact, destination):
        if artifact["id"] in (91, 92):
            (destination / "candidates.json").write_text(json.dumps(
                [{"id": candidate, "source": source} for candidate, source in sources.items()]))
            return
        if artifact["id"] not in outcomes:
            raise github.VerificationError("Invalid artifact archive")
        outcome, phase, reason = outcomes[artifact["id"]]
        record = {"outcome": outcome, "phase": phase, "pull-request": None, "run-ids": [],
                  "repairs-used": 0, "reason-code": reason}
        diagnostics = {"candidate-outcome": record}
        if phase == "unknown":
            diagnostics["action-outcome"] = "skipped"
        (destination / "candidate-diagnostics.json").write_text(json.dumps(diagnostics))

    monkeypatch.setattr(klaud, "github_items", items)
    monkeypatch.setattr(klaud, "download_json", download)

    recent, missing_weights = klaud.recent_candidate_outcomes("example/project", 24)
    assert missing_weights == {("dsr1", "fp4", "b200")}
    assert recent == {
        blocked: "readiness-blocked",
        unreadable: None,
        deferred: "capacity-deferred",
        superseded: "capacity-deferred",
        not_started: "agent-not-started",
        weightless: "weights-missing",
    }
    candidates = [{"id": candidate} for candidate in (deferred, blocked, untried, unreadable,
                                                      superseded, not_started)]
    assert [candidate["id"] for candidate in klaud.order_by_cooldown(candidates, recent)] == [
        untried, deferred, superseded, not_started]


def test_cooldown_ages_each_candidates_newest_artifact_including_resumed_sessions(monkeypatch):
    early, late, resumed = (f"{digit * 16}-{digit * 16}" for digit in "abc")
    artifacts = {
        # Started before the 24-hour cooldown began; a later wave resumed ``late``.
        39: [(1, early, "2026-10-14T11:00:00Z"), (30, late, "2026-10-15T11:00:00Z")],
        # The listing names ``resumed``'s first session before its second.
        41: [(20, resumed, "2026-10-15T05:00:00Z"), (21, resumed, "2026-10-15T10:00:00Z")],
    }
    outcomes = {1: "failed", 30: "readiness-blocked", 20: "failed", 21: "capacity-deferred"}
    queries = []

    def items(_repository, path, _key):
        if path.startswith("actions/workflows/klaud-plan.yml/runs?"):
            queries.append(path)
            return [{"id": 39, "created_at": "2026-10-14T06:00:00Z"},
                    {"id": 41, "created_at": "2026-10-15T00:00:00Z"},
                    {"id": 40, "created_at": "2026-10-12T23:59:59Z"}]
        run_id = int(path.split("/")[2])
        if run_id not in artifacts:
            pytest.fail(f"listed artifacts of run {run_id}")
        return [{"id": artifact, "expired": False, "created_at": created,
                 "name": f"klaud-candidate-{candidate}"}
                for artifact, candidate, created in artifacts[run_id]]

    def download(_repository, artifact, destination):
        record = {"outcome": outcomes[artifact["id"]], "phase": "targeted", "pull-request": None,
                  "run-ids": [], "repairs-used": 0}
        (destination / "candidate-diagnostics.json").write_text(
            json.dumps({"candidate-outcome": record}))

    class Clock(klaud.datetime):
        @classmethod
        def now(cls, tz=None):
            return lifecycle.utc("2026-10-15T12:00:00Z")

    monkeypatch.setattr(klaud, "datetime", Clock)
    monkeypatch.setattr(klaud, "github_items", items)
    monkeypatch.setattr(klaud, "download_json", download)

    assert klaud.recent_candidate_outcomes("example/project", 24) == (
        {late: "readiness-blocked", resumed: "capacity-deferred"}, set())
    # The cooldown plus the waves in which an older planner run may still resume sessions.
    assert queries == ["actions/workflows/klaud-plan.yml/runs?created=>=2026-10-13"]


def test_a_resumed_sessions_missing_weights_pause_its_model_from_the_planners_plan(monkeypatch):
    candidate = "1" * 16 + "-" + "2" * 16
    source = {"model": "dsr1", "precision": "fp4", "hardware": "b200"}
    # The planner run that owns the plan started before the scanned window; its resume run did not.
    runs = [{"id": 41, "created_at": "2000-01-01T00:00:00Z", "display_title": "Klaud Cold auto-sweep"},
            {"id": 4302, "created_at": "9999-01-01T00:00:00Z",
             "display_title": lifecycle.resume_title(41, candidate, 2)}]
    artifacts = {41: {"id": 91, "name": "klaud-plan"},
                 4302: {"id": 8, "name": f"klaud-candidate-{candidate}"}}

    def items(_repository, path, _key):
        if path.startswith("actions/workflows/klaud-plan.yml/runs?"):
            return runs
        return [{**artifacts[int(path.split("/")[2])], "expired": False,
                 "created_at": "9999-01-01T00:00:00Z"}]

    def download(_repository, artifact, destination):
        if artifact["id"] == 91:
            (destination / "candidates.json").write_text(
                json.dumps([{"id": candidate, "source": source}]))
            return
        record = {"outcome": "readiness-blocked", "phase": "targeted", "pull-request": None,
                  "run-ids": [], "repairs-used": 0, "reason-code": "weights-missing"}
        (destination / "candidate-diagnostics.json").write_text(
            json.dumps({"candidate-outcome": record}))

    monkeypatch.setattr(klaud, "github_items", items)
    monkeypatch.setattr(klaud, "download_json", download)

    assert klaud.recent_candidate_outcomes("example/project", 24) == (
        {candidate: "weights-missing"}, {("dsr1", "fp4", "b200")})


def test_overlap_index_drops_idle_prs_but_keeps_klaud_candidate_prs():
    now = klaud.datetime(2026, 10, 14, tzinfo=klaud.UTC)

    def pull(number, branch, updated):
        return {"number": number, "head": {"ref": branch}, "updated_at": updated}

    pulls = [
        pull(1, "feature/active", "2026-10-07T00:00:00Z"),
        pull(2, "klaud/qwen3.5-nightly", "2026-10-06T23:59:59Z"),
        pull(3, "klaud/auto-0123456789abcdef-0123456789abcdef", "2026-09-01T00:00:00Z"),
        pull(4, "klaude/auto-fedcba9876543210-fedcba9876543210", "2026-09-01T00:00:00Z"),
    ]

    assert [pr["number"] for pr in klaud.active_pulls(pulls, now, 7)] == [1, 3, 4]


def test_capacity_is_idle_headroom_on_clusters_without_waiting_human_jobs():
    now = "2026-10-15T00:05:00Z"

    def metric(name, schedulable, allocated):
        return {"clusterLabel": f"cluster:{name}", "schedulableNodes": schedulable,
                "allocatedNodes": allocated}

    def job(name, clusters, nodes, runner=None):
        return {"name": name, "clusterIds": [f"cluster:{cluster}" for cluster in clusters],
                "nodeCount": nodes, "runnerName": runner}

    metrics = [
        metric("idle", 4, 0), metric("human", 10, 2), metric("shared", 10, 2), metric("full", 6, 4),
        metric("over", 6, 5), metric("admitted", 8, 6), metric("unlabeled", 3, 1),
        metric("drained", 0, 0), metric("corrupt", 10, -1),
    ]
    queue = [
        job("sweep / p1 | klaud | dsr1 fp8", ["human"], 1),
        job("sweep / klaud | p3 | dsr1 fp8", ["shared"], 3),
        job("Klaud | p3 | dsr1 fp4", ["shared", "full", "over"], 2),
        job("sweep / p1 | dsr1 fp4", ["admitted"], 2, runner="admitted_00"),
        job("sweep / klaud | p3 | dsr1 fp4", ["admitted"], 2, runner="admitted_02"),
        job("e2e / klaud | p3 | dsr1 fp8", ["unlabeled"], None),
    ]

    def headroom(metrics=metrics, queue=queue, snapshot_at="2026-10-15T00:00:00Z",
                 generated_at=now, stalled=False):
        payload = {"kind": "inferencex.status.scheduler", "generatedAt": generated_at, "data": {
            "available": True, "snapshot": {"snapshotAt": snapshot_at, "stalled": stalled,
                                            "packingMetrics": metrics, "queue": queue},
        }}
        feed = Feed(url=api.ENDPOINTS["scheduler"], retrieved_at=now, sha256="0" * 64,
                    payload=payload)
        return api.cluster_headroom(feed, api.Policy(), api.utc(now))

    assert headroom() == {"idle": 4, "shared": 3, "admitted": 2, "unlabeled": 1}
    assert headroom(snapshot_at="2026-10-14T23:59:59Z") == {}
    assert headroom(generated_at="2026-10-15T00:02:59Z") == {}
    assert headroom(stalled=True) == {}
    assert headroom(metrics=[*metrics, metric("idle", 4, 0)]) == {}
    for malformed in ({"nodeCount": 0}, {"clusterIds": "cluster:idle"}, {"runnerName": 7}):
        assert headroom(queue=[*queue, {**queue[1], **malformed}]) == {}


@pytest.mark.parametrize(("run_status", "reconcile_status", "taken"), [
    ("completed", None, True),
    ("in_progress", "completed", True),
    ("in_progress", "in_progress", False),
    ("in_progress", None, False),
])  # fmt: skip
def test_recovery_lease_frees_once_the_holders_reconcile_job_ends(
    monkeypatch, run_status, reconcile_status, taken
):
    monkeypatch.setenv("GITHUB_RUN_ID", "200")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    monkeypatch.setattr(claims, "ref", lambda _r, _name: {"object": {"sha": "b" * 40}})
    monkeypatch.setattr(claims, "record", lambda _r, _value: {"run-id": 100, "run-attempt": 2})
    monkeypatch.setattr(github, "read", lambda _r, path: {
        "path": ".github/workflows/klaud-plan.yml", "head_branch": "main", "status": run_status,
        "conclusion": "success" if run_status == "completed" else None,
    } if path == "actions/runs/100" else pytest.fail(path))
    jobs = [{"name": "Run candidate x / Klaud Cold", "status": "completed"}] + (
        [{"name": "Reconcile interrupted candidates", "status": reconcile_status}]
        if reconcile_status else [])
    monkeypatch.setattr(github, "items", lambda _r, path, _key=None: (
        jobs if path == "actions/runs/100/attempts/2/jobs?per_page=100" else pytest.fail(path)))
    created = []
    monkeypatch.setattr(claims, "create", lambda _r, name, owner, _base, _previous: (
        created.append((name, owner)) or True))
    candidate = OwnedCandidate(id=CANDIDATE_ID, family=FAMILY, base="a" * 40)

    assert claims.recovery_lease("example/project", 42, candidate) is taken
    assert created == ([(f"klaud/recovery-42-{CANDIDATE_ID}", {"run-id": 200, "run-attempt": 1})]
                       if taken else [])


@pytest.mark.parametrize("other_recovery", ["in_progress", "completed"])
def test_sweep_frees_klaud_work_only_after_its_session_and_final_sweep_end(
    monkeypatch, other_recovery
):
    names = ["dead", "session", "sweeping", "human", "ready", "leftover",
             "closed-by-human", "stale-claim", "live-claim"]  # fmt: skip
    ids = {name: f"{digit * 16}-{digit * 16}" for name, digit in zip(names, "123456789a")}

    def pull(name, number, *, draft=True):
        return {
            "number": number, "draft": draft, "user": {"login": "Klaud-Cold"},
            "head": {"ref": f"klaud/auto-{ids[name]}", "repo": {"full_name": "example/project"}},
            "created_at": "2026-10-15T00:00:00Z", "merged_at": None,
        }  # fmt: skip

    open_pulls = [
        pull("dead", 1), pull("session", 2), pull("sweeping", 3), pull("human", 5),
        pull("ready", 6, draft=False),
    ]  # fmt: skip
    closed = {"leftover": (7, "Klaud-Cold"), "closed-by-human": (8, "maintainer")}
    smoke = {"dead": 501, "ready": 502}

    def items(_repository, path, _key=None):
        if path.startswith("actions/workflows/e2e-tests.yml/runs?status=queued"):
            return [{"id": run, "display_title": f"e2e Test - klaud-99-{ids[name]}",
                     "actor": {"login": "Klaud-Cold"}} for name, run in smoke.items()]
        if path.startswith("actions/workflows/run-sweep.yml/runs?status=in_progress"):
            return [{"head_branch": f"klaud/auto-{ids['sweeping']}"}]
        if path.startswith("actions/workflows/klaud-plan.yml/runs?status=in_progress"):
            return [{"id": 77}]
        if path.startswith("actions/runs/77/jobs"):
            return [{"name": f"Run candidate {ids['session']} / Klaud Cold"},
                    {"name": "Reconcile interrupted candidates", "status": other_recovery}]
        if path.startswith("actions/workflows/"):
            return []
        if path.startswith("pulls?state=open"):
            return open_pulls
        if path.startswith("pulls?state=all"):
            name = next(name for name, value in ids.items() if path.endswith(
                f"klaud%2Fauto-{value}&per_page=100"))
            return [pull(name, closed[name][0])]
        if path.startswith("pulls/"):
            author = "maintainer" if path.startswith("pulls/5/") else "github-actions[bot]"
            return [{"committer": {"login": author}}]
        if path == "git/matching-refs/heads/klaud/auto-":
            return [{"ref": f"refs/heads/klaud/auto-{ids[name]}", "object": {"sha": name}}
                    for name in ("dead", "session", "leftover", "closed-by-human")]
        if path == "git/matching-refs/heads/klaud/recovery-":
            return [{"ref": f"refs/heads/klaud/recovery-{parent}-{ids[name]}"}
                    for parent, name in ((1, "dead"), (1, "session"), (2, "leftover"))]
        raise AssertionError(path)

    def read(_repository, path):
        if path.startswith("actions/runs/"):
            active = path == "actions/runs/2"
            return {"status": "in_progress" if active else "completed",
                    "conclusion": None if active else "cancelled"}
        number = int(path.split("/")[1])
        return {"closed_by": {"login": next(c for n, c in closed.values() if n == number)}}

    writes, released = [], []
    monkeypatch.setenv("GITHUB_RUN_ID", "1000")
    monkeypatch.setattr(github, "items", items)
    monkeypatch.setattr(github, "read", read)
    monkeypatch.setattr(github, "write", lambda _r, path, method, _p=None: writes.append(
        (method, path)))
    monkeypatch.setattr(claims, "family_owners", lambda _r: [
        Ownership(run_id=run, candidates=[OwnedCandidate(
            id=ids[name], family=f"configs/nvidia-master.yaml:{name}", base="a" * 40)])
        for name, run in (("stale-claim", 1), ("live-claim", 2))
    ])
    monkeypatch.setattr(claims, "release_family", lambda _r, candidate, run: released.append(
        (candidate.id, run)))

    live = lifecycle.sweep_abandoned("example/project", set())

    assert live == {ids[name] for name in ("session", "sweeping", "human", "ready")}
    assert released == [(ids["stale-claim"], 1)]
    assert writes == [
        ("POST", "actions/runs/501/cancel"),
        *([("DELETE", f"git/refs/heads/klaud/recovery-1-{ids['dead']}")]
          if other_recovery == "completed" else []),
        ("POST", "issues/1/comments"),
        ("PATCH", "pulls/1"),
        ("DELETE", f"git/refs/heads/klaud/auto-{ids['dead']}"),
        ("DELETE", f"git/refs/heads/klaud/auto-{ids['leftover']}"),
    ]


UNEXPECTED_ERROR = CandidateOutcome(
    outcome="unexpected-error", phase="cleanup", pull_request=7, run_ids=[9], repairs_used=None,
)


def ended_session_sweep(monkeypatch, *, attempt=1, jobs=(), conclusion="cancelled", state="open",
                        artifacts=(), sessions=("failure",) * lifecycle.MAX_EXPIRED_SESSIONS,
                        reruns=(), planner="failure", gated=True):
    """Serve planner run 42 whose draft PR 7 has one final sweep, run 9, that ended
    ``conclusion`` (``None`` while it runs) at attempt ``attempt`` on 2026-10-15 00:00 UTC with
    ``jobs`` (a ``-timeout`` suffix marks a job GitHub stopped at its timeout).

    Session runs follow each other for 6 hours each from 2026-10-14 00:00 UTC: the planner
    run's attempts, then recovery's resume run 4200 + N for each later session N. The planner
    run's first attempt ran session 1; ``reruns`` adds the later attempts recovery once made
    instead of resume runs. A candidate job ended ``sessions[N - 1]`` or its ``reruns`` outcome
    (see ``candidate_job``). By default every session ran, so none is left to resume. Otherwise
    the planner run ended ``planner``; candidate jobs have the resume gate when ``gated``.
    Record GitHub writes, dispatched resume runs, finished outcomes, released family claims
    and posted PR comments; tests may add comments and resume runs and replace reads."""
    candidate = OwnedCandidate(id=f"{identity(FAMILY)[:16]}-{'2' * 16}", family=FAMILY,
                               base="a" * 40)
    branch = f"klaud/auto-{candidate.id}"
    start = lifecycle.utc("2026-10-14T00:00:00Z")
    attempts = 1 + len(reruns)
    running = sessions[-1] is None and len(sessions) == 1 and not reruns
    parent = {"id": 42, "head_branch": "main", "path": ".github/workflows/klaud-plan.yml",
              "event": "schedule", "status": "in_progress" if running else "completed",
              "conclusion": None if running else planner, "run_attempt": attempts,
              "created_at": stamp(start),
              "updated_at": stamp(start + timedelta(hours=6 * attempts))}
    pull = {"number": 7, "state": state, "draft": True, "merged_at": None,
            "user": {"login": "Klaud-Cold"}, "created_at": "2026-10-14T01:00:00Z",
            "labels": [{"name": "full-sweep-fail-fast"}],
            "head": {"ref": branch, "sha": "c" * 40, "repo": {"full_name": "example/project"}}}
    sweep = {"id": 9, "event": "pull_request", "head_sha": "c" * 40, "run_attempt": attempt,
             "status": "completed" if conclusion else "in_progress", "conclusion": conclusion,
             "actor": {"login": "Klaud-Cold"}, "created_at": "2026-10-14T02:00:00Z",
             "updated_at": "2026-10-15T00:00:00Z",
             "head_repository": {"full_name": "example/project"}, "pull_requests": [{"number": 7}]}

    def candidate_job(job_id, run_id, name, run_start, number, outcome):
        """The job ``name`` of attempt ``number`` of session run ``run_id``, started at
        ``run_start`` for 6 hours, which concluded ``outcome``: ``None`` while it runs,
        ``ungranted`` for a gate (or step before the agent) that failed it and ``gated-off`` for
        a gate that let it end without a session. A ``-copied`` suffix marks a copy, which keeps
        attempt 1's start."""
        created = run_start + timedelta(minutes=20)
        copied = bool(outcome) and outcome.endswith("-copied")
        kind = outcome.removesuffix("-copied") if outcome else None
        ended, gate, agent = {
            None: (None, None, None),
            "ungranted": ("failure", "failure", "skipped"),
            "gated-off": ("success", "success", "skipped"),
            "cancelled": ("cancelled", "success", "cancelled"),
        }.get(kind, (kind, "success", "success"))
        if copied:
            began = start + timedelta(minutes=21)
        else:
            began = created if kind == "skipped" else created + timedelta(minutes=1)
        steps = [] if kind == "skipped" else [
            {"name": "Set up job", "conclusion": "success"},
            *([{"name": lifecycle.GATE_STEP, "conclusion": gate}] if gated else []),
            {"name": lifecycle.AGENT_STEP, "conclusion": agent},
        ]
        return {"id": job_id, "run_id": run_id, "name": name, "run_attempt": number,
                "conclusion": ended, "runner_name": "GitHub Actions 1", "steps": steps,
                "created_at": stamp(created), "started_at": stamp(began),
                "completed_at": stamp(run_start + timedelta(hours=6)) if ended else None}

    own_name = f"Run candidate {candidate.id} / Klaud Cold"
    jobs_by_run = {42: [
        job
        for number, outcome in enumerate((sessions[0], *reruns), 1)
        for job in (
            candidate_job(1000 + number, 42, own_name, start + timedelta(hours=6 * (number - 1)),
                          number, outcome),
            candidate_job(2000 + number, 42, f"Run candidate {'3' * 16}-{'4' * 16} / Klaud Cold",
                          start + timedelta(hours=6 * (number - 1)), number,
                          "success" if number == 1 else "success-copied"),
        )
    ]}
    resume_runs = []

    def start_resume(number, outcome=None, run_id=None, actor="Klaud-Cold", hours=None):
        """Add resume run ``run_id`` (4200 + ``number`` by default), dispatched by ``actor``
        ``hours`` after 2026-10-14 00:00 UTC (the next free 6-hour slot by default), for session
        ``number``; its job ended ``outcome``."""
        run_id = run_id or 4200 + number
        slot = attempts + number - 2 if hours is None else hours / 6
        began = start + timedelta(hours=6 * slot)
        ended = {"ungranted": "failure", "gated-off": "success"}.get(outcome, outcome)
        resume_runs.append({
            "id": run_id, "display_title": lifecycle.resume_title(42, candidate.id, number),
            "head_branch": "main", "event": "workflow_dispatch", "actor": {"login": actor},
            "run_attempt": 1, "status": "in_progress" if outcome is None else "completed",
            "conclusion": ended, "created_at": stamp(began),
            "updated_at": stamp(began + timedelta(hours=6)),
        })
        jobs_by_run[run_id] = [candidate_job(run_id - 1200, run_id, own_name, began, 1, outcome)]

    for number, outcome in enumerate(sessions[1:], 2):
        start_resume(number, outcome)
    attempt_jobs = [
        {"id": 100 + index, "conclusion": job.removesuffix("-timeout"), "runner_name": "gpu-0"}
        for index, job in enumerate(jobs)
    ]
    listings = {
        f"actions/runs/9/attempts/{attempt}/jobs?per_page=100": attempt_jobs,
        **{
            f"check-runs/{100 + index}/annotations?per_page=100": [{"message": (
                "The job has exceeded the maximum execution time of 8h20m0s"
                if job.endswith("-timeout") else "The operation was canceled."
            )}]
            for index, job in enumerate(jobs)
        },
        "actions/runs/9/artifacts?per_page=100": list(artifacts),
        "pulls?state=open&per_page=100": [pull] if state == "open" else [],
        "pulls/7/commits?per_page=100": [{"committer": {"login": "github-actions[bot]"}}],
        "git/matching-refs/heads/klaud/recovery-": [
            {"ref": f"refs/heads/klaud/recovery-42-{candidate.id}"}],
        "git/matching-refs/heads/klaud/auto-": [
            {"ref": f"refs/heads/{branch}", "object": {"sha": "c" * 40}}],
        f"git/matching-refs/heads/{branch}": [
            {"ref": f"refs/heads/{branch}", "object": {"sha": "c" * 40}}],
    }
    comments = []

    def items(_repository, path, _key=None):
        if path in listings:
            return listings[path]
        if path.startswith("pulls?state=all&"):
            return [pull]
        if path.startswith("actions/workflows/run-sweep.yml/runs?per_page=100&created="):
            return [sweep]
        if path.startswith("actions/workflows/klaud-plan.yml/runs?"):
            query = dict(item.split("=", 1) for item in path.partition("?")[2].split("&"))
            return [run for run in resume_runs if query.get("status", run["status"]) == run["status"]]
        run, _, listing = path.removeprefix("actions/runs/").partition("/")
        if run.isdigit() and listing in ("jobs?filter=all&per_page=100", "jobs?per_page=100"):
            return jobs_by_run[int(run)]
        if path.startswith("issues/7/comments?"):
            return comments
        if path.startswith(("actions/workflows/", "actions/artifacts?")):
            return []
        raise AssertionError(path)

    reads = {"actions/runs/42": parent, "pulls/7": pull,
             "issues/7": {"closed_by": {"login": "Klaud-Cold"}},
             "actions/runs/42/attempts/1": {"updated_at": stamp(start + timedelta(hours=6))},
             f"actions/runs/9/attempts/{attempt}": {"run_started_at": "2026-10-14T20:00:00Z"}}
    world = SimpleNamespace(
        session=lifecycle.Session("example/project", parent, candidate, recovering=True),
        pull=pull, reads=reads, comments=comments, writes=[], dispatches=[], finished=[],
        released=[], start_resume=start_resume, sweep=sweep,
    )

    def write(_repository, path, method, payload=None):
        world.writes.append((method, path))
        if (method, path) == ("POST", "issues/7/comments"):
            comments.append({"id": 100 + len(world.writes), "user": {"login": "Klaud-Cold"},
                             "body": payload["body"]})
            return comments[-1]
        if (method, path) == DISPATCH:
            world.dispatches.append(payload)
        if method == "DELETE" and path.startswith("issues/comments/"):
            comments[:] = [comment for comment in comments
                           if path != f"issues/comments/{comment['id']}"]
        return {}

    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    monkeypatch.setattr(github, "items", items)
    monkeypatch.setattr(github, "read", lambda _r, path: (
        reads[path] if path in reads else pytest.fail(path)))
    monkeypatch.setattr(github, "write", write)
    monkeypatch.setattr(lifecycle.Session, "finish", lambda _session, outcome: (
        world.finished.append(outcome)))
    monkeypatch.setattr(claims, "family_owners", lambda _r: [
        Ownership(run_id=42, candidates=[candidate])])
    monkeypatch.setattr(claims, "recovery_lease", lambda *_: True)
    monkeypatch.setattr(claims, "release_family", lambda _r, owned, run: (
        world.released.append((owned.id, run))))
    return world


@pytest.mark.parametrize(("now", "attempt", "jobs", "state", "waiting", "rerun"), [
    pytest.param("2026-10-15T00:29:59Z", 1, ["success", "cancelled"], "open",
                 "Preempted run is waiting for its rerun", False, id="within-grace"),
    pytest.param("2026-10-15T00:30:00Z", 9, ["success", "cancelled"], "open",
                 "Reran preempted run", True, id="after-grace"),
    pytest.param("2026-10-15T23:59:59Z", 1, ["success", "cancelled"], "open",
                 "Reran preempted run", True, id="before-give-up"),
    pytest.param("2026-10-16T00:00:00Z", 1, ["success", "cancelled"], "open", None, False,
                 id="give-up"),
    pytest.param("2026-10-15T00:30:00Z", 10, ["success", "cancelled"], "open", None, False,
                 id="attempt-cap"),
    pytest.param("2026-10-15T00:29:59Z", 1, ["failure", "cancelled"], "open", None, False,
                 id="failed-job"),
    pytest.param("2026-10-15T00:30:00Z", 1, ["timed_out", "cancelled"], "open", None, False,
                 id="timed-out-job"),
    pytest.param("2026-10-15T00:30:00Z", 1, ["success", "cancelled-timeout"], "open", None,
                 False, id="job-hit-its-timeout"),
    pytest.param("2026-10-15T00:30:00Z", 1, ["success", "cancelled"], "closed", None, False,
                 id="closed-pr"),
])  # fmt: skip
def test_recovery_holds_a_preempted_final_sweep_and_reruns_it_after_the_grace_period(
    monkeypatch, now, attempt, jobs, state, waiting, rerun
):
    world = ended_session_sweep(monkeypatch, attempt=attempt, jobs=jobs, state=state)
    at = lifecycle.utc(now)

    if waiting:
        with pytest.raises(lifecycle.AwaitingRerun, match=waiting):
            lifecycle.reconcile(world.session, at)
    else:
        assert lifecycle.reconcile(world.session, at) is True
    assert world.writes == ([("POST", "actions/runs/9/rerun-failed-jobs")] if rerun else [])
    assert world.finished == ([] if waiting else [UNEXPECTED_ERROR])


@pytest.mark.parametrize(("broken", "waiting"), [
    ("items", "Preemption check unavailable; retrying next pass"),
    ("write", "Rerun request failed; retrying next pass"),
])  # fmt: skip
def test_recovery_retries_a_preempted_final_sweep_after_github_errors(
    monkeypatch, broken, waiting
):
    world = ended_session_sweep(monkeypatch, jobs=["success", "cancelled"])
    working = getattr(github, broken)

    def flaky(repository, path, *args):
        if path.startswith(("actions/runs/9/attempts/", "actions/runs/9/rerun-failed-jobs")):
            raise subprocess.CalledProcessError(1, ["gh", "api"])
        return working(repository, path, *args)

    monkeypatch.setattr(github, broken, flaky)

    with pytest.raises(lifecycle.AwaitingRerun, match=waiting):
        lifecycle.reconcile(world.session, lifecycle.utc("2026-10-15T01:00:00Z"))
    assert world.finished == []


def recovery_pass(monkeypatch, tmp_path, max_sessions=5, now="2026-10-15T01:00:00Z"):
    """Run planner run 1000's recovery at ``now``; return its step summary and output files."""
    summary, output = tmp_path / "summary.md", tmp_path / "output"
    for name, value in (("GITHUB_REPOSITORY", "example/project"), ("GITHUB_RUN_ID", "1000"),
                        ("RUNNER_TEMP", str(tmp_path)), ("GITHUB_STEP_SUMMARY", str(summary)),
                        ("GITHUB_OUTPUT", str(output))):
        monkeypatch.setenv(name, value)

    class Clock(lifecycle.datetime):
        @classmethod
        def now(cls, tz=None):
            return lifecycle.utc(now)

    monkeypatch.setattr(lifecycle, "datetime", Clock)
    lifecycle.recover(max_sessions)
    return summary, output


def test_recovery_keeps_the_pr_claim_and_lease_of_a_rerun_final_sweep(monkeypatch, tmp_path):
    world = ended_session_sweep(monkeypatch, jobs=["success", "cancelled"])

    summary, output = recovery_pass(monkeypatch, tmp_path)

    assert world.writes == [("POST", "actions/runs/9/rerun-failed-jobs")]
    assert world.finished == world.released == []
    assert json.loads((tmp_path / "klaud-recovery.json").read_text()) == [FAMILY]
    assert output.read_text() == "active-sessions=0\n"
    assert summary.read_text() == (
        f"Klaud `{world.session.candidate.id}`: pending / 等待后续收尾 (Reran preempted run). "
        "[Parent run / 父运行](https://github.com/example/project/actions/runs/42)\n\n")


@pytest.mark.parametrize("stale", ["deleted-claim-parent", "corrupt-artifact"])
def test_recovery_skips_an_unreadable_owner_and_reconciles_the_others(
    monkeypatch, tmp_path, capsys, stale
):
    world = ended_session_sweep(monkeypatch, jobs=["success", "cancelled"])
    family = "configs/amd-master.yaml:other"
    other = OwnedCandidate(id=f"{identity(family)[:16]}-{'6' * 16}", family=family, base="a" * 40)
    owners = [Ownership(run_id=42, candidates=[world.session.candidate])]
    if stale == "deleted-claim-parent":
        owners.append(Ownership(run_id=77, candidates=[other]))
    monkeypatch.setattr(claims, "family_owners", lambda _r: owners)
    serve, read = github.items, github.read

    def items(repository, path, key=None):
        if path.startswith("actions/artifacts?") and stale == "corrupt-artifact":
            return [{"id": 5, "expired": False, "size_in_bytes": 10, "workflow_run": {"id": 42}}]
        return serve(repository, path, key)

    def gone(repository, path):
        if path == "actions/runs/77":
            raise subprocess.CalledProcessError(1, ["gh", "api"])
        return read(repository, path)

    def corrupt(*_args):
        raise github.VerificationError("Invalid artifact archive")

    monkeypatch.setattr(github, "items", items)
    monkeypatch.setattr(github, "read", gone)
    monkeypatch.setattr(github, "download_json", corrupt)

    summary, _ = recovery_pass(monkeypatch, tmp_path)

    assert world.writes == [("POST", "actions/runs/9/rerun-failed-jobs")]
    rerun = (f"Klaud `{world.session.candidate.id}`: pending / 等待后续收尾 (Reran preempted "
             "run). [Parent run / 父运行](https://github.com/example/project/actions/runs/42)"
             "\n\n")
    if stale == "deleted-claim-parent":
        assert json.loads((tmp_path / "klaud-recovery.json").read_text()) == [
            "configs/amd-master.yaml:other", FAMILY]
        assert summary.read_text() == (
            f"Klaud `{other.id}`: needs inspection / 需要检查 (state unavailable or invalid). "
            "[Parent run / 父运行](https://github.com/example/project/actions/runs/77)\n\n" + rerun)
    else:
        assert json.loads((tmp_path / "klaud-recovery.json").read_text()) == [FAMILY]
        assert summary.read_text() == rerun
        assert "::warning::Klaud: ownership artifact 5 is unreadable; skipped.\n" in (
            capsys.readouterr().out)


def test_recovery_leaves_the_draft_of_a_planner_run_it_could_not_read(monkeypatch, tmp_path):
    world = ended_session_sweep(monkeypatch, sessions=("failure",))
    read, failed = github.read, []

    def flaky(repository, path):
        if path == "actions/runs/42" and not failed:
            failed.append(path)
            raise subprocess.CalledProcessError(1, ["gh", "api"])
        return read(repository, path)

    monkeypatch.setattr(github, "read", flaky)

    recovery_pass(monkeypatch, tmp_path)

    assert world.writes == world.released == []
    assert json.loads((tmp_path / "klaud-recovery.json").read_text()) == [FAMILY]


def test_abandoned_work_sweep_keeps_a_claim_whose_planner_run_is_gone(monkeypatch):
    world = ended_session_sweep(monkeypatch, sessions=("failure",))
    candidate = world.session.candidate
    family = "configs/amd-master.yaml:other"
    other = OwnedCandidate(id=f"{identity(family)[:16]}-{'6' * 16}", family=family, base="a" * 40)
    monkeypatch.setattr(claims, "family_owners", lambda _r: [
        Ownership(run_id=42, candidates=[candidate]), Ownership(run_id=77, candidates=[other])])
    read, serve = github.read, github.items
    draft = {**world.pull, "number": 8,
             "head": {**world.pull["head"], "ref": f"klaud/auto-{other.id}"}}

    def gone(repository, path):
        if path == "actions/runs/77":
            raise subprocess.CalledProcessError(1, ["gh", "api"])
        return read(repository, path)

    def items(repository, path, key=None):
        if path == "pulls?state=open&per_page=100":
            return [world.pull, draft]
        if path == "pulls/8/commits?per_page=100":
            return [{"committer": {"login": "github-actions[bot]"}}]
        return serve(repository, path, key)

    monkeypatch.setattr(github, "read", gone)
    monkeypatch.setattr(github, "items", items)

    assert lifecycle.sweep_abandoned("example/project", set()) == {other.id}
    assert world.released == [(candidate.id, 42)]
    assert world.writes == [
        ("DELETE", f"git/refs/heads/klaud/recovery-42-{candidate.id}"),
        ("POST", "issues/7/comments"), ("PATCH", "pulls/7"),
        ("DELETE", f"git/refs/heads/klaud/auto-{candidate.id}"),
    ]


def test_recovery_warns_why_a_passed_final_sweep_failed_validation(monkeypatch, capsys):
    world = ended_session_sweep(monkeypatch, attempt=2, conclusion="success", artifacts=[{
        "id": 1, "name": "klaud-sweep-manifest", "expired": False,
        "created_at": "2026-10-14T03:00:00Z", "workflow_run": {"id": 9, "head_sha": "c" * 40},
    }])

    assert lifecycle.reconcile(world.session, lifecycle.utc("2026-10-15T01:00:00Z")) is True
    assert capsys.readouterr().out == (
        f"::warning::Klaud: final sweep of candidate {world.session.candidate.id} failed "
        "validation: Manifest belongs to an earlier run attempt.\n")
    assert world.finished == [UNEXPECTED_ERROR]
    assert world.writes == []


DISPATCH = ("POST", "actions/workflows/klaud-plan.yml/dispatches")
GRANT = ("POST", "issues/7/comments")
NOW = lifecycle.utc("2026-10-15T01:00:00Z")


def recorded(session, marker="klaud-outcome:"):
    """A Klaud comment recording ``UNEXPECTED_ERROR`` under ``session``'s ``marker``."""
    record = {"head": "c" * 40, "outcome": UNEXPECTED_ERROR.model_dump(by_alias=True)}
    return {"id": 1, "user": {"login": "Klaud-Cold"}, "body": (
        session.marker.replace("klaud-outcome:", marker) + json.dumps(record) + "\n-->\nDone.")}


def grant_comment(world, number, author="Klaud-Cold"):
    """A PR comment granting the world's candidate session ``number`` of planner run 42."""
    return {"id": 50 + number, "user": {"login": author}, "body": (
        lifecycle.resume_marker(42, world.session.candidate.id, number) + "\nResumed.")}


def resume_check(tmp_path, monkeypatch, world, number, attempt=1, run=None, session=None):
    """Run the resume gate of the world's session ``number`` in ``run`` (planner run 42 for
    session 1, else resume run 4200 + ``number`` by default) at that run's ``attempt``, with
    ``session`` (``number`` by default) as the run's session input; return its exit code and
    outputs."""
    (tmp_path / "candidate.json").write_text(world.session.candidate.model_dump_json())
    output = tmp_path / "gate-output"
    run = run or (42 if number == 1 else 4200 + number)
    for name, value in (("GITHUB_REPOSITORY", "example/project"), ("GITHUB_RUN_ID", str(run)),
                        ("GITHUB_RUN_ATTEMPT", str(attempt)), ("KLAUD_PARENT_RUN", "42"),
                        ("KLAUD_SESSION", str(number) if session is None else session),
                        ("KLAUD_EVIDENCE", str(tmp_path)), ("GITHUB_OUTPUT", str(output))):
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(sys, "argv", ["klaud", "resume-check"])
    output.write_text("")
    code = klaud.main()
    return code, output.read_text()


def park_comment(world, number, runs=(9,)):
    """A Klaud comment recording that the world's session ``number`` parked on ``runs``."""
    return {"id": 70 + number, "user": {"login": "Klaud-Cold"}, "body": (
        lifecycle.park_marker(42, world.session.candidate.id, number)
        + json.dumps({"runs": list(runs)}) + "\n-->\nParked.")}


EXPIRED = lifecycle.MAX_EXPIRED_SESSIONS


@pytest.mark.parametrize("conclusion", [None, "failure"], ids=["running", "finished"])
@pytest.mark.parametrize(("sessions", "options", "started", "approved"), [
    pytest.param(("failure",), {}, 1, True, id="first-session"),
    pytest.param(("failure",) * (EXPIRED - 1), {}, EXPIRED - 1, True, id="below-the-cap"),
    pytest.param(("failure",) * EXPIRED, {}, EXPIRED, False, id="at-the-cap"),
    pytest.param(("failure", "gated-off", "failure"), {}, 2, True,
                 id="resume-run-without-its-grant-does-not-count"),
    pytest.param(("failure", "gated-off"), {}, 1, True, id="latest-started-session-decides"),
    pytest.param(("failure", "success"), {}, 2, False, id="latest-session-finished"),
    pytest.param(("failure",), {"planner": "cancelled"}, 1, True,
                 id="failed-job-in-a-cancelled-run"),
    pytest.param(("cancelled",), {"planner": "cancelled"}, 1, True,
                 id="cancelled-session-is-preempted"),
    pytest.param(("failure", "cancelled"), {}, 2, True, id="cancelled-resume-run-is-preempted"),
    pytest.param(("failure",), {"reruns": ("failure-copied", "failure", "failure")}, 3, True,
                 id="copied-planner-jobs-do-not-count"),
    pytest.param(("failure",), {"reruns": ("ungranted",) * 3}, 1, True,
                 id="ungranted-planner-reruns-do-not-count"),
    # Rerunning a planner run's failed jobs skipped its candidate job instead of resuming it.
    pytest.param(("failure",), {"reruns": ("skipped",)}, 1, True,
                 id="skipped-planner-rerun-still-resumes"),
    pytest.param(("failure",), {"gated": False}, 1, False, id="planner-run-predates-resumption"),
    pytest.param(("failure",), {"gated": False, "reruns": ("skipped",)}, 1, False,
                 id="rerun-of-a-planner-run-from-before-resumption"),
    pytest.param(("success",), {"parked": {1: (9,)}}, 1, True, id="parked-session"),
    pytest.param(("success",) * (EXPIRED + 1), {"parked": dict.fromkeys(range(1, EXPIRED + 2),
                 (9,))}, EXPIRED + 1, True, id="parked-sessions-never-expire"),
    pytest.param(("success",) * lifecycle.MAX_SESSIONS,
                 {"parked": dict.fromkeys(range(1, lifecycle.MAX_SESSIONS + 1), (9,))},
                 lifecycle.MAX_SESSIONS, False, id="at-the-session-cap"),
    pytest.param(("failure",) * (EXPIRED - 1) + ("success",), {"parked": {EXPIRED: ()}},
                 EXPIRED, False, id="parking-at-the-deadline-expires"),
    pytest.param(("failure",) * EXPIRED, {"parked": dict.fromkeys(range(1, EXPIRED + 1), (9,))},
                 EXPIRED, False, id="sessions-that-ran-out-after-parking-expire"),
])  # fmt: skip
def test_recovery_approves_resuming_only_sessions_a_resume_run_can_continue(
    monkeypatch, conclusion, sessions, options, started, approved
):
    harness = {key: value for key, value in options.items() if key != "parked"}
    world = ended_session_sweep(monkeypatch, conclusion=conclusion, sessions=sessions, **harness)
    for number, runs in options.get("parked", {}).items():
        world.comments.append(park_comment(world, number, runs))

    assert lifecycle.sessions_started(world.session) == started
    if conclusion is None:
        with pytest.raises(lifecycle.PendingCleanup, match="Owned work is still running"):
            lifecycle.reconcile(world.session, NOW)
    elif approved:
        with pytest.raises(lifecycle.ResumeSession, match="approved to resume") as raised:
            lifecycle.reconcile(world.session, NOW)
        assert raised.value.pull is world.pull
    else:
        assert lifecycle.reconcile(world.session, NOW) is True
    assert world.writes == world.released == []
    assert world.finished == ([UNEXPECTED_ERROR] if conclusion and not approved else [])


@pytest.mark.parametrize(("sessions", "updated", "now", "approved"), [
    pytest.param(("failure",), "2026-10-14T03:00:00Z", "2026-10-15T05:59:59Z", True,
                 id="within-a-day"),
    pytest.param(("failure",), "2026-10-14T03:00:00Z", "2026-10-15T06:00:00Z", False,
                 id="a-day-later"),
    pytest.param(("failure", "failure"), "2026-10-14T03:00:00Z", "2026-10-15T11:59:59Z", True,
                 id="within-a-day-of-the-resume-run"),
    pytest.param(("failure", "failure"), "2026-10-14T03:00:00Z", "2026-10-15T12:00:00Z", False,
                 id="a-day-after-the-resume-run"),
    pytest.param(("failure", "ungranted"), "2026-10-14T03:00:00Z", "2026-10-15T05:59:59Z", True,
                 id="within-a-day-of-the-last-started-session"),
    pytest.param(("failure", "ungranted"), "2026-10-14T03:00:00Z", "2026-10-15T06:00:00Z", False,
                 id="a-run-that-never-started-its-agent-does-not-extend-it"),
    pytest.param(("failure",), "2026-10-15T00:00:00Z", "2026-10-15T23:59:59Z", True,
                 id="within-a-day-of-the-last-run-ending"),
    pytest.param(("failure",), "2026-10-15T00:00:00Z", "2026-10-16T00:00:00Z", False,
                 id="a-day-after-the-last-run-ended"),
])  # fmt: skip
def test_recovery_stops_resuming_a_day_after_the_last_session_or_run_ended(
    monkeypatch, sessions, updated, now, approved
):
    world = ended_session_sweep(monkeypatch, conclusion="failure", sessions=sessions)
    world.sweep["updated_at"] = updated

    if approved:
        with pytest.raises(lifecycle.ResumeSession):
            lifecycle.reconcile(world.session, lifecycle.utc(now))
    else:
        assert lifecycle.reconcile(world.session, lifecycle.utc(now)) is True
    assert world.finished == ([] if approved else [UNEXPECTED_ERROR])


@pytest.mark.parametrize(("conclusion", "kept"), [(None, True), ("failure", False)],
                         ids=["run-live", "runs-ended"])
def test_recovery_keeps_a_draft_while_its_runs_are_live_however_long_they_queue(
    monkeypatch, conclusion, kept
):
    world = ended_session_sweep(monkeypatch, conclusion=conclusion, sessions=("failure",))

    assert lifecycle.retained(world.session, lifecycle.utc("2026-10-20T00:00:00Z")) is kept


@pytest.mark.parametrize(("sweep", "targeted", "rerun"), [
    pytest.param("failure", "cancelled", 11, id="preempted-targeted-run-after-a-failed-sweep"),
    pytest.param("cancelled", "success", 9, id="preempted-sweep-before-a-passed-targeted-run"),
])  # fmt: skip
def test_recovery_reruns_the_preempted_final_sweep_or_a_newer_targeted_run(
    monkeypatch, sweep, targeted, rerun
):
    world = ended_session_sweep(monkeypatch, conclusion=sweep, jobs=["success", "cancelled"])
    run = {
        "id": 11, "event": "workflow_dispatch", "head_sha": "a" * 40, "run_attempt": 1,
        "display_title": f"e2e Test - klaud-42-{world.session.candidate.id}",
        "actor": {"login": "Klaud-Cold"}, "head_repository": {"full_name": "example/project"},
        "status": "completed", "conclusion": targeted,
        "created_at": "2026-10-14T04:00:00Z", "updated_at": "2026-10-15T00:00:00Z",
    }
    serve = github.items

    def items(repository, path, key=None):
        if path.startswith("actions/workflows/e2e-tests.yml/runs?"):
            return [run]
        if path == "actions/runs/11/attempts/1/jobs?per_page=100":
            return [{"id": 300, "conclusion": "cancelled", "runner_name": "gpu-0"}]
        if path == "check-runs/300/annotations?per_page=100":
            return [{"message": "The operation was canceled."}]
        return serve(repository, path, key)

    monkeypatch.setattr(github, "items", items)

    with pytest.raises(lifecycle.AwaitingRerun, match="Reran preempted run"):
        lifecycle.reconcile(world.session, NOW)
    assert world.writes == [("POST", f"actions/runs/{rerun}/rerun-failed-jobs")]


@pytest.mark.parametrize(("session", "resumed"), [("failure", True), (None, False)],
                         ids=["own-job-ended", "own-job-running"])
def test_recovery_resumes_a_candidate_whose_job_ended_while_its_planner_run_continues(
    monkeypatch, tmp_path, session, resumed
):
    world = ended_session_sweep(monkeypatch, conclusion="failure", sessions=(session,))
    world.reads["actions/runs/42"] = {
        **world.session.parent, "status": "in_progress", "conclusion": None}

    _, output = recovery_pass(monkeypatch, tmp_path)

    assert world.writes == ([GRANT, DISPATCH] if resumed else [])
    assert output.read_text() == f"active-sessions={int(resumed)}\n"


def report_comment(world, run):
    """A Klaud attempt report on the world's PR for attempt 1 of owned run ``run``."""
    return {"id": 90 + run, "user": {"login": "Klaud-Cold"},
            "body": reporting.marker(world.session, f"run-{run}-1") + "{}\n-->\nQueued."}


@pytest.mark.parametrize(("conclusion", "reported", "overdue", "error"), [
    pytest.param(None, True, False, None, id="reported-live-run"),
    pytest.param(None, False, False, "Publish an attempt report", id="unreported-live-run"),
    pytest.param("failure", True, False, "No owned run is queued or running",
                 id="nothing-to-wait-on"),
    pytest.param("failure", False, True, None, id="at-the-deadline"),
])  # fmt: skip
def test_a_session_parks_only_on_reported_live_runs_or_at_its_deadline(
    monkeypatch, conclusion, reported, overdue, error
):
    world = ended_session_sweep(monkeypatch, conclusion=conclusion, sessions=(None,))
    session = lifecycle.Session("example/project", world.session.parent, world.session.candidate)
    if reported:
        world.comments.append(report_comment(world, 9))

    if error:
        with pytest.raises(github.VerificationError, match=error):
            lifecycle.park(session, 1, overdue=overdue)
        assert world.writes == []
    else:
        waiting = [] if conclusion else [9]
        assert lifecycle.park(session, 1, overdue=overdue) == waiting
        assert lifecycle.parked(session, world.pull, 1) == waiting
        lifecycle.park(session, 1, overdue=overdue)
        assert [method for method, _ in world.writes] == ["POST", "PATCH"]


@pytest.mark.parametrize(("parked", "conclusion", "blocked"), [
    pytest.param((9,), None, False, id="parked-on-a-live-run"),
    pytest.param((9,), "success", True, id="parked-runs-ended"),
    pytest.param((), "success", False, id="parked-at-the-deadline"),
    pytest.param(None, None, True, id="never-parked"),
])  # fmt: skip
def test_a_parked_session_may_stop_only_while_a_run_it_waits_on_is_live(
    tmp_path, monkeypatch, parked, conclusion, blocked
):
    world = ended_session_sweep(monkeypatch, conclusion=conclusion, sessions=(None,))
    if parked is not None:
        world.comments.append(park_comment(world, 1, parked))
    (tmp_path / "session-end.json").write_text('{"status": "parked"}\n')
    monkeypatch.setattr(lifecycle, "current_session", lambda: world.session)
    monkeypatch.setenv("KLAUD_EVIDENCE", str(tmp_path))
    monkeypatch.setenv("KLAUD_SESSION", "1")

    assert ("decision" in klaud.check_stop()) is blocked


@pytest.mark.parametrize(("marker", "end", "action", "verified"), [
    pytest.param(True, "parked", "success", True, id="parked"),
    pytest.param(True, "parked", "failure", False, id="kept-working-after-parking-then-ran-out"),
    pytest.param(False, None, "failure", False, id="ran-out"),
])  # fmt: skip
def test_diagnostics_verify_only_a_session_that_ended_parked(
    tmp_path, monkeypatch, marker, end, action, verified
):
    world = ended_session_sweep(monkeypatch, conclusion=None, sessions=(None,))
    if marker:
        world.comments.append(park_comment(world, 1))
    if end:
        (tmp_path / "session-end.json").write_text(json.dumps({"status": end}))
    monkeypatch.setattr(lifecycle, "current_session", lambda: world.session)
    monkeypatch.setenv("KLAUD_SESSION", "1")
    monkeypatch.setenv("KLAUD_EVIDENCE", str(tmp_path))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    output = tmp_path / "diagnostics.json"

    assert klaud.save_diagnostics(
        tmp_path / "execution.json", tmp_path / "structured.json", action, output
    ) is verified
    diagnostics = json.loads(output.read_text())
    if verified:
        assert diagnostics["session-state"] == "parked"
        assert diagnostics["candidate-outcome"] is None
    else:
        assert diagnostics["candidate-outcome"]["outcome"] == "unexpected-error"


@pytest.mark.parametrize(("elapsed", "warned"), [
    (lifecycle.PARK_AFTER_SECONDS - 1, False), (lifecycle.PARK_AFTER_SECONDS, True),
])  # fmt: skip
def test_the_deadline_hook_tells_a_session_to_park_near_its_time_limit(
    monkeypatch, elapsed, warned
):
    now = lifecycle.utc("2026-10-15T05:00:00Z")

    class Clock(klaud.datetime):
        @classmethod
        def now(cls, tz=None):
            return now

    monkeypatch.setattr(klaud, "datetime", Clock)
    monkeypatch.setenv("KLAUD_SESSION_STARTED", str(int(now.timestamp()) - elapsed))

    hook = klaud.check_deadline()
    assert bool(hook) is warned
    if warned:
        assert "About 30 minutes remain" in hook["hookSpecificOutput"]["additionalContext"]


@pytest.mark.parametrize("change", [
    "aborted", "closed", "outcome-recorded", "cleanup-pending",
])  # fmt: skip
def test_recovery_never_resumes_an_aborted_closed_or_recorded_pr(monkeypatch, change):
    world = ended_session_sweep(monkeypatch, state="closed" if change == "closed" else "open",
                                sessions=("failure",))
    if change == "aborted":
        world.pull["labels"].append({"name": lifecycle.ABORT_LABEL})
    elif change != "closed":
        world.comments.append(recorded(
            world.session, "klaud-cleanup:" if change == "cleanup-pending" else "klaud-outcome:"))

    lifecycle.resume(world.session, world.pull, NOW)

    assert world.writes == []


@pytest.mark.parametrize("conclusion", [None, "failure"], ids=["running", "finished"])
def test_klaud_abort_closes_the_candidate_pr_as_aborted(monkeypatch, conclusion):
    world = ended_session_sweep(monkeypatch, conclusion=conclusion, sessions=("failure",))
    world.pull["labels"].append({"name": lifecycle.ABORT_LABEL})

    assert lifecycle.reconcile(world.session, NOW) is True
    assert [outcome.outcome for outcome in world.finished] == ["aborted"]
    assert world.writes == []


def test_an_aborted_session_may_stop_without_its_own_outcome(tmp_path, monkeypatch):
    world = ended_session_sweep(monkeypatch, conclusion=None, sessions=(None,))
    world.pull["labels"].append({"name": lifecycle.ABORT_LABEL})
    monkeypatch.setattr(lifecycle, "current_session", lambda: world.session)
    monkeypatch.setenv("KLAUD_EVIDENCE", str(tmp_path))

    assert klaud.check_stop() == {}
    world.pull["labels"].pop()
    assert klaud.check_stop()["decision"] == "block"


def test_a_resumed_session_never_adopts_a_newer_runs_pr_on_its_branch(tmp_path, monkeypatch):
    world = ended_session_sweep(monkeypatch, conclusion="failure", sessions=("failure", "failure"))
    # A newer planner run opened this PR on the candidate branch after planner run 42 ended.
    world.pull["created_at"] = "2026-10-14T07:00:00Z"

    assert lifecycle.reconcile(world.session, NOW) is False
    assert world.writes == world.finished == world.released == []
    world.start_resume(3)
    world.comments.append(grant_comment(world, 3))
    assert resume_check(tmp_path, monkeypatch, world, 3) == (0, "run=false\nresume=false\n")


def test_recovery_dispatches_a_resume_run_that_continues_an_unfinished_session(
    monkeypatch, tmp_path
):
    world = ended_session_sweep(monkeypatch, conclusion="failure", sessions=("failure",))
    candidate = world.session.candidate.id
    link = "https://github.com/example/project/actions/runs/42"

    summary, output = recovery_pass(monkeypatch, tmp_path)

    assert world.writes == [GRANT, DISPATCH]
    assert world.dispatches == [{"ref": "main", "inputs": {
        "resume-parent": "42", "resume-candidate": candidate, "resume-session": "2"}}]
    assert [comment["body"] for comment in world.comments] == [
        f"<!-- klaud-resume:42:{candidate}:session-2 -->\n" + reporting.translated(
            f"**resumed** · The session ended before finishing; session 2 continues it in a "
            f"new run of [its planner]({link}).",
            f"**resumed** · 会话未完成即已结束；第 2 个会话将在[其规划运行]({link})的新运行中继续。",
        )
    ]
    assert world.finished == world.released == []
    assert json.loads((tmp_path / "klaud-recovery.json").read_text()) == [FAMILY]
    assert output.read_text() == "active-sessions=1\n"
    assert summary.read_text() == (
        f"Klaud `{candidate}`: resumed / 已恢复 (session 2). [Parent run / 父运行]({link})\n\n")

    # The dispatched run starts session 2 once; a rerun of it starts nothing.
    world.start_resume(2)
    assert resume_check(tmp_path, monkeypatch, world, 2) == (0, "run=true\nresume=true\n")
    assert resume_check(tmp_path, monkeypatch, world, 2, attempt=2) == (
        0, "run=false\nresume=false\n")

    # While it runs, the next pass keeps the candidate without dispatching another session.
    writes = len(world.writes)
    _, output = recovery_pass(monkeypatch, tmp_path)
    assert world.writes[writes:] == []
    assert output.read_text() == "active-sessions=1\nactive-sessions=1\n"
    assert json.loads((tmp_path / "klaud-recovery.json").read_text()) == [FAMILY]


def test_recovery_keeps_an_approved_session_without_a_resume_slot_for_the_next_wave(
    monkeypatch, tmp_path
):
    world = ended_session_sweep(monkeypatch, conclusion="failure", sessions=("failure",))

    summary, output = recovery_pass(monkeypatch, tmp_path, max_sessions=0)

    assert world.writes == world.finished == world.released == []
    assert json.loads((tmp_path / "klaud-recovery.json").read_text()) == [FAMILY]
    assert output.read_text() == "active-sessions=0\n"
    assert summary.read_text() == (
        f"Klaud `{world.session.candidate.id}`: pending / 等待后续收尾 (resume deferred: every "
        "session slot is in use). [Parent run / 父运行](https://github.com/example/project/actions/runs/42)"
        "\n\n")


def test_recovery_keeps_a_session_whose_resume_dispatch_failed_without_leaving_its_grant(
    monkeypatch, tmp_path, capsys
):
    world = ended_session_sweep(monkeypatch, conclusion="failure", sessions=("failure",))
    candidate = world.session.candidate.id
    recording = github.write

    def flaky(repository, path, method, payload=None):
        if (method, path) == DISPATCH:
            raise subprocess.CalledProcessError(1, ["gh", "api"])
        return recording(repository, path, method, payload)

    monkeypatch.setattr(github, "write", flaky)
    summary, output = recovery_pass(monkeypatch, tmp_path)

    assert world.writes == [GRANT, ("DELETE", "issues/comments/101")]
    assert world.comments == []
    assert world.finished == world.released == []
    assert json.loads((tmp_path / "klaud-recovery.json").read_text()) == [FAMILY]
    assert capsys.readouterr().out == (
        f"::warning::Klaud: resuming candidate {candidate} of planner run 42 failed: state "
        "unavailable or invalid.\n")
    assert summary.read_text() == (
        f"Klaud `{candidate}`: pending / 等待后续收尾 (resume failed, retrying next pass: state "
        "unavailable or invalid). [Parent run / 父运行]"
        "(https://github.com/example/project/actions/runs/42)\n\n")

    monkeypatch.setattr(github, "write", recording)
    recovery_pass(monkeypatch, tmp_path)

    assert world.writes[2:] == [GRANT, DISPATCH]
    assert [comment["body"].split("\n", 1)[0] for comment in world.comments] == [
        lifecycle.resume_marker(42, candidate, 2)]
    assert output.read_text() == "active-sessions=0\nactive-sessions=1\n"


@pytest.mark.parametrize("change", ["resume-run", "planner-rerun"])
def test_recovery_dispatches_nothing_for_a_session_that_resumed_meanwhile(
    monkeypatch, tmp_path, capsys, change
):
    world = ended_session_sweep(monkeypatch, conclusion="failure", sessions=("failure",))
    reconcile = lifecycle.reconcile

    def resumed_meanwhile(session, now):
        try:
            return reconcile(session, now)
        finally:
            if change == "resume-run":
                world.start_resume(2)
            else:
                world.reads["actions/runs/42"] = {
                    **world.session.parent, "status": "queued", "conclusion": None,
                    "run_attempt": 2,
                }

    monkeypatch.setattr(lifecycle, "reconcile", resumed_meanwhile)
    _, output = recovery_pass(monkeypatch, tmp_path)

    assert world.writes == world.finished == world.released == []
    assert output.read_text() == f"active-sessions={int(change == 'resume-run')}\n"
    assert capsys.readouterr().out == (
        f"::warning::Klaud: resuming candidate {world.session.candidate.id} of planner run 42 "
        "failed: Session resumed; leave the active session intact.\n")


@pytest.mark.parametrize(("now", "state", "kept"), [
    pytest.param("2026-10-15T23:59:59Z", "open", True, id="open-draft-within-a-day"),
    pytest.param("2026-10-16T00:00:00Z", "open", False, id="open-draft-a-day-later"),
    pytest.param("2026-10-15T23:59:59Z", "closed", False, id="closed-pr"),
])  # fmt: skip
def test_recovery_keeps_an_open_draft_it_could_not_reconcile_for_a_day(
    monkeypatch, tmp_path, now, state, kept
):
    world = ended_session_sweep(monkeypatch, state=state)
    serve = github.items

    def unavailable(repository, path, key=None):
        if path.startswith("actions/workflows/run-sweep.yml/runs?per_page=100&created="):
            raise subprocess.CalledProcessError(1, ["gh", "api"])
        return serve(repository, path, key)

    monkeypatch.setattr(github, "items", unavailable)
    recovery_pass(monkeypatch, tmp_path, now=now)

    candidate = world.session.candidate.id
    lease = ("DELETE", f"git/refs/heads/klaud/recovery-42-{candidate}")
    branch = ("DELETE", f"git/refs/heads/klaud/auto-{candidate}")
    closed = [("POST", "issues/7/comments"), ("PATCH", "pulls/7")] if state == "open" else []
    assert world.writes == ([] if kept else [lease, *closed, branch])
    assert world.released == ([] if kept else [(candidate, 42)])


@pytest.mark.parametrize(("sessions", "kept"), [
    pytest.param((None,), True, id="below-the-cap"),
    pytest.param(("failure",) * 3 + (None,), False, id="last-session"),
])  # fmt: skip
def test_an_unfinished_session_keeps_its_pr_and_claim_for_a_later_resume(
    monkeypatch, sessions, kept
):
    world = ended_session_sweep(monkeypatch, conclusion="failure", sessions=sessions)
    own = lifecycle.Session("example/project", world.session.parent, world.session.candidate)
    monkeypatch.setattr(lifecycle, "current_session", lambda: own)
    monkeypatch.setattr(sys, "argv", ["klaud", "recover-current"])

    if kept:
        with pytest.raises(lifecycle.PendingCleanup, match="Unfinished session kept for a later"):
            lifecycle.reconcile(own, NOW)
    assert klaud.main() == 0
    assert world.finished == ([] if kept else [UNEXPECTED_ERROR])
    assert world.released == ([] if kept else [(own.candidate.id, 42)])
    assert world.writes == []


@pytest.mark.parametrize(("attempt", "runs"), [(1, True), (2, False)])
def test_a_planner_job_runs_only_its_first_attempts_session_without_reading_github(
    tmp_path, monkeypatch, capsys, attempt, runs
):
    output = tmp_path / "output"
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", str(attempt))
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    for name in ("read", "items", "write"):
        monkeypatch.setattr(github, name, lambda *_args: pytest.fail("GitHub API call"))
    monkeypatch.setattr(sys, "argv", ["klaud", "resume-check"])

    assert klaud.main() == 0
    assert output.read_text() == capsys.readouterr().out == (
        f"run={str(runs).lower()}\nresume=false\n")


@pytest.mark.parametrize(("sessions", "change", "resumes"), [
    pytest.param(("failure", None), None, True, id="granted"),
    pytest.param(("failure",) * (lifecycle.MAX_SESSIONS - 1) + (None,), None, True,
                 id="last-session"),
    pytest.param(("failure", "gated-off", None), None, True,
                 id="ungranted-resume-run-not-counted"),
    pytest.param(("failure",) * lifecycle.MAX_SESSIONS + (None,), None, False, id="over-the-cap"),
    pytest.param(("failure", None), "outcome-recorded", False, id="outcome-recorded"),
    pytest.param(("failure", None), "aborted", False, id="aborted-despite-a-grant"),
    pytest.param(("failure", None), "ungranted", False, id="no-grant"),
    pytest.param(("failure", None), "other-session", False, id="grant-for-another-session"),
    pytest.param(("failure", None), "forged", False, id="grant-from-someone-else"),
])  # fmt: skip
def test_a_resume_run_starts_only_the_session_recovery_granted_it(
    tmp_path, monkeypatch, sessions, change, resumes
):
    world = ended_session_sweep(monkeypatch, conclusion=None, sessions=sessions)
    number = len(sessions)
    if change != "ungranted":
        world.comments.append(grant_comment(
            world, number * 10 if change == "other-session" else number,
            author="maintainer" if change == "forged" else "Klaud-Cold"))
    if change == "outcome-recorded":
        world.comments.append(recorded(world.session))
    if change == "aborted":
        world.pull["labels"].append({"name": lifecycle.ABORT_LABEL})

    decision = str(resumes).lower()
    assert resume_check(tmp_path, monkeypatch, world, number) == (
        0, f"run={decision}\nresume={decision}\n")
    assert world.writes == []


@pytest.mark.parametrize("session", ["", "1"])
def test_a_resume_run_without_a_later_session_number_never_starts_a_session(
    tmp_path, monkeypatch, session
):
    world = ended_session_sweep(monkeypatch, conclusion=None, sessions=("failure", None))
    world.comments.append(grant_comment(world, 2))

    assert resume_check(tmp_path, monkeypatch, world, 2, session=session) == (
        0, "run=false\nresume=false\n")


@pytest.mark.parametrize(("change", "resumes"), [
    pytest.param("dispatched-by-someone-else", False, id="dispatched-by-someone-else"),
    pytest.param("failure", False, id="second-run-of-a-started-session"),
    pytest.param("ungranted", True, id="retry-after-a-run-that-never-started-its-agent"),
])  # fmt: skip
def test_a_resume_run_never_repeats_or_replaces_recoverys_session(
    tmp_path, monkeypatch, change, resumes
):
    world = ended_session_sweep(monkeypatch, conclusion=None, sessions=("failure",))
    world.comments.append(grant_comment(world, 2))
    if change == "dispatched-by-someone-else":
        world.start_resume(2, actor="maintainer")
        run = 4202
    else:
        # An earlier resume run of session 2 already ended ``change``.
        world.start_resume(2, outcome=change, hours=6)
        world.start_resume(2, run_id=4302, hours=12)
        run = 4302

    decision = str(resumes).lower()
    assert resume_check(tmp_path, monkeypatch, world, 2, run=run) == (
        0, f"run={decision}\nresume={decision}\n")
    assert world.writes == []


def test_select_continues_after_one_baseline_state_failure(tmp_path, monkeypatch):
    first_id = "1" * 16 + "-" + "2" * 16
    second_id = "3" * 16 + "-" + "4" * 16
    base = "a" * 40
    families = [
        "configs/nvidia-master.yaml:first-family",
        "configs/nvidia-master.yaml:second-family",
    ]
    contexts = [
        {"id": candidate_id, "family": family, "base": base, "source": {},
         "clusters": ["cluster-a"], "node-demand": 1}
        for candidate_id, family in zip((first_id, second_id), families, strict=True)
    ]
    (tmp_path / "candidates.json").write_text(json.dumps(contexts))
    review = {
        "decisions": [
            {
                "candidate-id": candidate_id,
                "decision": "proceed",
                "family": family,
                "pull-requests": [],
                "baseline-model": "Model",
                "reason": "No overlap",
            }
            for candidate_id, family in zip(
                (first_id, second_id), families, strict=True
            )
        ]
    }

    def resolve(_repository, candidate, _context, _model, _goal, _producers, _matrix):
        if candidate.id == first_id:
            raise OSError("transient")
        return reporting.Baseline(
            family=candidate.family,
            date="2026-09-19",
            image="example/image:1",
            goal=reporting.Prose(en="Check the baseline.", zh="检查基线。"),
            sources=["https://inferencex.semianalysis.com/api/v1/benchmarks"],
            points=[],
        )

    (tmp_path / "producers.json").write_text("{}\n")
    monkeypatch.setenv("GITHUB_REPOSITORY", "example/project")
    monkeypatch.setenv("GITHUB_RUN_ID", "42")
    monkeypatch.setenv("KLAUD_PR_REVIEW", json.dumps(review))
    monkeypatch.setattr(klaud, "fetch_capacity", lambda _policy: {"cluster-a": 1})
    monkeypatch.setattr(reporting, "resolve_baseline", resolve)
    monkeypatch.setattr(validation, "canonical_matrix", lambda *_: {"single_node": {"all": []}})
    monkeypatch.setattr(claims, "claim_family", lambda *_args: True)

    klaud.select(tmp_path, 5)

    selection = json.loads((tmp_path / "selection.json").read_text())
    assert selection["candidates"] == [second_id]
    assert selection["baseline-deferred-candidates"] == [first_id]
    assert selection["baseline-deferral-reasons"] == {first_id: "OSError: transient"}
    assert selection["deferred-reason"] is None
    preflight = reporting.BaselinePreflight.model_validate_json(
        (tmp_path / second_id / "baseline-preflight.json").read_text()
    )
    assert preflight.candidate_id == second_id
    assert preflight.baseline_model == "Model"
    assert preflight.source_identity == identity({})
    selected = json.loads((tmp_path / second_id / "candidate.json").read_text())
    assert selected["baseline-model"] == "Model"


@pytest.mark.parametrize(("headroom", "selected", "deferred"), [
    ({"h100-dsxe": 2, "h200-dgxc": 1}, ["exact", "elsewhere"], ["spanning"]),
    ({"h100-cw": 1, "h100-dsxe": 2, "h200-dgxc": 1}, ["spanning", "elsewhere"], ["exact"]),
    ({"h100-cw": 1, "h100-dsxe": 3, "h200-dgxc": 1}, ["spanning", "exact", "elsewhere"], []),
])  # fmt: skip
def test_select_needs_headroom_on_every_target_cluster_and_shares_it_within_a_wave(
    tmp_path, monkeypatch, headroom, selected, deferred
):
    ids = {"spanning": "1" * 16 + "-" + "2" * 16, "exact": "3" * 16 + "-" + "4" * 16,
           "elsewhere": "5" * 16 + "-" + "6" * 16}
    targets = {"spanning": (["h100-cw", "h100-dsxe"], 1), "exact": (["h100-dsxe"], 2),
               "elsewhere": (["h200-dgxc"], 1)}
    contexts = [
        {"id": ids[name], "family": f"configs/nvidia-master.yaml:{name}", "base": "a" * 40,
         "source": {}, "clusters": clusters, "node-demand": demand}
        for name, (clusters, demand) in targets.items()
    ]  # fmt: skip
    (tmp_path / "candidates.json").write_text(json.dumps(contexts))
    (tmp_path / "producers.json").write_text("{}\n")
    review = {"decisions": [
        {"candidate-id": context["id"], "decision": "proceed", "family": context["family"],
         "pull-requests": [], "baseline-model": "Model", "reason": "No overlap"}
        for context in contexts
    ]}  # fmt: skip

    def resolve(_repository, candidate, _context, _model, _goal, _producers, _matrix):
        return reporting.Baseline(
            family=candidate.family, date="2026-09-19", image="example/image:1",
            goal=reporting.Prose(en="Check the baseline.", zh="检查基线。"),
            sources=["https://inferencex.semianalysis.com/api/v1/benchmarks"], points=[],
        )  # fmt: skip

    monkeypatch.setenv("GITHUB_REPOSITORY", "example/project")
    monkeypatch.setenv("GITHUB_RUN_ID", "42")
    monkeypatch.setenv("KLAUD_PR_REVIEW", json.dumps(review))
    monkeypatch.setattr(klaud, "fetch_capacity", lambda _policy: dict(headroom))
    monkeypatch.setattr(reporting, "resolve_baseline", resolve)
    monkeypatch.setattr(validation, "canonical_matrix", lambda *_: {"single_node": {"all": []}})
    monkeypatch.setattr(claims, "claim_family", lambda *_args: True)

    klaud.select(tmp_path, 5)

    selection = json.loads((tmp_path / "selection.json").read_text())
    assert selection["candidates"] == [ids[name] for name in selected]
    assert selection["capacity-deferred-candidates"] == [ids[name] for name in deferred]


@pytest.mark.parametrize(("active", "selected", "note"), [
    pytest.param(0, 3, "", id="no-active-sessions"),
    pytest.param(1, 2, " 1 running or resumed sessions leave 2 of 3 slots.", id="cap-reduced"),
    pytest.param(3, 0, " 3 running or resumed sessions leave no slot for new candidates.",
                 id="cap-filled"),
    pytest.param(4, 0, " 4 running or resumed sessions leave no slot for new candidates.",
                 id="over-the-cap"),
])  # fmt: skip
def test_select_leaves_the_slots_of_running_and_resumed_sessions_to_them(
    tmp_path, monkeypatch, active, selected, note
):
    ids = [f"{digit * 16}-{digit * 16}" for digit in "123"]
    contexts = [
        {"id": candidate_id, "family": f"configs/nvidia-master.yaml:family-{digit}",
         "base": "a" * 40, "source": {}, "clusters": ["cluster-a"], "node-demand": 1}
        for candidate_id, digit in zip(ids, "123", strict=True)
    ]  # fmt: skip
    (tmp_path / "candidates.json").write_text(json.dumps(contexts))
    (tmp_path / "producers.json").write_text("{}\n")
    review = {"decisions": [
        {"candidate-id": context["id"], "decision": "proceed", "family": context["family"],
         "pull-requests": [], "baseline-model": "Model", "reason": "No overlap"}
        for context in contexts
    ]}  # fmt: skip

    def resolve(_repository, candidate, _context, _model, _goal, _producers, _matrix):
        return reporting.Baseline(
            family=candidate.family, date="2026-09-19", image="example/image:1",
            goal=reporting.Prose(en="Check the baseline.", zh="检查基线。"),
            sources=["https://inferencex.semianalysis.com/api/v1/benchmarks"], points=[],
        )  # fmt: skip

    claimed = []
    summary = tmp_path / "summary.md"
    for name, value in (("GITHUB_REPOSITORY", "example/project"), ("GITHUB_RUN_ID", "42"),
                        ("KLAUD_PR_REVIEW", json.dumps(review)),
                        ("GITHUB_STEP_SUMMARY", str(summary))):
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    monkeypatch.setattr(klaud, "fetch_capacity", lambda _policy: {"cluster-a": 3})
    monkeypatch.setattr(reporting, "resolve_baseline", resolve)
    monkeypatch.setattr(validation, "canonical_matrix", lambda *_: {"single_node": {"all": []}})
    monkeypatch.setattr(claims, "claim_family", lambda _r, owned, _run: (
        claimed.append(owned.id) or True))

    klaud.select(tmp_path, 3, active_sessions=active)

    assert json.loads((tmp_path / "selection.json").read_text())["candidates"] == ids[:selected]
    assert claimed == ids[:selected]
    assert summary.read_text() == (
        f"Klaud Cold: selected {selected} of 3 eligible candidates.{note}\n\n"
        "See selection.json and review-diagnostics.json in the klaud-plan artifact.\n")


def test_prepare_baseline_uses_bound_preflight_and_rejects_source_drift(
    tmp_path, monkeypatch
):
    candidate = OwnedCandidate(
        id="1" * 16 + "-" + "2" * 16,
        family="configs/nvidia-master.yaml:test-family",
        base="a" * 40,
    )
    context = {"source": {"date": "2026-09-19", "image": "example/image:1"}}
    original_goal = reporting.Prose(en="Check the baseline.", zh="检查基线。")
    goal = reporting.Prose(en="Update the image.", zh="更新镜像。")
    baseline = reporting.Baseline(
        family=candidate.family,
        date="2026-09-19",
        image="example/image:1",
        goal=original_goal,
        sources=["https://inferencex.semianalysis.com/api/v1/benchmarks"],
        points=[
            reporting.Point(
                key="b" * 64,
                label="8k/1k c4",
                conc=4,
                scenario="fixed-seq-len",
                values=reporting.Values(total_tps_gpu=42),
                result="passed",
            )
        ],
    )
    preflight = reporting.BaselinePreflight(
        candidate_id=candidate.id,
        base=candidate.base,
        baseline_model="Model",
        source_identity=identity(context["source"]),
        baseline=baseline,
    )
    (tmp_path / "baseline-preflight.json").write_text(
        preflight.model_dump_json(by_alias=True)
    )
    monkeypatch.setenv("KLAUD_EVIDENCE", str(tmp_path))
    session = SimpleNamespace(repository="example/project", candidate=candidate)

    prepared = reporting.prepare_baseline(session, context, "Model", goal)
    assert prepared.goal == goal
    assert prepared.points[0].values.total_tps_gpu == 42
    with pytest.raises(github.VerificationError, match="preflight does not match"):
        reporting.prepare_baseline(
            session,
            {"source": {**context["source"], "image": "example/image:2"}},
            "Model",
            goal,
        )
    (tmp_path / "baseline-preflight.json").unlink()
    with pytest.raises(github.VerificationError, match="artifact is missing"):
        reporting.prepare_baseline(session, context, "Model", goal)


@pytest.mark.parametrize("frozen", [True, False], ids=["resumed-session", "first-session"])
def test_prepare_baseline_keeps_the_baseline_the_candidate_pr_froze(tmp_path, monkeypatch, frozen):
    candidate = OwnedCandidate(id=CANDIDATE_ID, family=FAMILY, base="a" * 40)
    source = {"date": "2026-09-19", "image": "example/image:1"}
    preflight = reporting.Baseline(
        family=FAMILY, date=source["date"], image=source["image"],
        goal=reporting.Prose(en="Check the baseline.", zh="检查基线。"),
        sources=["https://inferencex.semianalysis.com/api/v1/benchmarks"],
        points=[reporting.Point(
            key="b" * 64, label="8k/1k c4", conc=4, scenario="fixed-seq-len",
            values=reporting.Values(total_tps_gpu=42), result="passed",
        )],
    )  # fmt: skip
    published = preflight.model_copy(update={"goal": reporting.Prose(
        en="Update the image to 2.", zh="将镜像更新为 2。")})
    goal = reporting.Prose(en="Update the image to 3.", zh="将镜像更新为 3。")
    if not frozen:
        (tmp_path / "goal.json").write_text(goal.model_dump_json())
    (tmp_path / "candidate.json").write_text(json.dumps({"source": source}))
    (tmp_path / "baseline-preflight.json").write_text(reporting.BaselinePreflight(
        candidate_id=CANDIDATE_ID, base="a" * 40, baseline_model="Model",
        source_identity=identity(source), baseline=preflight,
    ).model_dump_json(by_alias=True))  # fmt: skip
    pull = {"number": 7, "body": (
        reporting.baseline_marker(published) if frozen else "<!-- klaud-baseline -->")}
    monkeypatch.setenv("KLAUD_EVIDENCE", str(tmp_path))
    monkeypatch.setattr(lifecycle, "current_session", lambda: SimpleNamespace(
        repository="example/project", parent={"id": 42}, candidate=candidate,
        pulls=lambda: [pull], refresh=lambda _pull: pull,
    ))
    monkeypatch.setattr(github, "items", lambda _repository, path, _key=None: (
        [] if path == "issues/7/comments?per_page=100" else pytest.fail(path)))
    output = tmp_path / "baseline.json"
    monkeypatch.setattr(sys, "argv", [
        "klaud", "prepare-baseline", "--model", "Model", "--goal-file",
        str(tmp_path / "goal.json"), "--output", str(output),
    ])  # fmt: skip

    assert klaud.main() == 0
    assert reporting.Baseline.model_validate_json(output.read_text()) == (
        published if frozen else preflight.model_copy(update={"goal": goal}))


@pytest.mark.parametrize(("family_key", "published_key", "fingerprint"), [
    ("test-family", "test-family", "fingerprint"),
    ("model-fp8-h200-trtllm-mtp", "model-fp8-h200-trt-mtp", None),
], ids=["fingerprinted", "legacy-row-under-aliased-key"])  # fmt: skip
def test_baseline_normalizes_enroot_image_and_rejects_unverified_provenance(
    monkeypatch, family_key, published_key, fingerprint
):
    base = "a" * 40
    historical_head = "b" * 40
    raw_image = "nvcr.io#nvidia/trtllm:1"
    public_image = "nvcr.io/nvidia/trtllm:1"
    entry = {
        "model-prefix": "Model",
        "runner": "h200",
        "framework": "trt",
        "precision": "fp8",
        "spec-decoding": "mtp",
        "disagg": False,
        "image": raw_image,
        "conc": [1, 2],
        "isl": 8000,
        "osl": 1000,
        "tp": 8,
        "ep": 1,
        "recipe-fingerprint": "fingerprint",
    }
    current = {
        "single_node": {
            "all": [
                {
                    **entry,
                    "conc": [1],
                    "runner": "cluster:h200-a",
                    "srt-recipe": "recipes/h200.yaml",
                    "recipe-fingerprint": "current-fingerprint",
                }
            ]
        }
    }
    historical_matrix = {"single_node": {"all": [entry]}}
    public_row = {
        "model": "Model",
        "hardware": "h200",
        "framework": "trt",
        "precision": "fp8",
        "spec_method": "mtp",
        "disagg": False,
        "is_multinode": False,
        "benchmark_type": "single_turn",
        "isl": 8000,
        "osl": 1000,
        "offload_mode": "off",
        "conc": 1,
        "image": raw_image,
        "prefill_tp": 8,
        "prefill_ep": 1,
        "prefill_dp_attention": False,
        "prefill_num_workers": 0,
        "decode_tp": 8,
        "decode_ep": 1,
        "decode_dp_attention": False,
        "decode_num_workers": 0,
        "recipe_fingerprint": fingerprint,
        "run_url": "https://github.com/example/project/actions/runs/42",
        "tput_per_gpu": 10.0,
        "output_tput_per_gpu": 8.0,
        "mean_ttft": 0.1,
        "mean_tpot": 0.02,
    }
    info = {
        "runs": [{"github_run_id": "42", "run_attempt": 1}],
        "runConfigs": [{"github_run_id": "42", "head_sha": historical_head}],
        "changelogs": [{"workflow_run_id": "42", "config_keys": [published_key]}],
    }

    feeds = {
        "benchmarks": [
            {**public_row, "image": None},
            public_row,
        ],
        "workflow-info": info,
        "evaluations": [],
    }

    def fetch(resource, **_kwargs):
        return Feed(
            url=f"https://inferencex.semianalysis.com/api/v1/{resource}",
            retrieved_at="2026-09-20T00:00:00Z",
            sha256="0" * 64,
            payload=feeds[resource],
        )

    monkeypatch.setattr(api, "fetch", fetch)
    producers = {historical_head: historical_matrix}
    candidate = OwnedCandidate(
        id="1" * 16 + "-" + "2" * 16,
        family=f"configs/nvidia-master.yaml:{family_key}",
        base=base,
    )
    context = {
        "source": {
            "date": "2026-09-19",
            "image": public_image,
            "model": "Model",
            "hardware": "h200",
            "framework": "trt",
            "precision": "fp8",
            "spec_method": "mtp",
            "disagg": False,
        }
    }

    baseline = reporting.resolve_baseline(
        "example/project",
        candidate,
        context,
        "Model",
        reporting.Prose(en="Update the image.", zh="更新镜像。"),
        producers,
        current,
    )

    assert [(point.conc, point.result) for point in baseline.points] == [
        (1, "passed"),
        (2, "unavailable"),
    ]

    feeds["benchmarks"] = [
        {
            **public_row,
            "run_url": "https://github.com/example/project/actions/runs/999",
        }
    ]
    with pytest.raises(
        github.VerificationError, match="producer provenance is unavailable"
    ):
        reporting.resolve_baseline(
            "example/project",
            candidate,
            context,
            "Model",
            reporting.Prose(en="Update the image.", zh="更新镜像。"),
            producers,
            current,
        )


@pytest.mark.parametrize("disagg, workers, published, matches", [
    (False, 1, (True, 1, 1, 8), True),
    (False, 1, (False, 0, 0, 8), True),
    (False, 2, (True, 2, 2, 8), True),
    (False, 2, (False, 0, 0, 8), False),
    (False, 1, (True, 1, 0, 8), False),
    (True, 1, (True, 1, 1, 8), True),
    (True, 1, (False, 0, 0, 8), False),
])  # fmt: skip
def test_multinode_identity_matches_published_aggregate_encodings(
    disagg, workers, published, matches
):
    entry = {
        "model-prefix": "dsv4", "runner": "cluster:b200-nscale", "framework": "dynamo-sglang",
        "precision": "fp4", "spec-decoding": "mtp", "disagg": disagg,
        "scenario-type": "agentic-coding", "kv-offloading": "dram", "conc": 4,
        "image": "example/image:1",
        "prefill": {"num-worker": workers, "tp": 8, "ep": 1, "dp-attn": False},
        "decode": {"num-worker": int(disagg), "tp": 8 if disagg else 4, "ep": 1, "dp-attn": False},
    }
    multinode, prefill_workers, decode_workers, decode_tp = published
    row = {
        "model": "dsv4", "hardware": "b200", "framework": "dynamo-sglang", "precision": "fp4",
        "spec_method": "mtp", "disagg": disagg, "is_multinode": multinode,
        "benchmark_type": "agentic_traces", "isl": None, "osl": None, "offload_mode": "on",
        "conc": 4, "image": "example/image:1", "prefill_tp": 8, "prefill_ep": 1,
        "prefill_dp_attention": False, "prefill_num_workers": prefill_workers,
        "decode_tp": decode_tp, "decode_ep": 1, "decode_dp_attention": False,
        "decode_num_workers": decode_workers,
    }  # fmt: skip

    assert reporting._matches_public_point(row, reporting.public_point(entry)) is matches


def test_sglang_disagg_identity_matches_its_published_mori_sglang_rows():
    worker = {"num-worker": 1, "tp": 4, "ep": 1, "dp-attn": False}
    entry = {
        "model-prefix": "dsv4", "runner": "cluster:mi355x-amds", "framework": "sglang-disagg",
        "precision": "fp4", "spec-decoding": "draft_model", "disagg": True,
        "scenario-type": "agentic-coding", "kv-offloading": "dram", "conc": 16,
        "image": "example/image:1", "prefill": worker, "decode": worker,
    }  # fmt: skip
    row = {
        "model": "dsv4", "hardware": "mi355x", "framework": "mori-sglang", "precision": "fp4",
        "spec_method": "draft_model", "disagg": True, "is_multinode": True,
        "benchmark_type": "agentic_traces", "isl": None, "osl": None, "offload_mode": "on",
        "conc": 16, "image": "example/image:1", "prefill_tp": 4, "prefill_ep": 1,
        "prefill_dp_attention": False, "prefill_num_workers": 1, "decode_tp": 4,
        "decode_ep": 1, "decode_dp_attention": False, "decode_num_workers": 1,
    }  # fmt: skip

    assert reporting._matches_public_point(row, reporting.public_point(entry)) is True


@pytest.fixture
def family_configs():
    master = {
        "fixture": {
            "image": "example/image:stable",
            "model": "example/model",
            "model-prefix": "dsr1",
            "precision": "fp8",
            "framework": "sglang",
            "runner": "cluster:fixture",
            "multinode": False,
            "srt-recipe-dir": "fixture",
            "scenarios": {
                "agentic-coding": [{
                    "search-space": [{
                        "tp": 1, "kv-offloading": "none", "conc-list": [2, 6],
                        "srt-recipe": "recipe.yaml",
                    }],
                }],
            },
        },
    }
    runners = {
        "labels": {"cluster:fixture": ["node-a"]},
        "clusters": {"fixture": {
            "gpus-per-node": 8, "arch": "x86_64", "scheduler": "slurm",
            "slurm": {"partition": "batch", "exclusive": True},
        }},
    }
    return master, runners


def write_fixture_recipe(root: Path) -> None:
    recipe = root / "benchmarks/single_node/srt-slurm-recipes/fixture/recipe.yaml"
    recipe.parent.mkdir(parents=True, exist_ok=True)
    recipe.write_text("{}\n")


@pytest.mark.parametrize("prefix", ["", "inferencex-e2e/"])
def test_canonical_family_matrix_reads_revision_layout(monkeypatch, family_configs, prefix):
    master, runners = family_configs
    # A fixed-seq-len curve of the same family is outside Klaud's AgentX scope.
    master["fixture"]["scenarios"]["fixed-seq-len"] = [{
        "isl": 8192, "osl": 1024,
        "search-space": [{"tp": 1, "conc-list": [4], "srt-recipe": "recipe.yaml"}],
    }]
    files = {
        prefix + "configs/nvidia-master.yaml": yaml.safe_dump(master),
        prefix + "configs/runners.yaml": yaml.safe_dump(runners),
        prefix + "benchmarks/single_node/srt-slurm-recipes/fixture/recipe.yaml":
            yaml.safe_dump(single_node_fragment(1)).encode(),
        **{prefix + path: text.encode() for path, text in shared_blocks().items()},
    }  # fmt: skip
    monkeypatch.setattr(
        github, "read",
        lambda *_: {"tree": [{"path": prefix.rstrip("/") or "configs"}]},
    )
    monkeypatch.setattr(github, "file_at", lambda _repo, _head, path: files[path])

    def family() -> list[dict]:
        matrix = validation.canonical_matrix(
            "example/project", "a" * 40, "configs/nvidia-master.yaml:fixture",
        )
        assert [row["conc"] for row in matrix["evals"]] == [6]
        return matrix["single_node"]["all"]

    rows = family()
    assert [(row.get("scenario-type"), row["conc"], row["image"]) for row in rows] == [
        ("agentic-coding", 2, "example/image:stable"),
        ("agentic-coding", 6, "example/image:stable"),
    ]
    block = prefix + "configs/srt-recipes/agentic.yaml"
    files[block] = files[block] + b"  env: {EXTRA: '1'}\n"
    changed = family()
    assert [row["conc"] for row in changed] == [2, 6]
    assert {row["recipe-fingerprint"] for row in changed}.isdisjoint(
        row["recipe-fingerprint"] for row in rows
    )


CANDIDATE_ID = "1" * 16 + "-" + "2" * 16


def publish(monkeypatch, head):
    """Serve public rows of FAMILY's two points, produced by run 42 at ``head``; return their source."""
    image = "example/image:stable"
    published = [
        {
            "id": str(conc), "workflow_run_id": "7",
            "model": "dsr1", "hardware": "fixture", "framework": "sglang", "precision": "fp8",
            "spec_method": "none", "disagg": False, "is_multinode": False,
            "benchmark_type": "agentic_traces", "isl": None, "osl": None, "offload_mode": "off",
            "conc": conc, "image": image, "prefill_tp": 1, "prefill_ep": 1,
            "prefill_dp_attention": False, "prefill_num_workers": 0, "decode_tp": 1,
            "decode_ep": 1, "decode_dp_attention": False, "decode_num_workers": 0,
            "run_url": "https://github.com/example/project/actions/runs/42",
            "tput_per_gpu": 5.0 * conc, "output_tput_per_gpu": 4.0 * conc, "mean_ttft": 0.1,
            "mean_tpot": 0.02,
        }
        for conc in (2, 6)
    ]
    feeds = {
        "benchmarks": published,
        "workflow-info": {
            "runs": [{"github_run_id": "42", "run_attempt": 1}],
            "runConfigs": [{"github_run_id": "42", "head_sha": head}],
            "changelogs": [{"workflow_run_id": "42", "config_keys": [KEY]}],
        },
        "benchmark-siblings": {
            "sku": {"github_run_id": 42, "dataset_slug": "cc-traces-weka-062126"},
        },
        "evaluations": [],
    }
    monkeypatch.setattr(api, "fetch", lambda resource, **_: Feed(
        url=f"https://inferencex.semianalysis.com/api/v1/{resource}",
        retrieved_at="2026-09-20T00:00:00Z", sha256="0" * 64, payload=feeds[resource],
    ))
    return {
        "date": "2026-09-19", "image": image, "model": "dsr1", "hardware": "fixture",
        "framework": "sglang", "precision": "fp8", "spec_method": "none", "disagg": False,
    }


def review_candidate(directory, monkeypatch, source):
    """Write one FAMILY candidate observed at ``source`` and a review that proceeds with it."""
    directory.mkdir()
    (directory / "candidates.json").write_text(json.dumps(
        [{"id": CANDIDATE_ID, "family": FAMILY, "base": "a" * 40, "source": source,
          "clusters": ["cluster-a"], "node-demand": 1}]
    ))
    monkeypatch.setenv("GITHUB_REPOSITORY", "example/project")
    monkeypatch.setenv("GITHUB_RUN_ID", "42")
    monkeypatch.setenv("KLAUD_PR_REVIEW", json.dumps({"decisions": [{
        "candidate-id": CANDIDATE_ID, "decision": "proceed", "family": FAMILY, "pull-requests": [],
        "baseline-model": "DeepSeek-R1", "reason": "No overlap",
    }]}))


@pytest.mark.parametrize(("regenerated", "selected", "deferred"), [
    (True, [CANDIDATE_ID], []),
    (False, [], [CANDIDATE_ID]),
])  # fmt: skip
def test_select_builds_baselines_from_regenerated_producers_without_starting_a_process(
    tmp_path, monkeypatch, regenerated, selected, deferred
):
    _, head = commit_history(tmp_path, "inferencex-e2e", master=AGENTX_MASTER)
    monkeypatch.chdir(tmp_path)
    base_family = validation.producer_matrix("example/project", head, FAMILY)
    monkeypatch.setattr(validation, "canonical_matrix", lambda *_: base_family)
    forbid_current_config_parsing(monkeypatch)
    directory = tmp_path / "klaud"
    review_candidate(directory, monkeypatch, publish(monkeypatch, head))
    if regenerated:
        klaud.regenerate_producers(directory)
    else:
        (directory / "producers.json").write_text("{}\n")

    def no_process(args, *_args, **_kwargs):
        raise AssertionError(f"select started {args!r}")

    monkeypatch.setattr(subprocess, "Popen", no_process)
    monkeypatch.setattr(klaud, "fetch_capacity", lambda _policy: {"cluster-a": 1})
    monkeypatch.setattr(claims, "claim_family", lambda *_args: True)

    klaud.select(directory, 5)

    selection = json.loads((directory / "selection.json").read_text())
    assert (selection["candidates"], selection["baseline-deferred-candidates"]) == (selected, deferred)
    if regenerated:
        preflight = reporting.BaselinePreflight.model_validate_json(
            (directory / CANDIDATE_ID / "baseline-preflight.json").read_text()
        )
        assert [
            (point.conc, point.result, point.head, point.values.total_tps_gpu)
            for point in preflight.baseline.points
        ] == [(2, "passed", head, 10.0), (6, "passed", head, 30.0)]
        assert {point.dataset for point in preflight.baseline.points} == {"cc-traces-weka-062126"}


def test_select_defers_a_baseline_the_current_family_no_longer_generates(tmp_path, monkeypatch):
    _, head = commit_history(tmp_path, "inferencex-e2e", master=AGENTX_MASTER)
    monkeypatch.chdir(tmp_path)
    producer_family = validation.producer_matrix("example/project", head, FAMILY)
    shrunk = {"single_node": {"all": [
        row for row in producer_family["single_node"]["all"] if row["conc"] == 2
    ]}}  # fmt: skip
    monkeypatch.setattr(validation, "canonical_matrix", lambda *_: shrunk)
    directory = tmp_path / "klaud"
    review_candidate(directory, monkeypatch, publish(monkeypatch, head))
    klaud.regenerate_producers(directory)

    def no_claim(*_args):
        raise AssertionError("a deferred candidate must not be claimed")

    monkeypatch.setattr(klaud, "fetch_capacity", lambda _policy: {"cluster-a": 1})
    monkeypatch.setattr(claims, "claim_family", no_claim)

    klaud.select(directory, 5)

    selection = json.loads((directory / "selection.json").read_text())
    assert selection["candidates"] == []
    assert selection["baseline-deferred-candidates"] == [CANDIDATE_ID]
    assert selection["baseline-mismatch-candidates"] == [
        {"id": CANDIDATE_ID, "reason": "baseline-point-mismatch", "missing-points": 1}
    ]
    assert not (directory / CANDIDATE_ID).exists()


PRODUCER = "b" * 40
DEPRECATED_1K1K = (
    "Deprecated since 2026-07-17 ([#2263](https://github.com/example/project/pull/2263)),"
    " to save cluster time."
)


def models_md(
    status=DEPRECATED_1K1K, deprecated="Single-turn 1k1k", active="Single-turn 8k1k",
    scenarios="", models="",
):
    """MODELS.md whose 1k/1k row has ``status`` and whose dsr1 row lists ``active`` and
    ``deprecated`` scenarios; ``scenarios`` and ``models`` append raw lines to either table."""
    return (
        "## Scenarios\n\n| Scenario | ISL/OSL | Status |\n|---|---|---|\n"
        "| Single-turn 8k1k | 8192 / 1024 | Active. |\n"
        f"| Single-turn 1k1k | 1024 / 1024 | {status} |\n{scenarios}\n"
        "## Model support matrix\n\n"
        "| Model architecture class | Prefix | Date added | Active scenarios | Deprecated scenarios |\n"
        "|---|---|---|---|---|\n"
        f"| DeepSeek-R1-0528 | `dsr1` | 2025-08-13 | {active} | {deprecated} |\n{models}"
    )


def dsr1_point(isl, conc):
    """One generated TP8 recipe of the dsr1 family at ISL ``isl`` and concurrencies ``conc``."""
    return {
        "image": "example/image:stable", "model": "example/model", "model-prefix": "dsr1",
        "precision": "fp8", "framework": "sglang", "runner": "fixture", "isl": isl, "osl": 1024,
        "tp": 8, "pp": 1, "dcp-size": 1, "pcp-size": 1, "ep": 1, "dp-attn": False,
        "spec-decoding": "none", "disagg": False, "conc": conc, "max-model-len": isl + 1280,
        "exp-name": f"dsr1_{isl}", "run-eval": False, "recipe-fingerprint": f"{isl:064x}",
    }


def serve_models_md(monkeypatch, markdown):
    """Serve ``markdown`` as the MODELS.md that may only be read at the candidate base."""

    def file_at(_repository, head, path):
        assert (head, path) == ("a" * 40, "inferencex-e2e/docs/MODELS.md")
        return markdown.encode()

    monkeypatch.setattr(github, "read", lambda *_: {"tree": [{"path": "inferencex-e2e"}]})
    monkeypatch.setattr(github, "file_at", file_at)


def serve_dsr1_baseline(monkeypatch, current, markdown):
    """Publish 1k/1k and 8k/1k c1 from PRODUCER on 2026-05-22 for a base family of ``current``
    rows and base ``markdown`` MODELS.md; return the producer families and the source."""
    published = [
        {
            "model": "dsr1", "hardware": "fixture", "framework": "sglang", "precision": "fp8",
            "spec_method": "none", "disagg": False, "is_multinode": False,
            "benchmark_type": "single_turn", "isl": isl, "osl": 1024, "offload_mode": "off",
            "conc": 1, "image": "example/image:stable", "prefill_tp": 8, "prefill_ep": 1,
            "prefill_dp_attention": False, "prefill_num_workers": 0, "decode_tp": 8,
            "decode_ep": 1, "decode_dp_attention": False, "decode_num_workers": 0,
            "recipe_fingerprint": f"{isl:064x}",
            "run_url": "https://github.com/example/project/actions/runs/42",
            "tput_per_gpu": 10.0, "output_tput_per_gpu": 8.0, "mean_ttft": 0.1,
            "mean_tpot": 0.02,
        }
        for isl in (1024, 8192)
    ]
    feeds = {
        "benchmarks": published,
        "workflow-info": {
            "runs": [{"github_run_id": "42", "run_attempt": 1}],
            "runConfigs": [{"github_run_id": "42", "head_sha": PRODUCER}],
            "changelogs": [{"workflow_run_id": "42", "config_keys": [KEY]}],
        },
        "evaluations": [],
    }
    monkeypatch.setattr(api, "fetch", lambda resource, **_: Feed(
        url=f"https://inferencex.semianalysis.com/api/v1/{resource}",
        retrieved_at="2026-09-20T00:00:00Z", sha256="0" * 64, payload=feeds[resource],
    ))
    monkeypatch.setattr(validation, "canonical_matrix", lambda *_: {"single_node": {"all": current}})
    serve_models_md(monkeypatch, markdown)
    producer = {"single_node": {"all": [dsr1_point(1024, [1]), dsr1_point(8192, [1])]}}
    source = {
        "date": "2026-05-22", "image": "example/image:stable", "model": "dsr1", "hardware": "fixture",
        "framework": "sglang", "precision": "fp8", "spec_method": "none", "disagg": False,
    }
    return {PRODUCER: producer}, source


@pytest.mark.parametrize(("markdown", "current", "retired"), [
    pytest.param(models_md(), [dsr1_point(8192, [1])], True, id="deprecated-after-baseline"),
    pytest.param(
        models_md(
            "**Deprecated for all models** since 2026-07-17"
            " ([#2263](https://github.com/example/project/pull/2263))."
        ),
        [dsr1_point(8192, [1])], True, id="deprecated-for-all-models",
    ),
    pytest.param(
        models_md(DEPRECATED_1K1K.replace("2026-07-17", "2026-05-22")), [dsr1_point(8192, [1])],
        False, id="deprecated-on-baseline-day",
    ),
    pytest.param(
        models_md(DEPRECATED_1K1K.replace("2026-07-17", "2026-09-31")), [dsr1_point(8192, [1])],
        False, id="invalid-date",
    ),
    pytest.param(
        models_md(DEPRECATED_1K1K.replace("example/project", "other/project")),
        [dsr1_point(8192, [1])], False, id="other-repository",
    ),
    pytest.param(
        models_md(DEPRECATED_1K1K.replace("[#2263]", "[#2264]")), [dsr1_point(8192, [1])],
        False, id="pr-number-mismatch",
    ),
    pytest.param(models_md("Active."), [dsr1_point(8192, [1])], False, id="only-absent"),
    pytest.param(
        models_md(scenarios="| Single-turn 1k1k | 1024/1024 | Active. |\n"),
        [dsr1_point(8192, [1])], False, id="duplicate-scenario-row",
    ),
    pytest.param(
        models_md(models="\n| Scenario | ISL/OSL | Status |\n|---|---|---|\n"),
        [dsr1_point(8192, [1])], False, id="duplicate-scenarios-table",
    ),
    pytest.param(
        models_md(scenarios="| Single-turn 4k1k | 4096 / 1024 | Active. | stray |\n"),
        [dsr1_point(8192, [1])], False, id="malformed-row",
    ),
    pytest.param(models_md(deprecated=""), [dsr1_point(8192, [1])], False, id="model-not-listed"),
    pytest.param(
        models_md(active="Single-turn 8k1k, Single-turn 1k1k"), [dsr1_point(8192, [1])], False,
        id="also-active",
    ),
    pytest.param(
        models_md(models="| DeepSeek-R1 | `dsr1` | 2025-01-20 | Single-turn 1k1k | |\n"),
        [dsr1_point(8192, [1])], False, id="duplicate-model-row",
    ),
    pytest.param(
        models_md(), [dsr1_point(1024, [2]), dsr1_point(8192, [1])], False,
        id="workload-still-run",
    ),
])  # fmt: skip
def test_select_exempts_only_points_models_md_retired_after_the_baseline(
    tmp_path, monkeypatch, markdown, current, retired
):
    producers, source = serve_dsr1_baseline(monkeypatch, current, markdown)
    directory = tmp_path / "klaud"
    review_candidate(directory, monkeypatch, source)
    (directory / "producers.json").write_text(json.dumps({FAMILY: producers}))
    monkeypatch.setattr(klaud, "fetch_capacity", lambda _policy: {"cluster-a": 1})
    monkeypatch.setattr(claims, "claim_family", lambda *_args: True)

    klaud.select(directory, 5)

    selection = json.loads((directory / "selection.json").read_text())
    if not retired:
        assert selection["candidates"] == []
        assert selection["baseline-mismatch-candidates"] == [
            {"id": CANDIDATE_ID, "reason": "baseline-point-mismatch", "missing-points": 1}
        ]
        return
    assert (selection["candidates"], selection["baseline-mismatch-candidates"]) == ([CANDIDATE_ID], [])
    baseline = reporting.BaselinePreflight.model_validate_json(
        (directory / CANDIDATE_ID / "baseline-preflight.json").read_text()
    ).baseline
    assert [(point.label.split(" c")[0], point.result) for point in baseline.points] == [
        ("1024/1024", "passed"), ("8192/1024", "passed"),
    ]
    assert baseline.retirements == [reporting.Retirement(
        scenario="Single-turn 1k1k", date="2026-07-17",
        pr_link="https://github.com/example/project/pull/2263", model_prefix="dsr1", isl=1024,
        osl=1024, points=[baseline.points[0].key],
    )]


def test_final_coverage_exempts_retired_points_and_reports_their_evidence(monkeypatch):
    current = [dsr1_point(8192, [1])]
    producers, source = serve_dsr1_baseline(monkeypatch, current, models_md())
    candidate = OwnedCandidate(id=CANDIDATE_ID, family=FAMILY, base="a" * 40)
    baseline = reporting.resolve_baseline(
        "example/project",
        candidate,
        {"source": source},
        "DeepSeek-R1",
        reporting.Prose(en="Update the image.", zh="更新镜像。"),
        producers,
        {"single_node": {"all": current}},
    )
    retired, kept = baseline.points

    reporting.check_baseline_coverage(
        "example/project", candidate, {"single_node": {"all": current}}, baseline
    )
    with pytest.raises(github.VerificationError, match="omits or changes frozen baseline points"):
        reporting.check_baseline_coverage(
            "example/project", candidate, {"single_node": {"all": [dsr1_point(8192, [2])]}},
            baseline,
        )

    final = reporting.Attempt(
        kind="final", number=0, head="c" * 40, image="example/image:next", run_id=7,
        run_attempt=1, status="passed",
        change=reporting.Prose(en="Validate the family.", zh="验证配置族。"),
        next=reporting.Prose(en="Mark ready.", zh="标记就绪。"),
        benchmarks_expected=1, benchmarks_passed=1, evals_expected=0, evals_passed=0,
        points=[kept.model_copy(update={"run_id": 7, "head": "c" * 40})],
    )
    for text in (
        reporting.render_body(baseline),
        reporting.render_attempt(final, baseline, "example/project"),
    ):
        [note] = [
            line for line in text.splitlines()
            if line.startswith("**Note:**") and f"1k/1k c1 TP8 EP1 {retired.key[:6]}" in line
        ]
        assert all(evidence in note for evidence in (
            "Single-turn 1k1k", "2026-07-17", "(https://github.com/example/project/pull/2263)",
        ))


def test_baseline_rejects_retirements_it_cannot_justify():
    fixed = [
        reporting.Point(
            key=key * 64, label=f"8k/1k c{conc}", conc=conc, scenario="fixed-seq-len",
            values=reporting.Values(), result="passed",
        )
        for key, conc in (("a", 1), ("b", 2))
    ]
    agentic = reporting.Point(
        key="c" * 64, label="AgentX c1", conc=1, scenario="agentic-coding",
        values=reporting.Values(), result="passed",
    )

    def baseline(roster, **changes):
        retirement = {
            "scenario": "Single-turn 1k1k", "date": "2026-07-17",
            "pr_link": "https://github.com/example/project/pull/2263", "model_prefix": "dsr1",
            "isl": 1024, "osl": 1024, "points": ["a" * 64],
        }
        return reporting.Baseline(
            family=FAMILY, date="2026-05-22", image="example/image:1",
            goal=reporting.Prose(en="Update the image.", zh="更新镜像。"), sources=[],
            points=roster, retirements=[reporting.Retirement(**{**retirement, **changes})],
        )

    for roster, changes, error in (
        (fixed, {"date": "2026-05-22"}, "after the baseline date"),
        (fixed, {"date": "2026-09-31"}, "day is out of range for month"),
        (fixed, {"points": ["d" * 64]}, "distinct frozen baseline points"),
        (fixed, {"points": ["a" * 64, "a" * 64]}, "distinct frozen baseline points"),
        (fixed, {"points": ["a" * 64, "b" * 64]}, "not retired"),
        ([*fixed, agentic], {"points": ["c" * 64]}, "Only fixed-seq-len points"),
        (fixed, {"pr_link": "https://example.com/pull/2263"}, "pr_link"),
    ):
        with pytest.raises(ValueError, match=error):
            baseline(roster, **changes)


def test_baselines_without_retirements_keep_the_earlier_record_format():
    baseline = reporting.Baseline(
        family=FAMILY, date="2026-05-22", image="example/image:1",
        goal=reporting.Prose(en="Update the image.", zh="更新镜像。"), sources=[],
        points=[reporting.Point(
            key="a" * 64, label="8192/1024 c1 TP8 EP1 aaaaaa", conc=1, scenario="fixed-seq-len",
            values=reporting.Values(), result="passed",
        )],
    )

    record = baseline.model_dump_json(by_alias=True)

    assert "retirements" not in json.loads(record)
    assert reporting.Baseline.model_validate_json(record) == baseline


def frozen_dsr1_baseline(monkeypatch):
    """The planner's baseline for a base family of 8k/1k c1 and c2 whose MODELS.md deprecated
    1k/1k after the 2026-05-22 publication: points are 1k/1k c1 (retired), 8k/1k c1 and c2."""
    current = [dsr1_point(8192, [1, 2])]
    producers, source = serve_dsr1_baseline(monkeypatch, current, models_md())
    return reporting.resolve_baseline(
        "example/project", OwnedCandidate(id=CANDIDATE_ID, family=FAMILY, base="a" * 40),
        {"source": source}, "DeepSeek-R1", reporting.Prose(en="Update the image.", zh="更新镜像。"),
        producers, {"single_node": {"all": current}},
    )


def check_final(tmp_path, monkeypatch, record, head_concs):
    """Run check-final for a PR whose body freezes ``record`` and whose head generates 8k/1k at
    ``head_concs``, while the candidate base still generates 8k/1k c1 and c2."""
    head = "c" * 40
    families = {"a" * 40: [dsr1_point(8192, [1, 2])], head: [dsr1_point(8192, head_concs)]}
    monkeypatch.setattr(
        validation, "canonical_matrix",
        lambda _repository, ref, _family: {"single_node": {"all": families[ref]}},
    )
    monkeypatch.setattr(api, "fetch", lambda *_args, **_kwargs: pytest.fail("public API call"))
    pull = {"number": 7, "head": {"sha": head}, "body": reporting.baseline_marker(record)}
    monkeypatch.setattr(lifecycle, "current_session", lambda: SimpleNamespace(
        repository="example/project",
        candidate=OwnedCandidate(id=CANDIDATE_ID, family=FAMILY, base="a" * 40),
        pulls=lambda: [pull], refresh=lambda _pull: pull,
    ))
    matrix = tmp_path / "final-matrix.json"
    matrix.write_text(json.dumps({
        "single_node": {"all": families[head]},
        "changelog_metadata": {"head_ref": head, "entries": [{"config-keys": [KEY]}]},
    }))
    monkeypatch.setattr(sys, "argv", ["klaud", "check-final", "--matrix-file", str(matrix)])
    return klaud.main()


def test_check_final_accepts_a_retirement_it_rederives_at_the_candidate_base(
    tmp_path, monkeypatch, capfd
):
    baseline = frozen_dsr1_baseline(monkeypatch)
    record = baseline.model_copy(update={"retirements": [reporting.Retirement(
        scenario="Single-turn 1k1k", date="2026-07-17",
        pr_link="https://github.com/example/project/pull/2263", model_prefix="dsr1", isl=1024,
        osl=1024, points=[baseline.points[0].key],
    )]})

    assert check_final(tmp_path, monkeypatch, record, [1, 2]) == 0
    assert capfd.readouterr().out == ""


@pytest.mark.parametrize(("retired", "changes", "head_concs", "status"), [
    pytest.param([0, 2], {}, [1], DEPRECATED_1K1K, id="unretired-point"),
    pytest.param([0], {"isl": 8192}, [1, 2], DEPRECATED_1K1K, id="other-workload"),
    pytest.param(
        [0], {"pr_link": "https://github.com/other/project/pull/2263"}, [1, 2], DEPRECATED_1K1K,
        id="other-repository",
    ),
    pytest.param([0], {}, [1, 2], "Active.", id="not-deprecated-at-base"),
])  # fmt: skip
def test_check_final_rejects_retirements_the_candidate_base_does_not_derive(
    tmp_path, monkeypatch, capfd, retired, changes, head_concs, status
):
    baseline = frozen_dsr1_baseline(monkeypatch)
    [retirement] = baseline.retirements
    record = baseline.model_copy(update={"retirements": [retirement.model_copy(update={
        **changes, "points": [baseline.points[index].key for index in retired],
    })]})
    serve_models_md(monkeypatch, models_md(status))

    assert check_final(tmp_path, monkeypatch, record, head_concs) == 1
    assert capfd.readouterr().out == (
        "::error::Klaud: Recorded baseline retirements do not match MODELS.md at the candidate"
        " base.\n"
    )


def test_finish_and_recovery_rederive_retirements_before_readiness(monkeypatch):
    baseline = frozen_dsr1_baseline(monkeypatch)
    [retirement] = baseline.retirements
    record = baseline.model_copy(update={"retirements": [retirement.model_copy(update={
        "points": [baseline.points[0].key, baseline.points[2].key],
    })]})
    pull = {"number": 7, "head": {"sha": "c" * 40}, "body": reporting.baseline_marker(record)}
    session = lifecycle.Session(
        "example/project", {"id": 42},
        OwnedCandidate(id=CANDIDATE_ID, family=FAMILY, base="a" * 40),
    )
    monkeypatch.setattr(session, "pulls", lambda: [pull])
    monkeypatch.setattr(session, "refresh", lambda _pull: pull)
    monkeypatch.setattr(lifecycle, "verify_sweep", lambda *_: (
        {"single_node": {"all": [dsr1_point(8192, [1])]}}, [], [],
    ))

    with pytest.raises(github.VerificationError, match="retirements do not match MODELS.md"):
        session.validation({"id": 9, "head_sha": "c" * 40, "run_attempt": 1})


def attempt_comment(number, en, zh, *, author="Klaud-Cold", parent=42, candidate=CANDIDATE_ID,
                    run=9, chunked=False):
    record = reporting.Attempt(
        kind="initial", number=0, head="c" * 40, image="example/image:next", run_id=run,
        run_attempt=1, status="cancelled", change=reporting.Prose(en="Pin the image.", zh="固定镜像。"),
        next=reporting.Prose.model_construct(en=en, zh=zh), benchmarks_expected=1,
        benchmarks_passed=0, evals_expected=0, evals_passed=0, points=[],
    ).model_dump(by_alias=True)
    data = {"record": record, "chunks": [f"run-{run}-1-part-1-{'0' * 16}"]} if chunked else record
    return {"id": number, "user": {"login": author}, "body": (
        f"<!-- klaud-report:{parent}:{candidate}:run-{run}-1\n{json.dumps(data)}\n-->\nRendered.")}


def finish_report(monkeypatch, outcome, comments, recovering=False, repairs=0):
    session = lifecycle.Session(
        "example/project", {"id": 42}, OwnedCandidate(id=CANDIDATE_ID, family=FAMILY, base="a" * 40),
        recovering=recovering,
    )
    validated = outcome == "validated"
    pull = {"number": 7, "head": {"sha": "c" * 40}, "merged_at": None, "draft": not validated,
            "state": "open" if validated else "closed",
            "labels": [{"name": "full-sweep-fail-fast"}] if validated else []}
    run = {"id": 9, "event": "pull_request", "head_sha": "c" * 40, "run_attempt": 1,
           "created_at": "2026-10-15T00:00:00Z", "status": "completed",
           "conclusion": "success" if validated else "cancelled"}
    posted = []
    monkeypatch.setattr(session, "pulls", lambda: [pull])
    monkeypatch.setattr(session, "runs", lambda: [run])
    monkeypatch.setattr(session, "refresh", lambda _pull: pull)
    monkeypatch.setattr(session, "validation", lambda _run: None)
    monkeypatch.setattr(reporting, "publish_final", lambda *_: None)
    monkeypatch.setattr(claims, "release_family", lambda *_: None)
    monkeypatch.setattr(github, "items", lambda _repository, path, _key=None: (
        comments if path == "issues/7/comments?per_page=100" else []))
    monkeypatch.setattr(github, "write", lambda _repository, _path, _method, payload: (
        posted.append(payload["body"])))

    session.finish(CandidateOutcome(
        outcome=outcome, phase="final-sweep" if validated else "targeted", pull_request=7,
        run_ids=[9], repairs_used=repairs,
    ))

    [report] = [body for body in posted if body.startswith(session.marker)]
    return report.split("\n-->\n", 1)[1]


@pytest.mark.parametrize("ended", [True, False], ids=["ended-meanwhile", "still-running"])
def test_finish_tolerates_only_a_cancel_refused_because_the_run_already_ended(
    monkeypatch, ended
):
    session = lifecycle.Session(
        "example/project", {"id": 42}, OwnedCandidate(id=CANDIDATE_ID, family=FAMILY, base="a" * 40)
    )
    pull = {"number": 7, "head": {"sha": "c" * 40}, "merged_at": None, "draft": True,
            "state": "closed", "labels": []}
    run = {"id": 9, "event": "pull_request", "head_sha": "c" * 40, "run_attempt": 1,
           "created_at": "2026-10-15T00:00:00Z", "status": "in_progress", "conclusion": None}
    finished = {**run, "status": "completed", "conclusion": "success"}
    current, writes = [run], []

    def write(_repository, path, method, _payload=None):
        writes.append((method, path))
        if path == "actions/runs/9/cancel":
            if ended:
                current[0] = finished
            raise subprocess.CalledProcessError(1, ["gh", "api"])
        return {}

    monkeypatch.setattr(session, "pulls", lambda: [pull])
    monkeypatch.setattr(session, "runs", lambda: list(current))
    monkeypatch.setattr(session, "refresh", lambda _pull: pull)
    monkeypatch.setattr(claims, "release_family", lambda *_: None)
    monkeypatch.setattr(github, "items", lambda *_args: [])
    monkeypatch.setattr(github, "read", lambda _repository, path: (
        current[0] if path == "actions/runs/9" else pytest.fail(path)))
    monkeypatch.setattr(github, "write", write)
    outcome = CandidateOutcome(outcome="failed", phase="targeted", pull_request=7, run_ids=[9],
                               repairs_used=0)

    if ended:
        assert session.finish(outcome).run_ids == [9]
        assert writes == [("POST", "issues/7/comments"), ("POST", "actions/runs/9/cancel"),
                          ("POST", "issues/7/comments")]
    else:
        with pytest.raises(subprocess.CalledProcessError):
            session.finish(outcome)
        assert writes == [("POST", "issues/7/comments"), ("POST", "actions/runs/9/cancel")]


@pytest.mark.parametrize(("labels", "accepted"), [
    (["full-sweep-fail-fast"], True),
    (["full-sweep-enabled"], False),
    (["non-canary-full-sweep-enabled"], False),
    (["full-sweep-fail-fast", "full-sweep-enabled"], False),
    (["full-sweep-fail-fast", "all-evals"], False),
])  # fmt: skip
def test_validated_outcome_requires_a_fail_fast_only_sweep(monkeypatch, labels, accepted):
    session = lifecycle.Session(
        "example/project", {"id": 42}, OwnedCandidate(id=CANDIDATE_ID, family=FAMILY, base="a" * 40)
    )
    pull = {"number": 7, "head": {"sha": "c" * 40}, "merged_at": None, "draft": False,
            "state": "open", "labels": [{"name": name} for name in labels]}
    run = {"id": 9, "event": "pull_request", "head_sha": "c" * 40, "run_attempt": 1,
           "created_at": "2026-10-15T00:00:00Z", "status": "completed", "conclusion": "success"}
    validated = []
    monkeypatch.setattr(session, "pulls", lambda: [pull])
    monkeypatch.setattr(session, "runs", lambda: [run])
    monkeypatch.setattr(session, "validation", validated.append)
    outcome = CandidateOutcome(
        outcome="validated", phase="final-sweep", pull_request=7, run_ids=[9], repairs_used=0
    )

    if accepted:
        session.verify(outcome, require_report=False)
        assert validated == [run]
    else:
        with pytest.raises(github.VerificationError, match="incompatible sweep labels"):
            session.verify(outcome, require_report=False)
        assert validated == []



STOP = ("Stop as uncertain: the new image stalls every deployment.",
        "以 uncertain 结束：新镜像使所有部署停滞。")


@pytest.mark.parametrize(("outcome", "latest", "chunked", "recovering", "shown"), [
    pytest.param("uncertain", None, False, False, False, id="no-own-attempt"),
    pytest.param("uncertain", STOP, False, False, True, id="own-attempt"),
    pytest.param("uncertain", STOP, True, False, True, id="chunked-own-attempt"),
    pytest.param("uncertain", ("Traceback:\n<pre>CUDA error</pre>", "日志"), False, False, False,
                 id="raw-log"),
    pytest.param("validated", STOP, False, False, False, id="validated"),
    pytest.param("uncertain", STOP, False, True, False, id="recovered-session"),
    pytest.param("aborted", STOP, False, False, False, id="aborted-by-label"),
])  # fmt: skip
def test_closing_report_states_why_from_the_latest_own_attempt(
    monkeypatch, outcome, latest, chunked, recovering, shown
):
    other = ("Retry with another image.", "换用其他镜像重试。")
    own = [
        attempt_comment(1, "Repair the launcher flags.", "修复启动参数。", run=8),
        attempt_comment(2, *latest, chunked=chunked),
    ] if latest else []
    comments = [
        *own,
        attempt_comment(3, *other, author="maintainer"),
        attempt_comment(4, *other, parent=41),
        attempt_comment(5, *other, candidate="3" * 16 + "-" + "4" * 16),
        {"id": 6, "user": {"login": "Klaud-Cold"}, "body": (
            f"<!-- klaud-report:42:{CANDIDATE_ID}:run-9-1-part-1-{'0' * 16}\n"
            '{"points": [], "evals": []}\n-->\nRendered.')},
    ]
    runs = "[9](https://github.com/example/project/actions/runs/9)"
    done = {
        "uncertain": ("PR closed; branch deleted for retry.", "PR 已关闭；分支已删除，可重新尝试。"),
        "aborted": ("PR closed; branch deleted for retry.", "PR 已关闭；分支已删除，可重新尝试。"),
        "validated": ("Full sweep verified; ready for review.", "完整 sweep 已验证；已就绪，等待审查。"),
    }[outcome]
    why = (f"**Reason:** {latest[0]}  \n", f"**原因：**{latest[1]}  \n") if shown else ("", "")
    if outcome == "aborted":
        why = ("**Reason:** Stopped by the klaud-abort label.  \n",
               "**原因：**已由 klaud-abort 标签停止。  \n")

    assert finish_report(monkeypatch, outcome, comments, recovering) == (
        f"**{outcome}** · Repairs: 0 · Runs: {runs}  \n{why[0]}All owned runs ended. {done[0]}"
        "\n\n<details>\n<summary>中文</summary>\n\n"
        f"**{outcome}** · 修复次数：0 · 运行：{runs}  \n{why[1]}所有自有运行均已结束。{done[1]}"
        "\n\n</details>"
    )


def test_closing_report_translates_an_unknown_repair_count(monkeypatch):
    report = finish_report(monkeypatch, "uncertain", [], recovering=True, repairs=None)
    english, chinese = report.split("<summary>中文</summary>")

    assert "· Repairs: unknown ·" in english
    assert "· 修复次数：未知 ·" in chinese


def test_every_candidate_pr_body_says_how_to_stop_the_candidate(monkeypatch):
    body = reporting.render_body(frozen_dsr1_baseline(monkeypatch))
    english, chinese = body.split("<summary>中文</summary>")

    assert ("**Stop:** Add the `klaud-abort` label at any time to stop this candidate now; Klaud "
            "cancels its runs, closes this PR and does not retry it. Remove the label to let the "
            "planner pick it up again.") in english
    assert ("**停止：**随时添加 `klaud-abort` 标签即可立即停止该候选：Klaud 会取消其运行、关闭此 PR "
            "且不再重试。移除该标签后，规划器会在下次扫描时重新考虑它。") in chinese


def test_the_planner_skips_aborted_candidates_until_the_label_is_removed(monkeypatch):
    aborted, other = "1" * 16 + "-" + "2" * 16, "3" * 16 + "-" + "4" * 16
    issues = [
        {"number": 7, "user": {"login": "Klaud-Cold"}, "pull_request": {}},
        {"number": 8, "user": {"login": "maintainer"}, "pull_request": {}},
        {"number": 9, "user": {"login": "Klaud-Cold"}},
    ]
    heads = {"pulls/7": f"klaud/auto-{aborted}", "pulls/8": f"klaud/auto-{other}"}
    monkeypatch.setattr(klaud, "github_items", lambda _repository, path, _key=None: (
        issues if path == "issues?labels=klaud-abort&state=all&per_page=100"
        else pytest.fail(path)))
    monkeypatch.setattr(klaud, "github_read", lambda _repository, path: (
        {"head": {"ref": heads[path]}}))

    assert klaud.aborted_candidates("example/project") == {aborted}


@pytest.mark.parametrize(("labels", "outcome", "released"), [
    pytest.param([], "aborted", True, id="label-removed"),
    pytest.param(["klaud-abort"], "aborted", False, id="still-labeled"),
    pytest.param([], "uncertain", False, id="closed-for-another-reason"),
])  # fmt: skip
def test_removing_klaud_abort_lifts_the_cooldown_of_an_aborted_candidate(
    monkeypatch, labels, outcome, released
):
    candidate = "1" * 16 + "-" + "2" * 16
    record = {"head": "c" * 40, "outcome": {**UNEXPECTED_ERROR.model_dump(by_alias=True),
                                            "outcome": outcome}}
    comments = [
        {"id": 1, "user": {"login": "Klaud-Cold"}, "body": f"<!-- klaud-outcome:41:{candidate}\n"
         + json.dumps({**record, "outcome": {**record["outcome"], "outcome": "uncertain"}})
         + "\n-->\nOlder."},
        {"id": 2, "user": {"login": "Klaud-Cold"},
         "body": f"<!-- klaud-outcome:42:{candidate}\n{json.dumps(record)}\n-->\nLatest."},
    ]
    pulls = [{"number": 7, "user": {"login": "Klaud-Cold"}, "created_at": "2026-10-15T00:00:00Z",
              "labels": [{"name": name} for name in labels]}]

    def items(_repository, path, _key=None):
        if path.startswith("pulls?state=closed&head=example%3Aklaud%2Fauto-"):
            return pulls
        if path == "issues/7/comments?per_page=100":
            return comments
        raise AssertionError(path)

    monkeypatch.setattr(klaud, "github_items", items)

    assert klaud.abort_released("example/project", candidate) is released
    assert [c["id"] for c in klaud.order_by_cooldown([{"id": candidate}], {candidate: "aborted"})] \
        == [candidate]


def test_producer_outside_the_local_clone_is_fetched_from_its_repository(tmp_path, monkeypatch):
    remote = tmp_path / "remote"
    remote.mkdir()
    base, head = commit_history(remote, master=AGENTX_MASTER)

    def remote_git(*args):
        subprocess.run(["git", *args], cwd=remote, check=True, capture_output=True)

    remote_git("update-ref", "refs/pull/42/head", head)
    remote_git("reset", "-q", "--hard", base)
    remote_git("config", "uploadpack.allowReachableSHA1InWant", "true")
    local = tmp_path / "local"
    subprocess.run(["git", "clone", "-q", "--no-local", str(remote), str(local)], check=True)
    subprocess.run(
        ["git", "config", f"url.{remote}.insteadOf", "https://github.com/example/project"],
        cwd=local, check=True,
    )
    assert subprocess.run(
        ["git", "cat-file", "-e", f"{head}^{{commit}}"], cwd=local, capture_output=True,
    ).returncode
    monkeypatch.chdir(local)

    matrix = validation.producer_matrix("example/project", head, FAMILY)

    assert [
        (row["model"], row["conc"], row["image"]) for row in matrix["single_node"]["all"]
    ] == POINTS
    with pytest.raises(validation.ProducerRegenerationError) as failure:
        validation.producer_matrix("example/project", "f" * 40, FAMILY)
    assert f"not our ref {'f' * 40}" in failure.value.stderr


def test_producer_predating_no_evals_regenerates_throughput_rows(tmp_path, monkeypatch):
    from infx.tests.historical_revision import GENERATOR

    commit_history(tmp_path, master=AGENTX_MASTER)
    legacy = GENERATOR.replace(
        '    parser.add_argument("--no-evals", action="store_true")\n', ""
    ).replace(
        "    return result\n",
        "    result[0] = {**result[0], \"run-eval\": True}\n"
        "    return result + [{**result[0], \"eval-only\": True, \"run-eval\": True}]\n",
    )
    (tmp_path / "infx/matrix/generate.py").write_text(legacy)
    subprocess.run(["git", "commit", "-qam", "generator without --no-evals"], cwd=tmp_path, check=True)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True, text=True, check=True
    ).stdout.strip()
    monkeypatch.chdir(tmp_path)

    rows = validation.producer_matrix("example/project", head, FAMILY)["single_node"]["all"]

    assert [(row["conc"], row["run-eval"], row.get("eval-only")) for row in rows] == [
        (2, False, None), (6, False, None)]


def test_failing_producer_keeps_its_stderr_and_never_sees_credentials(tmp_path, monkeypatch):
    commit_history(tmp_path)
    (tmp_path / "infx/matrix/generate.py").write_text(
        "import os, sys\n"
        "print(f\"generator failed; token={os.environ.get('GH_TOKEN')}\", file=sys.stderr)\n"
        "sys.exit(1)\n"
    )
    subprocess.run(["git", "commit", "-qam", "failing generator"], cwd=tmp_path, check=True)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True, text=True, check=True
    ).stdout.strip()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GH_TOKEN", "ghp_fixture")

    with pytest.raises(validation.ProducerRegenerationError) as failure:
        validation.producer_matrix("example/project", head, FAMILY)

    assert str(failure.value) == "Baseline producer family cannot be regenerated"
    assert failure.value.stderr == "generator failed; token=None\n"


@pytest.mark.parametrize("from_repository_root", [False, True])
def test_live_families_preserve_public_identity_after_move(tmp_path, family_configs, from_repository_root):
    master, runners = family_configs
    project = tmp_path / "inferencex-e2e"
    configs = project / "configs"
    configs.mkdir(parents=True)
    (configs / "nvidia-master.yaml").write_text(yaml.safe_dump(master))
    (configs / "runners.yaml").write_text(yaml.safe_dump(runners))
    write_fixture_recipe(project)

    families = klaud.live_families(tmp_path if from_repository_root else project)

    assert families == {
        ("dsr1", "fixture", "sglang", "fp8", "none", False, "agentic_traces",
         None, None, "example/image:stable"): {
            "configs/nvidia-master.yaml:fixture": ({"fixture"}, 1),
        },
    }


def test_live_families_name_atom_disagg_as_its_published_mooncake_atom(tmp_path, family_configs):
    master, runners = family_configs
    master["fixture"]["framework"] = "atom-disagg"
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "amd-master.yaml").write_text(yaml.safe_dump(master))
    (configs / "runners.yaml").write_text(yaml.safe_dump(runners))
    write_fixture_recipe(tmp_path)

    families = klaud.live_families(tmp_path)

    assert [key[2] for key in families] == ["mooncake-atom"]



def test_historical_families_ignore_leftover_nested_results(tmp_path, family_configs):
    master, runners = family_configs
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "nvidia-master.yaml").write_text(yaml.safe_dump(master))
    (configs / "runners.yaml").write_text(yaml.safe_dump(runners))
    write_fixture_recipe(tmp_path)

    def git(*args):
        return subprocess.check_output(
            ["git", *args], cwd=tmp_path, text=True, stderr=subprocess.DEVNULL,
        ).strip()

    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.com")
    git("add", ".")
    git("commit", "-qm", "historical layout")
    historical = git("rev-parse", "HEAD")
    nested = tmp_path / "inferencex-e2e"
    nested.mkdir()
    git("mv", "configs", "inferencex-e2e/")
    git("commit", "-qm", "new layout")
    (nested / "results").mkdir()
    (nested / "results/output.json").write_text("{}\n")
    git("checkout", "--quiet", historical)

    families = klaud.live_families(tmp_path)

    assert families == {
        ("dsr1", "fixture", "sglang", "fp8", "none", False, "agentic_traces",
         None, None, "example/image:stable"): {
            "configs/nvidia-master.yaml:fixture": ({"fixture"}, 1),
        },
    }


def test_choose_needs_node_demand_on_every_target_cluster_and_skips_unresolved_runners(
    tmp_path, family_configs, capsys
):
    master, runners = family_configs
    fixture, record = master["fixture"], runners["clusters"]["fixture"]
    worker = {"num-worker": 1, "tp": 8, "ep": 1, "dp-attn": False}
    master = {
        "exact": {**fixture, "runner": "cluster:h100-dsxe"},
        "wide": {**fixture, "runner": "cluster:h100-dsxe", "multinode": True, "scenarios": {
            "agentic-coding": [{"search-space": [
                {"num-nodes": nodes, "worker": worker, "kv-offloading": "none", "conc-list": [4],
                 "srt-recipe": "wide.yaml"}
                for nodes in (2, 4, 3)
            ]}],
        }},
        "unmapped": {**fixture, "runner": "cluster:h100-retired"},
    }
    runners = {
        "labels": {
            "cluster:h100-cw": ["h100-cw_00"],
            "cluster:h100-dsxe": ["h100-dsxe_00"],
        },
        "clusters": {"h100-cw": record, "h100-dsxe": record},
    }
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "nvidia-master.yaml").write_text(yaml.safe_dump(master))
    (configs / "runners.yaml").write_text(yaml.safe_dump(runners))
    write_fixture_recipe(tmp_path)
    recipe = tmp_path / "benchmarks/multi_node/srt-slurm-recipes/fixture/wide.yaml"
    recipe.parent.mkdir(parents=True)
    recipe.write_text("{}\n")

    families = klaud.live_families(tmp_path)

    assert capsys.readouterr().out == (
        "::warning::Klaud live-family-clusters-unresolved: configs/nvidia-master.yaml:unmapped\n"
    )
    assert families == {
        ("dsr1", "h100", "sglang", "fp8", "none", False, "agentic_traces",
         None, None, "example/image:stable"): {
            "configs/nvidia-master.yaml:exact": ({"h100-dsxe"}, 1),
            "configs/nvidia-master.yaml:wide": ({"h100-dsxe"}, 4),
        },
    }  # fmt: skip
    source = {
        "model": "dsr1", "hardware": "h100", "framework": "sglang", "precision": "fp8",
        "spec_method": "none", "disagg": False, "benchmark_type": "agentic_traces", "isl": None,
        "osl": None, "image": "example/image:stable", "date": "2026-09-19",
    }  # fmt: skip
    items = [{"source-status": "baseline", "source-id": "1", "source": source,
              "benchmark-age-days": 0}]

    def chosen(headroom):
        return sorted(
            (candidate["family"].removeprefix("configs/nvidia-master.yaml:"),
             candidate["clusters"], candidate["node-demand"])
            for candidate in klaud.choose(items, headroom, set(), families)
        )

    assert chosen({"h100-cw": 1}) == []
    assert chosen({"h100-dsxe": 3}) == [("exact", ["h100-dsxe"], 1)]
    assert chosen({"h100-cw": 1, "h100-dsxe": 4}) == [
        ("exact", ["h100-dsxe"], 1), ("wide", ["h100-dsxe"], 4),
    ]


def agentx_point(dataset, **values):
    return reporting.Point(
        key="d" * 64, label="AgentX c4 TP8 EP1 dddddd", conc=4, scenario="agentic-coding",
        dataset=dataset, values=reporting.Values(**values), result="passed",
    )  # fmt: skip


@pytest.mark.parametrize(("old", "new", "cell", "note"), [
    pytest.param("cc-traces-weka-062126", "semianalysis_cc_traces_weka_062126", "12 (+20%)", None,
                 id="loader-matches-dashboard-slug"),
    pytest.param("cc-traces-weka-062126", "semianalysisai/cc-traces-weka-062126", "12 (+20%)",
                 None, id="hf-id-matches-dashboard-slug"),
    pytest.param(None, "cc-traces-weka-062126", "12 (N/A)", "baseline dataset unknown",
                 id="legacy-baseline"),
    pytest.param("cc-traces-weka-062126", "cc-traces-weka-062126-256k", "12 (N/A)",
                 "dataset cc-traces-weka-062126 vs cc-traces-weka-062126-256k", id="other-corpus"),
])  # fmt: skip
def test_agentx_deltas_compare_only_the_same_replay_corpus(old, new, cell, note):
    baseline = reporting.Baseline(
        family=f"configs/nvidia-master.yaml:{KEY}", date="2026-09-19", image="example/image:1",
        goal=reporting.Prose(en="Update the image.", zh="更新镜像。"), sources=[],
        points=[agentx_point(old, output_tps_gpu=10.0, ttft_ms=100.0, tpot_ms=10.0)],
    )  # fmt: skip
    point = agentx_point(new, output_tps_gpu=12.0, ttft_ms=90.0, tpot_ms=11.0, request_errors=0)

    text = reporting.point_table([point], baseline)

    assert f"| 4 | {cell} |" in text
    notes = [line for line in text.splitlines() if "Δ N/A" in line]
    assert notes == ([f"**Note:** All rows: Δ N/A: {note}."] if note else [])


@pytest.mark.parametrize(("completed", "errors", "note"), [
    pytest.param(100, 0, None, id="no-failed-requests"),
    pytest.param(97, 3, "**Note:** All rows: 3 request errors.", id="failed-requests"),
])  # fmt: skip
def test_a_fixed_sequence_result_reports_its_failed_requests(completed, errors, note):
    from infx.results.fixed_sequence import build_result

    raw = {
        "max_concurrency": 4, "model_id": "m", "num_prompts": 100, "completed": completed,
        "total_token_throughput": 800.0, "output_throughput": 400.0,
        "mean_ttft_ms": 100.0, "mean_tpot_ms": 10.0,
        "benchmark_outcome": {"status": "passed", "requested": 100, "completed": completed,
                              "failed": 100 - completed, "max_failure_rate": 0.05},
    }  # fmt: skip
    env = {
        "RUNNER_TYPE": "h100", "FRAMEWORK": "sglang", "PRECISION": "fp8", "SPEC_DECODING": "none",
        "ISL": "1024", "OSL": "1024", "DISAGG": "false", "MODEL_PREFIX": "dsr1", "IMAGE": "i:1",
        "TP": "8", "EP_SIZE": "1", "DP_ATTENTION": "false",
    }  # fmt: skip
    values = reporting.values(build_result(raw, env))
    point = reporting.Point(
        key="f" * 64, label="1024/1024 c4 TP8 EP1 ffffff", conc=4, scenario="fixed-seq-len",
        values=values, result="passed",
    )  # fmt: skip

    text = reporting.point_table([point])

    assert values.request_errors == errors
    assert values.output_tps_gpu == 50.0
    notes = [line.rstrip() for line in text.splitlines() if "request error" in line]
    assert notes == ([note] if note else [])


def test_baseline_notes_name_concurrency_rows_like_their_table():
    points = [
        agentx_point("cc", output_tps_gpu=10.0, request_errors=2),
        agentx_point("cc", output_tps_gpu=10.0).model_copy(update={
            "key": "e" * 64, "label": "AgentX c8 TP8 EP1 eeeeee", "conc": 8}),
    ]  # fmt: skip
    baseline = reporting.Baseline(
        family=f"configs/nvidia-master.yaml:{KEY}", date="2026-09-19", image="example/image:1",
        goal=reporting.Prose(en="Update the image.", zh="更新镜像。"), sources=[], points=points,
    )  # fmt: skip

    text = reporting.baseline_table(baseline)

    assert "| 4 | N/A | 10 | N/A | N/A |" in text
    assert text.endswith("**Note:** c4: 2 request errors.")


@pytest.mark.parametrize(("points", "change"), [
    pytest.param([], "**变更：**更新镜像。  \n", id="queued-without-results"),
    pytest.param([agentx_point("cc", output_tps_gpu=10.0, request_errors=0)],
                 "**变更：**更新镜像。实测数值及异常说明见上表。  \n", id="with-results"),
])  # fmt: skip
def test_attempt_translation_points_to_a_table_only_when_one_is_shown(points, change):
    record = reporting.Attempt(
        kind="initial", number=0, head="c" * 40, image="example/image:2", run_id=7,
        run_attempt=1, status="running",
        change=reporting.Prose(en="Update the image.", zh="更新镜像。"),
        next=reporting.Prose(en="Wait.", zh="等待。"), benchmarks_expected=1,
        benchmarks_passed=0, evals_expected=0, evals_passed=0, points=points,
    )  # fmt: skip

    english, chinese = reporting.render_attempt(record, None, "example/project").split("<details>")

    assert ("| Concurrency |" in english) == bool(points)
    assert change in chinese


def test_chunked_attempt_tells_both_languages_where_results_continue(monkeypatch):
    head = "c" * 40
    session = SimpleNamespace(
        repository="example/project", parent={"id": 42},
        candidate=OwnedCandidate(id=CANDIDATE_ID, family=FAMILY, base="a" * 40),
        pulls=lambda: [{"number": 7}], refresh=lambda _pull: {"body": ""},
        runs=lambda: [{"id": 9, "head_sha": head, "run_attempt": 1, "status": "in_progress",
                       "conclusion": None}],
    )  # fmt: skip
    points = [
        agentx_point("cc", output_tps_gpu=10.0, request_errors=0).model_copy(update={
            "key": f"{conc:064x}", "label": f"AgentX c{conc} TP8 EP1 {conc:06x}", "conc": conc})
        for conc in range(1, 22)
    ]  # fmt: skip
    record = reporting.Attempt(
        kind="initial", number=0, head=head, image="example/image:2", run_id=9, run_attempt=1,
        status="running", change=reporting.Prose(en="Update the image.", zh="更新镜像。"),
        next=reporting.Prose(en="Wait.", zh="等待。"), benchmarks_expected=21,
        benchmarks_passed=0, evals_expected=0, evals_passed=0, points=points,
    )  # fmt: skip
    posted = []
    monkeypatch.setattr(github, "items", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        github, "write", lambda _repo, _path, _method, payload: posted.append(payload["body"])
    )

    reporting.publish(session, record)

    [main] = [body for body in posted if ":run-9-1\n" in body.split("-->", 1)[0]]
    english, chinese = main.split("<details>")
    assert "Results continue in numbered report comments." in english
    assert "结果见后续编号的报告评论。" in chinese


def eval_entry(tp):
    return {
        "model-prefix": "dsr1", "runner": "cluster:fixture-a", "framework": "sglang",
        "precision": "fp8", "spec-decoding": "none", "disagg": False,
        "scenario-type": "agentic-coding", "tp": tp, "ep": 1, "dp-attn": False, "conc": 8,
        "image": "example/image:stable", "run-eval": True, "eval-only": True,
    }  # fmt: skip


def public_eval(run, score, tp=8, workers=0):
    return {
        "model": "dsr1", "hardware": "fixture", "framework": "sglang", "precision": "fp8",
        "spec_method": "none", "disagg": bool(workers), "is_multinode": False, "prefill_tp": tp,
        "prefill_ep": 1, "prefill_dp_attention": False, "prefill_num_workers": workers,
        "decode_tp": tp, "decode_ep": 1, "decode_dp_attention": False,
        "decode_num_workers": workers,
        "task": "gsm8k", "conc": 8, "metrics": {"em_strict": score, "n_eff": 1319},
        "run_url": f"https://github.com/example/project/actions/runs/{run}",
    }  # fmt: skip


@pytest.mark.parametrize(("rows", "frozen"), [
    pytest.param([public_eval(42, 0.95), public_eval(99, 0.5), public_eval(42, 0.5, tp=4)], 0.95,
                 id="only-verified-producer-and-current-topology"),
    pytest.param([public_eval(42, 0.95, workers=1)], 0.95, id="single-node-worker-encoding"),
    pytest.param([public_eval(42, 0.95), public_eval(43, 0.5)], None, id="conflicting-producers"),
])  # fmt: skip
def test_baseline_evals_use_artifact_keys_of_verified_producers(monkeypatch, rows, frozen):
    monkeypatch.setattr(api, "fetch", lambda resource, **_: Feed(
        url=f"https://inferencex.semianalysis.com/api/v1/{resource}",
        retrieved_at="2026-09-20T00:00:00Z", sha256="0" * 64, payload=rows,
    ))  # fmt: skip
    matrix = {"evals": [eval_entry(8)]}

    _, evals = reporting.published_evals("example/project", matrix, {42, 43})

    if frozen is None:
        assert evals == []
        return
    [(key, _, _)] = validation.eval_identities(matrix)
    assert [(row.key, row.label, row.score, row.samples) for row in evals] == [
        (identity(key), "c8", frozen, 1319)
    ]
    baseline = reporting.Baseline(
        family=f"configs/nvidia-master.yaml:{KEY}", date="2026-09-19", image="example/image:1",
        goal=reporting.Prose(en="Update the image.", zh="更新镜像。"), sources=[], points=[],
        evals=evals,
    )  # fmt: skip
    attempt = evals[0].model_copy(update={"score": 0.96, "run_id": 7})
    assert "| gsm8k/em_strict · c8 | 96% (+1.00 pp) | 1,319 each |" in reporting.eval_table(
        [attempt], baseline
    )


def test_attempt_rejects_unnumbered_infrastructure_retries():
    record = {
        "kind": "infrastructure-retry", "number": 0, "head": "c" * 40, "image": "example/image:2",
        "run-id": 7, "run-attempt": 2, "status": "queued",
        "change": {"en": "Rerun.", "zh": "重跑。"}, "next": {"en": "Wait.", "zh": "等待。"},
        "benchmarks-expected": 0, "benchmarks-passed": 0, "evals-expected": 0,
        "evals-passed": 0, "points": [],
    }  # fmt: skip
    with pytest.raises(ValueError, match="numbered 1 or 2"):
        reporting.Attempt.model_validate(record)
    assert reporting.Attempt.model_validate({**record, "number": 2}).number == 2
