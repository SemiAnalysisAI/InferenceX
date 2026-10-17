import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from infx.klaud import __main__ as klaud
from infx.klaud import api, claims, github, lifecycle, reporting, validation
from infx.klaud.models import CandidateOutcome, Feed, OwnedCandidate, Ownership, identity
from infx.tests.historical_revision import (
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
    monkeypatch.setattr(sys, "argv", ["klaud", "recover"])
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


def test_cooldown_skips_recent_blocks_across_bases_and_retries_capacity_deferrals(
    monkeypatch,
):
    blocked, deferred, unreadable, superseded, untried, not_started, weightless = (
        f"{digit * 16}-{digit * 16}" for digit in "1234567"
    )
    outcomes = {
        1: ("readiness-blocked", "targeted", None),
        2: ("capacity-deferred", "targeted", None),
        4: ("capacity-deferred", "targeted", None),
        5: ("failed", "targeted", None),
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
            41: [(1, blocked), (3, unreadable), (5, superseded), (91, "plan")],
            42: [(2, deferred), (4, superseded), (7, not_started), (8, weightless), (92, "plan")],
            40: [(6, untried)],
        }
        run_id = int(path.split("/")[2])
        return [
            {"id": artifact, "expired": False,
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


def test_capacity_aborts_only_above_200_percent_with_queued_nodes():
    now = "2026-10-15T00:05:00Z"

    def metric(name, allocated, queued, schedulable):
        return {"clusterLabel": f"cluster:{name}", "allocatedNodes": allocated,
                "queuedNodes": queued, "schedulableNodes": schedulable}

    def feed(snapshot_at, metrics):
        return Feed(
            url="https://dash.inferencex.semianalysis.com/api/status/scheduler",
            retrieved_at=now, sha256="0" * 64,
            payload={"kind": "inferencex.status.scheduler", "generatedAt": now, "data": {
                "available": True,
                "snapshot": {"snapshotAt": snapshot_at, "stalled": False,
                             "packingMetrics": metrics},
            }},
        )

    metrics = [
        metric("at-200", 8, 12, 10), metric("over-200", 8, 13, 10),
        metric("idle", 0, 0, 4), metric("drained", 0, 0, 0), metric("corrupt", 1, -1, 10),
    ]
    policy = api.Policy()
    fresh, stale = "2026-10-15T00:00:00Z", "2026-10-14T23:59:59Z"
    assert api.available_clusters(feed(fresh, metrics), policy, api.utc(now)) == {"at-200", "idle"}
    assert api.available_clusters(feed(stale, metrics), policy, api.utc(now)) == set()
    pre_deploy = [{k: v for k, v in m.items() if k != "queuedNodes"} for m in metrics]
    assert api.available_clusters(feed(fresh, pre_deploy), policy, api.utc(now)) == set()


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
    names = ["dead", "session", "sweeping", "handoff", "human", "ready", "leftover",
             "closed-by-human", "stale-claim", "live-claim"]  # fmt: skip
    ids = {name: f"{digit * 16}-{digit * 16}" for name, digit in zip(names, "123456789a")}

    def pull(name, number, *, draft=True, labels=()):
        return {
            "number": number, "draft": draft, "user": {"login": "Klaud-Cold"},
            "head": {"ref": f"klaud/auto-{ids[name]}", "repo": {"full_name": "example/project"}},
            "labels": [{"name": label} for label in labels], "created_at": "2026-10-15T00:00:00Z",
            "merged_at": None,
        }  # fmt: skip

    open_pulls = [
        pull("dead", 1), pull("session", 2), pull("sweeping", 3),
        pull("handoff", 4, labels=["klaud-handoff"]), pull("human", 5), pull("ready", 6, draft=False),
    ]  # fmt: skip
    closed = {"leftover": (7, "Klaud-Cold"), "closed-by-human": (8, "maintainer")}
    smoke = {"dead": 501, "handoff": 502}

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

    assert live == {ids[name] for name in
                    ("session", "sweeping", "handoff", "human", "ready")}
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


def test_select_continues_after_one_baseline_state_failure(tmp_path, monkeypatch):
    first_id = "1" * 16 + "-" + "2" * 16
    second_id = "3" * 16 + "-" + "4" * 16
    base = "a" * 40
    families = [
        "configs/nvidia-master.yaml:first-family",
        "configs/nvidia-master.yaml:second-family",
    ]
    contexts = [
        {"id": candidate_id, "family": family, "base": base, "source": {}, "clusters": ["cluster-a"]}
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

    def resolve(_repository, candidate, _context, _model, _goal, _producers):
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
    monkeypatch.setattr(klaud, "fetch_capacity", lambda _policy: {"cluster-a"})
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


@pytest.mark.parametrize(("eligible", "selected", "deferred"), [
    ({"h100-dsxe"}, ["exact"], ["spanning"]),
    ({"h100-cw", "h100-dsxe"}, ["spanning", "exact"], []),
])  # fmt: skip
def test_select_needs_every_planner_target_cluster_eligible(
    tmp_path, monkeypatch, eligible, selected, deferred
):
    ids = {"spanning": "1" * 16 + "-" + "2" * 16, "exact": "3" * 16 + "-" + "4" * 16}
    targets = {"spanning": ["h100-cw", "h100-dsxe"], "exact": ["h100-dsxe"]}
    contexts = [
        {"id": ids[name], "family": f"configs/nvidia-master.yaml:{name}", "base": "a" * 40,
         "source": {}, "clusters": targets[name]}
        for name in ("spanning", "exact")
    ]  # fmt: skip
    (tmp_path / "candidates.json").write_text(json.dumps(contexts))
    (tmp_path / "producers.json").write_text("{}\n")
    review = {"decisions": [
        {"candidate-id": context["id"], "decision": "proceed", "family": context["family"],
         "pull-requests": [], "baseline-model": "Model", "reason": "No overlap"}
        for context in contexts
    ]}  # fmt: skip

    def resolve(_repository, candidate, _context, _model, _goal, _producers):
        return reporting.Baseline(
            family=candidate.family, date="2026-09-19", image="example/image:1",
            goal=reporting.Prose(en="Check the baseline.", zh="检查基线。"),
            sources=["https://inferencex.semianalysis.com/api/v1/benchmarks"], points=[],
        )  # fmt: skip

    monkeypatch.setenv("GITHUB_REPOSITORY", "example/project")
    monkeypatch.setenv("GITHUB_RUN_ID", "42")
    monkeypatch.setenv("KLAUD_PR_REVIEW", json.dumps(review))
    monkeypatch.setattr(klaud, "fetch_capacity", lambda _policy: eligible)
    monkeypatch.setattr(reporting, "resolve_baseline", resolve)
    monkeypatch.setattr(validation, "canonical_matrix", lambda *_: {"single_node": {"all": []}})
    monkeypatch.setattr(claims, "claim_family", lambda *_args: True)

    klaud.select(tmp_path, 5)

    selection = json.loads((tmp_path / "selection.json").read_text())
    assert selection["candidates"] == [ids[name] for name in selected]
    assert selection["capacity-deferred-candidates"] == [ids[name] for name in deferred]


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


def test_baseline_normalizes_enroot_image_and_rejects_unverified_provenance(
    monkeypatch,
):
    base = "a" * 40
    historical_head = "b" * 40
    raw_image = "nvcr.io#nvidia/trtllm:1"
    public_image = "nvcr.io/nvidia/trtllm:1"
    entry = {
        "model-prefix": "Model",
        "runner": "h200",
        "framework": "trtllm",
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
        "framework": "trtllm",
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
        "recipe_fingerprint": "fingerprint",
        "run_url": "https://github.com/example/project/actions/runs/42",
        "tput_per_gpu": 10.0,
        "output_tput_per_gpu": 8.0,
        "mean_ttft": 0.1,
        "mean_tpot": 0.02,
        "errors": 0,
    }
    info = {
        "runs": [{"github_run_id": "42", "run_attempt": 1}],
        "runConfigs": [{"github_run_id": "42", "head_sha": historical_head}],
        "changelogs": [{"workflow_run_id": "42", "config_keys": ["test-family"]}],
    }

    def canonical_matrix(_repository, head, _family):
        assert head == base
        return current

    feeds = {
        "benchmarks": [
            {**public_row, "image": None},
            public_row,
        ],
        "workflow-info": info,
    }

    def fetch(resource, **_kwargs):
        return Feed(
            url=f"https://inferencex.semianalysis.com/api/v1/{resource}",
            retrieved_at="2026-09-20T00:00:00Z",
            sha256="0" * 64,
            payload=feeds[resource],
        )

    monkeypatch.setattr(validation, "canonical_matrix", canonical_matrix)
    monkeypatch.setattr(api, "fetch", fetch)
    producers = {historical_head: historical_matrix}
    candidate = OwnedCandidate(
        id="1" * 16 + "-" + "2" * 16,
        family="configs/nvidia-master.yaml:test-family",
        base=base,
    )
    context = {
        "source": {
            "date": "2026-09-19",
            "image": public_image,
            "model": "Model",
            "hardware": "h200",
            "framework": "trtllm",
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


@pytest.fixture
def family_configs():
    master = {
        "fixture": {
            "image": "example/image:stable",
            "model": "example/model",
            "model-prefix": "dsr1",
            "precision": "fp8",
            "framework": "sglang",
            "runner": "fixture",
            "multinode": False,
            "srt-recipe-dir": "fixture",
            "scenarios": {
                "fixed-seq-len": [{
                    "isl": 1024, "osl": 1024,
                    "search-space": [{"tp": 1, "conc-list": [2, 6], "srt-recipe": "recipe.yaml"}],
                }],
            },
        },
    }
    runners = {
        "labels": {"fixture": ["node-a"], "cluster:fixture": ["node-a"]},
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
        assert matrix["evals"] == []
        return matrix["single_node"]["all"]

    rows = family()
    assert [(row["model"], row["conc"], row["image"]) for row in rows] == [
        ("example/model", 2, "example/image:stable"),
        ("example/model", 6, "example/image:stable"),
    ]
    block = prefix + "configs/srt-recipes/fixed-sequence-single.yaml"
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
            "model": "dsr1", "hardware": "fixture", "framework": "sglang", "precision": "fp8",
            "spec_method": "none", "disagg": False, "is_multinode": False,
            "benchmark_type": "single_turn", "isl": 8192, "osl": 1024, "offload_mode": "off",
            "conc": conc, "image": image, "prefill_tp": 1, "prefill_ep": 1,
            "prefill_dp_attention": False, "prefill_num_workers": 0, "decode_tp": 1,
            "decode_ep": 1, "decode_dp_attention": False, "decode_num_workers": 0,
            "run_url": "https://github.com/example/project/actions/runs/42",
            "tput_per_gpu": 5.0 * conc, "output_tput_per_gpu": 4.0 * conc, "mean_ttft": 0.1,
            "mean_tpot": 0.02, "errors": 0,
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
          "clusters": ["cluster-a"]}]
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
    _, head = commit_history(tmp_path, "inferencex-e2e")
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
    monkeypatch.setattr(klaud, "fetch_capacity", lambda _policy: {"cluster-a"})
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


def test_select_defers_a_baseline_the_current_family_no_longer_generates(tmp_path, monkeypatch):
    _, head = commit_history(tmp_path, "inferencex-e2e")
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

    monkeypatch.setattr(klaud, "fetch_capacity", lambda _policy: {"cluster-a"})
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
            "mean_tpot": 0.02, "errors": 0,
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
    monkeypatch.setattr(klaud, "fetch_capacity", lambda _policy: {"cluster-a"})
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
    producers, source = serve_dsr1_baseline(monkeypatch, [dsr1_point(8192, [1, 2])], models_md())
    return reporting.resolve_baseline(
        "example/project", OwnedCandidate(id=CANDIDATE_ID, family=FAMILY, base="a" * 40),
        {"source": source}, "DeepSeek-R1", reporting.Prose(en="Update the image.", zh="更新镜像。"),
        producers,
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


def finish_report(monkeypatch, outcome, comments, recovering=False):
    session = lifecycle.Session(
        "example/project", {"id": 42}, OwnedCandidate(id=CANDIDATE_ID, family=FAMILY, base="a" * 40),
        recovering=recovering,
    )
    validated = outcome == "validated"
    pull = {"number": 7, "head": {"sha": "c" * 40}, "merged_at": None, "draft": not validated,
            "state": "open" if validated else "closed",
            "labels": [{"name": "full-sweep-enabled"}] if validated else []}
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
        run_ids=[9], repairs_used=0,
    ))

    [report] = [body for body in posted if body.startswith(session.marker)]
    return report.split("\n-->\n", 1)[1]


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
        "validated": ("Full sweep verified; ready for review.", "完整 sweep 已验证；已就绪，等待审查。"),
    }[outcome]
    why = (f"**Reason:** {latest[0]}  \n", f"**原因：**{latest[1]}  \n") if shown else ("", "")

    assert finish_report(monkeypatch, outcome, comments, recovering) == (
        f"**{outcome}** · Repairs: 0 · Runs: {runs}  \n{why[0]}All owned runs ended. {done[0]}"
        "\n\n<details>\n<summary>中文</summary>\n\n"
        f"**{outcome}** · 修复次数：0 · 运行：{runs}  \n{why[1]}所有自有运行均已结束。{done[1]}"
        "\n\n</details>"
    )


def test_producer_outside_the_local_clone_is_fetched_from_its_repository(tmp_path, monkeypatch):
    remote = tmp_path / "remote"
    remote.mkdir()
    base, head = commit_history(remote)

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

    commit_history(tmp_path)
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
        ("dsr1", "fixture", "sglang", "fp8", "none", False, "single_turn",
         1024, 1024, "example/image:stable"): {"configs/nvidia-master.yaml:fixture": {"fixture"}},
    }


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
        ("dsr1", "fixture", "sglang", "fp8", "none", False, "single_turn",
         1024, 1024, "example/image:stable"): {"configs/nvidia-master.yaml:fixture": {"fixture"}},
    }


def test_choose_needs_every_target_cluster_of_a_family_and_skips_unresolved_runners(
    tmp_path, family_configs, capsys
):
    master, runners = family_configs
    fixture, record = master["fixture"], runners["clusters"]["fixture"]
    master = {
        "spanning": {**fixture, "runner": "h100"},
        "exact": {**fixture, "runner": "cluster:h100-dsxe"},
        "unmapped": {**fixture, "runner": "h100-retired"},
    }
    runners = {
        "labels": {
            "h100": ["h100-cw_00", "h100-dsxe_00"],
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

    families = klaud.live_families(tmp_path)

    assert capsys.readouterr().out == (
        "::warning::Klaud live-family-clusters-unresolved: configs/nvidia-master.yaml:unmapped\n"
    )
    assert families == {
        ("dsr1", "h100", "sglang", "fp8", "none", False, "single_turn",
         1024, 1024, "example/image:stable"): {
            "configs/nvidia-master.yaml:spanning": {"h100-cw", "h100-dsxe"},
            "configs/nvidia-master.yaml:exact": {"h100-dsxe"},
        },
    }  # fmt: skip
    source = {
        "model": "dsr1", "hardware": "h100", "framework": "sglang", "precision": "fp8",
        "spec_method": "none", "disagg": False, "benchmark_type": "single_turn", "isl": 1024,
        "osl": 1024, "image": "example/image:stable", "date": "2026-09-19",
    }  # fmt: skip
    items = [{"source-status": "baseline", "source-id": "1", "source": source}]

    def chosen(available):
        return sorted(
            (candidate["family"].removeprefix("configs/nvidia-master.yaml:"), candidate["clusters"])
            for candidate in klaud.choose(items, available, set(), families)
        )

    assert chosen({"h100-cw"}) == []
    assert chosen({"h100-dsxe"}) == [("exact", ["h100-dsxe"])]
    assert chosen({"h100-cw", "h100-dsxe"}) == [
        ("exact", ["h100-dsxe"]), ("spanning", ["h100-cw", "h100-dsxe"]),
    ]
