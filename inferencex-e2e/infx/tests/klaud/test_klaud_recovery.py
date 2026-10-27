import json
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime, timedelta

import pytest

from infx.klaud import __main__ as klaud
from infx.klaud import claims, github, lifecycle
from infx.klaud.models import OwnedCandidate, Ownership, identity
from infx.tests.historical_revision import FAMILY
from infx.tests.klaud.test_klaud_github import (
    DISPATCH,
    EXPIRED,
    GRANT,
    UNEXPECTED_ERROR,
    ended_session_sweep,
    park_comment,
    recovery_pass,
    report_comment,
)

FAMILY_OWNERS = claims.family_owners
CLAIM_SHA = "d" * 40


def http_error(status):
    """A failed gh call whose stderr reports HTTP ``status``."""
    return subprocess.CalledProcessError(1, ["gh", "api"], stderr=f"gh: Request failed (HTTP {status})\n")


def serve_ownership_artifact(monkeypatch, world):
    """Serve planner run 42's ownership artifact, naming the world's candidate."""
    serve = github.items

    def items(repository, path, key=None):
        if path.startswith("actions/artifacts?"):
            return [{"id": 5, "expired": False, "workflow_run": {"id": 42}}]
        return serve(repository, path, key)

    def download(_repository, _artifact, destination):
        (destination / "ownership.json").write_text(
            Ownership(run_id=42, candidates=[world.session.candidate]).model_dump_json(by_alias=True))

    monkeypatch.setattr(github, "items", items)
    monkeypatch.setattr(github, "download_json", download)


def serve_claim(monkeypatch, world, *, listing=None, commit=None):
    """Serve the world's family claim through the real claim reader: its listing raises
    ``listing`` if given, and its commit read raises ``commit`` or returns it."""
    monkeypatch.setattr(claims, "family_owners", FAMILY_OWNERS)
    monkeypatch.setattr(claims, "commits", {})
    serve, read = github.items, github.read
    claim = {"ref": "refs/heads/" + claims.family_name(world.session.candidate),
             "object": {"sha": CLAIM_SHA}}

    def items(repository, path, key=None):
        if path == "git/matching-refs/heads/klaud/claim-":
            if listing:
                raise listing
            return [claim]
        return serve(repository, path, key)

    def reader(repository, path):
        if path == "git/commits/" + CLAIM_SHA:
            if isinstance(commit, Exception):
                raise commit
            return commit
        return read(repository, path)

    monkeypatch.setattr(github, "items", items)
    monkeypatch.setattr(github, "read", reader)


@pytest.mark.parametrize(("listing", "commit"), [
    pytest.param(github.ListingUnavailable("Incomplete GitHub listing"), None, id="claim-listing"),
    pytest.param(None, http_error(502), id="claim-commit"),
])  # fmt: skip
def test_unreadable_family_claims_stay_claimed_while_recovery_resumes_the_rest(
    monkeypatch, tmp_path, capsys, listing, commit
):
    world = ended_session_sweep(monkeypatch, conclusion="failure", sessions=("failure",))
    serve_ownership_artifact(monkeypatch, world)
    serve_claim(monkeypatch, world, listing=listing, commit=commit)

    _, output = recovery_pass(monkeypatch, tmp_path)

    assert world.writes == [GRANT, DISPATCH]
    assert world.released == []
    assert output.read_text() == "active-sessions=1\n"
    assert json.loads((tmp_path / "klaud-recovery.json").read_text()) == [FAMILY]
    printed = capsys.readouterr().out
    assert "::warning::Klaud: family claims unreadable; every claimed family stays claimed.\n" in printed
    assert "::warning::Klaud abandoned-work sweep skipped: owned work unavailable.\n" in printed


def test_a_malformed_family_claim_still_stops_recovery_for_inspection(monkeypatch, tmp_path):
    world = ended_session_sweep(monkeypatch, conclusion="failure", sessions=("failure",))
    serve_ownership_artifact(monkeypatch, world)
    serve_claim(monkeypatch, world, commit={"message": "not an owner"})

    with pytest.raises(github.VerificationError, match="manual inspection required"):
        recovery_pass(monkeypatch, tmp_path)
    assert world.writes == []


@pytest.mark.parametrize(("deleted", "released"), [
    pytest.param(None, True, id="deleted"),
    pytest.param(404, True, id="already-gone"),
    pytest.param(422, True, id="already-deleted"),
    pytest.param(500, False, id="server-error"),
])  # fmt: skip
def test_releasing_a_claim_reuses_its_commit_and_accepts_one_another_pass_deleted(
    monkeypatch, deleted, released
):
    candidate = OwnedCandidate(id=f"{identity(FAMILY)[:16]}-{'2' * 16}", family=FAMILY, base="a" * 40)
    owner = Ownership(run_id=42, candidates=[candidate])
    claim = {"ref": "refs/heads/" + claims.family_name(candidate), "object": {"sha": CLAIM_SHA}}
    reads, writes = [], []
    monkeypatch.setattr(claims, "commits", {})
    monkeypatch.setattr(github, "items", lambda _r, path, _key=None: (
        [claim] if path.startswith("git/matching-refs/heads/klaud/claim-") else pytest.fail(path)))

    def read(_repository, path):
        reads.append(path)
        return {"message": json.dumps(owner.model_dump(by_alias=True))}

    def write(_repository, path, method, _payload=None):
        writes.append((method, path))
        if deleted:
            raise http_error(deleted)

    monkeypatch.setattr(github, "read", read)
    monkeypatch.setattr(github, "write", write)

    assert claims.family_owners("example/project") == [owner]
    if released:
        claims.release_family("example/project", candidate, 42)
    else:
        with pytest.raises(subprocess.CalledProcessError):
            claims.release_family("example/project", candidate, 42)
    assert reads == ["git/commits/" + CLAIM_SHA]
    assert writes == [("DELETE", "git/refs/heads/" + claims.family_name(candidate))]


@pytest.mark.parametrize("comment", [422, 502])
def test_sweep_steps_go_on_past_finished_targets_and_name_a_failed_write(
    monkeypatch, capsys, comment
):
    world = ended_session_sweep(monkeypatch, conclusion="failure", sessions=("failure",))
    candidate = world.session.candidate
    stale = "4" * 16 + "-" + "5" * 16
    serve = github.items

    def items(repository, path, key=None):
        if path.startswith("actions/workflows/e2e-tests.yml/runs?status=queued&"):
            return [{"id": 501, "display_title": f"e2e Test - klaud-42-{candidate.id}",
                     "actor": {"login": "Klaud-Cold"}}]
        if path == "git/matching-refs/heads/klaud/auto-":
            return [*serve(repository, path, key),
                    {"ref": f"refs/heads/klaud/auto-{stale}", "object": {"sha": "e" * 40}}]
        return serve(repository, path, key)

    # The run ended and another pass deleted the lease after this pass listed them.
    failures = {("POST", "actions/runs/501/cancel"): 409,
                ("DELETE", f"git/refs/heads/klaud/recovery-42-{candidate.id}"): 422,
                ("POST", "issues/7/comments"): comment}
    attempted = []

    def write(_repository, path, method, _payload=None):
        attempted.append((method, path))
        if (method, path) in failures:
            raise http_error(failures[method, path])
        return {}

    monkeypatch.setattr(github, "items", items)
    monkeypatch.setattr(github, "write", write)

    owners = claims.family_owners("example/project")
    assert lifecycle.sweep_abandoned("example/project", set(), owners, set()) == set()
    assert world.released == [(candidate.id, 42)]
    assert attempted == [
        ("POST", "actions/runs/501/cancel"),
        ("DELETE", f"git/refs/heads/klaud/recovery-42-{candidate.id}"),
        ("POST", "issues/7/comments"),
        ("DELETE", f"git/refs/heads/klaud/auto-{stale}"),
    ]
    assert capsys.readouterr().out == (
        f"::warning::Klaud abandoned-work sweep: POST issues/7/comments failed (HTTP {comment}).\n")


@pytest.mark.parametrize(("failing", "warning", "blocked"), [
    pytest.param(("DELETE", "lease"), "DELETE git/refs/heads/klaud/recovery-42-{id} failed (HTTP 500)",
                 [], id="later-step-fails"),
    pytest.param(("GET", "pulls?state=open&per_page=100"), "GET pulls failed (HTTP 500)", [FAMILY],
                 id="live-work-unknown"),
])  # fmt: skip
def test_recovery_frees_families_unless_the_sweep_cannot_tell_live_work(
    monkeypatch, tmp_path, capsys, failing, warning, blocked
):
    world = ended_session_sweep(monkeypatch, state="closed", sessions=("failure",))
    candidate = world.session.candidate
    lease = f"git/refs/heads/klaud/recovery-42-{candidate.id}"
    serve, write = github.items, github.write

    def items(repository, path, key=None):
        if failing == ("GET", path):
            raise http_error(500)
        return serve(repository, path, key)

    def flaky(repository, path, method, payload=None):
        if failing == ("DELETE", "lease") and (method, path) == ("DELETE", lease):
            raise http_error(500)
        return write(repository, path, method, payload)

    def mismatch(_session, _now):
        raise github.VerificationError("Candidate ownership mismatch")

    monkeypatch.setattr(github, "items", items)
    monkeypatch.setattr(github, "write", flaky)
    monkeypatch.setattr(lifecycle, "reconcile", mismatch)

    recovery_pass(monkeypatch, tmp_path)

    assert json.loads((tmp_path / "klaud-recovery.json").read_text()) == blocked
    printed = capsys.readouterr().out
    assert f"::warning::Klaud abandoned-work sweep: {warning.format(id=candidate.id)}.\n" in printed
    assert (("DELETE", f"git/refs/heads/klaud/auto-{candidate.id}") in world.writes) is not blocked


def parked_on_live_sweeps(monkeypatch, count):
    """Serve ``count`` candidates of planner run 42, claimed and in its ownership artifact, each
    with its own draft PR, parked by its first session on that PR's still-running final sweep.
    Return GitHub call counts, the (call, path) pairs read and the candidates whose recovery
    lease was taken."""
    world = ended_session_sweep(monkeypatch, conclusion=None, sessions=("success",))
    world.comments.append(park_comment(world, 1))
    others = [OwnedCandidate(id=f"{identity(family)[:16]}-{'2' * 16}", family=family, base="a" * 40)
              for family in (f"configs/nvidia-master.yaml:parked-{n}" for n in range(1, count))]
    candidates = [world.session.candidate, *others]
    monkeypatch.setattr(claims, "family_owners", lambda _r: [
        Ownership(run_id=42, candidates=[candidate]) for candidate in candidates])
    monkeypatch.setattr(github, "download_json", lambda _r, _artifact, destination: (
        destination / "ownership.json").write_text(
            Ownership(run_id=42, candidates=candidates).model_dump_json(by_alias=True)))
    pulls = {candidate.id: {**world.pull, "number": 100 + n,
                            "head": {**world.pull["head"], "ref": f"klaud/auto-{candidate.id}"}}
             for n, candidate in enumerate(others, 1)}
    serve = github.items
    jobs = serve("example/project", "actions/runs/42/jobs?filter=all&per_page=100", "jobs")
    own = next(job for job in jobs if world.session.candidate.id in job["name"])

    def items(repository, path, key=None):
        if path == "actions/runs/42/jobs?filter=all&per_page=100":
            return [*jobs, *({**own, "id": 5000 + n, "name": f"Run candidate {candidate.id} / Klaud Cold"}
                             for n, candidate in enumerate(others, 1))]
        if path.startswith("actions/artifacts?"):
            return [{"id": 5, "expired": False, "workflow_run": {"id": 42}}]
        if path == "pulls?state=open&per_page=100":
            return [world.pull, *pulls.values()]
        for n, candidate in enumerate(others, 1):
            pull = pulls[candidate.id]
            if path.startswith("pulls?state=all&") and candidate.id in path:
                return [pull]
            if path.startswith("actions/workflows/run-sweep.yml/runs?per_page=100&created=") and (
                    candidate.id in path):
                return [{**world.sweep, "id": 900 + n, "pull_requests": [{"number": pull["number"]}]}]
            if path == f"issues/{pull['number']}/comments?per_page=100":
                return [{"id": 70, "user": {"login": "Klaud-Cold"}, "body": (
                    lifecycle.park_marker(42, candidate.id, 1) + json.dumps({"runs": [900 + n]})
                    + "\n-->\nParked.")}]
        return serve(repository, path, key)

    calls, paths, leases = Counter(), [], []
    real = {"read": github.read, "items": items, "write": github.write}

    def counted(name):
        def call(repository, path, *args, **kwargs):
            calls[name] += 1
            paths.append((name, path))
            return real[name](repository, path, *args, **kwargs)
        return call

    for name in real:
        monkeypatch.setattr(github, name, counted(name))
    monkeypatch.setattr(claims, "recovery_lease", lambda _r, _parent, candidate: (
        leases.append(candidate.id) or True))
    return calls, paths, leases, [FAMILY, *(candidate.family for candidate in others)]


def test_a_pass_writes_nothing_and_reads_little_per_candidate_parked_on_live_runs(
    monkeypatch, tmp_path
):
    reads = {}
    for count in (2, 6):
        calls, paths, leases, families = parked_on_live_sweeps(monkeypatch, count)

        summary, output = recovery_pass(monkeypatch, tmp_path)

        assert leases == []
        assert calls["write"] == 0
        assert output.read_text().endswith("active-sessions=0\n")
        assert json.loads((tmp_path / "klaud-recovery.json").read_text()) == sorted(families)
        # Every session of the planner run reuses its first terminal-run read.
        assert paths.count(("read", "actions/runs/42")) == 2
        # Recovery and the abandoned-work sweep share one scan of active planner runs.
        assert sum(path.startswith("actions/workflows/klaud-plan.yml/runs?status=")
                   for _, path in paths) == len(lifecycle.ACTIVE_STATUSES)
        reads[count] = calls["read"] + calls["items"]
    assert summary.read_text().count(
        "pending / 等待后续收尾 (Owned work is still running; retained for the next recovery "
        "pass)") == 2 + 6
    # Each more candidate lists only its own PR, PR comments and final sweep.
    assert (reads[6] - reads[2]) / 4 <= 3


def rerun_comment(world, number, due="2026-10-15T01:00:00Z"):
    """A Klaud comment recording that session ``number`` parked for a rerun of run 9's
    failed attempt 1 after ``due``."""
    record = {"runs": [], "rerun": {"run": 9, "attempt": 1, "not-before": due}}
    return {"id": 70 + number, "user": {"login": "Klaud-Cold"}, "body": (
        lifecycle.park_marker(42, world.session.candidate.id, number) + json.dumps(record)
        + "\n-->\nParked.")}


def test_recovery_reruns_a_parked_infrastructure_failure_once_when_due_then_resumes_after_it(
    monkeypatch, tmp_path
):
    # Three sessions already expired; the fourth parked for the rerun and must not expire.
    sessions = ("failure",) * (EXPIRED - 1) + ("success",)
    world = ended_session_sweep(monkeypatch, conclusion="failure", sessions=sessions)
    world.comments.append(rerun_comment(world, EXPIRED))
    rerun = ("POST", "actions/runs/9/rerun-failed-jobs")
    leases = []
    monkeypatch.setattr(claims, "recovery_lease", lambda _r, _parent, candidate: (
        leases.append(candidate.id) or True))

    summary, _ = recovery_pass(monkeypatch, tmp_path, now="2026-10-15T00:59:59Z")
    assert world.writes == leases == []
    assert "(Infrastructure rerun is waiting for its requested time)" in summary.read_text()
    fresh = lifecycle.Session("example/project", world.session.parent, world.session.candidate)
    assert lifecycle.status_label(fresh, world.pull) == "klaud-parked"

    recovery_pass(monkeypatch, tmp_path, now="2026-10-15T01:00:00Z")
    assert world.writes == [rerun]
    assert world.dispatches == []

    world.sweep.update(run_attempt=2, status="in_progress", conclusion=None)
    recovery_pass(monkeypatch, tmp_path, now="2026-10-15T02:00:00Z")
    assert world.writes == [rerun]
    assert len(leases) == 1

    world.sweep.update(status="completed", conclusion="failure", updated_at="2026-10-15T03:00:00Z")
    _, output = recovery_pass(monkeypatch, tmp_path, now="2026-10-15T03:00:01Z")
    assert world.writes == [rerun, GRANT, DISPATCH]
    assert world.dispatches[0]["inputs"]["resume-session"] == str(EXPIRED + 1)
    assert world.finished == []


@pytest.mark.parametrize(("now", "outcome"), [
    pytest.param("2026-10-15T00:59:59Z", "Infrastructure rerun is waiting", id="before-its-time"),
    pytest.param("2026-10-15T01:00:00Z", "Reran failed jobs at the requested time", id="on-time"),
    pytest.param("2026-10-16T00:59:59Z", "Reran failed jobs at the requested time",
                 id="before-giving-up"),
    pytest.param("2026-10-16T01:00:00Z", None, id="given-up"),
])  # fmt: skip
def test_a_requested_rerun_waits_for_its_time_and_gives_up_after_a_day(monkeypatch, now, outcome):
    world = ended_session_sweep(monkeypatch, conclusion="failure", sessions=("success",))
    world.comments.append(rerun_comment(world, 1))

    if outcome:
        with pytest.raises(lifecycle.AwaitingRerun, match=outcome):
            lifecycle.reconcile(world.session, lifecycle.utc(now))
    else:
        assert lifecycle.reconcile(world.session, lifecycle.utc(now)) is True
    rerun = [("POST", "actions/runs/9/rerun-failed-jobs")]
    assert world.writes == (rerun if outcome and outcome.startswith("Reran") else [])
    assert world.finished == ([] if outcome else [UNEXPECTED_ERROR])


def park_cli(tmp_path, monkeypatch, world, *args):
    """Run ``park`` with ``args`` in the world's running session 1; return its exit code."""
    (tmp_path / "candidate.json").write_text(world.session.candidate.model_dump_json())
    for name, value in (("GITHUB_REPOSITORY", "example/project"), ("GITHUB_RUN_ID", "42"),
                        ("KLAUD_PARENT_RUN", "42"), ("KLAUD_SESSION", "1"),
                        ("KLAUD_SESSION_STARTED", str(int(time.time()))),
                        ("KLAUD_EVIDENCE", str(tmp_path))):
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(sys, "argv", ["klaud", "park", *args])
    return klaud.main()


@pytest.mark.parametrize("minutes", [15, 240])
def test_park_records_when_recovery_may_rerun_a_reported_failed_run(tmp_path, monkeypatch, minutes):
    world = ended_session_sweep(monkeypatch, conclusion="failure", jobs=["failure", "cancelled"],
                                sessions=(None,))
    world.comments.append(report_comment(world, 9))
    before = datetime.now(UTC)

    assert park_cli(tmp_path, monkeypatch, world, "--rerun-run", "9",
                    "--rerun-after-minutes", str(minutes)) == 0

    after = datetime.now(UTC)
    fresh = lifecycle.Session("example/project", world.session.parent, world.session.candidate)
    record = lifecycle.park_record(fresh, world.pull, 1)
    due = lifecycle.utc(record["rerun"].pop("not-before"))
    assert record == {"runs": [], "rerun": {"run": 9, "attempt": 1}}
    assert before + timedelta(minutes=minutes) <= due <= after + timedelta(minutes=minutes)
    assert json.loads((tmp_path / "session-end.json").read_text()) == {"status": "parked"}


@pytest.mark.parametrize(("conclusion", "attempt", "jobs", "reported", "run", "error"), [
    pytest.param(None, 1, ["failure"], True, 9, "Only an ended owned run", id="still-running"),
    pytest.param("success", 1, ["success"], True, 9, "Only an ended owned run", id="nothing-failed"),
    pytest.param("failure", 1, ["failure"], True, 77, "Only an ended owned run", id="not-owned"),
    pytest.param("failure", lifecycle.MAX_RUN_ATTEMPTS, ["failure"], True, 9,
                 "Only an ended owned run", id="out-of-attempts"),
    pytest.param("failure", 1, ["failure"], False, 9, "Publish an attempt report", id="unreported"),
])  # fmt: skip
def test_park_refuses_a_rerun_recovery_could_not_make(
    monkeypatch, conclusion, attempt, jobs, reported, run, error
):
    world = ended_session_sweep(monkeypatch, conclusion=conclusion, attempt=attempt, jobs=jobs,
                                sessions=(None,))
    if reported:
        world.comments.append(report_comment(world, 9))
    session = lifecycle.Session("example/project", world.session.parent, world.session.candidate)

    with pytest.raises(github.VerificationError, match=error):
        lifecycle.park(session, 1, overdue=False, rerun=(run, lifecycle.utc("2026-10-15T01:00:00Z")))
    assert world.writes == []


@pytest.mark.parametrize("args", [
    pytest.param(["--rerun-run", "9"], id="no-delay"),
    pytest.param(["--rerun-after-minutes", "45"], id="no-run"),
    pytest.param(["--rerun-run", "9", "--rerun-after-minutes", "14"], id="too-soon"),
    pytest.param(["--rerun-run", "9", "--rerun-after-minutes", "241"], id="too-late"),
])  # fmt: skip
def test_park_cli_rejects_an_incomplete_or_out_of_range_rerun(tmp_path, monkeypatch, args):
    world = ended_session_sweep(monkeypatch, conclusion="failure", sessions=(None,))

    with pytest.raises(SystemExit) as exited:
        park_cli(tmp_path, monkeypatch, world, *args)
    assert exited.value.code == 2
    assert world.writes == []
    assert not (tmp_path / "session-end.json").exists()


def mark_unstarted(monkeypatch, runs):
    """Mark the candidate jobs of session runs ``runs`` as agents that never used a tool."""
    serve = github.items

    def items(repository, path, key=None):
        listed = serve(repository, path, key)
        run = path.removeprefix("actions/runs/").partition("/")[0]
        if path.endswith("/jobs?filter=all&per_page=100") and run.isdigit() and int(run) in runs:
            return [{**job, "steps": [*job["steps"], {"name": lifecycle.UNSTARTED_STEP,
                                                      "conclusion": "success"}]} for job in listed]
        return listed

    monkeypatch.setattr(github, "items", items)


@pytest.mark.parametrize(("unstarted", "approved"), [
    pytest.param((), False, id="every-session-expired"),
    pytest.param((4203,), True, id="one-agent-never-used-a-tool"),
])  # fmt: skip
def test_a_session_whose_agent_never_used_a_tool_does_not_expire(monkeypatch, unstarted, approved):
    world = ended_session_sweep(monkeypatch, conclusion="failure", sessions=("failure",) * EXPIRED)
    mark_unstarted(monkeypatch, unstarted)
    later = lifecycle.utc("2026-10-15T02:00:00Z")

    assert lifecycle.sessions_started(world.session) == EXPIRED
    if approved:
        with pytest.raises(lifecycle.ResumeSession):
            lifecycle.reconcile(world.session, later)
    else:
        assert lifecycle.reconcile(world.session, later) is True
    assert world.finished == ([] if approved else [UNEXPECTED_ERROR])


@pytest.mark.parametrize(("sessions", "unstarted", "parked", "minutes"), [
    pytest.param(("failure",), (), False, 10, id="after-one"),
    pytest.param(("failure",) * 2, (), False, 30, id="after-two-in-a-row"),
    pytest.param(("failure",) * 3, (), False, 90, id="after-three-in-a-row"),
    pytest.param(("failure",) * 4, (4203, 4204), False, 90, id="after-four-whose-agents-never-ran"),
    pytest.param(("failure", "failure", "success"), (), True, 0, id="none-after-a-clean-park"),
])  # fmt: skip
def test_resumes_back_off_after_sessions_that_ended_without_parking(
    monkeypatch, sessions, unstarted, parked, minutes
):
    world = ended_session_sweep(monkeypatch, conclusion="failure", sessions=sessions)
    world.sweep["updated_at"] = "2026-10-14T05:00:00Z"
    mark_unstarted(monkeypatch, unstarted)
    if parked:
        world.comments.append(park_comment(world, len(sessions)))
    ended = lifecycle.utc(lifecycle.candidate_jobs(world.session)[-1]["completed_at"])
    due = ended + timedelta(minutes=minutes)

    if minutes:
        with pytest.raises(lifecycle.PendingCleanup, match="backing off"):
            lifecycle.reconcile(world.session, due - timedelta(seconds=1))
    with pytest.raises(lifecycle.ResumeSession):
        lifecycle.reconcile(world.session, due)
    assert world.writes == world.finished == []
