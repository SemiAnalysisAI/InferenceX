"""Finish owned sessions; resume or reconcile interrupted ones in the next autosweep."""

from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode

from . import claims, github
from .github import VerificationError
from .models import ABORT_LABEL, CandidateOutcome, OwnedCandidate, Ownership, identity, utc
from .reporting import Prose, latest_next, translated
from .validation import verify_sweep

BOT = "Klaud-Cold"
COMMITTER = "github-actions[bot]"
ACTIVE_STATUSES = ("requested", "waiting", "pending", "queued", "in_progress")
CANDIDATE_ID = r"[0-9a-f]{16}-[0-9a-f]{16}"
SWEEP_LABELS = {
    "full-sweep-enabled",
    "non-canary-full-sweep-enabled",
    "full-sweep-fail-fast",
    "all-evals",
    "evals-only",
    "agentx-fast",
}
# Leaves the scheduler time to rerun what it preempted before recovery reruns it.
RERUN_GRACE_SECONDS = 30 * 60
# Recovery stops waiting for a preempted final sweep that stayed cancelled this long.
RERUN_GIVE_UP_SECONDS = 24 * 60 * 60
MAX_SWEEP_ATTEMPTS = 10
# Sessions per candidate: its planner run's session plus three resume runs.
MAX_SESSIONS = 4
# Planner waves start this often (klaud-plan.yml's schedule).
WAVE_HOURS = 6
# Recovery stops resuming, or keeping unreconciled sessions, this long after the last ended.
RESUME_GIVE_UP_SECONDS = 24 * 60 * 60
# The klaud-candidate.yml step that lets a resume run start only a session recovery granted.
GATE_STEP = "Check whether this session should run"
# The klaud-candidate.yml step that runs the session's agent.
AGENT_STEP = "Let Klaud Cold own the candidate PR"
# GitHub concludes a job stopped at its timeout-minutes as cancelled, with this annotation.
TIMEOUT_ANNOTATION = "exceeded the maximum execution time"


def resume_title(parent: int, candidate: str, session: int) -> str:
    """The run name klaud-plan.yml gives the resume run of ``candidate``'s ``session``."""
    return f"Klaud Cold resume {parent} {candidate} session {session}"


def resume_parent(title: str) -> int | None:
    """The planner run whose session a resume run named ``title`` continues, else None."""
    match = re.fullmatch(rf"Klaud Cold resume (\d+) {CANDIDATE_ID} session [1-9][0-9]*", title)
    return int(match[1]) if match else None


class PendingCleanup(VerificationError):  # noqa: N818
    """GitHub has accepted a transition but its child jobs are not terminal yet."""


class AwaitingRerun(PendingCleanup):
    """A preempted final sweep will run again; keep its PR, branch, claim and lease."""


class ResumeSession(PendingCleanup):
    """Recovery may resume this unfinished session; keep its PR, branch, claim and lease."""

    def __init__(self, pull: dict) -> None:
        super().__init__("Unfinished session approved to resume in a new run")
        self.pull = pull


def terminal(run: dict) -> bool:
    return run["status"] == "completed" and bool(run.get("conclusion"))


class Session:
    def __init__(
        self,
        repository: str,
        parent: dict,
        candidate: OwnedCandidate,
        *,
        recovering: bool = False,
    ) -> None:
        self.repository, self.parent, self.candidate = repository, parent, candidate
        self.recovering = recovering
        self.branch = f"klaud/auto-{candidate.id}"
        self.marker = f"<!-- klaud-outcome:{parent['id']}:{candidate.id}\n"
        self.validations = {}
        self.first_attempt_end: datetime | None = None

    def validation(self, run: dict) -> tuple[dict, list[dict], list[dict]]:
        key = (run["id"], run["head_sha"], run["run_attempt"])
        if key not in self.validations:
            from .reporting import baseline_for, check_baseline_coverage

            evidence = verify_sweep(self.repository, run, self.candidate.family)
            check_baseline_coverage(
                self.repository, self.candidate, evidence[0], baseline_for(self, self.pulls()[0])
            )
            self.validations[key] = evidence
        return self.validations[key]

    def pull_window_end(self) -> datetime | None:
        """When the planner run's first attempt ended, or None while it runs.

        Only a planner run's first session opens its candidate PR; later sessions resume it.
        """
        if self.parent["run_attempt"] == 1:
            return utc(self.parent["updated_at"]) if terminal(self.parent) else None
        if self.first_attempt_end is None:
            first = github.read(self.repository, f"actions/runs/{self.parent['id']}/attempts/1")
            self.first_attempt_end = utc(first["updated_at"])
        return self.first_attempt_end

    def pulls(self) -> list[dict]:
        query = urlencode(
            {
                "state": "all",
                "head": self.repository.split("/")[0] + ":" + self.branch,
                "per_page": 100,
            }
        )
        end = self.pull_window_end()
        pulls = [
            pull
            for pull in github.items(self.repository, "pulls?" + query)
            if utc(pull["created_at"]) >= utc(self.parent["created_at"])
            and (end is None or utc(pull["created_at"]) <= end)
        ]
        if len(pulls) > 1:
            raise VerificationError("Ambiguous candidate PR")
        if pulls:
            pull = pulls[0]
            if pull["user"]["login"] != BOT or pull["head"]["repo"]["full_name"] != self.repository:
                raise VerificationError("Candidate ownership mismatch")
        return pulls

    def resume_runs(self) -> list[dict]:
        """Recovery's resume runs of this candidate's session, oldest first."""
        prefix = resume_title(self.parent["id"], self.candidate.id, 0).removesuffix("0")
        # A session resumes within RESUME_GIVE_UP_SECONDS of the last one ending, and each
        # session lasts at most a GitHub job's 6 hours, so every resume run starts by then.
        until = utc(self.parent["created_at"]) + MAX_SESSIONS * timedelta(
            seconds=RESUME_GIVE_UP_SECONDS, hours=6
        )
        runs = github.items(
            self.repository,
            "actions/workflows/klaud-plan.yml/runs?"
            + urlencode(
                {
                    "event": "workflow_dispatch",
                    "actor": BOT,
                    "created": f"{self.parent['created_at']}..{until:%Y-%m-%dT%H:%M:%SZ}",
                    "per_page": 100,
                }
            ),
            "workflow_runs",
        )
        return sorted(
            (
                run
                for run in runs
                if run["display_title"].startswith(prefix)
                and run["display_title"].removeprefix(prefix).isdigit()
                and run["head_branch"] == "main"
                and run["actor"]["login"] == BOT
            ),
            key=lambda run: run["created_at"],
        )

    def last_session_end(self) -> str | None:
        """When this candidate's latest session run ended, or None while one is active."""
        runs = [self.parent, *self.resume_runs()]
        if not all(terminal(run) for run in runs):
            return None
        return max((run["updated_at"] for run in runs), key=utc)

    def runs(self) -> list[dict]:
        ended = self.last_session_end()
        end = ended or "*"
        query = {"per_page": 100, "created": f"{self.parent['created_at']}..{end}"}
        targeted = github.items(
            self.repository,
            "actions/workflows/e2e-tests.yml/runs?"
            + urlencode({**query, "event": "workflow_dispatch"}),
            "workflow_runs",
        )
        title = f"e2e Test - klaud-{self.parent['id']}-{self.candidate.id}"
        targeted = [
            run
            for run in targeted
            if run["display_title"] == title
            and run["actor"]["login"] == BOT
            and run["head_repository"]["full_name"] == self.repository
        ]
        sweeps = github.items(
            self.repository,
            "actions/workflows/run-sweep.yml/runs?"
            + urlencode(
                {
                    "per_page": 100,
                    "created": ">=" + self.parent["created_at"],
                    "event": "pull_request",
                    "branch": self.branch,
                }
            ),
            "workflow_runs",
        )
        pulls = self.pulls()
        sweeps = [
            run
            for run in sweeps
            if pulls
            and run["head_repository"]["full_name"] == self.repository
            and utc(run["created_at"]) >= utc(pulls[0]["created_at"])
            and (
                not run.get("pull_requests")
                or any(pr["number"] == pulls[0]["number"] for pr in run["pull_requests"])
            )
        ]
        if self.recovering and any(
            ended and utc(run["created_at"]) > utc(ended) and run["actor"]["login"] != BOT
            for run in sweeps
        ):
            raise VerificationError("Maintainer started runs after the session; leave them intact")
        return targeted + sweeps

    def aborted(self, pull: dict) -> bool:
        return any(label["name"] == ABORT_LABEL for label in pull["labels"])

    def refresh(self, pull: dict) -> dict:
        self.check_parent()
        current = github.read(self.repository, f"pulls/{pull['number']}")
        if (
            current["head"]["sha"] != pull["head"]["sha"]
            or current["merged_at"]
            or current["user"]["login"] != BOT
        ):
            raise VerificationError("Ownership changed; leave maintainer work intact")
        return current

    def check_parent(self) -> None:
        if self.recovering and (
            not terminal(github.read(self.repository, f"actions/runs/{self.parent['id']}"))
            or not all(terminal(run) for run in self.resume_runs())
        ):
            raise VerificationError("Session resumed; leave the active session intact")

    def report(self, pull: dict) -> dict | None:
        comments = github.items(self.repository, f"issues/{pull['number']}/comments?per_page=100")
        matches = [
            comment
            for comment in comments
            if comment["user"]["login"] == BOT and comment["body"].startswith(self.marker)
        ]
        if not matches:
            return None
        body = max(matches, key=lambda c: c["id"])["body"]
        return json.loads(body[len(self.marker) :].split("\n-->", 1)[0])

    def pending(self, pull: dict) -> CandidateOutcome | None:
        marker = self.marker.replace("klaud-outcome:", "klaud-cleanup:")
        comments = github.items(self.repository, f"issues/{pull['number']}/comments?per_page=100")
        matches = [
            c for c in comments if c["user"]["login"] == BOT and c["body"].startswith(marker)
        ]
        if not matches:
            return None
        body = max(matches, key=lambda c: c["id"])["body"]
        record = json.loads(body[len(marker) :].split("\n-->", 1)[0])
        if record["head"] != pull["head"]["sha"]:
            raise VerificationError("PR head changed after cleanup was requested")
        outcome = CandidateOutcome.model_validate(record["outcome"])
        if outcome.outcome == "validated":
            raise VerificationError("Invalid pending cleanup category")
        return outcome

    def branch_exists(self) -> bool:
        refs = github.items(self.repository, "git/matching-refs/heads/" + self.branch)
        return any(ref["ref"] == "refs/heads/" + self.branch for ref in refs)

    def delete_branch(self, pull: dict) -> None:
        """Delete only this session's unchanged exact-head candidate branch."""
        if not self.branch_exists():
            return
        self.refresh(pull)
        ref = github.read(self.repository, "git/ref/heads/" + self.branch)
        if ref["object"]["sha"] != pull["head"]["sha"]:
            raise VerificationError("Branch moved during cleanup")
        github.write(self.repository, "git/refs/heads/" + self.branch, "DELETE")

    def verify(
        self,
        outcome: CandidateOutcome,
        *,
        require_report: bool = True,
        require_ready: bool = True,
        require_branch_deleted: bool = True,
    ) -> None:
        pulls = self.pulls()
        pull = pulls[0] if pulls else None
        if pull and pull["merged_at"]:
            raise VerificationError("Maintainer owns the PR")
        runs = self.runs()
        if any(not terminal(run) for run in runs):
            raise PendingCleanup("Owned jobs are unfinished")
        if set(outcome.run_ids) != {run["id"] for run in runs}:
            raise VerificationError("Outcome omits or invents owned runs")
        if outcome.pull_request != (pull["number"] if pull else None):
            raise VerificationError("Outcome PR mismatch")
        if outcome.outcome == "validated":
            if not pull or pull["state"] != "open" or (require_ready and pull["draft"]):
                raise VerificationError("Validated PR must remain ready for review")
            # Klaud runs every final sweep fail-fast; any other sweep label is invalid.
            sweep_labels = {label["name"] for label in pull["labels"]} & SWEEP_LABELS
            if sweep_labels != {"full-sweep-fail-fast"}:
                raise VerificationError("Validated PR has incompatible sweep labels")
            finals = [
                run
                for run in runs
                if run["event"] == "pull_request"
                and run["head_sha"] == pull["head"]["sha"]
                and run["conclusion"] != "skipped"
            ]
            if not finals:
                raise VerificationError("No exact-head final sweep")
            latest = max(finals, key=lambda run: run["created_at"])
            if latest["conclusion"] != "success":
                raise VerificationError("Latest exact-head final sweep failed")
            receipt = self.report(pull) if require_report else None
            proof = {
                "head": pull["head"]["sha"],
                "run-id": latest["id"],
                "run-attempt": latest["run_attempt"],
            }
            if receipt and receipt.get("validation") == proof:
                inventory = github.artifacts(self.repository, latest["id"])
                if not any(
                    a["name"] == "klaud-sweep-manifest" and not a["expired"] for a in inventory
                ):
                    raise VerificationError("Verified sweep artifacts have expired")
            if not receipt or receipt.get("validation") != proof:
                self.validation(latest)
        else:
            if pull and (
                pull["state"] != "closed"
                or not pull["draft"]
                or any(label["name"] in SWEEP_LABELS for label in pull["labels"])
            ):
                raise VerificationError("Failure/deferral PR cleanup is incomplete")
            if pull and require_branch_deleted and self.branch_exists():
                raise VerificationError("Incorrect cleanup branch disposition")
        if pull and require_report:
            receipt = self.report(pull)
            if (
                not receipt
                or receipt.get("head") != pull["head"]["sha"]
                or receipt["outcome"] != outcome.model_dump(by_alias=True, exclude_unset=True)
            ):
                raise VerificationError("Missing verified completion report")

    def finish(self, outcome: CandidateOutcome) -> CandidateOutcome:
        pulls = self.pulls()
        pull = pulls[0] if pulls else None
        runs = self.runs()
        if pull and pull["merged_at"]:
            raise VerificationError("Merged PR belongs to maintainers")
        outcome = outcome.model_copy(
            update={
                "pull_request": pull["number"] if pull else None,
                "run_ids": sorted(run["id"] for run in runs),
            }
        )
        if outcome.outcome == "validated" and pull:
            self.verify(outcome, require_report=False, require_ready=False)
            from .reporting import publish_final

            final = max(
                (
                    run
                    for run in runs
                    if run["event"] == "pull_request"
                    and run["head_sha"] == pull["head"]["sha"]
                    and run["conclusion"] != "skipped"
                ),
                key=lambda run: run["created_at"],
            )
            publish_final(self, final, self.validation(final))
            current = self.refresh(pull)
            if current["draft"]:
                subprocess_ready(self.repository, pull["number"], undo=False)
        if outcome.outcome != "validated":
            if pull and not self.report(pull):
                self.refresh(pull)
                if not self.pending(pull):
                    pending = self.marker.replace("klaud-outcome:", "klaud-cleanup:")
                    request = {
                        "head": pull["head"]["sha"],
                        "outcome": outcome.model_dump(by_alias=True, exclude_unset=True),
                    }
                    body = (
                        pending
                        + json.dumps(request)
                        + "\n-->\n"
                        + translated(
                            f"**{outcome.outcome}** · Cleanup pending. Stop and confirm owned runs before closing.",
                            f"**{outcome.outcome}** · 清理待完成。先停止并确认自有运行结束，再关闭 PR。",
                        )
                    )
                    github.write(
                        self.repository,
                        f"issues/{pull['number']}/comments",
                        "POST",
                        {"body": body},
                    )
            for run in runs:
                if not terminal(run):
                    self.check_parent()
                    if pull:
                        self.refresh(pull)
                    github.write(self.repository, f"actions/runs/{run['id']}/cancel", "POST")
            if any(not terminal(run) for run in self.runs()):
                raise PendingCleanup(
                    "Cancellation requested; wait for terminal jobs then finish again"
                )
            if pull:
                current = self.refresh(pull)
                for label in current["labels"]:
                    if label["name"] in SWEEP_LABELS:
                        self.refresh(pull)
                        github.write(
                            self.repository,
                            f"issues/{pull['number']}/labels/{label['name']}",
                            "DELETE",
                        )
                current = self.refresh(pull)
                if not current["draft"]:
                    subprocess_ready(self.repository, pull["number"])
                self.refresh(pull)
                if current["state"] != "closed":
                    github.write(
                        self.repository,
                        f"pulls/{pull['number']}",
                        "PATCH",
                        {"state": "closed"},
                    )
                self.delete_branch(pull)
        outcome = outcome.model_copy(update={"run_ids": sorted(run["id"] for run in self.runs())})
        self.verify(outcome, require_report=False)
        if pull:
            self.refresh(pull)
            proof = None
            if outcome.outcome == "validated":
                final = max(
                    (
                        run
                        for run in self.runs()
                        if run["event"] == "pull_request"
                        and run["head_sha"] == pull["head"]["sha"]
                        and run["conclusion"] != "skipped"
                    ),
                    key=lambda run: run["created_at"],
                )
                proof = {
                    "head": pull["head"]["sha"],
                    "run-id": final["id"],
                    "run-attempt": final["run_attempt"],
                }
            record = {
                "head": pull["head"]["sha"],
                "outcome": outcome.model_dump(by_alias=True, exclude_unset=True),
                "validation": proof,
            }
            if self.report(pull) != record:
                links = (
                    ", ".join(
                        f"[{run}](https://github.com/{self.repository}/actions/runs/{run})"
                        for run in outcome.run_ids
                    )
                    or "—"
                )
                repairs = outcome.repairs_used if outcome.repairs_used is not None else "unknown"
                reason_en = f" · Reason: {outcome.reason_code}" if outcome.reason_code else ""
                reason_zh = f" · 原因：{outcome.reason_code}" if outcome.reason_code else ""
                why = (
                    Prose(en="Stopped by the klaud-abort label.", zh="已由 klaud-abort 标签停止。")
                    if outcome.outcome == "aborted"
                    else None
                    if proof or self.recovering
                    else latest_next(self, pull)
                )
                why_en = f"**Reason:** {why.en}  \n" if why else ""
                why_zh = f"**原因：**{why.zh}  \n" if why else ""
                body = (
                    self.marker
                    + json.dumps(record)
                    + "\n-->\n"
                    + translated(
                        f"**{outcome.outcome}**{reason_en} · Repairs: {repairs} · Runs: {links}  \n{why_en}All owned runs ended. "
                        + (
                            "Full sweep verified; ready for review."
                            if proof
                            else "PR closed; branch deleted for retry."
                        ),
                        f"**{outcome.outcome}**{reason_zh} · 修复次数：{repairs} · 运行：{links}  \n{why_zh}所有自有运行均已结束。"
                        + (
                            "完整 sweep 已验证；已就绪，等待审查。"
                            if proof
                            else "PR 已关闭；分支已删除，可重新尝试。"
                        ),
                    )
                )
                github.write(
                    self.repository,
                    f"issues/{pull['number']}/comments",
                    "POST",
                    {"body": body},
                )
        claims.release_family(self.repository, self.candidate, self.parent["id"])
        return outcome


def subprocess_ready(repository: str, number: int, *, undo: bool = True) -> None:
    subprocess.run(
        ["gh", "pr", "ready", str(number), "--repo", repository] + (["--undo"] if undo else []),
        check=True,
        capture_output=True,
        timeout=60,
    )


def current_session() -> Session:
    context = json.loads((Path(os.environ["KLAUD_EVIDENCE"]) / "candidate.json").read_text())
    candidate = OwnedCandidate.model_validate(
        {key: context[key] for key in ("id", "family", "base")}
    )
    repository = os.environ["GITHUB_REPOSITORY"]
    # A resume run continues the session of the planner run that owns the candidate.
    parent = os.environ.get("KLAUD_PARENT_RUN") or os.environ["GITHUB_RUN_ID"]
    return Session(repository, github.read(repository, f"actions/runs/{int(parent)}"), candidate)


def hit_timeout(repository: str, job: dict) -> bool:
    return bool(job.get("runner_name")) and any(
        TIMEOUT_ANNOTATION in (annotation.get("message") or "")
        for annotation in github.items(
            repository, f"check-runs/{job['id']}/annotations?per_page=100"
        )
    )


def preempted(repository: str, run: dict) -> bool:
    """Cancelled by someone else: no job of its current attempt failed or hit its timeout."""
    return run["conclusion"] == "cancelled" and not any(
        job["conclusion"] in ("failure", "timed_out")
        or (job["conclusion"] == "cancelled" and hit_timeout(repository, job))
        for job in github.items(
            repository,
            f"actions/runs/{run['id']}/attempts/{run['run_attempt']}/jobs?per_page=100",
            "jobs",
        )
    )


def await_rerun(session: Session, pull: dict, run: dict, now: datetime) -> None:
    """Keep a preempted final sweep's candidate alive until the sweep runs again."""
    try:
        if not preempted(session.repository, run):
            return
    except (OSError, subprocess.SubprocessError, VerificationError) as error:
        raise AwaitingRerun("Preemption check unavailable; retrying next pass") from error
    if now - utc(run["updated_at"]) < timedelta(seconds=RERUN_GRACE_SECONDS):
        raise AwaitingRerun("Preempted final sweep is waiting for its rerun")
    if run["run_attempt"] >= MAX_SWEEP_ATTEMPTS:
        return
    session.refresh(pull)
    try:
        github.write(session.repository, f"actions/runs/{run['id']}/rerun-failed-jobs", "POST")
    except (OSError, subprocess.SubprocessError) as error:
        raise AwaitingRerun("Rerun request failed; retrying next pass") from error
    raise AwaitingRerun("Reran preempted final sweep")


def unfinished_draft(session: Session, pull: dict | None) -> bool:
    """Whether ``pull`` is the bot's open, unmerged draft from this repository."""
    return bool(
        pull
        and pull["state"] == "open"
        and pull["draft"]
        and not pull["merged_at"]
        and pull["user"]["login"] == BOT
        and (pull["head"]["repo"] or {}).get("full_name") == session.repository
    )


def resumable(session: Session, pull: dict | None) -> bool:
    """Whether a later session may continue this candidate's unfinished draft PR."""
    return (
        unfinished_draft(session, pull)
        and not session.aborted(pull)
        and session.branch_exists()
        and not session.report(pull)
        and not session.pending(pull)
    )


def candidate_jobs(session: Session) -> list[dict]:
    """The candidate's job in its planner run's attempts and its resume runs, oldest first.

    Rerunning failed jobs copies the other jobs into the new attempt with their original start;
    those copies are left out.
    """
    prefix = f"Run candidate {session.candidate.id}"
    runs = [session.parent["id"], *(run["id"] for run in session.resume_runs())]
    return sorted(
        (
            job
            for run in runs
            for job in github.items(
                session.repository, f"actions/runs/{run}/jobs?filter=all&per_page=100", "jobs"
            )
            if job["name"].startswith(prefix)
            and job["started_at"]
            and utc(job["started_at"]) >= utc(job["created_at"])
        ),
        key=lambda job: job["created_at"],
    )


def started_session(job: dict) -> bool:
    """Whether a candidate job started its agent, or still may while its gate runs.

    A rerun planner job skips the agent step, and so does a resume run without its grant.
    """
    return job["conclusion"] != "skipped" and not any(
        step["name"] == AGENT_STEP and step["conclusion"] == "skipped" for step in job["steps"]
    )


def sessions_started(session: Session) -> int:
    """Sessions started for the candidate, a running one included."""
    return sum(started_session(job) for job in candidate_jobs(session))


def resumes(session: Session, pull: dict | None, now: datetime) -> bool:
    """Whether a new resume run would continue this unfinished session."""
    if session.last_session_end() is None or not resumable(session, pull):
        return False
    started = [job for job in candidate_jobs(session) if started_session(job)]
    return (
        0 < len(started) < MAX_SESSIONS
        # Runs that failed before their agent started never extend this.
        and bool(started[-1].get("completed_at"))
        and now - utc(started[-1]["completed_at"]) < timedelta(seconds=RESUME_GIVE_UP_SECONDS)
        # A planner run from before resumption has no gate.
        and any(step["name"] == GATE_STEP for step in started[-1]["steps"])
        # A session that runs out fails its job, and a cancelled session was preempted like any
        # other Klaud work; only klaud-abort stops a candidate.
        and started[-1]["conclusion"] in ("failure", "timed_out", "cancelled")
    )


def resume(session: Session, pull: dict | None, now: datetime) -> None:
    """Approve resuming an unfinished session; recovery grants approved ones after its pass."""
    if session.recovering and resumes(session, pull, now):
        raise ResumeSession(pull)


def resume_marker(parent: int, candidate: str, number: int) -> str:
    return f"<!-- klaud-resume:{parent}:{candidate}:session-{number} -->"


def granted(session: Session, pull: dict, number: int) -> bool:
    """Whether recovery granted this candidate's session ``number``."""
    marker = resume_marker(session.parent["id"], session.candidate.id, number)
    return any(
        comment["user"]["login"] == BOT and comment["body"].startswith(marker)
        for comment in github.items(
            session.repository, f"issues/{pull['number']}/comments?per_page=100"
        )
    )


def grant(session: Session, pull: dict) -> int:
    """Grant an unfinished session its next session, then dispatch the resume run that runs it."""
    repository, parent = session.repository, session.parent["id"]
    session.refresh(pull)
    number = sessions_started(session) + 1
    if not 2 <= number <= MAX_SESSIONS:
        raise VerificationError(f"Session {number} cannot resume")
    posted = None
    if not granted(session, pull, number):
        link = f"https://github.com/{repository}/actions/runs/{parent}"
        body = (
            resume_marker(parent, session.candidate.id, number)
            + "\n"
            + translated(
                f"**resumed** · The session ended before finishing; session {number} continues "
                f"it in a new run of [its planner]({link}).",
                f"**resumed** · 会话未完成即已结束；第 {number} 个会话将在"
                f"[其规划运行]({link})的新运行中继续。",
            )
        )
        posted = github.write(
            repository, f"issues/{pull['number']}/comments", "POST", {"body": body}
        )["id"]
    try:
        github.write(
            repository,
            "actions/workflows/klaud-plan.yml/dispatches",
            "POST",
            {
                "ref": "main",
                "inputs": {
                    "resume-parent": str(parent),
                    "resume-candidate": session.candidate.id,
                    "resume-session": str(number),
                },
            },
        )
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        # A grant without its resume run would let a later run start an uncounted session.
        if posted is not None:
            with contextlib.suppress(
                OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError
            ):
                github.write(repository, f"issues/comments/{posted}", "DELETE")
        raise
    return number


def retained(session: Session, now: datetime) -> bool:
    """Whether recovery keeps a session it could not reconcile for a later pass.

    Until RESUME_GIVE_UP_SECONDS after its last session ended, an open draft PR may still
    resume; a PR or run that cannot be read counts as one.
    """
    try:
        ended = session.last_session_end()
        if ended and now - utc(ended) >= timedelta(seconds=RESUME_GIVE_UP_SECONDS):
            return False
        pulls = session.pulls()
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        return True
    return bool(pulls) and unfinished_draft(session, pulls[0])


def summarize(repository: str, candidate: OwnedCandidate, parent: int, status: str) -> None:
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a") as output:
            output.write(
                f"Klaud `{candidate.id}`: {status}. "
                f"[Parent run / 父运行](https://github.com/{repository}/actions/runs/{parent})\n\n"
            )


def reconcile(session: Session, now: datetime) -> bool:
    pulls = session.pulls()
    pull = pulls[0] if pulls else None
    if pull and pull["merged_at"]:
        return False
    record = session.report(pull) if pull else None
    if record:
        outcome = CandidateOutcome.model_validate(record["outcome"])
        session.verify(outcome, require_branch_deleted=False)
        changed = False
        if outcome.outcome != "validated" and pull and session.branch_exists():
            session.delete_branch(pull)
            changed = True
        session.verify(outcome)
        return changed
    runs = session.runs()
    if not pull and not runs:
        return False
    pending = session.pending(pull) if pull else None
    if pull and not pending and session.aborted(pull):
        session.finish(
            CandidateOutcome(
                outcome="aborted",
                phase="cleanup",
                pull_request=pull["number"],
                run_ids=[run["id"] for run in runs],
                repairs_used=None,
            )
        )
        return True
    final = [
        run
        for run in runs
        if pull
        and run["event"] == "pull_request"
        and run["head_sha"] == pull["head"]["sha"]
        and run["conclusion"] != "skipped"
    ]
    if not pending and any(not terminal(run) for run in runs):
        resume(session, pull, now)
        raise PendingCleanup("Owned work is still running; retained for the next recovery pass")
    latest = max(final, key=lambda r: r["created_at"]) if final else None
    if (
        not pending
        and latest
        and pull["state"] == "open"
        and latest["conclusion"] == "cancelled"
        and now - utc(latest["updated_at"]) < timedelta(seconds=RERUN_GIVE_UP_SECONDS)
    ):
        await_rerun(session, pull, latest, now)
    validated = bool(latest and terminal(latest) and latest["conclusion"] == "success")
    if validated and not pending:
        try:
            session.validation(latest)
        except VerificationError as error:
            print(
                f"::warning::Klaud: final sweep of candidate {session.candidate.id} "
                f"failed validation: {error}."
            )
            validated = False
    if not pending and not validated:
        resume(session, pull, now)
        if (
            not session.recovering
            and resumable(session, pull)
            and sessions_started(session) < MAX_SESSIONS
        ):
            raise PendingCleanup("Unfinished session kept for a later resume")
    outcome = pending or CandidateOutcome(
        outcome="validated" if validated else "unexpected-error",
        phase="final-sweep" if validated else "cleanup",
        pull_request=pull["number"] if pull else None,
        run_ids=[r["id"] for r in runs],
        repairs_used=None,
    )
    session.finish(outcome)
    return True


def recover(max_resumes: int) -> None:
    repository = os.environ["GITHUB_REPOSITORY"]
    now = datetime.now(UTC)
    inventory = github.items(
        repository, "actions/artifacts?name=klaud-ownership&per_page=100", "artifacts"
    )
    owners = [(owner, None) for owner in claims.family_owners(repository)]
    for artifact in inventory:
        if artifact["expired"]:
            continue
        parent = github.read(repository, f"actions/runs/{artifact['workflow_run']['id']}")
        if (
            parent["head_branch"] != "main"
            or parent["path"] != ".github/workflows/klaud-plan.yml"
            or parent["event"] not in ("schedule", "workflow_dispatch")
        ):
            continue
        with tempfile.TemporaryDirectory(prefix="klaud-ownership-") as temp:
            github.download_json(repository, artifact, Path(temp))
            owner = Ownership.model_validate_json((Path(temp) / "ownership.json").read_text())
        if owner.run_id != parent["id"]:
            raise VerificationError("Ownership parent mismatch")
        owners.append((owner, artifact))
    parents = {
        owner.run_id: github.read(repository, f"actions/runs/{owner.run_id}") for owner, _ in owners
    }
    if any(
        parent["head_branch"] != "main"
        or parent["path"] != ".github/workflows/klaud-plan.yml"
        or parent["event"] not in ("schedule", "workflow_dispatch")
        for parent in parents.values()
    ):
        raise VerificationError("Untrusted ownership parent")
    results = {}
    blocked = set()
    live = set()
    approved: list[tuple[Session, dict]] = []
    for owner, _ in sorted(owners, key=lambda pair: pair[0].run_id, reverse=True):
        parent = parents[owner.run_id]
        for candidate in owner.candidates:
            key = (parent["id"], candidate.id)
            if key in results:
                continue
            results[key] = False
            if not terminal(parent):
                blocked.add(candidate.family)
                live.add(candidate.id)
                continue
            session = Session(repository, parent, candidate, recovering=True)
            try:
                if not all(terminal(run) for run in session.resume_runs()):
                    blocked.add(candidate.family)
                    live.add(candidate.id)
                    continue
                if not claims.recovery_lease(repository, parent["id"], candidate):
                    blocked.add(candidate.family)
                    live.add(candidate.id)
                    continue
                changed = reconcile(session, now)
                claims.release_family(repository, candidate, parent["id"])
            except (
                OSError,
                ValueError,
                KeyError,
                TypeError,
                subprocess.SubprocessError,
            ) as error:
                blocked.add(candidate.family)
                if isinstance(error, ResumeSession):
                    live.add(candidate.id)
                    approved.append((session, error.pull))
                    continue
                if isinstance(error, AwaitingRerun) or retained(session, now):
                    live.add(candidate.id)
                reason = (
                    str(error)
                    if isinstance(error, VerificationError)
                    else "state unavailable or invalid"
                )
                status = (
                    "pending / 等待后续收尾"
                    if isinstance(error, PendingCleanup)
                    else "needs inspection / 需要检查"
                ) + f" ({reason})"
            else:
                results[key] = True
                status = "reconciled / 已收尾" if changed else None
            if status:
                summarize(repository, candidate, parent["id"], status)
    resumed = 0
    # Oldest planner runs first; resumed sessions share this wave's candidate cap.
    approved.sort(key=lambda pair: (pair[0].parent["id"], pair[0].candidate.id))
    for index, (session, pull) in enumerate(approved):
        run_id = session.parent["id"]
        if index >= max_resumes:
            status = "pending / 等待后续收尾 (resume deferred: no slot left in this wave)"
        else:
            try:
                number = grant(session, pull)
            except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
                reason = (
                    str(error)
                    if isinstance(error, VerificationError)
                    else "state unavailable or invalid"
                )
                print(
                    f"::warning::Klaud: resuming candidate {session.candidate.id} of planner run "
                    f"{run_id} failed: {reason}."
                )
                status = f"pending / 等待后续收尾 (resume failed, retrying next pass: {reason})"
            else:
                resumed += 1
                status = f"resumed / 已恢复 (session {number})"
        summarize(repository, session.candidate, run_id, status)
    for owner, artifact in owners:
        if artifact and all(
            results[(owner.run_id, candidate.id)] for candidate in owner.candidates
        ):
            current = github.read(repository, f"actions/runs/{owner.run_id}")
            if terminal(current) and current["run_attempt"] == parents[owner.run_id]["run_attempt"]:
                github.write(repository, f"actions/artifacts/{artifact['id']}", "DELETE")
    try:
        live |= sweep_abandoned(repository, live)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(f"::warning::Klaud abandoned-work sweep skipped: {type(error).__name__}")
    else:
        prefixes = {candidate.split("-", 1)[0] for candidate in live}
        blocked = {family for family in blocked if identity(family)[:16] in prefixes}
    (Path(os.environ["RUNNER_TEMP"]) / "klaud-recovery.json").write_text(
        json.dumps(sorted(blocked))
    )
    if filename := os.environ.get("GITHUB_OUTPUT"):
        with open(filename, "a") as output:
            output.write(f"resumed={resumed}\n")


def owned_activity(repository: str) -> tuple[set[str], dict[str, list[int]], set[int]]:
    live: set[str] = set()
    targeted: dict[str, list[int]] = {}
    planners: set[int] = set()
    for status in ACTIVE_STATUSES:
        query = urlencode({"status": status, "per_page": 100})
        for run in github.items(
            repository, f"actions/workflows/e2e-tests.yml/runs?{query}", "workflow_runs"
        ):
            match = re.fullmatch(rf"e2e Test - klaud-\d+-({CANDIDATE_ID})", run["display_title"])
            if match and run["actor"]["login"] == BOT:
                targeted.setdefault(match[1], []).append(run["id"])
        for run in github.items(
            repository, f"actions/workflows/run-sweep.yml/runs?{query}", "workflow_runs"
        ):
            if match := re.fullmatch(rf"klaud/auto-({CANDIDATE_ID})", run["head_branch"] or ""):
                live.add(match[1])
        for run in github.items(
            repository, f"actions/workflows/klaud-plan.yml/runs?{query}", "workflow_runs"
        ):
            for job in github.items(
                repository, f"actions/runs/{run['id']}/jobs?per_page=100", "jobs"
            ):
                if match := re.search(rf"Run candidate ({CANDIDATE_ID})", job["name"]):
                    live.add(match[1])
                elif job["name"] == claims.RECONCILE_JOB and job["status"] != "completed":
                    planners.add(run["id"])
    return live, targeted, planners


def bot_committed(commits: list[dict]) -> bool:
    return bool(commits) and all(
        (commit.get("committer") or {}).get("login") == COMMITTER for commit in commits
    )


def sweep_abandoned(repository: str, live: set[str]) -> set[str]:
    sessions, targeted, planners = owned_activity(repository)
    live = live | sessions
    pulls = github.items(repository, "pulls?state=open&per_page=100")
    abandoned = []
    for pull in pulls:
        match = re.fullmatch(rf"klaud/auto-({CANDIDATE_ID})", pull["head"]["ref"])
        if not match or match[1] in live:
            continue
        if (
            pull["draft"]
            and pull["user"]["login"] == BOT
            and (pull["head"]["repo"] or {}).get("full_name") == repository
            and bot_committed(
                github.items(repository, f"pulls/{pull['number']}/commits?per_page=100")
            )
        ):
            abandoned.append(pull)
        else:
            live.add(match[1])
    for candidate, runs in targeted.items():
        if candidate not in live:
            for run in runs:
                github.write(repository, f"actions/runs/{run}/cancel", "POST")
    for owner in claims.family_owners(repository):
        candidate = owner.candidates[0]
        if candidate.id not in live and terminal(
            github.read(repository, f"actions/runs/{owner.run_id}")
        ):
            claims.release_family(repository, candidate, owner.run_id)
    if planners <= {int(os.environ.get("GITHUB_RUN_ID", "0"))}:
        finished: dict[int, bool] = {}
        for lease in github.items(repository, "git/matching-refs/heads/klaud/recovery-"):
            match = re.fullmatch(rf"refs/heads/klaud/recovery-(\d+)-({CANDIDATE_ID})", lease["ref"])
            if not match or match[2] in live:
                continue
            parent = int(match[1])
            if parent not in finished:
                finished[parent] = terminal(github.read(repository, f"actions/runs/{parent}"))
            if finished[parent]:
                github.write(repository, "git/" + lease["ref"], "DELETE")
    for pull in abandoned:
        github.write(
            repository,
            f"issues/{pull['number']}/comments",
            "POST",
            {
                "body": translated(
                    "**abandoned** · The owning Klaud session ended without cleanup. Closed, "
                    "with any leftover smoke runs cancelled, so a new candidate can take this "
                    "family.",
                    "**abandoned** · 所属 Klaud 会话已结束但未完成清理。已关闭并取消残留的冒烟运行，"
                    "新的候选可以接手该配置族。",
                )
            },
        )
        github.write(repository, f"pulls/{pull['number']}", "PATCH", {"state": "closed"})
        github.write(repository, f"git/refs/heads/{pull['head']['ref']}", "DELETE")
    open_heads = {pull["head"]["ref"] for pull in pulls}
    for ref in github.items(repository, "git/matching-refs/heads/klaud/auto-"):
        branch = ref["ref"].removeprefix("refs/heads/")
        match = re.fullmatch(rf"klaud/auto-({CANDIDATE_ID})", branch)
        if not match or match[1] in live or branch in open_heads:
            continue
        history = github.items(
            repository,
            "pulls?"
            + urlencode(
                {
                    "state": "all",
                    "head": repository.split("/", maxsplit=1)[0] + ":" + branch,
                    "per_page": 100,
                }
            ),
        )
        if history:
            latest = max(history, key=lambda pull: pull["created_at"])
            closer = github.read(repository, f"issues/{latest['number']}").get("closed_by") or {}
            if latest["user"]["login"] != BOT or closer.get("login") != BOT:
                continue
        elif not bot_committed([github.read(repository, f"commits/{ref['object']['sha']}")]):
            continue
        github.write(repository, f"git/refs/heads/{branch}", "DELETE")
    return live
