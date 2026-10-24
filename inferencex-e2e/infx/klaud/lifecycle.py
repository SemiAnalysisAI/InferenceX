"""Finish owned sessions; park, resume or reconcile interrupted ones in later recovery passes.

A session that waits on owned runs parks instead of idling: recovery reruns preempted runs and
starts the next session only once every owned run has ended. A candidate gets at most
``MAX_SESSIONS`` sessions, of which at most ``MAX_EXPIRED_SESSIONS`` may end without parking or
finishing. Planner waves start every ``WAVE_HOURS`` as scheduled in klaud-plan.yml.
"""

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
RERUN_GRACE_SECONDS = 30 * 60
RERUN_GIVE_UP_SECONDS = 24 * 60 * 60
MAX_RUN_ATTEMPTS = 10
MAX_SESSIONS = 12
MAX_EXPIRED_SESSIONS = 4
WAVE_HOURS = 3
RESUME_GIVE_UP_SECONDS = 24 * 60 * 60
SESSION_SECONDS = 330 * 60
PARK_AFTER_SECONDS = 300 * 60
GATE_STEP = "Check whether this session should run"
AGENT_STEP = "Let Klaud Cold own the candidate PR"
WAKE_TITLE = "Klaud Cold wake"
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
    """A preempted run will run again; keep its PR, branch, claim and lease."""


class ResumeSession(PendingCleanup):
    """Recovery may resume this unfinished session; keep its PR, branch, claim and lease."""

    def __init__(self, pull: dict) -> None:
        """Carry the resumable ``pull`` with a fixed public message."""
        super().__init__("Unfinished session approved to resume in a new run")
        self.pull = pull


def terminal(run: dict) -> bool:
    """Return True if workflow ``run`` has completed with a conclusion."""
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
        """Track ``candidate`` of planner run ``parent`` in ``repository``, as recovery if set."""
        self.repository, self.parent, self.candidate = repository, parent, candidate
        self.recovering = recovering
        self.branch = f"klaud/auto-{candidate.id}"
        self.marker = f"<!-- klaud-outcome:{parent['id']}:{candidate.id}\n"
        self.validations = {}
        self.first_attempt_end: datetime | None = None
        self.listings: dict[str, tuple[int, list[dict]]] = {}

    def listing(self, path: str, key: str | None = None) -> list[dict]:
        """``path``'s listing, reused until the next GitHub write or ownership check."""
        cached = self.listings.get(path)
        if cached is None or cached[0] != github.writes:
            cached = self.listings[path] = (github.writes, github.items(self.repository, path, key))
        return cached[1]

    def validation(self, run: dict) -> tuple[dict, list[dict], list[dict]]:
        """Return the sweep evidence of final sweep ``run``, cached per run attempt.

        Raises VerificationError if the sweep or its baseline coverage fails verification.
        """
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
        """Return this candidate's PR as a list of at most one, raising VerificationError.

        A PR opened outside the planner's first attempt is ignored. More than one, or one not
        owned by the bot from this repository, is an error.
        """
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
            for pull in self.listing("pulls?" + query)
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
        runs = self.listing(
            "actions/workflows/klaud-plan.yml/runs?"
            + urlencode(
                {
                    "event": "workflow_dispatch",
                    "actor": BOT,
                    "created": ">=" + self.parent["created_at"],
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

    def planner_jobs(self) -> list[dict]:
        """This candidate's jobs in every attempt of its planner run."""
        prefix = f"Run candidate {self.candidate.id}"
        return [
            job
            for job in self.listing(
                f"actions/runs/{self.parent['id']}/jobs?filter=all&per_page=100", "jobs"
            )
            if job["name"].startswith(prefix)
        ]

    def idle(self) -> bool:
        """Return True once no job of this candidate's sessions can still run.

        Sibling candidates may keep the planner run going after this candidate's job ended.
        """
        if not all(terminal(run) for run in self.resume_runs()):
            return False
        parent = github.read(self.repository, f"actions/runs/{self.parent['id']}")
        if terminal(parent):
            return True
        jobs = [
            job for job in self.planner_jobs() if job.get("run_attempt") == parent["run_attempt"]
        ]
        return bool(jobs) and all(job.get("completed_at") for job in jobs)

    def last_session_end(self) -> str | None:
        """When this candidate's latest session job ended, or None while one may still run."""
        if not self.idle():
            return None
        ends = [job["completed_at"] for job in self.planner_jobs() if job.get("completed_at")]
        ends += [run["updated_at"] for run in self.resume_runs()]
        return max(ends, key=utc) if ends else self.parent["updated_at"]

    def runs(self) -> list[dict]:
        """Return the candidate's targeted e2e runs followed by its PR sweep runs.

        In recovery, raises VerificationError if a maintainer started a sweep after the session.
        """
        ended = self.last_session_end()
        end = ended or "*"
        query = {"per_page": 100, "created": f"{self.parent['created_at']}..{end}"}
        targeted = self.listing(
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
        sweeps = self.listing(
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
        """Return True if ``pull`` carries the abort label."""
        return any(label["name"] == ABORT_LABEL for label in pull["labels"])

    def refresh(self, pull: dict) -> dict:
        """Return fresh ``pull``, raising VerificationError if it moved, merged or changed hands."""
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
        """Drop cached listings and, in recovery, raise VerificationError if the session resumed."""
        self.listings.clear()
        if self.recovering and not self.idle():
            raise VerificationError("Session resumed; leave the active session intact")

    def report(self, pull: dict) -> dict | None:
        """Return the outcome record of this session's latest bot report on ``pull``, or None."""
        comments = self.listing(f"issues/{pull['number']}/comments?per_page=100")
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
        """Return the outcome of this session's pending cleanup request on ``pull``, or None.

        Raises VerificationError if the PR head moved since or the outcome claims validated.
        """
        marker = self.marker.replace("klaud-outcome:", "klaud-cleanup:")
        comments = self.listing(f"issues/{pull['number']}/comments?per_page=100")
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
        """Return True if the candidate branch exists."""
        refs = self.listing("git/matching-refs/heads/" + self.branch)
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
        """Raise unless GitHub state matches ``outcome`` for this session, else return None.

        Raises PendingCleanup while owned jobs run and VerificationError on any mismatch. The
        flags relax the report, ready-for-review and deleted-branch checks.
        """
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
        """Apply ``outcome`` to this session and return it with the real PR and run IDs.

        Validated PRs are published and marked ready. Others have runs cancelled, the PR closed
        and the branch deleted. Then the report is posted and the family claim released.
        """
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
                    try:
                        github.write(self.repository, f"actions/runs/{run['id']}/cancel", "POST")
                    except (OSError, subprocess.SubprocessError):
                        if not terminal(github.read(self.repository, f"actions/runs/{run['id']}")):
                            raise
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
                repairs_zh = outcome.repairs_used if outcome.repairs_used is not None else "未知"
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
                        f"**{outcome.outcome}**{reason_zh} · 修复次数：{repairs_zh} · 运行：{links}  \n{why_zh}所有自有运行均已结束。"
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
    """Mark PR ``number`` ready for review, or back to draft if ``undo``, counting a write."""
    github.writes += 1
    subprocess.run(
        ["gh", "pr", "ready", str(number), "--repo", repository] + (["--undo"] if undo else []),
        check=True,
        capture_output=True,
        timeout=60,
    )


def current_session() -> Session:
    """Return the Session for the candidate in ``$KLAUD_EVIDENCE/candidate.json``.

    A resume run passes ``KLAUD_PARENT_RUN``, the planner run that owns the candidate.
    """
    context = json.loads((Path(os.environ["KLAUD_EVIDENCE"]) / "candidate.json").read_text())
    candidate = OwnedCandidate.model_validate(
        {key: context[key] for key in ("id", "family", "base")}
    )
    repository = os.environ["GITHUB_REPOSITORY"]
    parent = os.environ.get("KLAUD_PARENT_RUN") or os.environ["GITHUB_RUN_ID"]
    return Session(repository, github.read(repository, f"actions/runs/{int(parent)}"), candidate)


def hit_timeout(repository: str, job: dict) -> bool:
    """Return True if ``job`` ran and GitHub annotated it as stopped at its timeout-minutes."""
    return bool(job.get("runner_name")) and any(
        TIMEOUT_ANNOTATION in (annotation.get("message") or "")
        for annotation in github.items(
            repository, f"check-runs/{job['id']}/annotations?per_page=100"
        )
    )


def preempted(repository: str, run: dict) -> bool:
    """Cancelled by someone else: no job of its current attempt failed or hit its timeout.

    Failures are ruled out first because a fail-fast sweep cancels every other job.
    """
    if run["conclusion"] != "cancelled":
        return False
    jobs = github.items(
        repository,
        f"actions/runs/{run['id']}/attempts/{run['run_attempt']}/jobs?per_page=100",
        "jobs",
    )
    return not any(job["conclusion"] in ("failure", "timed_out") for job in jobs) and not any(
        job["conclusion"] == "cancelled" and hit_timeout(repository, job) for job in jobs
    )


def await_rerun(session: Session, pull: dict, run: dict, now: datetime) -> None:
    """Raise AwaitingRerun to keep ``session`` while preempted owned ``run`` can rerun.

    Waits RERUN_GRACE_SECONDS for the scheduler's own rerun, then requests one. Returns if
    ``run`` was not preempted or is out of attempts.
    """
    try:
        if not preempted(session.repository, run):
            return
    except (OSError, subprocess.SubprocessError, VerificationError) as error:
        raise AwaitingRerun("Preemption check unavailable; retrying next pass") from error
    if now - utc(run["updated_at"]) < timedelta(seconds=RERUN_GRACE_SECONDS):
        raise AwaitingRerun("Preempted run is waiting for its rerun")
    if run["run_attempt"] >= MAX_RUN_ATTEMPTS:
        return
    session.refresh(pull)
    try:
        github.write(session.repository, f"actions/runs/{run['id']}/rerun-failed-jobs", "POST")
    except (OSError, subprocess.SubprocessError) as error:
        raise AwaitingRerun("Rerun request failed; retrying next pass") from error
    raise AwaitingRerun("Reran preempted run")


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
            for job in session.listing(f"actions/runs/{run}/jobs?filter=all&per_page=100", "jobs")
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


def park_marker(parent: int, candidate: str, number: int) -> str:
    """Return the comment marker that records where session ``number`` parked."""
    return f"<!-- klaud-park:{parent}:{candidate}:session-{number}\n"


def parked(session: Session, pull: dict, number: int) -> list[int] | None:
    """The owned runs session ``number`` parked on, or None if it never parked.

    An empty list means the session parked at its deadline with no owned run left to wait on.
    """
    marker = park_marker(session.parent["id"], session.candidate.id, number)
    matches = [
        comment
        for comment in session.listing(f"issues/{pull['number']}/comments?per_page=100")
        if comment["user"]["login"] == BOT and comment["body"].startswith(marker)
    ]
    if not matches:
        return None
    body = max(matches, key=lambda comment: comment["id"])["body"]
    return json.loads(body[len(marker) :].split("\n-->", 1)[0])["runs"]


def within_budget(session: Session, pull: dict) -> bool:
    """Return True while the candidate may start another session.

    A session that parked on owned runs and then ended cleanly (or is still running) is free up
    to MAX_SESSIONS. One that ran out, crashed, kept working after parking, or parked only at its
    deadline counts toward MAX_EXPIRED_SESSIONS.
    """
    started = [job for job in candidate_jobs(session) if started_session(job)]
    expired = sum(
        not (parked(session, pull, number) and job.get("conclusion") in (None, "success"))
        for number, job in enumerate(started, 1)
    )
    return len(started) < MAX_SESSIONS and expired < MAX_EXPIRED_SESSIONS


def park(session: Session, number: int, *, overdue: bool) -> list[int]:
    """Record that session ``number`` waits on its live owned runs, and return their IDs.

    Each waited-on run needs a published attempt report so the next session can find it.
    Without live runs only an ``overdue`` session may park. Raises VerificationError otherwise.
    """
    from .reporting import reported_runs

    pulls = session.pulls()
    pull = pulls[0] if pulls else None
    if not resumable(session, pull):
        raise VerificationError("Only an unfinished draft candidate PR can park; finish instead")
    waiting = sorted(run["id"] for run in session.runs() if not terminal(run))
    if not waiting and not overdue:
        raise VerificationError("No owned run is queued or running; continue or finish")
    if set(waiting) - reported_runs(session, pull):
        raise VerificationError("Publish an attempt report for every owned run before parking")
    session.refresh(pull)
    repository = session.repository
    links = ", ".join(
        f"[{run}](https://github.com/{repository}/actions/runs/{run})" for run in waiting
    )
    body = (
        park_marker(session.parent["id"], session.candidate.id, number)
        + json.dumps({"runs": waiting})
        + "\n-->\n"
        + (
            translated(
                f"**parked** · Session {number} is waiting on {links}. Recovery reruns "
                "preempted runs and starts the next session once they end.",
                f"**parked** · 第 {number} 个会话正在等待 {links}。恢复流程会重跑被抢占的运行，"
                "并在运行结束后启动下一个会话。",
            )
            if waiting
            else translated(
                f"**parked** · Session {number} reached its time limit; the next session "
                "continues it.",
                f"**parked** · 第 {number} 个会话已到时限；下一个会话将继续。",
            )
        )
    )
    previous = [
        comment
        for comment in session.listing(f"issues/{pull['number']}/comments?per_page=100")
        if comment["user"]["login"] == BOT
        and comment["body"].startswith(
            park_marker(session.parent["id"], session.candidate.id, number)
        )
    ]
    if previous:
        github.write(
            repository,
            f"issues/comments/{max(previous, key=lambda c: c['id'])['id']}",
            "PATCH",
            {"body": body},
        )
    else:
        github.write(repository, f"issues/{pull['number']}/comments", "POST", {"body": body})
    return waiting


def resumes(session: Session, pull: dict | None, now: datetime) -> bool:
    """Return True if a new resume run would continue this unfinished session at ``now``.

    Every owned run must have ended, and the last started job must have run the gate step and
    parked, failed, timed out or been cancelled. Resuming stops RESUME_GIVE_UP_SECONDS after
    the later of that job's end and the last owned run's update, or when the session budget is
    spent. Only klaud-abort stops a candidate otherwise.
    """
    if session.last_session_end() is None or not resumable(session, pull):
        return False
    runs = session.runs()
    started = [job for job in candidate_jobs(session) if started_session(job)]
    if (
        not started
        or any(not terminal(run) for run in runs)
        or not started[-1].get("completed_at")
        or not any(step["name"] == GATE_STEP for step in started[-1]["steps"])
    ):
        return False
    activity = max([started[-1]["completed_at"], *(run["updated_at"] for run in runs)], key=utc)
    return (
        now - utc(activity) < timedelta(seconds=RESUME_GIVE_UP_SECONDS)
        and within_budget(session, pull)
        and (
            started[-1]["conclusion"] in ("failure", "timed_out", "cancelled")
            or parked(session, pull, len(started)) is not None
        )
    )


def resume(session: Session, pull: dict | None, now: datetime) -> None:
    """Raise ResumeSession with ``pull`` when recovery may resume this session, else return.

    Recovery grants the approved sessions after its pass.
    """
    if session.recovering and resumes(session, pull, now):
        raise ResumeSession(pull)


def resume_marker(parent: int, candidate: str, number: int) -> str:
    """Return the comment marker that records the grant of session ``number``."""
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
    """Grant an unfinished session its next session, dispatch its resume run, return its number.

    If the dispatch fails the grant comment is deleted, so no uncounted session can start.
    """
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
        if posted is not None:
            with contextlib.suppress(
                OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError
            ):
                github.write(repository, f"issues/comments/{posted}", "DELETE")
        raise
    return number


def retained(session: Session, now: datetime) -> bool:
    """Whether recovery keeps a session it could not reconcile for a later pass.

    An open draft PR is kept while any owned run is live, then until RESUME_GIVE_UP_SECONDS
    after the later of its last session's end and its last run's update. Unreadable runs count
    only the session's end; an unreadable PR or session counts as kept.
    """
    try:
        ended = session.last_session_end()
        try:
            runs = session.runs()
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
            runs = []
        if any(not terminal(run) for run in runs):
            return True
        if ended:
            activity = max([ended, *(run["updated_at"] for run in runs)], key=utc)
            if now - utc(activity) >= timedelta(seconds=RESUME_GIVE_UP_SECONDS):
                return False
        pulls = session.pulls()
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        return True
    return bool(pulls) and unfinished_draft(session, pulls[0])


def summarize(repository: str, candidate: OwnedCandidate, parent: int, status: str) -> None:
    """Append ``status`` for ``candidate`` of run ``parent`` to the job summary, if one exists."""
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a") as output:
            output.write(
                f"Klaud `{candidate.id}`: {status}. "
                f"[Parent run / 父运行](https://github.com/{repository}/actions/runs/{parent})\n\n"
            )


def reconcile(session: Session, now: datetime) -> bool:
    """Settle interrupted ``session`` at ``now`` and return True if anything changed.

    Raises PendingCleanup subclasses while owned work runs, while the latest exact-head final
    sweep or a newer targeted run was preempted and may rerun, or when the session may resume.
    Returns False for a merged PR or no owned work.
    """
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
        raise PendingCleanup("Owned work is still running; retained for the next recovery pass")
    latest = max(final, key=lambda r: r["created_at"]) if final else None
    targeted = max(
        (run for run in runs if run["event"] != "pull_request" and run["conclusion"] != "skipped"),
        key=lambda r: r["created_at"],
        default=None,
    )
    if targeted and latest and targeted["created_at"] < latest["created_at"]:
        targeted = None
    for run in (latest, targeted):
        if (
            not pending
            and run
            and pull
            and pull["state"] == "open"
            and run["conclusion"] == "cancelled"
            and now - utc(run["updated_at"]) < timedelta(seconds=RERUN_GIVE_UP_SECONDS)
        ):
            await_rerun(session, pull, run, now)
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
        if not session.recovering and resumable(session, pull) and within_budget(session, pull):
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


def recover(max_sessions: int) -> None:
    """Reconcile every owned candidate, then resume approved sessions, oldest planners first.

    Resumes only while sessions already running plus those resumed here stay below
    ``max_sessions``. Writes still-blocked families to ``$RUNNER_TEMP/klaud-recovery.json`` and
    the running plus resumed session count to ``$GITHUB_OUTPUT``. Unreadable state counts as
    live work.
    """
    repository = os.environ["GITHUB_REPOSITORY"]
    now = datetime.now(UTC)
    inventory = github.items(
        repository, "actions/artifacts?name=klaud-ownership&per_page=100", "artifacts"
    )
    owners = [(owner, None) for owner in claims.family_owners(repository)]
    for artifact in inventory:
        if artifact["expired"]:
            continue
        try:
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
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
            print(f"::warning::Klaud: ownership artifact {artifact['id']} is unreadable; skipped.")
            continue
        if owner.run_id != parent["id"]:
            raise VerificationError("Ownership parent mismatch")
        owners.append((owner, artifact))
    results = {}
    blocked = set()
    live = set()
    parents = {}
    for owner, _ in owners:
        if owner.run_id in parents:
            continue
        try:
            parents[owner.run_id] = github.read(repository, f"actions/runs/{owner.run_id}")
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
            for candidate in owner.candidates:
                results[(owner.run_id, candidate.id)] = False
                blocked.add(candidate.family)
                live.add(candidate.id)
                summarize(
                    repository,
                    candidate,
                    owner.run_id,
                    "needs inspection / 需要检查 (state unavailable or invalid)",
                )
    owners = [pair for pair in owners if pair[0].run_id in parents]
    if any(
        parent["head_branch"] != "main"
        or parent["path"] != ".github/workflows/klaud-plan.yml"
        or parent["event"] not in ("schedule", "workflow_dispatch")
        for parent in parents.values()
    ):
        raise VerificationError("Untrusted ownership parent")
    approved: list[tuple[Session, dict]] = []
    for owner, _ in sorted(owners, key=lambda pair: pair[0].run_id, reverse=True):
        parent = parents[owner.run_id]
        for candidate in owner.candidates:
            key = (parent["id"], candidate.id)
            if key in results:
                continue
            results[key] = False
            session = Session(repository, parent, candidate, recovering=True)
            try:
                if not session.idle():
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
    try:
        running = len(running_sessions(repository))
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        print("::warning::Klaud: running sessions unavailable; resuming none this pass.")
        running = max_sessions
    resumed = 0
    approved.sort(key=lambda pair: (pair[0].parent["id"], pair[0].candidate.id))
    for session, pull in approved:
        run_id = session.parent["id"]
        if running + resumed >= max_sessions:
            status = "pending / 等待后续收尾 (resume deferred: every session slot is in use)"
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
            output.write(f"active-sessions={running + resumed}\n")


def running_sessions(repository: str) -> set[str]:
    """Return the candidates whose session job in an active klaud-plan run has not finished."""
    running = set()
    for status in ACTIVE_STATUSES:
        query = urlencode({"status": status, "per_page": 100})
        for run in github.items(
            repository, f"actions/workflows/klaud-plan.yml/runs?{query}", "workflow_runs"
        ):
            for job in github.items(
                repository, f"actions/runs/{run['id']}/jobs?per_page=100", "jobs"
            ):
                match = re.search(rf"Run candidate ({CANDIDATE_ID})", job["name"])
                if match and not job.get("completed_at"):
                    running.add(match[1])
    return running


def owned_activity(repository: str) -> tuple[set[str], dict[str, list[int]], set[int]]:
    """Return active Klaud work in ``repository``.

    That is live candidate IDs, active targeted run IDs per candidate, and planner runs whose
    reconcile job is unfinished.
    """
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
    """Return True if ``commits`` is non-empty and the Actions bot committed all of them."""
    return bool(commits) and all(
        (commit.get("committer") or {}).get("login") == COMMITTER for commit in commits
    )


def sweep_abandoned(repository: str, live: set[str]) -> set[str]:
    """Clean up Klaud work no live session owns and return the widened ``live`` candidate IDs.

    Closes abandoned draft PRs and cancels orphan targeted runs. Deletes stale claims, leases
    and branches. A claim whose planner run cannot be read counts as live.
    """
    sessions, targeted, planners = owned_activity(repository)
    live = live | sessions
    claimed = []
    for owner in claims.family_owners(repository):
        candidate = owner.candidates[0]
        if candidate.id in live:
            continue
        try:
            parent = github.read(repository, f"actions/runs/{owner.run_id}")
        except (OSError, ValueError, subprocess.SubprocessError):
            live.add(candidate.id)
            continue
        claimed.append((owner, candidate, parent))
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
    for owner, candidate, parent in claimed:
        if candidate.id not in live and terminal(parent):
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
