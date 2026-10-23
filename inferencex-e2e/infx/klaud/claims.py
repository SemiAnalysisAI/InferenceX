"""Small Git-ref claims: atomic family selection and per-session recovery ownership.

No time-based stealing. Recovery leases change hands only after their owning
workflow ends, using a non-force fast-forward as compare-and-swap.
"""

from __future__ import annotations

import json
import os
import subprocess

from . import github
from .github import VerificationError
from .models import OwnedCandidate, Ownership, identity

RECONCILE_JOB = "Reconcile interrupted candidates"


def ref(repository: str, name: str) -> dict | None:
    """Return the ref record for branch ``name`` in ``repository``, or None if absent."""
    return next(
        (
            row
            for row in github.items(repository, "git/matching-refs/heads/" + name)
            if row["ref"] == "refs/heads/" + name
        ),
        None,
    )


def record(repository: str, value: dict) -> dict:
    """Return the JSON owner stored in the commit message of ref ``value``.

    Raises VerificationError when the message is not readable JSON.
    """
    commit = github.read(repository, "git/commits/" + value["object"]["sha"])
    try:
        return json.loads(commit["message"])
    except (KeyError, ValueError) as error:
        raise VerificationError("Unknown claim owner; manual inspection required") from error


def create(
    repository: str, name: str, payload: dict, base: str, previous: dict | None = None
) -> bool:
    """Point branch ``name`` at a new commit whose message is ``payload`` as JSON.

    The commit sits on ``previous`` (fast-forward, no force) or on ``base`` for a new ref.
    Return True if this caller won the ref, False if a concurrent writer did.
    """
    parent = previous["object"]["sha"] if previous else base
    commit = github.read(repository, "git/commits/" + parent)
    new = github.write(
        repository,
        "git/commits",
        "POST",
        {
            "message": json.dumps(payload, sort_keys=True),
            "tree": commit["tree"]["sha"],
            "parents": [parent],
        },
    )
    try:
        if previous:
            github.write(
                repository,
                "git/refs/heads/" + name,
                "PATCH",
                {"sha": new["sha"], "force": False},
            )
        else:
            github.write(
                repository,
                "git/refs",
                "POST",
                {"ref": "refs/heads/" + name, "sha": new["sha"]},
            )
    except subprocess.CalledProcessError:
        current = ref(repository, name)
        if current and current["object"]["sha"] != (previous or {}).get("object", {}).get("sha"):
            return current["object"]["sha"] == new["sha"]
        raise
    return True


def family_name(candidate: OwnedCandidate) -> str:
    """Return the claim branch name derived from the candidate's family."""
    return "klaud/claim-" + identity(candidate.family)[:16]


def claim_family(repository: str, candidate: OwnedCandidate, parent: int) -> bool:
    """Claim the candidate's family for run ``parent`` and return True if this call won."""
    name = family_name(candidate)
    if ref(repository, name):
        return False
    ownership = Ownership(run_id=parent, candidates=[candidate])
    return create(repository, name, ownership.model_dump(by_alias=True), candidate.base)


def release_family(repository: str, candidate: OwnedCandidate, parent: int) -> None:
    """Delete the candidate's family claim if run ``parent`` owns exactly that candidate."""
    name = family_name(candidate)
    value = ref(repository, name)
    if not value:
        return
    ownership = Ownership.model_validate(record(repository, value))
    if ownership.run_id != parent or ownership.candidates != [candidate]:
        return
    github.write(repository, "git/refs/heads/" + name, "DELETE")


def family_owners(repository: str) -> list[Ownership]:
    """Return every family claim's ownership, raising VerificationError on a malformed one."""
    result = []
    for value in github.items(repository, "git/matching-refs/heads/klaud/claim-"):
        owner = Ownership.model_validate(record(repository, value))
        if len(owner.candidates) != 1 or value["ref"] != "refs/heads/" + family_name(
            owner.candidates[0]
        ):
            raise VerificationError("Invalid family claim")
        result.append(owner)
    return result


def recovery_lease(repository: str, parent: int, candidate: OwnedCandidate) -> bool:
    """Take the recovery lease for ``candidate`` of run ``parent`` and return True if held.

    A prior holder hands over only once its planner run or its reconcile job has completed.
    """
    name = f"klaud/recovery-{parent}-{candidate.id}"
    owner = {
        "run-id": int(os.environ["GITHUB_RUN_ID"]),
        "run-attempt": int(os.environ.get("GITHUB_RUN_ATTEMPT", "1")),
    }
    previous = ref(repository, name)
    if previous:
        prior = record(repository, previous)
        if prior == owner:
            return True
        run = github.read(repository, f"actions/runs/{int(prior['run-id'])}")
        if run["path"] != ".github/workflows/klaud-plan.yml" or run["head_branch"] != "main":
            return False
        if run["status"] != "completed" or not run.get("conclusion"):
            jobs = github.items(
                repository,
                f"actions/runs/{int(prior['run-id'])}/attempts/{int(prior['run-attempt'])}"
                "/jobs?per_page=100",
                "jobs",
            )
            if not any(
                job["name"] == RECONCILE_JOB and job["status"] == "completed" for job in jobs
            ):
                return False
    return create(repository, name, owner, candidate.base, previous)
