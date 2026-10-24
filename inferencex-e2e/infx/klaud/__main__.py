"""Select Klaud Cold candidates or check capacity before a benchmark dispatch."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import secrets
import shlex
import subprocess
import tempfile
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

from pydantic import ValidationError

from .api import PUBLIC, ReadError, fetch_capacity, fetch_catalog
from .github import (
    VerificationError,
    download_json,
    items as github_items,
    read as github_read,
)
from .models import (
    ABORT_LABEL,
    CandidateOutcome,
    OwnedCandidate,
    Ownership,
    Policy,
    PRReview,
    SessionEnd,
    identity,
    normalized_image,
    utc,
)

if TYPE_CHECKING:
    from infx.clusters import RunnerInventory

PRODUCERS = "producers.json"
RETRY_NEXT_WAVE = frozenset({"capacity-deferred", "agent-not-started", "aborted"})


def observation_key(row: dict) -> tuple:
    """Return the public identity tuple of feed ``row``, with its image normalized."""
    image = row.get("image")
    if isinstance(image, str):
        image = normalized_image(image)
    return tuple(
        image if key == "image" else row.get(key)
        for key in (
            "model",
            "hardware",
            "framework",
            "precision",
            "spec_method",
            "disagg",
            "benchmark_type",
            "isl",
            "osl",
            "image",
        )
    )


def target_clusters(entries: list[dict], inventory: RunnerInventory) -> frozenset[str]:
    """Return the cluster IDs serving the runners of ``entries``, or none if one is unknown."""
    try:
        return frozenset(
            inventory.cluster_for(node).id
            for runner in {entry["runner"] for entry in entries}
            for node in inventory.labels.get(runner, [runner])
        )
    except ValueError:
        return frozenset()


def live_families(root: Path) -> dict[tuple, dict[str, tuple[frozenset[str], int]]]:
    """Resolve public AgentX identities, target clusters and node demand from the generator.

    Archives and fixed-seq-len points are excluded. A family's node demand is the node count of
    its largest generated AgentX job.
    """
    from infx.clusters import load_inventory
    from infx.config import project_root
    from infx.matrix.generate import _hardware_family, generate_test_config_sweep
    from infx.matrix.validation import load_config_files, load_runner_file
    from infx.results.identity import public_framework

    from .validation import in_scope

    root = project_root(root)
    runners = load_runner_file(str(root / "configs/runners.yaml"))
    inventory = load_inventory(runners)
    index: dict[tuple, dict[str, tuple[frozenset[str], int]]] = {}
    paths = sorted((root / "configs").glob("*-master.yaml"))
    if not paths:
        raise ReadError("live-configs-unavailable")
    for path in paths:
        configs = load_config_files([str(path)])
        for key in configs:
            family = f"{path.relative_to(root).as_posix()}:{key}"
            try:
                entries = generate_test_config_sweep(
                    SimpleNamespace(config_keys=[key]), configs, runners, root
                )
            except ValueError:
                print(f"::warning::Klaud live-family-unrenderable: {family}")
                continue
            entries = [entry for entry in entries if in_scope(entry)]
            if not entries:
                continue
            clusters = target_clusters(entries, inventory)
            if not clusters:
                print(f"::warning::Klaud live-family-clusters-unresolved: {family}")
                continue
            demand = max(entry.get("node-count", 1) for entry in entries)
            for entry in entries:
                row = {
                    "model": entry["model-prefix"],
                    "hardware": _hardware_family(entry["runner"]),
                    "framework": public_framework(entry["framework"]),
                    "precision": entry["precision"],
                    "spec_method": entry["spec-decoding"],
                    "disagg": entry.get("disagg", False),
                    "benchmark_type": "agentic_traces",
                    "isl": None,
                    "osl": None,
                    "image": entry["image"],
                }
                index.setdefault(observation_key(row), {})[family] = (clusters, demand)
    return index


def choose(
    items: list[dict],
    headroom: dict[str, int],
    occupied: set[str],
    families: dict[tuple, dict[str, tuple[frozenset[str], int]]],
) -> list[dict]:
    """Order one candidate per unclaimed family that fits the headroom of every target cluster.

    The order is a weighted shuffle: a family's weight is its benchmark age in days plus one, so
    older images tend to lead without starving recent ones. Headroom is checked but not consumed
    here, so the overlap review keeps alternatives.
    """
    occupied = {re.sub(r"^klaud[e]?/auto-", "klaud/auto-", branch) for branch in occupied}
    selected = []
    keys: list[float] = []
    seen = set()
    valid = [item for item in items if item["source-status"] == "baseline"]
    for item in sorted(
        valid,
        key=lambda item: (item["source"]["date"], item["source-id"]),
        reverse=True,
    ):
        row = item["source"]
        legacy = identity(
            {
                key: row[key]
                for key in (
                    "model",
                    "hardware",
                    "framework",
                    "precision",
                    "spec_method",
                    "disagg",
                )
            }
        )[:16]
        image_key = identity(observation_key(row)[-1])[:16]
        for family, (clusters, demand) in sorted(families.get(observation_key(row), {}).items()):
            prefix = "klaud/auto-" + identity(family)[:16] + "-"
            branch = prefix + image_key
            claims = {
                branch,
                prefix,
                f"klaud/auto-{legacy}-{image_key}",
                f"klaud/auto-{legacy}-",
            }
            if (
                family in seen
                or claims & occupied
                or any(headroom.get(cluster, 0) < demand for cluster in clusters)
            ):
                continue
            selected.append(
                {
                    "id": branch.removeprefix("klaud/auto-"),
                    "family": family,
                    "clusters": sorted(clusters),
                    "node-demand": demand,
                    "source": row,
                    "branch": branch,
                }
            )
            keys.append(random.expovariate(item["benchmark-age-days"] + 1))
            seen.add(family)
    ordered = sorted(zip(keys, selected, strict=True), key=lambda pair: pair[0])
    return [candidate for _, candidate in ordered]


def weights_key(source: dict) -> tuple:
    """Return the (model, precision, hardware) key that identifies the weights ``source`` needs."""
    return (source.get("model"), source.get("precision"), source.get("hardware"))


def recent_candidate_outcomes(
    repository: str, cooldown_hours: int
) -> tuple[dict[str, str | None], set[tuple]]:
    """Return each candidate's latest outcome within ``cooldown_hours`` and missing-weights keys.

    Resumed sessions record outcomes in later waves, so history reaches MAX_SESSIONS waves back.
    The newest session artifact counts. Unreadable history means no cooldown.
    """
    from .lifecycle import MAX_SESSIONS, WAKE_TITLE, WAVE_HOURS, resume_parent

    cutoff = datetime.now(UTC) - timedelta(hours=cooldown_hours)
    since = cutoff - timedelta(hours=MAX_SESSIONS * WAVE_HOURS)
    try:
        runs = github_items(
            repository,
            "actions/workflows/klaud-plan.yml/runs?created=>=" + since.date().isoformat(),
            "workflow_runs",
        )
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        print("::warning::Klaud cooldown history unavailable; continuing without cooldown")
        return {}, set()
    dated = []
    plans: dict[int, int] = {}
    for run in runs:
        try:
            created = utc(run["created_at"])
            if created >= since and run.get("display_title") != WAKE_TITLE:
                dated.append((created, int(run["id"])))
                plans[int(run["id"])] = resume_parent(run.get("display_title") or "") or int(
                    run["id"]
                )
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
    listings: dict[int, list[dict]] = {}
    latest: dict[str, tuple[dict, int]] = {}
    for _, run_id in sorted(dated, reverse=True):
        try:
            listings[run_id] = github_items(
                repository, f"actions/runs/{run_id}/artifacts?per_page=100", "artifacts"
            )
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
            continue
        for artifact in listings[run_id]:
            try:
                match = re.fullmatch(
                    r"klaud-candidate-([0-9a-f]{16}-[0-9a-f]{16})", artifact["name"]
                )
                if (
                    not match
                    or artifact.get("expired", True)
                    or utc(artifact["created_at"]) < cutoff
                ):
                    continue
                if match[1] not in latest or artifact["id"] > latest[match[1]][0]["id"]:
                    latest[match[1]] = (artifact, run_id)
            except (AttributeError, KeyError, TypeError, ValueError):
                continue
    result: dict[str, str | None] = {}
    blocked: dict[int, set[str]] = {}
    for candidate, (artifact, run_id) in latest.items():
        result[candidate] = recorded_outcome(repository, artifact)
        if result[candidate] == "weights-missing":
            blocked.setdefault(plans[run_id], set()).add(candidate)
    missing_weights = set()
    for plan_run, candidates in blocked.items():
        if plan_run not in listings:
            try:
                listings[plan_run] = github_items(
                    repository, f"actions/runs/{plan_run}/artifacts?per_page=100", "artifacts"
                )
            except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
                continue
        missing_weights.update(
            weights_key(planned["source"])
            for planned in planned_candidates(repository, listings[plan_run])
            if planned.get("id") in candidates
        )
    return result, missing_weights


def planned_candidates(repository: str, artifacts: list[dict]) -> list[dict]:
    """Return the candidates in the klaud-plan artifact among ``artifacts``, or [] if unreadable."""
    plan = next((a for a in artifacts if a.get("name") == "klaud-plan"), None)
    if not plan:
        return []
    with tempfile.TemporaryDirectory() as scratch:
        try:
            download_json(repository, plan, Path(scratch))
            candidates = json.loads((Path(scratch) / "candidates.json").read_text())
            return [
                c for c in candidates if isinstance(c, dict) and isinstance(c.get("source"), dict)
            ]
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
            return []


def recorded_outcome(repository: str, artifact: dict) -> str | None:
    """Return the outcome a diagnostics ``artifact`` recorded, or None if it is unreadable.

    A skipped agent yields ``agent-not-started`` and a missing-weights failure ``weights-missing``.
    """
    with tempfile.TemporaryDirectory() as scratch:
        try:
            download_json(repository, artifact, Path(scratch))
            diagnostics = json.loads((Path(scratch) / "candidate-diagnostics.json").read_text())
            if diagnostics.get("action-outcome") == "skipped":
                return "agent-not-started"
            outcome = CandidateOutcome.model_validate(diagnostics["candidate-outcome"])
            return (
                outcome.reason_code if outcome.reason_code == "weights-missing" else outcome.outcome
            )
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            subprocess.SubprocessError,
        ):
            return None


def order_by_cooldown(candidates: list[dict], recent: dict[str, str | None]) -> list[dict]:
    """Return ``candidates`` never tried recently, then those whose recent outcome retries now.

    An abort is lifted by removing its label, so it never delays the next attempt by itself.
    """
    fresh = [candidate for candidate in candidates if candidate["id"] not in recent]
    deferred = [
        candidate for candidate in candidates if recent.get(candidate["id"], "") in RETRY_NEXT_WAVE
    ]
    return fresh + deferred


def aborted_candidates(repository: str) -> set[str]:
    """Candidates whose Klaud PR carries klaud-abort; the planner never retries them.

    Removing the label returns the candidate to the pool at the planner's next scan.
    """
    from .lifecycle import BOT

    aborted = set()
    for issue in github_items(repository, f"issues?labels={ABORT_LABEL}&state=all&per_page=100"):
        if "pull_request" not in issue or issue["user"]["login"] != BOT:
            continue
        pull = github_read(repository, f"pulls/{issue['number']}")
        if match := re.fullmatch(r"klaud/auto-([0-9a-f]{16}-[0-9a-f]{16})", pull["head"]["ref"]):
            aborted.add(match[1])
    return aborted


def abort_released(repository: str, candidate_id: str) -> bool:
    """Whether the candidate's latest Klaud PR ended in an abort whose label was since removed."""
    from urllib.parse import urlencode

    from .lifecycle import BOT

    query = urlencode(
        {
            "state": "closed",
            "head": f"{repository.split('/', maxsplit=1)[0]}:klaud/auto-{candidate_id}",
            "per_page": 100,
        }
    )
    try:
        pulls = [
            pull
            for pull in github_items(repository, "pulls?" + query)
            if pull["user"]["login"] == BOT
        ]
        if not pulls:
            return False
        latest = max(pulls, key=lambda pull: pull["created_at"])
        if any(label["name"] == ABORT_LABEL for label in latest["labels"]):
            return False
        comments = github_items(repository, f"issues/{latest['number']}/comments?per_page=100")
        for comment in reversed(comments):
            if comment["user"]["login"] == BOT and comment["body"].startswith(
                "<!-- klaud-outcome:"
            ):
                record = json.loads(comment["body"].split("\n", 1)[1].split("\n-->", 1)[0])
                return record["outcome"]["outcome"] == "aborted"
    except (OSError, ValueError, KeyError, TypeError, IndexError, subprocess.SubprocessError):
        print(f"::warning::Klaud abort history unavailable for {candidate_id}; keeping cooldown")
    return False


def active_pulls(pulls: list[dict], now: datetime, stale_days: int) -> list[dict]:
    """Return ``pulls`` on Klaud branches or updated within ``stale_days`` before ``now``."""
    cutoff = now - timedelta(days=stale_days)
    return [
        pr
        for pr in pulls
        if re.match(r"^klaud[e]?/auto-", pr["head"]["ref"])
        or datetime.fromisoformat(pr["updated_at"].replace("Z", "+00:00")) >= cutoff
    ]


def plan(
    root: Path, directory: Path, review_batch_size: int, cooldown_hours: int, stale_pr_days: int
) -> None:
    """Write ``candidates.json`` and ``open-prs.json`` under ``directory`` for overlap review.

    Unclaimed, unaborted feed candidates at ``root``'s HEAD are cooled down by
    ``cooldown_hours`` and capped at ``review_batch_size``. PRs idle ``stale_pr_days`` drop out.
    """
    from . import claims

    policy = Policy()
    repository = os.environ["GITHUB_REPOSITORY"]
    base = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, timeout=30
    ).strip()
    items, issues = fetch_catalog(policy)
    if issues:
        raise ReadError("public-feed-invalid: " + ", ".join(issues))
    refs = github_read(repository, "git/matching-refs/heads/klaud", paginate=True)
    occupied = {ref["ref"].removeprefix("refs/heads/") for page in refs for ref in page}
    pulls = github_read(repository, "pulls?state=open&per_page=100", paginate=True)
    occupied.update(pr["head"]["ref"] for page in pulls for pr in page)
    occupied.update(
        pr["head"]["ref"].rsplit("-", 1)[0] + "-"
        for page in pulls
        for pr in page
        if re.match(r"^klaud[e]?/auto-", pr["head"]["ref"])
    )
    candidates = choose(items, fetch_capacity(policy), occupied, live_families(root))
    blocked = {
        candidate.family
        for owner in claims.family_owners(repository)
        for candidate in owner.candidates
    }
    recovery_file = Path(os.environ.get("RUNNER_TEMP", directory.parent)) / "klaud-recovery.json"
    if recovery_file.exists():
        blocked.update(json.loads(recovery_file.read_text()))
    candidates = [candidate for candidate in candidates if candidate["family"] not in blocked]
    aborted = aborted_candidates(repository)
    candidates = [candidate for candidate in candidates if candidate["id"] not in aborted]
    recent, missing_weights = recent_candidate_outcomes(repository, cooldown_hours)
    for candidate in candidates:
        held = candidate["id"] in recent and recent[candidate["id"]] not in RETRY_NEXT_WAVE
        if held and abort_released(repository, candidate["id"]):
            recent[candidate["id"]] = "aborted"
    candidates = [
        candidate
        for candidate in candidates
        if weights_key(candidate["source"]) not in missing_weights
    ]
    candidates = order_by_cooldown(candidates, recent)
    candidates = candidates[:review_batch_size]
    contexts = [
        {
            **candidate,
            "base": base,
            "repository": repository,
            "public-api": {
                "schema": PUBLIC + "/api/openapi.json",
                "images": PUBLIC + "/api/v1/latest-images",
                "benchmarks": PUBLIC + "/api/v1/benchmarks",
                "workflow-info": PUBLIC + "/api/v1/workflow-info",
            },
        }
        for candidate in candidates
    ]
    open_prs = [
        {
            "number": pr["number"],
            "title": pr["title"],
            "url": pr["html_url"],
            "author": pr["user"]["login"],
            "draft": pr["draft"],
            "branch": pr["head"]["ref"],
            "head": pr["head"]["sha"],
        }
        for pr in active_pulls(
            [pr for page in pulls for pr in page], datetime.now(UTC), stale_pr_days
        )
    ]
    if candidates:
        for pr in open_prs:
            try:
                pages = github_read(
                    repository,
                    f"pulls/{pr['number']}/files?per_page=100",
                    paginate=True,
                )
            except (subprocess.SubprocessError, ValueError):
                pr.update({"files": [], "files-complete": False})
                continue
            pr["files"] = sorted(
                {
                    name
                    for page in pages
                    for file in page
                    for name in (file["filename"], file.get("previous_filename"))
                    if name
                }
            )
            pr["files-complete"] = sum(len(page) for page in pages) < 3000
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "candidates.json").write_text(
        json.dumps(contexts, indent=2, allow_nan=False) + "\n"
    )
    (directory / "open-prs.json").write_text(json.dumps(open_prs, indent=2) + "\n")
    if filename := os.environ.get("GITHUB_OUTPUT"):
        schema = PRReview.model_json_schema(by_alias=True)
        schema["properties"]["decisions"].update(minItems=len(contexts), maxItems=len(contexts))
        with open(filename, "a") as output:
            output.write(f"has_candidates={str(bool(candidates)).lower()}\n")
            output.write(f"review_schema={json.dumps(schema)}\n")


def denied_bash_category(denial: dict) -> str:
    """Return a fixed category for the Bash command in permission ``denial``.

    Command text, arguments and paths are never retained.
    """
    command = denial.get("tool_input", {})
    command = command.get("command") if isinstance(command, dict) else None
    if not isinstance(command, str):
        return "unknown"
    try:
        words = shlex.split(command)
    except ValueError:
        return "unknown"
    if not words:
        return "unknown"
    if words[0] in {"cd", "env", "bash", "sh"} or "=" in words[0]:
        return "shell-wrapper"
    if words[0] in {
        "cat",
        "ls",
        "find",
        "sed",
        "awk",
        "rg",
        "grep",
        "head",
        "tail",
        "pwd",
        "wc",
        "jq",
    }:
        return "file-read-or-filter"
    if words[0] in {"python", "python3", "uv"}:
        return "python-or-uv"
    if words[0] == "git":
        return "git"
    if words[0] == "gh":
        return "github-cli"
    return "other"


def execution_diagnostics(path: Path) -> dict:
    """Return numeric metrics, termination and denial categories from the log at ``path``."""
    try:
        messages = json.loads(path.read_text())
        if not isinstance(messages, list):
            raise ValueError("Expected execution message list")
    except (OSError, ValueError):
        return {"execution-log": "unavailable"}
    result = next(
        (
            message
            for message in reversed(messages)
            if isinstance(message, dict) and message.get("type") == "result"
        ),
        {},
    )
    metrics = {
        key: value
        for key in ("duration_ms", "num_turns", "total_cost_usd")
        if (
            type(value := result.get(key)) is int or (type(value) is float and math.isfinite(value))
        )
        and value >= 0
    }
    denials = result.get("permission_denials", [])
    if not isinstance(denials, list):
        denials = []
    tools = {
        "Read",
        "Glob",
        "Grep",
        "Bash",
        "WebFetch",
        "WebSearch",
        "Write",
        "Edit",
        "Agent",
        "Task",
        "StructuredOutput",
    }
    denied_tools = Counter(
        denial["tool_name"]
        if isinstance(denial.get("tool_name"), str) and denial["tool_name"] in tools
        else "other"
        for denial in denials
        if isinstance(denial, dict)
    )
    subtypes = {
        "success",
        "error_max_turns",
        "error_during_execution",
        "error_max_budget_usd",
        "error_max_structured_output_retries",
    }
    subtype = result.get("subtype")
    return {
        "execution-log": "available",
        "result-present": bool(result),
        **metrics,
        "structured-output-present": isinstance(result.get("structured_output"), dict),
        "termination": subtype if isinstance(subtype, str) and subtype in subtypes else "unknown",
        "is-error": result.get("is_error") if type(result.get("is_error")) is bool else None,
        "permission-denials": dict(denied_tools),
        "denied-bash-categories": dict(
            Counter(
                denied_bash_category(denial)
                for denial in denials
                if isinstance(denial, dict) and denial.get("tool_name") == "Bash"
            )
        ),
    }


def check_stop() -> dict:
    """Return the Stop hook decision, blocking unless a verified terminal outcome exists.

    An aborted candidate never blocks, since the recover-current step cleans it up. A session
    that parked may stop while a run it parked on is still live, or if it parked at its deadline.
    """
    from .lifecycle import current_session, parked, terminal

    try:
        session = current_session()
        pulls = session.pulls()
        if pulls and session.aborted(pulls[0]):
            return {}
        evidence = Path(os.environ["KLAUD_EVIDENCE"])
        end = evidence / "session-end.json"
        if end.exists() and SessionEnd.model_validate_json(end.read_text()).status == "parked":
            waiting = parked(session, pulls[0], int(os.environ["KLAUD_SESSION"])) if pulls else None
            if waiting is None:
                raise VerificationError("Session has not parked")
            if waiting and all(terminal(run) for run in session.runs() if run["id"] in waiting):
                return {
                    "decision": "block",
                    "reason": "Every run this session parked on has ended. Continue: inspect them, report the attempt, then repair, park again or finish.",
                }
            return {}
        outcome = CandidateOutcome.model_validate_json((evidence / "outcome.json").read_text())
        session.verify(outcome)
    except (KeyError, TypeError, ValueError, subprocess.SubprocessError, OSError):
        return {
            "decision": "block",
            "reason": "No verified terminal outcome. Continue monitoring and repairs, then run the documented finish command to validate or complete failure/deferral cleanup and reporting, or park while owned runs are queued or running. Inspect read failures; never dispatch replacements or cancel healthy work just to stop.",
        }
    return {}


def check_deadline() -> dict:
    """Return the PostToolUse hook output telling a session near its time limit to park."""
    from .lifecycle import PARK_AFTER_SECONDS, SESSION_SECONDS

    started = os.environ.get("KLAUD_SESSION_STARTED")
    if not started:
        return {}
    elapsed = datetime.now(UTC).timestamp() - int(started)
    if elapsed < PARK_AFTER_SECONDS:
        return {}
    minutes = max(0, int((SESSION_SECONDS - elapsed) // 60))
    return {
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": f"About {minutes} minutes remain in this session. Publish attempt reports for owned runs now, then run park, or finish if the outcome is terminal, and end with StructuredOutput.",
        }
    }


def resume_check() -> dict[str, bool]:
    """The session gate's outputs.

    A planner run's job runs its session in the run's first attempt only. A resume run runs the
    session recovery granted it, once: only in its first attempt, only if recovery dispatched it
    for that later session, and never after another run already started that session.
    """
    no = {"run": False, "resume": False}
    run = os.environ.get("GITHUB_RUN_ID")
    attempt = int(os.environ.get("GITHUB_RUN_ATTEMPT", "1"))
    if (os.environ.get("KLAUD_PARENT_RUN") or run) == run:
        return {"run": attempt == 1, "resume": False}
    number = int(os.environ.get("KLAUD_SESSION") or "0")
    if attempt != 1 or number < 2:
        return no
    from .lifecycle import (
        MAX_SESSIONS,
        candidate_jobs,
        current_session,
        granted,
        resumable,
        resume_title,
        started_session,
    )

    session = current_session()
    title = resume_title(session.parent["id"], session.candidate.id, number)
    same = {resume["id"] for resume in session.resume_runs() if resume["display_title"] == title}
    if run is None or int(run) not in same:
        return no
    jobs = [job for job in candidate_jobs(session) if started_session(job)]
    if len(jobs) > MAX_SESSIONS or any(
        job["run_id"] in same and job["run_id"] != int(run) for job in jobs
    ):
        return no
    pulls = session.pulls()
    pull = pulls[0] if pulls else None
    if not resumable(session, pull):
        return no
    resume = granted(session, pull, number)
    return {"run": resume, "resume": resume}


def ended_parked(action_outcome: str) -> bool:
    """Return True if the agent step succeeded and the session's last end record is a park."""
    evidence = os.environ.get("KLAUD_EVIDENCE")
    if action_outcome != "success" or not evidence:
        return False
    try:
        end = SessionEnd.model_validate_json((Path(evidence) / "session-end.json").read_text())
    except (OSError, ValueError):
        return False
    return end.status == "parked"


def save_diagnostics(
    execution_file: Path, outcome_file: Path, action_outcome: str, output: Path
) -> bool:
    """Write sanitized session diagnostics to ``output`` and return False if unverified.

    The outcome comes from the posted receipt, the evidence outcome, then ``outcome_file``. A
    session whose agent ended cleanly right after parking on its unfinished draft PR has no
    outcome and counts as verified.
    """
    diagnostics = {
        "action-outcome": action_outcome,
        **execution_diagnostics(execution_file),
    }
    failure = None
    outcome = None
    parked_runs = None
    try:
        from .lifecycle import current_session, parked, resumable

        session = current_session()
        pulls, runs = session.pulls(), session.runs()
        diagnostics["observed-session"] = {
            "pull-request": pulls[0]["number"] if pulls else None,
            "head": pulls[0]["head"]["sha"] if pulls else None,
            "runs": [
                {
                    "id": run["id"],
                    "attempt": run["run_attempt"],
                    "terminal": run["status"] == "completed",
                }
                for run in runs
            ],
        }
        receipt = session.report(pulls[0]) if pulls else None
        if not receipt and pulls and ended_parked(action_outcome) and resumable(session, pulls[0]):
            parked_runs = parked(session, pulls[0], int(os.environ.get("KLAUD_SESSION") or "1"))
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        failure = "session-state-unavailable"
    else:
        if receipt:
            try:
                outcome = CandidateOutcome.model_validate(receipt["outcome"])
            except (ValueError, KeyError, TypeError):
                failure = "receipt-invalid"
            else:
                diagnostics["outcome-source"] = "verified-receipt"
        elif parked_runs is None:
            evidence = os.environ.get("KLAUD_EVIDENCE")
            if evidence:
                try:
                    outcome = CandidateOutcome.model_validate_json(
                        (Path(evidence) / "outcome.json").read_text()
                    )
                except (OSError, ValueError, TypeError):
                    outcome = None
                else:
                    diagnostics["outcome-source"] = "verified-outcome"
            if outcome is None:
                try:
                    outcome = SessionEnd.model_validate_json(outcome_file.read_text()).outcome
                except (OSError, ValueError, TypeError):
                    outcome = None
                if outcome is None:
                    failure = "candidate-outcome-unavailable"
                else:
                    diagnostics["outcome-source"] = "structured-response"
        if outcome is not None and failure is None:
            try:
                session.verify(outcome)
            except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
                failure = "lifecycle-unverified"
    if parked_runs is not None and failure is None:
        diagnostics["session-state"] = "parked"
        diagnostics["parked-runs"] = len(parked_runs)
        diagnostics["candidate-outcome"] = None
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(diagnostics, indent=2) + "\n")
        if filename := os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(filename, "a") as summary:
                summary.write(
                    f"Parked on {len(parked_runs)} owned runs; recovery resumes the candidate. / "
                    f"已等待 {len(parked_runs)} 个自有运行；恢复流程将继续该候选。\n"
                )
        return True
    if failure is not None:
        observed = diagnostics.get("observed-session", {})
        outcome = CandidateOutcome(
            outcome="unexpected-error",
            phase="unknown",
            pull_request=observed.get("pull-request"),
            run_ids=[run["id"] for run in observed.get("runs", [])],
            repairs_used=None,
        )
        diagnostics["outcome-error"] = failure
        diagnostics["outcome-report"] = "unavailable-or-invalid"
    else:
        diagnostics["outcome-report"] = "available"
        diagnostics["session-state"] = "finished"
    diagnostics["candidate-outcome"] = outcome.model_dump(by_alias=True, exclude_unset=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(diagnostics, indent=2) + "\n")
    if filename := os.environ.get("GITHUB_STEP_SUMMARY"):
        repository = os.environ["GITHUB_REPOSITORY"]
        prefix = f"https://github.com/{repository}"
        pr = (
            f"[#{outcome.pull_request}]({prefix}/pull/{outcome.pull_request})"
            if outcome.pull_request
            else "—"
        )
        runs = ", ".join(f"[{run}]({prefix}/actions/runs/{run})" for run in outcome.run_ids) or "—"
        repairs = (
            str(outcome.repairs_used) if outcome.repairs_used is not None else "unknown / 未知"
        )
        reason = outcome.reason_code or "—"
        with open(filename, "a") as summary:
            summary.write(
                "| Outcome / 结果 | Phase / 阶段 | Reason / 原因 | PR | Repairs / 修复 | Runs / 运行 |\n"
                "| --- | --- | --- | --- | --- | --- |\n"
                f"| {outcome.outcome} | {outcome.phase} | {reason} | {pr} | {repairs} | {runs} |\n"
            )
    return outcome.outcome != "unexpected-error"


def reviewed(contexts: list[dict]) -> tuple[PRReview, str | None]:
    """The overlap review of ``contexts`` (``KLAUD_PR_REVIEW``), or none and why selection defers."""
    if not contexts:
        return PRReview(decisions=[]), None
    if os.environ.get("KLAUD_REVIEW_OUTCOME", "success") != "success":
        return PRReview(decisions=[]), "review-action-failed"
    try:
        review = PRReview.model_validate_json(os.environ.get("KLAUD_PR_REVIEW", ""))
        ids = [decision.candidate_id for decision in review.decisions]
        if len(set(ids)) != len(ids) or set(ids) != {candidate["id"] for candidate in contexts}:
            raise ValueError("Review must cover every candidate exactly once")
        expected = {candidate["id"]: candidate["family"] for candidate in contexts}
        if any(
            decision.family != expected[decision.candidate_id]
            for decision in review.decisions
            if decision.decision == "proceed"
        ):
            raise ValueError("Review changed the resolved live family")
    except ValueError:
        return PRReview(decisions=[]), "invalid-review-output"
    return review, None


def print_untrusted(title: str, text: str) -> None:
    """Print another revision's output in a log group where it cannot issue workflow commands."""
    token = secrets.token_hex(16)
    print(f"::group::{title}\n::stop-commands::{token}\n{text.rstrip()}\n::{token}::\n::endgroup::")


def regenerate_producers(directory: Path) -> None:
    """Write ``producers.json``: each proceed candidate's producer families, by family and head.

    Producer revisions regenerate their own families, so this step runs without credentials.
    A producer that fails is logged and left out; ``select`` defers the candidates needing it.
    """
    from .reporting import publication
    from .validation import ProducerRegenerationError, producer_matrix

    contexts = json.loads((directory / "candidates.json").read_text())
    review, _ = reviewed(contexts)
    models = {
        decision.candidate_id: decision.baseline_model
        for decision in review.decisions
        if decision.decision == "proceed"
    }
    repository = os.environ["GITHUB_REPOSITORY"]
    families: dict[str, dict[str, dict]] = {}
    for candidate in contexts:
        if (model := models.get(candidate["id"])) is None:
            continue
        try:
            heads = publication(repository, candidate, model).heads
        except (OSError, ValueError, KeyError, TypeError):
            print(
                f"::warning::Klaud: public producers of candidate {candidate['id']} are unavailable."
            )
            continue
        regenerated = families.setdefault(candidate["family"], {})
        for head in heads:
            if head in regenerated:
                continue
            try:
                regenerated[head] = producer_matrix(repository, head, candidate["family"])
            except ProducerRegenerationError as error:
                print_untrusted(
                    f"Producer {head} of {candidate['family']} cannot be regenerated", error.stderr
                )
    (directory / PRODUCERS).write_text(json.dumps(families, allow_nan=False) + "\n")


def select(
    directory: Path,
    max_candidates: int,
    execution_file: Path | None = None,
    active_sessions: int = 0,
) -> None:
    """Claim reviewed candidates that fit capacity and have a verified baseline.

    Takes up to ``max_candidates`` less the ``active_sessions`` already running or resumed,
    which share the cap, and writes their context, ownership.json and selection.json under
    ``directory``.
    """
    from . import claims, validation
    from .reporting import BaselinePreflight, Prose, missing_baseline_points, resolve_baseline

    contexts = json.loads((directory / "candidates.json").read_text())
    producers = json.loads((directory / PRODUCERS).read_text())
    review, deferred = reviewed(contexts)
    decisions = {decision.candidate_id: decision for decision in review.decisions}
    selected = []
    slots = max(0, max_candidates - active_sessions)
    headroom: dict[str, int] = {}
    if slots and any(decision.decision == "proceed" for decision in review.decisions):
        try:
            headroom = fetch_capacity(Policy())
        except ReadError:
            deferred = "capacity-unavailable"
    capacity_deferred = []
    baseline_deferred = []
    baseline_reasons: dict[str, str] = {}
    baseline_mismatches = []
    preflights = {}
    families = {decision.family for decision in review.decisions if decision.decision != "proceed"}
    for candidate in contexts:
        if len(selected) >= slots:
            break
        decision = decisions.get(candidate["id"])
        if decision is None or decision.decision != "proceed" or decision.family in families:
            continue
        clusters, demand = candidate["clusters"], candidate["node-demand"]
        if any(headroom.get(cluster, 0) < demand for cluster in clusters):
            capacity_deferred.append(candidate["id"])
            continue
        owned = OwnedCandidate.model_validate(
            {key: candidate[key] for key in ("id", "family", "base")}
        )
        try:
            matrix = validation.canonical_matrix(
                os.environ["GITHUB_REPOSITORY"], owned.base, owned.family
            )
            baseline = resolve_baseline(
                os.environ["GITHUB_REPOSITORY"],
                owned,
                candidate,
                decision.baseline_model,
                Prose(
                    en="Verify the complete published baseline before candidate dispatch.",
                    zh="在调度候选任务前验证完整的已发布基线。",
                ),
                producers.get(candidate["family"], {}),
                matrix,
            )
            preflight = BaselinePreflight(
                candidate_id=candidate["id"],
                base=candidate["base"],
                baseline_model=decision.baseline_model,
                source_identity=identity(candidate["source"]),
                baseline=baseline,
            )
            missing = missing_baseline_points(matrix, baseline)
        except (VerificationError, ReadError) as error:
            baseline_deferred.append(candidate["id"])
            baseline_reasons[candidate["id"]] = str(error)
            families.add(decision.family)
            continue
        except subprocess.SubprocessError as error:
            baseline_deferred.append(candidate["id"])
            baseline_reasons[candidate["id"]] = type(error).__name__
            families.add(decision.family)
            continue
        except (OSError, ValueError, KeyError, TypeError) as error:
            baseline_deferred.append(candidate["id"])
            baseline_reasons[candidate["id"]] = f"{type(error).__name__}: {str(error)[:300]}"
            families.add(decision.family)
            continue
        if missing:
            baseline_deferred.append(candidate["id"])
            baseline_reasons[candidate["id"]] = "baseline-point-mismatch"
            baseline_mismatches.append(
                {
                    "id": candidate["id"],
                    "reason": "baseline-point-mismatch",
                    "missing-points": len(missing),
                }
            )
            families.add(decision.family)
            continue
        if not claims.claim_family(
            os.environ["GITHUB_REPOSITORY"], owned, int(os.environ["GITHUB_RUN_ID"])
        ):
            continue
        for cluster in clusters:
            headroom[cluster] -= demand
        selected.append(
            {
                **candidate,
                "baseline-model": decision.baseline_model,
                "pr-review": decision.model_dump(by_alias=True),
            }
        )
        preflights[candidate["id"]] = preflight
        families.add(decision.family)
    for candidate in selected:
        target = directory / candidate["id"]
        target.mkdir(parents=True, exist_ok=True)
        (target / "candidate.json").write_text(
            json.dumps(candidate, indent=2, allow_nan=False) + "\n"
        )
        (target / "baseline-preflight.json").write_text(
            preflights[candidate["id"]].model_dump_json(by_alias=True) + "\n"
        )
    ownership = Ownership(
        run_id=int(os.environ["GITHUB_RUN_ID"]),
        candidates=[
            OwnedCandidate.model_validate({key: candidate[key] for key in ("id", "family", "base")})
            for candidate in selected
        ],
    )
    (directory / "ownership.json").write_text(ownership.model_dump_json(by_alias=True) + "\n")
    candidates = [candidate["id"] for candidate in selected]
    (directory / "selection.json").write_text(
        json.dumps(
            {
                "candidates": candidates,
                "deferred-reason": deferred,
                "capacity-deferred-candidates": capacity_deferred,
                "baseline-deferred-candidates": baseline_deferred,
                "baseline-deferral-reasons": baseline_reasons,
                "baseline-mismatch-candidates": baseline_mismatches,
                **review.model_dump(by_alias=True),
            },
            indent=2,
        )
        + "\n"
    )
    if execution_file is not None:
        (directory / "review-diagnostics.json").write_text(
            json.dumps(execution_diagnostics(execution_file), indent=2) + "\n"
        )
    summary = f"Klaud Cold: selected {len(candidates)} of {len(contexts)} eligible candidates."
    if active_sessions:
        summary += f" {active_sessions} running or resumed sessions leave " + (
            f"{slots} of {max_candidates} slots." if slots else "no slot for new candidates."
        )
    if deferred:
        summary += f" Selection stopped: {deferred}."
        print(f"::warning::{summary}")
    if capacity_deferred:
        summary += (
            f" {len(capacity_deferred)} reviewed candidates deferred by the latest capacity check"
            " (a target cluster had queued human jobs or too few idle nodes)."
        )
    if baseline_deferred:
        summary += f" {len(baseline_deferred)} candidates lacked a verifiable full baseline: " + (
            "; ".join(
                f"{count}× {reason[:120]}"
                for reason, count in Counter(baseline_reasons.values()).most_common()
            )
            + "."
        )
    if baseline_mismatches:
        summary += (
            f" {len(baseline_mismatches)} of them have frozen baseline points"
            " the current family no longer generates."
        )
    if filename := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(filename, "a") as output:
            output.write(
                summary
                + "\n\nSee selection.json and review-diagnostics.json in the klaud-plan artifact.\n"
            )
    if filename := os.environ.get("GITHUB_OUTPUT"):
        with open(filename, "a") as output:
            output.write(
                f"selected={str(bool(candidates)).lower()}\ncandidates={json.dumps(candidates)}\n"
            )


def main() -> int:
    """Run the Klaud subcommand named on the command line and return its exit status."""
    parser = argparse.ArgumentParser(prog="python -m infx.klaud", description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="InferenceX checkout used to resolve the candidate base SHA",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("plan", help="Prepare candidates and open PRs for overlap review")
    prepare.add_argument(
        "--directory",
        type=Path,
        required=True,
        help="Output directory for candidate context",
    )
    prepare.add_argument(
        "--review-batch-size",
        type=int,
        required=True,
        help="Maximum shuffled candidates sent to one overlap review (1-256)",
    )
    prepare.add_argument(
        "--cooldown-hours",
        type=int,
        required=True,
        help="Hours a recently attempted candidate waits before another agent (1-168)",
    )
    prepare.add_argument(
        "--stale-pr-days",
        type=int,
        required=True,
        help="Days without activity after which another open PR stops owning a family (1-90)",
    )
    regeneration = commands.add_parser(
        "regenerate-producers",
        help="Regenerate reviewed candidates' baseline producer families; run without credentials",
    )
    regeneration.add_argument("--directory", type=Path, required=True)
    selection = commands.add_parser(
        "select", help="Validate KLAUD_PR_REVIEW and select nonoverlapping candidates"
    )
    selection.add_argument("--directory", type=Path, required=True)
    selection.add_argument(
        "--max-candidates-per-run",
        type=int,
        required=True,
        help="Maximum candidates to select (1-256)",
    )
    selection.add_argument(
        "--execution-file",
        type=Path,
        help="Claude execution log; retain numeric metrics and fixed denial categories",
    )
    selection.add_argument(
        "--active-sessions",
        type=int,
        default=0,
        help="Klaud sessions running or resumed by recovery; each takes one candidate slot",
    )
    commands.add_parser(
        "check-stop",
        help="Claude Stop hook: require finished runs and validated or closed PRs, or a park",
    )
    commands.add_parser(
        "check-deadline",
        help="Claude PostToolUse hook: tell a session near its time limit to park",
    )
    commands.add_parser(
        "park",
        help="Record that this session waits on its reported live owned runs, then stop",
    )
    recovery = commands.add_parser(
        "recover", help="Reconcile interrupted sessions from completed autosweeps"
    )
    recovery.add_argument(
        "--max-sessions",
        type=int,
        required=True,
        help="Most Klaud sessions running at once, counting the resumes it grants (0-256)",
    )
    commands.add_parser(
        "recover-current",
        help="Reconcile after an interrupted agent without waiting for healthy children",
    )
    commands.add_parser(
        "resume-check",
        help="Gate a candidate job: run the planner's session or a resume run's granted session",
    )
    preflight = commands.add_parser(
        "check-final",
        help="Check a generated final matrix against the complete exact-head family",
    )
    preflight.add_argument("--matrix-file", type=Path, required=True)
    baseline = commands.add_parser(
        "prepare-baseline",
        help="Freeze the planner's verified baseline preflight, or keep the one the PR froze",
    )
    baseline.add_argument(
        "--model", required=True, help="Display model name from the public OpenAPI enum"
    )
    baseline.add_argument(
        "--goal-file",
        type=Path,
        required=True,
        help="JSON with en/zh goal sentences naming the engine and old/new images",
    )
    baseline.add_argument("--output", type=Path, required=True)
    report = commands.add_parser(
        "report",
        help="Publish a typed baseline or owned-attempt record using canonical templates",
    )
    report.add_argument("--kind", choices=["baseline", "attempt"], required=True)
    report.add_argument("--file", type=Path, required=True)
    schema = commands.add_parser("report-schema", help="Print the baseline or attempt JSON schema")
    schema.add_argument("--kind", choices=["baseline", "attempt"], required=True)
    finish = commands.add_parser(
        "finish", help="Verify validation or finish owned cleanup and reporting"
    )
    finish.add_argument("--outcome-file", type=Path, required=True)
    commands.add_parser(
        "session-end-schema", help="Print the public-safe schema of how a session ended"
    )
    diagnostics = commands.add_parser(
        "diagnostics",
        help="Save sanitized candidate outcomes, termination metrics and permission categories",
    )
    diagnostics.add_argument("--execution-file", type=Path, required=True)
    diagnostics.add_argument("--structured-outcome-file", type=Path, required=True)
    diagnostics.add_argument("--output", type=Path, required=True)
    diagnostics.add_argument(
        "--outcome",
        choices=["success", "failure", "cancelled", "skipped", "unknown"],
        default="unknown",
    )
    args = parser.parse_args()
    try:
        if args.command in ("report", "report-schema", "prepare-baseline"):
            from .lifecycle import current_session
            from .reporting import (
                Attempt,
                Baseline,
                Prose,
                baseline_for,
                check_reported_datasets,
                prepare_baseline,
                publish,
            )

            if args.command == "prepare-baseline":
                if args.output.exists():
                    Baseline.model_validate_json(args.output.read_text())
                    return 0
                session = current_session()
                pulls = session.pulls()
                record = baseline_for(session, pulls[0]) if pulls else None
                if record is None:
                    context = json.loads(
                        (Path(os.environ["KLAUD_EVIDENCE"]) / "candidate.json").read_text()
                    )
                    goal = Prose.model_validate_json(args.goal_file.read_text())
                    record = prepare_baseline(session, context, args.model, goal)
                with args.output.open("x") as output:
                    output.write(record.model_dump_json(by_alias=True) + "\n")
                return 0
            model = Baseline if args.kind == "baseline" else Attempt
            if args.command == "report-schema":
                print(json.dumps(model.model_json_schema(by_alias=True)))
            else:
                record = model.model_validate_json(args.file.read_text())
                if isinstance(record, Attempt):
                    check_reported_datasets(record)
                publish(current_session(), record)
            return 0
        if args.command == "check-final":
            from .lifecycle import current_session
            from .reporting import baseline_for, check_baseline_coverage
            from .validation import canonical_matrix, check_matrix

            session = current_session()
            pulls = session.pulls()
            if not pulls:
                raise VerificationError("Candidate PR not found")
            pull = pulls[0]
            head = pull["head"]["sha"]
            matrix = json.loads(args.matrix_file.read_text())
            canonical = canonical_matrix(session.repository, head, session.candidate.family)
            check_matrix(
                matrix,
                canonical,
                head,
                session.candidate.family,
            )
            check_baseline_coverage(
                session.repository, session.candidate, canonical, baseline_for(session, pull)
            )
            return 0
        if args.command == "recover-current":
            from . import claims
            from .lifecycle import PendingCleanup, current_session, reconcile

            session = current_session()
            try:
                reconcile(session, datetime.now(UTC))
            except PendingCleanup:
                pass
            else:
                claims.release_family(
                    session.repository,
                    session.candidate,
                    session.parent["id"],
                )
            return 0
        if args.command == "resume-check":
            outputs = "".join(
                f"{key}={str(value).lower()}\n" for key, value in resume_check().items()
            )
            print(outputs, end="")
            if filename := os.environ.get("GITHUB_OUTPUT"):
                with open(filename, "a") as output:
                    output.write(outputs)
            return 0
        if args.command == "recover":
            from .lifecycle import recover

            if not 0 <= args.max_sessions <= 256:
                parser.error("--max-sessions must be between 0 and 256")
            recover(args.max_sessions)
            return 0
        if args.command == "finish":
            from .lifecycle import current_session

            outcome = CandidateOutcome.model_validate_json(args.outcome_file.read_text())
            if (
                outcome.outcome == "failed"
                and outcome.phase == "baseline"
                and not outcome.reason_code
            ):
                raise VerificationError("Baseline failure requires a fixed reason-code")
            outcome = current_session().finish(outcome)
            evidence = Path(os.environ["KLAUD_EVIDENCE"])
            (evidence / "outcome.json").write_text(
                outcome.model_dump_json(by_alias=True, exclude_unset=True) + "\n"
            )
            (evidence / "session-end.json").write_text(
                SessionEnd(status="finished", outcome=outcome).model_dump_json(
                    by_alias=True, exclude_unset=True
                )
                + "\n"
            )
            return 0
        if args.command == "park":
            from .lifecycle import PARK_AFTER_SECONDS, current_session, park

            started = int(os.environ["KLAUD_SESSION_STARTED"])
            park(
                current_session(),
                int(os.environ["KLAUD_SESSION"]),
                overdue=datetime.now(UTC).timestamp() - started >= PARK_AFTER_SECONDS,
            )
            (Path(os.environ["KLAUD_EVIDENCE"]) / "session-end.json").write_text(
                SessionEnd(status="parked").model_dump_json(by_alias=True, exclude_unset=True)
                + "\n"
            )
            return 0
        if args.command == "session-end-schema":
            print(json.dumps(SessionEnd.model_json_schema(by_alias=True)))
            return 0
        if args.command == "check-deadline":
            print(json.dumps(check_deadline()))
            return 0
        if args.command == "check-stop":
            print(json.dumps(check_stop()))
            return 0
        if args.command == "diagnostics":
            return (
                0
                if save_diagnostics(
                    args.execution_file,
                    args.structured_outcome_file,
                    args.outcome,
                    args.output,
                )
                else 1
            )
        if args.command == "plan":
            if not 1 <= args.review_batch_size <= 256:
                parser.error("--review-batch-size must be between 1 and 256")
            if not 1 <= args.cooldown_hours <= 168:
                parser.error("--cooldown-hours must be between 1 and 168")
            if not 1 <= args.stale_pr_days <= 90:
                parser.error("--stale-pr-days must be between 1 and 90")
            plan(
                args.root,
                args.directory,
                args.review_batch_size,
                args.cooldown_hours,
                args.stale_pr_days,
            )
            return 0
        if args.command == "regenerate-producers":
            regenerate_producers(args.directory)
            return 0
        if not 1 <= args.max_candidates_per_run <= 256:
            parser.error("--max-candidates-per-run must be between 1 and 256")
        if args.active_sessions < 0:
            parser.error("--active-sessions must not be negative")
        select(
            args.directory, args.max_candidates_per_run, args.execution_file, args.active_sessions
        )
        return 0
    except ReadError as error:
        if args.command == "plan":
            message = f"Klaud Cold preparation failed: {error}"
            print(f"::error::{message}")
            if filename := os.environ.get("GITHUB_STEP_SUMMARY"):
                with open(filename, "a") as output:
                    output.write(message + "\n")
        return 1
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
    ) as error:
        if isinstance(error, VerificationError):
            reason = str(error)
        elif isinstance(error, ValidationError) and args.command == "report":
            reason = "Invalid record: " + "; ".join(
                ".".join(str(part) for part in item["loc"]) + ": " + item["msg"]
                if item["loc"]
                else item["msg"]
                for item in error.errors(include_url=False)
            )
        else:
            reason = "State unavailable or invalid; inspect GitHub before retrying"
        print(f"::error::Klaud: {reason}.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
