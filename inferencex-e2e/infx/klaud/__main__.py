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
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, NamedTuple

from pydantic import ValidationError

from . import github
from .api import (
    KLAUD_QUEUE_MAX_AGE,
    PUBLIC,
    Capacity,
    ReadError,
    fetch_capacity,
    fetch_catalog,
)
from .github import (
    VerificationError,
    download_json,
    items as github_items,
    read as github_read,
)
from .models import (
    ASSET_REASONS,
    INFRA_REASONS,
    CandidateOutcome,
    OwnedCandidate,
    Ownership,
    Policy,
    PRReview,
    SessionEnd,
    identity,
    key_families,
    normalized_image,
    utc,
)

if TYPE_CHECKING:
    from infx.clusters import RunnerInventory

PRODUCERS = "producers.json"
CLUSTER_STATE = "cluster-state.json"
RETRY_NEXT_WAVE = frozenset({"no-capacity", "agent-not-started"})
MAINTAINER_CLOSE_HOLD_DAYS = 7
REQUEUE_LABEL = "klaud-requeue"
INFRA_OUTCOMES = frozenset({"missing-assets", "weights-missing"})
INFRA_COOLDOWN_DAYS = 7
LONG_HOLD_HOURS = 72
LONG_HOLD_OUTCOMES = frozenset({"already-updated", "duplicate", "route-mismatch"})
PAUSED_WEIGHTS = frozenset(ASSET_REASONS)
UNDATED_IMAGE_HOLD_DAYS = 7
DATED_TAG = re.compile(r"(?<!\d)20\d\d(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])(?!\d)")
INFRA_BACKOFF_HOURS = 48
INFRA_BACKOFF_OUTCOMES = 2
PRODUCER_MARGIN = 5
EVAL_FIELDS = frozenset(
    {"run-eval", "eval-only", "eval-conc", "eval-all-concs", "eval-framework", "eval-suite"}
)


class Family(NamedTuple):
    """A live family's target clusters, node demands and whether its smoke matrix is complete.

    ``node_demand`` is the node count of its largest AgentX job, ``sweep_node_demand`` the
    nodes of every job of its full AgentX sweep.
    """

    clusters: frozenset[str]
    node_demand: int
    sweep_node_demand: int
    smoke_covers_full: bool


class History(NamedTuple):
    """Recent candidate outcomes that still hold candidates, weights, images or clusters."""

    recent: dict[str, str | None]
    paused_weights: set[tuple]
    held_images: dict[str, datetime]
    infra_clusters: set[str]


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


def point_identities(rows: list[dict]) -> set[tuple]:
    """Return the (settings, concurrency) identity of each benchmark point in matrix ``rows``."""
    from infx.matrix.generate import freeze_config_value

    return {
        (
            freeze_config_value(
                {
                    key: value
                    for key, value in row.items()
                    if key != "conc" and key not in EVAL_FIELDS
                }
            ),
            conc,
        )
        for row in rows
        if not row.get("eval-only")
        for conc in (row["conc"] if isinstance(row["conc"], list) else [row["conc"]])
    }


def live_families(root: Path) -> dict[tuple, dict[str, Family]]:
    """Resolve public AgentX identities, target clusters and node demand from the generator.

    Archives and fixed-seq-len points are excluded. A family is one AgentX dashboard curve: every
    master key of one master file whose points share model, hardware, framework and precision,
    since the newest run replaces that whole curve. Curves spanning master files or using several
    images are left out. A family's smoke matrix covers its full sweep when the generator's
    ``--smoke`` selection keeps every benchmark point and default eval.
    """
    from infx.clusters import load_inventory
    from infx.config import project_root
    from infx.matrix.generate import (
        _hardware_family,
        generate_test_config_sweep,
        mark_eval_entries,
        select_matrix_evals,
    )
    from infx.matrix.validation import load_config_files, load_runner_file
    from infx.results.identity import public_framework

    from .validation import expected_evals, in_scope

    root = project_root(root)
    runners = load_runner_file(str(root / "configs/runners.yaml"))
    inventory = load_inventory(runners)
    curves: dict[tuple, dict[tuple[str, str], list[dict]]] = {}
    paths = sorted((root / "configs").glob("*-master.yaml"))
    if not paths:
        raise ReadError("live-configs-unavailable")
    for path in paths:
        source = path.relative_to(root).as_posix()
        configs = load_config_files([str(path)])
        for key in configs:
            try:
                entries = generate_test_config_sweep(
                    SimpleNamespace(config_keys=[key]), configs, runners, root
                )
            except ValueError:
                print(f"::warning::Klaud live-family-unrenderable: {source}:{key}")
                continue
            for entry in entries:
                if in_scope(entry):
                    curve = (
                        entry["model-prefix"],
                        _hardware_family(entry["runner"]),
                        public_framework(entry["framework"]),
                        entry["precision"],
                    )
                    curves.setdefault(curve, {}).setdefault((source, key), []).append(entry)
    index: dict[tuple, dict[str, Family]] = {}
    for members in curves.values():
        sources = {source for source, _ in members}
        if len(sources) > 1:
            print(
                f"::notice::Klaud skips {', '.join(sorted(f'{s}:{k}' for s, k in members))}: "
                "its AgentX curve spans several master configs."
            )
            continue
        family = sources.pop() + ":" + ",".join(sorted(key for _, key in members))
        entries = [entry for rows in members.values() for entry in rows]
        clusters = target_clusters(entries, inventory)
        if not clusters:
            print(f"::warning::Klaud live-family-clusters-unresolved: {family}")
            continue
        if len({normalized_image(entry["image"]) for entry in entries}) > 1:
            print(f"::notice::Klaud skips {family}: its keys use different images.")
            continue
        evals = [
            dict(row, **{"eval-only": True})
            for row in mark_eval_entries(deepcopy(entries), root)
            if row.get("run-eval")
        ]
        smoke = select_matrix_evals(deepcopy(entries), mode="smoke", root=root)
        smoke_evals = [row for row in smoke if row.get("eval-only")]
        record = Family(
            clusters=clusters,
            node_demand=max(entry.get("node-count", 1) for entry in entries),
            sweep_node_demand=sum(entry.get("node-count", 1) for entry in [*entries, *evals]),
            smoke_covers_full=point_identities(smoke) == point_identities(entries)
            and expected_evals({"evals": smoke_evals}) == expected_evals({"evals": evals}),
        )
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
            index.setdefault(observation_key(row), {})[family] = record
    return index


def choose(
    items: list[dict],
    headroom: dict[str, int],
    occupied: set[str],
    families: dict[tuple, dict[str, Family]],
) -> list[dict]:
    """Order one candidate per unclaimed family that fits the headroom of every target cluster.

    A branch or claim of the family, or of a single-key family of one of its keys, occupies it.
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
        for family, record in sorted(families.get(observation_key(row), {}).items()):
            branch = "klaud/auto-" + identity(family)[:16] + "-" + image_key
            claims = {f"klaud/auto-{legacy}-{image_key}", f"klaud/auto-{legacy}-"}
            for member in {family, *key_families(family)}:
                name = identity(member)[:16]
                claims |= {f"klaud/auto-{name}-", f"klaud/auto-{name}-{image_key}"}
                claims.add(f"klaud/claim-{name}")
            if (
                family in seen
                or claims & occupied
                or any(headroom.get(cluster, 0) < record.node_demand for cluster in record.clusters)
            ):
                continue
            selected.append(
                {
                    "id": branch.removeprefix("klaud/auto-"),
                    "family": family,
                    "clusters": sorted(record.clusters),
                    "node-demand": record.node_demand,
                    "sweep-node-demand": record.sweep_node_demand,
                    "smoke-covers-full": record.smoke_covers_full,
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


def hold(label: str | None, phase: str | None, cooldown_hours: int) -> timedelta:
    """Return how long outcome ``label``, recorded in ``phase``, keeps its candidate out of plans.

    Infrastructure outcomes (INFRA_OUTCOMES) hold for INFRA_COOLDOWN_DAYS. Outcomes a retry soon
    would repeat (LONG_HOLD_OUTCOMES, incompatible at resolve) hold for LONG_HOLD_HOURS, others
    for ``cooldown_hours``.
    """
    if label in INFRA_OUTCOMES:
        return timedelta(days=INFRA_COOLDOWN_DAYS)
    if label in LONG_HOLD_OUTCOMES or (label == "incompatible" and phase == "resolve"):
        return timedelta(hours=max(cooldown_hours, LONG_HOLD_HOURS))
    return timedelta(hours=cooldown_hours)


def recent_candidate_outcomes(repository: str, cooldown_hours: int) -> History:
    """Return the recent outcomes that still hold candidates, weights, images or clusters.

    Each candidate's newest session artifact counts and holds the candidate for ``hold``. While
    held, missing weights and route mismatches (PAUSED_WEIGHTS) pause their (model, precision,
    hardware) key. An already-updated outcome holds its image from its newest time, and
    INFRA_BACKOFF_OUTCOMES infrastructure failures within INFRA_BACKOFF_HOURS back off a target
    cluster. Only scheduled auto-sweeps and Klaud Cold's resume runs hold sessions. Resumed
    sessions record outcomes in later waves, so history reaches MAX_SESSIONS waves back.
    Unreadable history means no cooldown.
    """
    from .lifecycle import BOT, MAX_SESSIONS, WAKE_TITLE, WAVE_HOURS, resume_parent

    now = datetime.now(UTC)
    window = min(now - timedelta(hours=cooldown_hours), now - timedelta(days=INFRA_COOLDOWN_DAYS))
    since = window - timedelta(hours=MAX_SESSIONS * WAVE_HOURS)
    try:
        runs = [
            run
            for query in ("event=schedule", f"event=workflow_dispatch&actor={BOT}")
            for run in github_items(
                repository,
                f"actions/workflows/klaud-plan.yml/runs?{query}&created=>={since.date()}",
                "workflow_runs",
            )
        ]
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        print("::warning::Klaud cooldown history unavailable; continuing without cooldown")
        return History(recent={}, paused_weights=set(), held_images={}, infra_clusters=set())
    dated = set()
    plans: dict[int, int] = {}
    for run in runs:
        try:
            created = utc(run["created_at"])
            if created >= since and run.get("display_title") != WAKE_TITLE:
                dated.add((created, int(run["id"])))
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
                    or utc(artifact["created_at"]) < window
                ):
                    continue
                if match[1] not in latest or artifact["id"] > latest[match[1]][0]["id"]:
                    latest[match[1]] = (artifact, run_id)
            except (AttributeError, KeyError, TypeError, ValueError):
                continue
    result: dict[str, str | None] = {}
    held_images: dict[str, datetime] = {}
    planned_labels: dict[int, dict[str, str]] = {}
    for candidate, (artifact, run_id) in latest.items():
        label, phase = recorded_outcome(repository, artifact)
        created = utc(artifact["created_at"])
        if label == "already-updated":
            image = candidate.split("-", 1)[1]
            held_images[image] = max(held_images.get(image, created), created)
        if label in INFRA_REASONS and now - created < timedelta(hours=INFRA_BACKOFF_HOURS):
            planned_labels.setdefault(plans[run_id], {})[candidate] = label
        if now - created >= hold(label, phase, cooldown_hours):
            continue
        result[candidate] = label
        if label in PAUSED_WEIGHTS:
            planned_labels.setdefault(plans[run_id], {})[candidate] = label
    paused_weights = set()
    infra: Counter[str] = Counter()
    for plan_run, labels in planned_labels.items():
        if plan_run not in listings:
            try:
                listings[plan_run] = github_items(
                    repository, f"actions/runs/{plan_run}/artifacts?per_page=100", "artifacts"
                )
            except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
                continue
        for planned in planned_candidates(repository, listings[plan_run]):
            label = labels.get(planned.get("id"))
            if label in PAUSED_WEIGHTS:
                paused_weights.add(weights_key(planned["source"]))
            elif label in INFRA_REASONS:
                infra.update(planned.get("clusters", []))
    return History(
        recent=result,
        paused_weights=paused_weights,
        held_images=held_images,
        infra_clusters={cluster for cluster, n in infra.items() if n >= INFRA_BACKOFF_OUTCOMES},
    )


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


def recorded_outcome(repository: str, artifact: dict) -> tuple[str | None, str | None]:
    """Return the outcome label and phase a diagnostics ``artifact`` recorded, or Nones.

    A skipped agent's label is ``agent-not-started``, and an asset or infrastructure reason code
    labels its outcome. An unreadable artifact has no label.
    """
    with tempfile.TemporaryDirectory() as scratch:
        try:
            download_json(repository, artifact, Path(scratch))
            diagnostics = json.loads((Path(scratch) / "candidate-diagnostics.json").read_text())
            if diagnostics.get("action-outcome") == "skipped":
                return "agent-not-started", None
            outcome = CandidateOutcome.model_validate(diagnostics["candidate-outcome"])
            reason = outcome.reason_code
            labelled = reason in ASSET_REASONS or reason in INFRA_REASONS
            return (reason if labelled else outcome.outcome), outcome.phase
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            subprocess.SubprocessError,
        ):
            return None, None


def order_by_cooldown(candidates: list[dict], recent: dict[str, str | None]) -> list[dict]:
    """Return ``candidates`` never tried recently, then those whose recent outcome retries now."""
    fresh = [candidate for candidate in candidates if candidate["id"] not in recent]
    deferred = [
        candidate for candidate in candidates if recent.get(candidate["id"], "") in RETRY_NEXT_WAVE
    ]
    return fresh + deferred


def image_held(candidate: dict, held_images: dict[str, datetime], now: datetime) -> bool:
    """Whether an already-updated outcome on ``candidate``'s image still holds it at ``now``.

    Dated nightly tags are republished often, so they hold for LONG_HOLD_HOURS and other tags
    for UNDATED_IMAGE_HOLD_DAYS.
    """
    since = held_images.get(candidate["id"].split("-", 1)[1])
    if since is None:
        return False
    name = candidate["source"]["image"].split("@", 1)[0].rpartition("/")[2]
    dated = ":" in name and DATED_TAG.search(name.rpartition(":")[2])
    return now - since < (
        timedelta(hours=LONG_HOLD_HOURS) if dated else timedelta(days=UNDATED_IMAGE_HOLD_DAYS)
    )


def maintainer_closed_families(repository: str, now: datetime) -> set[str]:
    """Family prefixes of Klaud PRs someone else closed unmerged in the last hold window.

    Closing a Klaud PR stops that candidate, and the planner leaves its family alone for
    MAINTAINER_CLOSE_HOLD_DAYS, unless the PR carries REQUEUE_LABEL, which returns the family to
    the pool. Closed PRs are read newest-updated first until the window ends.
    """
    from .lifecycle import BOT

    cutoff = now - timedelta(days=MAINTAINER_CLOSE_HOLD_DAYS)
    held = set()
    page = 1
    while True:
        pulls = github_read(
            repository,
            f"pulls?state=closed&sort=updated&direction=desc&per_page=100&page={page}",
        )
        for pull in pulls:
            match = re.fullmatch(r"klaud/auto-([0-9a-f]{16})-[0-9a-f]{16}", pull["head"]["ref"])
            if (
                match
                and pull["user"]["login"] == BOT
                and not pull["merged_at"]
                and utc(pull["closed_at"]) >= cutoff
                and REQUEUE_LABEL not in {label["name"] for label in pull.get("labels", [])}
                and match[1] not in held
            ):
                closer = github_read(repository, f"issues/{pull['number']}").get("closed_by") or {}
                if closer.get("login") != BOT:
                    held.add(match[1])
        if len(pulls) < 100 or utc(pulls[-1]["updated_at"]) < cutoff:
            return held
        page += 1


def committed_demand(
    claimed: set[str], families: dict[tuple, dict[str, Family]], size: dict[str, int]
) -> dict[str, int]:
    """Return the nodes of each cluster that the full sweeps of ``claimed`` families will need.

    One sweep never holds more than its whole cluster, so each family's demand is capped at the
    cluster's ``size``.
    """
    records = {
        family: record for options in families.values() for family, record in options.items()
    }
    committed: Counter[str] = Counter()
    for family in claimed & records.keys():
        for cluster in records[family].clusters:
            committed[cluster] += min(records[family].sweep_node_demand, size.get(cluster, 0))
    return dict(committed)


def idle_first(
    candidates: list[dict], headroom: dict[str, int], size: dict[str, int]
) -> list[dict]:
    """Stably order ``candidates`` by the free fraction of their tightest target cluster."""
    return sorted(
        candidates,
        key=lambda candidate: (
            -min(headroom[cluster] / size[cluster] for cluster in candidate["clusters"])
        ),
    )


class WaveCapacity:
    """Headroom one wave uses up as it takes candidates, one per infra-backoff cluster.

    ``select`` and ``regenerate-producers`` walk candidates the same way, so producers are
    prepared for the candidates a wave can actually take, not those capacity will skip.
    """

    def __init__(self, headroom: dict[str, int], size: dict[str, int], backoff: set[str]) -> None:
        """Start from ``headroom`` of clusters of ``size`` nodes, with ``backoff`` clusters."""
        self.headroom, self.size, self.backoff = dict(headroom), size, backoff
        self.backed_off: set[str] = set()

    def blocker(self, candidate: dict) -> str | None:
        """Return why the wave cannot take ``candidate`` now ("backoff" or "capacity"), or None."""
        clusters = candidate["clusters"]
        if self.backed_off.intersection(clusters):
            return "backoff"
        if any(self.headroom.get(cluster, 0) < candidate["node-demand"] for cluster in clusters):
            return "capacity"
        return None

    def take(self, candidate: dict) -> None:
        """Reserve ``candidate``'s full sweep on each target cluster for the rest of the wave."""
        for cluster in candidate["clusters"]:
            self.headroom[cluster] -= min(candidate["sweep-node-demand"], self.size[cluster])
        self.backed_off.update(self.backoff.intersection(candidate["clusters"]))


def klaud_pulls(pulls: list[dict]) -> list[dict]:
    """Return ``pulls`` on Klaud candidate branches. Other PRs never own a family."""
    return [pr for pr in pulls if re.match(r"^klaud[e]?/auto-", pr["head"]["ref"])]


def plan(root: Path, directory: Path, review_batch_size: int, cooldown_hours: int) -> None:
    """Write ``candidates.json``, ``open-prs.json`` and CLUSTER_STATE under ``directory``.

    Unclaimed feed candidates at ``root``'s HEAD that fit the headroom left by claimed families
    are ordered toward idle clusters, cooled down by ``cooldown_hours`` and the recent-outcome
    holds, held while a maintainer recently closed their family's PR, and capped at
    ``review_batch_size``. Only open Klaud PRs are indexed. CLUSTER_STATE gives ``select`` the
    claimed families' committed demand and the clusters under infrastructure backoff, and gives
    ``regenerate-producers`` this plan's headroom and cluster sizes.
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
    owners = claims.family_owners(repository)
    blocked = {candidate.family for owner in owners for candidate in owner.candidates}
    families = live_families(root)
    capacity = fetch_capacity(policy) or Capacity(idle={}, queued={}, size={})
    committed = committed_demand(blocked, families, capacity.size)
    headroom = capacity.headroom(committed)
    candidates = idle_first(choose(items, headroom, occupied, families), headroom, capacity.size)
    recovery_file = Path(os.environ.get("RUNNER_TEMP", directory.parent)) / "klaud-recovery.json"
    if recovery_file.exists():
        blocked.update(json.loads(recovery_file.read_text()))
    blocked_keys = {member for family in blocked for member in key_families(family)}
    candidates = [
        candidate
        for candidate in candidates
        if not key_families(candidate["family"]) & blocked_keys
    ]
    now = datetime.now(UTC)
    held = maintainer_closed_families(repository, now)
    candidates = [
        candidate
        for candidate in candidates
        if not {
            identity(member)[:16]
            for member in {candidate["family"], *key_families(candidate["family"])}
        }
        & held
    ]
    history = recent_candidate_outcomes(repository, cooldown_hours)
    candidates = [
        candidate
        for candidate in candidates
        if weights_key(candidate["source"]) not in history.paused_weights
        and not image_held(candidate, history.held_images, now)
    ]
    candidates = order_by_cooldown(candidates, history.recent)
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
        for pr in klaud_pulls([pr for page in pulls for pr in page])
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
    (directory / CLUSTER_STATE).write_text(
        json.dumps(
            {
                "committed-demand": committed,
                "infra-backoff": sorted(history.infra_clusters),
                "headroom": headroom,
                "size": capacity.size,
            },
            indent=2,
        )
        + "\n"
    )
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

    A PR a maintainer closed never blocks, since the recover-current step cleans it up. A session
    that parked may stop while a run it parked on is still live, or if it parked at its deadline.
    """
    from .lifecycle import current_session, parked, terminal

    try:
        session = current_session()
        pulls = session.pulls()
        if pulls and session.closed_by_maintainer(pulls[0]):
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
    rerun = None
    try:
        from .lifecycle import current_session, park_record, resumable

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
            record = park_record(session, pulls[0], int(os.environ.get("KLAUD_SESSION") or "1"))
            if record is not None:
                parked_runs, rerun = record["runs"], record.get("rerun")
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
                    f"Parked to rerun the failed jobs of run {rerun['run']} after"
                    f" {rerun['not-before']}; recovery then resumes the candidate. / "
                    f"已等待：恢复流程将在 {rerun['not-before']} 后重跑运行 {rerun['run']}"
                    "的失败作业，然后继续该候选。\n"
                    if rerun
                    else f"Parked on {len(parked_runs)} owned runs; recovery resumes the"
                    f" candidate. / 已等待 {len(parked_runs)} 个自有运行；恢复流程将继续该候选。\n"
                )
        return True
    if failure is not None:
        observed = diagnostics.get("observed-session", {})
        outcome = CandidateOutcome(
            outcome="no-outcome",
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
    return outcome.outcome != "no-outcome"


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


def regenerate_producers(directory: Path, max_candidates: int, active_sessions: int = 0) -> None:
    """Write ``producers.json``: proceed candidates' producer families, by family and head.

    ``select`` takes candidates in order up to ``max_candidates`` less ``active_sessions``,
    skipping those the wave's capacity cannot take. This walks them the same way, with the plan's
    headroom from CLUSTER_STATE, and regenerates that many plus PRODUCER_MARGIN replacements, none
    without a slot. Producer revisions regenerate their own families, so this step runs without
    credentials. A producer that fails is logged and left out; ``select`` defers the candidates
    needing it.
    """
    from .reporting import publication
    from .validation import ProducerRegenerationError, producer_matrix

    contexts = json.loads((directory / "candidates.json").read_text())
    review, _ = reviewed(contexts)
    excluded = {decision.family for decision in review.decisions if decision.decision != "proceed"}
    models = {
        decision.candidate_id: decision.baseline_model
        for decision in review.decisions
        if decision.decision == "proceed" and decision.family not in excluded
    }
    slots = max(0, max_candidates - active_sessions)
    state = json.loads((directory / CLUSTER_STATE).read_text())
    wave = WaveCapacity(state["headroom"], state["size"], set(state["infra-backoff"]))
    proceeding = []
    taken: set[str] = set()
    for candidate in contexts:
        if len(proceeding) >= (slots + PRODUCER_MARGIN if slots else 0):
            break
        if (
            candidate["id"] in models
            and candidate["family"] not in taken
            and not wave.blocker(candidate)
        ):
            wave.take(candidate)
            taken.add(candidate["family"])
            proceeding.append(candidate)
    repository = os.environ["GITHUB_REPOSITORY"]
    families: dict[str, dict[str, dict]] = {}
    for candidate in proceeding:
        model = models[candidate["id"]]
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
    if slots and github.budget_low():
        slots, deferred = 0, deferred or "api-budget-low"
    state = json.loads((directory / CLUSTER_STATE).read_text())
    backoff = set(state["infra-backoff"])
    capacity = None
    if slots and any(decision.decision == "proceed" for decision in review.decisions):
        try:
            capacity = fetch_capacity(Policy())
        except ReadError:
            deferred = "capacity-unavailable"
        else:
            deferred = None if capacity else "capacity-snapshot-unusable"
    wave = WaveCapacity(
        capacity.headroom(state["committed-demand"]) if capacity else {},
        capacity.size if capacity else {},
        backoff,
    )
    capacity_deferred = []
    backoff_deferred = []
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
        blocker = wave.blocker(candidate)
        if blocker == "backoff":
            backoff_deferred.append(candidate["id"])
            continue
        if blocker:
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
        wave.take(candidate)
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
                "infra-backoff-clusters": sorted(backoff),
                "backoff-deferred-candidates": backoff_deferred,
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
            " (a target cluster had queued human jobs, Klaud jobs queued over"
            f" {KLAUD_QUEUE_MAX_AGE.total_seconds() / 3600:g}h, or too few idle nodes"
            " beyond the full sweeps of claimed families)."
        )
    if backoff:
        summary += (
            f" Infrastructure backoff on {', '.join(sorted(backoff))}: at least"
            f" {INFRA_BACKOFF_OUTCOMES} infrastructure failures in {INFRA_BACKOFF_HOURS}h, so at"
            f" most one new candidate per cluster this wave ({len(backoff_deferred)} deferred)."
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
    regeneration = commands.add_parser(
        "regenerate-producers",
        help="Regenerate reviewed candidates' baseline producer families; run without credentials",
    )
    selection = commands.add_parser(
        "select", help="Validate KLAUD_PR_REVIEW and select nonoverlapping candidates"
    )
    for slotted in (regeneration, selection):
        slotted.add_argument("--directory", type=Path, required=True)
        slotted.add_argument(
            "--max-candidates-per-run",
            type=int,
            required=True,
            help="Maximum candidates to select (1-256)",
        )
        slotted.add_argument(
            "--active-sessions",
            type=int,
            default=0,
            help="Klaud sessions running or resumed by recovery; each takes one candidate slot",
        )
    selection.add_argument(
        "--execution-file",
        type=Path,
        help="Claude execution log; retain numeric metrics and fixed denial categories",
    )
    commands.add_parser(
        "check-stop",
        help="Claude Stop hook: require finished runs and validated or closed PRs, or a park",
    )
    commands.add_parser(
        "check-deadline",
        help="Claude PostToolUse hook: tell a session near its time limit to park",
    )
    parking = commands.add_parser(
        "park",
        help="Record that this session waits on its reported live owned runs, then stop",
    )
    parking.add_argument(
        "--rerun-run",
        type=int,
        help="Reported, ended owned run whose failed jobs recovery reruns once, later",
    )
    parking.add_argument(
        "--rerun-after-minutes",
        type=int,
        help="Minutes (15-240) recovery waits before that rerun, then the next session starts",
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
            from .lifecycle import PendingCleanup, current_session, reconcile, show_status

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
            finally:
                show_status(session)
                github.api_budget()
            return 0
        if args.command == "resume-check":
            github.api_budget()
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

            if (args.rerun_run is None) != (args.rerun_after_minutes is None):
                parser.error("--rerun-run and --rerun-after-minutes go together")
            if args.rerun_after_minutes is not None and not 15 <= args.rerun_after_minutes <= 240:
                parser.error("--rerun-after-minutes must be between 15 and 240")
            now = datetime.now(UTC)
            started = int(os.environ["KLAUD_SESSION_STARTED"])
            park(
                current_session(),
                int(os.environ["KLAUD_SESSION"]),
                overdue=now.timestamp() - started >= PARK_AFTER_SECONDS,
                rerun=None
                if args.rerun_run is None
                else (args.rerun_run, now + timedelta(minutes=args.rerun_after_minutes)),
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
            plan(args.root, args.directory, args.review_batch_size, args.cooldown_hours)
            return 0
        if not 1 <= args.max_candidates_per_run <= 256:
            parser.error("--max-candidates-per-run must be between 1 and 256")
        if args.active_sessions < 0:
            parser.error("--active-sessions must not be negative")
        if args.command == "regenerate-producers":
            regenerate_producers(args.directory, args.max_candidates_per_run, args.active_sessions)
            return 0
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
        elif isinstance(error, subprocess.CalledProcessError) and "rate limit" in str(
            error.stderr or ""
        ):
            reason = "GitHub API rate limit exhausted; retry after it resets"
        else:
            reason = "State unavailable or invalid; inspect GitHub before retrying"
        print(f"::error::Klaud: {reason}.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
