import json
import sys
import threading
from collections import Counter
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs

import pytest
import yaml

from infx.klaud import __main__ as klaud
from infx.klaud import api, claims, github, lifecycle, reporting, validation
from infx.klaud.models import (
    CandidateOutcome,
    Feed,
    OwnedCandidate,
    Ownership,
    identity,
    stamp,
    utc,
)
from infx.tests.klaud.test_klaud_github import (
    ended_session_sweep,
    family_configs,
    park_comment,
    write_fixture_recipe,
)

NOW = "2026-10-26T12:00:00Z"
RUNS = "actions/workflows/klaud-plan.yml/runs?"


def ago(hours):
    return stamp(utc(NOW) - timedelta(hours=hours))


def frozen_clock(monkeypatch):
    class Clock(klaud.datetime):
        @classmethod
        def now(cls, tz=None):
            return utc(NOW)

    monkeypatch.setattr(klaud, "datetime", Clock)


def workflow_run(run_id, event, title="Klaud Cold auto-sweep", actor="maintainer", hours=1):
    return {"id": run_id, "event": event, "actor": {"login": actor}, "display_title": title,
            "created_at": ago(hours)}  # fmt: skip


def session(artifact_id, candidate, hours):
    return {"id": artifact_id, "name": f"klaud-candidate-{candidate}", "expired": False,
            "created_at": ago(hours)}  # fmt: skip


def plan_artifact(artifact_id):
    return {"id": artifact_id, "name": "klaud-plan", "expired": False, "created_at": ago(1)}


def outcome(name, phase="targeted", reason=None):
    return {"outcome": name, "phase": phase, "pull-request": None, "run-ids": [],
            "repairs-used": 0, "reason-code": reason}  # fmt: skip


def serve_history(monkeypatch, runs, artifacts, records):
    """Serve workflow runs filtered like GitHub's run listing, and their artifacts' records."""
    listed = []

    def items(_repository, path, _key):
        if path.startswith(RUNS):
            query = parse_qs(path.removeprefix(RUNS))
            return [
                run for run in runs
                if run["event"] == query["event"][0]
                and run["actor"]["login"] == query.get("actor", [run["actor"]["login"]])[0]
                and run["created_at"] >= query["created"][0].removeprefix(">=")
            ]  # fmt: skip
        run_id = int(path.split("/")[2])
        listed.append(run_id)
        return artifacts.get(run_id, [])

    def download(_repository, artifact, destination):
        record = records[artifact["id"]]
        if artifact["name"] == "klaud-plan":
            plan = record if isinstance(record, dict) else {"candidates": record}
            (destination / "candidates.json").write_text(json.dumps(plan["candidates"]))
            (destination / "selection.json").write_text(json.dumps(plan.get("selection", {})))
        else:
            (destination / "candidate-diagnostics.json").write_text(
                json.dumps({"candidate-outcome": record}))

    frozen_clock(monkeypatch)
    monkeypatch.setattr(klaud, "github_items", items)
    monkeypatch.setattr(klaud, "download_json", download)
    return listed


def candidate_id(family, image="0" * 16):
    return f"{identity(family)[:16]}-{image}"


def test_history_lists_only_auto_sweeps_and_klaud_resume_runs(monkeypatch):
    swept, resumed, woken = candidate_id("swept"), candidate_id("resumed"), candidate_id("woken")
    runs = [
        workflow_run(1, "schedule", hours=5),
        workflow_run(2, "schedule", title="Klaud Cold wake", hours=4),
        workflow_run(3, "workflow_dispatch", title="Klaud Cold wake", actor="github-actions[bot]"),
        workflow_run(4, "workflow_dispatch", title=f"Klaud Cold resume 1 {resumed} session 2",
                     actor="Klaud-Cold", hours=2),
        workflow_run(5, "workflow_dispatch", actor="maintainer", hours=3),
    ]  # fmt: skip
    artifacts = {1: [session(10, swept, 5)], 2: [session(20, woken, 4)],
                 3: [session(30, woken, 1)], 4: [session(40, resumed, 2)],
                 5: [session(50, woken, 3)]}  # fmt: skip
    records = {10: outcome("failed"), 20: outcome("failed"), 30: outcome("failed"),
               40: outcome("already-updated", "resolve"), 50: outcome("failed")}  # fmt: skip
    listed = serve_history(monkeypatch, runs, artifacts, records)

    history = klaud.recent_candidate_outcomes("example/project", 24)

    assert sorted(listed) == [1, 4]
    assert history.recent == {swept: "failed", resumed: "already-updated"}
    assert history.held_images == {"0" * 16: utc(ago(2))}


@pytest.mark.parametrize(("record", "hours", "held", "paused"), [
    pytest.param(outcome("already-updated", "resolve"), 71 + 59 / 60, True, False, id="no-op-held"),
    pytest.param(outcome("already-updated", "resolve"), 72, False, False, id="no-op-released"),
    pytest.param(outcome("duplicate", "resolve"), 71 + 59 / 60, True, False, id="duplicate-held"),
    pytest.param(outcome("duplicate", "resolve"), 72, False, False, id="duplicate-released"),
    pytest.param(outcome("incompatible", "resolve"), 71 + 59 / 60, True, False,
                 id="resolve-incompatible-held"),
    pytest.param(outcome("incompatible", "resolve"), 72, False, False,
                 id="resolve-incompatible-released"),
    pytest.param(outcome("incompatible", "targeted"), 24, False, False,
                 id="targeted-incompatible-cools-down-like-a-failure"),
    pytest.param(outcome("missing-assets", reason="route-mismatch"), 71 + 59 / 60, True, True,
                 id="route-mismatch-paused"),
    pytest.param(outcome("missing-assets", reason="route-mismatch"), 72, False, False,
                 id="route-mismatch-released"),
    pytest.param(outcome("missing-assets", reason="weights-missing"), 72, True, True,
                 id="weights-missing-outlasts-a-route-mismatch"),
    pytest.param(outcome("failed", "final-sweep", "infra-node-fault"), 71 + 59 / 60, True, False,
                 id="node-fault-held"),
    pytest.param(outcome("failed", "final-sweep", "infra-node-fault"), 72, False, False,
                 id="node-fault-released"),
    pytest.param(outcome("inconclusive", "targeted", "infra-hf-rate-limit"), 71 + 59 / 60, True,
                 False, id="hf-rate-limit-held"),
])  # fmt: skip
def test_outcomes_hold_their_candidate_and_weights_until_their_hold_ends(
    monkeypatch, record, hours, held, paused
):
    candidate = candidate_id("family")
    source = {"model": "dsr1", "precision": "fp8", "hardware": "b200"}
    serve_history(
        monkeypatch,
        [workflow_run(1, "schedule", hours=hours)],
        {1: [session(10, candidate, hours), plan_artifact(11)]},
        {10: record, 11: [{"id": candidate, "source": source, "clusters": ["b200-nscale"]}]},
    )

    history = klaud.recent_candidate_outcomes("example/project", 24)

    assert (candidate in history.recent) is held
    assert history.paused_weights == ({("dsr1", "fp8", "b200")} if paused else set())


@pytest.mark.parametrize(("reason", "hours", "held"), [
    pytest.param("baseline-point-mismatch", 71 + 59 / 60, True, id="mismatch-held"),
    pytest.param("baseline-point-mismatch", 72, False, id="mismatch-released"),
    pytest.param("Public baseline dataset provenance is unavailable", 71 + 59 / 60, True,
                 id="dataset-provenance-held"),
    pytest.param("Baseline producer family cannot be regenerated", 1, False,
                 id="unprepared-producers-retry-next-wave"),
])  # fmt: skip
def test_a_wave_s_baseline_point_mismatch_holds_its_candidate(monkeypatch, reason, hours, held):
    candidate = candidate_id("stale-baseline")
    serve_history(
        monkeypatch,
        [workflow_run(1, "schedule", hours=hours)],
        {1: [plan_artifact(11)]},
        {11: {"candidates": [], "selection": {"baseline-deferral-reasons": {candidate: reason}}}},
    )

    history = klaud.recent_candidate_outcomes("example/project", 24)

    assert (candidate in history.recent) is held


@pytest.mark.parametrize(("hours", "record", "label"), [
    pytest.param(30, outcome("failed"), "baseline-point-mismatch",
                 id="an-earlier-session-keeps-the-deferral"),
    pytest.param(5, outcome("no-capacity"), "no-capacity", id="a-later-session-replaces-it"),
])  # fmt: skip
def test_a_baseline_deferral_holds_unless_a_session_started_after_it(
    monkeypatch, hours, record, label
):
    candidate = candidate_id("stale-baseline")
    reasons = {candidate: "baseline-point-mismatch"}
    serve_history(
        monkeypatch,
        [workflow_run(1, "schedule", hours=10), workflow_run(2, "schedule", hours=hours)],
        {1: [plan_artifact(11)], 2: [session(20, candidate, hours)]},
        {11: {"candidates": [], "selection": {"baseline-deferral-reasons": reasons}}, 20: record},
    )

    history = klaud.recent_candidate_outcomes("example/project", 24)

    assert history.recent == {candidate: label}


@pytest.mark.parametrize(("hours", "held"), [(167 + 59 / 60, True), (168, False)])
def test_a_cluster_time_limit_holds_the_family_on_every_image(monkeypatch, hours, held):
    tried = candidate_id("slow-family")
    serve_history(
        monkeypatch,
        [workflow_run(1, "schedule", hours=hours)],
        {1: [session(10, tried, hours)]},
        {10: outcome("failed", "final-sweep", reason="cluster-time-limit")},
    )

    history = klaud.recent_candidate_outcomes("example/project", 24)

    assert history.held_families == ({tried.split("-", 1)[0]} if held else set())


def test_a_cluster_time_limit_holds_every_key_of_its_curve(monkeypatch):
    curve = "configs/nvidia-master.yaml:agg,mtp"
    tried = candidate_id(curve)
    serve_history(
        monkeypatch,
        [workflow_run(1, "schedule", hours=10)],
        {1: [session(10, tried, 10), plan_artifact(11)]},
        {10: outcome("failed", "final-sweep", reason="cluster-time-limit"),
         11: [{"id": tried, "family": curve, "source": {}, "clusters": ["gb300"]}]},
    )  # fmt: skip

    history = klaud.recent_candidate_outcomes("example/project", 24)

    assert history.held_families == {
        identity(family)[:16] for family in
        (curve, "configs/nvidia-master.yaml:agg", "configs/nvidia-master.yaml:mtp")}  # fmt: skip


@pytest.mark.parametrize("failing", [RUNS, "actions/runs/"])
def test_unreadable_history_fails_the_plan_instead_of_dropping_every_hold(monkeypatch, failing):
    serve_history(
        monkeypatch,
        [workflow_run(1, "schedule")],
        {1: [session(10, candidate_id("held"), 1)]},
        {10: outcome("failed")},
    )
    served = klaud.github_items

    def items(repository, path, key):
        if path.startswith(failing):
            raise github.ListingUnavailable("Incomplete GitHub listing")
        return served(repository, path, key)

    monkeypatch.setattr(klaud, "github_items", items)

    with pytest.raises(api.ReadError, match="candidate-history-unavailable"):
        klaud.recent_candidate_outcomes("example/project", 24)


def test_history_lists_runs_in_parallel_and_downloads_each_plan_once(monkeypatch):
    deferred, weightless = candidate_id("deferred"), candidate_id("weightless")
    source = {"model": "dsr1", "precision": "fp4", "hardware": "b200"}
    serve_history(
        monkeypatch,
        [workflow_run(1, "schedule", hours=2), workflow_run(2, "schedule")],
        {1: [session(10, weightless, 2), plan_artifact(11)]},
        {10: outcome("missing-assets", reason="weights-missing"),
         11: {"candidates": [{"id": weightless, "source": source}],
              "selection": {"baseline-deferral-reasons": {deferred: "baseline-point-mismatch"}}}},
    )  # fmt: skip
    served_items, served_download = klaud.github_items, klaud.download_json
    # Serial listings never meet here, and the wait times out.
    listings = threading.Barrier(2, timeout=5)
    downloads = Counter()

    def items(repository, path, key):
        if not path.startswith(RUNS):
            listings.wait()
        return served_items(repository, path, key)

    def download(repository, artifact, destination):
        downloads[artifact["id"]] += 1
        served_download(repository, artifact, destination)

    monkeypatch.setattr(klaud, "github_items", items)
    monkeypatch.setattr(klaud, "download_json", download)

    history = klaud.recent_candidate_outcomes("example/project", 24)

    assert history.recent == {weightless: "weights-missing", deferred: "baseline-point-mismatch"}
    assert history.paused_weights == {("dsr1", "fp4", "b200")}
    assert downloads == {10: 1, 11: 1}


def test_an_already_updated_image_is_held_from_its_newest_outcome(monkeypatch):
    image = "1" * 16
    older, newer = candidate_id("older", image), candidate_id("newer", image)
    serve_history(
        monkeypatch,
        [workflow_run(1, "schedule", hours=150)],
        {1: [session(10, older, 150), session(11, newer, 100)]},
        {10: outcome("already-updated", "resolve"), 11: outcome("already-updated", "resolve")},
    )

    history = klaud.recent_candidate_outcomes("example/project", 24)

    assert history.recent == {}
    assert history.held_images == {image: utc(ago(100))}


@pytest.mark.parametrize(("image", "hours", "held"), [
    ("lmsysorg/sglang:nightly-20261025", 71 + 59 / 60, True),
    ("lmsysorg/sglang:nightly-20261025", 72, False),
    ("nvcr.io#nvidia/tritonserver:25.10-20261024-py3", 72, False),
    ("lmsysorg/sglang:v0.5.22-cu130", 72, True),
    ("lmsysorg/sglang:v0.5.22-cu130", 167 + 59 / 60, True),
    ("lmsysorg/sglang:v0.5.22-cu130", 168, False),
    # Neither a digest nor a registry port is a dated tag.
    ("lmsysorg/sglang:v0.5.22@sha256:" + "20261025" * 8, 72, True),
    ("registry:20261025/vllm/vllm-openai:minimax-m31-flash", 72, True),
    ("lmsysorg/sglang:dev-20261399", 72, True),
])  # fmt: skip
def test_dated_nightly_images_are_held_shorter_than_other_tags(image, hours, held):
    candidate = {"id": candidate_id("family", "1" * 16), "source": {"image": image}}

    assert klaud.image_held(candidate, {"1" * 16: utc(ago(hours))}, utc(NOW)) is held
    assert klaud.image_held(candidate, {"2" * 16: utc(ago(hours))}, utc(NOW)) is False


def test_a_mixed_image_curve_is_held_only_while_every_published_image_is():
    held, fresh = (identity(image)[:16] for image in ("example/sglang:v1", "example/vllm:v2"))
    candidate = {"id": candidate_id("configs/nvidia-master.yaml:agg,mtp", held),
                 "source": {"image": "example/sglang:v1"},
                 "sources": {"agg": {"image": "example/sglang:v1"},
                             "mtp": {"image": "example/vllm:v2"}}}  # fmt: skip
    since = utc(ago(1))

    assert klaud.image_held(candidate, {held: since}, utc(NOW)) is False
    assert klaud.image_held(candidate, {held: since, fresh: since}, utc(NOW)) is True


def test_a_mixed_curve_with_an_unbaselined_key_stays_held_after_already_updated(monkeypatch):
    tried = candidate_id("configs/nvidia-master.yaml:agg,mtp", identity("example/sglang:v1")[:16])
    candidate = {"id": tried, "source": {"image": "example/sglang:v1"},
                 "sources": {"agg": {"image": "example/sglang:v1"}},
                 "unbaselined-keys": {"mtp": "example/vllm:v2"}}  # fmt: skip
    serve_history(
        monkeypatch,
        [workflow_run(1, "schedule", hours=30)],
        {1: [session(10, tried, 30)]},
        {10: outcome("already-updated", "resolve")},
    )

    history = klaud.recent_candidate_outcomes("example/project", 24)

    assert klaud.image_held(candidate, history.held_images, utc(NOW)) is True


@pytest.mark.parametrize(("hours", "backoff"), [
    pytest.param(47 + 59 / 60, {"gb300"}, id="second-failure-inside-the-window"),
    pytest.param(48, set(), id="second-failure-outside-the-window"),
])  # fmt: skip
def test_two_recent_infrastructure_failures_back_off_their_cluster(monkeypatch, hours, backoff):
    node, registry, dataset, failed = (candidate_id(name) for name in ("n", "r", "d", "f"))
    planned = [
        {"id": node, "source": {}, "clusters": ["gb300"]},
        {"id": registry, "source": {}, "clusters": ["b300", "gb300"]},
        {"id": dataset, "source": {}, "clusters": ["mi355x"]},
        {"id": failed, "source": {}, "clusters": ["mi355x"]},
    ]
    serve_history(
        monkeypatch,
        [workflow_run(1, "schedule", hours=50)],
        {1: [session(10, node, hours), session(11, registry, 1), session(12, dataset, 1),
             session(13, failed, 1), plan_artifact(14)]},
        {10: outcome("failed", "final-sweep", "infra-node-fault"),
         11: outcome("inconclusive", "targeted", "infra-registry-rate-limit"),
         12: outcome("failed", "targeted", "infra-hf-rate-limit"),
         13: outcome("failed", "targeted"),
         14: planned},
    )  # fmt: skip

    history = klaud.recent_candidate_outcomes("example/project", 24)

    assert history.infra_clusters == backoff
    assert history.recent[registry] == "infra-registry-rate-limit"


@pytest.mark.parametrize(("name", "phase", "reason", "valid"), [
    ("missing-assets", "targeted", "route-mismatch", True),
    ("failed", "targeted", "route-mismatch", False),
    ("failed", "final-sweep", "infra-node-fault", True),
    ("inconclusive", "targeted", "infra-hf-rate-limit", True),
    ("failed", "resolve", "infra-registry-rate-limit", True),
    ("failed", "baseline", "infra-registry-rate-limit", False),
    ("missing-assets", "targeted", "infra-node-fault", False),
    ("validated", "final-sweep", "infra-node-fault", False),
])  # fmt: skip
def test_infrastructure_and_route_reason_codes_pair_with_their_outcomes(name, phase, reason, valid):
    record = {**outcome(name, phase, reason), "pull-request": 7, "run-ids": [1]}
    if valid:
        assert CandidateOutcome.model_validate(record).reason_code == reason
    else:
        with pytest.raises(ValueError):
            CandidateOutcome.model_validate(record)


def scheduler(queue, metrics, now=NOW):
    payload = {"kind": "inferencex.status.scheduler", "generatedAt": now, "data": {
        "available": True, "snapshot": {"snapshotAt": now, "stalled": False,
                                        "packingMetrics": metrics, "queue": queue},
    }}  # fmt: skip
    feed = Feed(url=api.ENDPOINTS["scheduler"], retrieved_at=now, sha256="0" * 64, payload=payload)
    return api.cluster_capacity(feed, api.Policy(), utc(now))


def metric(name, schedulable, allocated=0):
    return {"clusterLabel": f"cluster:{name}", "schedulableNodes": schedulable,
            "allocatedNodes": allocated}  # fmt: skip


def job(name, cluster, nodes, hours, runner=None):
    return {"name": name, "clusterIds": [f"cluster:{cluster}"], "nodeCount": nodes,
            "runnerName": runner, "queuedAt": ago(hours)}  # fmt: skip


def test_capacity_skips_clusters_whose_klaud_queue_waited_over_six_hours():
    capacity = scheduler(
        [job("sweep / klaud | p3 | a", "waited", 2, 6), job("sweep / klaud | p3 | b", "stuck", 1, 1),
         job("sweep / klaud | p3 | c", "stuck", 1, 6 + 1 / 3600),
         job("sweep / klaud | p3 | d", "admitted", 1, 30, runner="admitted_00")],
        [metric("waited", 8, 1), metric("stuck", 8), metric("admitted", 8, 1)],
    )  # fmt: skip

    assert capacity == api.Capacity(
        idle={"waited": 7, "admitted": 7},
        queued={"waited": 2, "stuck": 2},
        size={"waited": 8, "stuck": 8, "admitted": 8},
    )
    # Claimed families' committed demand includes their queued jobs: only the larger counts.
    assert capacity.headroom({"waited": 1, "admitted": 5}) == {"waited": 5, "admitted": 2}
    assert capacity.headroom({"waited": 6, "admitted": 7}) == {"waited": 1}
    unqueued = {**job("sweep / klaud | p3 | e", "waited", 1, 1), "queuedAt": None}
    assert scheduler([unqueued], [metric("waited", 8)]) is None


def test_claimed_families_commit_their_full_sweep_capped_at_cluster_size():
    def family(clusters, sweep):
        return klaud.Family(frozenset(clusters), 1, sweep, False)

    families = {
        ("observation", 1): {"configs/nvidia-master.yaml:wide": family(["gb300", "b300"], 30),
                             "configs/nvidia-master.yaml:idle": family(["gb300"], 4)},
        ("observation", 2): {"configs/nvidia-master.yaml:small": family(["gb300"], 3)},
    }  # fmt: skip
    claimed = {"configs/nvidia-master.yaml:wide", "configs/nvidia-master.yaml:small",
               "configs/nvidia-master.yaml:retired"}  # fmt: skip

    assert klaud.committed_demand(claimed, families, {"gb300": 35, "b300": 8}) == {
        "gb300": 33, "b300": 8}


def wave_context(name, clusters, demand=1, sweep=None):
    return {"id": candidate_id(name), "family": f"configs/nvidia-master.yaml:{name}",
            "base": "a" * 40, "source": {}, "clusters": clusters, "node-demand": demand,
            "sweep-node-demand": demand if sweep is None else sweep}  # fmt: skip


def select_wave(tmp_path, monkeypatch, contexts, capacity, committed=None, backoff=(),
                backlog=None, active=0, running=None):  # fmt: skip
    """Select a wave whose review proceeds with every context; return selection and summary."""
    (tmp_path / "candidates.json").write_text(json.dumps(contexts))
    (tmp_path / "producers.json").write_text("{}\n")
    (tmp_path / klaud.CLUSTER_STATE).write_text(
        json.dumps({"committed-demand": committed or {}, "infra-backoff": list(backoff),
                    "finished-backlog": backlog or {}}))  # fmt: skip
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

    summary = tmp_path / "summary.md"
    for name, value in (("GITHUB_REPOSITORY", "example/project"), ("GITHUB_RUN_ID", "42"),
                        ("KLAUD_PR_REVIEW", json.dumps(review)),
                        ("GITHUB_STEP_SUMMARY", str(summary))):  # fmt: skip
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    monkeypatch.setattr(klaud, "fetch_capacity", capacity)
    monkeypatch.setattr(reporting, "resolve_baseline", resolve)
    monkeypatch.setattr(validation, "canonical_matrix", lambda *_: {"single_node": {"all": []}})
    monkeypatch.setattr(claims, "claim_family", lambda *_args: True)
    if running:
        monkeypatch.setattr(klaud, "running_sessions", running)

    klaud.select(tmp_path, 5, active_sessions=active)

    return json.loads((tmp_path / "selection.json").read_text()), summary.read_text()


def test_select_gates_on_committed_demand_and_commits_each_selected_sweep(tmp_path, monkeypatch):
    capacity = api.Capacity(idle={"a": 10, "b": 4}, queued={"a": 3}, size={"a": 10, "b": 4})
    contexts = [wave_context("too-wide", ["a"], 4), wave_context("fits", ["a"], 3, sweep=25),
                wave_context("after-commit", ["a"]), wave_context("other", ["b"], 4)]  # fmt: skip

    selection, _ = select_wave(
        tmp_path, monkeypatch, contexts, lambda _policy: capacity, committed={"a": 7})

    ids = {context["family"].rsplit(":", 1)[1]: context["id"] for context in contexts}
    assert selection["candidates"] == [ids["fits"], ids["other"]]
    assert selection["capacity-deferred-candidates"] == [ids["too-wide"], ids["after-commit"]]
    assert selection["deferred-reason"] is None


def test_select_takes_one_candidate_per_backed_off_cluster_and_says_so(tmp_path, monkeypatch):
    capacity = api.Capacity(idle={"gb300": 4, "b300": 4}, queued={},
                            size={"gb300": 4, "b300": 4})  # fmt: skip
    contexts = [wave_context("oversized", ["gb300"], 5), wave_context("first", ["gb300"]),
                wave_context("second", ["b300", "gb300"]), wave_context("elsewhere", ["b300"])]  # fmt: skip

    selection, summary = select_wave(
        tmp_path, monkeypatch, contexts, lambda _policy: capacity, backoff=["gb300"])

    ids = [context["id"] for context in contexts]
    assert selection["candidates"] == [ids[1], ids[3]]
    assert selection["capacity-deferred-candidates"] == [ids[0]]
    assert selection["backoff-deferred-candidates"] == [ids[2]]
    assert selection["infra-backoff-clusters"] == ["gb300"]
    assert ("Infrastructure backoff on gb300: at least 2 infrastructure failures in 48h, so at"
            " most one new candidate per cluster this wave (1 deferred).") in summary


def unavailable(_policy):
    raise api.ReadError("scheduler:http-503")


@pytest.mark.parametrize(("capacity", "reason"), [
    (lambda _policy: None, "capacity-snapshot-unusable"),
    (unavailable, "capacity-unavailable"),
])  # fmt: skip
def test_select_names_why_capacity_could_not_be_checked(tmp_path, monkeypatch, capacity, reason):
    contexts = [wave_context("only", ["a"])]

    selection, summary = select_wave(tmp_path, monkeypatch, contexts, capacity)

    assert selection["deferred-reason"] == reason
    assert selection["capacity-deferred-candidates"] == [contexts[0]["id"]]
    assert f"Selection stopped: {reason}." in summary


def unreadable_runs(_repository):
    raise github.ListingUnavailable("Incomplete GitHub listing")


@pytest.mark.parametrize(("active", "running", "selected", "reason"), [
    pytest.param(0, lambda _repository: 3, 2, None, id="sessions-started-after-recovery-count"),
    pytest.param(4, lambda _repository: 1, 1, None, id="recoverys-larger-count-stands"),
    pytest.param(0, unreadable_runs, 0, "active-sessions-unavailable", id="unreadable-stops"),
])  # fmt: skip
def test_select_counts_the_sessions_running_when_it_selects(
    tmp_path, monkeypatch, active, running, selected, reason
):
    capacity = api.Capacity(idle=dict.fromkeys("abcd", 4), queued={}, size=dict.fromkeys("abcd", 4))
    contexts = [wave_context(name, [name]) for name in "abcd"]

    selection, summary = select_wave(tmp_path, monkeypatch, contexts, lambda _policy: capacity,
                                     active=active, running=running)  # fmt: skip

    assert selection["candidates"] == [context["id"] for context in contexts[:selected]]
    assert selection["deferred-reason"] == reason
    if reason:
        assert f"Selection stopped: {reason}." in summary


def test_select_defers_candidates_of_a_master_config_with_a_finished_pr_backlog(
    tmp_path, monkeypatch
):
    capacity = api.Capacity(idle={"a": 4, "b": 4}, queued={}, size={"a": 4, "b": 4})
    busy = wave_context("busy", ["a"])
    quiet = {**wave_context("quiet", ["b"]), "family": "configs/amd-master.yaml:quiet"}

    selection, summary = select_wave(
        tmp_path, monkeypatch, [busy, quiet], lambda _policy: capacity,
        backlog={"nvidia-master.yaml": 8, "amd-master.yaml": 7})  # fmt: skip

    assert selection["candidates"] == [quiet["id"]]
    assert selection["backlog-deferred-candidates"] == [busy["id"]]
    assert selection["backlogged-configs"] == ["nvidia-master.yaml"]
    assert (
        "1 reviewed candidates wait because nvidia-master.yaml already have 8 or more"
        " finished Klaud PRs awaiting review."
    ) in summary


def test_a_wave_without_a_free_slot_skips_the_overlap_review_and_selection():
    workflow = yaml.safe_load(
        (Path(__file__).parents[4] / ".github/workflows/klaud-plan.yml").read_text()
    )
    cap = workflow["env"]["MAX_CANDIDATES_PER_RUN"]

    assert (f"fromJSON(needs.recover.outputs.active-sessions || '0') < {cap}"
            in workflow["jobs"]["review"]["if"])  # fmt: skip
    assert "review" in workflow["jobs"]["select"]["needs"]
    assert "if" not in workflow["jobs"]["select"]


@pytest.mark.parametrize(("concs", "covers", "sweep"), [
    pytest.param([2], True, 2, id="one-concurrency"),
    pytest.param([2, 6], False, 3, id="smoke-trims-concurrency"),
])  # fmt: skip
def test_live_families_say_whether_smoke_covers_the_full_sweep(
    tmp_path, family_configs, concs, covers, sweep
):
    master, runners = family_configs
    master["fixture"]["scenarios"]["agentic-coding"][0]["search-space"][0]["conc-list"] = concs
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "nvidia-master.yaml").write_text(yaml.safe_dump(master))
    (configs / "runners.yaml").write_text(yaml.safe_dump(runners))
    write_fixture_recipe(tmp_path)

    [records] = klaud.live_families(tmp_path).values()

    assert records["configs/nvidia-master.yaml:fixture"] == klaud.Family(
        frozenset({"fixture"}), 1, sweep, covers)


def feed_source(model, image):
    return {"model": model, "hardware": "h100", "framework": "sglang", "precision": "fp8",
            "spec_method": "none", "disagg": False, "benchmark_type": "agentic_traces",
            "isl": None, "osl": None, "image": image, "date": "2026-10-20"}  # fmt: skip


def test_plan_holds_no_op_images_and_claimed_demand_and_orders_idle_clusters_first(
    tmp_path, monkeypatch
):
    undated, dated = "example/sglang:v1", "example/sglang:nightly-20261020"
    sources = {"held-a": feed_source("m1", undated), "held-b": feed_source("m2", undated),
               "dated": feed_source("m3", dated), "idle": feed_source("m4", "example/vllm:v2"),
               "big": feed_source("m5", "example/vllm:v3")}  # fmt: skip
    clusters = {"held-a": "b", "held-b": "b", "dated": "a", "idle": "b", "big": "big"}
    families = {
        klaud.observation_key(source): {
            f"configs/nvidia-master.yaml:{name}": klaud.Family(
                frozenset({clusters[name]}), 1, 2, name == "idle")
        }
        for name, source in sources.items()
    }  # fmt: skip
    families[("unpublished",)] = {
        "configs/nvidia-master.yaml:claimed": klaud.Family(frozenset({"big"}), 1, 30, False)}
    items = [{"source-status": "baseline", "source-id": name, "source": source,
              "benchmark-age-days": 1} for name, source in sources.items()]  # fmt: skip
    claimed = OwnedCandidate(id=candidate_id("claimed"), family="configs/nvidia-master.yaml:claimed",
                             base="a" * 40)  # fmt: skip

    def read(_repository, path, paginate=False):
        if path.startswith("pulls?state=closed&"):
            return []
        assert paginate, path
        return [[]]

    frozen_clock(monkeypatch)
    monkeypatch.setenv("GITHUB_REPOSITORY", "example/project")
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    monkeypatch.setattr(klaud.subprocess, "check_output", lambda *_args, **_kwargs: "a" * 40)
    monkeypatch.setattr(klaud, "fetch_catalog", lambda _policy: (items, []))
    monkeypatch.setattr(klaud, "github_read", read)
    monkeypatch.setattr(klaud, "live_families", lambda _root: families)
    monkeypatch.setattr(claims, "family_owners", lambda _repository: [
        Ownership(run_id=7, candidates=[claimed])])
    monkeypatch.setattr(klaud, "fetch_capacity", lambda _policy: api.Capacity(
        idle={"a": 2, "b": 6, "big": 20}, queued={}, size={"a": 8, "b": 8, "big": 20}))
    monkeypatch.setattr(klaud, "recent_candidate_outcomes", lambda _repository, _hours: (
        klaud.History(recent={}, paused_weights=set(), infra_clusters={"a"}, held_families=set(),
                      held_images={identity(undated)[:16]: utc(ago(80)),
                                   identity(dated)[:16]: utc(ago(80))})))
    directory = tmp_path / "klaud"

    klaud.plan(tmp_path, directory, 64, 24, 5)

    contexts = json.loads((directory / "candidates.json").read_text())
    assert [(context["family"].rsplit(":", 1)[1], context["sweep-node-demand"],
             context["smoke-covers-full"]) for context in contexts] == [
        ("idle", 2, True), ("dated", 2, False)]  # fmt: skip
    assert json.loads((directory / klaud.CLUSTER_STATE).read_text()) == {
        "committed-demand": {"big": 20}, "infra-backoff": ["a"], "headroom": {"a": 2, "b": 6},
        "size": {"a": 8, "b": 8, "big": 20}, "finished-backlog": {}}  # fmt: skip


def open_pull(number, finished):
    """A ready Klaud PR whose status label a maintainer removed, or a draft with a stale label."""
    return {"number": number, "title": f"Klaud PR {number}", "html_url": f"https://pr/{number}",
            "user": {"login": "Klaud-Cold"}, "draft": not finished, "created_at": ago(1),
            "head": {"ref": f"klaud/auto-{number:016x}-{'0' * 16}", "sha": "a" * 40,
                     "repo": {"full_name": "example/project"}},
            "labels": [] if finished else [{"name": "klaud-finished"}]}  # fmt: skip


def plan_curves(tmp_path, monkeypatch, curves, *, history=None, closed=None, pulls=(), batch=64,
                slots=5):  # fmt: skip
    """Plan ``curves`` that all serve one feed observation; return their order and cluster state.

    ``pulls`` lists (number, master config, finished) of open Klaud PRs.
    """
    source = feed_source("m1", "example/sglang:v1")
    target = klaud.Family(frozenset({"a"}), 1, 1, False)
    masters = {number: master for number, master, _ in pulls}

    def read(_repository, path, paginate=False):
        if path.startswith("pulls?state=open&"):
            return [[open_pull(number, finished) for number, _, finished in pulls]]
        if path.startswith("pulls/"):
            return [[{"filename": f"inferencex-e2e/configs/{masters[int(path.split('/')[1])]}"},
                     {"filename": "inferencex-e2e/perf-changelog.yaml"}]]  # fmt: skip
        return [[]]

    frozen_clock(monkeypatch)
    monkeypatch.setenv("GITHUB_REPOSITORY", "example/project")
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    monkeypatch.setattr(klaud.subprocess, "check_output", lambda *_args, **_kwargs: "a" * 40)
    monkeypatch.setattr(klaud, "fetch_catalog", lambda _policy: ([
        {"source-status": "baseline", "source-id": "1", "source": source,
         "benchmark-age-days": 0}], []))  # fmt: skip
    monkeypatch.setattr(klaud, "github_read", read)
    monkeypatch.setattr(klaud, "live_families", lambda _root: {
        klaud.observation_key(source): dict.fromkeys(curves, target)})  # fmt: skip
    monkeypatch.setattr(claims, "family_owners", lambda _repository: [])
    monkeypatch.setattr(klaud, "fetch_capacity", lambda _policy: api.Capacity(
        idle={"a": 8}, queued={}, size={"a": 8}))  # fmt: skip
    monkeypatch.setattr(klaud, "maintainer_closed_families", lambda _repository, _now: (
        closed or (set(), set())))  # fmt: skip
    monkeypatch.setattr(klaud, "recent_candidate_outcomes", lambda _repository, _hours: (
        history or klaud.History(recent={}, paused_weights=set(), held_images={},
                                 infra_clusters=set(), held_families=set())))  # fmt: skip

    klaud.plan(tmp_path, tmp_path / "klaud", batch, 24, slots)

    contexts = json.loads((tmp_path / "klaud/candidates.json").read_text())
    state = json.loads((tmp_path / "klaud" / klaud.CLUSTER_STATE).read_text())
    return [context["family"] for context in contexts], state


def test_plan_holds_a_curve_sharing_a_key_with_a_timed_out_curve(tmp_path, monkeypatch):
    timed_out = identity("configs/nvidia-master.yaml:agg")[:16]
    history = klaud.History(recent={}, paused_weights=set(), held_images={}, infra_clusters=set(),
                            held_families={timed_out})  # fmt: skip
    curves = ["configs/nvidia-master.yaml:agg,mtp", "configs/nvidia-master.yaml:other"]

    planned, _ = plan_curves(tmp_path, monkeypatch, curves, history=history)

    assert planned == ["configs/nvidia-master.yaml:other"]


def test_plan_retries_a_requeued_candidate_despite_its_maintainer_close(tmp_path, monkeypatch):
    requeued, closed = "configs/nvidia-master.yaml:requeued", "configs/nvidia-master.yaml:closed"
    ids = {curve: candidate_id(curve, identity("example/sglang:v1")[:16])
           for curve in (requeued, closed)}  # fmt: skip
    history = klaud.History(recent=dict.fromkeys(ids.values(), "closed-by-maintainer"),
                            paused_weights=set(), held_images={}, infra_clusters=set(),
                            held_families=set())  # fmt: skip

    planned, _ = plan_curves(tmp_path, monkeypatch, [requeued, closed], history=history,
                             closed=(set(), {ids[requeued]}))  # fmt: skip

    assert planned == [requeued]


def test_plan_counts_finished_prs_per_master_config_and_reviews_backlogged_ones_last(
    tmp_path, monkeypatch
):
    busy, quiet = "configs/nvidia-master.yaml:busy", "configs/amd-master.yaml:quiet"
    pulls = [*((number, "nvidia-master.yaml", True) for number in range(8)),
             (8, "nvidia-master.yaml", False), (9, "amd-master.yaml", True)]  # fmt: skip

    planned, state = plan_curves(tmp_path, monkeypatch, [busy, quiet], pulls=pulls)

    assert planned == [quiet, busy]
    assert state["finished-backlog"] == {"nvidia-master.yaml": 8, "amd-master.yaml": 1}


@pytest.mark.parametrize(("slots", "batch", "reviewed"), [
    pytest.param(1, 64, 7, id="three-per-free-slot-plus-four"),
    pytest.param(5, 10, 10, id="capped-at-the-batch-size"),
])  # fmt: skip
def test_plan_sizes_the_review_batch_from_the_free_slots(
    tmp_path, monkeypatch, slots, batch, reviewed
):
    curves = [f"configs/nvidia-master.yaml:curve-{n:02}" for n in range(20)]

    planned, _ = plan_curves(tmp_path, monkeypatch, curves, batch=batch, slots=slots)

    assert len(planned) == reviewed


def regeneration_wave(tmp_path, monkeypatch, proceeding, blocked=0, cluster=None, backlog=None):
    """Write a review of ``proceeding`` candidates plus a duplicate and its family's sibling.

    Each candidate targets its own cluster, or all share ``cluster``. ``blocked`` more
    candidates lead the order on cluster "full", which has no headroom.
    """
    contexts = [wave_context(f"family-{n}", [cluster or f"c{n}"]) for n in range(proceeding)]
    duplicate, sibling = wave_context("duplicate", ["d"]), wave_context("duplicate", ["d"])
    sibling["id"] = candidate_id("duplicate", "1" * 16)
    contexts[1:1] = [duplicate, sibling]
    contexts[:0] = [wave_context(f"full-{n}", ["full"]) for n in range(blocked)]
    (tmp_path / "candidates.json").write_text(json.dumps(contexts))
    clusters = {c for context in contexts for c in context["clusters"]} - {"full"}
    (tmp_path / klaud.CLUSTER_STATE).write_text(json.dumps({
        "committed-demand": {}, "infra-backoff": [], "headroom": dict.fromkeys(clusters, 100),
        "size": {**dict.fromkeys(clusters, 100), "full": 8},
        "finished-backlog": backlog or {}}))  # fmt: skip
    monkeypatch.setenv("GITHUB_REPOSITORY", "example/project")
    monkeypatch.setenv("KLAUD_PR_REVIEW", json.dumps({"decisions": [
        {"candidate-id": context["id"],
         "decision": "duplicate" if context is duplicate else "proceed",
         "family": context["family"], "pull-requests": [9] if context is duplicate else [],
         "baseline-model": None if context is duplicate else "Model", "reason": "Reviewed"}
        for context in contexts
    ]}))  # fmt: skip
    published = []
    monkeypatch.setattr(reporting, "publication", lambda _repository, candidate, _model: (
        published.append(candidate["family"].rsplit(":", 1)[1]) or SimpleNamespace(heads=[])))
    return published


@pytest.mark.parametrize(("active", "regenerated"), [
    pytest.param(0, [f"family-{n}" for n in range(8)], id="five-slots-and-a-margin-of-five"),
    pytest.param(4, [f"family-{n}" for n in range(6)], id="one-slot-and-a-margin-of-five"),
    pytest.param(5, [], id="no-slot"),
])  # fmt: skip
def test_regeneration_covers_only_the_candidates_select_can_reach(
    tmp_path, monkeypatch, active, regenerated
):
    published = regeneration_wave(tmp_path, monkeypatch, 8)
    monkeypatch.setattr(sys, "argv", [
        "klaud", "regenerate-producers", "--max-candidates-per-run", "5",
        "--active-sessions", str(active), "--directory", str(tmp_path)])  # fmt: skip

    assert klaud.main() == 0
    assert published == regenerated
    assert (tmp_path / klaud.PRODUCERS).read_text() == (
        json.dumps({family: {} for family in [f"configs/nvidia-master.yaml:{name}"
                                              for name in regenerated]}) + "\n")  # fmt: skip


def test_regeneration_limit_stops_at_the_slots_plus_margin(tmp_path, monkeypatch):
    published = regeneration_wave(tmp_path, monkeypatch, 12)

    klaud.regenerate_producers(tmp_path, 2, active_sessions=0)

    assert published == [f"family-{n}" for n in range(7)]


def test_regeneration_skips_candidates_the_waves_capacity_cannot_take(tmp_path, monkeypatch):
    published = regeneration_wave(tmp_path, monkeypatch, 8, blocked=6)

    klaud.regenerate_producers(tmp_path, 1, active_sessions=0)

    assert published == [f"family-{n}" for n in range(6)]


def test_regeneration_prepares_a_few_alternates_per_cluster(tmp_path, monkeypatch):
    published = regeneration_wave(tmp_path, monkeypatch, 6, cluster="a")

    klaud.regenerate_producers(tmp_path, 5, active_sessions=0)

    assert published == ["family-0", "family-1", "family-2"]


@pytest.mark.parametrize(("finished", "regenerated"), [
    pytest.param(7, ["family-0", "family-1", "family-2"], id="below-the-cap"),
    pytest.param(8, [], id="at-the-cap"),
])  # fmt: skip
def test_regeneration_skips_candidates_of_a_backlogged_master_config(
    tmp_path, monkeypatch, finished, regenerated
):
    published = regeneration_wave(tmp_path, monkeypatch, 3,
                                  backlog={"nvidia-master.yaml": finished})  # fmt: skip

    klaud.regenerate_producers(tmp_path, 5, active_sessions=0)

    assert published == regenerated


@pytest.mark.parametrize(("record", "summary"), [
    pytest.param({"runs": [9]}, "Parked on 1 owned runs; recovery resumes the candidate.",
                 id="waiting-on-a-run"),
    pytest.param({"runs": [], "rerun": {"run": 9, "attempt": 1,
                                        "not-before": "2026-10-26T12:45:00Z"}},
                 "Parked to rerun the failed jobs of run 9 after 2026-10-26T12:45:00Z;"
                 " recovery then resumes the candidate.", id="delayed-rerun"),
])  # fmt: skip
def test_a_parked_sessions_summary_names_its_scheduled_rerun(
    tmp_path, monkeypatch, record, summary
):
    world = ended_session_sweep(monkeypatch, conclusion=None, sessions=(None,))
    world.comments.append({**park_comment(world, 1), "body": (
        lifecycle.park_marker(42, world.session.candidate.id, 1) + json.dumps(record)
        + "\n-->\nParked.")})  # fmt: skip
    (tmp_path / "session-end.json").write_text(json.dumps({"status": "parked"}))
    monkeypatch.setattr(lifecycle, "current_session", lambda: world.session)
    monkeypatch.setenv("KLAUD_SESSION", "1")
    monkeypatch.setenv("KLAUD_EVIDENCE", str(tmp_path))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary.md"))

    assert klaud.save_diagnostics(
        tmp_path / "execution.json", tmp_path / "structured.json", "success",
        tmp_path / "diagnostics.json",
    )  # fmt: skip
    assert (tmp_path / "summary.md").read_text().startswith(summary + " / ")
