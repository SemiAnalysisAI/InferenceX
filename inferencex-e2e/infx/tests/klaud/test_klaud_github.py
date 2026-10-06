import json
import os
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from infx.klaud import __main__ as klaud
from infx.klaud import api, claims, github, lifecycle, reporting, validation
from infx.klaud.models import CandidateOutcome, Feed, OwnedCandidate, identity
from infx.tests.historical_revision import (
    FAMILY,
    KEY,
    POINTS,
    commit_history,
    forbid_current_config_parsing,
)


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


def test_recent_candidates_use_all_current_base_workflow_artifacts(monkeypatch):
    candidate_id = "1" * 16 + "-" + "2" * 16
    base = "a" * 40
    calls = []

    def items(_repository, path, key):
        calls.append((path, key))
        if path.startswith("actions/workflows/klaud-plan.yml/runs?"):
            return [
                {"id": 42, "created_at": "9999-01-01T00:00:00Z", "head_sha": base},
                {"id": 43, "created_at": "9999-01-01T00:00:00Z", "head_sha": "b" * 40},
            ]
        if path == "actions/runs/42/artifacts?per_page=100":
            return [
                {
                    "name": f"klaud-candidate-{candidate_id}",
                    "expired": False,
                    "workflow_run": None,
                },
                {"name": "other-artifact", "expired": False},
            ]
        raise AssertionError(path)

    monkeypatch.setattr(klaud, "github_items", items)

    assert klaud.recent_candidate_ids("example/project", base, 24) == {candidate_id}
    assert calls[-1] == ("actions/runs/42/artifacts?per_page=100", "artifacts")


def test_select_continues_after_one_baseline_state_failure(tmp_path, monkeypatch):
    first_id = "1" * 16 + "-" + "2" * 16
    second_id = "3" * 16 + "-" + "4" * 16
    base = "a" * 40
    families = [
        "configs/nvidia-master.yaml:first-family",
        "configs/nvidia-master.yaml:second-family",
    ]
    contexts = [
        {"id": candidate_id, "family": family, "base": base, "source": {}}
        for candidate_id, family in zip((first_id, second_id), families, strict=True)
    ]
    (tmp_path / "candidates.json").write_text(json.dumps(contexts))
    review = {
        "decisions": [
            {
                "candidate-id": candidate_id,
                "decision": "proceed",
                "family": family,
                "telemetry-clusters": ["cluster-a"],
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
    assert selection["deferred-reason"] is None
    preflight = reporting.BaselinePreflight.model_validate_json(
        (tmp_path / second_id / "baseline-preflight.json").read_text()
    )
    assert preflight.candidate_id == second_id
    assert preflight.baseline_model == "Model"
    assert preflight.source_identity == identity({})
    selected = json.loads((tmp_path / second_id / "candidate.json").read_text())
    assert selected["baseline-model"] == "Model"


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
    # The current generator routes by cluster and records the recipe path; the producer did
    # not. The same public point must still freeze once, under the current key.
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
            "scenarios": {
                "fixed-seq-len": [{
                    "isl": 1024, "osl": 1024,
                    "search-space": [{"tp": 1, "conc-list": [2, 6]}],
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


@pytest.mark.parametrize("prefix", ["", "inferencex-e2e/"])
def test_canonical_family_matrix_reads_revision_layout(monkeypatch, family_configs, prefix):
    master, runners = family_configs
    files = {
        prefix + "configs/nvidia-master.yaml": yaml.safe_dump(master),
        prefix + "configs/runners.yaml": yaml.safe_dump(runners),
    }
    monkeypatch.setattr(
        github, "read",
        lambda *_: {"tree": [{"path": prefix.rstrip("/") or "configs"}]},
    )
    monkeypatch.setattr(github, "file_at", lambda _repo, _head, path: files[path])

    matrix = validation.canonical_matrix(
        "example/project", "a" * 40, "configs/nvidia-master.yaml:fixture",
    )

    assert [
        (row["model"], row["conc"], row["image"])
        for row in matrix["single_node"]["all"]
    ] == [
        ("example/model", 2, "example/image:stable"),
        ("example/model", 6, "example/image:stable"),
    ]
    assert matrix["evals"] == []


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
        [{"id": CANDIDATE_ID, "family": FAMILY, "base": "a" * 40, "source": source}]
    ))
    monkeypatch.setenv("GITHUB_REPOSITORY", "example/project")
    monkeypatch.setenv("GITHUB_RUN_ID", "42")
    monkeypatch.setenv("KLAUD_PR_REVIEW", json.dumps({"decisions": [{
        "candidate-id": CANDIDATE_ID, "decision": "proceed", "family": FAMILY,
        "telemetry-clusters": ["cluster-a"], "pull-requests": [],
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
    # The published baseline has c2 and c6; the current family dropped c6.
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

    families = klaud.live_families(tmp_path if from_repository_root else project)

    assert families == {
        ("dsr1", "fixture", "sglang", "fp8", "none", False, "single_turn",
         1024, 1024, "example/image:stable"): {"configs/nvidia-master.yaml:fixture"},
    }


def test_historical_families_ignore_leftover_nested_results(tmp_path, family_configs):
    master, runners = family_configs
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "nvidia-master.yaml").write_text(yaml.safe_dump(master))
    (configs / "runners.yaml").write_text(yaml.safe_dump(runners))

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
         1024, 1024, "example/image:stable"): {"configs/nvidia-master.yaml:fixture"},
    }
