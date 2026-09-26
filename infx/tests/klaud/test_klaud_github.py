import json
import os
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest

from infx.klaud import __main__ as klaud
from infx.klaud import api, claims, github, lifecycle, reporting, validation
from infx.klaud.models import CandidateOutcome, Feed, OwnedCandidate, identity


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

    def resolve(_repository, candidate, _context, _model, _goal):
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

    monkeypatch.setenv("GITHUB_REPOSITORY", "example/project")
    monkeypatch.setenv("GITHUB_RUN_ID", "42")
    monkeypatch.setenv("KLAUD_PR_REVIEW", json.dumps(review))
    monkeypatch.setattr(klaud, "fetch_capacity", lambda _policy: {"cluster-a"})
    monkeypatch.setattr(reporting, "resolve_baseline", resolve)
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
    selected = json.loads((tmp_path / second_id / "candidate.json").read_text())
    assert selected["baseline-preflight-required"] is True


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
        reporting.prepare_baseline(
            session, {**context, "baseline-preflight-required": True}, "Model", goal
        )


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
    current = {"single_node": {"all": [{**entry, "conc": [1]}]}}
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

    def canonical_matrix(_repository, head, _family, *, historical=False):
        assert (head, historical) in ((base, False), (historical_head, True))
        return historical_matrix if historical else current

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
        )
