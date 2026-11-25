import os
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from specs import released_sets, video_spec

SPECS = Path(__file__).resolve().parents[1] / "specs"


@pytest.fixture(scope="module")
def spec() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model": {"id": "org/model", "revision": "a" * 40},
        "generation": {"fps": 24},
        "prompts": {"p": {"text": "a prompt", "requires_motion": True, "requires_sound": False}},
        "sets": {
            "s": {
                "plan_id": "plan",
                "repetitions": 2,
                "warmup_runs": 1,
                "cases": [{"case_id": "c", "prompt": "p", "seed": 7}],
            },
        },
        "reference": {},
    }


def test_hashes_ignore_yaml_formatting(spec: dict[str, Any], tmp_path: Path) -> None:
    body = yaml.safe_dump(spec, sort_keys=False, indent=4, width=50)
    rewritten = tmp_path / "h3.yaml"
    rewritten.write_text(f"# rewritten copy\n{body}# end\n\n", encoding="utf-8")
    reloaded = video_spec.load_spec(rewritten)
    assert video_spec.plan_sha256(video_spec.plan_for(reloaded, "s")) == video_spec.plan_sha256(
        video_spec.plan_for(spec, "s")
    )
    assert video_spec.spec_sha256(reloaded) == video_spec.spec_sha256(spec)


def test_count_overrides_rename_the_plan_and_change_its_hash(spec: dict[str, Any]) -> None:
    full = video_spec.plan_for(spec, "s")
    assert full == {
        "model_id": "org/model",
        "model_revision": "a" * 40,
        "generation": {"fps": 24},
        "plan_id": "plan",
        "repetitions": 2,
        "warmup_runs": 1,
        "cases": [
            {
                "case_id": "c",
                "prompt": "a prompt",
                "seed": 7,
                "requires_motion": True,
                "requires_sound": False,
            }
        ],
    }
    single = video_spec.plan_for(spec, "s", repetitions=1, warmup_runs=0)
    assert single == {**full, "plan_id": "plan-r1-w0", "repetitions": 1, "warmup_runs": 0}
    assert video_spec.plan_sha256(single) != video_spec.plan_sha256(full)
    assert video_spec.plan_for(spec, "s", repetitions=2, warmup_runs=1) == full
    with pytest.raises(ValueError, match=re.escape("repetitions must be an integer >= 1")):
        video_spec.plan_for(spec, "s", repetitions=0)


RECORDED_FROZEN_PLAN_SHA256 = "8d51a472caa7db0f5d4c33adec1e9c9f852e7144cf25356085802edaf7d80d68"


def test_frozen_plan_reproduces_the_hash_its_reference_run_recorded() -> None:
    spec = video_spec.load_spec(SPECS / "h3.yaml")
    recorded = spec["reference"]["sglang"]
    plan = video_spec.plan_for(spec, "frozen-c124")
    assert plan["plan_id"] == recorded["plan_id"]
    assert video_spec.schema_version(plan) == 1
    assert (
        video_spec.plan_sha256(plan)
        == recorded["recorded_plan_sha256"]
        == RECORDED_FROZEN_PLAN_SHA256
    )


def test_v01_plan_declares_schema_2_with_its_fairness_and_warmup() -> None:
    spec = video_spec.load_spec(SPECS / "h3-v2.yaml")
    plan = video_spec.plan_for(spec, "release-v1")
    assert video_spec.schema_version(plan) == 2
    assert plan["fairness"] == spec["fairness"]
    assert plan["warmup_case"]["case_id"] == "warmup-only"


def _reuse_text(source: str, target: str) -> Callable[[dict[str, Any]], None]:
    return lambda doc: doc["prompts"][target].update(text=doc["prompts"][source]["text"])


@pytest.mark.parametrize(
    ("name", "mutate", "message"),
    [
        (
            "h3-v2.yaml",
            lambda doc: doc.update(schema_version=1),
            "spec must be a mapping with exactly the keys",
        ),
        (
            "h3-v2.yaml",
            lambda doc: doc["sets"]["release-v1"]["cases"].pop(),
            "needs 20 distinct measured prompts",
        ),
        (
            "h3-v2.yaml",
            _reuse_text("ceramic-wheel", "paper-fold"),
            "needs 20 distinct measured prompts",
        ),
        (
            "h3-v2.yaml",
            lambda doc: doc["sets"]["release-v1"]["warmup_case"].update(prompt="ceramic-wheel"),
            "warmup_case must be disjoint",
        ),
        (
            "h3-v2.yaml",
            _reuse_text("ceramic-wheel", "warmup-only"),
            "warmup_case must be disjoint",
        ),
        (
            "h3-v2.yaml",
            lambda doc: doc["fairness"]["schedule"].update(denoising_updates=50),
            "fairness.schedule must match",
        ),
        (
            "h3.yaml",
            lambda doc: doc["sets"]["engine-compare-v1"]["cases"][1].update(seed=11),
            "cases must not repeat a prompt text with the same seed",
        ),
    ],
)
def test_loader_rejects_invalid_workloads(
    tmp_path: Path, name: str, mutate: Callable[[dict[str, Any]], None], message: str
) -> None:
    document = yaml.safe_load((SPECS / name).read_text())
    mutate(document)
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match=re.escape(message)):
        video_spec.load_spec(path)


def test_canonical_json_bytes_is_sorted_compact_utf8_and_finite() -> None:
    assert (
        video_spec.canonical_json_bytes({"b": "é", "a": [1, 2.5]})
        == '{"a":[1,2.5],"b":"é"}'.encode()
    )
    with pytest.raises(ValueError, match="Out of range float values"):
        video_spec.canonical_json_bytes({"a": float("nan")})


def _git(repo: Path, *args: str) -> str:
    identity = ["-c", "user.name=test", "-c", "user.email=test@example.invalid"]
    environment = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    return subprocess.run(
        ["git", *identity, *args],
        cwd=repo,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _commit(repo: Path, spec: dict[str, Any], commit: dict[str, Any]) -> str:
    """Commit a specs tree and return the commit id.

    ``commit`` maps each set id in ``sets`` to its repetitions; ``steps``, ``file`` and ``loader``
    override the denoising steps, the spec file written and the loader's source.
    """
    specs = repo / released_sets.SPECS
    loader = (SPECS / "video_spec.py").read_text() + commit.get("loader", "")
    (specs / "video_spec.py").write_text(loader)
    document = {
        **spec,
        "generation": {"num_inference_steps": commit.get("steps", 40)},
        "sets": {
            set_id: {**spec["sets"]["s"], "repetitions": repetitions}
            for set_id, repetitions in commit["sets"].items()
        },
    }
    (specs / commit.get("file", "clip.yaml")).write_text(yaml.safe_dump(document))
    _git(repo, "add", "--all")
    _git(repo, "commit", "--quiet", "--message", "specs")
    return _git(repo, "rev-parse", "HEAD")


def _history(repo: Path, spec: dict[str, Any], commits: list[dict[str, Any]]) -> list[str]:
    """Start a repository with one commit per entry and return the commit ids, oldest first."""
    (repo / released_sets.SPECS).mkdir(parents=True)
    _git(repo, "init", "--quiet")
    return [_commit(repo, spec, commit) for commit in commits]


V1 = {"release-v1": 2}
V2_ADDED = {"sets": V1 | {"release-v2": 2}}
V2_EDITED = {"sets": V1 | {"release-v2": 3}}


@pytest.mark.parametrize(
    ("commits", "violation"),
    [
        # A release-v2 that changes the denoising steps changes release-v1 too.
        ([{"sets": V1}, {**V2_ADDED, "steps": 39}], "clip.yaml release-v1: plan changed"),
        ([{"sets": V1}, {"sets": {"release-v2": 2}}], "clip.yaml release-v1: removed"),
        (
            [
                {"sets": V1},
                {"sets": V1, "loader": "\n\ndef plan_sha256(plan):\n    return _sha256([plan])\n"},
            ],
            "clip.yaml release-v1: plan changed",
        ),
        # The first and last commits agree on every set the first one has.
        ([{"sets": V1}, V2_ADDED, V2_EDITED], "clip.yaml release-v2: plan changed"),
    ],
)
def test_a_released_set_cannot_change_or_disappear(
    spec: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    commits: list[dict[str, Any]],
    violation: str,
) -> None:
    start = _history(tmp_path, spec, commits)[0]
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match=violation):
        released_sets.main(["released_sets", start])


@pytest.mark.parametrize(
    "commits",
    [
        [{"sets": V1 | {"draft": 2}}, V2_ADDED, {"sets": V2_ADDED["sets"] | {"release-v3": 2}}],
        # New generation parameters go in a new spec file.
        [{"sets": V1}, {"sets": V1, "file": "longer-clip.yaml", "steps": 30}],
        # Undoing an edit restores the released plan.
        [{"sets": V1}, V2_ADDED, V2_EDITED, V2_ADDED],
        # A commit that cannot load its specs is skipped once a later commit loads again.
        [{"sets": V1}, {"sets": V1, "loader": "\n\nraise RuntimeError\n"}, {"sets": V1}],
    ],
)
def test_history_that_keeps_every_released_plan_passes(
    spec: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    commits: list[dict[str, Any]],
) -> None:
    start = _history(tmp_path, spec, commits)[0]
    monkeypatch.chdir(tmp_path)
    released_sets.main(["released_sets", start])


def test_a_later_commit_does_not_change_an_older_commits_verdict(
    spec: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    start, older, newer = _history(tmp_path, spec, [{"sets": V1}, V2_ADDED, V2_EDITED])
    _git(tmp_path, "checkout", "--quiet", "--detach", older)
    monkeypatch.chdir(tmp_path)
    released_sets.main(["released_sets", start])
    with pytest.raises(SystemExit, match="does not descend"):
        released_sets.main(["released_sets", newer])


def test_a_pull_request_may_edit_a_set_before_it_merges(
    spec: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (base,) = _history(tmp_path, spec, [{"sets": V1}])
    _git(tmp_path, "checkout", "--quiet", "-b", "feat/release-v2")
    _commit(tmp_path, spec, V2_ADDED)
    _commit(tmp_path, spec, V2_EDITED)
    _git(tmp_path, "checkout", "--quiet", "--detach", base)
    _git(tmp_path, "merge", "--quiet", "--no-ff", "--message", "merge", "feat/release-v2")
    monkeypatch.chdir(tmp_path)
    released_sets.main(["released_sets", base])
