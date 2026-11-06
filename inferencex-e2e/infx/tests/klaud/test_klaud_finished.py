import os
import subprocess

import pytest

from infx.klaud import __main__ as klaud
from infx.klaud import finished, github
from infx.klaud.models import OwnedCandidate, Ownership, identity, utc
from infx.tests.klaud.test_klaud_github import ended_session_sweep, recovery_pass
from infx.workflows.validate_perf_changelog import parse_changelog

REPOSITORY = "SemiAnalysisAI/InferenceX"
CANDIDATE = "1" * 16 + "-" + "2" * 16
BRANCH = f"klaud/auto-{CANDIDATE}"
CHANGELOG = "inferencex-e2e/perf-changelog.yaml"
CONFIG = "inferencex-e2e/configs/nvidia-master.yaml"


def git(cwd, *args):
    return subprocess.run(
        ["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.com", *args],
        cwd=cwd, check=True, capture_output=True, text=True,
    ).stdout.strip()  # fmt: skip


def block(key, link):
    return f'- config-keys:\n    - {key}\n  description:\n    - "Update {key}"\n  pr-link: {link}\n'


def commit(repo, message, **files):
    for path, text in files.items():
        (repo / path).write_text(text)
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", message)
    return git(repo, "rev-parse", "HEAD")


def finished_pr(head, number=7, branch=BRANCH, created="2026-10-20T00:00:00Z"):
    return {"number": number, "draft": False, "user": {"login": "Klaud-Cold"},
            "created_at": created,
            "head": {"ref": branch, "sha": head, "repo": {"full_name": REPOSITORY}}}  # fmt: skip


OLDER = f"klaud/auto-{'3' * 16}-{'4' * 16}"


def conflicting_world(tmp_path, monkeypatch, *, config_conflict=False, owned=False, older=False):
    """Serve finished PR 7 whose branch and main both appended to the changelog, and main
    and the PR also edited the master config when ``config_conflict``. With ``older``, PR 8 on
    OLDER, opened a day earlier, also appended. Return the bare remote, each PR branch's head,
    main's head and the local full-history checkout recovery runs in. Git ignores global
    config, so recovery must bring its own commit identity."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    seed = tmp_path / "seed"
    (seed / "inferencex-e2e/configs").mkdir(parents=True)
    git(seed, "init", "-q", "-b", "main")
    base = block("base-config", f"https://github.com/{REPOSITORY}/pull/1")
    start = commit(seed, "base", **{CHANGELOG: base, CONFIG: "image: v1\n"})
    heads = {}
    for branch, key, edits in [
        (BRANCH, "pr-config", {CONFIG: "image: v2\n"} if config_conflict else {}),
        *([(OLDER, "older-config", {})] if older else []),
    ]:
        git(seed, "checkout", "-qb", branch, start)
        heads[branch] = commit(seed, key, **{CHANGELOG: base + "\n" + block(key, "XXX"), **edits})
    git(seed, "checkout", "-q", "main")
    main = commit(seed, "main", **{
        CHANGELOG: base + "\n" + block("main-config", f"https://github.com/{REPOSITORY}/pull/41"),
        **({CONFIG: "image: v3\n"} if config_conflict else {})})  # fmt: skip
    remote, local = tmp_path / "remote.git", tmp_path / "local"
    git(tmp_path, "clone", "-q", "--bare", str(seed), str(remote))
    git(tmp_path, "clone", "-q", str(remote), str(local))
    monkeypatch.chdir(local)
    monkeypatch.setenv("GH_TOKEN", "fixture-token")
    monkeypatch.setattr(finished, "remote", lambda _repository: str(remote))
    pulls = [finished_pr(heads[BRANCH])] + (
        [finished_pr(heads[OLDER], 8, OLDER, "2026-10-19T00:00:00Z")] if older else []
    )
    artifacts = [{"id": 5, "expired": False, "workflow_run": {"id": 42}}] if owned else []

    def items(_repository, path, _key=None):
        if path == "pulls?state=open&per_page=100":
            return pulls
        if path.startswith("actions/artifacts?name=klaud-ownership"):
            return artifacts
        if path.startswith(
            ("actions/workflows/e2e-tests.yml/runs?", "actions/workflows/run-sweep.yml/runs?")
        ):
            return []
        raise AssertionError(path)

    def read(_repository, path):
        pull = next((pull for pull in pulls if path == f"pulls/{pull['number']}"), None)
        return {**pull, "mergeable": False} if pull else pytest.fail(path)

    monkeypatch.setattr(github, "items", items)
    monkeypatch.setattr(github, "read", read)
    monkeypatch.setattr(github, "download_json", lambda _r, _artifact, destination: (
        destination / "ownership.json").write_text(Ownership(run_id=42, candidates=[
            OwnedCandidate(id=CANDIDATE, family="configs/nvidia-master.yaml:fixture",
                           base="a" * 40)]).model_dump_json(by_alias=True)))  # fmt: skip
    return remote, heads, main, local


def test_a_changelog_only_conflict_is_merged_with_the_resolver_and_pushed_without_a_sweep(
    tmp_path, monkeypatch
):
    remote, heads, main, local = conflicting_world(tmp_path, monkeypatch)
    calls, run = [], finished.git
    monkeypatch.setattr(finished, "git", lambda *args, **kwargs: (
        calls.append((args[0], kwargs.get("env") or {})) or run(*args, **kwargs)))  # fmt: skip

    finished.sync_conflicts(REPOSITORY, set(), [])

    pushed = git(remote, "rev-parse", f"refs/heads/{BRANCH}")
    assert git(remote, "rev-list", "--parents", "-n1", pushed).split()[1:] == [heads[BRANCH], main]
    # [skip-sweep] keeps run-sweep.yml from starting a new sweep on the synchronize event.
    assert git(remote, "log", "-1", "--format=%s%n%cn", pushed) == (
        f"Merge main into {BRANCH} [skip-sweep]\ngithub-actions[bot]"
    )
    resolved = parse_changelog(git(remote, "show", f"{pushed}:{CHANGELOG}").encode() + b"\n", "x")
    assert [entry["config-keys"][0] for entry in resolved] == [
        "base-config",
        "main-config",
        "pr-config",
    ]
    assert resolved[-1]["pr-link"] == f"https://github.com/{REPOSITORY}/pull/7"
    assert len(git(local, "worktree", "list").splitlines()) == 1
    # A checkout without persisted credentials still reads and pushes a private repository.
    remote_calls = [env for command, env in calls if command in ("fetch", "push")]
    assert len(remote_calls) == 2
    assert all(env.get("GIT_CONFIG_VALUE_0", "").startswith("AUTHORIZATION: basic ")
               for env in remote_calls)  # fmt: skip


@pytest.mark.parametrize(("config_conflict", "owned"), [
    pytest.param(True, False, id="conflict-beyond-the-changelog"),
    pytest.param(False, True, id="owned-by-a-session"),
])  # fmt: skip
def test_other_conflicts_and_owned_prs_are_left_untouched(
    tmp_path, monkeypatch, config_conflict, owned
):
    remote, heads, _, local = conflicting_world(
        tmp_path, monkeypatch, config_conflict=config_conflict, owned=owned
    )

    finished.sync_conflicts(REPOSITORY, set(), [])

    assert git(remote, "rev-parse", f"refs/heads/{BRANCH}") == heads[BRANCH]
    assert len(git(local, "worktree", "list").splitlines()) == 1


def test_a_pass_syncs_at_most_the_limit_oldest_first(tmp_path, monkeypatch, capsys):
    remote, heads, _, _ = conflicting_world(tmp_path, monkeypatch, older=True)
    monkeypatch.setattr(finished, "SYNC_LIMIT", 1)

    finished.sync_conflicts(REPOSITORY, set(), [])

    assert git(remote, "rev-parse", f"refs/heads/{OLDER}") != heads[OLDER]
    assert git(remote, "rev-parse", f"refs/heads/{BRANCH}") == heads[BRANCH]
    assert "changelog sync of PR #7 and later waits for the next pass" in capsys.readouterr().out


def test_a_conflict_beyond_the_changelog_does_not_use_up_the_limit(tmp_path, monkeypatch):
    remote, heads, _, _ = conflicting_world(tmp_path, monkeypatch, config_conflict=True, older=True)
    monkeypatch.setattr(finished, "SYNC_LIMIT", 1)
    monkeypatch.setattr(
        github,
        "read",
        lambda _r, path: (
            {**finished_pr(heads[BRANCH], created="2026-10-18T00:00:00Z"), "mergeable": False}
            if path == "pulls/7"
            else {**finished_pr(heads[OLDER], 8, OLDER, "2026-10-19T00:00:00Z"), "mergeable": False}
        ),
    )

    finished.sync_conflicts(REPOSITORY, set(), [])

    assert git(remote, "rev-parse", f"refs/heads/{BRANCH}") == heads[BRANCH]
    assert git(remote, "rev-parse", f"refs/heads/{OLDER}") != heads[OLDER]


def test_offered_images_name_the_image_of_the_candidate_choose_builds():
    source = {"model": "m1", "hardware": "h100", "framework": "sglang", "precision": "fp8",
              "spec_method": "none", "disagg": False, "benchmark_type": "agentic_traces",
              "isl": None, "osl": None, "image": "example/sglang:v2", "date": "2026-10-20"}  # fmt: skip
    family = "configs/nvidia-master.yaml:agg,mtp"
    families = {klaud.observation_key(source): {
        family: klaud.Family(frozenset({"a"}), 1, 1, False)}}  # fmt: skip
    items = [{"source-status": "baseline", "source-id": "1", "source": source,
              "benchmark-age-days": 0}]  # fmt: skip

    [candidate] = klaud.choose(items, {"a": 8}, set(), families)
    prefix, image = candidate["id"].split("-")

    offered = klaud.offered_images(items, families)
    assert offered[prefix] == {image}
    assert offered[identity("configs/nvidia-master.yaml:agg")[:16]] == {image}


NEWER = {"1" * 16: {"3" * 16}}


@pytest.mark.parametrize(("created", "offered", "owned", "closed"), [
    pytest.param("2026-10-20T00:00:00Z", NEWER, False, True, id="stale-with-newer-image"),
    pytest.param("2026-10-22T00:00:01Z", NEWER, False, False, id="fresh"),
    pytest.param("2026-10-20T00:00:00Z", {"1" * 16: {"2" * 16, "3" * 16}}, False, False,
                 id="image-still-offered"),
    pytest.param("2026-10-20T00:00:00Z", {}, False, False, id="family-not-offered"),
    pytest.param("2026-10-20T00:00:00Z", NEWER, True, False, id="owned-by-a-session"),
])  # fmt: skip
def test_a_stale_finished_pr_whose_family_moved_to_a_newer_image_is_requeued(
    monkeypatch, created, offered, owned, closed
):
    pull = {**finished_pr("c" * 40), "created_at": created}
    writes = []
    monkeypatch.setenv("KLAUD_REQUEUE_TOKEN", "pat-fixture")
    monkeypatch.setattr(
        github,
        "items",
        lambda _r, path, _key=None: (
            [] if path.startswith("actions/artifacts?") else pytest.fail(path)
        ),
    )
    monkeypatch.setattr(
        github,
        "write",
        lambda _r, path, method, payload=None, *, token=None: writes.append(
            (method, path, payload, token)
        ),
    )
    claimed = [
        Ownership(
            run_id=42,
            candidates=[
                OwnedCandidate(
                    id=CANDIDATE, family="configs/nvidia-master.yaml:fixture", base="a" * 40
                )
            ],
        )
    ]

    numbers = finished.requeue_superseded(
        REPOSITORY,
        [pull, {**pull, "number": 8, "draft": True}],
        offered,
        claimed if owned else [],
        utc("2026-10-27T00:00:00Z"),
    )

    assert numbers == ([7] if closed else [])
    assert [(method, path, token) for method, path, _, token in writes] == ([
        ("POST", "issues/7/labels", "pat-fixture"),
        ("POST", "issues/7/comments", "pat-fixture"),
        ("PATCH", "pulls/7", "pat-fixture"),
    ] if closed else [])  # fmt: skip
    if closed:
        assert writes[0][2] == {"labels": ["klaud-requeue"]}
        assert writes[1][2]["body"].startswith("**requeued** · ")
        assert writes[2][2] == {"state": "closed"}


def test_recovery_syncs_finished_prs_against_live_work_before_the_sweep_marks_open_prs_live(
    monkeypatch, tmp_path
):
    world = ended_session_sweep(monkeypatch, jobs=["success", "cancelled"])
    ready = {**world.pull, "number": 8, "draft": False,
             "head": {**world.pull["head"], "ref": BRANCH}}  # fmt: skip
    serve, calls = github.items, []
    monkeypatch.setattr(github, "items", lambda repository, path, key=None: (
        [world.pull, ready] if path == "pulls?state=open&per_page=100"
        else serve(repository, path, key)))  # fmt: skip
    # A slow sync must not cost the wave its recovery outputs, so both exist before it starts.
    monkeypatch.setattr(finished, "sync_conflicts", lambda *args: calls.append((
        *args, (tmp_path / "klaud-recovery.json").exists(),
        (tmp_path / "output").read_text())))  # fmt: skip

    recovery_pass(monkeypatch, tmp_path)

    [(repository, live, claimed, blocked_written, output)] = calls
    assert repository == "example/project"
    # The sweep counts every open Klaud PR as live; the sync must still see the finished one.
    assert CANDIDATE not in live
    assert [owned.id for owner in claimed for owned in owner.candidates] == [
        world.session.candidate.id
    ]
    assert blocked_written
    assert output.startswith("active-sessions=")
