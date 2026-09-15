import copy
import io
import json
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest
from hypothesis import given, strategies as st

from infx import github
from infx.workflows import reuse, reuse_comment
from test_acknowledge_sweep_reuse import bot_status, event_for, make_request_case


@given(pinned=st.one_of(st.none(), st.integers(1, 10**15)), following=st.integers(1, 10**15),
       indent=st.text(alphabet=" \t", max_size=8), newline=st.sampled_from(["\n", "\r\n"]),
       earlier=st.booleans())
def test_reuse_commands_are_line_scoped(pinned, following, indent, newline, earlier):
    command = indent + "/reuse-sweep-run" + (f" {pinned}" if pinned else "") + indent
    body = newline.join([*(["/reuse-sweep-run 99"] if earlier else []), command, str(following), "Review complete."])
    assert reuse.parse_reuse_command(body) == (True, pinned)
    assert reuse.parse_reuse_command("Please use " + command + " later") == (False, None)


@given(pinned=st.booleans(), association=st.sampled_from(["OWNER", "MEMBER", "COLLABORATOR", "CONTRIBUTOR", "NONE", ""]),
       conclusion=st.sampled_from(["success", "failure", "cancelled", "timed_out"]),
       expired=st.booleans(), complete=st.booleans(), related=st.booleans(),
       artifact=st.sampled_from(["results_bmk", "eval_results_all", "bmk_agentic_fixture", "logs"]),
       labels=st.sets(st.sampled_from(["sweep-enabled", "full-sweep", "evals-only", "agentx-fast"])))
def test_reuse_acknowledgment_is_fail_closed(pinned, association, conclusion, expired, complete, related, artifact, labels):
    with pytest.MonkeyPatch.context() as patch, redirect_stdout(io.StringIO()):
        case = make_request_case(patch)
        case["comment"].update(body="/reuse-sweep-run" + (" 123" if pinned else ""), author_association=association)
        case["run"].update(conclusion=conclusion, status="completed" if complete else "in_progress")
        case["commits"] = [{"sha": "tested-sha" if related else "foreign-sha"}]
        case["artifacts"] = [{"name": artifact, "expired": expired}]
        case["pr"]["labels"] = [{"name": label} for label in labels]
        human = {"id": 1, "content": "+1", "user": {"login": "maintainer"}}
        case["reactions"] = [copy.deepcopy(human)]
        expected = (association in {"OWNER", "MEMBER", "COLLABORATOR"} and complete and related
                    and not expired and artifact != "logs" and not labels.intersection({"evals-only", "agentx-fast"})
                    and conclusion in ({"success", "failure", "cancelled"} if pinned else {"success"}))
        for _ in range(2):
            code = reuse_comment.acknowledge("example/project", event_for(case), "test-token")
            assert code == (0 if expected else 1)
            assert bot_status(case) == (["+1"] if expected else ["-1"])
            assert human in case["reactions"]


@given(count=st.integers(0, 450), wrapped=st.booleans(), filter_value=st.text(max_size=40))
def test_github_pagination_keeps_filters_and_all_items(count, wrapped, filter_value):
    expected = [{"id": i} for i in range(count)]
    calls = []
    def api(repo, path, token, params=None):
        calls.append(dict(params))
        page = int(params["page"])
        records = expected[(page - 1) * 100:page * 100]
        return {"items": records} if wrapped else records
    params = {"branch": filter_value}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(github, "api", api)
        actual = github.paginate("example/project", "/items", "test-token", "items" if wrapped else "", params)
    assert actual == expected
    assert [int(call["page"]) for call in calls] == list(range(1, count // 100 + 2))
    assert all(call["branch"] == filter_value for call in calls)
    assert params == {"branch": filter_value}


@given(operations=st.lists(st.sampled_from(["accept", "reject", "withdraw", "unavailable", "stale"]), min_size=1, max_size=20))
def test_reactions_follow_request_edits_without_touching_humans(operations):
    with pytest.MonkeyPatch.context() as patch, redirect_stdout(io.StringIO()):
        case = make_request_case(patch)
        human = {"id": 1, "content": "heart", "user": {"login": "maintainer"}}
        case["reactions"] = [copy.deepcopy(human)]
        expected = []
        for operation in operations:
            case["comment"]["body"] = {"withdraw": "withdrawn", "reject": "/reuse-sweep-run nope"}.get(operation, "/reuse-sweep-run 123")
            case["fail_path"] = "/actions/runs/123" if operation == "unavailable" else None
            event = event_for(case, action="edited")
            if operation == "stale":
                case["comment"]["body"] = "/reuse-sweep-run 456"
            else:
                expected = [] if operation == "withdraw" else ["+1" if operation == "accept" else "-1"]
            reuse_comment.acknowledge("example/project", event, "test-token")
            assert bot_status(case) == expected
            assert human in case["reactions"]


@given(mode=st.sampled_from(["push-main", "push-branch", "synchronize", "opened", "manual"]),
       authorized=st.booleans(), merged=st.booleans(), usable=st.booleans(), pinned=st.booleans(),
       associations=st.integers(0, 2))
def test_reuse_cli_never_skips_work_without_a_valid_authorized_source(mode, authorized, merged, usable, pinned, associations):
    with tempfile.TemporaryDirectory() as directory, pytest.MonkeyPatch.context() as patch:
        case = make_request_case(patch)
        case["comment"].update(author_association="OWNER" if authorized else "NONE",
                               body="/reuse-sweep-run" + (" 123" if pinned else ""))
        case["pr"]["merged_at"] = "2026-01-02T00:00:00Z" if merged else None
        case["artifacts"][0]["expired"] = not usable
        original_api = github.api
        def api(repo, path, token, params=None, **kwargs):
            if path == "/commits/merge-sha/pulls":
                return [{"number": number} for number in range(7, 7 + associations)]
            if path == "/issues/8/comments":
                return []
            return original_api(repo, path, token, params, **kwargs)
        patch.setattr(github, "api", api)
        patch.setenv("GH_TOKEN", "test-token")
        output = Path(directory) / "output"
        event = "push" if mode.startswith("push") else "pull_request" if mode in {"synchronize", "opened"} else "workflow_dispatch"
        patch.setattr("sys.argv", ["reuse", "--repo", "example/project", "--commit-sha", "merge-sha",
                                  "--event-name", event, "--event-action", mode, "--pr-number", "7",
                                  "--ref", "refs/heads/main" if mode == "push-main" else "refs/heads/feature",
                                  "--github-output", str(output)])
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr), pytest.raises(SystemExit) as error:
            reuse.cli()
        fails = authorized and ((mode == "synchronize" and not usable) or
                                (mode == "push-main" and associations > 0 and
                                 (associations > 1 or not merged or not usable)))
        assert error.value.code == int(fails), stderr.getvalue()
        if fails:
            assert not output.exists()
            assert stdout.getvalue() == ""
        else:
            actual = json.loads(stdout.getvalue())
            assert actual["reuse-enabled"] == str(mode == "push-main" and authorized and associations == 1 and merged and usable).lower()
            assert actual["skip-pr-sweep"] == str(mode == "synchronize" and authorized and usable).lower()
            assert dict(line.split("=", 1) for line in output.read_text().splitlines()) == actual
