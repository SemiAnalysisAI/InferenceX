"""Exercise the real workflow scripts with controlled GitHub responses."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
OPERATIONS = ("stage-results", "trusted-external-sweep")


def workflow(name, root=ROOT):
    return yaml.load((root / f".github/workflows/{name}.yml").read_text(), Loader=yaml.BaseLoader)


def scenario(operation):
    """Hand-built PR and artifact facts, independent of the production selectors."""
    pull = {
        "number": 42, "state": "open", "draft": False,
        "head": {"sha": "approved-head", "ref": "feature", "repo": {"full_name": "outside/repo"}},
        "base": {"sha": "base-head"}, "merge_commit_sha": "approved-merge",
        "labels": [{"name": "full-sweep-fail-fast"}],
    }
    payload = {
        "repository": {"full_name": "example/repo", "default_branch": "main"},
        "pull_request": copy.deepcopy(pull), "issue": {"number": 42, "pull_request": {}},
        "comment": {"body": "/stage-results", "user": {"login": "commenter"}},
        "label": {"name": "full-sweep-fail-fast"},
    }
    payload["action"] = "created" if operation == "stage-results" else "labeled"
    run = {
        "id": 101, "run_attempt": 2, "created_at": "2026-01-02T10:00:00Z",
        "head_sha": "approved-head", "status": "completed", "conclusion": "success",
        "path": ".github/workflows/run-sweep.yml", "event": "pull_request",
        "pull_requests": [{"number": 42}], "html_url": "https://example.test/runs/101",
    }
    return {
        "context": {"actor": "requester", "repo": {"owner": "example", "repo": "repo"},
                    "issue": {"number": 42}, "payload": payload},
        "permission": {"role_name": "write", "permission": "write"},
        "data": {
            "pull": pull, "commits": [{"sha": "approved-head"}], "runs": [run],
            "timeline": [{"event": "labeled", "label": {"name": "full-sweep-fail-fast"}, "created_at": "2026-01-01T00:00:00Z"}],
            "artifacts": {"101": [{"name": "changelog-metadata", "expired": False}, {"name": "results_bmk", "expired": False}]},
            "dispatchedRuns": [{"display_title": "e2e Test - External PR #42 @ approved-hea", "created_at": "2026-01-02T12:00:00Z", "html_url": "https://example.test/runs/303"}],
        },
    }


def run_workflow(operation, case):
    job = next(iter(workflow(operation)["jobs"].values()))
    return run_scripts(job["steps"], case)


def run_scripts(steps, case):
    result = subprocess.run(
        ["node", str(ROOT / "utils/changelog_gate_tests/workflow_script_runner.cjs")],
        input=json.dumps({**case, "steps": steps}), cwd=ROOT,
        capture_output=True, text=True, timeout=10, check=True,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize("operation,expected_actor", [
    ("stage-results", "commenter"), ("trusted-external-sweep", "requester"),
])
def test_permission_lookup_uses_original_requester(operation, expected_actor):
    case = scenario(operation)
    case["context"]["triggering_actor"] = "admin-rerunner"
    case["context"]["payload"]["sender"] = {"login": "payload-sender"}
    result = run_workflow(operation, case)
    assert not result["failures"]
    assert result["permissionRequests"] == [{
        "owner": "example", "repo": "repo", "username": expected_actor,
    }]


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("role,legacy,allowed", [
    ("admin", "admin", True), ("maintain", "write", True),
    ("maintain", "maintain", True), ("write", "write", True),
    ("triage", "read", False), ("read", "read", False), ("none", "none", False),
    ("custom-role", "admin", False), ("custom-role", "write", False), ("custom-role", "read", False),
    ("admin", "read", False), ("read", "admin", False),
    ("admin", "unknown", False), ("WRITE", "write", False), (" write ", "write", False),
])
def test_workflow_dispatch_requires_the_requested_repository_role(operation, role, legacy, allowed):
    case = scenario(operation)
    case["permission"] = {"role_name": role, "permission": legacy}
    result = run_workflow(operation, case)
    dispatches = [w for w in result["writes"] if w["method"] != "issues.createComment"]
    assert bool(dispatches) is allowed
    assert bool(result["failures"]) is not allowed
    if not allowed:
        assert f'permission "{legacy}"' in result["failures"][0]
        assert f'role "{role}"' in result["failures"][0]
        if operation == "stage-results":
            assert len(result["writes"]) == 1
            comment = result["writes"][0]["body"]
            assert "@commenter" in comment
            assert f'permission "{legacy}"' in comment
            assert f'role "{role}"' in comment


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("permission,error", [
    ({"role_name": "admin", "permission": "admin"}, True),
    (None, False), ([], False), ("not an object", False), ({}, False),
    ({"permission": True, "role_name": "admin"}, False),
    ({"permission": None, "role_name": "admin"}, False),
    ({"permission": "", "role_name": "admin"}, False),
    ({"permission": "admin", "role_name": []}, False),
    ({"permission": "admin", "role_name": {}}, False),
    ({"role_name": "admin"}, False),
    ({"permission": "admin"}, False),
    ({"permission": "admin", "role_name": None}, False),
    ({"permission": "admin", "role_name": True}, False),
    ({"permission": "admin", "role_name": ""}, False),
])
def test_unavailable_role_never_reaches_protected_effects(operation, permission, error):
    case = scenario(operation)
    case.update(permission=permission, permissionError=error)
    result = run_workflow(operation, case)
    assert result["failures"]
    assert result["writes"] == []
    assert result["outputs"] == {}


def test_staging_keeps_source_outputs_and_dispatch_contract():
    result = run_workflow("stage-results", scenario("stage-results"))
    assert result["failures"] == []
    assert result["outputs"]["request"] == {
        "run-id": "101", "run-attempt": "2", "run-date": "2026-01-02",
        "requested-by": "commenter", "run-url": "https://example.test/runs/101",
    }
    assert result["writes"][-1] == {
        "method": "repos.createDispatchEvent", "owner": "SemiAnalysisAI", "repo": "InferenceX-app",
        "event_type": "stage-results", "client_payload": {
            "source-repository": "example/repo", "pr-number": "42", "run-id": "101",
            "run-attempt": "2", "run-date": "2026-01-02", "requested-by": "commenter", "comment-id": "501",
        },
    }


def test_invalid_staging_command_preserves_usage_reply_without_role_lookup():
    case = scenario("stage-results")
    case["context"]["payload"]["comment"]["body"] = "/stage-results invalid"
    result = run_workflow("stage-results", case)
    assert result["failures"] == ["Unsupported /stage-results syntax"]
    assert [w["body"] for w in result["writes"]] == ["Usage: `/stage-results` or `/stage-results <run-id>`."]
    assert result["permissionRequests"] == []


@pytest.mark.parametrize("field,value", [
    ("head_sha", "unrelated-head"), ("event", "push"), ("path", "other.yml"),
    ("status", "in_progress"), ("conclusion", "skipped"),
])
def test_staging_role_does_not_override_run_eligibility(field, value):
    case = scenario("stage-results")
    case["data"]["runs"][0][field] = value
    result = run_workflow("stage-results", case)
    assert result["failures"]
    assert result["writes"] == []


@pytest.mark.parametrize("conclusion", ["success", "failure", "cancelled"])
@pytest.mark.parametrize("artifact", ["results_bmk", "eval_results_all", "bmk_agentic_example"])
def test_staging_preserves_partial_results_for_each_supported_artifact(conclusion, artifact):
    case = scenario("stage-results")
    case["data"]["runs"][0]["conclusion"] = conclusion
    case["data"]["artifacts"]["101"][1]["name"] = artifact
    result = run_workflow("stage-results", case)
    assert not result["failures"]
    assert result["writes"][-1]["event_type"] == "stage-results"


@pytest.mark.parametrize("change", ["missing-metadata", "expired-results", "no-current-label", "no-historical-label"])
def test_staging_role_does_not_override_labels_or_artifacts(change):
    case = scenario("stage-results")
    data = case["data"]
    if change == "missing-metadata":
        data["artifacts"]["101"].pop(0)
    elif change == "expired-results":
        data["artifacts"]["101"][1]["expired"] = True
    elif change == "no-current-label":
        data["pull"]["labels"] = []
    else:
        data["timeline"] = []
    result = run_workflow("stage-results", case)
    assert result["failures"]
    assert all(w["method"] == "issues.createComment" for w in result["writes"])


@pytest.mark.parametrize("pinned,associated,expected", [(True, True, True), (True, False, False), (False, True, False)])
def test_historical_staging_association_requires_explicit_run_id(pinned, associated, expected):
    case = scenario("stage-results")
    case["data"]["runs"][0].update(head_sha="historical-head", pull_requests=[{"number": 42 if associated else 43}])
    if pinned:
        case["context"]["payload"]["comment"]["body"] = "/stage-results 101"
    result = run_workflow("stage-results", case)
    assert bool(result["failures"]) is not expected
    assert any(w["method"] == "repos.createDispatchEvent" for w in result["writes"]) is expected


@pytest.mark.parametrize("change", ["closed", "draft", "same-repo", "advanced-head", "conflicting-labels", "no-merge"])
def test_external_approval_does_not_override_pr_integrity(change):
    case = scenario("trusted-external-sweep")
    pull = case["data"]["pull"]
    if change == "closed":
        pull["state"] = "closed"
    elif change == "draft":
        pull["draft"] = True
    elif change == "same-repo":
        pull["head"]["repo"]["full_name"] = "example/repo"
    elif change == "advanced-head":
        pull["head"]["sha"] = "new-head"
    elif change == "conflicting-labels":
        pull["labels"].append({"name": "sweep-enabled"})
    else:
        pull["merge_commit_sha"] = None
    result = run_workflow("trusted-external-sweep", case)
    assert result["failures"]
    assert result["writes"] == []


def test_external_dispatch_preserves_approved_refs_and_options():
    case = scenario("trusted-external-sweep")
    case["data"]["pull"]["labels"] += [{"name": "all-evals"}, {"name": "agentx-fast"}]
    result = run_workflow("trusted-external-sweep", case)
    assert result["failures"] == []
    assert result["writes"][0] == {
        "method": "actions.createWorkflowDispatch", "owner": "example", "repo": "repo",
        "workflow_id": "e2e-tests.yml", "ref": "main", "inputs": {
            "test-name": "External PR #42 @ approved-hea", "ref": "approved-merge",
            "changelog-base-ref": "base-head", "changelog-head-ref": "approved-head",
            "trim-conc": "false", "all-evals": "true", "evals-only": "false", "fail-fast": "true",
            "agentx-fast": "true", "pr-labels-json": '["full-sweep-fail-fast","all-evals","agentx-fast"]',
        },
    }


def signoff_case(event='pull_request_target'):
    case = scenario('trusted-external-sweep')
    case['context'].update(eventName=event, runId=99)
    case['context']['payload']['action'] = 'synchronize'
    case['data'].update(comments=[], reviews=[], inlineComments=[], statuses=[])
    case['data']['pull']['head']['sha'] = 'resolved-head'
    case['data']['pull']['merge_commit_sha'] = None
    return case


def signoff(identifier=11, timestamp='2026-01-01T12:00:00Z', **changes):
    return {'id': identifier, 'body': 'As a PR reviewer and CODEOWNER, I have reviewed this and have:',
            'user': {'login': 'reviewer', 'type': 'User'}, 'state': 'APPROVED',
            'commit_id': 'conflicting-head', 'submitted_at': timestamp, 'updated_at': timestamp, **changes}


@pytest.mark.parametrize('collection,kind,path', [
    ('comments', 'conversation comment', 'issues/comments/11'),
    ('reviews', 'review summary', 'pulls/42/reviews/11'),
    ('inlineComments', 'inline review comment', 'pulls/comments/11'),
])
def test_head_update_recovers_each_signoff_kind_on_the_current_head(collection, kind, path):
    case = signoff_case()
    case['data'][collection] = [signoff()]
    result = run_workflow('codeowner-signoff-verify', case)
    assert result['failures'] == []
    assert result['outputs']['resolve'] == {
        'proceed': 'true', 'pr-number': '42', 'head-sha': 'resolved-head',
        'signoff-author': 'reviewer', 'signoff-kind': kind,
        'signoff-fetch-cmd': f'gh api repos/example/repo/{path} --jq .body',
    }
    status_read = next(request for request in result['requests']
                       if request['method'] == 'repos.listCommitStatusesForRef')
    assert status_read['ref'] == 'resolved-head'
    assert result['writes'] == []


def test_head_update_uses_latest_existing_signoff_across_sources():
    case = signoff_case()
    case['data']['comments'] = [signoff(12, '2026-01-02T00:00:00Z')]
    case['data']['reviews'] = [signoff(13), signoff(14, '2026-01-03T00:00:00Z', state='DISMISSED')]
    case['data']['inlineComments'] = [signoff(15, '2026-01-04T00:00:00Z', body='withdrawn')]
    result = run_workflow('codeowner-signoff-verify', case)
    assert result['failures'] == []
    assert result['outputs']['resolve']['signoff-fetch-cmd'] == 'gh api repos/example/repo/issues/comments/12 --jq .body'


@pytest.mark.parametrize('permission', [
    {'permission': 'read', 'role_name': 'read'}, {'permission': 'write', 'role_name': 'custom'},
    {'permission': 'write'}, None,
])
@pytest.mark.parametrize('earlier_signoff', [False, True])
def test_catchup_cannot_substitute_an_unauthorized_signer(permission, earlier_signoff):
    case = signoff_case()
    case['permissionsByUser'] = {'outsider': permission}
    case['data']['reviews'] = [signoff()] if earlier_signoff else []
    case['data']['comments'] = [signoff(12, '2026-01-02T00:00:00Z', user={'login': 'outsider', 'type': 'User'})]
    result = run_workflow('codeowner-signoff-verify', case)
    assert result['failures'] == []
    assert result['outputs']['resolve']['proceed'] == str(earlier_signoff).lower()
    if earlier_signoff:
        assert result['outputs']['resolve']['signoff-fetch-cmd'] == 'gh api repos/example/repo/pulls/42/reviews/11 --jq .body'
    assert result['writes'] == []


@pytest.mark.parametrize('change', ['closed', 'draft', 'absent', 'withdrawn', 'dismissed', 'bot'])
def test_head_update_does_not_invent_a_signoff(change):
    case = signoff_case()
    if change != 'absent':
        case['data']['reviews'] = [signoff()]
    if change in {'closed', 'draft'}:
        case['data']['pull'].update(state='closed' if change == 'closed' else 'open', draft=change == 'draft')
    elif change == 'withdrawn':
        case['data']['reviews'][0]['body'] = 'withdrawn'
    elif change == 'dismissed':
        case['data']['reviews'][0]['state'] = 'DISMISSED'
    elif change == 'bot':
        case['data']['reviews'][0]['user']['type'] = 'Bot'
    result = run_workflow('codeowner-signoff-verify', case)
    assert result['failures'] == []
    assert result['outputs']['resolve']['proceed'] == 'false'
    assert 'head-sha' not in result['outputs']['resolve']


@pytest.mark.parametrize('state,creator,target,duplicate', [
    ('success', 'github-actions[bot]', 'pull/42#issuecomment-123', (True, True)),
    ('failure', 'github-actions[bot]', 'pull/42#issuecomment-123', (True, True)),
    ('pending', 'github-actions[bot]', 'pull/42#issuecomment-123', (False, False)),
    ('success', 'contributor', 'pull/42#issuecomment-123', (False, True)),
    ('success', 'github-actions[bot]', 'pull/43#issuecomment-123', (False, False)),
    ('failure', 'github-actions[bot]', 'actions/runs/99', (False, False)),
    ('error', 'github-actions[bot]', 'actions/runs/99', (False, False)),
])
@pytest.mark.parametrize('previous_pass', [False, True])
def test_head_update_skips_only_a_completed_verdict_for_this_pr(state, creator, target, duplicate, previous_pass):
    case = signoff_case()
    case['data']['reviews'] = [signoff()]
    case['data']['statuses'] = [{'context': 'codeowner-signoff-verify', 'state': state, 'creator': {'login': creator},
                                'target_url': f'https://github.com/example/repo/{target}'}]
    if previous_pass:
        case['data']['statuses'].append({'context': 'codeowner-signoff-verify', 'state': 'success',
                                        'creator': {'login': 'github-actions[bot]'},
                                        'target_url': 'https://github.com/example/repo/pull/42#issuecomment-100'})
    result = run_workflow('codeowner-signoff-verify', case)
    assert result['failures'] == []
    assert result['outputs']['resolve']['proceed'] == str(not duplicate[previous_pass]).lower()


@pytest.mark.parametrize('method', ['repos.listCommitStatusesForRef', 'issues.listComments',
                                  'pulls.listReviews', 'pulls.listReviewComments'])
def test_head_update_discovery_fails_closed_when_github_is_unavailable(method):
    case = signoff_case()
    case['data']['reviews'] = [signoff()]
    case['failMethod'] = method
    result = run_workflow('codeowner-signoff-verify', case)
    assert result['failures'] == ['GitHub unavailable']
    assert result['outputs'].get('resolve', {}).get('proceed') != 'true'


def test_head_update_finds_a_signoff_after_the_first_page():
    case = signoff_case()
    case['data']['reviews'] = [signoff(identifier, body='Looks good') for identifier in range(120)] + [signoff(121)]
    result = run_workflow('codeowner-signoff-verify', case)
    assert result['failures'] == []
    assert result['outputs']['resolve']['signoff-fetch-cmd'] == 'gh api repos/example/repo/pulls/42/reviews/121 --jq .body'


@pytest.mark.parametrize('event', ['issue_comment', 'pull_request_review', 'pull_request_review_comment',
                                 'pull_request_target', 'workflow_dispatch'])
@pytest.mark.parametrize('permission', [
    {'role_name': 'read', 'permission': 'read'},
    {'role_name': 'custom', 'permission': 'write'},
    {'permission': 'admin'},
])
def test_unauthorized_signoff_requests_do_not_start_the_verifier(event, permission):
    case = signoff_case(event)
    case['permission'] = permission
    case['context']['payload']['review' if event == 'pull_request_review' else 'comment'] = signoff()
    case['context']['payload']['inputs'] = {'comment_url': 'https://github.com/example/repo/pull/42#pullrequestreview-11'}
    case['data']['reviews'] = [signoff()]
    result = run_workflow('codeowner-signoff-verify', case)
    assert result['outputs']['resolve']['proceed'] == 'false'
    assert result['writes'] == []


@pytest.mark.parametrize('author,run,outcome,state', [
    (7, '99 attempt=1', 'success', 'success'),
    (8, '99 attempt=1', 'success', 'failure'),
    (7, '98 attempt=1', 'success', 'failure'),
    (7, '99 attempt=2', 'success', 'failure'),
    (7, '99 attempt=1', 'failure', 'failure'),
    (7, '99 attempt=1', 'cancelled', 'failure'),
    (7, '99 attempt=1', 'skipped', 'failure'),
])
def test_verdict_requires_the_verifier_and_current_attempt(author, run, outcome, state):
    case = signoff_case()
    case['context'].update(run_id=99, run_attempt=1)
    case['stepsState'] = {'verify': {'outcome': outcome}, 'prepare': {'outputs': {'verifier-id': '7'}}}
    case['needs'] = {'gate': {'outputs': {'head-sha': 'pinned-head', 'pr-number': '42'}}}
    case['data']['comments'] = [{
        'body': f'<!-- codeowner-signoff-verify sha=pinned-head -->\n<!-- verification-run={run} -->\n**Verdict: PASS**',
        'user': {'id': author}, 'html_url': 'https://example.test/verdict',
    }]
    step = workflow('codeowner-signoff-verify')['jobs']['verify']['steps'][-1]
    result = run_scripts([step], case)
    assert result['failures'] == []
    [status] = result['writes']
    assert status['state'] == state
    assert status['sha'] == 'pinned-head'
    assert status['context'] == 'codeowner-signoff-verify'
    assert status['target_url'] == ('https://example.test/verdict' if state == 'success'
                                    else 'https://github.com/example/repo/actions/runs/99')
    if outcome == 'cancelled':
        assert status['description'] == 'Verification cancelled; rerun required'


@pytest.mark.parametrize('event,collection,fragment,kind,path', [
    ('issue_comment', 'comments', 'issuecomment', 'conversation comment', 'issues/comments/11'),
    ('pull_request_review', 'reviews', 'pullrequestreview', 'review summary', 'pulls/42/reviews/11'),
    ('pull_request_review_comment', 'inlineComments', 'discussion_r', 'inline review comment', 'pulls/comments/11'),
])
@pytest.mark.parametrize('manual', [False, True])
def test_explicit_signoff_requests_can_repeat_verification(event, collection, fragment, kind, path, manual):
    case = signoff_case('workflow_dispatch' if manual else event)
    case['data'][collection] = [signoff()]
    case['data']['statuses'] = [{'context': 'codeowner-signoff-verify', 'state': 'success',
                                'target_url': 'https://github.com/example/repo/pull/42#issuecomment-123',
                                'creator': {'login': 'github-actions[bot]'}}]
    if manual:
        separator = '' if fragment == 'discussion_r' else '-'
        case['context']['payload']['inputs'] = {'comment_url': f'https://github.com/example/repo/pull/42#{fragment}{separator}11'}
        case['data']['pull'].update(state='closed', draft=True)
    else:
        case['context']['payload']['review' if collection == 'reviews' else 'comment'] = signoff()
    result = run_workflow('codeowner-signoff-verify', case)
    assert result['failures'] == []
    assert result['outputs']['resolve'] == {
        'proceed': 'true', 'pr-number': '42', 'head-sha': 'resolved-head',
        'signoff-author': 'reviewer', 'signoff-kind': kind,
        'signoff-fetch-cmd': f'gh api repos/example/repo/{path} --jq .body',
    }


@pytest.mark.parametrize('api_exit', [0, 1])
def test_verification_starts_with_a_pending_status_on_the_pinned_head(tmp_path, api_exit):
    import os

    case = signoff_case()
    case.update(renderShell=True, secrets={'CLAUDE_PAT': 'verifier-token'})
    case['needs'] = {'gate': {'outputs': {'head-sha': 'pinned-head'}}}
    step = workflow('codeowner-signoff-verify')['jobs']['verify']['steps'][0]
    [prepared] = run_scripts([step], case)['shell']
    gh = tmp_path / 'gh'
    gh.write_text('#!/bin/sh\nprintf "%s\\n" "$GH_TOKEN" "$@" > "$RECORD"\nexit "$API_EXIT"\n')
    gh.chmod(0o755)
    record = tmp_path / 'request'
    result = subprocess.run(['bash', '-euo', 'pipefail', '-c', prepared['script']],
                            cwd=tmp_path, capture_output=True, text=True, timeout=10,
                            env={**os.environ, **prepared['env'], 'PATH': f'{tmp_path}:{os.environ["PATH"]}',
                                 'API_EXIT': str(api_exit), 'RECORD': str(record), 'GITHUB_REPOSITORY': 'example/repo',
                                 'GITHUB_SERVER_URL': 'https://github.com', 'GITHUB_RUN_ID': '99'})
    assert result.returncode == api_exit, result.stderr
    token, *args = record.read_text().splitlines()
    assert token == 'workflow-token'
    assert args[:2] == ['api', 'repos/example/repo/statuses/pinned-head']
    assert args[args.index('--method') + 1] == 'POST'
    assert dict(arg.split('=', 1) for arg in args if '=' in arg) == {
        'state': 'pending', 'context': 'codeowner-signoff-verify',
        'description': 'Verifying CODEOWNER sign-off',
        'target_url': 'https://github.com/example/repo/actions/runs/99',
    }


@pytest.mark.parametrize('api_exit', [0, 1])
def test_verification_preparation_uses_the_verifier_credential_and_stops_on_lookup_failure(tmp_path, api_exit):
    import os

    step = next(step for step in workflow('codeowner-signoff-verify')['jobs']['verify']['steps']
                if step.get('id') == 'prepare')
    case = signoff_case()
    case.update(renderShell=True, secrets={'CLAUDE_PAT': 'verifier-token'})
    case['context']['repository'] = 'example/repo'
    case['needs'] = {'gate': {'outputs': {'head-sha': 'pinned-head', 'pr-number': '42'}}}
    [prepared] = run_scripts([step], case)['shell']
    tools_dir = tmp_path / 'bin'
    tools_dir.mkdir()
    gh = tools_dir / 'gh'
    gh.write_text('#!/bin/sh\n[ "$GH_TOKEN" = "verifier-token" ] || exit 2\n'
                  '[ "$*" = "api user --jq .id" ] || exit 2\nprintf "7\\n"\nexit "$API_EXIT"\n')
    gh.chmod(0o755)
    (tmp_path / '.github').mkdir()
    (tmp_path / '.github/codeowner-signoff-verify-prompt.md').write_text('PR #${PR_NUMBER} at ${HEAD_SHA}')
    prompt = tmp_path / 'prompt.md'
    result = subprocess.run(['bash', '-euo', 'pipefail', '-c',
                             prepared['script'].replace('/tmp/codeowner-signoff-verify-prompt.md', str(prompt))],
                            cwd=tmp_path, capture_output=True, text=True, timeout=10,
                            env={**os.environ, **prepared['env'], 'PATH': f'{tools_dir}:{os.environ["PATH"]}',
                                 'API_EXIT': str(api_exit), 'GITHUB_OUTPUT': str(tmp_path / 'outputs')})
    assert result.returncode == api_exit, result.stderr
    if api_exit:
        assert not (tmp_path / 'outputs').exists()
        assert not prompt.exists()
    else:
        assert (tmp_path / 'outputs').read_text() == 'verifier-id=7\n'
        assert prompt.read_text() == 'PR #42 at pinned-head'
