import shutil

import pytest
from hypothesis import given, strategies as st

from changelog_gate_tests.test_workflow_authorization import (
    OPERATIONS, run_scripts, run_workflow, scenario, signoff, signoff_case, workflow,
)


pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="Workflow JavaScript needs a local Node runtime")
ROLES = st.sampled_from(["admin", "maintain", "write", "read", "triage", "custom", "", None, True, 1, [], {}])
PERMISSIONS = st.one_of(
    st.fixed_dictionaries({"permission": ROLES, "role_name": ROLES}),
    st.fixed_dictionaries({}, optional={"permission": ROLES, "role_name": ROLES}),
    st.sampled_from([None, [], "admin", True]),
)


@given(collection=st.sampled_from(['comments', 'reviews', 'inlineComments']),
       preceding=st.integers(0, 220), closed=st.booleans(), draft=st.booleans(),
       completed=st.booleans(), interrupted=st.booleans(), withdrawn=st.booleans(), outsider=st.booleans())
def test_signoff_catchup_pins_current_head_and_deduplicates(collection, preceding, closed, draft, completed, interrupted, withdrawn, outsider):
    case = signoff_case()
    case['data']['pull'].update(state='closed' if closed else 'open', draft=draft)
    case['data'][collection] = [signoff(identifier, body='ordinary comment') for identifier in range(preceding)]
    case['data'][collection].append(signoff(1000, body='withdrawn' if withdrawn else 'As a PR reviewer and CODEOWNER'))
    if outsider:
        case['permissionsByUser'] = {'outsider': {'permission': 'read', 'role_name': 'read'}}
        case['data'][collection].append(signoff(1001, '2026-01-02T00:00:00Z', user={'login': 'outsider', 'type': 'User'}))
    if completed:
        case['data']['statuses'] = [{'context': 'codeowner-signoff-verify', 'state': 'success',
                                    'target_url': 'https://github.com/example/repo/pull/42#issuecomment-123',
                                    'creator': {'login': 'github-actions[bot]'}}]
    if interrupted:
        case['data']['statuses'].insert(0, {'context': 'codeowner-signoff-verify', 'state': 'failure',
                                           'target_url': 'https://github.com/example/repo/actions/runs/98',
                                           'creator': {'login': 'github-actions[bot]'}})
    expected = not any([closed, draft, completed and not interrupted, withdrawn])
    result = run_workflow('codeowner-signoff-verify', case)
    assert result['failures'] == []
    assert result['outputs']['resolve']['proceed'] == str(expected).lower()
    if expected:
        assert result['outputs']['resolve']['head-sha'] == 'resolved-head'
        assert '/1000 --jq .body' in result['outputs']['resolve']['signoff-fetch-cmd']
        case['data']['statuses'] = [{'context': 'codeowner-signoff-verify', 'state': 'success',
                                    'target_url': 'https://github.com/example/repo/pull/42#issuecomment-123',
                                    'creator': {'login': 'github-actions[bot]'}}]
        repeated = run_workflow('codeowner-signoff-verify', case)
        assert repeated['outputs']['resolve']['proceed'] == 'false'


@given(event=st.sampled_from(['issue_comment', 'pull_request_review', 'pull_request_review_comment',
                            'pull_request_target', 'workflow_dispatch']),
       permission=PERMISSIONS, bot=st.booleans(), unavailable=st.booleans())
def test_signoff_gate_rejects_unauthorized_requesters(event, permission, bot, unavailable):
    case = signoff_case(event)
    case.update(permission=permission, permissionError=unavailable)
    case['context']['actor'] = 'automation[bot]' if bot else 'requester'
    case['context']['payload'].update(comment=signoff(), review=signoff(), inputs={
        'comment_url': 'https://github.com/example/repo/pull/42#pullrequestreview-11'})
    case['data']['reviews'] = [signoff()]
    result = run_workflow('codeowner-signoff-verify', case)
    allowed = (not bot and not unavailable and isinstance(permission, dict)
               and permission.get('permission') in ['admin', 'write']
               and permission.get('role_name') in ['admin', 'maintain', 'write'])
    assert (result['outputs'].get('resolve', {}).get('proceed') == 'true') is allowed
    assert result['writes'] == []


@given(trusted=st.booleans(), identity_known=st.booleans(), current_head=st.booleans(),
       current_run=st.booleans(), current_attempt=st.booleans(), passed=st.booleans(),
       outcome=st.sampled_from(['success', 'failure', 'cancelled', 'skipped']), preceding=st.integers(0, 220))
def test_signoff_verdict_cannot_reuse_other_authors_heads_or_attempts(
    trusted, identity_known, current_head, current_run, current_attempt, passed, outcome, preceding,
):
    case = signoff_case()
    case['context'].update(run_id=99, run_attempt=2)
    case['stepsState'] = {'verify': {'outcome': outcome},
                          'prepare': {'outputs': {'verifier-id': '7' if identity_known else ''}}}
    case['needs'] = {'gate': {'outputs': {'head-sha': 'pinned-head', 'pr-number': '42'}}}
    head = 'pinned-head' if current_head else 'previous-head'
    run = 99 if current_run else 98
    attempt = 2 if current_attempt else 1
    verdict = '## ✅✅✅ **Verdict: PASS** ✅✅✅' if passed else '## ❌❌❌ **REJECTED** ❌❌❌'
    case['data']['comments'] = [{'body': 'ordinary comment'} for _ in range(preceding)] + [{
        'body': f'<!-- codeowner-signoff-verify sha={head} -->\n{verdict}\n<!-- verification-run={run} attempt={attempt} -->',
        'user': {'id': 7 if trusted else 8}, 'html_url': 'https://github.com/example/repo/pull/42#issuecomment-123',
    }]
    publisher = workflow('codeowner-signoff-verify')['jobs']['verify']['steps'][-1]
    result = run_scripts([publisher], case)
    assert result['failures'] == []
    [status] = result['writes']
    accepted = all([trusted, identity_known, current_head, current_run, current_attempt, passed, outcome == 'success'])
    assert status['state'] == ('success' if accepted else 'failure')
    assert status['sha'] == 'pinned-head'


@pytest.mark.parametrize("operation", OPERATIONS)
@given(permission=PERMISSIONS, unavailable=st.booleans())
def test_permissions_gate_all_protected_workflow_effects(operation, permission, unavailable):
    case = scenario(operation)
    case.update(permission=permission, permissionError=unavailable)
    result = run_workflow(operation, case)
    expected = (not unavailable and isinstance(permission, dict)
                and permission.get("permission") in ["admin", "maintain", "write"]
                and permission.get("role_name") in ["admin", "maintain", "write"])
    dispatches = [write for write in result["writes"] if write["method"] != "issues.createComment"]
    assert bool(dispatches) is expected
    assert bool(result["failures"]) is not expected
    if not expected:
        assert not result["outputs"]


@given(closed=st.booleans(), draft=st.booleans(), advanced=st.booleans(), same_repo=st.booleans(),
       mergeable=st.booleans(), conflicting=st.booleans(), label_removed=st.booleans(),
       primary=st.sampled_from(["sweep-enabled", "full-sweep-enabled", "full-sweep-fail-fast",
                               "non-canary-full-sweep-enabled", "full-sweep-fail-fast-no-canary"]),
       modifiers=st.sets(st.sampled_from(["all-evals", "evals-only", "agentx-fast"])))
def test_external_sweep_requires_current_approval(closed, draft, advanced, same_repo, mergeable, conflicting, label_removed, primary, modifiers):
    case = scenario("trusted-external-sweep")
    pull = case["data"]["pull"]
    pull.update(state="closed" if closed else "open", draft=draft,
                merge_commit_sha="approved-merge" if mergeable else None)
    pull["head"].update(sha="new-head" if advanced else "approved-head")
    pull["head"]["repo"]["full_name"] = "example/repo" if same_repo else "outside/repo"
    labels = ([] if label_removed else [primary]) + sorted(modifiers)
    if conflicting:
        labels += ["sweep-enabled" if primary != "sweep-enabled" else "full-sweep-enabled"]
    pull["labels"] = [{"name": label} for label in labels]
    case["context"]["payload"]["label"]["name"] = primary
    result = run_workflow("trusted-external-sweep", case)
    expected = mergeable and not any([closed, draft, advanced, same_repo, conflicting, label_removed])
    dispatches = [write for write in result["writes"] if write["method"] == "actions.createWorkflowDispatch"]
    assert bool(dispatches) is expected
    assert bool(result["failures"]) is not expected
    if expected:
        [dispatch] = dispatches
        assert dispatch["ref"] == "main"
        inputs = dispatch["inputs"]
        assert inputs["ref"] == "approved-merge"
        assert inputs["changelog-base-ref"] == "base-head"
        assert inputs["changelog-head-ref"] == "approved-head"
        assert inputs["trim-conc"] == str(primary == "sweep-enabled").lower()
        assert inputs["fail-fast"] == str(primary in {"full-sweep-fail-fast", "full-sweep-fail-fast-no-canary"}).lower()
        for modifier in ("all-evals", "evals-only", "agentx-fast"):
            assert inputs[modifier] == str(modifier in modifiers).lower()


@given(pinned=st.booleans(), historical=st.booleans(), associated=st.booleans(), current_label=st.booleans(),
       removed_before_run=st.booleans(), expired=st.booleans(), metadata=st.booleans(),
       complete=st.booleans(), conclusion=st.sampled_from(["success", "failure", "cancelled", "skipped"]),
       artifact=st.sampled_from(["results_bmk", "eval_results_all", "bmk_agentic_fixture", "logs"]))
def test_staging_requires_label_history_and_usable_source(pinned, historical, associated, current_label, removed_before_run, expired, metadata, complete, conclusion, artifact):
    case = scenario("stage-results")
    if pinned:
        case["context"]["payload"]["comment"]["body"] += " 101"
    data = case["data"]
    data["runs"][0].update(head_sha="historical" if historical else "approved-head",
                           pull_requests=[{"number": 42 if associated else 43}], conclusion=conclusion,
                           status="completed" if complete else "in_progress")
    if not current_label:
        data["pull"]["labels"] = []
    data["timeline"].append({"event": "unlabeled", "label": {"name": "full-sweep-fail-fast"},
                             "created_at": "2026-01-02T09:00:00Z" if removed_before_run else "2026-01-02T11:00:00Z"})
    data["artifacts"]["101"] = ([{"name": "changelog-metadata"}] if metadata else []) + [{"name": artifact, "expired": expired}]
    expected = (current_label and not removed_before_run and (not historical or pinned and associated)
                and complete and conclusion != "skipped" and metadata and not expired and artifact != "logs")
    result = run_workflow("stage-results", case)
    dispatches = [write for write in result["writes"] if write["method"] == "repos.createDispatchEvent"]
    assert bool(dispatches) is expected
    assert bool(result["failures"]) is not expected
    if expected:
        assert result["outputs"]["request"]["run-id"] == "101"
        assert dispatches[0]["client_payload"]["run-attempt"] == "2"
