import shutil

import pytest
from hypothesis import given, strategies as st

from changelog_gate_tests.test_workflow_authorization import (
    OPERATIONS, run_workflow, scenario, signoff, signoff_case,
)


pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="Workflow JavaScript needs a local Node runtime")
INVALID_ROLE = st.one_of(
    st.none(), st.booleans(), st.integers(), st.lists(st.text(max_size=20), max_size=3),
    st.dictionaries(st.text(max_size=20), st.integers(), max_size=3),
    st.text(max_size=40).map(lambda value: "unknown:" + value),
)
INVALID_PERMISSION = st.one_of(
    INVALID_ROLE,
    st.fixed_dictionaries({"permission": st.just("admin"), "role_name": INVALID_ROLE}),
    st.fixed_dictionaries({"permission": INVALID_ROLE, "role_name": st.just("admin")}),
)


@pytest.mark.parametrize("operation", [*OPERATIONS, "codeowner-signoff-verify"])
@given(permission=INVALID_PERMISSION)
def test_malformed_permissions_cannot_authorize_work(operation, permission):
    case = signoff_case() if operation == "codeowner-signoff-verify" else scenario(operation)
    case["permission"] = permission
    case["data"]["reviews"] = [signoff()]
    result = run_workflow(operation, case)
    assert all(write["method"] == "issues.createComment" for write in result["writes"])
    if operation == "codeowner-signoff-verify":
        assert result["outputs"]["resolve"]["proceed"] == "false"
    else:
        assert result["failures"]
        assert result["outputs"] == {}
