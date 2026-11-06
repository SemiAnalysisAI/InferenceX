import pytest

from infx.workflows.pr_checklist import checklist_complete

TEMPLATE = """## Summary

- [ ] Unrelated task outside the checklist

<details>
<summary>PR checklist and change type</summary>

## Type of Change

- [ ] Bug fix

## Checklist

- [{tested}] I have tested my changes locally
- [{changelog}] **If `python -m infx.matrix.changed` lists affected config keys, I have appended one entry**
- [ ] **Before merging via reuse, an authorized maintainer has commented `/use <run_id>`**

</details>

- [ ] Trailing task after the section
"""


@pytest.mark.parametrize(
    ("tested", "changelog", "expected"),
    [
        ("x", "x", True),
        ("X", " ", True),
        (" ", "x", False),
    ],
)
def test_only_conditional_items_may_stay_unchecked(tested, changelog, expected):
    body = TEMPLATE.format(tested=tested, changelog=changelog)

    assert checklist_complete(body) is expected


def test_missing_or_empty_checklist_is_incomplete():
    assert checklist_complete("## Summary\n\n- [x] Done\n") is False
    assert checklist_complete("## Checklist\n\nNothing here.\n\n## Notes\n- [x] Done\n") is False
