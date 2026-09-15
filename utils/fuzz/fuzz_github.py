import pytest
from hypothesis import given, strategies as st

from infx import github
from infx.workflows import reuse


@given(pinned=st.one_of(st.none(), st.integers(1, 10**15)), following=st.integers(1, 10**15),
       indent=st.text(alphabet=" \t", max_size=8), newline=st.sampled_from(["\n", "\r\n"]),
       earlier=st.booleans())
def test_reuse_commands_are_line_scoped(pinned, following, indent, newline, earlier):
    command = indent + "/reuse-sweep-run" + (f" {pinned}" if pinned else "") + indent
    body = newline.join([*(["/reuse-sweep-run 99"] if earlier else []), command, str(following), "Review complete."])
    assert reuse.parse_reuse_command(body) == (True, pinned)
    assert reuse.parse_reuse_command("Please use " + command + " later") == (False, None)


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
