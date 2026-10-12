import copy
import json
from urllib.parse import urlencode

import pytest

from infx.workflows import stage_results_callback as callback


@pytest.fixture
def event():
    return {
        "action": "stage-results-completed",
        "client_payload": {
            "source-repository": "SemiAnalysisAI/InferenceX-app",
            "run-id": "123",
            "run-date": "2026-10-12",
            "pr-number": "7",
            "requested-by": "reviewer",
            "app-run-id": "456",
            "comment-id": "81",
            "stage-succeeded": "true",
        },
    }


def chart_url(**params):
    return callback.STAGING_SITE + "/inference?" + urlencode({
        "g_model": "Kimi-K3", "i_seq": "agentic-coding", "i_prec": "fp4",
        "i_gpus": "mi355x-vllm-disagg", "i_dates": "2026-10-12~r123",
        **params,
    })


def capture(monkeypatch):
    calls = []
    monkeypatch.setattr(callback.github, "api", lambda *args, **kwargs: calls.append((args, kwargs)))
    return calls


def test_completion_updates_acknowledgment_with_each_exact_curve(event, monkeypatch, tmp_path):
    first, second = chart_url(), chart_url(g_model="DeepSeek-R1", i_seq="8192x1024", i_gpus="h200-sglang")
    event["client_payload"]["chart-links"] = [
        {"label": "Kimi-K3 · AgentX · MI355X [vLLM]", "url": first},
        {"label": "DeepSeek-R1 · 8k/1k · H200", "url": second},
    ]
    path = tmp_path / "event.json"
    path.write_text(json.dumps(event))
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(path))
    monkeypatch.setenv("GITHUB_REPOSITORY", "example/project")
    monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.com")
    monkeypatch.setenv("GH_TOKEN", "token")
    calls = capture(monkeypatch)

    callback.main()

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == ("example/project", "/issues/comments/81", "token")
    assert kwargs["method"] == "PATCH"
    body = kwargs["data"]["body"]
    assert f"- [Kimi-K3 · AgentX · MI355X \\[vLLM\\]]({first})" in body
    assert f"- [DeepSeek-R1 · 8k/1k · H200]({second})" in body
    assert "[Staging workflow](https://github.com/SemiAnalysisAI/InferenceX-app/actions/runs/456)" in body


def test_older_app_callback_retains_run_qualified_link(event, monkeypatch):
    del event["client_payload"]["comment-id"]
    calls = capture(monkeypatch)
    callback.report("example/project", event, "token", "https://github.com")
    args, kwargs = calls[0]
    assert args[1] == "/issues/7/comments"
    assert kwargs["method"] == "POST"
    assert callback.STAGING_SITE + "/inference?i_dates=2026-10-12~r123" in kwargs["data"]["body"]


@pytest.mark.parametrize("action,succeeded,expected", [
    ("stage-results-started", "", "is now running"),
    ("stage-results-completed", "false", "staging run 123 failed"),
])
def test_progress_and_failure_do_not_publish_curve_links(event, monkeypatch, action, succeeded, expected):
    event["action"] = action
    event["client_payload"].update({"stage-succeeded": succeeded, "chart-links": "not ready"})
    calls = capture(monkeypatch)
    callback.report("example/project", event, "token", "https://github.com")
    body = calls[0][1]["data"]["body"]
    assert expected in body
    assert callback.STAGING_SITE not in body


def test_eval_only_completion_does_not_invent_a_pareto_line(event, monkeypatch):
    event["client_payload"]["chart-links"] = []
    calls = capture(monkeypatch)
    callback.report("example/project", event, "token", "https://github.com")
    body = calls[0][1]["data"]["body"]
    assert "no staged performance curves" in body
    assert "/inference?" not in body


@pytest.mark.parametrize("links", [
    {},
    ["invalid"],
    [{"label": "curve", "url": 123}],
    [{"label": "", "url": chart_url()}],
    [{"label": "curve\nnew paragraph", "url": chart_url()}],
    [{"label": "curve", "url": chart_url().replace(callback.STAGING_SITE, "https://example.com")}],
    [{"label": "curve", "url": chart_url(i_dates="2026-10-12~r999")}],
    [{"label": "curve", "url": chart_url(i_dates="2026-10-12")}],
    [{"label": "curve", "url": chart_url(i_gpus="")}],
    [{"label": "curve", "url": chart_url() + "#other"}],
])
def test_invalid_links_never_edit_the_comment(event, monkeypatch, links):
    event["client_payload"]["chart-links"] = copy.deepcopy(links)
    calls = capture(monkeypatch)
    with pytest.raises(RuntimeError):
        callback.report("example/project", event, "token", "https://github.com")
    assert calls == []


@pytest.mark.parametrize("key,value", [
    ("source-repository", "untrusted/repo"), ("run-id", "x"),
    ("pr-number", "7/8"), ("app-run-id", ""), ("comment-id", "-1"),
    ("run-date", "today"), ("requested-by", "a\n@other"), ("stage-succeeded", "maybe"),
])
def test_invalid_callback_never_posts(event, monkeypatch, key, value):
    event["client_payload"][key] = value
    calls = capture(monkeypatch)
    with pytest.raises(RuntimeError):
        callback.report("example/project", event, "token", "https://github.com")
    assert calls == []
