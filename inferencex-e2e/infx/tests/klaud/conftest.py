import pytest

from infx.klaud import github


@pytest.fixture(autouse=True)
def unknown_api_budget(monkeypatch):
    """Keep tests off the real ``gh api rate_limit``; an unreadable budget never defers work."""
    monkeypatch.setattr(github, "api_budget", lambda: None)
