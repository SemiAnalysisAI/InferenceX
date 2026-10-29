import pytest

from infx.klaud import __main__ as klaud
from infx.klaud import github


@pytest.fixture(autouse=True)
def unknown_api_budget(monkeypatch):
    """Keep tests off the real ``gh api rate_limit``; an unreadable budget never defers work."""
    monkeypatch.setattr(github, "api_budget", lambda: None)


@pytest.fixture(autouse=True)
def no_running_sessions(monkeypatch):
    """Keep ``select`` off the real planner run listing; tests that count sessions override it."""
    monkeypatch.setattr(klaud, "running_sessions", lambda _repository: 0)
