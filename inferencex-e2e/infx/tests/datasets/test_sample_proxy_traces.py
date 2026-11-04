from __future__ import annotations

import argparse
import sys
import types
from datetime import UTC, datetime

import pytest

if "psycopg" not in sys.modules:
    # The sampler's DB driver is not a test dependency; nothing here opens a connection.
    _psycopg = types.ModuleType("psycopg")
    _psycopg.rows = types.ModuleType("psycopg.rows")
    _psycopg.rows.dict_row = object()
    sys.modules["psycopg"] = _psycopg
    sys.modules["psycopg.rows"] = _psycopg.rows

from infx.datasets.sample_proxy_traces import (  # noqa: E402
    _redact_url,
    _sort_and_limit,
    parse_args,
)

TIED_TS = datetime(2026, 5, 1, tzinfo=UTC)
OLDER_TS = datetime(2026, 4, 1, tzinfo=UTC)


@pytest.mark.parametrize("sampling", ["top", "recent"])
def test_limit_picks_the_same_sessions_whatever_order_the_db_returns(sampling: str) -> None:
    rows = [
        {"session_id": "sess-b", "req_count": 10, "last_ts": TIED_TS},
        {"session_id": "sess-a", "req_count": 10, "last_ts": TIED_TS},
        {"session_id": "sess-c", "req_count": 5, "last_ts": OLDER_TS},
    ]
    args = argparse.Namespace(sampling=sampling, limit=1, seed=None)

    picked = [_sort_and_limit(order, args)[0]["session_id"] for order in (rows, rows[::-1])]

    assert picked == ["sess-a", "sess-a"]


def test_zero_limit_is_rejected(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["sample", "--out", "out", "--sampling", "top", "--limit", "0"]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exit_info:
        parse_args()

    assert exit_info.value.code == 2
    assert "--limit must be >= 1" in capsys.readouterr().err


def test_keyword_conninfo_password_is_not_logged() -> None:
    redacted = _redact_url("host=db.example dbname=proxy user=reader password=hunter2-secret")

    assert "hunter2-secret" not in redacted
