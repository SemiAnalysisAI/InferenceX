import json
from pathlib import Path

import pytest

from infx.evals import minimax_m3_full_eval as full, minimax_provider_eval as smoke


@pytest.fixture
def minimax_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    (source / "verify.py").write_bytes(b"pass\n")
    (source / "sample.jsonl").write_bytes(b"{}\n" * 102)
    monkeypatch.setattr(
        full,
        "REQUIRED_SOURCE_SHA256",
        {
            "verify.py": "9f56e761d79bfdb34304a012586cb04d16b435ef6130091a97702e559260a2f2",
            "sample.jsonl": "28398be8a08be34c60bb6f76d444dbcaa5577a591aa305499638ec779e79afd6",
        },
    )
    return source


@pytest.fixture
def minimax_smoke_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    fixture = tmp_path / "fixture.json"
    fixture.write_text(
        json.dumps(
            {
                "source": smoke.UPSTREAM_SOURCE,
                "ref": smoke.UPSTREAM_REF,
                "indices": [71],
                "license": "Test license\n",
                "rows": [
                    {
                        "messages": [{"role": "user", "content": "Find café hours"}],
                        "tools": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        smoke,
        "EXPECTED_LICENSE_SHA256",
        "c24d5f6da316a4bec6612e644e5fdcc0243fcb3a3ebcec4a2a16389ada6c520c",
    )
    monkeypatch.setattr(
        smoke,
        "EXPECTED_CASE_SHA256",
        {71: "05b119b71e4dcc69cf439da009703721d993c5ec05c502da3c9daed79ec9a48a"},
    )
    return fixture
