"""Unit coverage for the B300 Mooncake recovery patch."""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def mooncake_patch_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Small package sources exercise the patcher's file-write contract."""
    import hashlib
    from runners import patch_kimik3_mooncake_recovery as recovery

    original = {"first.py": "value = 1\n", "second.py": "value = 2\n"}
    updated = {"first.py": "value = 3\n", "second.py": "value = 4\n"}
    patches = []
    for name, source in original.items():
        (tmp_path / name).write_text(source)
        patches.append((
            name, hashlib.sha256(source.encode()).hexdigest(),
            hashlib.sha256(updated[name].encode()).hexdigest(),
            ((source, updated[name]),),
        ))
    monkeypatch.setattr(recovery, "PATCHES", patches)
    return recovery, tmp_path, original, updated


def test_mooncake_patch_is_complete_and_idempotent(mooncake_patch_fixture):
    recovery, root, _, expected = mooncake_patch_fixture
    assert recovery.patch_mooncake(root) is True
    assert {name: (root / name).read_text() for name in expected} == expected
    assert recovery.patch_mooncake(root) is False
    assert {name: (root / name).read_text() for name in expected} == expected


@pytest.mark.parametrize("state", ["unknown", "partial", "missing"])
def test_mooncake_patch_preflights_all_sources_before_writing(
    mooncake_patch_fixture, state: str,
):
    recovery, root, _, updated = mooncake_patch_fixture
    second = root / "second.py"
    if state == "unknown":
        second.write_text("different upstream revision\n")
    elif state == "partial":
        second.write_text(updated["second.py"])
    else:
        second.unlink()
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    with pytest.raises((RuntimeError, OSError)):
        recovery.patch_mooncake(root)
    assert {path.name: path.read_bytes() for path in root.iterdir()} == before
