import hashlib
import importlib.metadata

from utils.install_minimaxm3_aiter import (
    WHEELS,
    installed_wheels_match,
    sha256_file,
)


def test_sha256_file(tmp_path):
    path = tmp_path / "wheel.whl"
    path.write_bytes(b"pinned wheel")

    assert sha256_file(path) == hashlib.sha256(b"pinned wheel").hexdigest()


def test_installed_wheels_match(monkeypatch):
    versions = {wheel.distribution: wheel.version for wheel in WHEELS}
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda distribution: versions[distribution],
    )

    assert installed_wheels_match()

    versions["amd-aiter"] = "0.1.13.post1"
    assert not installed_wheels_match()


def test_pinned_wheels_have_sha256_digests():
    assert [wheel.distribution for wheel in WHEELS] == ["flydsl", "amd-aiter"]
    assert all(len(wheel.sha256) == 64 for wheel in WHEELS)
    assert all(int(wheel.sha256, 16) >= 0 for wheel in WHEELS)
