"""Exercise the content-keyed pyslurm build cache."""

import zipfile
from pathlib import Path

from infx.runners.pyslurm_build import _cached_build


def _fake_wheel(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    wheel = out_dir / "pyslurm-25.11.2-cp312-cp312-linux_x86_64.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("pyslurm/__init__.py", "__version__ = '25.11.2'\n")
    return wheel


def test_builds_once_then_reuses_the_cache(tmp_path: Path) -> None:
    builds = []

    def build(out_dir: Path) -> Path:
        builds.append(out_dir)
        return _fake_wheel(out_dir)

    first = _cached_build(tmp_path, build, "abc123", "25.05-cp312")
    second = _cached_build(tmp_path, build, "abc123", "25.05-cp312")

    assert first == second == tmp_path / "pyslurm-25.05-cp312-abc123"
    assert len(builds) == 1
    assert (first / "pyslurm" / "__init__.py").is_file()
    assert (first / ".complete").read_text().startswith("pyslurm-25.11.2")


def test_new_key_rebuilds_and_leaves_no_staging(tmp_path: Path) -> None:
    _cached_build(tmp_path, _fake_wheel, "one", "25.05-cp312")
    _cached_build(tmp_path, _fake_wheel, "two", "25.05-cp312")

    names = sorted(p.name for p in tmp_path.iterdir() if not p.name.startswith("."))
    assert names == ["pyslurm-25.05-cp312-one", "pyslurm-25.05-cp312-two"]
