"""``benchmarks/bench_python.sh``: the container's python3, else uv's standalone CPython."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from infx.tests.bench.stubs import executable

HELPER = Path(__file__).resolve().parents[3] / "benchmarks" / "bench_python.sh"
UV = """#!/bin/sh
echo "$*" >> "$UV_LOG"
[ "$*" = "python find 3.11" ] && echo /opt/cpython-3.11/bin/python3.11
exit 0
"""
# What the installer pipe runs: it drops the uv stub where UV_INSTALL_DIR names.
CURL = """#!/bin/sh
cat <<'EOF'
cp "$UV_STUB" "$UV_INSTALL_DIR/uv"
EOF
"""


@pytest.fixture
def tools(tmp_path: Path) -> Path:
    """A PATH with the helper's tools and a curl stub, and no python3."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("mktemp", "env", "sh", "cp", "cat"):
        (bin_dir / name).symlink_to(shutil.which(name))
    executable(bin_dir / "curl", CURL)
    executable(tmp_path / "uv", UV)
    (tmp_path / "tmp").mkdir()
    return bin_dir


def bench_python(tools: Path) -> subprocess.CompletedProcess:
    tmp = tools.parent
    return subprocess.run(
        [shutil.which("bash"), "-c", f'source "{HELPER}" && bench_python'],
        env={
            "PATH": str(tools),
            "TMPDIR": str(tmp / "tmp"),
            "UV_LOG": str(tmp / "uv.log"),
            "UV_STUB": str(tmp / "uv"),
        },
        capture_output=True,
        text=True,
        check=False,
    )


def test_container_python3_is_used_as_is(tools: Path) -> None:
    python3 = executable(tools / "python3", "#!/bin/sh\n")

    result = bench_python(tools)

    assert result.stdout == f"{python3}\n"
    assert not (tools.parent / "uv.log").exists()


def test_image_without_python_runs_on_uv_standalone_cpython(tools: Path) -> None:
    result = bench_python(tools)

    assert result.returncode == 0, result.stderr
    assert result.stdout == "/opt/cpython-3.11/bin/python3.11\n"
    assert (tools.parent / "uv.log").read_text().splitlines() == [
        "python install 3.11",
        "python find 3.11",
    ]
