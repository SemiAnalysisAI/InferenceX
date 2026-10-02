"""The isolated AIPerf venv that the ``agentic`` command builds before re-executing."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from infx.bench.agentic.venv import Runtime, bootstrap
from infx.tests.bench.stubs import executable

FAKE_UV = r"""#!/bin/sh
printf '%s\n' "$@" >> "$UV_CALLS"
printf '%s\n' "$UV_CACHE_DIR" >> "$UV_CACHES"
if [ "$1" = venv ]; then
    [ "$FAIL" = venv ] && exit 3
    for target; do :; done
    mkdir -p "$target/bin"
elif [ "$1" = pip ]; then
    [ "$FAIL" = pip ] && exit 2
    [ "$FAIL" = incomplete ] && exit 0
    bin="$(dirname "$4")"
    printf '#!/bin/sh\nexit 0\n' > "$bin/aiperf"
    printf '#!/bin/sh\nexit 0\n' > "$bin/hf"
    chmod +x "$bin/aiperf" "$bin/hf"
fi
"""


@pytest.fixture
def tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A PATH with only coreutils plus tripwires for git and apt-get."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for forbidden in ("git", "apt-get"):
        executable(bin_dir / forbidden, f'#!/bin/sh\necho {forbidden} >> "$FORBIDDEN"\nexit 97\n')
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    monkeypatch.setenv("UV_CALLS", str(tmp_path / "uv-calls"))
    monkeypatch.setenv("UV_CACHES", str(tmp_path / "uv-caches"))
    monkeypatch.setenv("FORBIDDEN", str(tmp_path / "forbidden"))
    monkeypatch.delenv("FAIL", raising=False)
    return bin_dir


@pytest.mark.parametrize("uv_on_path", [True, False])
def test_bootstrap_builds_a_pinned_venv_with_editable_aiperf(
    tmp_path: Path, tools: Path, uv_on_path: bool
):
    uv = executable(tmp_path / "uv", FAKE_UV)
    if uv_on_path:
        shutil.copy(uv, tools / "uv")
    else:
        # Astral's installer, piped from curl to sh, drops uv into UV_INSTALL_DIR.
        installer = tmp_path / "install.sh"
        installer.write_text(f'mkdir -p "$UV_INSTALL_DIR" && cp {uv} "$UV_INSTALL_DIR/uv"\n')
        executable(tools / "curl", f"#!/bin/sh\ncat {installer}\n")
    runtime = Runtime(tmp_path / "runtime")
    runtime.venv.mkdir(parents=True)
    (runtime.venv / "stale").write_text("from an earlier point")
    source = tmp_path / "checkout with spaces" / "utils" / "aiperf"

    assert bootstrap(runtime, "3.11", source) == 0

    calls = (tmp_path / "uv-calls").read_text().splitlines()
    venv = tmp_path / "runtime" / "venv"
    assert calls[:4] == ["venv", "--python", "3.11", str(venv)]
    install = ["pip", "install", "--python", str(venv / "bin/python"), "-e", str(source)]
    assert calls[4:10] == install
    assert not (venv / "stale").exists()
    caches = set((tmp_path / "uv-caches").read_text().splitlines())
    assert caches == {str(tmp_path / "runtime" / "uv-cache")}
    assert not (tmp_path / "forbidden").exists()
    assert (tmp_path / "runtime/uv/bin/uv").is_file() == (not uv_on_path)


def test_failed_venv_creation_returns_uv_status_without_installing(tmp_path, tools, monkeypatch):
    executable(tools / "uv", FAKE_UV)
    monkeypatch.setenv("FAIL", "venv")

    assert bootstrap(Runtime(tmp_path / "runtime"), "3.11", tmp_path / "aiperf") == 3

    assert "pip" not in (tmp_path / "uv-calls").read_text().splitlines()


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("pip", "ERROR: benchmark client dependency bootstrap failed"),
        ("incomplete", "ERROR: isolated AIPerf environment is incomplete"),
        # No uv on PATH, and Astral's installer does not produce one.
        ("installer", "ERROR: uv installation did not create"),
    ],
)
def test_unusable_runtime_fails_with_the_reason(
    tmp_path, tools, monkeypatch, capsys, failure, message
):
    if failure == "installer":
        executable(tools / "curl", "#!/bin/sh\nexit 6\n")
    else:
        executable(tools / "uv", FAKE_UV)
        monkeypatch.setenv("FAIL", failure)

    assert bootstrap(Runtime(tmp_path / "runtime"), "3.11", tmp_path / "aiperf") == 1

    assert message in capsys.readouterr().err
