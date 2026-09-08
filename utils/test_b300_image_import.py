"""Exercise the launcher's local import, cache publication, and failure paths."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def write_executable(path: Path, source: str, *, python: bool = False) -> None:
    path.write_text(f"#!{sys.executable if python else '/bin/bash'}\n" + source)
    path.chmod(0o755)


@pytest.fixture
def importer(tmp_path: Path) -> tuple[Path, dict[str, str], Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    squash_dir = tmp_path / "squash"
    squash_dir.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    # macOS has no util-linux flock or GNU timeout. Keep real advisory locking
    # through fcntl; the cluster smoke uses the actual Linux commands.
    write_executable(
        bin_dir / "flock",
        "import fcntl, sys\nfcntl.flock(int(sys.argv[-1]), fcntl.LOCK_EX)\n",
        python=True,
    )
    write_executable(bin_dir / "timeout", 'shift 2\nexec "$@"\n')
    write_executable(bin_dir / "srun", 'echo "unexpected Slurm allocation" >&2\nexit 99\n')
    write_executable(
        bin_dir / "unsquashfs",
        "import pathlib, sys\np = pathlib.Path(sys.argv[2])\n"
        "sys.exit(0 if p.exists() and p.read_bytes() == b'valid' else 1)\n",
        python=True,
    )
    write_executable(
        bin_dir / "enroot",
        """import os, pathlib, sys, time
active = pathlib.Path(os.environ['IMPORT_LOG'] + '.active')
active.mkdir()  # Concurrent cold imports must not overlap on the login host.
try:
    with open(os.environ['IMPORT_LOG'], 'a') as log:
        log.write(os.environ['ENROOT_TEMP_PATH'] + '\\n')
    p = pathlib.Path(sys.argv[3])
    p.write_bytes(b'partial')
    time.sleep(0.2)
    mode = os.environ.get('IMPORT_MODE', 'success')
    if mode == 'failure':
        sys.exit(42)
    if mode != 'invalid':
        p.write_bytes(b'valid')
finally:
    active.rmdir()
""",
        python=True,
    )
    source = (ROOT / "runners/launch_b300-dsxe.sh").read_text()
    start = source.index("import_squash_image() {")
    end = source.index('\nif [[ "$IS_MULTINODE"', start)
    harness = tmp_path / "import.sh"
    harness.write_text(
        source[start:end] + '\nimport_squash_image example/test:tag "$1" || exit 1\n'
    )
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "SQUASH_DIR": str(squash_dir),
        "ENROOT_TEMP_PATH": str(scratch),
        "IMPORT_LOG": str(tmp_path / "imports.log"),
    }
    return harness, env, squash_dir / "image.sqsh"


def run_import(importer: tuple[Path, dict[str, str], Path]) -> subprocess.CompletedProcess:
    harness, env, squash = importer
    return subprocess.run(
        ["bash", str(harness), str(squash)], env=env,
        capture_output=True, text=True, timeout=10,
    )


def test_warm_image_needs_no_import(importer) -> None:
    _, env, squash = importer
    squash.write_bytes(b"valid")
    assert run_import(importer).returncode == 0
    assert not Path(env["IMPORT_LOG"]).exists()


@pytest.mark.parametrize("distinct_images", [False, True])
def test_concurrent_callers_serialize_and_clean_local_scratch(importer, distinct_images: bool) -> None:
    harness, env, squash = importer
    targets = [squash, squash.with_name("second.sqsh") if distinct_images else squash]
    processes = [
        subprocess.Popen(
            ["bash", str(harness), str(target)], env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        for target in targets
    ]
    for process in processes:
        _, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, stderr
    assert all(target.read_bytes() == b"valid" for target in targets)
    imports = Path(env["IMPORT_LOG"]).read_text().splitlines()
    assert len(imports) == (2 if distinct_images else 1)
    for path in imports:
        assert Path(path).parent == Path(env["ENROOT_TEMP_PATH"])
        assert not Path(path).exists()
    assert not list(squash.parent.glob("*.tmp.*"))


@pytest.mark.parametrize("mode", ["failure", "invalid"])
def test_failed_import_never_publishes_partial_image(importer, mode: str) -> None:
    _, env, squash = importer
    env["IMPORT_MODE"] = mode
    squash.write_bytes(b"previous-invalid-image")
    result = run_import(importer)
    assert result.returncode != 0
    assert squash.read_bytes() == b"previous-invalid-image"
    assert not list(squash.parent.glob("*.tmp.*"))
    assert not list(Path(env["ENROOT_TEMP_PATH"]).iterdir())
