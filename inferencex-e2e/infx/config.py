"""Default repository paths used by sweep planning."""

import os
import subprocess
from pathlib import Path

MASTER_CONFIGS = ["configs/amd-master.yaml", "configs/nvidia-master.yaml"]
RUNNER_CONFIG = "configs/runners.yaml"
# The matrix generator module. Revisions that predate it shipped a script, which
# historical append-only planning still runs from their own snapshot.
GENERATOR_MODULE = "infx.matrix.generate"
GENERATOR_MODULE_PATH = "infx/matrix/generate.py"
LEGACY_GENERATOR_SCRIPT = "utils/matrix_logic/generate_sweep_configs.py"


def repository_root() -> Path:
    if root := os.environ.get("INFERENCEX_REPOSITORY_ROOT"):
        return project_root(Path(root))
    source_root = Path(__file__).resolve().parent.parent
    if (source_root / "configs").is_dir():
        return source_root
    return project_root(Path.cwd())


def project_root(checkout: Path) -> Path:
    """Locate the end-to-end project in a current or historical checkout."""
    nested = checkout / "inferencex-e2e"
    return nested if (nested / RUNNER_CONFIG).is_file() else checkout


def git_repository_root(cwd: Path | None = None) -> Path:
    """Return the root of the working Git checkout, independently of installed code."""
    return Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=cwd, text=True).strip()
    ).resolve()


def git_path_at_ref(ref: str, path: str, *, cwd: Path | None = None) -> str:
    """Resolve a project file across the repository-layout migration."""
    requested = Path(path)
    if requested.is_absolute() or ".." in requested.parts:
        requested = requested if requested.is_absolute() else (cwd or Path.cwd()) / requested
        path = requested.resolve().relative_to(git_repository_root(cwd)).as_posix()
    else:
        path = requested.as_posix()
    relative = path.removeprefix("inferencex-e2e/")
    for candidate in (f"inferencex-e2e/{relative}", relative):
        result = subprocess.run(
            ["git", "cat-file", "-e", f"{ref}:{candidate}"],
            cwd=cwd,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            return candidate
    raise ValueError(f"Could not find {path!r} at {ref!r}")
