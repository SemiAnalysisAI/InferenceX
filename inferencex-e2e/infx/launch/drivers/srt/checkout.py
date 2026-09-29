"""The job-local srt-slurm checkout: clone, patches, recipe staging, srtctl, and ``make setup``."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from infx.config import repository_root
from infx.launch import proc
from infx.launch.context import LaunchError
from infx.launch.drivers.srt.recipe import RECIPES_MIRROR
from infx.launch.drivers.srt.run import SrtRun, require

if TYPE_CHECKING:
    from infx.launch.drivers.srt.lanes import SrtLane

SUBMODULE = Path("utils/srt-slurm")
PATCHES = Path("runners/srt-slurm/patches")
SETUP_LOG = "srt-setup.log"
UV_INSTALLER = "https://astral.sh/uv/install.sh"


@dataclass(frozen=True)
class SrtFork:
    """A framework that runs on a fork of srt-slurm instead of the pinned submodule.

    Fork checkouts get no InferenceX patches and no ``benchmark.stream_output``;
    their job id is read from srtctl's human-readable output, and they are never
    passed ``--no-preflight``, which they predate.
    """

    url: str
    commit: str


SRT_FORKS: dict[str, SrtFork] = {
    # TileRT still needs its legacy runtime until the native backend and router land.
    "tilert": SrtFork(
        "https://github.com/SemiAnalysisAI/srt-slurm.git",
        "6bc3f306bdafa1edfb5dded2fcda8f1ccede1bde",
    ),
}


@dataclass(frozen=True)
class Checkout:
    """A job-local srt-slurm checkout with InferenceX recipes staged."""

    root: Path
    commit: str
    fork: bool


def _git(*args: str | Path, capture: bool = False) -> str:
    """Run git, raising ``LaunchError`` on failure; return stdout when captured."""
    try:
        result = proc.run(["git", *args], check=True, capture=capture)
    except (OSError, subprocess.CalledProcessError) as error:
        raise LaunchError(f"git {args[0]} failed: {error}") from error
    return result.stdout.strip() if capture else ""


def _checked(result: subprocess.CompletedProcess[str], action: str) -> None:
    """Raise ``LaunchError`` when ``result`` failed."""
    if result.returncode:
        raise LaunchError(f"{action} failed (exit {result.returncode})")


def checkout_dir(run: SrtRun, lane: SrtLane, *, shared: bool) -> Path:
    """Where the lane's checkout lives: shared-run-root, per-run, or ``<workspace>/srt-slurm``."""
    request = run.request
    if shared:
        root = run.srt.shared_run_root
        if root is None:
            raise LaunchError(f"cluster {run.cluster.id!r} has no srt-slurm.shared-run-root")
        run_id = request.env.get("GITHUB_RUN_ID", "")
        attempt = request.env.get("GITHUB_RUN_ATTEMPT", "")
        return root / f"srt-slurm-{run_id}-{attempt}-{request.runner_name}-{os.getpid()}"
    if lane.per_run_checkout:
        require(request, "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT")
        # Twelve hex digits of the RESULT_FILENAME's SHA-1 keep concurrent points apart.
        digest = hashlib.sha1(request.result_filename.encode(), usedforsecurity=False)
        run_id, attempt = request.env["GITHUB_RUN_ID"], request.env["GITHUB_RUN_ATTEMPT"]
        return run.workspace / f"srt-slurm-{run_id}-{attempt}-{digest.hexdigest()[:12]}"
    return run.workspace / "srt-slurm"


def prepare_checkout(run: SrtRun, destination: Path, *, power: bool) -> Checkout:
    """Check out srt-slurm at ``destination`` and stage every InferenceX recipe into it.

    A checkout left at ``destination`` by an earlier run is removed first. Clones the
    pinned submodule (``--no-hardlinks`` keeps job writes out of it) and applies
    runners/srt-slurm/patches, or checks out the framework's fork; records
    ``srt-slurm-sha.txt`` (and ``power-producer-sha.txt`` for power lanes).
    """
    if destination.exists():
        print(f"Removing existing {destination}...", flush=True)
        shutil.rmtree(destination)
    fork = SRT_FORKS.get(run.request.framework)
    if fork is not None:
        _git("init", "--quiet", destination)
        _git("-C", destination, "remote", "add", "origin", fork.url)
        _git("-C", destination, "fetch", "--quiet", "--depth=1", "origin", fork.commit)
        _git("-C", destination, "checkout", "--quiet", "--detach", fork.commit)
        commit = fork.commit
    else:
        source = repository_root() / SUBMODULE
        if not (source / ".git").exists():
            raise LaunchError(
                "Missing srt-slurm submodule; run git submodule update --init before launching."
            )
        commit = _git("-C", source, "rev-parse", "HEAD", capture=True)
        _git(
            "-c",
            "advice.detachedHead=false",
            "clone",
            "--quiet",
            "--no-hardlinks",
            source,
            destination,
        )
        # Temporary fixes awaiting upstream merge; see runners/srt-slurm/patches/README.md.
        for patch in sorted((run.workspace / PATCHES).glob("*.patch")):
            _git("-C", destination, "apply", patch)
    head = _git("-C", destination, "rev-parse", "HEAD", capture=True)
    if head != commit:
        raise LaunchError(f"srt-slurm checkout is at {head}, expected {commit}")
    print(f"Using srt-slurm revision {commit}", flush=True)
    sha_file = run.workspace / "srt-slurm-sha.txt"
    sha_file.write_text(f"{head}\n")
    if power:
        shutil.copyfile(sha_file, run.workspace / "power-producer-sha.txt")
    recipes = run.workspace / RECIPES_MIRROR
    (destination / "benchmarks/multi_node").mkdir(parents=True, exist_ok=True)
    shutil.copytree(recipes, destination / "recipes", symlinks=True, dirs_exist_ok=True)
    # Both CONFIG_FILE spellings (recipes/... and the mirror path) occur in master configs.
    (destination / RECIPES_MIRROR).symlink_to("../../recipes")
    shutil.copytree(recipes / "configs", destination / "configs", symlinks=True, dirs_exist_ok=True)
    return Checkout(destination, commit, fork is not None)


def _uv(run: SrtRun) -> str:
    """Return a uv binary, installing it (curl | sh) when none is on PATH."""
    found = shutil.which("uv", path=run.env.get("PATH"))
    if found:
        return found
    root = run.srt.uv_cache_root
    install_dir = root / "bin" if root is not None else Path.home() / ".local/bin"
    install_dir.mkdir(parents=True, exist_ok=True)
    env = {**run.env, "UV_INSTALL_DIR": str(install_dir), "UV_NO_MODIFY_PATH": "1"}
    _checked(proc.run(["sh", "-c", f"curl -LsSf {UV_INSTALLER} | sh"], env=env), "installing uv")
    run.prepend_path(install_dir)
    return str(install_dir / "uv")


def install_srtctl(run: SrtRun, checkout: Checkout, *, python: str | None) -> Path:
    """Create ``<checkout>/.venv``, install the checkout into it, and put it on PATH.

    ``--seed`` installs pip, which srtctl's dynamo wheel prefetch needs.
    """
    uv = _uv(run)
    root = run.srt.uv_cache_root
    if root is not None:
        # One cache per runner: concurrent builds in a shared NFS cache race.
        cache = root / f"cache-{run.request.runner_name}"
        cache.mkdir(parents=True, exist_ok=True)
        run.env["UV_CACHE_DIR"] = str(cache)
        # Where uv installs Pythons, for this build and for the jobs srtctl submits.
        run.env["UV_PYTHON_INSTALL_DIR"] = str(root / "python")
    venv = checkout.root / ".venv"
    argv = [uv, "venv", "--quiet", "--seed", *(["--python", python] if python else []), str(venv)]
    _checked(proc.run(argv, env=run.env, cwd=checkout.root), "uv venv")
    install = [
        uv,
        "pip",
        "install",
        "--quiet",
        "--python",
        str(venv / "bin/python"),
        "-e",
        str(checkout.root),
    ]
    _checked(
        proc.run(install, env={**run.env, "VIRTUAL_ENV": str(venv)}, cwd=checkout.root),
        "installing srtctl",
    )
    run.prepend_path(venv / "bin")
    if shutil.which("srtctl", path=run.env["PATH"]) is None:
        raise LaunchError("Failed to install srtctl")
    return venv


def _run_logged(argv: list[str], log: Path, *, env: dict[str, str], cwd: Path) -> int:
    """Echo ``argv`` and run it with stdout and stderr appended to ``log``."""
    proc.echo(argv, env)
    with log.open("a") as handle:
        return subprocess.run(
            argv, env=env, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, check=False
        ).returncode


def _archive_ok(argv: list[str | Path]) -> bool:
    """Whether an archive integrity check exits 0 (a missing tool counts as a failure)."""
    try:
        return proc.run(argv, capture=True).returncode == 0
    except OSError:
        return False


def _discard_corrupt_archives(configs: Path) -> bool:
    """Delete NATS/etcd downloads that fail an integrity check; return whether any did."""
    discarded = False
    checks = (
        ("nats-server-v*.deb", "NATS", ["dpkg-deb", "--contents"]),
        ("etcd-*.tar.gz", "etcd", ["tar", "-tzf"]),
    )
    for pattern, name, check in checks:
        for archive in sorted(configs.glob(pattern)):
            if not _archive_ok([*check, archive]):
                print(f"Removing incomplete {name} archive: {archive}", file=sys.stderr)
                archive.unlink()
                discarded = True
    return discarded


def run_setup(run: SrtRun, checkout: Checkout, *, attempts: int) -> int:
    """Run ``make setup`` in the checkout; return its exit code.

    Output goes to ``srt-setup.log`` and is printed only on failure. With
    ``attempts`` > 1, a failure is retried only after discarding a corrupt
    NATS/etcd download.
    """
    log = run.workspace / SETUP_LOG
    for attempt in range(1, attempts + 1):
        suffix = f" (attempt {attempt}/{attempts})" if attempts > 1 else ""
        print(f"Setting up srt-slurm{suffix} (details: {SETUP_LOG})", flush=True)
        argv = ["make", "setup", f"ARCH={run.cluster.arch}"]
        rc = _run_logged(argv, log, env=run.env, cwd=checkout.root)
        if rc == 0:
            print("srt-slurm setup complete", flush=True)
            return 0
        sys.stderr.write(log.read_text(errors="replace"))
        if attempts == 1:
            return rc
        if not _discard_corrupt_archives(checkout.root / "configs"):
            print(
                "ERROR: srt-slurm setup failed without an invalid NATS/etcd archive; not retrying",
                file=sys.stderr,
            )
            return 1
        if attempt < attempts:
            time.sleep(attempt * 5)
    print(f"ERROR: srt-slurm setup failed after {attempts} attempts", file=sys.stderr)
    return 1


# Left out of a compute-visible workspace copy: this repository's history, srt-slurm
# checkouts and outputs, staged logs, and squash images.
_WORKSPACE_EXCLUDES = (".git/", "srt-slurm*/", "outputs/", "LOGS/", "*.sqsh")


def compute_workspace(run: SrtRun, checkout: Checkout, *, shared: bool) -> Path:
    """INFMAX_WORKSPACE: the runner checkout as the job's compute nodes see it.

    Lanes whose srt-slurm checkout sits on shared-run-root may run on runners whose
    workspace compute nodes cannot see; the backend then stages a copy beside the
    checkout.
    """
    if not shared:
        return run.workspace
    name = checkout.root.name.replace("srt-slurm-", "infmax-workspace-", 1)
    staging = checkout.root.parent / name
    return run.backend.stage_workspace(run.workspace, staging, exclude=_WORKSPACE_EXCLUDES)
