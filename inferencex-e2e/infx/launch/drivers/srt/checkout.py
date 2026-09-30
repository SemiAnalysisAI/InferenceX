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

from infx.config import repository_root
from infx.launch import proc
from infx.launch.context import LaunchError
from infx.launch.drivers.srt.recipe import RECIPES_MIRROR
from infx.launch.drivers.srt.run import SrtRun, require

SUBMODULE = Path("utils/srt-slurm")
PATCHES = Path("runners/srt-slurm/patches")
SETUP_LOG = "srt-setup.log"
SETUP_ATTEMPTS = 5
UV_INSTALLER = "https://astral.sh/uv/install.sh"


@dataclass(frozen=True)
class SrtFork:
    """A framework's srt-slurm fork, checked out instead of the pinned submodule.

    Forks get no InferenceX patches, predate ``--json``, ``--no-preflight`` and
    ``benchmark.stream_output``, and report their job only in prose. Their jobs keep
    srtctl's own health-check default, and may run recipes the workspace mirror lacks.
    """

    url: str
    commit: str


SRT_FORKS: dict[str, SrtFork] = {
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

    @property
    def venv(self) -> Path:
        return self.root / ".venv"


def _git(*args: str | Path, capture: bool = False) -> str:
    try:
        result = proc.run(["git", *args], check=True, capture=capture)
    except (OSError, subprocess.CalledProcessError) as error:
        raise LaunchError(f"git {args[0]} failed: {error}") from error
    return result.stdout.strip() if capture else ""


def _checked(result: subprocess.CompletedProcess[str], action: str) -> None:
    if result.returncode:
        raise LaunchError(f"{action} failed (exit {result.returncode})")


def checkout_dir(run: SrtRun, *, shared: bool) -> Path:
    """Where a multi-node checkout lives, named for its run: shared-run-root or the workspace."""
    request = run.request
    if shared:
        root = run.srt.shared_run_root
        if root is None:
            raise LaunchError(f"cluster {run.cluster.id!r} has no srt-slurm.shared-run-root")
        run_id = request.env.get("GITHUB_RUN_ID", "")
        attempt = request.env.get("GITHUB_RUN_ATTEMPT", "")
        return root / f"srt-slurm-{run_id}-{attempt}-{request.runner_name}-{os.getpid()}"
    require(request, "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT")
    digest = hashlib.sha1(request.result_filename.encode(), usedforsecurity=False)
    run_id, attempt = request.env["GITHUB_RUN_ID"], request.env["GITHUB_RUN_ATTEMPT"]
    return run.workspace / f"srt-slurm-{run_id}-{attempt}-{digest.hexdigest()[:12]}"


def prepare_checkout(run: SrtRun, destination: Path, *, power: bool) -> Checkout:
    """Check out srt-slurm at ``destination``, replacing an earlier run's, and stage the recipes.

    Records ``srt-slurm-sha.txt``, and for power lanes ``power-producer-sha.txt``.
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
    (destination / RECIPES_MIRROR).symlink_to("../../recipes")
    shutil.copytree(recipes / "configs", destination / "configs", symlinks=True, dirs_exist_ok=True)
    return Checkout(destination, commit, fork is not None)


def _uv(run: SrtRun) -> str:
    """The uv on PATH, else a fresh install."""
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


def install_srtctl(run: SrtRun, checkout: Checkout, *, python: str | None = None) -> None:
    """Install the checkout into its venv and put the venv on PATH."""
    uv = _uv(run)
    root = run.srt.uv_cache_root
    if root is not None:
        cache = root / f"cache-{run.request.runner_name}"
        cache.mkdir(parents=True, exist_ok=True)
        run.env["UV_CACHE_DIR"] = str(cache)
        run.env["UV_PYTHON_INSTALL_DIR"] = str(root / "python")
    venv = checkout.venv
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


def _run_logged(argv: list[str], log: Path, *, env: dict[str, str], cwd: Path) -> int:
    proc.echo(argv, env)
    with log.open("a") as handle:
        return subprocess.run(
            argv, env=env, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, check=False
        ).returncode


def _discard_corrupt_archives(configs: Path) -> bool:
    """Delete NATS/etcd downloads that fail an integrity check; return whether any did."""
    discarded = False
    checks = (
        ("nats-server-v*.deb", "NATS", ["dpkg-deb", "--contents"]),
        ("etcd-*.tar.gz", "etcd", ["tar", "-tzf"]),
    )
    for pattern, name, check in checks:
        for archive in sorted(configs.glob(pattern)):
            try:
                intact = proc.run([*check, archive], capture=True).returncode == 0
            except OSError:
                intact = False
            if not intact:
                print(f"Removing incomplete {name} archive: {archive}", file=sys.stderr)
                archive.unlink()
                discarded = True
    return discarded


def run_setup(run: SrtRun, checkout: Checkout) -> int:
    """Run ``make setup`` in the checkout, logging to SETUP_LOG; return its exit code."""
    log = run.workspace / SETUP_LOG
    for attempt in range(1, SETUP_ATTEMPTS + 1):
        print(f"Setting up srt-slurm, attempt {attempt} (details: {SETUP_LOG})", flush=True)
        argv = ["make", "setup", f"ARCH={run.cluster.arch}"]
        rc = _run_logged(argv, log, env=run.env, cwd=checkout.root)
        if rc == 0:
            print("srt-slurm setup complete", flush=True)
            return 0
        sys.stderr.write(log.read_text(errors="replace"))
        if not _discard_corrupt_archives(checkout.root / "configs"):
            return rc
        if attempt < SETUP_ATTEMPTS:
            time.sleep(attempt * 5)
    print(f"ERROR: srt-slurm setup failed after {SETUP_ATTEMPTS} attempts", file=sys.stderr)
    return 1


def compute_workspace(run: SrtRun, checkout: Checkout, *, shared: bool) -> Path:
    """INFMAX_WORKSPACE: the runner checkout as the job's compute nodes see it.

    Compute nodes may not see a shared-run-root lane's runner workspace; the backend then
    stages a copy beside the checkout.
    """
    if not shared:
        return run.workspace
    name = checkout.root.name.replace("srt-slurm-", "infmax-workspace-", 1)
    exclude = (".git/", "srt-slurm*/", "outputs/", "LOGS/", "*.sqsh")
    return run.backend.stage_workspace(run.workspace, checkout.root.parent / name, exclude=exclude)
