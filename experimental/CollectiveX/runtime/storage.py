"""Image-cache identity, imports, and isolated compute-visible source staging."""

from __future__ import annotations

from contextlib import nullcontext, suppress
import json
import os
from pathlib import Path
import platform
import pwd
import re
import shutil
import subprocess
import tempfile
import time

from . import probe
from .scheduler import locked, log, log_path, log_tail, run


# Keep private state and generated results out of a fresh execution's source tree.
EXCLUDES = {
    "__pycache__",
    "results",
    ".shards",
    ".collx_workloads",
    ".collx_backend",
    ".collx_sources",
    ".venv",
    ".pytest_cache",
    "private-infra.md",
    "goal.md",
    "notes.md",
}


class ImportFailure(RuntimeError):
    """A node-side importer status that the allocation retry loop must preserve."""

    def __init__(self, message: str, status: int):
        super().__init__(message)
        self.status = status


def select_image(image: str, env: dict[str, str]) -> dict[str, str]:
    """Import by tag, using a digest only to detect a moved tag in its sidecar.

    Enroot cannot reliably import a digest-qualified Docker Hub reference without interaction.
    An unresolved digest reuses the staged squash; the refresh flag remains the manual hatch.
    """
    if not re.fullmatch(r"[A-Za-z0-9._/-]+:[A-Za-z0-9._-]+", image):
        raise ValueError("configured image reference is malformed")
    result = {**env, "COLLECTIVEX_IMAGE": image}
    if not probe.DIGEST_PATTERN.fullmatch(env.get("COLLX_IMAGE_DIGEST", "")):
        digest = probe.resolve_image_digest(image)
        if digest:
            result["COLLX_IMAGE_DIGEST"] = digest
            log(f"image digest {digest}")
        else:
            result.pop("COLLX_IMAGE_DIGEST", None)
            log("image digest unresolved; staged squash reused as-is")
    return result


def squash_path(directory: Path, image: str, image_platform: str) -> Path:
    """One cache file per platform/tag, never per run or per transient digest lookup."""
    suffix = {"linux/amd64": "", "linux/arm64": "_linux_arm64"}[image_platform]
    return directory / f"{suffix}_{re.sub(r'[/:@#]', '_', image)}.sqsh"


def squash_verdict(path: Path, digest: str, refresh_epoch: int | None) -> str:
    """Order absence, refresh, digest change, and reuse while the caller holds the lock."""
    if not path.exists():
        return "absent"
    if refresh_epoch is not None:
        try:
            mtime = int(path.stat().st_mtime)
        except OSError:
            mtime = 0
        if mtime < refresh_epoch:
            return "refresh-requested"
    stamp = ""
    with suppress(OSError):
        content = Path(f"{path}.digest").read_text(errors="replace")
        # Shell read required a terminating newline; incomplete sidecars never claim a digest.
        if "\n" in content:
            stamp = content.split("\n", 1)[0]
    return "digest-moved" if digest and stamp and digest != stamp else "reuse"


def import_image(options: dict, env: dict[str, str] | None = None) -> Path:
    """Import on the chosen host using the original cache and scratch policies."""
    env = dict(os.environ if env is None else env)
    supported = {"linux/amd64": {"x86_64", "amd64"}, "linux/arm64": {"aarch64", "arm64"}}
    if platform.machine() not in supported.get(options["platform"], set()):
        raise ImportFailure(
            "container image platform does not match the allocated architecture", 13
        )
    mode = options["mode"]
    scratch = options.get("scratch", "")
    if (
        mode == "compute"
        and scratch
        and (not Path(scratch).is_absolute() or not Path(scratch).is_dir())
    ):
        raise ImportFailure("invalid container import scratch directory", 14)
    path = Path(options["path"])
    lock = Path(options["lock"])
    image, digest = options["image"], options.get("digest", "")
    local_scratch = mode == "compute" or options.get("local_scratch", False)
    prefix = (
        "inferencex-collectivex-home." if mode == "compute" else "inferencex-collectivex-enroot."
    )
    context = (
        tempfile.TemporaryDirectory(prefix=prefix, dir=scratch or "/tmp")
        if local_scratch
        else nullcontext(None)
    )
    with context as temporary:
        if temporary is not None:
            base = Path(temporary)
            if mode == "compute":
                env.update(HOME=temporary, XDG_CACHE_HOME=str(base / ".cache"))
            for key, name in (
                ("TEMP", "tmp"),
                ("CACHE", "cache"),
                ("DATA", "data"),
                ("RUNTIME", "run"),
            ):
                directory = base / (f"enroot-{name}" if mode == "compute" else name)
                directory.mkdir()
                env[f"ENROOT_{key}_PATH"] = str(directory)
        path.parent.mkdir(parents=True, exist_ok=True)
        # A B300 import can hold this lock for ~18 minutes. The local wait must outlast it;
        # remote shared storage serializes imports, while node-local caches import in parallel.
        with locked(lock, 2700 if mode == "local" else None, private=False):
            verdict = squash_verdict(path, digest, options.get("refresh_epoch"))
            if (
                verdict == "reuse"
                and run(["unsquashfs", "-l", str(path)], env=env, check=False).returncode
            ):
                verdict = "invalid"
            if verdict == "reuse":
                log("container squash ready (reusing staged import)")
                return path
            log(f"importing configured container image ({verdict})")
            if mode == "compute":
                commands = [
                    ["enroot", "version"],
                    ["df", "-hT", env["ENROOT_TEMP_PATH"], str(path.parent)],
                    [
                        "findmnt",
                        "-T",
                        env["ENROOT_TEMP_PATH"],
                        "-o",
                        "TARGET,SOURCE,FSTYPE,OPTIONS",
                    ],
                ]
                commands += [
                    ["df", "-hT", directory]
                    for directory in ("/tmp", "/var/tmp", "/dev/shm", "/scratch", "/local")
                    if Path(directory).is_dir()
                ]
                converter = shutil.which("enroot-aufs2ovlfs", path=env.get("PATH"))
                if converter:
                    commands.append(["getcap", converter])
                for command in commands:
                    with suppress(OSError):
                        run(command, env=env, check=False, capture=False)
            path.unlink(missing_ok=True)
            Path(f"{path}.digest").unlink(missing_ok=True)
            run(["enroot", "import", "-o", str(path), f"docker://{image}"], env=env, capture=False)
            run(["unsquashfs", "-l", str(path)], env=env, capture=mode == "compute")
            # Other accounts must be able to reuse the squash rather than receiving Pyxis's
            # "Invalid image format" against a 0600 file. Sidecar writes remain best effort.
            with suppress(OSError):
                path.chmod(path.stat().st_mode | 0o444)
            with suppress(OSError):
                Path(f"{path}.digest").write_text(digest + "\n")
            # Retired per-run names are never reused; protect concurrent older runs for two days.
            sanitized = re.sub(r"[/:@#]", "_", image)
            for previous in path.parent.glob(f"*_{sanitized}.sqsh"):
                with suppress(OSError):
                    if (
                        previous != path
                        and not previous.is_symlink()
                        and previous.is_file()
                        and int((time.time() - previous.stat().st_mtime) / 60) > 2880
                    ):
                        previous.unlink()
    return path


def ensure_image(allocation, env: dict[str, str], *, local: bool = False, attempt: int = 1) -> Path:
    """Retry compute imports with independent logs; architecture mismatches never retry."""
    path = squash_path(
        Path(env["COLLX_SQUASH_DIR"]), env["COLLECTIVEX_IMAGE"], env["COLLX_IMAGE_PLATFORM"]
    )
    lock_dir = Path(env.get("COLLX_LOCK_DIR") or path.parent / ".locks")
    options = {
        "path": str(path),
        "lock": str(lock_dir / f"{path.stem}.lock"),
        "image": env["COLLECTIVEX_IMAGE"],
        "platform": env["COLLX_IMAGE_PLATFORM"],
        "digest": env.get("COLLX_IMAGE_DIGEST", ""),
        "mode": "local" if local else "compute",
        "scratch": "" if local else env.get("COLLX_IMPORT_TMPDIR", ""),
        "local_scratch": local,
        "refresh_epoch": int(env.get("COLLX_LAUNCH_EPOCH") or time.time())
        if env.get("COLLX_IMAGE_REFRESH", "0") == "1"
        else None,
    }
    label = "container-import" + (f"-a{attempt}" if attempt > 1 else "")
    attempts = 1 if local else int(env.get("COLLX_IMPORT_ATTEMPTS", "3"))
    for index in range(1, attempts + 1):
        output = log_path(allocation.root, label + (f"-r{index}" if index > 1 else ""))
        try:
            if local:
                # Keep imports in a child process so importer environment and scratch stay local.
                run(
                    [
                        "python3",
                        str(Path(__file__).with_name("node.py")),
                        "import-image",
                        json.dumps(options),
                    ],
                    env=env,
                    path=output,
                )
            else:
                allocation.host(
                    int(env["COLLX_NODES"]), ["import-image", json.dumps(options)], output
                )
            return path
        except subprocess.CalledProcessError as exc:
            log_tail(output)
            if exc.returncode == 13:
                raise RuntimeError(
                    "container image platform does not match the allocated architecture"
                ) from exc
            # GB300 soft-mounted NFS/RDMA may briefly fail mkdir. Re-taking the lock and
            # removing a partial import makes a retry safe, with the original 30s * attempt backoff.
            if index < attempts:
                log(
                    f"container import attempt {index}/{attempts} failed (rc={exc.returncode}); retrying"
                )
                time.sleep(index * 30)
    raise RuntimeError(f"container import failed after {attempts} attempts")


def implicit_stage_base(home: str = "", isolation_key: str = "") -> Path:
    """Use the compute-visible passwd home, never the workflow's temporary HOME."""
    if isolation_key and not all(char.isalnum() or char in "._-" for char in isolation_key):
        raise ValueError("invalid stage isolation key")
    base = Path(home or pwd.getpwuid(os.getuid()).pw_dir).resolve()
    path = base / (".inferencex-collectivex-stage" + (f"-{isolation_key}" if isolation_key else ""))
    path.mkdir(mode=0o700, exist_ok=True)
    return path


def prepare_stage_dir(runner: str, env: dict[str, str]) -> dict[str, str]:
    """Resolve the existing pool-specific compute-visible stage base."""
    result = dict(env)
    if env.get("COLLECTIVEX_CANONICAL_GHA", "0") != "1":
        return result
    squash = env.get("COLLX_SQUASH_DIR", "")
    if not squash:
        raise RuntimeError("canonical CollectiveX execution requires shared container storage")
    selected = "" if runner in ("b300", "gb300") else env.get("COLLX_STAGE_DIR", "")
    if not selected:
        if runner in ("h100-dgxc", "b200-nscale"):
            # These passwd homes are login-local; the squash parent is compute-visible.
            selected = implicit_stage_base(str(Path(squash).parent))
        elif runner in ("b300", "gb300"):
            selected = implicit_stage_base(
                isolation_key=env.get("COLLECTIVEX_EXECUTION_ID") or env.get("GITHUB_RUN_ID", "")
            )
        elif runner == "h200-dgxc":
            selected = implicit_stage_base()
        elif runner in ("mi300x", "mi325x", "mi355x"):
            temporary = Path(env.get("RUNNER_TEMP", ""))
            if not temporary.is_absolute() or temporary.parts[-2:] != ("_work", "_temp"):
                raise RuntimeError("canonical AMD execution requires a standard shared runner temp")
            selected = implicit_stage_base(str(temporary.parent.parent))
        else:
            raise RuntimeError(
                "canonical CollectiveX execution requires a configured shared stage directory"
            )
    elif runner == "mi300x":
        selected = Path(selected).resolve()
        if not selected.is_dir():
            raise RuntimeError(
                "canonical MI300X execution cannot resolve the shared stage directory"
            )
    result["COLLX_STAGE_DIR"] = str(selected)
    return result


def stage_path(repo: Path, env: dict[str, str]) -> Path:
    """Resolve and validate the child before copying, so interrupted copies can be cleaned."""
    tag = env.get("COLLECTIVEX_EXECUTION_ID") or env.get("GITHUB_RUN_ID") or f"manual-{os.getpid()}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", tag):
        raise ValueError("invalid staging execution identity")
    selected = env.get("COLLX_STAGE_DIR", "")
    if not selected or selected == str(repo):
        if not env.get("COLLX_SQUASH_DIR"):
            raise RuntimeError("CollectiveX staging requires COLLX_STAGE_DIR or COLLX_SQUASH_DIR")
        base, name = Path(env["COLLX_SQUASH_DIR"]), f".collectivex-stage-{tag}"
    else:
        base, name = Path(selected), f"job_{tag}"
    child = base / name
    if child.exists() or base.resolve() == Path("/"):
        raise ValueError("invalid staging directory")
    for excluded in (str(repo), env.get("COLLX_JOB_ROOT"), env.get("GITHUB_WORKSPACE")):
        if excluded and base.resolve() == Path(excluded).resolve():
            raise ValueError("invalid staging base")
    return child


def stage_repository(repo: Path, destination: Path) -> None:
    """Copy only CollectiveX, preserving the existing private-file exclusions."""
    destination.mkdir(mode=0o700)
    log("staging CollectiveX on compute-visible storage")
    try:
        shutil.copytree(
            repo / "experimental/CollectiveX",
            destination / "experimental/CollectiveX",
            ignore=shutil.ignore_patterns(*EXCLUDES),
        )
    except Exception:
        try:
            shutil.rmtree(destination)
        except OSError as exc:
            log(f"ERROR: cannot remove the incomplete execution stage: {exc}")
        raise


def collect_results(source: Path, repo: Path) -> None:
    """Copy staged JSONs back to the checkout read by upload-artifact, including failed cases."""
    if source == repo:
        return
    destination = repo / "experimental/CollectiveX/results"
    destination.mkdir(parents=True, exist_ok=True)
    files = sorted((source / "experimental/CollectiveX/results").glob("*.json"))
    if not files:
        raise RuntimeError("staged run produced no result JSON")
    for path in files:
        shutil.copyfile(path, destination / path.name)
    log("collected staged results for artifact validation")


def cleanup_stage(source: Path, repo: Path) -> None:
    """Remove only a real generated stage directory after allocation teardown."""
    if source == repo:
        return
    if not source.is_dir() or source.is_symlink() or source == Path("/"):
        raise RuntimeError("refusing to remove an invalid stage directory")
    shutil.rmtree(source)
    log("removed generated per-execution stage directory")
