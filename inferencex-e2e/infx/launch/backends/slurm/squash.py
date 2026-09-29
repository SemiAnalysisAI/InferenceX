"""Enroot squash images for Pyxis: cache paths, import locks, validation and import.

Without a squash cache, jobs get the registry reference and Pyxis imports the image itself.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import glob
import os
import re
import socket
import stat
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Literal

from infx.clusters.slurm import SquashPolicy
from infx.launch import proc
from infx.launch.backends.base import BackendError, Job
from infx.launch.backends.slurm.cli import srun

IMPORT_ATTEMPTS = 3
RETRY_DELAY_S = 30.0
_EXIT_NOT_STAGED = 66
_EXIT_LOCK_TIMEOUT = 75
_ENROOT_DIRS = ("ENROOT_TEMP_PATH", "ENROOT_CACHE_PATH", "ENROOT_DATA_PATH", "ENROOT_RUNTIME_PATH")
_KEY_CHARACTERS = re.compile(r"[/:@#]")
_TEMP_GLOB = ".tmp.*.[0-9]*"


class ImageError(BackendError):
    """An image could not be imported, validated, or located."""


class _PermanentImageError(ImageError):
    """A failure that retrying cannot fix (lock timeout, missing pre-staged image)."""


def squash_key(image: str) -> str:
    """Filesystem-safe image key: ``/ : @ #`` become ``_``."""
    return _KEY_CHARACTERS.sub("_", image)


def squash_path(image: str, policy: SquashPolicy) -> Path:
    """Squash file for ``image`` under the policy's directory, named by its key style."""
    if policy.key_style == "underscore":
        return policy.dir / f"{squash_key(image)}.sqsh"
    if policy.key_style == "plus-strip-nvcr":
        image = image.removeprefix("nvcr.io/")
    return policy.dir / f"{_KEY_CHARACTERS.sub('+', image)}.sqsh"


def lock_path(image: str, policy: SquashPolicy) -> Path:
    path = squash_path(image, policy)
    if policy.lock_file == "locks-dir":
        return path.parent / ".locks" / f"{squash_key(image)}.lock"
    return path.with_name(f"{path.name}.lock")


def enroot_uri(image: str) -> str:
    """Enroot import URI for an image reference, keeping digest pins.

    Enroot 3.x cannot parse ``tag@digest``, so a pinned image becomes
    ``registry#repository:digest`` (the digest is immutable, so the tag is dropped).
    Pyxis-style ``registry#repo`` input is read as ``registry/repo``.
    """
    image = image.replace("#", "/", 1)
    without_digest, digest = image, ""
    if "@sha256:" in image:
        without_digest, _, digest = image.rpartition("@")
    first = without_digest.split("/", 1)[0]
    if "/" in without_digest and ("." in first or ":" in first or first == "localhost"):
        registry, repository = first, without_digest.split("/", 1)[1]
    else:
        registry, repository = "registry-1.docker.io", without_digest
    if not digest:
        if registry == "registry-1.docker.io":
            return f"docker://{image}"
        return f"docker://{registry}#{repository}"
    directory, _, name = repository.rpartition("/")
    name = name.split(":", 1)[0]
    repository = f"{directory}/{name}" if directory else name
    if registry == "registry-1.docker.io" and "/" not in repository:
        repository = f"library/{repository}"
    return f"docker://{registry}#{repository}:{digest}"


def registry_reference(image: str) -> str:
    """``--container-image`` value that makes Pyxis import ``image`` itself."""
    return enroot_uri(image).removeprefix("docker://")


def is_valid_squash(path: Path) -> bool:
    if not os.access(path, os.R_OK):
        return False
    try:
        result = proc.run(["unsquashfs", "-l", path], capture=True)
    except FileNotFoundError:
        return False
    return result.returncode == 0


def reuse_or_registry(image: str, policy: SquashPolicy) -> str:
    """``--container-image`` for a job that imports nothing first.

    A valid shared squash is reused; otherwise Pyxis imports the registry reference inside
    the job, with no lock and no extra allocation. Node-local squashes are invisible here.
    """
    if policy.visibility == "shared":
        path = squash_path(image, policy)
        if is_valid_squash(path):
            return str(path)
    return registry_reference(image)


@contextlib.contextmanager
def _locked(path: Path, timeout_s: int) -> Iterator[None]:
    """Hold an exclusive ``flock`` on ``path`` for at most ``timeout_s`` of waiting.

    A lock file another account created is opened read-only: ``flock`` needs no write access.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT, 0o666)
    except OSError:
        descriptor = os.open(path, os.O_RDONLY)
    try:
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as error:
                if error.errno not in {errno.EAGAIN, errno.EACCES}:
                    raise
                if time.monotonic() >= deadline:
                    raise _PermanentImageError(
                        f"timed out after {timeout_s}s waiting for image lock {path}"
                    ) from None
                time.sleep(1)
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _import_here(image: str, policy: SquashPolicy) -> None:
    """Import on this host into a temp file, validate, ``chmod a+r``, rename into place."""
    path = squash_path(image, policy)
    with _locked(lock_path(image, policy), policy.lock_timeout_s):
        if is_valid_squash(path):
            print(
                f"Squash file already exists and is valid, skipping import: {path}", file=sys.stderr
            )
            return
        stale = glob.glob(glob.escape(str(path)) + _TEMP_GLOB)
        for leftover in [path, *map(Path, stale)]:
            leftover.unlink(missing_ok=True)
        temporary = path.with_name(f"{path.name}.tmp.{socket.gethostname()}.{os.getpid()}")
        try:
            with tempfile.TemporaryDirectory(prefix="infx-enroot.") as private:
                env = {**os.environ}
                for name in _ENROOT_DIRS:
                    directory = Path(private) / name.removeprefix("ENROOT_").lower()
                    directory.mkdir()
                    env[name] = str(directory)
                uri = enroot_uri(image)
                imported = proc.run(["enroot", "import", "-o", temporary, uri], env=env, input="")
            if imported.returncode != 0:
                raise ImageError(f"enroot import failed for {uri}")
            if not is_valid_squash(temporary):
                raise ImageError(f"enroot import produced an invalid squash file: {temporary}")
            temporary.chmod(temporary.stat().st_mode | stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


_NODE_SCRIPT = r"""
set -eo pipefail
sq="$1"; lock="$2"; uri="$3"; lock_timeout="$4"; action="$5"
host="$(hostname)"
if [ "$action" = validate ]; then
  if [ -r "$sq" ] && unsquashfs -l "$sq" >/dev/null 2>&1; then
    echo "$host: pre-staged image is valid: $sq"
    exit 0
  fi
  echo "$host: ERROR: pre-staged image is missing or invalid: $sq" >&2
  exit 66
fi
mkdir -p "$(dirname "$sq")" "$(dirname "$lock")"
# Another account's lock file opens read-only; flock needs no write access.
if ! { exec 9>"$lock"; } 2>/dev/null; then
  exec 9<"$lock" || { echo "$host: ERROR: cannot open image lock $lock" >&2; exit 1; }
fi
# flock exits 1 on a -w timeout; anything else (e.g. 127, not installed) is an error.
flock -w "$lock_timeout" 9 || {
  rc=$?
  [ "$rc" -ne 1 ] || { echo "$host: ERROR: timed out waiting for image lock $lock" >&2; exit 75; }
  echo "$host: ERROR: flock failed (exit $rc) on $lock" >&2
  exit "$rc"
}
if [ -r "$sq" ] && unsquashfs -l "$sq" >/dev/null 2>&1; then
  echo "$host: squash file already exists and is valid, skipping import: $sq"
  exit 0
fi
private="$(mktemp -d "${TMPDIR:-/tmp}/infx-enroot.XXXXXX")"
tmp="$sq.tmp.$host.$$"
trap 'rm -rf -- "$private"; rm -f -- "$tmp"' EXIT
export ENROOT_TEMP_PATH="$private/temp" ENROOT_CACHE_PATH="$private/cache"
export ENROOT_DATA_PATH="$private/data" ENROOT_RUNTIME_PATH="$private/runtime"
mkdir -p "$ENROOT_TEMP_PATH" "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH" "$ENROOT_RUNTIME_PATH"
# Only our own dead temp files (<squash>.tmp.<host>.<pid>, see _TEMP_GLOB).
rm -f -- "$sq" "$sq".tmp.*.[0-9]*
echo "$host: importing $uri -> $sq"
enroot import -o "$tmp" "$uri" </dev/null
if ! unsquashfs -l "$tmp" >/dev/null 2>&1; then
  echo "$host: ERROR: enroot import produced an invalid squash file: $tmp" >&2
  exit 1
fi
chmod a+r "$tmp" || true
mv -f -- "$tmp" "$sq"
"""


def _run_node_script(
    image: str,
    policy: SquashPolicy,
    action: Literal["import", "validate"],
    *,
    job: Job | None,
    alloc_args: Sequence[str],
) -> None:
    """Run the node script on ``job``'s node, or in a one-node step ``alloc_args`` allocates."""
    argv = [
        "bash",
        "-c",
        _NODE_SCRIPT,
        "infx-image",
        str(squash_path(image, policy)),
        str(lock_path(image, policy)),
        enroot_uri(image),
        str(policy.lock_timeout_s),
        action,
    ]
    step_args = ["--nodes=1", "--ntasks-per-node=1", "--chdir=/tmp", "--label"]
    rc = srun(job, argv, extra=[*(alloc_args if job is None else ()), *step_args])
    if rc == _EXIT_NOT_STAGED:
        raise _PermanentImageError(
            f"pre-staged image {image} is missing or invalid at {squash_path(image, policy)} "
            "on at least one node; stage it there before running"
        )
    if rc == _EXIT_LOCK_TIMEOUT:
        raise _PermanentImageError(f"timed out waiting for the import lock of {image}")
    if rc != 0:
        raise ImageError(f"image {action} of {image} failed on the allocated nodes (exit {rc})")


def _retrying(action: str, attempt_once: Callable[[], None]) -> None:
    """Run ``attempt_once`` up to IMPORT_ATTEMPTS times; network storage errors are transient."""
    for attempt in range(1, IMPORT_ATTEMPTS + 1):
        try:
            attempt_once()
            return
        except _PermanentImageError:
            raise
        except (ImageError, OSError) as error:
            if attempt == IMPORT_ATTEMPTS:
                raise ImageError(
                    f"{action} failed after {IMPORT_ATTEMPTS} attempts: {error}"
                ) from error
            print(
                f"{action} attempt {attempt}/{IMPORT_ATTEMPTS} failed ({error}); retrying",
                file=sys.stderr,
            )
            time.sleep(attempt * RETRY_DELAY_S)


def ensure_image(
    image: str,
    policy: SquashPolicy,
    *,
    job: Job | None,
    alloc_args: Sequence[str] = (),
) -> str:
    """Make the squash of ``image`` available and return its path for ``--container-image``.

    Node-side modes run on ``job``'s node; a ``compute`` import without a job runs in a
    one-node step allocated by ``alloc_args``. A valid shared squash is reused as is.
    Raises ``ImageError`` when the image cannot be made available.
    """
    mode = policy.import_mode
    path = squash_path(image, policy)
    if mode == "unchecked":
        return str(path)
    node_local = policy.visibility == "node-local"
    if node_local and job is None:
        raise ImageError(f"node-local image storage for {image} requires a job allocation")
    if mode == "pre-staged":
        if node_local:
            _retrying(
                f"validation of pre-staged {image}",
                lambda: _run_node_script(image, policy, "validate", job=job, alloc_args=()),
            )
        elif not is_valid_squash(path):
            raise ImageError(
                f"pre-staged image {image} is missing or invalid at {path}; stage it there before running"
            )
        return str(path)
    if mode == "submit-host":
        _retrying(f"import of {image}", lambda: _import_here(image, policy))
        return str(path)
    if not node_local and is_valid_squash(path):
        print(f"Squash file already exists and is valid, skipping import: {path}", file=sys.stderr)
        return str(path)
    if job is None and (mode == "all-nodes" or not alloc_args):
        raise ImageError(f"{mode} import of {image} needs a job allocation or allocation arguments")
    _retrying(
        f"import of {image}",
        lambda: _run_node_script(image, policy, "import", job=job, alloc_args=alloc_args),
    )
    return str(path)
