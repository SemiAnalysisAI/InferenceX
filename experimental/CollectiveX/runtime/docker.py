"""Slurm-less Taiwan pools: direct Docker/torchrun calls with tracked container ownership."""

from __future__ import annotations

from contextlib import suppress
import hashlib
import os
from pathlib import Path
import re
import subprocess
import tempfile

from . import build, config, storage
from .scheduler import log, run, write_json


def docker_command(env: dict[str, str], *, swap_blocks: bool) -> list[str]:
    """Keep the pool's Docker-group versus passwordless-sudo selection."""
    if run(["docker", "ps"], env=env, check=False).returncode == 0:
        return ["docker"]
    if run(["sudo", "-n", "docker", "ps"], env=env, check=False).returncode:
        raise RuntimeError("Docker is unavailable to the runner account")
    return ["sudo", "-n", "docker"] if swap_blocks else ["sudo", "docker"]


def remove_containers(state: dict, env: dict[str, str]) -> None:
    """Remove only names recorded for this execution, including after process cancellation."""
    command = state.get("docker", [])
    if command not in (["docker"], ["sudo", "docker"], ["sudo", "-n", "docker"]):
        raise ValueError("invalid recorded Docker command")
    for name in state.get("containers", []):
        if not re.fullmatch(r"cx(?:ep|swap|build)_[A-Za-z0-9._-]+", name):
            raise ValueError("invalid recorded container name")
        with suppress(OSError, subprocess.SubprocessError):
            run([*command, "rm", "-f", name], env=env, check=False)


def _record_container(root: Path, state: dict, name: str) -> None:
    """Persist ownership before launching so the independent workflow finalizer can recover."""
    if name not in state["containers"]:
        state["containers"].append(name)
    write_json(root / "execution.json", state)


def _reap_previous(command: list[str], image: str, env: dict[str, str]) -> None:
    """Keep the established one-runner-per-node recovery of earlier pinned-image containers.

    docker --rm only acts on exit; a killed Actions tree can leave a root container holding
    GPUs, causing hipIpcGetMemHandle failures for the next multi-rank case.
    """
    result = run([*command, "ps", "-q", "--filter", f"ancestor={image}"], env=env, check=False)
    for container in result.stdout.splitlines():
        started = run(
            [*command, "inspect", "-f", "{{.State.StartedAt}}", container], env=env, check=False
        )
        if started.returncode == 0:
            log(
                f"reaping stray container {container[:12]} from an earlier leg (started {started.stdout.strip()})"
            )
            run([*command, "rm", "-f", container], env=env, check=False)


def _uccl_prefix(plan, repo: Path, root: Path, state: dict, command: list[str]) -> Path:
    """Preserve the commit/image-content/arch keyed Docker build cache outside the workspace."""
    build.stage_backend_source(repo, "uccl-ep", root)
    arch = config._platforms()[plan.runner]["arch"]
    image = plan.env["COLLX_IMAGE"]
    inspected = run(
        [*command, "image", "inspect", "--format", "{{.Id}}", image], env=plan.env, check=False
    )
    identity = inspected.stdout.rstrip("\n") if inspected.returncode == 0 else image
    material = f"{build.UCCL_COMMIT}\0{identity}\0{arch}".encode()
    key = hashlib.sha1(material).hexdigest()[:16]
    prefix = Path(f"/tmp/collx-uccl-pfx-{key}")
    if (prefix / ".ready").is_file():
        log(f"uccl-ep: reusing persisted build at {prefix}")
        return prefix
    # Container writes are root-owned. The prefix is deliberately outside the runner job root;
    # publish only after a successful import check, so interrupted copies never look ready.
    build.remove(prefix)
    temporary = Path(tempfile.mkdtemp(prefix="collx-uccl-pfx.", dir="/tmp"))
    name = f"cxbuild_{plan.env.get('COLLECTIVEX_EXECUTION_ID') or 'manual'}"
    _record_container(root, state, name)
    try:
        run(
            [
                *command,
                "run",
                "--rm",
                "--name",
                name,
                "--device",
                "/dev/kfd",
                "--device",
                "/dev/dri",
                "--group-add",
                "video",
                "--group-add",
                "render",
                "--ipc",
                "host",
                "--shm-size",
                "32g",
                "--cap-add",
                "SYS_PTRACE",
                "--security-opt",
                "seccomp=unconfined",
                "-e",
                "PYTHONDONTWRITEBYTECODE=1",
                "-e",
                "COLLX_VENDOR=amd",
                "-e",
                "COLLX_BACKEND_SOURCE_ROOT=/cx/.collx_sources",
                "-v",
                f"{repo}/experimental/CollectiveX:/cx",
                "-v",
                f"{temporary}:/uccl_pfx",
                "-w",
                "/cx",
                image,
                "python3",
                "runtime/node.py",
                "prepare-docker-uccl",
                arch,
            ],
            env=plan.env,
            capture=False,
        )
        try:
            temporary.rename(prefix)
        except OSError:
            build.remove(temporary)
        if not (prefix / ".ready").is_file():
            raise RuntimeError("uccl-ep: build did not publish a ready cache")
    finally:
        if temporary.exists():
            build.remove(temporary)
    log(f"uccl-ep: build persisted to {prefix}")
    return prefix


def _swap(plan, repo: Path, root: Path, state: dict, command: list[str]) -> int:
    """Run each copy layout as the runner user, keeping its two destination guard blocks."""
    env = plan.env = storage.select_image(plan.env["COLLX_SWAP_IMAGE"], plan.env)
    image = env["COLLX_SWAP_IMAGE"]
    if run([*command, "image", "inspect", image], env=env, check=False).returncode:
        run([*command, "pull", image], env=env, capture=False)
    groups = []
    for name in ("video", "render"):
        group = run(["getent", "group", name], env=env).stdout.split(":")
        if len(group) > 2 and group[2]:
            groups += ["--group-add", group[2]]
    name = f"cxswap_{env['COLLECTIVEX_EXECUTION_ID']}"
    _record_container(root, state, name)
    for layout in ("contiguous", "random"):
        forwarded = []
        for key in ("COLLECTIVEX_IMAGE", "COLLECTIVEX_SOURCE_SHA", "COLLX_SHARD_SKU"):
            forwarded += ["-e", f"{key}={env[key]}"]
        run(
            [
                *command,
                "run",
                "--rm",
                "--name",
                name,
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                *groups,
                "--device",
                "/dev/kfd",
                "--device",
                "/dev/dri",
                "--ipc",
                "host",
                "--security-opt",
                "seccomp=unconfined",
                "--entrypoint",
                "python3",
                "-e",
                "HOME=/tmp",
                "-e",
                "PYTHONDONTWRITEBYTECODE=1",
                *forwarded,
                "-v",
                f"{repo}:/ix",
                "-w",
                "/ix/experimental/CollectiveX",
                image,
                *config.swap_arguments(env, layout),
            ],
            env=env,
            capture=False,
        )
    return 0


def execute_docker(plan, repo: Path, root: Path, state: dict, timestamp: str) -> int:
    """Execute the same two-attempt EP cases, forwarding argv directly to torchrun."""
    from .execution import case_arguments, control_document

    command = docker_command(plan.env, swap_blocks=plan.swap)
    state["docker"] = command
    write_json(root / "execution.json", state)
    (repo / "experimental/CollectiveX/results").mkdir(parents=True, exist_ok=True)
    if plan.swap:
        return _swap(plan, repo, root, state, command)
    image, env = plan.env["COLLX_IMAGE"], plan.env
    _reap_previous(command, image, env)
    if run([*command, "image", "inspect", image], env=env, check=False).returncode:
        run([*command, "pull", image], env=env, capture=False)
    mounts = []
    if plan.backend == "uccl-ep":
        prefix = _uccl_prefix(plan, repo, root, state, command)
        mounts = ["-v", f"{prefix}:/uccl_pfx"]
        settings = {
            "PYTHONPATH": "/uccl_pfx",
            "UCCL_EP_ENABLE_AGGRESSIVE_ATOMIC": env.get("UCCL_EP_ENABLE_AGGRESSIVE_ATOMIC") or "1",
        }
    else:
        # MoRI SDMA queue initialization fails on mi300x-tw's kernel thunk. Keep its IPC
        # fallback while mi325x-tw retains SDMA; explicit caller overrides still win.
        settings = {
            "MORI_DISABLE_AUTO_XGMI": env.get("MORI_DISABLE_AUTO_XGMI") or "0",
            "MORI_ENABLE_SDMA": env.get("MORI_ENABLE_SDMA")
            or ("0" if plan.runner == "mi300x-tw" else "1"),
            "MORI_APP_LOG_LEVEL": env.get("MORI_APP_LOG_LEVEL") or "info",
        }
    # New runtime imports must not leave root-owned __pycache__ directories in the mounted
    # runner source tree. Backend build caches remain outside the workspace as before.
    settings.update(
        HSA_NO_SCRATCH_RECLAIM="1",
        PYTHONDONTWRITEBYTECODE="1",
        COLLECTIVEX_SOURCE_SHA=env.get("COLLECTIVEX_SOURCE_SHA", ""),
    )
    forwarded = [item for key, value in settings.items() for item in ("-e", f"{key}={value}")]
    document = control_document(repo, env)
    failed = False
    for index in range(len(document["cases"])):
        try:
            argv = case_arguments(document, index, plan, timestamp)
        except (ValueError, KeyError, SystemExit):
            log(f"case {index}: argv generation failed")
            failed = True
            continue
        for attempt in (1, 2):
            # A cold torchrun can fail before any worker starts; retain the one retry and
            # the same output filename, which the successful attempt overwrites.
            execution = env.get("COLLECTIVEX_EXECUTION_ID") or "manual"
            name = f"cxep_{execution}_c{index}_a{attempt}"
            _record_container(root, state, name)
            log(
                f"case {index}/{len(document['cases'])} attempt {attempt}: docker torchrun --nproc-per-node={plan.world}"
            )
            result = run(
                [
                    *command,
                    "run",
                    "--rm",
                    "--name",
                    name,
                    "--label",
                    f"collectivex.leg={execution}",
                    "--device",
                    "/dev/kfd",
                    "--device",
                    "/dev/dri",
                    "--group-add",
                    "video",
                    "--group-add",
                    "render",
                    "--ipc",
                    "host",
                    "--shm-size",
                    "32g",
                    "--cap-add",
                    "SYS_PTRACE",
                    "--security-opt",
                    "seccomp=unconfined",
                    "--network",
                    "host",
                    *forwarded,
                    "-v",
                    f"{repo}/experimental/CollectiveX:/cx",
                    *mounts,
                    "-w",
                    "/cx",
                    image,
                    "torchrun",
                    "--standalone",
                    f"--nproc-per-node={plan.world}",
                    "bench/run_ep.py",
                    *argv,
                ],
                env=env,
                check=False,
                capture=False,
            )
            if result.returncode == 0:
                break
            log(f"case {index} attempt {attempt} returned nonzero")
        else:
            log(f"case {index} failed after 2 attempts")
            failed = True
    return int(failed)
