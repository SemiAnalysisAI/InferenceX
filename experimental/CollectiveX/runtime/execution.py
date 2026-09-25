"""One shard's allocation, preparation, sequential cases, and recoverable cleanup."""

from __future__ import annotations

from contextlib import suppress
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import time

from . import build, config, probe, storage
from .config import Plan, make_plan
from .scheduler import interrupted, log, log_path, log_tail, run, write_json
from .scheduler import SlurmAllocation, HOST_EXPORTS, existing_exclusions


ROOT_NAME = r"inferencex-collectivex-([0-9]+)-([0-9]+)-([A-Za-z0-9._-]+)"
PARENT_NAME = r"inferencex-collectivex-parent-([0-9]+)-([0-9]+)-([A-Za-z0-9._-]+)"


def validate_job_root(root: Path) -> None:
    """Preserve the workflow's private-root guard before recording or cancelling jobs."""
    match = re.fullmatch(ROOT_NAME, root.name)
    parent = re.fullmatch(PARENT_NAME, root.parent.name)
    valid = match and (
        root.parent == Path("/tmp")
        or (root.parent.parent == Path("/tmp") and parent and parent.groups() == match.groups())
    )
    if not valid or not root.is_dir() or root.is_symlink():
        raise ValueError("CollectiveX isolated root is invalid")
    status = root.stat()
    if status.st_uid != os.getuid() or status.st_mode & 0o7777 != 0o700:
        raise ValueError("CollectiveX isolated root is not private")


def control_document(repo: Path, env: dict[str, str]) -> dict:
    """Resolve the shard once, retaining the legacy relative-to-CollectiveX lookup."""
    path = Path(env["COLLX_SHARD_FILE"])
    if not path.is_file():
        path = repo / "experimental/CollectiveX" / path
    document = config.load(str(path))
    if not document["cases"]:
        raise ValueError("shard declares no cases")
    return document


def case_arguments(document: dict, index: int, plan: Plan, timestamp: str) -> list[str]:
    """Prove placement before constructing argv for either Slurm or Docker."""
    return config.case_arguments(
        document,
        index,
        plan.runner,
        timestamp,
        tuple(map(str, (plan.world, plan.nodes, plan.gpus, plan.domain))),
    )


def _validate_allocation(allocation: SlurmAllocation, plan: Plan, attempt: int) -> str:
    """Return the failing phase while keeping probe evidence and per-pool retry policy."""
    suffix = f"-a{attempt}" if attempt > 1 else ""
    if plan.nodes > 1 and plan.transport != "mnnvl":
        path = log_path(allocation.root, f"network-profile{suffix}")
        try:
            allocation.host(
                plan.nodes,
                [
                    "network-profile",
                    plan.env.get("COLLX_SOCKET_IFNAME", ""),
                    plan.env["COLLX_RDMA_DEVICES"],
                    plan.env.get("COLLX_IB_GID_INDEX", ""),
                    plan.env.get("COLLX_RDMA_FABRIC", ""),
                ],
                path,
            )
            plan.env = probe.validated_selectors(path.read_text(), plan.nodes, plan.env)
            allocation.env = plan.env
        except (subprocess.CalledProcessError, RuntimeError):
            log_tail(path)
            return "network"
    if plan.pool.kind == "nvidia":
        probes = (["cuda-context"] if plan.runner == "b300" else []) + ["gpu-health"]
        for name in probes:
            path = log_path(allocation.root, f"{name}{suffix}")
            options = {"gres": f"gpu:{plan.gpus}"}
            command = [name]
            if name == "gpu-health":
                # A collective is paced by its slowest GPU. Slurm's five-minute guard bounds
                # nvidia-smi wedged in D-state, which Python's own timeout cannot reap.
                options["time"] = 5
            else:
                command.append(str(plan.gpus))
            try:
                allocation.host(plan.nodes, command, path, **options)
            except subprocess.CalledProcessError:
                log_tail(path)
                return name
    return ""


def allocate_image(allocation: SlurmAllocation, plan: Plan) -> Path:
    """Acquire healthy nodes and import the image with the original pool-specific retries."""
    excluded = plan.env.get("COLLX_EXCLUDE_NODES", "")
    if plan.swap and excluded:
        excluded = existing_exclusions(excluded)
    attempts = 1 if plan.swap or plan.pool.kind == "gb" else 3
    for attempt in range(1, attempts + 1):
        allocation.allocate(plan.allocation_args(excluded), attempt)
        reason = (
            ""
            if plan.swap or plan.pool.kind == "gb"
            else _validate_allocation(allocation, plan, attempt)
        )
        if not reason and plan.pool.kind == "amd" and not plan.swap:
            try:
                return storage.ensure_image(allocation, plan.env, attempt=attempt)
            except RuntimeError:
                reason = "container-import"
        if not reason:
            break
        # AMD retries mismatched RoCE selectors and transient container-import failures unless
        # nodes were explicitly pinned. NVIDIA retries H100 network, B300 CUDA context, and
        # throttled GPUs on every SKU. A rejected node is excluded from the next allocation.
        retry = (
            (plan.pool.kind == "amd" and not plan.env.get("COLLX_NODELIST"))
            or reason == "gpu-health"
            or (plan.runner, reason) in (("h100-dgxc", "network"), ("b300", "cuda-context"))
        )
        if not retry or attempt == attempts:
            raise RuntimeError(f"allocated nodes failed {reason} validation")
        rejected = allocation.nodes()
        log(f"allocated nodes failed {reason} validation; retrying elsewhere")
        allocation.release()
        excluded = ",".join(filter(None, (excluded, rejected)))
    if (
        plan.swap
        and plan.env.get("COLLX_IMAGE_REFRESH") == "0"
        and plan.env.get("COLLX_SWAP_STAGED_DIR")
    ):
        # H100 serving caches use the tag filename without CollectiveX's platform prefix.
        image = re.sub(r"[/:@#]", "_", plan.env["COLLECTIVEX_IMAGE"])
        staged = Path(plan.env["COLLX_SWAP_STAGED_DIR"]) / f"{image}.sqsh"
        if run(["unsquashfs", "-l", str(staged)], check=False).returncode == 0:
            log(f"using operator-staged image: {staged}")
            return staged
        log(f"requested image is not staged: {staged}")
    return storage.ensure_image(allocation, plan.env, local=plan.runner == "b300" and not plan.swap)


def rendezvous(allocation: SlurmAllocation, plan: Plan) -> None:
    """Resolve global rank zero on the validated primary interface, then rebuild selectors."""
    path = log_path(allocation.root, "rendezvous")
    interface = plan.env.get("COLLX_SOCKET_IFNAME", "")
    selected = interface if re.fullmatch(probe.INTERFACE, interface) else ""
    allocation.host(1, ["address", selected], path, relative=0)
    # srun diagnostics share the private log; they must never become MASTER_ADDR.
    matches = re.findall(r"^\[collectivex-private\] rendezvous=(.+)$", path.read_text(), re.M)
    address = matches[0] if len(matches) == 1 else ""
    pattern = r"([0-9]{1,3}\.){3}[0-9]{1,3}" if selected else r"[A-Za-z0-9][A-Za-z0-9._-]*"
    if not re.fullmatch(pattern, address):
        log_tail(path)
        raise RuntimeError("could not resolve the allocated primary node/interface")
    port = plan.env.get("COLLX_MASTER_PORT") or "29551"
    if not re.fullmatch(r"[1-9][0-9]*", port) or int(port) > 65535:
        raise ValueError("invalid distributed rendezvous port")
    plan.env.update(MASTER_ADDR=address, MASTER_PORT=port)
    plan.env = probe.network_environment(plan.env, plan.nodes, plan.transport)
    allocation.env = plan.env


def run_ep_cases(
    allocation: SlurmAllocation, plan: Plan, document: dict, options: dict, timestamp: str
) -> int:
    """Prepare once per node, then run each case on one Slurm task per GPU."""
    rendezvous(allocation, plan)
    output = log_path(allocation.root, "backend-prepare")
    try:
        allocation.step(
            {
                **options,
                "nodes": plan.nodes,
                "ntasks": plan.nodes,
                "ntasks_per_node": 1,
                "chdir": "/tmp",
                "export": "ALL",
            },
            ["python3", "runtime/node.py", "prepare"],
            output,
        )
    except subprocess.CalledProcessError:
        log_tail(output)
        raise
    failures = 0
    for index in range(len(document["cases"])):
        argv = case_arguments(document, index, plan, timestamp)
        log(f"EP{plan.world}[{index + 1}/{len(document['cases'])}] {plan.backend}")
        output = log_path(allocation.root, f"runtime-c{index:03d}")
        try:
            # This is a hang guard, not a work budget: 900s killed FP8 prefill with complete
            # artifacts, and 1800s killed degraded EP16 fabrics (~34GB/s/node). Keep 5400s.
            allocation.step(
                {
                    **options,
                    "nodes": plan.nodes,
                    "ntasks": plan.world,
                    "ntasks_per_node": plan.gpus,
                    "chdir": "/tmp",
                    "export": "ALL",
                },
                ["python3", "runtime/node.py", "rank", "--", *argv],
                output,
                timeout=int(plan.env.get("COLLX_RUN_TIMEOUT") or "5400"),
            )
        except subprocess.CalledProcessError:
            log(f"ERROR: case {index} failed")
            log_tail(output)
            failures += 1
    if failures:
        log(f"ERROR: {failures}/{len(document['cases'])} case(s) failed")
    return int(failures > 0)


def cleanup(root: Path, repo: Path, env: dict[str, str]) -> None:
    """Recover resources from private state even when the launcher process was killed."""
    validate_job_root(root)
    state_file = root / "execution.json"
    state = json.loads(state_file.read_text()) if state_file.is_file() else {}
    allocation = SlurmAllocation(root, env)
    record = root / "jobid"
    if record.is_file():
        allocation.job_id = record.read_text().strip()
        if not re.fullmatch(r"[1-9][0-9]*", allocation.job_id):
            raise ValueError("invalid cleanup allocation")
        # Pyxis global-scope writable containers otherwise survive allocation teardown and
        # accumulate tens of GB until ENOSPC. Best effort, bounded, and before scancel.
        with suppress(OSError, subprocess.SubprocessError):
            allocation.step(
                {"nodes": state.get("nodes", 1), "ntasks_per_node": 1, "chdir": "/tmp"},
                ["enroot", "remove", "-f", f"pyxis_cxep_{allocation.job_id}"],
                log_path(root, "container-cleanup"),
                timeout=120,
                kill_after=None,
            )
        allocation.release()  # Do not touch results or source while an allocation may still write.
    if state.get("containers"):
        from .docker import remove_containers

        remove_containers(state, env)
    source = Path(state["stage"]) if state.get("stage") else None
    error = None
    if source and source != repo and source.exists():
        try:
            storage.collect_results(source, repo)
        except Exception as exc:
            log(f"ERROR: {exc}")
            error = exc
        storage.cleanup_stage(source, repo)
        state["stage"] = None
        write_json(state_file, state)
    if error:
        raise error


def execute(repo: Path, incoming: dict[str, str]) -> int:
    """Run the selected EP or block-copy cell with one cleanup path for every exit."""
    os.umask(0o077)
    root = Path(incoming["COLLX_JOB_ROOT"])
    validate_job_root(root)
    env = {
        **incoming,
        "COLLX_LAUNCH_EPOCH": incoming.get("COLLX_LAUNCH_EPOCH") or str(int(time.time())),
    }
    if env.get("COLLX_BENCH") == "swap-blocks":
        config.validate_swap_environment(env)
    plan = make_plan(env["COLLX_SHARD_SKU"], env.get("COLLX_BENCH", ""), env)
    state = {"nodes": plan.nodes, "stage": None, "containers": [], "docker": []}
    write_json(root / "execution.json", state)
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    with interrupted():
        failed = False
        try:
            if plan.pool.kind == "docker":
                from .docker import execute_docker

                return execute_docker(plan, repo, root, state, timestamp)
            if plan.pool.kind != "amd" or plan.swap:
                plan.env = storage.select_image(
                    plan.env["COLLX_SWAP_IMAGE"] if plan.swap else plan.env["COLLX_IMAGE"], plan.env
                )
            source = storage.stage_path(repo, plan.env)
            state["stage"] = str(source)
            write_json(root / "execution.json", state)
            storage.stage_repository(repo, source)
            build.stage_backend_source(source, plan.backend, root)
            if plan.pool.kind == "amd" and not plan.swap:
                plan.env = storage.select_image(plan.env["COLLX_IMAGE"], plan.env)
            cache = ""
            if not plan.swap and plan.pool.kind in ("nvidia", "gb"):
                cache = probe.prepare_cache(plan.env["COLLX_SQUASH_DIR"])
            allocation = SlurmAllocation(root, plan.env)
            image = allocate_image(allocation, plan)
            if not plan.swap and plan.pool.kind == "amd" and plan.backend == "uccl-ep":
                # AMD squash storage is node-local/root-owned. Keep the cross-allocation
                # backend cache beside the shared stage, surviving stage cleanup.
                cache = probe.prepare_cache(plan.env["COLLX_STAGE_DIR"])
            if cache:
                plan.env["COLLX_BACKEND_CACHE_ROOT"] = "/cx-cache"
            if not plan.swap and (plan.pool.kind in ("nvidia", "gb") or plan.backend == "uccl-ep"):
                plan.env["COLLX_BACKEND_SOURCE_ROOT"] = (
                    "/ix/experimental/CollectiveX/.collx_sources"
                )
            allocation.env = plan.env
            options = plan.container_options(source, image, allocation.job_id, cache)
            (source / "experimental/CollectiveX/results").mkdir(parents=True, exist_ok=True)
            if plan.swap:
                return run_swap_cases(allocation, plan, options)
            return run_ep_cases(
                allocation, plan, control_document(repo, plan.env), options, timestamp
            )
        except BaseException:
            failed = True
            raise
        finally:
            try:
                cleanup(root, repo, plan.env)
            except Exception as exc:
                if not failed:
                    raise
                # Preserve a signal/setup failure's original status, including 128+signal.
                # The durable allocation record remains for the workflow's independent retry.
                log(f"ERROR: cleanup failed: {exc}")


def run_swap_cases(allocation, plan, options: dict) -> int:
    """Execute both layouts using one process and the existing restricted export set."""
    for layout in ("contiguous", "random"):
        path = log_path(allocation.root, f"swap-blocks-{layout}")
        try:
            allocation.step(
                {
                    **options,
                    "nodes": 1,
                    "ntasks": 1,
                    "ntasks_per_node": 1,
                    "chdir": "/tmp",
                    "export": f"{HOST_EXPORTS},COLLECTIVEX_IMAGE,COLLECTIVEX_SOURCE_SHA,COLLX_SHARD_SKU",
                },
                ["python3", *config.swap_arguments(plan.env, layout)],
                path,
            )
        except subprocess.CalledProcessError as exc:
            log_tail(path)
            raise RuntimeError(f"swap-blocks {layout} failed") from exc
        print(path.read_text(errors="replace"), end="")
    return 0
