#!/usr/bin/env python3
"""CollectiveX v1 EP benchmark entrypoint for torchrun or rank environments."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import hmac
import json
import os
import platform
import re
import shlex
import socket
import subprocess
import sys

# Make the sibling tests/ modules importable when run as `tests/run_ep.py` under
# torchrun (it executes the file as __main__, not as a package).
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

import ep_harness  # noqa: E402  (stdlib-only; safe before torch)
import identity  # noqa: E402


ALLOCATION_STRATUM_CONTRACT = "collectivex-allocation-stratum-v1"
PRIVATE_FABRIC_ENV = {
    "ib_gid_index": "CX_IB_GID_INDEX",
    "rdma_devices": "CX_RDMA_DEVICES",
    "rdma_service_level": "CX_RDMA_SERVICE_LEVEL",
    "rdma_traffic_class": "CX_RDMA_TRAFFIC_CLASS",
    "socket_ifname": "CX_SOCKET_IFNAME",
}


def _numeric_version(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command, capture_output=True, check=False, text=True, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    match = re.search(r"\b[0-9]+(?:\.[0-9]+){1,3}\b", result.stdout)
    return match.group(0) if match else None


def _loaded_collective_version() -> str | None:
    try:
        with open("/proc/self/maps", encoding="utf-8") as handle:
            paths = {
                os.path.realpath(line.rstrip().split()[-1])
                for line in handle
                if any(name in line for name in ("libnccl.so", "librccl.so"))
                and os.path.isfile(line.rstrip().split()[-1])
            }
        if len(paths) != 1:
            return None
        version = ctypes.c_int()
        library = ctypes.CDLL(paths.pop())
        if library.ncclGetVersion(ctypes.byref(version)) != 0:
            return None
        return ep_harness.format_collective_version(version.value)
    except (AttributeError, OSError):
        return None


def _runtime_fingerprint(
    torch, device, *, machine: str, vendor: str, arch: str
) -> dict:
    """Return strict runtime facts without hosts, addresses, UUIDs, or paths."""
    properties = torch.cuda.get_device_properties(device)
    if vendor == "nvidia":
        driver = _numeric_version(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        )
        runtime_kind, runtime_version, collective_kind = (
            "cuda",
            torch.version.cuda,
            "nccl",
        )
    else:
        driver = _numeric_version(["rocm-smi", "--showdriverversion"])
        runtime_kind, runtime_version, collective_kind = (
            "hip",
            torch.version.hip,
            "rccl",
        )
    return {
        "accelerator_runtime": {"kind": runtime_kind, "version": runtime_version},
        "collective_library": {
            "kind": collective_kind,
            "version": _loaded_collective_version(),
        },
        "device": {
            "arch": arch,
            "compute_units": int(properties.multi_processor_count),
            "memory_bytes": int(properties.total_memory),
            "product": torch.cuda.get_device_name(device),
            "warp_size": int(properties.warp_size),
        },
        "driver_version": driver,
        "framework": {"kind": "torch", "version": str(torch.__version__)},
        "machine": machine,
        "python_version": platform.python_version(),
        "vendor": vendor,
    }


def _summarize_realized_placement(
    records: list[tuple[str, int]],
    *,
    expected_nodes: int,
    expected_gpus_per_node: int,
    expected_world_size: int,
) -> dict:
    """Validate private host/rank records and return only publication-safe aggregates."""
    if expected_nodes < 1 or expected_gpus_per_node < 1:
        raise ValueError("requested placement dimensions must be positive")
    if expected_nodes * expected_gpus_per_node != expected_world_size:
        raise ValueError("requested nodes x GPUs per node differs from world size")
    if len(records) != expected_world_size:
        raise ValueError("realized rank count differs from world size")

    by_host: dict[str, list[int]] = {}
    for host, local_rank in records:
        if not isinstance(host, str) or not host or type(local_rank) is not int:
            raise ValueError("realized placement record has invalid types")
        by_host.setdefault(host, []).append(local_rank)

    counts = sorted(len(local_ranks) for local_ranks in by_host.values())
    complete_local_ranks = all(
        sorted(local_ranks) == list(range(expected_gpus_per_node))
        for local_ranks in by_host.values()
    )
    unique_pairs = len(set(records)) == len(records)
    if len(by_host) != expected_nodes:
        raise ValueError(
            f"realized node count {len(by_host)} differs from requested {expected_nodes}"
        )
    if counts != [expected_gpus_per_node] * expected_nodes:
        raise ValueError("realized ranks per node differ from requested GPUs per node")
    if not complete_local_ranks or not unique_pairs:
        raise ValueError("realized local ranks are incomplete or duplicated")
    return {
        "gpus_per_node": expected_gpus_per_node,
        "nodes": expected_nodes,
        "ranks_per_node": expected_gpus_per_node,
        "unique_local_ranks": True,
        "valid": True,
    }


def _common_runtime_fingerprint(records: list[dict]) -> dict:
    """Return the shared sanitized fingerprint, rejecting heterogeneous ranks."""
    if not records:
        raise ValueError("runtime fingerprint evidence is empty")
    canonical = {
        json.dumps(record, allow_nan=False, sort_keys=True, separators=(",", ":"))
        for record in records
    }
    if len(canonical) != 1:
        raise ValueError("runtime fingerprint differs across distributed ranks")
    return records[0]


def _all_gather_json(
    value,
    *,
    torch_module,
    dist_module,
    device,
    world_size: int,
) -> list:
    """Gather bounded JSON values using device tensors, not Python object collectives."""
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if not payload or len(payload) > 1_048_576:
        raise ValueError("distributed consensus payload size is invalid")
    size = torch_module.tensor([len(payload)], dtype=torch_module.int64, device=device)
    sizes = [torch_module.zeros_like(size) for _ in range(world_size)]
    dist_module.all_gather(sizes, size)
    lengths = [int(item.item()) for item in sizes]
    if any(length < 1 or length > 1_048_576 for length in lengths):
        raise ValueError("distributed consensus size evidence is invalid")
    width = max(lengths)
    encoded = torch_module.zeros(width, dtype=torch_module.uint8, device=device)
    encoded[: len(payload)] = torch_module.tensor(
        list(payload), dtype=torch_module.uint8, device=device
    )
    gathered = [torch_module.empty_like(encoded) for _ in range(world_size)]
    dist_module.all_gather(gathered, encoded)
    records = []
    for tensor, length in zip(gathered, lengths, strict=True):
        raw = bytes(tensor[:length].cpu().tolist())
        records.append(json.loads(raw.decode("utf-8")))
    return records


def _allocation_stratum_sha256(
    physical_hosts: list[str],
    *,
    audit_salt: str | None,
    fabric_selectors: dict[str, str | None],
    required: bool,
) -> str | None:
    """Commit private allocation/fabric identity without exposing its inputs."""
    if audit_salt in (None, ""):
        if required:
            raise ValueError("canonical execution requires a private allocation audit salt")
        return None
    if not isinstance(audit_salt, str) or not re.fullmatch(r"[0-9a-f]{64}", audit_salt):
        raise ValueError("allocation audit salt is invalid")
    if set(fabric_selectors) != set(PRIVATE_FABRIC_ENV):
        raise ValueError("private fabric selector set differs from the stratum contract")
    for value in fabric_selectors.values():
        if value is not None and (
            not isinstance(value, str)
            or not value
            or len(value) > 512
            or any(ord(char) < 32 or ord(char) == 127 for char in value)
        ):
            raise ValueError("private fabric selector is invalid")
    if not physical_hosts or any(
        not isinstance(host, str)
        or not host
        or len(host) > 255
        or any(ord(char) < 32 or ord(char) == 127 for char in host)
        for host in physical_hosts
    ):
        raise ValueError("physical allocation host evidence is invalid")
    payload = json.dumps(
        {
            "contract": ALLOCATION_STRATUM_CONTRACT,
            "fabric_selectors": fabric_selectors,
            "physical_hosts": sorted(set(physical_hosts)),
        },
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hmac.new(bytes.fromhex(audit_salt), payload, hashlib.sha256).hexdigest()


def _common_allocation_stratum(
    records: list[str | None], *, required: bool
) -> str | None:
    """Require every distributed rank to derive the same private stratum."""
    if not records or any(
        value is not None
        and (not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value))
        for value in records
    ):
        raise ValueError("allocation stratum evidence is invalid")
    distinct = set(records)
    if len(distinct) != 1:
        raise ValueError("allocation stratum differs across distributed ranks")
    value = records[0]
    if required and value is None:
        raise ValueError("canonical execution requires an allocation stratum")
    return value


def main() -> int:
    ap = argparse.ArgumentParser(description="CollectiveX EP dispatch/combine sweep")
    ap.add_argument(
        "--backend",
        required=True,
        choices=[
            "deepep",
            "deepep-v2",
            "deepep-hybrid",
            "mori",
            "uccl",
            "nccl-ep",
        ],
    )
    ep_harness.add_common_args(ap)
    args = ap.parse_args()

    if args.mode == ep_harness.LOW_LATENCY_MODE:
        if args.backend not in {"deepep", "uccl"}:
            print(
                "ERROR: low-latency mode is supported only by deepep and uccl",
                file=sys.stderr,
            )
            return 2
        if args.phase != "decode":
            print("ERROR: low-latency mode requires --phase decode", file=sys.stderr)
            return 2
    if args.case_id and not identity.is_typed_id(args.case_id, "case"):
        print(f"ERROR: invalid native case ID {args.case_id!r}", file=sys.stderr)
        return 2
    if args.case_id and args.seed != ep_harness.ROUTING_SEED:
        print(
            f"ERROR: scheduled v1 cases require seed={ep_harness.ROUTING_SEED}; got {args.seed}",
            file=sys.stderr,
        )
        return 2
    if args.qualification_index not in range(1, ep_harness.QUALIFICATION_RUNS + 1):
        print(
            f"ERROR: qualification index must be in 1..{ep_harness.QUALIFICATION_RUNS}",
            file=sys.stderr,
        )
        return 2

    sampling_error = ep_harness.sampling_contract_error(
        args.iters, args.trials, args.warmup
    )
    if sampling_error:
        print(f"ERROR: {sampling_error}", file=sys.stderr)
        return 2

    try:
        import torch
        import torch.distributed as dist
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: torch unavailable: {exc!r}", file=sys.stderr)
        return 3

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")

    import capability

    sku = capability.PLATFORMS.get(args.runner)
    if sku is None:
        print(f"ERROR: unknown runner identity {args.runner!r}", file=sys.stderr)
        return 5
    machine = {"x86_64": "amd64", "aarch64": "arm64"}.get(
        platform.machine(), platform.machine()
    )
    props = torch.cuda.get_device_properties(device)
    if torch.version.hip:
        vendor = "amd"
        accelerator = str(getattr(props, "gcnArchName", "")).split(":", 1)[0]
    else:
        vendor = "nvidia"
        major, minor = torch.cuda.get_device_capability(device)
        accelerator = f"sm{major}{minor}"
    device_name = torch.cuda.get_device_name(device)
    device_count = torch.cuda.device_count()
    identity_issues = capability.runtime_identity_issues(
        args.runner,
        vendor=vendor,
        arch=accelerator,
        machine=machine,
        device_name=device_name,
        device_count=device_count,
        world_size=world_size,
    )
    if identity_issues:
        print(
            f"ERROR: runtime identity does not match {args.runner}: "
            + "; ".join(identity_issues),
            file=sys.stderr,
        )
        return 5
    observed_gpus_per_node = args.gpus_per_node or device_count
    if observed_gpus_per_node != sku["gpus_per_node"]:
        print(
            f"ERROR: {args.runner} requires {sku['gpus_per_node']} GPUs per node",
            file=sys.stderr,
        )
        return 5
    if world_size % observed_gpus_per_node:
        print("ERROR: distributed world is not divisible by GPUs per node", file=sys.stderr)
        return 5
    observed_nodes = world_size // observed_gpus_per_node
    topology = capability.topology_for(args.runner, world_size)
    observed_topology = {
        "nodes": observed_nodes,
        "gpus_per_node": observed_gpus_per_node,
        "scale_up_domain": args.scale_up_domain or observed_gpus_per_node,
        "scope": args.scope,
        "scale_up_transport": args.scale_up_transport,
        "scale_out_transport": args.scale_out_transport or None,
        "transport": args.transport,
        "topology_class": args.topology_class,
    }
    if topology is None or any(
        observed_topology[field] != topology[field] for field in observed_topology
    ):
        print(
            f"ERROR: runtime topology does not match {args.runner} EP{world_size}",
            file=sys.stderr,
        )
        return 5
    schedulable, reason = capability.resolve(
        args.runner,
        args.backend,
        ep=world_size,
        nodes=observed_nodes,
        routing=args.routing,
        eplb=False,
        mode=args.mode,
    )
    if not schedulable:
        print(f"ERROR: scheduled case is unsupported: {reason}", file=sys.stderr)
        return 5
    args.runtime_device_product = device_name
    args.runtime_device_count = device_count
    args.allocation_execution_id = os.environ.get("COLLECTIVEX_EXECUTION_ID")

    # Reproduction provenance (recorded in the artifact). Rack launchers provide ranks directly
    # through srun, while single-node launchers use torchrun; do not claim torchrun for both.
    if os.environ.get("TORCHELASTIC_RUN_ID"):
        args.distributed_launcher = "torchrun"
        prefix = f"torchrun --nproc_per_node={world_size}"
    else:
        args.distributed_launcher = "rank-environment"
        prefix = f"RANK={rank} WORLD_SIZE={world_size} LOCAL_RANK={local_rank} python3"
    args.reproduction_command = f"{prefix} tests/run_ep.py {shlex.join(sys.argv[1:])}"
    args.image = os.environ.get("COLLECTIVEX_IMAGE", "")
    args.image_digest = os.environ.get("COLLECTIVEX_IMAGE_DIGEST", "")
    args.image_digest_verified = (
        os.environ.get("COLLECTIVEX_IMAGE_DIGEST_VERIFIED") == "1"
    )
    # Container architecture and local squash hash for Enroot/Pyxis.
    args.image_arch = machine
    args.squash_sha256 = os.environ.get("COLLECTIVEX_SQUASH_SHA256")
    # GitHub provenance: repo, run ID, attempt, ref, source SHA, job,
    # artifact. A result is only publication-'official' when these are present (validity gate).
    _run = {
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "ref": os.environ.get("GITHUB_REF_NAME") or os.environ.get("GITHUB_REF"),
        "source_sha": os.environ.get("COLLECTIVEX_SOURCE_SHA")
        or os.environ.get("GITHUB_SHA"),
        "repo": os.environ.get("GITHUB_REPOSITORY"),
        "job": os.environ.get("GITHUB_JOB"),
        "artifact": os.environ.get("COLLECTIVEX_ARTIFACT_NAME"),
    }
    if any(_run.values()):
        _run["qualification_index"] = args.qualification_index
        args.git_run = _run
    else:
        args.git_run = None

    # Import the backend class only after torch initializes. The selected mode is an
    # explicit case dimension; adapters do not infer it from the token ladder.
    if args.backend == "mori":
        from ep_mori import MoRIBackend as Backend
    elif args.backend == "nccl-ep":
        from ep_nccl import NCCLBackend as Backend
    elif args.backend == "uccl":
        from ep_uccl import UCCLBackend as Backend
    elif args.backend == "deepep-hybrid":
        from ep_deepep_hybrid import DeepEPHybridBackend as Backend
    elif args.backend == "deepep-v2":
        from ep_deepep_v2 import DeepEPV2Backend as Backend
    else:
        from ep_deepep import DeepEPBackend as Backend

    # MoRI registers the default GPU process group with its SHMEM runtime. Keep that
    # group device-only so scale-out does not also depend on a host Gloo fabric.
    if not dist.is_initialized():
        if args.backend == "mori":
            dist.init_process_group(
                backend="nccl",
                rank=rank,
                world_size=world_size,
                device_id=device,
            )
        elif args.backend == "deepep-v2":
            # PR #605 reuses PyTorch's NCCL communicator through ``_comm_ptr``. Supplying
            # device_id eagerly forms it before ElasticBuffer construction.
            dist.init_process_group("nccl", device_id=device)
        else:
            dist.init_process_group("nccl")

    args.runtime_fingerprint = _runtime_fingerprint(
        torch, device, machine=machine, vendor=vendor, arch=accelerator
    )

    gpus_per_node = args.gpus_per_node or sku["gpus_per_node"]
    try:
        expected_nodes = int(
            os.environ.get("SLURM_NNODES", str(world_size // gpus_per_node))
        )
    except ValueError as exc:
        raise ValueError("SLURM_NNODES must be a positive integer") from exc
    realized_records = _all_gather_json(
        [socket.gethostname(), local_rank, args.runtime_fingerprint],
        torch_module=torch,
        dist_module=dist,
        device=device,
        world_size=world_size,
    )
    args.realized_placement = _summarize_realized_placement(
        [(record[0], record[1]) for record in realized_records],
        expected_nodes=expected_nodes,
        expected_gpus_per_node=gpus_per_node,
        expected_world_size=world_size,
    )
    args.runtime_fingerprint = _common_runtime_fingerprint(
        [record[2] for record in realized_records]
    )
    canonical = bool(args.workload_dir)
    local_stratum = _allocation_stratum_sha256(
        [record[0] for record in realized_records],
        audit_salt=os.environ.get("CX_AUDIT_SALT"),
        fabric_selectors={
            field: os.environ.get(environment) or None
            for field, environment in PRIVATE_FABRIC_ENV.items()
        },
        required=canonical,
    )
    stratum_records = _all_gather_json(
        local_stratum,
        torch_module=torch,
        dist_module=dist,
        device=device,
        world_size=world_size,
    )
    args.allocation_stratum_sha256 = _common_allocation_stratum(
        stratum_records, required=canonical
    )

    # Construct + run inside a try so a backend exception (esp. a new adapter on GPU) prints its
    # FULL traceback to STDOUT — torchrun captures per-rank stdout but only summarizes stderr, so an
    # uncaught exception is otherwise invisible in CI. Print on every rank (prefixed) then re-raise.
    try:
        backend = Backend(args, rank, world_size, local_rank, device)
        if rank == 0:
            print(
                f"[run_ep] backend={args.backend} phase={args.phase} mode={args.mode} "
                f"world={world_size} ep_size={world_size} hidden={args.hidden} "
                f"topk={args.topk} experts={args.experts} dtype=bf16 "
                f"routing={args.routing} seed={args.seed} "
                f"qualification_index={args.qualification_index}"
            )
        rc = ep_harness.run_sweep(args, backend, torch, dist, device, rank, world_size)
    except Exception:
        import traceback

        print(
            f"[run_ep][rank{rank}] backend={args.backend} FAILED:\n"
            + traceback.format_exc(),
            flush=True,
        )
        raise
    # finalize() handles backend-specific teardown: DeepEP returns rc cleanly;
    # MoRI hard-exits past its post-shmem_finalize teardown assertion.
    return backend.finalize(rc)


if __name__ == "__main__":
    raise SystemExit(main())
