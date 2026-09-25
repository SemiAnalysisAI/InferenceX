#!/usr/bin/env python3
"""Load private runner settings, the public backend registry, and shard controls."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import json
import os
import sys


OPERATOR_FIELDS = {
    "partition",
    "account",
    "qos",
    "squash_dir",
    "stage_dir",
    "enroot_cache_path",
    "exclude_nodes",
    "nodelist",
    "lock_dir",
}
NETWORK_FIELDS = {
    "socket_ifname",
    "rdma_devices",
    "ib_gid_index",
    "rdma_service_level",
    "rdma_traffic_class",
    "rail_isolated",
    "single_node_rdma_devices",
    "rdma_fabric",
}
# Timing knobs, in the order the legacy colon-string encoded them, paired with the run_ep flag
# each one drives. The names match configs/sweep.json so a case's timing block is readable
# rather than positional.
_TIMING_FLAGS = (
    ("iters_per_trial", "--iters"),
    ("trials_per_point", "--trials"),
    ("warmup_iters_per_trial", "--warmup"),
    ("chain_iters_per_trial", "--chain-iters"),
    ("chain_trials_per_point", "--chain-trials"),
    ("chain_drop", "--chain-drop"),
)


def _migrate_timing(timing: object) -> dict:
    """The one place a legacy colon-string timing profile is decoded.

    Current shards carry an object keyed by the _TIMING_FLAGS names. A replayed pre-chain
    shard carries "iters:trials:warmup" and a replayed post-chain one all six, positionally;
    three fields emit no --chain-* flags, so run_ep's argparse defaults supply them rather
    than this file duplicating the values. Anything else fails closed.

    Worth knowing when replaying an old shard: timing was never part of case_id, so a
    pre-chain case re-run today lands under its original identity while measured with
    today's chain budget.
    """
    names = tuple(key for key, _ in _TIMING_FLAGS)
    if isinstance(timing, dict):
        if set(timing) not in (set(names), set(names[:3])):
            print(f"unrecognised timing object {timing!r}", file=sys.stderr)
            raise SystemExit(1)
        return timing
    fields = str(timing).split(":")
    if len(fields) not in (3, 6):
        print(f"unrecognised timing profile {timing!r}", file=sys.stderr)
        raise SystemExit(1)
    return dict(zip(names, fields))


def _platforms() -> dict:
    """The per-SKU platform registry (configs/platform_config.json). Callers
    fail closed on a missing file, unknown SKU, or missing field."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs",
        "platform_config.json",
    )
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)["platforms"]


def _network_overlay(runner: str) -> dict[str, object]:
    """Repo-tracked per-SKU scale-out RDMA selectors — the `network` block of the
    SKU's configs/platform_config.json entry — overlaid onto the base operator
    config. Only NETWORK_FIELDS are taken, so identity keys and notes are ignored;
    a missing/invalid file is a no-op fallback to the base/secret network fields."""
    try:
        block = _platforms().get(runner, {}).get("network", {})
    except (KeyError, OSError, TypeError, json.JSONDecodeError):
        return {}
    return {key: value for key, value in block.items() if key in NETWORK_FIELDS}


def operator_values(path: str, runner: str) -> dict:
    """Resolve registry, operator overrides, and tracked network selectors as data."""
    try:
        platform = _platforms()[runner]
        # The registry's tracked per-SKU `operator` block is the baseline
        # (de-secreted by operator decision); an operator config document, when
        # provided, overrides it per field. Path "-" means registry-only.
        selected = dict(platform.get("operator", {}))
        if path != "-":
            with open(path, encoding="utf-8") as stream:
                document = json.load(stream)
            selected.update(document["runners"].get(runner, {}))
        # Overlay repo-tracked scale-out RDMA selectors onto the base runner config;
        # SKUs without a platform_config.json network block keep their base/secret
        # network fields.
        selected.update(_network_overlay(runner))
        allowed = OPERATOR_FIELDS | NETWORK_FIELDS | {"storage_roots"}
        if set(selected) - allowed:
            raise ValueError
        roots = selected.pop("storage_roots", None)
        if roots:
            for root in roots:
                squash = os.path.join(root, "collectivex", "containers")
                stage = os.path.join(root, "collectivex", "stage")
                try:
                    os.makedirs(squash, mode=0o700, exist_ok=True)
                    os.makedirs(stage, mode=0o700, exist_ok=True)
                    selected.update(squash_dir=squash, stage_dir=stage)
                    break
                except OSError:
                    continue
            else:
                raise ValueError
        if any(
            not isinstance(value, (str, int)) or "\0" in str(value) for value in selected.values()
        ):
            raise ValueError
        selected.update(image=platform["image"], image_platform=platform["image_platform"])
        return selected
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        print("validation-invalid-config", file=sys.stderr)
        raise SystemExit(1)


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def case_argv(case: dict, version: object, runner: str, ts: str, index: int) -> list[str]:
    """Build the benchmark argument list without a temporary file or shell decoder."""
    get = lambda key, default="": str(case.get(key) or default)
    argv = [
        "--backend",
        str(case["backend"]),
        "--mode",
        str(case["mode"]),
        "--precision",
        str(case["precision"]),
        "--phase",
        str(case["phase"]),
        "--routing",
        str(case["routing"]),
        "--gpus-per-node",
        str(case["gpus_per_node"]),
        "--scale-up-domain",
        str(case["scale_up_domain"]),
        "--scope",
        str(case["scope"]),
        "--scale-up-transport",
        str(case["scale_up_transport"]),
        "--scale-out-transport",
        get("scale_out_transport"),
        "--tokens-ladder",
        str(case["ladder"]),
        "--hidden",
        str(case["hidden"]),
        "--topk",
        str(case["topk"]),
        "--experts",
        str(case["experts"]),
        "--seed",
        str(case["seed"]),
        "--runner",
        runner,
        "--topology-class",
        str(case["topology_class"]),
        "--transport",
        str(case["transport"]),
        "--case-id",
        str(case["case_id"]),
        "--suite",
        str(case["suite"]),
        "--workload-name",
        str(case["workload"]),
        "--version",
        str(version),
    ]
    timing = _migrate_timing(case["timing"])
    for key, flag in _TIMING_FLAGS:
        if key in timing:
            argv += [flag, str(timing[key])]
    # case_id is the canonical identity (sku==runner, backend, workload, mode, phase, ep, routing,
    # precision), so a new identity axis cannot be omitted from the filename the way mode once was.
    # ts + the per-shard case index disambiguate legs that share one results/ directory.
    out = f"results/{case['case_id']}_{ts}-c{index:03d}.json"
    argv += ["--out", out]
    return argv


def case_arguments(
    document: dict, index: int, runner: str, timestamp: str, placement: tuple[str, str, str, str]
) -> list[str]:
    """Validate an allocation's placement and return its benchmark argv directly."""
    cases = document["cases"]
    if not 0 <= index < len(cases):
        raise SystemExit(1)
    case = cases[index]
    observed = tuple(
        str(case.get(field, "")) for field in ("ep", "nodes", "gpus_per_node", "scale_up_domain")
    )
    if observed != placement:
        print(f"case placement {observed} differs from the allocation", file=sys.stderr)
        raise SystemExit(1)
    return case_argv(case, document["version"], runner, timestamp, index)


# Fresh rank tasks receive only these backend-created settings. Network configuration is
# applied separately at the rank boundary, preserving Slurm's exact HCA selector handling.
RANK_ENV_VARS = (
    "PATH",
    "VIRTUAL_ENV",
    "LD_LIBRARY_PATH",
    "PYTHONPATH",
    "CUDA_HOME",
    "CPATH",
    "NVCC_PREPEND_FLAGS",
    "NVSHMEM_DIR",
    "EP_NCCL_ROOT_DIR",
    "EP_NVSHMEM_ROOT_DIR",
    "EP_JIT_CACHE_DIR",
    "EP_REUSE_NCCL_COMM",
    "NCCL_CUMEM_ENABLE",
    "UCCL_EP_ENABLE_AGGRESSIVE_ATOMIC",
)
DEEPEP_UNSETS = ("EP_SUPPRESS_NCCL_CHECK",)


@dataclass(frozen=True)
class Pool:
    kind: str
    default_time: int
    account: bool = False
    memory_all: bool = False
    remap_root: bool = False
    mpi_none: bool = False
    cpus: int = 0
    devices: bool = False


POOLS = {
    "h100-dgxc": Pool("nvidia", 45, account=True),
    "h200-dgxc": Pool("nvidia", 45, remap_root=True),
    # Bare-metal B200's native IB + gdrdrv enables LL EP16. Retain its full-node memory
    # request and the manual 45-minute budget that covers first-run backend builds.
    "b200-nscale": Pool("nvidia", 45, account=True, memory_all=True),
    "b300": Pool("nvidia", 45, account=True, memory_all=True, remap_root=True, mpi_none=True),
    "gb200": Pool("gb", 30, account=True, memory_all=True, remap_root=True, cpus=35),
    "gb300": Pool("gb", 90, account=True, memory_all=True, remap_root=True, cpus=35),
    "mi300x": Pool("amd", 60, cpus=256, devices=True, remap_root=True),
    "mi325x": Pool("amd", 60, cpus=256, devices=True, remap_root=True),
    "mi355x": Pool("amd", 60, cpus=128, remap_root=True),
    "mi300x-tw": Pool("docker", 0),
    "mi325x-tw": Pool("docker", 0),
}
BACKENDS = {
    "nvidia": ("deepep-v2", "uccl-ep", "nccl-ep"),
    "gb": ("deepep-v2", "nccl-ep", "flashinfer-ep"),
    "amd": ("mori", "uccl-ep"),
    "docker": ("mori", "uccl-ep"),
}
CONFIG_UNSETS = {f"COLLX_{name.upper()}" for name in OPERATOR_FIELDS | NETWORK_FIELDS} | {
    "COLLX_IMAGE",
    "COLLX_IMAGE_PLATFORM",
    "COLLX_MASTER_PORT",
    "ENROOT_CACHE_PATH",
    "MASTER_ADDR",
    "MASTER_PORT",
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
}


def require(env: dict[str, str], *names: str) -> None:
    """Reject missing and empty inputs at the Python caller boundary."""
    missing = [name for name in names if not env.get(name)]
    if missing:
        raise ValueError("missing platform or runner configuration: " + " ".join(missing))


def configured_environment(runner: str, incoming: dict[str, str]) -> dict[str, str]:
    """Replace caller network/operator state using the same registry/override precedence."""
    env = {name: value for name, value in incoming.items() if name not in CONFIG_UNSETS}
    default = (
        Path(incoming.get("XDG_CONFIG_HOME") or Path(incoming["HOME"]) / ".config")
        / "inferencex/collectivex.json"
    )
    path = Path(incoming.get("COLLECTIVEX_OPERATOR_CONFIG") or default)
    selected = operator_values(str(path) if path.exists() else "-", runner)
    env.update({f"COLLX_{key.upper()}": str(value) for key, value in selected.items()})
    env.pop("COLLECTIVEX_OPERATOR_CONFIG", None)
    return env


@dataclass
class Plan:
    runner: str
    backend: str
    pool: Pool
    env: dict[str, str]
    nodes: int
    gpus: int
    world: int
    domain: int
    minutes: int
    transport: str

    @property
    def swap(self) -> bool:
        return self.backend == "swap-blocks"

    def allocation_args(self, excluded: str = "") -> list[str]:
        """Preserve each pool's resource request and nodelist/exclude precedence."""
        env, kind = self.env, self.pool.kind
        args = [
            f"--partition={env['COLLX_PARTITION']}",
            f"--nodes={self.nodes}",
            f"--gres=gpu:{self.gpus}",
            f"--ntasks-per-node={self.gpus}",
            f"--time={self.minutes}",
        ]
        if self.swap or kind in ("nvidia", "gb") or self.runner == "mi355x":
            args.append("--exclusive")
        if not self.swap and self.pool.memory_all:
            args.append("--mem=0")
        if not self.swap and self.pool.cpus:
            cpus = self.pool.cpus // self.gpus if kind == "amd" else self.pool.cpus
            args.append(f"--cpus-per-task={cpus}")
        fields = (
            ("account", "qos", "nodelist")
            if self.swap or kind == "nvidia"
            else ("account",)
            if kind == "gb"
            else ()
        )
        for field in fields:
            if env.get(f"COLLX_{field.upper()}"):
                args.append(f"--{field}={env[f'COLLX_{field.upper()}']}")
        if not self.swap and kind == "amd" and env.get("COLLX_NODELIST"):
            args.append(f"--nodelist={env['COLLX_NODELIST']}")
        elif excluded:
            args.append(f"--exclude={excluded}")
        return args

    def container_options(self, source: Path, image: Path, job_id: str, cache: str = "") -> dict:
        """One Pyxis option model for preparation, ranks, and block copies."""
        mounts = [f"{source}:/ix"]
        if self.pool.devices:
            mounts += ["/dev/kfd:/dev/kfd", "/dev/dri:/dev/dri"]
        if cache:
            mounts.append(f"{cache}:/cx-cache")
        return {
            "container_mounts": ",".join(mounts),
            "no_container_mount_home": True,
            "container_workdir": "/ix/experimental/CollectiveX",
            "no_container_entrypoint": True,
            "container_name": f"cxep_{job_id}",
            "container_image": str(image),
            "container_writable": True,
            "container_remap_root": self.swap or self.pool.remap_root,
            "mpi": "none" if self.pool.mpi_none and not self.swap else None,
        }


def make_plan(runner: str, backend: str, incoming: dict[str, str]) -> Plan:
    """Resolve explicit workflow inputs and retain the historical manual-launch defaults."""
    from .probe import network_environment
    from .storage import prepare_stage_dir

    pool = POOLS[runner]
    amd = pool.kind in ("amd", "docker")
    backend = backend or ("mori" if amd else "deepep-v2")
    if backend != "swap-blocks" and backend not in BACKENDS[pool.kind]:
        raise ValueError(f"unsupported {runner} EP backend: {backend}")
    env = configured_environment(runner, incoming)
    env.update(COLLX_RUNNER=runner, COLLX_BENCH=backend, COLLX_VENDOR="amd" if amd else "nvidia")
    swap = backend == "swap-blocks"
    if not swap:
        require(env, "COLLX_SHARD_FILE")
    if pool.kind != "docker":
        env = prepare_stage_dir(runner, env)
    nodes = int(env.get("COLLX_NODES") or (2 if pool.kind == "gb" else 1))
    gpus = int(env.get("COLLX_GPUS_PER_NODE") or (4 if pool.kind == "gb" else 8))
    domain = int(env.get("COLLX_SCALE_UP_DOMAIN") or (72 if pool.kind == "gb" else 8))
    world = (
        nodes * gpus
        if swap or pool.kind == "docker"
        else int(env.get("COLLX_NGPUS") or nodes * gpus)
    )
    if nodes < 1 or world != nodes * gpus:
        raise ValueError("invalid shard launcher placement")
    if pool.kind == "docker" and nodes != 1:
        raise ValueError("the -tw AMD pools are single-node scale-up only")
    if swap and (nodes != 1 or gpus != 1):
        raise ValueError("swap-blocks requires one GPU process")
    transport = "mnnvl" if pool.kind == "gb" else "xgmi" if amd else "nvlink"
    if nodes > 1 and transport != "mnnvl":
        transport += "-rdma"
    env.update(
        COLLX_NGPUS=str(world),
        COLLX_NODES=str(nodes),
        COLLX_GPUS_PER_NODE=str(gpus),
        COLLX_SCALE_UP_DOMAIN=str(domain),
        COLLX_TRANSPORT=transport,
    )
    if pool.kind == "amd" and not swap:
        for name, default in {
            "MORI_DISABLE_AUTO_XGMI": "0",
            "MORI_ENABLE_SDMA": "1",
            "MORI_APP_LOG_LEVEL": "info",
            "MORI_SHMEM_LOG_LEVEL": "info",
            "MORI_IO_LOG_LEVEL": "info",
        }.items():
            env[name] = env.get(name) or default
    if not amd and not swap:
        env["NCCL_CUMEM_ENABLE"] = "1"
    if pool.kind == "gb" and not swap:
        env.update(NCCL_MNNVL_ENABLE="1", MC_FORCE_MNNVL="1")
    if pool.kind != "docker" and not swap:
        env = network_environment(env, nodes, transport)
    require(env, "COLLX_IMAGE")
    if pool.kind != "docker":
        require(env, "COLLX_IMAGE_PLATFORM", "COLLX_PARTITION", "COLLX_SQUASH_DIR")
        if pool.account and not swap:
            require(env, "COLLX_ACCOUNT")
        if swap or pool.kind in ("amd", "gb") or runner in ("h100-dgxc", "b300"):
            require(env, "COLLX_STAGE_DIR")
        if runner == "gb300" and not swap:
            require(env, "COLLX_ENROOT_CACHE_PATH")
        if env.get("COLLX_ENROOT_CACHE_PATH") and (swap or pool.kind == "gb"):
            env["ENROOT_CACHE_PATH"] = env["COLLX_ENROOT_CACHE_PATH"]
    minutes = (
        int(env["COLLX_SWAP_TIME"]) if swap else int(env.get("COLLX_TIME") or pool.default_time)
    )
    return Plan(runner, backend, pool, env, nodes, gpus, world, domain, minutes, transport)


def validate_swap_environment(env: dict[str, str]) -> None:
    """Retain the swap launcher's required inputs and positive integer guards."""
    require(
        env,
        "COLLX_SHARD_SKU",
        "COLLX_NODES",
        "COLLX_GPUS_PER_NODE",
        "COLLX_SWAP_IMAGE",
        "COLLX_SWAP_MAX_PAYLOAD_BYTES",
        "COLLX_SWAP_BLOCK_BYTES",
        "COLLX_SWAP_NUM_BLOCKS",
        "COLLX_SWAP_WARMUP",
        "COLLX_SWAP_ITERATIONS",
        "COLLX_SWAP_SEED",
        "COLLX_SWAP_DEVICE",
        "COLLX_SWAP_TIME",
        "COLLX_JOB_ROOT",
        "COLLECTIVEX_SOURCE_SHA",
        "COLLECTIVEX_EXECUTION_ID",
        "COLLECTIVEX_CANONICAL_GHA",
        "COLLX_VENDOR",
        "COLLX_IMAGE_REFRESH",
    )
    if not re.fullmatch(r"vllm/vllm-openai(-rocm)?:[A-Za-z0-9._-]+", env["COLLX_SWAP_IMAGE"]):
        raise ValueError("swap-blocks requires a tagged official vLLM image")
    for name in ("COLLX_SWAP_BLOCK_BYTES", "COLLX_SWAP_NUM_BLOCKS"):
        if not re.fullmatch(r"[1-9][0-9]*( [1-9][0-9]*)*", env[name]):
            raise ValueError("block sizes and counts must be space-separated positive integers")
    for name in ("COLLX_SWAP_ITERATIONS", "COLLX_SWAP_TIME", "COLLX_SWAP_MAX_PAYLOAD_BYTES"):
        if not re.fullmatch(r"[1-9][0-9]*", env[name]):
            raise ValueError("iterations, time, and payload budget must be positive integers")
    for name in ("COLLX_SWAP_WARMUP", "COLLX_SWAP_SEED", "COLLX_SWAP_DEVICE"):
        if not re.fullmatch(r"[0-9]+", env[name]):
            raise ValueError("warmup, seed, and device must be non-negative integers")


def swap_arguments(env: dict[str, str], layout: str) -> list[str]:
    """Build the unchanged block-copy workload and output filename."""
    return [
        "bench/run_swap_blocks.py",
        "--directions",
        "h2d",
        "d2h",
        "d2d",
        "--block-bytes",
        *env["COLLX_SWAP_BLOCK_BYTES"].split(),
        "--num-blocks",
        *env["COLLX_SWAP_NUM_BLOCKS"].split(),
        "--layout",
        layout,
        "--seed",
        env["COLLX_SWAP_SEED"],
        "--device",
        env["COLLX_SWAP_DEVICE"],
        "--max-payload-bytes",
        env["COLLX_SWAP_MAX_PAYLOAD_BYTES"],
        "--warmup",
        env["COLLX_SWAP_WARMUP"],
        "--iterations",
        env["COLLX_SWAP_ITERATIONS"],
        "--output",
        f"results/swap-blocks-{layout}.json",
    ]
