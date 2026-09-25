#!/usr/bin/env python3
"""Stdlib node utilities, backend preparation, and the per-GPU process boundary."""

from __future__ import annotations

import argparse
from importlib import metadata
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import traceback

# This file runs both directly inside a container and from a stdlib zipapp on a compute host.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime import build, probe, storage
from runtime.scheduler import log, run
from runtime.config import RANK_ENV_VARS, DEEPEP_UNSETS


def package_root(package: str, component: str) -> str:
    """Find the requested wheel's actual nvidia/ component in this interpreter."""
    distribution = metadata.distribution(package)
    prefix = f"nvidia/{component}/"
    entries = [str(entry).replace("\\", "/") for entry in distribution.files or ()]
    root = Path(distribution.locate_file(PurePosixPath("nvidia") / component)).resolve()
    if not any(entry.startswith(prefix) for entry in entries) or not root.is_dir():
        raise ValueError(f"{package} component {component} is unavailable")
    return str(root)


def cuda_arch() -> str:
    """Query the allocated GPU in a short-lived process, before selecting a build cache."""
    import torch

    major, minor = torch.cuda.get_device_capability()
    return f"{major}.{minor}"


def check_deepep() -> None:
    import deep_ep

    if not inspect.isclass(deep_ep.ElasticBuffer):
        raise RuntimeError("DeepEP V2 import probe failed")


def check_uccl(*, docker: bool = False) -> None:
    import torch  # noqa: F401 - load libc10 before uccl.ep dlopens
    from deep_ep import Buffer

    if not hasattr(Buffer, "get_dispatch_layout") or (
        not docker and not hasattr(Buffer, "low_latency_dispatch")
    ):
        raise RuntimeError("UCCL import probe failed")


def check_nccl() -> None:
    import torch  # noqa: F401 - load libc10/libnccl before the bindings
    import nccl.core  # noqa: F401
    import nccl.ep

    print(
        f"nccl.ep: libnccl_ep {nccl.ep.get_lib_version()} at {nccl.ep.get_lib_path()}",
        file=sys.stderr,
    )


def check_flashinfer() -> None:
    import flashinfer
    from flashinfer.comm import Mapping  # noqa: F401
    from flashinfer.comm.mnnvl import MnnvlConfig  # noqa: F401
    from flashinfer.comm.trtllm_moe_alltoall import MoeAlltoAll, moe_a2a_get_workspace_size_per_rank  # noqa: F401

    print(f"FlashInfer {getattr(flashinfer, '__version__', 'unknown')} one-sided A2A available")


def check_mori() -> None:
    import mori  # noqa: F401


CHECKS = {
    "deepep-v2": check_deepep,
    "uccl-ep": check_uccl,
    "uccl-docker": lambda: check_uccl(docker=True),
    "nccl-ep": check_nccl,
    "mori": check_mori,
    "flashinfer-ep": check_flashinfer,
}


def prepare() -> None:
    """Prepare once per node, then publish rank settings as private data."""
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    env = dict(os.environ)
    backend, runner = env["COLLX_BENCH"], env["COLLX_RUNNER"]
    log(f"backend preparation: runner={runner} bench={backend} nodes={env.get('COLLX_NODES', '1')}")
    env = probe.network_environment(
        env, int(env.get("COLLX_NODES", "1")), env.get("COLLX_TRANSPORT", "")
    )
    probe.validate_container_network(env)
    active = build.prepare_backend(backend, runner, env)
    build.write_rank_environment(root, env.get("SLURM_NODEID", "0"), backend, active)
    log(f"backend preparation: bench={backend} rc=0")


def rank_environment(root: Path, env: dict[str, str]) -> dict[str, str]:
    """Load only backend-owned settings, then derive rank identity from Slurm itself."""
    node = env.get("SLURM_NODEID", "")
    if not re.fullmatch(r"[0-9]+", node):
        raise SystemExit(66)
    try:
        document = json.loads((root / ".collx_backend/env" / f"node-{node}.json").read_text())
        settings, unsets = document["set"], document["unset"]
        if (
            not isinstance(settings, dict)
            or not isinstance(unsets, list)
            or set(settings) - set(RANK_ENV_VARS)
            or set(unsets) - set(DEEPEP_UNSETS)
            or any(not isinstance(value, str) for value in settings.values())
        ):
            raise ValueError("invalid rank environment")
        result = {**env, **settings}
        for key in unsets:
            result.pop(key, None)
    except (OSError, KeyError, TypeError, ValueError):
        raise SystemExit(66) from None
    fields = ("SLURM_PROCID", "SLURM_NTASKS", "SLURM_LOCALID", "SLURM_NODEID")
    if any(not re.fullmatch(r"[0-9]+", result.get(key, "")) for key in fields):
        raise SystemExit(67)
    if result["SLURM_NTASKS"] != result.get("COLLX_NGPUS") or int(result["SLURM_LOCALID"]) >= int(
        result["COLLX_GPUS_PER_NODE"]
    ):
        raise SystemExit(67)
    if int(result.get("COLLX_NODES", "1")) > 1 and result.get("COLLX_TRANSPORT") != "mnnvl":
        try:
            if not result.get("COLLX_SOCKET_IFNAME"):
                result["COLLX_SOCKET_IFNAME"] = probe.default_route_interface()
                if not re.fullmatch(probe.INTERFACE, result["COLLX_SOCKET_IFNAME"]):
                    raise ValueError("invalid primary interface")
            # Slurm can strip NCCL_IB_HCA's '=' while exporting it. Reapply the exact selector
            # at the container boundary; an inherited prefix match can select the wrong rails.
            result = probe.network_environment(
                result, int(result["COLLX_NODES"]), result["COLLX_TRANSPORT"]
            )
        except (OSError, KeyError, ValueError):
            raise SystemExit(68) from None
    result.update(
        RANK=result["SLURM_PROCID"],
        WORLD_SIZE=result["SLURM_NTASKS"],
        LOCAL_RANK=result["SLURM_LOCALID"],
        LOCAL_WORLD_SIZE=result["COLLX_GPUS_PER_NODE"],
    )
    return result


def rank(arguments: list[str]) -> None:
    """Replace the bootstrap process with the unchanged benchmark entry point."""
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    root = Path(__file__).resolve().parents[1]
    env = rank_environment(root, dict(os.environ))
    os.chdir(root)
    os.execvpe("python3", ["python3", "bench/run_ep.py", *arguments], env)


def address(interface: str) -> str:
    """Prefer the validated primary network over a management-network hostname."""
    if interface:
        output = run(
            ["ip", "-o", "-4", "address", "show", "dev", interface, "scope", "global"]
        ).stdout
        lines = output.splitlines()
        return lines[0].split()[3].split("/")[0] if lines else ""
    return run(["hostname", "-s"]).stdout.splitlines()[0]


def network_profile(socket_names: str, rdma_devices: str, gid_index: str, fabric: str) -> None:
    try:
        probe.validate_network_profile(socket_names, rdma_devices, gid_index, fabric)
    except SystemExit as exc:
        if exc.code:
            probe._emit_fabric_inventory()
        raise


def main(argv: list[str] | None = None) -> int:
    """Dispatch explicit node operations without shell interpolation or vendor imports at startup."""
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    commands = {
        "import-image": (lambda options: storage.import_image(json.loads(options)), ("options",)),
        "network-profile": (
            network_profile,
            ("socket_names", "rdma_devices", "gid_index", "fabric"),
        ),
        "gpu-health": (probe.validate_gpu_health, ()),
        "cuda-context": (
            lambda expected: probe.validate_cuda_context(int(expected)),
            ("expected",),
        ),
        "address": (address, ("interface",)),
        "cuda-arch": (cuda_arch, ()),
        "package-root": (package_root, ("package", "component")),
        "check-backend": (lambda backend: CHECKS[backend](), ("backend",)),
        "prepare": (prepare, ()),
        "prepare-docker-uccl": (
            lambda arch: build.install_uccl(Path("/uccl_pfx"), arch, dict(os.environ), docker=True),
            ("arch",),
        ),
        "rank": (rank, ("arguments",)),
    }
    for name, (handler, names) in commands.items():
        command = subcommands.add_parser(name)
        command.set_defaults(handler=handler)
        for key in names:
            command.add_argument(key, nargs=argparse.REMAINDER if key == "arguments" else None)
    args = vars(parser.parse_args(argv))
    args.pop("command")
    handler = args.pop("handler")
    try:
        result = handler(**args)
        if result is not None:
            print(result)
        return 0
    except storage.ImportFailure as exc:
        log(f"ERROR: {exc}")
        return exc.status
    except Exception as exc:
        log(f"ERROR: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
