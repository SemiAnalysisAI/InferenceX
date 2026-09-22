"""Prepare native SRT server/client commands for the existing Docker runner."""

from __future__ import annotations

import argparse
import os
import re
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from infx.srt_slurm.single_node import runtime_arguments, select_recipe
from infx.srt_slurm.synthetic_acceptance import build_overrides


def shell_command(command: list[str], environment: Mapping[str, str]) -> str:
    """Quote argv and literal recipe environment without embedding caller secrets."""
    lines = ["#!/usr/bin/env bash", "set -eo pipefail"]
    for key, value in environment.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"Invalid environment key: {key!r}")
        lines.append(f"export {key}={shlex.quote(value)}")
    lines.append(f"exec {shlex.join(command)}")
    return "\n".join(lines) + "\n"


def prepare(config: str, environment: Mapping[str, str]) -> tuple[str, str]:
    """Use the pinned SRT schema and backend builder; Docker stays pool-owned."""
    from srtctl.core.config import expand_engine_config_defaults, resolve_config_with_defaults
    from srtctl.core.overrides import apply_overrides_to_recipe, parse_overrides
    from srtctl.core.runtime import Nodes, RuntimeContext
    from srtctl.core.schema import SrtConfig
    from srtctl.core.topology import Process

    selected, recipe = select_recipe(config, environment)
    if environment["FRAMEWORK"] != "sglang":
        raise ValueError("The Docker runner currently supports native SGLang recipes only")
    local_model = environment.get("MODEL_PATH")
    port = int(environment["PORT"])
    if not 1 <= port <= 65535:
        raise ValueError("PORT must be between 1 and 65535")
    overrides = runtime_arguments(selected, environment)
    apply_overrides_to_recipe(recipe, parse_overrides(overrides[1::2], []))
    golden = build_overrides(recipe, environment["FRAMEWORK"], environment)
    # Fixed-sequence jobs remove any stale simulation flags, as native apply does.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--unset", action="append", default=[])
    parsed = parser.parse_args(golden)
    apply_overrides_to_recipe(recipe, parse_overrides(parsed.set, parsed.unset))
    resolved = resolve_config_with_defaults(recipe, {})
    expand_engine_config_defaults(resolved)
    native = SrtConfig.Schema().load(resolved)
    if (
        native.frontend.type != "sglang"
        or native.services
        or native.setup_script
        or native.host_setup.enabled
        or native.dynamo.sidecar
        or native.extra_mount
        or native.container_mounts
    ):
        raise ValueError(
            "Docker fixed-sequence recipes require one direct server without services or setup"
        )
    runtime = RuntimeContext(
        job_id="docker",
        run_name=native.name,
        nodes=Nodes(head="127.0.0.1", bench="127.0.0.1", infra="127.0.0.1", worker=("127.0.0.1",)),
        head_node_ip="127.0.0.1",
        infra_node_ip="127.0.0.1",
        log_dir=Path("/logs"),
        model_path=Path(local_model or environment["MODEL"]),
        container_image=Path(environment["IMAGE"]),
        gpus_per_node=native.resources.gpus_per_node,
        network_interface=None,
        # Docker already exposes MODEL_PATH through its existing mounts. Use
        # the native builder's literal path mode, without Slurm's /model mount.
        is_hf_model=True,
        frontend_port=port,
    )
    process = Process(
        node="127.0.0.1",
        gpu_indices=frozenset(range(int(environment["GPU_COUNT"]))),
        sys_port=port,
        http_port=port,
        endpoint_mode="agg",
        endpoint_index=0,
    )
    server = native.backend.build_worker_command(
        process, [process], runtime, frontend_type="sglang"
    )
    server_env = {**native.backend.get_environment_for_mode("agg"), **native.environment}
    benchmark: dict[str, Any] = recipe["benchmark"]
    client_env = {
        **benchmark["env"],
        "SRT_FRONTEND_HOST": "127.0.0.1",
        "SRT_FRONTEND_PORT": str(port),
    }
    return shell_command(server, server_env), shell_command(
        shlex.split(benchmark["command"]), client_env
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    server, client = prepare(args.recipe, os.environ)
    (args.output / "srt-docker-server.sh").write_text(server)
    (args.output / "srt-docker-client.sh").write_text(client)


if __name__ == "__main__":
    main()
