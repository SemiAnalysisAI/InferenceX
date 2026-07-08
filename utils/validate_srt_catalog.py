#!/usr/bin/env python3
"""Validate every configured srt-slurm recipe against the locked toolchain."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

import yaml

from prepare_srt_config import (
    MODEL_PATH_ALIAS,
    NGINX_CONTAINER_ALIAS,
    WORKLOAD_CONTAINER_ALIAS,
    prepare_config,
    resolve_recipe,
    split_config_arg,
)
from runner_config import load_config, resolve_model

try:
    from srtctl.core.schema import SrtConfig
except ImportError as error:  # pragma: no cover - exercised by the CI command
    raise SystemExit(
        "srtctl is not importable; add the pinned srt-slurm src directory "
        "to PYTHONPATH"
    ) from error


@dataclass(frozen=True)
class CatalogEntry:
    """One master-config search-space reference to an srt recipe."""

    config_key: str
    config_arg: str
    cluster: str
    model_prefix: str
    precision: str
    framework: str
    model: str
    isl: int | None
    osl: int | None
    concurrencies: str
    prefill: dict[str, Any]
    decode: dict[str, Any]


LAUNCHER_CLUSTERS = {
    "h100-dgxc-slurm": "h100-dgxc",
    "h200-dgxc-slurm": "h200-dgxc",
    "b200-dgxc": "b200-dgxc",
    "b200-dgxc-slurm": "b200-dgxc",
    "b300-nv": "b300-nv",
    "gb200-nv": "gb200-nv",
    "gb300-nv": "gb300-nv",
}


def config_args(value: Any) -> set[str]:
    """Find CONFIG_FILE assignments recursively within one search space."""
    if isinstance(value, dict):
        return set().union(*(config_args(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(config_args(item) for item in value))
    if isinstance(value, str) and value.startswith("CONFIG_FILE="):
        return {value.removeprefix("CONFIG_FILE=")}
    return set()


def cluster_for_runner(runner: str, runner_config: dict[str, Any]) -> str | None:
    """Resolve a master-config runner selector to one supported SRT cluster."""
    concrete_runners = runner_config["labels"].get(runner, [runner])
    matches = {
        cluster
        for launcher, cluster in LAUNCHER_CLUSTERS.items()
        if any(
            concrete.split("_", 1)[0] == launcher
            for concrete in concrete_runners
        )
    }
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(f"runner selector {runner!r} spans SRT clusters: {matches}")
    return matches.pop()


def load_catalog(
    master_path: Path,
    runner_config: dict[str, Any],
) -> list[CatalogEntry]:
    """Load every selected recipe occurrence with its expected topology."""
    with master_path.open(encoding="utf-8") as config_file:
        master = yaml.safe_load(config_file)

    entries: list[CatalogEntry] = []
    for config_key, config in master.items():
        if not config.get("multinode"):
            continue
        cluster = cluster_for_runner(config["runner"], runner_config)
        if cluster is None:
            continue
        for scenario_name, scenario_groups in config["scenarios"].items():
            for scenario_group in scenario_groups:
                for search_space in scenario_group["search-space"]:
                    references = config_args(search_space)
                    if not references:
                        continue
                    if len(references) != 1:
                        raise ValueError(
                            f"{config_key}: search-space entry has multiple "
                            f"CONFIG_FILE values: {sorted(references)}"
                        )
                    concurrencies = search_space.get("conc-list", [])
                    entries.append(
                        CatalogEntry(
                            config_key=config_key,
                            config_arg=references.pop(),
                            cluster=cluster,
                            model_prefix=config["model-prefix"],
                            precision=config["precision"],
                            framework=config["framework"],
                            model=config["model"],
                            isl=scenario_group.get("isl"),
                            osl=scenario_group.get("osl"),
                            concurrencies="x".join(
                                str(value) for value in concurrencies
                            ),
                            prefill=search_space["prefill"],
                            decode=search_space["decode"],
                        )
                    )
    return entries


def ensure_legacy_revision(srt_root: Path, revision: str) -> None:
    """Fetch the data-only legacy recipe snapshot when it is not present."""
    present = subprocess.run(
        ["git", "-C", str(srt_root), "cat-file", "-e", f"{revision}^{{commit}}"],
        check=False,
        capture_output=True,
    )
    if present.returncode == 0:
        return
    subprocess.run(
        ["git", "-C", str(srt_root), "fetch", "--depth", "1", "origin", revision],
        check=True,
    )


def stage_recipe(
    config_arg: str,
    *,
    repo_root: Path,
    srt_root: Path,
    legacy_revision: str,
) -> str:
    """Stage a local, main, or legacy recipe and retain its selector."""
    relative_path, selector = split_config_arg(config_arg)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"CONFIG_FILE must be repository-relative: {config_arg}")

    destination = srt_root / relative_path
    local_recipe = (
        repo_root
        / "benchmarks"
        / "multi_node"
        / "srt-slurm-recipes"
        / relative_path.relative_to("recipes")
    )
    if local_recipe.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_recipe, destination)
    elif not destination.is_file():
        ensure_legacy_revision(srt_root, legacy_revision)
        contents = subprocess.run(
            [
                "git",
                "-C",
                str(srt_root),
                "show",
                f"{legacy_revision}:{relative_path.as_posix()}",
            ],
            check=True,
            capture_output=True,
        ).stdout
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(contents)

    if not destination.is_file():
        raise ValueError(f"recipe did not resolve: {config_arg}")
    suffix = f":{selector}" if selector else ""
    return f"{destination}{suffix}"


def validate_topology(entry: CatalogEntry, config: SrtConfig) -> list[str]:
    """Compare recipe worker/GPU topology with the benchmark declaration."""
    errors: list[str] = []
    resources = config.resources
    if resources.is_disaggregated:
        expected = (
            (
                "prefill",
                entry.prefill,
                resources.num_prefill,
                resources.gpus_per_prefill,
            ),
            (
                "decode",
                entry.decode,
                resources.num_decode,
                resources.gpus_per_decode,
            ),
        )
    else:
        expected = (
            (
                "aggregate",
                entry.prefill,
                resources.num_agg,
                resources.gpus_per_agg,
            ),
        )
        if entry.decode["num-worker"] != 0:
            errors.append(
                "aggregate recipe requires master decode num-worker=0, got "
                f"{entry.decode['num-worker']}"
            )
    for role, declared, workers, gpus_per_worker in expected:
        if workers != declared["num-worker"]:
            errors.append(
                f"{role} workers: master={declared['num-worker']} recipe={workers}"
            )
        declared_world_size = max(declared["tp"], declared.get("ep", 1))
        if gpus_per_worker != declared_world_size:
            errors.append(
                f"{role} GPUs/worker: master world-size={declared_world_size} "
                f"recipe={gpus_per_worker}"
            )
    return errors


def validate_catalog(
    *,
    repo_root: Path,
    srt_root: Path,
    master_path: Path,
    runner_config_path: Path,
) -> tuple[int, int]:
    """Stage, transform, and schema-check every catalog occurrence."""
    runner_config = load_config(runner_config_path)
    srt_settings = runner_config["srt-slurm"]
    actual_revision = subprocess.run(
        ["git", "-C", str(srt_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_revision != srt_settings["revision"]:
        raise ValueError(
            f"srt-slurm checkout is {actual_revision}, expected "
            f"{srt_settings['revision']}"
        )

    entries = load_catalog(master_path, runner_config)
    staged = {
        config_arg: stage_recipe(
            config_arg,
            repo_root=repo_root,
            srt_root=srt_root,
            legacy_revision=srt_settings["legacy-recipes-revision"],
        )
        for config_arg in {entry.config_arg for entry in entries}
    }

    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="inferencex-srt-catalog-") as tmp:
        output_root = Path(tmp)
        for index, entry in enumerate(entries):
            try:
                model_mapping = resolve_model(
                    runner_config,
                    entry.cluster,
                    entry.model_prefix,
                    entry.precision,
                    entry.framework,
                    entry.model,
                )
                _, source_recipe = resolve_recipe(staged[entry.config_arg])
                source_model_path = str(source_recipe["model"]["path"])
                local_model_name = Path(
                    model_mapping["MODEL_PATH"].rstrip("/")
                ).name
                output = prepare_config(
                    staged[entry.config_arg],
                    output=output_root / f"{index}.yaml",
                    isl=entry.isl,
                    osl=entry.osl,
                    concurrencies=entry.concurrencies,
                    random_range_ratio=0.8,
                    default_served_model_name=local_model_name,
                )
                config = SrtConfig.from_yaml(output)
                topology_errors = validate_topology(entry, config)

                if config.model.container != WORKLOAD_CONTAINER_ALIAS:
                    topology_errors.append(
                        f"workload container is {config.model.container!r}"
                    )
                if (
                    config.frontend.enable_multiple_frontends
                    and config.frontend.nginx_container != NGINX_CONTAINER_ALIAS
                ):
                    topology_errors.append(
                        f"nginx container is {config.frontend.nginx_container!r}"
                    )

                recipe_model = str(config.model.path)
                if (
                    not recipe_model.startswith("hf:")
                    and recipe_model != MODEL_PATH_ALIAS
                ):
                    topology_errors.append(
                        f"model alias recipe={recipe_model!r} "
                        f"expected={MODEL_PATH_ALIAS!r}"
                    )
                if str(config.benchmark.command).endswith("srt_benchmark.sh"):
                    served_name_fallback = (
                        Path(source_model_path.removeprefix("hf:")).name
                        if source_model_path.startswith("hf:")
                        else local_model_name
                    )
                    expected_served_name = config.backend.get_served_model_name(
                        served_name_fallback
                    )
                    actual_served_name = config.benchmark.env.get("MODEL_NAME")
                    if actual_served_name != expected_served_name:
                        topology_errors.append(
                            f"served model benchmark={actual_served_name!r} "
                            f"server={expected_served_name!r}"
                        )

                if topology_errors:
                    failures.append(
                        f"{entry.config_key} [{entry.config_arg}]: "
                        + "; ".join(topology_errors)
                    )
            except Exception as error:  # report the full catalog in one run
                failures.append(
                    f"{entry.config_key} [{entry.config_arg}]: "
                    f"{type(error).__name__}: {error}"
                )

    if failures:
        raise ValueError(
            f"srt-slurm catalog has {len(failures)} invalid entries:\n"
            + "\n".join(failures)
        )
    return len(entries), len(staged)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--srt-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--master-config",
        type=Path,
        default=Path("configs/nvidia-master.yaml"),
    )
    parser.add_argument(
        "--runner-config",
        type=Path,
        default=Path("configs/runners.yaml"),
    )
    args = parser.parse_args()

    occurrences, unique = validate_catalog(
        repo_root=args.repo_root.resolve(),
        srt_root=args.srt_root.resolve(),
        master_path=args.master_config.resolve(),
        runner_config_path=args.runner_config.resolve(),
    )
    print(
        f"Validated {occurrences} srt-slurm catalog occurrences "
        f"across {unique} unique CONFIG_FILE values"
    )


if __name__ == "__main__":
    main()
