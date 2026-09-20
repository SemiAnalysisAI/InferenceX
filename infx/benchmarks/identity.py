"""Capture and verify prepared installed-client identity, with no dependency installation."""

from __future__ import annotations

import argparse
import json
import subprocess
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

from .common import child_environment, read_json, verify_file, write_json
from .spec import RuntimeSpec

AGENTX_REVISION = "754356e9a39acc6cc6afb242d123bb57c3fb6f75"
LM_EVAL_REVISION = "b315ef3b05176acc9732bb7fdec116abe1ecc476"


def require_source_revision(identity: dict[str, Any], name: str, revision: str) -> None:
    distribution = identity.get("distributions", {}).get(name, {})
    source = distribution.get("direct_url") or {}
    if source.get("dir_info", {}).get("editable"):
        raise ValueError(
            f"editable client installation is not an immutable prepared runtime: {name}"
        )
    if source.get("vcs_info", {}).get("commit_id") != revision:
        raise ValueError(f"{name} must be installed from immutable reviewed revision {revision}")


def validate_cache_manifests(runtime: RuntimeSpec, root: Path) -> None:
    """Use the selected client's real schemas, including its nested dataset metadata."""
    manifests = sorted(root.glob("*/manifest.json"))
    if not manifests:
        raise ValueError("client produced no mmap manifest")
    script = (
        "import sys; from pathlib import Path; "
        "from aiperf.dataset.mmap_cache import CacheManifest, MANIFEST_VERSION; "
        "from aiperf.common.models import DatasetMetadata; "
        "\nfor value in sys.argv[1:]:\n"
        " manifest = CacheManifest.model_validate_json(Path(value).read_text())\n"
        " if manifest.version != MANIFEST_VERSION: raise ValueError('cache schema version differs')\n"
        " DatasetMetadata.model_validate_json(manifest.dataset_metadata_json)\n"
    )
    result = subprocess.run(
        [runtime.python, "-I", "-c", script, *(str(path) for path in manifests)],
        env=child_environment(runtime),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if result.returncode:
        raise ValueError("installed client rejected mmap manifest or nested dataset metadata")


def capture_identity(
    python: str,
    distributions: list[str],
    *,
    dataset_loader: str | None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    resource = files("infx.benchmarks").joinpath("identity_probe.py")
    with as_file(resource) as probe:
        argv = [python, "-I", str(probe)]
        for name in distributions:
            argv.extend(("--distribution", name))
        if dataset_loader is not None:
            argv.extend(("--dataset-loader", dataset_loader))
        result = subprocess.run(
            argv, env=env, capture_output=True, check=True, text=True, timeout=600
        )
    return json.loads(result.stdout)


def verify_runtime(
    runtime: RuntimeSpec, *, dataset_loader: str | None, source_pins: dict[str, str] | None = None
) -> dict[str, Any]:
    expected = read_json(verify_file(runtime.identity))
    for asset in runtime.assets:
        verify_file(asset)
    actual = capture_identity(
        runtime.python,
        runtime.distributions,
        dataset_loader=dataset_loader,
        env=child_environment(runtime),
    )
    if actual != expected:
        raise ValueError(
            "installed client/interpreter/plugin identity differs from prepared identity"
        )
    for name, revision in (source_pins or {}).items():
        require_source_revision(actual, name, revision)
    return actual


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True)
    parser.add_argument("--distribution", action="append", required=True)
    parser.add_argument("--dataset-loader")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(
        args.output,
        capture_identity(args.python, args.distribution, dataset_loader=args.dataset_loader),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
