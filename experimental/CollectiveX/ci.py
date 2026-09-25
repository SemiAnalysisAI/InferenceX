#!/usr/bin/env python3
"""CollectiveX workflow entry point: plan, extract, execute, finalize, and clean up."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

from runtime import execution
from runtime.scheduler import log


ROOT = Path(__file__).resolve().parent


def output(name: str, value: object, env: dict[str, str]) -> None:
    """Append a workflow output while leaving ordinary diagnostics on stdout/stderr."""
    with open(env["GITHUB_OUTPUT"], "a", encoding="utf-8") as stream:
        stream.write(f"{name}={value}\n")


def matrix(env: dict[str, str]) -> None:
    """Generate the same artifact and run-specific queue tokens as the original setup step."""
    from sweep_matrix import resolve_matrix
    from swap_matrix import build_matrix

    backend = env["INPUT_BACKEND"]
    sku, excluded = env.get("INPUT_ONLY_SKU", ""), env.get("INPUT_EXCLUDE_SKUS", "")
    ep, modes = env.get("INPUT_EP_SIZES", ""), env.get("INPUT_MODES", "")
    if backend == "swap-blocks":
        if ep or modes:
            raise ValueError("swap-blocks does not accept EP filters")
        platforms = json.loads((ROOT / "configs/platform_config.json").read_text())["platforms"]
        document = build_matrix(platforms, sku, excluded)
        serialized = json.dumps(document)
    else:
        document = resolve_matrix(
            backend=backend, only_sku=sku, exclude_skus=excluded, ep_sizes=ep, modes=modes
        )
        serialized = json.dumps(document, sort_keys=True, separators=(",", ":"))
        runnable = sum(item["disposition"] == "runnable" for item in document["requested_cases"])
        print(
            f"resolved {len(document['include'])} shard-cells, {runnable} runnable and "
            f"{len(document['requested_cases']) - runnable} unsupported cases",
            file=sys.stderr,
        )
    (ROOT / "matrix_full.json").write_text(serialized + "\n")
    # The artifact is written before queue tokens are added, preserving its reusable control
    # representation. Only the execution matrix receives run/attempt-specific scheduler tokens.
    cells = document["include"]
    for index, cell in enumerate(cells):
        canonical = json.dumps(cell, sort_keys=True, separators=(",", ":"))
        material = f"{env['RUN_ID']}:{env['RUN_ATTEMPT']}:{index}:{canonical}".encode()
        cell["queue-token"] = hashlib.sha256(material).hexdigest()[:32]
    output("matrix", json.dumps({"include": cells}, separators=(",", ":")), env)
    output("n", len(cells), env)
    print(f"execution-cells: {len(cells)}")


def extract(env: dict[str, str]) -> None:
    from sweep_matrix import extract_shard

    extract_shard(
        str(Path(env["COLLX_JOB_ROOT"]) / "control/matrix_full.json"),
        env["MATRIX_ID"],
        str(ROOT / env["COLLX_SHARD_FILE"]),
    )


def execute(env: dict[str, str]) -> int:
    """Support workflow isolation and a private recovery directory for manual invocations."""
    if not env.get("COLLX_JOB_ROOT"):
        env = {
            **env,
            "COLLX_JOB_ROOT": tempfile.mkdtemp(
                prefix=f"inferencex-collectivex-0-0-manual-{os.getpid()}-", dir="/tmp"
            ),
        }
        log(f"execution-root={env['COLLX_JOB_ROOT']}")
    return execution.execute(ROOT.parent.parent, env)


def finalize(env: dict[str, str]) -> None:
    """Recover resources first, then render and stage every JSON emitted by the shard."""
    root, repo = Path(env["COLLX_JOB_ROOT"]), ROOT.parent.parent
    execution.cleanup(root, repo, env)
    results = ROOT / "results"
    if env["COLLX_BENCH"] != "swap-blocks":
        import bandwidth
        import summarize

        # These are best-effort views of raw artifacts. Benchmark return codes gate the leg;
        # a renderer failure must not suppress the raw documents, including failed outcomes.
        for renderer in (summarize, bandwidth):
            try:
                documents = summarize.load_results(str(results), None, None)
                if renderer is bandwidth:
                    documents = [doc for doc in documents if doc["outcome"]["status"] == "success"]
                with open(env["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as summary:
                    summary.write(renderer.render(documents) + "\n")
            except Exception as exc:
                log(f"summary unavailable: {exc}")
    files = sorted(results.glob("*.json"))
    if not files:
        output("staged", "false", env)
        print("No result JSON to stage; leg produced none.")
        return
    for path in files:
        shutil.copyfile(path, root / "artifact" / path.name)
    output("staged", "true", env)


def cleanup_workspace(env: dict[str, str]) -> None:
    """Delete only the workflow's validated private root after resource finalization."""
    root, parent = Path(env["COLLX_JOB_ROOT"]), Path(env["COLLX_JOB_PARENT"])
    execution.validate_job_root(root)
    if root.parent != parent or env["COLLX_SOURCE_ROOT"] != str(root / "source"):
        raise ValueError("CollectiveX cleanup source is invalid")
    if parent != Path("/tmp") and not re.fullmatch(execution.PARENT_NAME, parent.name):
        raise ValueError("CollectiveX cleanup parent is invalid")
    if (root / "jobid").exists():
        raise RuntimeError("allocation cleanup is incomplete; retaining the isolated workspace")
    shutil.rmtree(root)
    if parent != Path("/tmp"):
        parent.unlink()


COMMANDS = {
    "matrix": matrix,
    "extract": extract,
    "execute": execute,
    "finalize": finalize,
    "cleanup": cleanup_workspace,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    args = parser.parse_args(argv)
    try:
        return COMMANDS[args.command](dict(os.environ)) or 0
    except Exception as exc:
        log(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
