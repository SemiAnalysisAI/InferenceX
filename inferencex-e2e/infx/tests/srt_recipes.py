"""Launchable srt-slurm recipes for tests that plan or fingerprint matrix rows."""

from pathlib import Path

import yaml

from infx.srt_slurm.workload import SHARED_BLOCKS, TELEMETRY_BLOCK


def shared_blocks() -> dict[str, str]:
    """Each lane's shared block (its benchmark client) and the DCGM telemetry block, by path."""
    blocks = {
        block.as_posix(): yaml.safe_dump({"benchmark": {"type": "custom", "command": (
            "bash srt_agentic.sh" if agentic else f"bash {'multi' if multinode else 'single'}.sh"
        )}})
        for (agentic, multinode), block in SHARED_BLOCKS.items()
    }  # fmt: skip
    blocks[TELEMETRY_BLOCK.as_posix()] = yaml.safe_dump(
        {"telemetry": {"enabled": True, "dcgm_exporter": {"container_image": "dcgm"}}}
    )
    return blocks


def write_shared_blocks(project: Path) -> None:
    for path, text in shared_blocks().items():
        (project / path).parent.mkdir(parents=True, exist_ok=True)
        (project / path).write_text(text)


def single_node_fragment(tp: int, **args) -> dict:
    """A single-node sglang fragment serving TP ``tp`` points without speculation."""
    return {"schema": 2, "engine": "sglang", "roles": {"agg": {
        "nodes": 1, "workers": 1, "gpus": tp, "args": {"tensor-parallel-size": tp, **args},
    }}}  # fmt: skip
