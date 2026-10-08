"""Launchable srt-slurm recipes for tests that plan or fingerprint matrix rows.

The planner fingerprints a row by selecting and binding its recipe as the launcher would, so
planned fixtures need recipes that serve their points.
"""

from pathlib import Path

import yaml

from infx.srt_slurm.workload import SHARED_BLOCKS


def shared_blocks() -> dict[str, str]:
    """Each fixed-sequence lane's shared block, by path: its benchmark client."""
    return {
        block.as_posix(): yaml.safe_dump({"benchmark": {
            "type": "custom", "command": f"bash {'multi' if multinode else 'single'}.sh",
        }})
        for multinode, block in SHARED_BLOCKS.items()
    }  # fmt: skip


def write_shared_blocks(project: Path) -> None:
    for path, text in shared_blocks().items():
        (project / path).parent.mkdir(parents=True, exist_ok=True)
        (project / path).write_text(text)


def single_node_fragment(tp: int, **args) -> dict:
    """A fixed-sequence sglang fragment serving TP ``tp`` points without speculation."""
    return {"schema": 2, "engine": "sglang", "roles": {"agg": {
        "nodes": 1, "workers": 1, "gpus": tp, "args": {"tensor-parallel-size": tp, **args},
    }}}  # fmt: skip


def agentic_recipe(tp: int, *, model: str, image: str, precision: str) -> dict:
    """A complete single-node AgentX sglang recipe for TP ``tp`` points of ``model``."""
    return {
        **single_node_fragment(tp),
        "model": {"path": f"hf:{model}", "container": image, "precision": precision},
        "benchmark": {"type": "custom", "command": "bash srt_agentic.sh", "env": {"MODEL": model}},
    }
