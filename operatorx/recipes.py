"""The InferenceX recipe a case runs under.

InferenceX serves every single-node configuration from a native srt-slurm recipe
(inferencex-e2e/benchmarks/single_node/srt-slurm-recipes/<model>/<framework>/<hardware>-<precision>*/*.yaml).
A case takes the recipe InferenceX runs for its source checkpoint on the runner's
hardware: the recipe's image, launch env and server arguments. InferenceX's own code
expands a recipe into its variants (infx.srt_slurm.synthetic_acceptance.selected_recipes).
Without a recipe the case runs on the backend's default image.
"""
from __future__ import annotations

import functools
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1] / "inferencex-e2e"  # infx, its recipes and srt-slurm
RECIPES = REPO / "benchmarks/single_node/srt-slurm-recipes"
FRAMEWORKS = ("vllm",)

# Server arguments a layer benchmark does not apply: artifacts it never loads (weights,
# tokenizer and the parsers built on it, draft model), the parallel layout (a case's own),
# the KV tiers and prefix cache of a long-running server, and the frontend.
_NOT_APPLIED = {"model", "tokenizer", "tokenizer-mode", "reasoning-parser", "tool-call-parser",
                "enable-auto-tool-choice", "speculative-config", "served-model-name", "load-format",
                "safetensors-load-strategy", "skip-tokenizer-init", "tensor-parallel-size",
                "data-parallel-size", "pipeline-parallel-size", "enable-expert-parallel",
                "expert-parallel-size", "decode-context-parallel-size", "dcp-comm-backend",
                "distributed-executor-backend", "kv-transfer-config", "kv-events-config",
                "enable-prefix-caching", "prefix-match-unit", "enable-cumem-allocator",
                "gpu-memory-utilization", "kv-cache-memory-bytes", "max-num-seqs", "max-model-len",
                "enforce-eager"}


@functools.cache
def _variants(framework: str, hardware: str) -> tuple[tuple[str, dict], ...]:
    """Every expanded variant InferenceX runs for a framework on a hardware family."""
    import yaml

    for path in (str(REPO), str(REPO / "utils/srt-slurm/src")):
        if path not in sys.path:
            sys.path.append(path)
    from infx.srt_slurm.synthetic_acceptance import selected_recipes

    out = []
    for f in sorted(RECIPES.glob(f"*/{framework}/{hardware}-*/*.yaml")):
        for name, recipe in selected_recipes(yaml.safe_load(f.read_text()), None):
            out.append((str(f.relative_to(REPO)) + (f":{name}" if name else ""), recipe))
    return tuple(out)


def find(framework: str, hardware: str, checkpoints: list[str]) -> dict | None:
    """The recipe InferenceX runs for the first of the checkpoints served on this hardware."""
    if framework not in FRAMEWORKS or not RECIPES.is_dir():
        return None
    variants = _variants(framework, hardware)
    for ckpt in checkpoints:
        for key, r in variants:
            if r["model"]["path"] == f"hf:{ckpt}":
                role = r["roles"]["agg"]
                return {"recipe": key, "checkpoint": ckpt, "image": r["model"]["container"],
                        "env": {k: str(v) for k, v in (role.get("env") or {}).items()},
                        "engine_args": role.get("args") or {}}
    return None


def checkpoints(sources) -> list[str]:
    """The distinct checkpoints ("org/model") a case's "<org>/<model>/<role>" sources name."""
    return list(dict.fromkeys(s.rsplit("/", 1)[0] for s in sources or ()))


def serve_argv(args: dict[str, Any]) -> list[str]:
    """A recipe's server arguments as `vllm serve` command-line arguments, less the ones a
    layer benchmark does not apply."""
    out = []
    for k, v in args.items():
        if k in _NOT_APPLIED or v is None or v is False:
            continue
        out.append(f"--{k}")
        if v is not True:
            out.append(v if isinstance(v, str) else json.dumps(v))
    return out
