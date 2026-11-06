"""The workspace recipe mirror, and the edits multi-node lanes make to the job's copies.

Only disposable copies in the job's srt-slurm checkout are edited: the staged mirror and the
recipe the binder writes.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from infx import yamlio
from infx.launch.context import LaunchError

if TYPE_CHECKING:
    from infx.launch.drivers.srt.lanes import SrtLane
    from infx.launch.request import LaunchRequest

RECIPES_MIRROR = Path("benchmarks/multi_node/srt-slurm-recipes")
HEALTH_ATTEMPTS = 720
FORCED_ACCEPTANCE_MARKER = "TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS"


def recipe_mirror_path(workspace: Path, srt_recipe: str) -> Path:
    """``srt_recipe``'s workspace copy, without its ``:<selector>``."""
    return workspace / srt_recipe.split(":", 1)[0]


def staged_recipe(srt_recipe: str) -> str:
    """The srtctl file argument: the recipe's copy in the checkout's ``recipes/``, selector kept."""
    relative = srt_recipe.removeprefix(f"{RECIPES_MIRROR.as_posix()}/")
    if relative == srt_recipe:
        raise LaunchError(f"{srt_recipe} is not under {RECIPES_MIRROR}")
    return f"recipes/{relative}"


def prepare_recipe(recipe: Path, job_name: str, dist_timeout_s: int | None) -> None:
    """Edit the bound recipe, the variant srtctl submits, for this job: ``job_name`` becomes
    srtctl's run name, the health check waits at least HEALTH_ATTEMPTS attempts, and each role
    with a ``watchdog-timeout`` gets the cluster's ``dist-timeout``."""
    bound = yamlio.safe_load(recipe.read_text())
    if "name" in bound:
        bound["name"] = job_name
    if "max_attempts" in (health := bound.get("health_check") or {}):
        health["max_attempts"] = max(health["max_attempts"], HEALTH_ATTEMPTS)
    for role in (bound.get("roles") or {}).values():
        args = (role or {}).get("args") or {}
        if dist_timeout_s is not None and "watchdog-timeout" in args:
            args["dist-timeout"] = dist_timeout_s
    recipe.write_text(yaml.safe_dump(bound, sort_keys=False))


def strip_forced_acceptance(recipes: Path) -> None:
    """Drop forced TRT acceptance from every staged trtllm recipe."""
    for recipe in sorted(recipes.rglob("*.yaml")):
        relative = recipe.relative_to(recipes.parent).as_posix()
        if recipe.is_file() and fnmatch.fnmatchcase(relative, "recipes/*/trtllm/*"):
            lines = recipe.read_text().splitlines(keepends=True)
            kept = [line for line in lines if FORCED_ACCEPTANCE_MARKER not in line]
            recipe.write_text("".join(kept))


def eval_overrides(recipes: Path, lane: SrtLane, request: LaunchRequest) -> list[str]:
    """srtctl overrides of an eval run; eval-only real verification also edits ``recipes``."""
    overrides: list[str] = []
    if request.eval_only and lane.real_verification is not None and lane.real_verification(request):
        strip_forced_acceptance(recipes)
        if lane.head_frontend is not None and lane.head_frontend(request):
            overrides += ["--set", "frontend.placement.node=head"]
    if request.run_eval or request.eval_only:
        for key in lane.eval_unsets:
            overrides += ["--unset", key]
    return overrides
