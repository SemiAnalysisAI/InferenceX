"""The workspace recipe mirror, and the text edits multi-node lanes make to the job's copies.

Only disposable copies in the job's srt-slurm checkout are edited: the staged mirror and the
recipe the binder writes.
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import TYPE_CHECKING

from infx.launch.context import LaunchError

if TYPE_CHECKING:
    from infx.launch.drivers.srt.lanes import SrtLane
    from infx.launch.request import LaunchRequest

RECIPES_MIRROR = Path("benchmarks/multi_node/srt-slurm-recipes")
HEALTH_ATTEMPTS = 720
_HEALTH_CHECK = re.compile(r"( *)health_check:")
_MAX_ATTEMPTS = re.compile(r"(\bmax_attempts:\s*)(\d+)")
FORCED_ACCEPTANCE_MARKER = "TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS"


def recipe_relpath(srt_recipe: str) -> str:
    """``srt_recipe`` without its ``:<selector>``."""
    return srt_recipe.split(":", 1)[0]


def recipe_mirror_path(workspace: Path, srt_recipe: str) -> Path:
    return workspace / recipe_relpath(srt_recipe)


def staged_recipe(srt_recipe: str) -> str:
    """The srtctl file argument: the recipe's copy in the checkout's ``recipes/``, selector kept."""
    relative = srt_recipe.removeprefix(f"{RECIPES_MIRROR.as_posix()}/")
    if relative == srt_recipe:
        raise LaunchError(f"{srt_recipe} is not under {RECIPES_MIRROR}")
    return f"recipes/{relative}"


def rename_job(text: str, name: str) -> str:
    """Set the top-level ``name:``, the job name srtctl submits."""
    return re.sub(r"(?m)^name:.*$", lambda _: f'name: "{name}"', text)


def raise_health_attempts(text: str) -> str:
    """Raise each ``health_check`` block's ``max_attempts`` to at least HEALTH_ATTEMPTS."""
    lines = text.splitlines(keepends=True)
    block: int | None = None
    for index, line in enumerate(lines):
        content = line.strip()
        indent = len(line) - len(line.lstrip(" "))
        if block is not None and content and not content.startswith("#") and indent <= block:
            block = None
        if heading := _HEALTH_CHECK.match(line):
            block = len(heading[1])
        if block is not None:
            lines[index] = _MAX_ATTEMPTS.sub(_at_least_the_floor, line)
    return "".join(lines)


def _at_least_the_floor(match: re.Match[str]) -> str:
    return f"{match[1]}{max(int(match[2]), HEALTH_ATTEMPTS)}"


def add_dist_timeout(text: str, seconds: int) -> str:
    """Insert ``dist-timeout: seconds`` after each role's ``watchdog-timeout``."""
    lines = []
    for line in text.splitlines(keepends=True):
        lines.append(line if line.endswith("\n") else line + "\n")
        if line.startswith("      watchdog-timeout:"):
            lines.append(f"      dist-timeout: {seconds}\n")
    return "".join(lines)


def prepare_recipe(recipe: Path, job_name: str, dist_timeout_s: int | None) -> None:
    """Edit the bound recipe, the variant srtctl submits, for this job."""
    text = raise_health_attempts(rename_job(recipe.read_text(), job_name))
    if dist_timeout_s is not None:
        text = add_dist_timeout(text, dist_timeout_s)
    recipe.write_text(text)


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
