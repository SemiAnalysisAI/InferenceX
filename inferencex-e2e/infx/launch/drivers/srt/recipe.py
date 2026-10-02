"""The workspace recipe mirror, and the edits multi-node lanes make to the job's recipe copy.

Only the disposable copy staged in the job's srt-slurm checkout is edited. Text edits keep
its comments; concurrency injection rewrites it through YAML, which drops them.
"""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

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
    """The recipe path of ``SRT_RECIPE``, without its ``:<override>`` selector."""
    return srt_recipe.split(":", 1)[0]


def recipe_mirror_path(workspace: Path, srt_recipe: str) -> Path:
    """The workspace mirror of ``SRT_RECIPE``'s recipe."""
    return workspace / RECIPES_MIRROR / recipe_relpath(srt_recipe).removeprefix("recipes/")


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


def prepare_recipe(
    checkout: Path,
    srt_recipe: str,
    job_name: str,
    dist_timeout_s: int | None,
    conc_list: str | None,
) -> None:
    """Edit the checkout's staged copy of ``srt_recipe`` for this job."""
    config_path = checkout / recipe_relpath(srt_recipe)
    if not config_path.is_file():
        raise LaunchError(f"SRT_RECIPE does not exist after srt-slurm setup: {config_path}")
    text = raise_health_attempts(rename_job(config_path.read_text(), job_name))
    if dist_timeout_s is not None:
        text = add_dist_timeout(text, dist_timeout_s)
    config_path.write_text(text)
    if conc_list is not None:
        try:
            inject_concurrencies(config_path, parse_concurrencies(conc_list))
        except ValueError as error:
            raise LaunchError(str(error)) from error


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


def parse_concurrencies(conc_list: str) -> list[int]:
    """Parse a whitespace-separated CONC_LIST of unique, canonical positive integers."""
    values = []
    for word in conc_list.split():
        if not word.isascii() or not word.isdecimal() or str(int(word)) != word or int(word) <= 0:
            raise ValueError(f"CONC_LIST entries must be canonical positive integers: {word!r}")
        values.append(int(word))
    if not values or len(set(values)) != len(values):
        raise ValueError("concurrencies must be positive unique integers")
    return values


def inject_concurrencies(recipe_path: Path, concurrencies: Sequence[int]) -> None:
    """Set the recipe's top-level ``benchmark.concurrencies``.

    Raises ``ValueError``, leaving the file untouched, for unreadable YAML or no top-level
    ``benchmark`` mapping (override bundles keep theirs under ``base``).
    """
    try:
        recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"failed to load recipe {recipe_path}: {error}") from error
    if not isinstance(recipe, dict) or not isinstance(recipe.get("benchmark"), dict):
        raise ValueError(f"recipe {recipe_path} must contain a benchmark mapping")
    recipe["benchmark"]["concurrencies"] = list(concurrencies)
    recipe_path.write_text(yaml.safe_dump(recipe, sort_keys=False), encoding="utf-8")
