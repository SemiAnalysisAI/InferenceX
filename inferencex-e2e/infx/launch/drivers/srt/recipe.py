"""The workspace recipe mirror and the edits multi-node lanes make to a staged recipe copy.

Only the disposable copies staged in the job's srt-slurm checkout are edited. The text
edits keep comments and layout; power concurrency injection rewrites the recipe through
YAML, which drops comments.
"""

from __future__ import annotations

import fnmatch
import os
import re
import tempfile
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
# Forced TRT acceptance: eval-only AgentX TRT runs strip it to verify drafts for real.
FORCED_ACCEPTANCE_MARKER = "TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS"


def recipe_relpath(config_file: str) -> str:
    """Return the recipe path of ``CONFIG_FILE``, without its ``:<override>`` selector."""
    return config_file.split(":", 1)[0]


def recipe_mirror_path(workspace: Path, config_file: str) -> Path:
    """Return the workspace recipe mirror for ``CONFIG_FILE`` (``recipes/`` stripped)."""
    rel = recipe_relpath(config_file).removeprefix("recipes/")
    return Path(workspace) / RECIPES_MIRROR / rel


def rename_job(text: str, name: str) -> str:
    """Set the top-level ``name:``, the job name srtctl submits."""
    return re.sub(r"(?m)^name:.*$", lambda _: f'name: "{name}"', text)


def raise_health_attempts(text: str) -> str:
    """Raise each health-check ``max_attempts`` below 720 to 720; longer budgets stay."""
    return re.sub(
        r"(\bmax_attempts:\s*)(\d+)",
        lambda match: f"{match[1]}{max(int(match[2]), HEALTH_ATTEMPTS)}",
        text,
    )


def add_dist_timeout(text: str, seconds: int) -> str:
    """Insert ``dist-timeout: seconds`` after each role's ``watchdog-timeout``."""
    lines = []
    for line in text.splitlines(keepends=True):
        lines.append(line if line.endswith("\n") else line + "\n")
        if line.startswith("      watchdog-timeout:"):
            lines.append(f"      dist-timeout: {seconds}\n")
    return "".join(lines)


def edit_recipe(config_path: Path, job_name: str, dist_timeout_s: int | None) -> None:
    """Apply the text rewrites every lane makes to the staged recipe copy."""
    text = raise_health_attempts(rename_job(config_path.read_text(), job_name))
    if dist_timeout_s is not None:
        text = add_dist_timeout(text, dist_timeout_s)
    config_path.write_text(text)


def prepare_recipe(
    checkout: Path,
    config_file: str,
    job_name: str,
    dist_timeout_s: int | None,
    conc_list: str | None,
) -> None:
    """Edit the checkout's staged copy of ``config_file`` for this job.

    ``conc_list`` is injected on power lanes, which validate one window per concurrency.
    """
    config_path = checkout / recipe_relpath(config_file)
    if not config_path.is_file():
        raise LaunchError(f"CONFIG_FILE does not exist after srt-slurm setup: {config_path}")
    edit_recipe(config_path, job_name, dist_timeout_s)
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
    """srtctl overrides of an eval run; eval-only real verification also edits ``recipes``.

    Accuracy runs use real speculative verification, so forced TRT acceptance is
    stripped from the staged recipes where the lane says so.
    """
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
    """Atomically set top-level ``benchmark.concurrencies`` in ``recipe_path``.

    Power lanes validate one power window per requested concurrency, so the recipe
    must benchmark exactly CONC_LIST. Raises ``ValueError`` for unreadable YAML or a recipe
    without a top-level ``benchmark`` mapping (override-format recipes keep it under
    ``base`` and are rejected).
    """
    recipe_path = Path(recipe_path)
    try:
        recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"failed to load recipe {recipe_path}: {error}") from error
    if not isinstance(recipe, dict) or not isinstance(recipe.get("benchmark"), dict):
        raise ValueError(f"recipe {recipe_path} must contain a benchmark mapping")
    recipe["benchmark"]["concurrencies"] = list(concurrencies)
    descriptor, temporary = tempfile.mkstemp(dir=recipe_path.parent, prefix=f".{recipe_path.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            yaml.safe_dump(recipe, handle, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary).replace(recipe_path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
