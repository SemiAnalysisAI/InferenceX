"""Resolve srt-slurm CONFIG_FILE selectors into the recipe a job actually runs."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import infx.config
from infx.matrix import generate, plan
from infx.srt_slurm import recipe_selector
from infx.srt_slurm.recipe_selector import materialize, resolve_variant

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "utils/srt-slurm/src"))

# The pinned upstream implementation is the contract every selector must match.
from srtctl.core.config import generate_override_configs  # noqa: E402

RECIPE = {
    "schema": 2,
    "base": {
        "name": "fam",
        "roles": {"prefill": {"nodes": 1}, "decode": {"nodes": 2, "env": {"A": "1"}}},
        "benchmark": {"concurrencies": [4, 8]},
        "health_check": {"max_attempts": 180},
    },
    "zip_override_grid": {
        "name": ["fam-a", "fam-b"],
        "roles": {"decode": {"nodes": [2, 4], "env": [{"B": "2"}, None]}},
        "benchmark": {"concurrencies": [[4], [8, 16]]},
        "health_check": {"max_attempts": [None]},
    },
    "zip_override_auto": {"roles": {"prefill": {"nodes": [3, 5]}}},
    "override_solo": {"roles": {"decode": {"nodes": 7}}},
}


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    recipe = tmp_path / "benchmarks/multi_node/srt-slurm-recipes/fam/disagg.yaml"
    recipe.parent.mkdir(parents=True)
    recipe.write_text(yaml.safe_dump(RECIPE, sort_keys=False))
    (tmp_path / "configs").mkdir()
    monkeypatch.setattr(infx.config, "__file__", str(tmp_path / "infx/config.py"))
    recipe_selector.recipe_identities.cache_clear()
    yield tmp_path
    recipe_selector.recipe_identities.cache_clear()


@pytest.mark.parametrize(
    "selector",
    ["base", "zip_override_grid[0]", "zip_override_grid[1]", "zip_override_auto[1]", "override_solo"],
)
def test_resolve_variant_matches_srtctl(selector):
    assert resolve_variant(RECIPE, selector) == generate_override_configs(RECIPE, selector)[0][1]


def test_resolve_variant_slices_deletes_and_names():
    assert resolve_variant(RECIPE, "zip_override_grid[1]") == {
        "name": "fam-b",
        "roles": {"prefill": {"nodes": 1}, "decode": {"nodes": 4}},
        "benchmark": {"concurrencies": [8, 16]},
        "health_check": {},
        "schema": 2,
    }
    assert resolve_variant(RECIPE, "zip_override_auto[0]")["name"] == "fam_auto_0"


@pytest.mark.parametrize(
    ("selector", "message"),
    [
        ("zip_override_grid[2]", "out of range"),
        ("zip_override_grid", "Unsupported"),
        ("override_*", "Unsupported"),
    ],
)
def test_resolve_variant_rejects_selectors_that_are_not_one_job(selector, message):
    with pytest.raises(ValueError, match=message):
        resolve_variant(RECIPE, selector)


def test_materialize_cli_writes_a_flat_recipe_beside_its_source(repo):
    result = subprocess.run(
        [
            sys.executable, "-m", "infx.srt_slurm.recipe_selector", "materialize",
            "recipes/fam/disagg.yaml:zip_override_grid[0]", "--repo-root", str(repo),
        ],
        capture_output=True, text=True, check=True, cwd=ROOT,
    )

    assert result.stdout.strip() == "recipes/fam/disagg.zip_override_grid-0.resolved.yaml"
    written = repo / "benchmarks/multi_node/srt-slurm-recipes/fam/disagg.zip_override_grid-0.resolved.yaml"
    assert yaml.safe_load(written.read_text()) == {
        "name": "fam-a",
        "roles": {"prefill": {"nodes": 1}, "decode": {"nodes": 2, "env": {"A": "1", "B": "2"}}},
        "benchmark": {"concurrencies": [4]},
        "health_check": {},
        "schema": 2,
    }
    # Launchers patch top-level and nested keys by indentation.
    assert written.read_text().startswith("name: fam-a\n")
    assert materialize("recipes/fam/disagg.yaml", repo) == "recipes/fam/disagg.yaml"


def test_materialize_cli_reports_bad_selectors(repo):
    result = subprocess.run(
        [
            sys.executable, "-m", "infx.srt_slurm.recipe_selector", "materialize",
            "recipes/fam/disagg.yaml:zip_override_missing[0]", "--repo-root", str(repo),
        ],
        capture_output=True, text=True, cwd=ROOT,
    )

    assert result.returncode == 1
    assert "zip_override_missing" in result.stderr


def test_launcher_helper_replaces_selector_config_files(repo, tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # Stub uv: drop its own arguments and run the requested command locally.
    (bin_dir / "uv").write_text('#!/usr/bin/env bash\nshift 5\nexec "$@"\n')
    (bin_dir / "uv").chmod(0o755)
    (bin_dir / "python3").write_text(f'#!/usr/bin/env bash\nexec {sys.executable} "$@"\n')
    (bin_dir / "python3").chmod(0o755)
    script = (
        f'source {ROOT}/runners/slurm_utils.sh && materialize_srt_configs && '
        'bash -c \'echo "$CONFIG_FILE|$EVAL_CONFIG_FILE"\''
    )
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "GITHUB_WORKSPACE": str(repo),
        "CONFIG_FILE": "recipes/fam/disagg.yaml:override_solo",
        "EVAL_CONFIG_FILE": "recipes/fam/flat.yaml",
    }

    result = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, env=env, check=True
    )

    assert result.stdout.strip().splitlines()[-1] == (
        "recipes/fam/disagg.override_solo.resolved.yaml|recipes/fam/flat.yaml"
    )


def test_node_count_reads_the_selected_variant(repo):
    def settings(config_file):
        return {"additional-settings": [f"CONFIG_FILE={config_file}"]}

    assert generate.recipe_node_count(settings("recipes/fam/disagg.yaml:zip_override_grid[1]"), {}) == 5
    assert generate.recipe_node_count(settings("recipes/fam/disagg.yaml:override_solo"), {}) == 8
    # Without a selector srtctl submits every variant; the master estimate applies.
    assert generate.recipe_node_count(settings("recipes/fam/disagg.yaml"), {}) is None


def test_fingerprint_keeps_the_identity_of_a_consolidated_recipe(repo):
    def entry(config_file):
        return {
            "model": "m",
            "conc": [4],
            "prefill": {"tp": 4, "additional-settings": [f"CONFIG_FILE={config_file}", "X=1"]},
            "decode": {"tp": 8, "additional-settings": []},
        }

    old = entry("recipes/fam/disagg-1p2d.yaml")
    new = entry("recipes/fam/disagg.yaml:zip_override_grid[0]")
    before = plan.recipe_fingerprint(new)
    (repo / recipe_selector.IDENTITIES_FILE).write_text(
        yaml.safe_dump({"recipes/fam/disagg.yaml:zip_override_grid[0]": "recipes/fam/disagg-1p2d.yaml"})
    )
    recipe_selector.recipe_identities.cache_clear()

    assert before != plan.recipe_fingerprint(old)
    assert plan.recipe_fingerprint(new) == plan.recipe_fingerprint(old)
    assert plan.recipe_fingerprint(entry("recipes/fam/disagg.yaml:zip_override_grid[1]")) != (
        plan.recipe_fingerprint(old)
    )
