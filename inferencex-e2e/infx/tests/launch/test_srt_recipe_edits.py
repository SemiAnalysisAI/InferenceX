"""Edits the multi-node lanes apply to the bound recipe."""

import yaml

from infx.launch.drivers.srt.recipe import prepare_recipe


def test_the_job_names_the_run_and_gives_each_watchdog_role_the_dist_timeout(tmp_path):
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(yaml.safe_dump({
        "schema": 2, "name": "upstream",
        "roles": {
            "prefill": {"name": "keep-me", "args": {"watchdog-timeout": 600, "tp": 4}},
            "decode": {"args": {"tp": 4}},
        },
        "benchmark": {"type": "sa-bench"},
    }, sort_keys=False))  # fmt: skip

    prepare_recipe(recipe, "runner_00", 1800)

    assert yaml.safe_load(recipe.read_text()) == {
        "schema": 2, "name": "runner_00",
        "roles": {
            "prefill": {"name": "keep-me", "args": {"watchdog-timeout": 600, "tp": 4, "dist-timeout": 1800}},
            "decode": {"args": {"tp": 4}},
        },
        "benchmark": {"type": "sa-bench"},
    }  # fmt: skip
