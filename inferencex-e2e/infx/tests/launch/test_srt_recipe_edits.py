"""Edits the multi-node lanes apply to the staged recipe copy."""

import pytest
import yaml

from infx.launch.drivers.srt.recipe import (
    add_dist_timeout,
    inject_concurrencies,
    parse_concurrencies,
    raise_health_attempts,
    rename_job,
    replace_health_check,
)

RECIPE = """name: "upstream"
roles:
  prefill:
    name: keep-me
    args:
      watchdog-timeout: 600
      tp: 4
health_check:
  max_attempts: 1440

  interval_seconds: 10
benchmark:
  type: sa-bench
"""


def test_rename_touches_only_the_top_level_name():
    renamed = yaml.safe_load(rename_job(RECIPE, "runner_00"))
    assert renamed["name"] == "runner_00"
    assert renamed["roles"]["prefill"]["name"] == "keep-me"


def test_health_attempts_are_raised_but_never_shortened():
    # A recipe may request more than 720 attempts, as GLM-5.2's 1440 x 10s does.
    assert raise_health_attempts(RECIPE) == RECIPE
    short = RECIPE.replace("max_attempts: 1440", "max_attempts: 100")
    assert yaml.safe_load(raise_health_attempts(short))["health_check"]["max_attempts"] == 720


def test_health_check_block_is_replaced_and_later_keys_survive():
    replaced = replace_health_check(RECIPE)
    assert replaced.count("health_check:") == 1
    parsed = yaml.safe_load(replaced)
    assert parsed["health_check"] == {"max_attempts": 720, "interval_seconds": 10}
    assert parsed["benchmark"] == {"type": "sa-bench"}
    assert parsed["roles"]["prefill"]["args"] == {"watchdog-timeout": 600, "tp": 4}
    # Without a block the budget is only appended.
    bare = "name: x\nbenchmark:\n  type: custom\n"
    assert yaml.safe_load(replace_health_check(bare))["health_check"]["max_attempts"] == 720


def test_dist_timeout_follows_each_role_watchdog_timeout():
    edited = yaml.safe_load(add_dist_timeout(RECIPE, 1800))
    assert edited["roles"]["prefill"]["args"] == {"watchdog-timeout": 600, "dist-timeout": 1800, "tp": 4}
    assert add_dist_timeout("watchdog-timeout: 1\n", 1800) == "watchdog-timeout: 1\n"


# Power lanes benchmark exactly CONC_LIST.


def test_injection_replaces_only_the_benchmark_concurrencies(tmp_path):
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("name: x\nbenchmark:\n  type: sa-bench\n  concurrencies: 1x2x4\n")
    inject_concurrencies(recipe, [4, 16])
    assert yaml.safe_load(recipe.read_text()) == {
        "name": "x", "benchmark": {"type": "sa-bench", "concurrencies": [4, 16]},
    }  # fmt: skip
    assert [path.name for path in tmp_path.iterdir()] == ["recipe.yaml"]


@pytest.mark.parametrize(("text", "message"), [
    ("base:\n  benchmark:\n    concurrencies: [1]\n", "benchmark mapping"),
    ("benchmark: [\n", "failed to load"),
])  # fmt: skip
def test_an_unusable_recipe_is_left_untouched(tmp_path, text, message):
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(text)
    with pytest.raises(ValueError, match=message):
        inject_concurrencies(recipe, [4])
    assert recipe.read_text() == text


def test_conc_list_must_be_canonical_positive_integers():
    assert parse_concurrencies(" 4 8\t16 ") == [4, 8, 16]
    for bad in ("", "08", "0", "-4", "4.0", "4 4", "+4"):
        with pytest.raises(ValueError):
            parse_concurrencies(bad)
