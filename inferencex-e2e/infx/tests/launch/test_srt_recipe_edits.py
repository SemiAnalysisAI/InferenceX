"""Edits the multi-node lanes apply to the staged recipe copy."""

from pathlib import Path

import pytest
import yaml

from infx.launch.drivers.srt.recipe import (
    RECIPES_MIRROR,
    add_dist_timeout,
    inject_concurrencies,
    parse_concurrencies,
    raise_health_attempts,
    rename_job,
)
from infx.srt_slurm.recipe_images import resolve_images

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


def test_every_health_budget_is_raised_to_720_attempts_but_never_shortened():
    assert raise_health_attempts(RECIPE) == RECIPE
    bundle = (
        "base:\n  health_check:\n    max_attempts: 360\n  frontend:\n    max_attempts: 3\n"
        "override_x:\n  health_check: {max_attempts: 100}\n  benchmark: {max_attempts: 2}\n"
    )
    raised = yaml.safe_load(raise_health_attempts(bundle))
    assert raised["base"]["health_check"]["max_attempts"] == raised["override_x"]["health_check"]["max_attempts"] == 720
    assert (raised["base"]["frontend"]["max_attempts"], raised["override_x"]["benchmark"]["max_attempts"]) == (3, 2)


def test_dist_timeout_follows_each_role_watchdog_timeout():
    edited = yaml.safe_load(add_dist_timeout(RECIPE, 1800))
    assert edited["roles"]["prefill"]["args"] == {"watchdog-timeout": 600, "dist-timeout": 1800, "tp": 4}
    assert add_dist_timeout("watchdog-timeout: 1\n", 1800) == "watchdog-timeout: 1\n"


def test_injection_replaces_only_the_benchmark_concurrencies(tmp_path):
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("name: x\nbenchmark:\n  type: sa-bench\n  concurrencies: 1x2x4\n")
    inject_concurrencies(recipe, [4, 16])
    assert yaml.safe_load(recipe.read_text()) == {
        "name": "x", "benchmark": {"type": "sa-bench", "concurrencies": [4, 16]},
    }  # fmt: skip


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


def test_recipe_images_resolve_the_selected_override_and_deduplicate(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "utils/srt-slurm/src"))
    path = tmp_path / RECIPES_MIRROR / "images.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(
        yaml.safe_dump(
            {
                "base": {
                    "model": {"container": "worker:base"},
                    "frontend": {"container_image": "router:v1"},
                },
                "override_changed": {"model": {"container": "worker:v2"}},
                "override_shared": {"frontend": {"container_image": "worker:base"}},
            }
        )
    )
    assert resolve_images(f"{path}:override_changed", "worker:v2") == [
        "worker:v2",
        "router:v1",
    ]
    assert resolve_images(f"{path}:override_shared", "worker:base") == [
        "worker:base"
    ]
    with pytest.raises(ValueError, match="exactly one"):
        resolve_images(str(path), "worker:base")
    with pytest.raises(ValueError, match="must match"):
        resolve_images(f"{path}:override_changed", "worker:base")


@pytest.mark.parametrize("frontend", [{"container_image": "bad image"}, "not-a-mapping"])
def test_recipe_images_reject_invalid_frontend_identities(tmp_path, frontend, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "utils/srt-slurm/src"))
    path = tmp_path / RECIPES_MIRROR / "images.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump({"model": {"container": "worker:v1"}, "frontend": frontend}))
    with pytest.raises(ValueError, match="invalid recipe images"):
        resolve_images(str(path), "worker:v1")
