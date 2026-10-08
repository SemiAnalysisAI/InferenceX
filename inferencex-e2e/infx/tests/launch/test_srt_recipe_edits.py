"""Edits the multi-node lanes apply to the staged recipe copy."""

import yaml

from infx.launch.drivers.srt.recipe import add_dist_timeout, raise_health_attempts, rename_job

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
