"""Tests for runtime injection of AgentX power measurement points."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


def test_injects_positive_unique_concurrencies_without_adding_telemetry(tmp_path: Path):
    from runners.inject_srt_power_concurrencies import inject_concurrencies

    recipe_path = tmp_path / "recipe.yaml"
    recipe_path.write_text(
        "name: agentx\nbenchmark:\n  type: custom\n  command: run-agentx\n",
        encoding="utf-8",
    )

    inject_concurrencies(recipe_path, [8, 16, 32])

    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    assert recipe["benchmark"]["concurrencies"] == [8, 16, 32]
    assert "telemetry" not in recipe
    assert recipe["benchmark"]["command"] == "run-agentx"


@pytest.mark.parametrize(
    "concurrencies",
    [[], [0], [-1], [8, 8], [True], [8, 1.5]],
)
def test_rejects_invalid_concurrency_contract(
    tmp_path: Path,
    concurrencies: list,
):
    from runners.inject_srt_power_concurrencies import inject_concurrencies

    recipe_path = tmp_path / "recipe.yaml"
    recipe_path.write_text("benchmark:\n  type: custom\n", encoding="utf-8")

    with pytest.raises(ValueError, match="positive unique integers"):
        inject_concurrencies(recipe_path, concurrencies)


@pytest.mark.parametrize(
    "recipe_text",
    ["[]\n", "name: missing-benchmark\n", "benchmark: []\n"],
)
def test_rejects_recipe_without_benchmark_mapping(tmp_path: Path, recipe_text: str):
    from runners.inject_srt_power_concurrencies import inject_concurrencies

    recipe_path = tmp_path / "recipe.yaml"
    recipe_path.write_text(recipe_text, encoding="utf-8")

    with pytest.raises(ValueError, match="benchmark mapping"):
        inject_concurrencies(recipe_path, [8])
