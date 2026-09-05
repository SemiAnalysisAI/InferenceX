import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
INJECTOR = REPO_ROOT / "runners" / "inject_synthetic_acceptance.py"
LAUNCHER = REPO_ROOT / "runners" / "launch_gb300-nv.sh"
MASTER = REPO_ROOT / "configs" / "nvidia-master.yaml"
RECIPES = REPO_ROOT / "benchmarks" / "multi_node" / "srt-slurm-recipes"
GOLDEN_AL = REPO_ROOT / "golden_al_distribution" / "dsv4-pro-0813-dspark.yaml"
CONFIG_KEYS = (
    "dsv4-fp4-gb300-dynamo-sglang-agentic-agg",
    "dsv4-fp4-gb300-dynamo-sglang-agentic-disagg",
)


def run_injector(
    recipe: Path, *, eval_only: bool, acceptance_length: str = "3.77"
) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "EVAL_ONLY": str(eval_only).lower(),
        "SYNTHETIC_ACCEPTANCE": "true",
        "SYNTHETIC_ACCEPTANCE_LENGTH": acceptance_length,
    }
    return subprocess.run(
        [sys.executable, str(INJECTOR), str(recipe), "dynamo-sglang"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def _configured_recipe_paths(master: dict) -> set[Path]:
    recipe_paths = set()
    for config_key in CONFIG_KEYS:
        assert master[config_key]["model"] == "deepseek-ai/DeepSeek-V4-Pro-0813"
        for group in master[config_key]["scenarios"]["agentic-coding"]:
            for point in group["search-space"]:
                assert point["spec-decoding"] == "draft_model"
                worker = point.get("prefill", point.get("worker"))
                assert worker is not None
                settings = worker["additional-settings"]
                assert "SYNTHETIC_ACCEPTANCE=true" in settings
                assert "SYNTHETIC_ACCEPTANCE_LENGTH=3.77" in settings
                config = next(
                    item for item in settings if item.startswith("CONFIG_FILE=")
                )
                recipe_paths.add(RECIPES / config.removeprefix("CONFIG_FILE=recipes/"))
    return recipe_paths


def test_sglang_injection_is_throughput_only(tmp_path: Path) -> None:
    original = "  prefill_environment:\n    A: B\n  decode_environment:\n    C: D\n"

    throughput_recipe = tmp_path / "throughput.yaml"
    throughput_recipe.write_text(original)
    result = run_injector(throughput_recipe, eval_only=False)
    assert result.returncode == 0, result.stderr
    assert throughput_recipe.read_text().count("SGLANG_SIMULATE_ACC_LEN") == 2

    eval_recipe = tmp_path / "eval.yaml"
    eval_recipe.write_text(original)
    result = run_injector(eval_recipe, eval_only=True)
    assert result.returncode == 0, result.stderr
    assert eval_recipe.read_text() == original


def test_gb300_dsv4_configs_inject_only_at_launch() -> None:
    master = yaml.safe_load(MASTER.read_text())
    recipe_paths = _configured_recipe_paths(master)

    assert len(recipe_paths) == 7
    for recipe in recipe_paths:
        text = recipe.read_text()
        parsed = yaml.safe_load(text)
        worker_count = 1 if recipe.name.startswith("agg-") else 2
        assert "SGLANG_SIMULATE_ACC_" not in text
        assert text.count('SGLANG_RAGGED_VERIFY_MODE: "static"') == worker_count
        assert text.count("speculative-algorithm: DSPARK") == worker_count
        assert text.count("speculative-dspark-block-size: 6") == worker_count
        assert text.count("speculative-num-draft-tokens: 7") == worker_count
        assert "speculative-algorithm: EAGLE" not in text
        assert 'path: "deepseek-v4-pro-0813"' in text
        assert "AIPERF_TOKENIZER" not in parsed["benchmark"]["env"]

    launcher = LAUNCHER.read_text()
    assert 'MODEL == "deepseek-ai/DeepSeek-V4-Pro-0813"' in launcher
    assert 'MODEL_PATH="/scratch/models/DeepSeek-V4-Pro-0813"' in launcher
    assert 'SRT_SLURM_MODEL_PREFIX="deepseek-v4-pro-0813"' in launcher
    source = 'source "$(dirname "${BASH_SOURCE[0]}")/slurm_utils.sh"'
    inject = 'inject_synthetic_acceptance "$CONFIG_PATH" "$FRAMEWORK" || exit 1'
    apply = "SRTCTL_OUTPUT=$(srtctl apply"
    assert launcher.index(source) < launcher.index(inject) < launcher.index(apply)


def test_gb300_dsv4_dspark6_matches_committed_golden_curve() -> None:
    golden = yaml.safe_load(GOLDEN_AL.read_text())
    assert golden["deepseek-v4-pro-0813"]["thinking_on"][6] == 3.77

    master = yaml.safe_load(MASTER.read_text())
    for config_key in CONFIG_KEYS:
        for group in master[config_key]["scenarios"]["agentic-coding"]:
            for point in group["search-space"]:
                worker = point.get("prefill", point.get("worker"))
                assert worker is not None
                settings = worker["additional-settings"]
                assert "SYNTHETIC_ACCEPTANCE_LENGTH=3.77" in settings


def test_gb300_dsv4_dspark6_uses_golden_acceptance_only_for_throughput(
    tmp_path: Path,
) -> None:
    master = yaml.safe_load(MASTER.read_text())
    for recipe in _configured_recipe_paths(master):
        original = recipe.read_text()
        worker_count = 1 if recipe.name.startswith("agg-") else 2

        throughput_recipe = tmp_path / recipe.name
        throughput_recipe.write_text(original)
        result = run_injector(throughput_recipe, eval_only=False)
        assert result.returncode == 0, result.stderr
        throughput_text = throughput_recipe.read_text()
        assert throughput_text.count('SGLANG_SIMULATE_ACC_LEN: "3.77"') == worker_count
        assert (
            throughput_text.count('SGLANG_RAGGED_VERIFY_MODE: "static"') == worker_count
        )

        eval_recipe = tmp_path / f"eval-{recipe.name}"
        eval_recipe.write_text(original)
        result = run_injector(eval_recipe, eval_only=True)
        assert result.returncode == 0, result.stderr
        eval_text = eval_recipe.read_text()
        assert eval_text == original
        assert "SGLANG_SIMULATE_ACC_" not in eval_text
        assert eval_text.count('SGLANG_RAGGED_VERIFY_MODE: "static"') == worker_count
