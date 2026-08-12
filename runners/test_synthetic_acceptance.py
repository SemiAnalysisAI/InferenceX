import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
INJECTOR = REPO_ROOT / "runners" / "inject_synthetic_acceptance.py"
GB300_LAUNCHER = REPO_ROOT / "runners" / "launch_gb300-nv.sh"
NVIDIA_MASTER = REPO_ROOT / "configs" / "nvidia-master.yaml"
CHECKED_IN_RECIPES = REPO_ROOT / "benchmarks" / "multi_node" / "srt-slurm-recipes"
SPEC_CONFIG_RE = re.compile(r"speculative-config:\s*'([^']+)'")
GB300_DSV4_KEYS = (
    "dsv4-fp4-gb300-dynamo-vllm-agentic-mtp-agg",
    "dsv4-fp4-gb300-dynamo-vllm-agentic-mtp-disagg",
)


def run_injector(recipe: Path, *, eval_only: bool) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "EVAL_ONLY": "true" if eval_only else "false",
        "SYNTHETIC_ACCEPTANCE": "true",
        "SYNTHETIC_ACCEPTANCE_LENGTH": "2.49",
    }
    return subprocess.run(
        [sys.executable, str(INJECTOR), str(recipe), "dynamo-vllm"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_throughput_injects_synthetic_acceptance(tmp_path: Path) -> None:
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        '      speculative-config: \'{"method":"mtp","num_speculative_tokens":3}\'\n'
    )

    result = run_injector(recipe, eval_only=False)

    assert result.returncode == 0, result.stderr
    match = SPEC_CONFIG_RE.search(recipe.read_text())
    assert match is not None
    assert json.loads(match.group(1)) == {
        "method": "mtp",
        "num_speculative_tokens": 3,
        "rejection_sample_method": "synthetic",
        "synthetic_acceptance_length": 2.49,
    }


def test_eval_only_keeps_real_mtp_recipe(tmp_path: Path) -> None:
    recipe = tmp_path / "recipe.yaml"
    original = (
        '      speculative-config: \'{"method":"mtp","num_speculative_tokens":3}\'\n'
    )
    recipe.write_text(original)

    result = run_injector(recipe, eval_only=True)

    assert result.returncode == 0, result.stderr
    assert recipe.read_text() == original
    assert "EVAL_ONLY=true: keeping real MTP recipe" in result.stdout


def test_gb300_agentic_configs_inject_synthetic_only_at_launch() -> None:
    master = yaml.safe_load(NVIDIA_MASTER.read_text())
    recipe_paths: set[Path] = set()

    for config_key in GB300_DSV4_KEYS:
        search_space = master[config_key]["scenarios"]["agentic-coding"][0][
            "search-space"
        ]
        for point in search_space:
            settings = point["prefill"]["additional-settings"]
            assert "SYNTHETIC_ACCEPTANCE=true" in settings
            assert "SYNTHETIC_ACCEPTANCE_LENGTH=2.49" in settings
            config_setting = next(
                setting for setting in settings if setting.startswith("CONFIG_FILE=")
            )
            config_path = config_setting.removeprefix("CONFIG_FILE=")
            recipe_paths.add(CHECKED_IN_RECIPES / config_path.removeprefix("recipes/"))

    assert len(recipe_paths) == 5
    for recipe_path in recipe_paths:
        content = recipe_path.read_text()
        matches = SPEC_CONFIG_RE.findall(content)
        assert matches, recipe_path
        for raw_config in matches:
            spec_config = json.loads(raw_config)
            assert spec_config["method"] == "mtp"
            assert spec_config["num_speculative_tokens"] == 3
            assert "rejection_sample_method" not in spec_config
            assert "synthetic_acceptance_length" not in spec_config


def test_gb300_launcher_runs_acceptance_normalizer_before_srtctl() -> None:
    launcher = GB300_LAUNCHER.read_text()

    source_offset = launcher.index(
        'source "$(dirname "${BASH_SOURCE[0]}")/slurm_utils.sh"'
    )
    inject_offset = launcher.index(
        'inject_synthetic_acceptance "$CONFIG_PATH" "$FRAMEWORK" || exit 1'
    )
    apply_offset = launcher.index("SRTCTL_OUTPUT=$(srtctl apply")

    assert source_offset < inject_offset < apply_offset
