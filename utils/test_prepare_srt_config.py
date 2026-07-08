"""Tests for srt-slurm recipe normalization."""

from pathlib import Path
import os
import subprocess

import pytest
import yaml

from prepare_srt_config import prepare_config, use_inferencex_benchmark


def write_recipe(tmp_path: Path) -> Path:
    path = tmp_path / "recipe.yaml"
    path.write_text(
        """
base:
  name: test
  model:
    path: deepseek-v4-pro
    container: stale/recipe:image
  resources:
    gpus_per_node: 4
    prefill_nodes: 1
    prefill_workers: 1
    decode_nodes: 2
    decode_workers: 2
    gpus_per_decode: 2
  frontend:
    type: dynamo
    enable_multiple_frontends: true
  backend:
    sglang_config:
      prefill:
        served-model-name: DeepSeek-V4-Pro
  benchmark:
    type: sa-bench
    req_rate: 300
    use_chat_template: true
    custom_tokenizer: sa_bench_tokenizers.sglang_deepseek_v4.SGLangDeepseekV4Tokenizer
zip_override_curve:
  benchmark:
    isl: [1024, 8192]
    osl: 1024
    concurrencies: ["8x16", "32"]
zip_override_named:
  name: [explicit-zero, explicit-one]
  benchmark:
    isl: 1024
    osl: 1024
    concurrencies: ["8", "16"]
""",
        encoding="utf-8",
    )
    return path


def test_resolves_selector_and_uses_custom_benchmark(tmp_path: Path) -> None:
    source = write_recipe(tmp_path)
    output = prepare_config(
        f"{source}:zip_override_curve[1]",
        isl=4096,
        concurrencies="4x8",
        random_range_ratio=1.0,
    )
    recipe = yaml.safe_load(output.read_text(encoding="utf-8"))
    benchmark = recipe["benchmark"]

    assert recipe["name"] == "test_curve_1"
    assert recipe["model"]["container"] == "inferencex-workload"
    assert recipe["model"]["path"] == "inferencex-model"
    assert recipe["frontend"]["nginx_container"] == "inferencex-nginx"
    assert benchmark["type"] == "custom"
    assert benchmark["command"].endswith("srt_benchmark.sh")
    assert benchmark["env"] == {
        "INFMAX_CONTAINER_WORKSPACE": "/infmax-workspace",
        "RESULT_ROOT": "/logs",
        "PORT": "8000",
        "TOKENIZER_PATH": "/model",
        "TOKENIZER_MODE": "deepseek_v4",
        "ISL": "4096",
        "OSL": "1024",
        "CONC_LIST": "4x8",
        "REQUEST_RATE": "300",
        "RANDOM_RANGE_RATIO": "1.0",
        "NUM_PROMPTS_MULTIPLIER": "10",
        "NUM_WARMUP_MULTIPLIER": "2",
        "MODEL_NAME": "DeepSeek-V4-Pro",
        "TOTAL_GPUS": "8",
        "PREFILL_GPUS": "4",
        "DECODE_GPUS": "4",
        "USE_CHAT_TEMPLATE": "true",
        "DSV4_CHAT_TEMPLATE": "true",
    }


def test_preserves_existing_custom_benchmark(tmp_path: Path) -> None:
    source = tmp_path / "custom.yaml"
    source.write_text(
        """
name: custom
benchmark:
  type: custom
  command: bash existing.sh
  aiperf_server_metrics: true
  env:
    VALUE: "1"
""",
        encoding="utf-8",
    )

    output = prepare_config(str(source))
    recipe = yaml.safe_load(output.read_text(encoding="utf-8"))

    assert recipe["benchmark"]["command"] == "bash existing.sh"
    assert "aiperf_server_metrics" not in recipe["benchmark"]
    assert recipe["benchmark"]["env"] == {
        "VALUE": "1",
        "INFERENCEX_AIPERF_SERVER_METRICS": "true",
    }


def test_preserves_explicit_zip_override_name(tmp_path: Path) -> None:
    source = write_recipe(tmp_path)

    output = prepare_config(f"{source}:zip_override_named[1]")
    recipe = yaml.safe_load(output.read_text(encoding="utf-8"))

    assert recipe["name"] == "explicit-one"


def test_normalizes_legacy_sa_bench_fields(tmp_path: Path) -> None:
    source = write_recipe(tmp_path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["base"]["benchmark"].update(
        tokenizer_mode="deepseek_v4",
        warmup_req_rate="inf",
    )
    source.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    output = prepare_config(
        f"{source}:base", isl=1024, osl=1024, concurrencies="8"
    )
    benchmark = yaml.safe_load(output.read_text(encoding="utf-8"))["benchmark"]

    assert "tokenizer_mode" not in benchmark
    assert "warmup_req_rate" not in benchmark
    assert benchmark["env"]["DSV4_CHAT_TEMPLATE"] == "true"


def test_explicit_recipe_random_range_ratio_wins_over_workflow_default() -> None:
    recipe = {
        "model": {"path": "model"},
        "resources": {
            "gpus_per_node": 8,
            "agg_nodes": 1,
            "agg_workers": 1,
        },
        "benchmark": {
            "type": "sa-bench",
            "isl": 1024,
            "osl": 1024,
            "concurrencies": [8],
            "random_range_ratio": 1.0,
        },
    }

    benchmark = use_inferencex_benchmark(
        recipe,
        random_range_ratio=0.8,
    )["benchmark"]

    assert benchmark["env"]["RANDOM_RANGE_RATIO"] == "1.0"


def test_dsv4_raw_prompts_still_use_direct_tokenizer_loader() -> None:
    recipe = {
        "model": {"path": "deepseek-v4-pro"},
        "resources": {
            "gpus_per_node": 8,
            "agg_nodes": 1,
            "agg_workers": 1,
        },
        "benchmark": {
            "type": "sa-bench",
            "isl": 1024,
            "osl": 1024,
            "concurrencies": [8],
            "use_chat_template": False,
            "custom_tokenizer": (
                "sa_bench_tokenizers.sglang_deepseek_v4."
                "SGLangDeepseekV4Tokenizer"
            ),
        },
    }

    benchmark = use_inferencex_benchmark(recipe)["benchmark"]

    assert benchmark["env"]["TOKENIZER_MODE"] == "deepseek_v4"
    assert benchmark["env"]["USE_CHAT_TEMPLATE"] == "false"
    assert benchmark["env"]["DSV4_CHAT_TEMPLATE"] == "false"


def test_matches_srt_gpu_accounting_for_partial_workers() -> None:
    recipe = {
        "model": {"path": "model"},
        "resources": {
            "gpus_per_node": 8,
            "prefill_nodes": 1,
            "prefill_workers": 2,
            "gpus_per_prefill": 2,
            "decode_nodes": 1,
            "decode_workers": 1,
            "gpus_per_decode": 4,
        },
        "benchmark": {
            "type": "sa-bench",
            "isl": 1024,
            "osl": 1024,
            "concurrencies": [8],
        },
    }

    benchmark = use_inferencex_benchmark(recipe)["benchmark"]

    assert benchmark["env"]["PREFILL_GPUS"] == "4"
    assert benchmark["env"]["DECODE_GPUS"] == "4"
    assert benchmark["env"]["TOTAL_GPUS"] == "8"


def test_aggregate_accounting_reports_provisioned_gpus() -> None:
    recipe = {
        "model": {"path": "model"},
        "resources": {
            "gpus_per_node": 8,
            "agg_nodes": 2,
            "agg_workers": 1,
            "gpus_per_agg": 4,
        },
        "benchmark": {
            "type": "sa-bench",
            "isl": 1024,
            "osl": 1024,
            "concurrencies": [8],
        },
    }

    benchmark = use_inferencex_benchmark(recipe)["benchmark"]

    assert benchmark["env"]["PREFILL_GPUS"] == "0"
    assert benchmark["env"]["DECODE_GPUS"] == "0"
    assert benchmark["env"]["TOTAL_GPUS"] == "16"


def test_hugging_face_model_is_used_as_tokenizer_path() -> None:
    recipe = {
        "model": {"path": "hf:org/model"},
        "resources": {
            "gpus_per_node": 8,
            "agg_nodes": 1,
            "agg_workers": 1,
        },
        "benchmark": {
            "type": "sa-bench",
            "isl": 1024,
            "osl": 1024,
            "concurrencies": [8],
        },
    }

    benchmark = use_inferencex_benchmark(recipe)["benchmark"]

    assert benchmark["env"]["TOKENIZER_PATH"] == "org/model"
    assert benchmark["env"]["TOKENIZER_MODE"] == "auto"
    assert benchmark["env"]["MODEL_NAME"] == "model"
    assert recipe["model"]["path"] == "hf:org/model"


def test_model_alias_uses_same_served_name_as_srt_slurm() -> None:
    recipe = {
        "model": {"path": "nvidia/GLM-5-NVFP4"},
        "resources": {
            "gpus_per_node": 4,
            "agg_nodes": 1,
            "agg_workers": 1,
        },
        "benchmark": {
            "type": "sa-bench",
            "isl": 1024,
            "osl": 1024,
            "concurrencies": [8],
        },
    }

    benchmark = use_inferencex_benchmark(recipe)["benchmark"]

    assert benchmark["env"]["MODEL_NAME"] == "GLM-5-NVFP4"


def test_cluster_model_basename_is_default_served_name() -> None:
    recipe = {
        "model": {
            "path": "dsr1",
            "container": "image",
        },
        "resources": {
            "gpus_per_node": 8,
            "agg_nodes": 1,
            "agg_workers": 1,
        },
        "benchmark": {
            "type": "sa-bench",
            "isl": 1024,
            "osl": 1024,
            "concurrencies": [8],
        },
    }

    benchmark = use_inferencex_benchmark(
        recipe,
        default_served_model_name="DeepSeek-R1-0528-NVFP4-v2",
    )["benchmark"]

    assert benchmark["env"]["MODEL_NAME"] == "DeepSeek-R1-0528-NVFP4-v2"


def test_shell_helper_allows_agentic_config_without_sequence_lengths(
    tmp_path: Path,
) -> None:
    source = tmp_path / "agentic.yaml"
    source.write_text(
        """
name: agentic
benchmark:
  type: custom
  command: bash agentic.sh
  env: {}
""",
        encoding="utf-8",
    )
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$GITHUB_WORKSPACE/runners/lib/srt_slurm.sh"; '
            'prepare_srt_benchmark "$1"',
            "bash",
            str(source),
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "GITHUB_WORKSPACE": str(repo_root),
            "CONC_LIST": "4 8",
            "RANDOM_RANGE_RATIO": "0.8",
        },
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip().endswith(".inferencex-agentic.yaml")


@pytest.mark.parametrize(
    "config_arg",
    (
        "/tmp/recipe.yaml",
        "../recipe.yaml",
        "recipes/../../recipe.yaml",
        "other/recipe.yaml",
    ),
)
def test_shell_recipe_staging_rejects_paths_outside_recipes(
    tmp_path: Path,
    config_arg: str,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$GITHUB_WORKSPACE/runners/lib/srt_slurm.sh"; '
            'stage_srt_recipe "$1"',
            "bash",
            config_arg,
        ],
        cwd=tmp_path,
        env={**os.environ, "GITHUB_WORKSPACE": str(repo_root)},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "CONFIG_FILE" in result.stderr
