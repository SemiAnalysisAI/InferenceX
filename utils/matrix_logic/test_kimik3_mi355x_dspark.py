from pathlib import Path
import re

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
MTP_WRAPPER = (
    REPO_ROOT / "benchmarks/single_node/agentic/kimik3_fp4_mi355x_mtp.sh"
)
BASE_RECIPE = REPO_ROOT / "benchmarks/single_node/agentic/kimik3_fp4_mi355x.sh"
MASTER_CONFIG = REPO_ROOT / "configs/amd-master.yaml"


def _shell_default(script: str, variable: str) -> str:
    match = re.search(
        rf'export {re.escape(variable)}="\$\{{{re.escape(variable)}:-(.*?)\}}"',
        script,
    )
    assert match is not None, f"{variable} must have an overridable default"
    return match.group(1)


def test_dspark_wrapper_matches_oren_rocm_baseline() -> None:
    wrapper = MTP_WRAPPER.read_text()

    assert _shell_default(wrapper, "KV_CACHE_DTYPE") == "auto"
    assert _shell_default(wrapper, "GPU_MEM_UTIL") == "0.95"
    assert _shell_default(wrapper, "MAX_NUM_SEQS") == "16"
    assert _shell_default(wrapper, "EVAL_MAX_NUM_SEQS") == "128"
    assert _shell_default(wrapper, "MAX_NUM_BATCHED_TOKENS") == "4096"
    assert _shell_default(wrapper, "LANGUAGE_MODEL_ONLY") == "false"
    assert _shell_default(wrapper, "PREFIX_CACHING") == "auto"
    assert _shell_default(wrapper, "ENFORCE_EAGER") == "false"


def test_dspark_defaults_to_real_block_rejection() -> None:
    recipe = BASE_RECIPE.read_text()
    spec_section = recipe.split("# ---- Eval-only path", maxsplit=1)[0].split(
        "SPEC_ARGS=()", maxsplit=1
    )[1]

    assert '\\"rejection_sample_method\\":\\"block\\"' in spec_section
    assert "synthetic_acceptance_length" not in spec_section


def test_reference_prefix_cache_mode_emits_no_override() -> None:
    recipe = BASE_RECIPE.read_text()
    prefix_section = recipe.split("# The upstream DSpark config", maxsplit=1)[0].split(
        "PREFIX_CACHE_ARGS=", maxsplit=1
    )[1]

    assert "auto)" in prefix_section
    assert "PREFIX_CACHE_ARGS=()" in prefix_section


def test_dspark_diagnostic_matrix_is_gpu_only_c1() -> None:
    master = yaml.safe_load(MASTER_CONFIG.read_text())
    search_space = master["kimik3-fp4-mi355x-vllm-agentic-dspark"]["scenarios"][
        "agentic-coding"
    ][0]["search-space"]

    assert search_space == [
        {
            "tp": 8,
            "kv-offloading": "none",
            "spec-decoding": "mtp",
            "conc-list": [1],
        }
    ]
