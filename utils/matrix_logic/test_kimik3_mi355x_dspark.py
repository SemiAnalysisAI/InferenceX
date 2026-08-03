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


def test_dspark_wrapper_is_perf_tuned_not_reproducer_faithful() -> None:
    """Branch kimik3-dspark-perf DELIBERATELY diverges from the AMD reproducer.

    On the base branch this test asserts fidelity to the upstream Oren ROCm
    baseline (KV_CACHE_DTYPE=auto, PREFIX_CACHING=auto, MLA_ASM_PAD=0). That
    fidelity is the wrong objective for a throughput arm:

      * PREFIX_CACHING=auto means vLLM resolves the flag to False for this
        model, and every DSpark cell so far measured "Prefix cache hit rate:
        0.0%" against 91.7% on the non-DSpark asm c8 arm. On an agentic trace
        (ISL mean 335K, theoretical hit 98.1%) that is a ~300K-token prefix
        recomputed every turn.
      * KV_CACHE_DTYPE=auto leaves a 2,156,093-token pool while c8 at ~300K
        contexts wants ~2.4M, and it also skips the gist mla_gluon patch, which
        the launcher gates on this exact flag.

    So this branch asserts the perf config instead. The reproducer-fidelity
    assertions still guard the base branch; do not merge this file back.
    """
    wrapper = MTP_WRAPPER.read_text()

    # Deltas from the reproducer, each proven on the non-DSpark arm.
    assert _shell_default(wrapper, "KV_CACHE_DTYPE") == "fp8"
    assert _shell_default(wrapper, "PREFIX_CACHING") == "true"
    assert _shell_default(wrapper, "MLA_ASM_PAD") == "0"  # asm verify closed under fp8
    assert _shell_default(wrapper, "DSPARK_ASM_VERIFY") == "0"  # see wrapper comment

    # Unchanged from the reproducer.
    assert _shell_default(wrapper, "GPU_MEM_UTIL") == "0.95"
    assert _shell_default(wrapper, "MAX_NUM_SEQS") == "8"  # mla_gluon batch ceiling
    assert _shell_default(wrapper, "EVAL_MAX_NUM_SEQS") == "128"
    assert _shell_default(wrapper, "MAX_NUM_BATCHED_TOKENS") == "4096"
    assert _shell_default(wrapper, "LANGUAGE_MODEL_ONLY") == "false"
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
    search_space = master["kimik3-fp4-mi355x-vllm-agentic-mtp"]["scenarios"][
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
