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


def test_dspark_wrapper_matches_blog_baseline() -> None:
    """Aligned to vllm.ai/blog/2026-07-27-k3, not to the Oren reproducer.

    Deltas from the reproducer, each with a measured reason:
      PREFIX_CACHING true  -- the blog says --enable-prefix-caching is required
                              and not on by default; the reproducer emitted no
                              override, so DSpark ran at 0.0% hit.
      GPU_MEM_UTIL 0.88    -- 0.95 fails the startup free-memory check on
                              mi355x-amds (~272 of 288 GiB free).
      MLA_ASM_PAD 0 /      -- padded asm verify does not verify (574/574,
      DSPARK_ASM_VERIFY 0     per-position all 1.000). Gluon verify is the only
                              path that verifies.
      DSPARK_PR50619 1 /   -- vllm-project/vllm#50619's causal native MTP verify
      DSPARK_MQA_FIX 0        replaces the expanded-row path. Both patchers own
                              forward_mqa's Gluon branch, so exactly one may be
                              on. The row map they replace hands every verify
                              row the request's whole KV range, letting draft
                              token t attend t+1..7 -- the target grading the
                              draft against itself, which is what produced
                              acceptance 6.41-8.00 of 8 and 8.00 flat at conc 1.
      KV_CACHE_DTYPE auto  -- fp8 needs the rope-outside cache-write port; the
                              query-capability half of #50619 is already in the
                              patcher but is inert while KV is bf16.
    """
    wrapper = MTP_WRAPPER.read_text()

    assert _shell_default(wrapper, "PREFIX_CACHING") == "true"
    assert _shell_default(wrapper, "MLA_ASM_PAD") == "0"
    assert _shell_default(wrapper, "DSPARK_ASM_VERIFY") == "0"
    assert _shell_default(wrapper, "DSPARK_PR50619") == "1"
    assert _shell_default(wrapper, "DSPARK_MQA_FIX") == "0"
    assert "export DSPARK_MTP_NATIVE" not in wrapper, (
        "DSPARK_MTP_NATIVE had no patcher in this lineage -- exporting it made "
        "run 30875699986 silently run the stock non-causal flatten. Do not "
        "reintroduce it without the code that reads it."
    )
    assert _shell_default(wrapper, "KV_CACHE_DTYPE") == "auto"
    assert _shell_default(wrapper, "GPU_MEM_UTIL") == "0.88"
    assert _shell_default(wrapper, "MAX_NUM_SEQS") == "8"
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
