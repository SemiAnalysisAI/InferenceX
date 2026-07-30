from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_RECIPE = (
    REPO_ROOT / "benchmarks/single_node/agentic/kimik3_fp4_mi355x.sh"
).read_text()
MTP_RECIPE = (
    REPO_ROOT / "benchmarks/single_node/agentic/kimik3_fp4_mi355x_mtp.sh"
).read_text()


def test_dspark_patches_rocm_aiter_mla_for_multi_token_queries() -> None:
    assert (
        'if [ "${KV_CACHE_DTYPE:-}" = "fp8" ] '
        '|| [ "${SPEC_DECODE:-false}" = "true" ]; then'
        in BASE_RECIPE
    )
    assert "MTP-aware qlen/q_pos indexing" in BASE_RECIPE
    assert "ATTENTION_BACKEND" not in MTP_RECIPE
