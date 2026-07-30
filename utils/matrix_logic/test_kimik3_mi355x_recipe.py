from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_RECIPE = (
    REPO_ROOT / "benchmarks/single_node/agentic/kimik3_fp4_mi355x.sh"
).read_text()
MTP_RECIPE = (
    REPO_ROOT / "benchmarks/single_node/agentic/kimik3_fp4_mi355x_mtp.sh"
).read_text()


def test_dspark_routes_target_mla_away_from_rocm_aiter() -> None:
    assert 'ATTENTION_BACKEND="${ATTENTION_BACKEND:-TRITON_MLA}"' in MTP_RECIPE
    assert 'ATTENTION_BACKEND_ARGS=(--attention-backend "$ATTENTION_BACKEND")' in (
        BASE_RECIPE
    )
    assert '"${ATTENTION_BACKEND_ARGS[@]}"' in BASE_RECIPE
