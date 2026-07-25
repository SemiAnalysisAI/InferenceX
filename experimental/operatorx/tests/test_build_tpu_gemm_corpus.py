from scripts.build_tpu_gemm_corpus import normalize_shapes, operand_bytes, pack_shapes


def _gemm(m, n, k, dtype="fp8"):
    return {
        "type": "gemm",
        "args": {
            "m": m,
            "n": n,
            "k": k,
            "dtype_a": dtype,
            "dtype_b": dtype,
            "dtype_out": dtype,
        },
    }


def test_normalize_deduplicates_dtype_variants_and_excludes_nonlocal_m():
    shapes, excluded = normalize_shapes(
        [_gemm(16, 32, 64), _gemm(16, 32, 64, "bf16"), _gemm(65, 32, 64)],
        max_m=64,
    )

    assert len(shapes) == 1
    assert shapes[0]["args"]["dtype_a"] == "bf16"
    assert shapes[0]["name"].endswith("-m16-n32-k64")
    assert len(excluded) == 1
    assert "exceeds per-kernel ceiling" in excluded[0]["exclusion_reason"]


def test_pack_shapes_respects_memory_budget_and_preserves_all_shapes():
    shapes, _ = normalize_shapes(
        [_gemm(4, 8, 16), _gemm(8, 16, 32), _gemm(16, 32, 64)],
        max_m=64,
    )
    budget = max(operand_bytes(shape) for shape in shapes) * 2
    batches = pack_shapes(shapes, budget)

    assert sorted(shape["name"] for batch in batches for shape in batch) == sorted(
        shape["name"] for shape in shapes
    )
    assert all(sum(operand_bytes(shape) for shape in batch) <= budget for batch in batches)
