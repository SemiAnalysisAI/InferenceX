#!/usr/bin/env python3
"""Graft ROCm/aiter#4417 ("Fix large-token FlyDSL MoE launch and output limits")
onto the aiter shipped in the 2026-08-12 vLLM ROCm nightly.

Why not `git apply` the upstream diff: the nightly's aiter predates the
`dict[str, dict]` typing modernization, so every hunk's context is off by the
`Dict[str, Dict]` spelling. The substance of #4417 is five small anchored edits,
so we apply those directly. Idempotent — re-running is a no-op.

Without this patch, DSv4-Pro FP8/FP4 stage2 uses buffer atomics whose 32-bit
byte offsets overflow once token_num*model_dim*2 exceeds 4 GiB, which shows up
as `Memory access fault by GPU node-N ... Reason: Unknown` during the vLLM
profile run (determine_available_memory), killing the engine core.
"""

import sys

HELPERS = '''
# HIP limits grid.y/grid.z to 65535.
_HIP_MAX_GRID_DIM_Y = 65535


def resolve_flydsl_grid_y_persist_m(
    num_m_blocks: int, requested_persist_m: int = 0
) -> int:
    """Increase persist_m as needed to keep grid.y within HIP's limit."""
    num_m_blocks = max(int(num_m_blocks), 0)
    requested_persist_m = max(int(requested_persist_m), 1)
    required_persist_m = max(
        1, (num_m_blocks + _HIP_MAX_GRID_DIM_Y - 1) // _HIP_MAX_GRID_DIM_Y
    )
    return max(requested_persist_m, required_persist_m)


def requires_flydsl_stage2_reduce(
    token_num: int, model_dim: int, element_size: int
) -> bool:
    """Return whether stage2 atomic output exceeds 32-bit byte offsets."""
    return int(token_num) * int(model_dim) * int(element_size) > 0xFFFFFFFF


'''

# (description, exact old text, new text) — each old text must be unique.
EDITS = [
    (
        "add _HIP_MAX_GRID_DIM_Y + the two guard helpers",
        "def resolve_flydsl_stage2_tile_k(inter_dim: int, tile_k: int) -> int:",
        HELPERS.lstrip("\n")
        + "def resolve_flydsl_stage2_tile_k(inter_dim: int, tile_k: int) -> int:",
    ),
    (
        "stage1: cap grid.y via persist_m",
        "    _persist_m = persist_m if persist_m > 0 else 1",
        "    _persist_m = resolve_flydsl_grid_y_persist_m(_grid_y, persist_m)",
    ),
    (
        "stage2: switch to reduce when atomic output exceeds 4 GiB",
        '    if os.environ.get("AITER_FLYDSL_FORCE_REDUCE", "0") == "1":\n'
        "        mode = \"reduce\"\n",
        '    if os.environ.get("AITER_FLYDSL_FORCE_REDUCE", "0") == "1":\n'
        '        mode = "reduce"\n'
        "    elif (\n"
        '        mode != "reduce"\n'
        "        and not return_per_slot\n"
        "        and requires_flydsl_stage2_reduce(token_num, model_dim, 2)\n"
        "    ):\n"
        "        # Buffer atomics use 32-bit offsets; reduce outputs larger than 4 GiB.\n"
        '        mode = "reduce"\n',
    ),
    (
        "stage2 fp8: cap grid.y via persist_m",
        '    if a_dtype == "fp8":\n        _persist_m = 1\n',
        '    if a_dtype == "fp8":\n'
        "        # FP8 uses non-persistent scheduling, so cap grid.y via persist_m.\n"
        "        _persist_m = resolve_flydsl_grid_y_persist_m(m_blocks)\n",
    ),
]


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        import importlib.util as u

        spec = u.find_spec("aiter.ops.flydsl.moe_kernels")
        if spec is None or spec.origin is None:
            print("FAIL: cannot locate aiter.ops.flydsl.moe_kernels", file=sys.stderr)
            return 1
        path = spec.origin

    src = open(path).read()

    if "requires_flydsl_stage2_reduce" in src and "resolve_flydsl_grid_y_persist_m" in src:
        print(f"SKIP: aiter#4417 already present in {path}")
        return 0

    for desc, old, new in EDITS:
        n = src.count(old)
        if n != 1:
            print(f"FAIL: anchor for '{desc}' matched {n} times (want 1)", file=sys.stderr)
            return 1
        src = src.replace(old, new, 1)
        print(f"  ok: {desc}")

    open(path, "w").write(src)
    print(f"OK: grafted aiter#4417 into {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
