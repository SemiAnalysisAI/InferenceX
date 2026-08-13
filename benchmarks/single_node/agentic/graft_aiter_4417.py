#!/usr/bin/env python3
"""Graft ROCm/aiter #4417 into an aiter that predates it.

#4417 "Fix large-token FlyDSL MoE launch and output limits" (merged 2026-07-30,
single file, +32/-2) adds two guards to aiter/ops/flydsl/moe_kernels.py:

  * requires_flydsl_stage2_reduce() -- stage2 buffer atomics address the output
    with 32-bit byte offsets, so a >4 GiB output silently walks off the end.
    The guard flips mode to "reduce" before that happens.
  * resolve_flydsl_grid_y_persist_m() -- HIP caps grid.y at 65535; the guard
    raises persist_m enough to keep the launch legal.

Neither guard fires at the DSv4-Pro TP8 shape this campaign measures --
requires_flydsl_stage2_reduce(65536, 7168, 2) is False (~939 MB, well under the
4 GiB threshold) -- so #4417 does NOT explain the inter_dim=384 profile-run
memfault. That fault is a stage2 tile_k mismatch: the tuned CSV names
opus_moe2_*_t64x256x256_* for inter_dim=384, and resolve_flydsl_stage2_tile_k
only guards flydsl_* kernel names, so tile_k=256 runs against K=384. #4417 is
grafted anyway because it is a genuine gap in the base that any larger-token
sweep row would hit.

The upstream .diff does NOT apply to the 08-12 nightly base: that aiter predates
both #4417 and aiter's typing modernization, so every context line still reads
`Dict[str, Dict]` / `Optional[X]` where the diff expects `dict[str, dict]` /
`X | None`. The three hunks are grafted here by anchor instead of by context.
Idempotent: re-running on an already-grafted (or already-new) file is a no-op.
"""

import sys

HELPERS = '''

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


def graft(path):
    src = open(path).read()
    if "requires_flydsl_stage2_reduce" in src and "_HIP_MAX_GRID_DIM_Y" in src:
        print("SKIP already grafted/present:", path)
        return 0
    orig = src
    done = []

    # hunk 1a: the grid.y constant, right after the _KERNEL_PARAMS declaration.
    a = "_KERNEL_PARAMS: Dict[str, Dict] = {}\n"
    if a not in src:
        a = "_KERNEL_PARAMS: dict[str, dict] = {}\n"
    if src.count(a) != 1:
        print("FAIL anchor _KERNEL_PARAMS count=", src.count(a))
        return 1
    src = src.replace(a, a + "\n# HIP limits grid.y/grid.z to 65535.\n"
                          "_HIP_MAX_GRID_DIM_Y = 65535\n", 1)
    done.append("const")

    # hunk 1b: the two helpers, ahead of resolve_flydsl_stage2_tile_k.
    b = "\ndef resolve_flydsl_stage2_tile_k("
    if src.count(b) != 1:
        print("FAIL anchor stage2_tile_k count=", src.count(b))
        return 1
    src = src.replace(b, HELPERS + "\ndef resolve_flydsl_stage2_tile_k(", 1)
    done.append("helpers")

    # hunk 2: stage1 caps grid.y through persist_m.
    c = "    _persist_m = persist_m if persist_m > 0 else 1\n"
    if src.count(c) != 1:
        print("FAIL anchor stage1 persist_m count=", src.count(c))
        return 1
    src = src.replace(c, "    _persist_m = resolve_flydsl_grid_y_persist_m(_grid_y, persist_m)\n", 1)
    done.append("stage1_persist")

    # hunk 3a: stage2 falls back to reduce when the atomic output exceeds 4 GiB.
    d = ('    if os.environ.get("AITER_FLYDSL_FORCE_REDUCE", "0") == "1":\n'
         '        mode = "reduce"\n')
    if src.count(d) != 1:
        print("FAIL anchor FORCE_REDUCE count=", src.count(d))
        return 1
    src = src.replace(d, d + (
        "    elif (\n"
        "        mode != \"reduce\"\n"
        "        and not return_per_slot\n"
        "        and requires_flydsl_stage2_reduce(token_num, model_dim, 2)\n"
        "    ):\n"
        "        # Buffer atomics use 32-bit offsets; reduce outputs larger than 4 GiB.\n"
        "        mode = \"reduce\"\n"), 1)
    done.append("stage2_reduce")

    # hunk 3b: fp8 stage2 is non-persistent, so cap grid.y the same way.
    e = '    if a_dtype == "fp8":\n        _persist_m = 1\n'
    if src.count(e) != 1:
        print("FAIL anchor fp8 persist_m count=", src.count(e))
        return 1
    src = src.replace(e, '    if a_dtype == "fp8":\n'
                         '        # FP8 uses non-persistent scheduling, so cap grid.y via persist_m.\n'
                         '        _persist_m = resolve_flydsl_grid_y_persist_m(m_blocks)\n', 1)
    done.append("stage2_fp8_persist")

    if src == orig:
        print("FAIL no change")
        return 1
    open(path + ".pre4417", "w").write(orig)
    open(path, "w").write(src)
    print("GRAFTED", path, "hunks:", ",".join(done))
    return 0


if __name__ == "__main__":
    sys.exit(graft(sys.argv[1]))
