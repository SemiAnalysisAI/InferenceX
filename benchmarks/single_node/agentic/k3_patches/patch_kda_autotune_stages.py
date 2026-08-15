"""Make the KDA recompute_w_u autotune stage list env-selectable.

vLLM #50649 drops num_stages=4 for the wide gfx950 path, but stage 4 materially
improves tail latency at low concurrency. Gate it: VLLM_K3_KDA_SAFE_STAGES=1
selects [2, 3] (needed for the wide conc-8+ path), default keeps [2, 3, 4].
Idempotent.
"""

import py_compile
import shutil

F = (
    "/usr/local/lib/python3.12/dist-packages/vllm/models/kimi_k3/amd/ops/"
    "third_party/kda/chunk.py"
)
s = open(F).read()

if "_RECOMPUTE_W_U_NUM_STAGES" in s:
    print("  KDA autotune stage gate already present")
else:
    shutil.copy2(F, F + ".pre_stages.bak")

    anchor = "NUM_WARPS_AUTOTUNE = [2, 4, 8, 16] if is_amd else [4, 8, 16, 32]"
    assert s.count(anchor) == 1, f"NUM_WARPS_AUTOTUNE anchor count {s.count(anchor)}"
    block = (
        anchor
        + "\n\n"
        + "# vLLM #50649 excludes num_stages=4 for the wide c8 gfx950 path. Keep\n"
        + "# the original stage-4 candidate for c1/c4, where it materially improves tail\n"
        + "# latency, and select the safe list before Triton builds the autotune configs.\n"
        + '_RECOMPUTE_W_U_SAFE_STAGES = os.environ.get(\n'
        + '    "VLLM_K3_KDA_SAFE_STAGES", "0"\n'
        + ').lower() in ("1", "true", "yes", "on")\n'
        + "_RECOMPUTE_W_U_NUM_STAGES = (\n"
        + "    [2, 3] if _RECOMPUTE_W_U_SAFE_STAGES else [2, 3, 4]\n"
        + ")"
    )
    s = s.replace(anchor, block, 1)

    if "\nimport os\n" not in s:
        assert s.count("\nimport torch\n") >= 1, "torch import anchor"
        s = s.replace("\nimport torch\n", "\nimport os\n\nimport torch\n", 1)

    # Two autotune lists share this line; only the recompute_w_u one (keyed on
    # H/K/V/BT/BK/BV/IS_VARLEN) is gated, so anchor on that key list.
    old = (
        "        for num_stages in [2, 3, 4]\n"
        "    ],\n"
        '    key=["H", "K", "V", "BT", "BK", "BV", "IS_VARLEN"],'
    )
    new = (
        "        for num_stages in _RECOMPUTE_W_U_NUM_STAGES\n"
        "    ],\n"
        '    key=["H", "K", "V", "BT", "BK", "BV", "IS_VARLEN"],'
    )
    assert s.count(old) == 1, f"recompute_w_u anchor count {s.count(old)}"
    s = s.replace(old, new, 1)

    open(F, "w").write(s)
    py_compile.compile(F, doraise=True)
    print("  KDA autotune stage gate applied")
