"""NaN-guard tl.argmax in the spec-decode rejection sampler.

NaN target logits make tl.argmax return an out-of-range block index (into the
padded region), which OOB-reads the local argmax buffer and aborts the queue
with HSA_STATUS_ERROR_EXCEPTION 0x1016 on the first live multi-token verify.
Map NaN to -inf so argmax stays in range. Idempotent.
"""

import py_compile
import shutil

F = (
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/"
    "rejection_sampler_utils.py"
)
s = open(F).read()

if "NaN breaks tl.argmax index bounds" in s:
    print("  rejection NaN guard already present")
else:
    shutil.copy2(F, F + ".pre_nan_guard.bak")

    a1 = "    max_block_idx = tl.argmax(local_max, axis=0)"
    n1 = (
        "    # See _insert_resampled_kernel: NaN breaks tl.argmax index bounds.\n"
        '    local_max = tl.where(local_max != local_max, float("-inf"), local_max)\n'
        "    max_block_idx = tl.argmax(local_max, axis=0)"
    )
    assert s.count(a1) == 1, f"target argmax anchor count {s.count(a1)}"
    s = s.replace(a1, n1, 1)

    a2 = "    resampled_max_block_idx = tl.argmax(resampled_local_max, axis=0)"
    n2 = (
        "    # NaN max values (from NaN target logits) make tl.argmax return an\n"
        "    # out-of-range block index (into the padded region), causing an OOB read\n"
        "    # of resampled_local_argmax. Map NaN to -inf so argmax stays in range.\n"
        "    resampled_local_max = tl.where(\n"
        "        resampled_local_max != resampled_local_max,\n"
        '        float("-inf"),\n'
        "        resampled_local_max,\n"
        "    )\n"
        "    resampled_max_block_idx = tl.argmax(resampled_local_max, axis=0)"
    )
    assert s.count(a2) == 1, f"resampled argmax anchor count {s.count(a2)}"
    s = s.replace(a2, n2, 1)

    open(F, "w").write(s)
    py_compile.compile(F, doraise=True)
    print("  rejection NaN guard applied (2 sites)")
