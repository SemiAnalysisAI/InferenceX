#!/usr/bin/env python3
"""Accept non-contiguous / multi-dim `state_indices` in fused_recurrent_kda.

The vendored K3 AMD KDA kernel wrapper rejects any `state_indices` that is not
already 1-D and unit-stride:

    if state_indices.ndim != 1 or state_indices.stride(0) != 1:
        raise ValueError("`state_indices` must be contiguous and one-dimensional.")

Under a captured cudagraph the caller always hands it a flat persistent buffer,
so the check never fires. On the eager warmup path -- warmup.py:369
warmup_kernels -> _run_decode_step, which builds mixed spec / non-spec batches
that no graph covers -- the spec state indices arrive as a view, and the check
is reached. Rather than reject, flatten it.

This is a normalisation, not a semantic change: reshape(-1).contiguous() is a
no-op on the shapes the graph path already supplies.
"""

from __future__ import annotations

import argparse
import difflib
import py_compile
import sys

REL = "vllm/models/kimi_k3/amd/ops/third_party/kda/fused_recurrent.py"

OLD = '''    if state_indices.ndim != 1 or state_indices.stride(0) != 1:
        raise ValueError("`state_indices` must be contiguous and one-dimensional.")'''

NEW = '''    if state_indices.ndim != 1 or state_indices.stride(0) != 1:
        state_indices = state_indices.reshape(-1).contiguous()'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diff-out")
    args = ap.parse_args()

    import vllm  # noqa: F401  (resolve the installed tree, not a source checkout)
    import os

    root = os.path.dirname(os.path.dirname(vllm.__file__))
    path = os.path.join(root, REL)

    if not os.path.isfile(path):
        print(f"ERROR: {path} missing", file=sys.stderr)
        return 1

    src = open(path).read()

    if NEW in src:
        print(f"vllm_kda_state_indices.py: already patched {path}")
        return 0

    n = src.count(OLD)
    if n != 1:
        # Loud, not silent: an unpatched run would be indistinguishable from
        # the runs this patch exists to change.
        print(
            f"ERROR: expected exactly 1 anchor match in {path}, found {n}. "
            "The installed file has drifted; re-derive the anchor.",
            file=sys.stderr,
        )
        return 1

    out = src.replace(OLD, NEW)
    open(path, "w").write(out)
    py_compile.compile(path, doraise=True)

    if args.diff_out:
        with open(args.diff_out, "w") as fh:
            fh.writelines(
                difflib.unified_diff(
                    src.splitlines(True), out.splitlines(True),
                    fromfile=f"a/{REL}", tofile=f"b/{REL}",
                )
            )

    print(f"vllm_kda_state_indices.py: patched {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
