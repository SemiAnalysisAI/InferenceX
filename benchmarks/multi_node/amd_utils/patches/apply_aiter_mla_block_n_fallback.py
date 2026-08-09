#!/usr/bin/env python3
"""Stop aiter's fp8 MLA decode from KeyError-ing on untabulated nhead*qlen.

``aiter/mla.py:get_meta_param`` sizes the KV-split search from a hard-coded table::

    get_block_n_fp8 = {8:64, 16:128, 24:128, 32:128, 48:64,
                       64:64, 128:32, 256:32, 384:32, 512:32}
    min_block_n = get_block_n_fp8[int(nhead * max_seqlen_q)]

so the fp8 decode path only accepts ``nhead * max_seqlen_q`` values that happen to
be in that table. Kimi-K3 at TP8 pads to 16 heads, which covers qlen 1/2/3/4/8/16
but not, for example, the qlen the spec-decode warmup step uses -- that lands on
16*15 = 240 and dies with ``KeyError: 240`` during compile_or_warm_up_model.

``min_block_n`` only bounds ``num_kv_splits`` (a scheduling heuristic); it does
not enter the math. Falling back to the nearest tabulated key is therefore safe
-- worst case the split count is picked from a neighbouring bucket -- and it is
far better than refusing to run.

Idempotent. Fails loudly if the anchor moves.
"""

from __future__ import annotations

import sys
from pathlib import Path

MARKER = "# INFERENCEX-BLOCK-N-FALLBACK"

ANCHOR = "        min_block_n = get_block_n_fp8[int(nhead * max_seqlen_q)]"

REPLACEMENT = f"""        {MARKER}
        _k = int(nhead * max_seqlen_q)
        if _k in get_block_n_fp8:
            min_block_n = get_block_n_fp8[_k]
        else:
            # Nearest tabulated key. min_block_n only caps num_kv_splits, so an
            # adjacent bucket costs at most a slightly different split count.
            _nk = min(get_block_n_fp8, key=lambda t: (abs(t - _k), t))
            min_block_n = get_block_n_fp8[_nk]"""


def main() -> None:
    import aiter  # noqa: PLC0415

    path = Path(aiter.__file__).parent / "mla.py"
    if not path.is_file():
        sys.exit(f"[block-n-fallback] ERROR: not found: {path}")

    src = path.read_text()
    if MARKER in src:
        print(f"[block-n-fallback] already applied: {path}")
        return

    n = src.count(ANCHOR)
    if n != 1:
        sys.exit(
            f"[block-n-fallback] ERROR: anchor matched {n} times (expected 1) "
            f"in {path}. aiter changed upstream -- update this patch."
        )

    backup = path.with_suffix(path.suffix + ".inferencex.bak")
    if not backup.exists():
        backup.write_text(src)
    path.write_text(src.replace(ANCHOR, REPLACEMENT, 1))

    import ast

    ast.parse(path.read_text())
    print(f"[block-n-fallback] Patched {path}")


if __name__ == "__main__":
    main()
