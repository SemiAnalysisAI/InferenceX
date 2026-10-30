#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Let a middle pipeline stage build DeepseekV2Model (models/deepseek_v2.py).

WHAT: the gemm-output allocator sizing in `DeepseekV2Model.__init__` reads
`self.embed_tokens.embedding_dim` twice. Only the first pipeline stage (and the
last one when it carries an MTP draft layer) holds a real embedding; every other
stage has a `PPMissingLayer`, so `--pp-size` > 2 dies at load time with
AttributeError on PP rank 1. The embedding width is the hidden size, so both
reads become `config.hidden_size`; behaviour on PP1 and on every stage that has
an embedding is unchanged.

Needed by the GLM-5.2 PD example's TP1 x PP4 prefill. Not upstream as of v0.5.22.

Self-locating and idempotent; exits non-zero when the anchor drifted.
Marker: `_infera_pp_embed_dim`.
"""

import importlib.util
import py_compile
import sys
from pathlib import Path

TAG = "[deepseek-v2-pp-embed-dim]"
MARKER = "_infera_pp_embed_dim"
OLD = "self.embed_tokens.embedding_dim"
NEW = "config.hidden_size"
FIRST_OLD = f"            and {OLD} == 7168\n"
FIRST_NEW = f"            and {NEW} == 7168  # {MARKER}: PP middle stages have no embedding\n"


def main() -> int:
    spec = importlib.util.find_spec("sglang")
    if not spec or not spec.origin:
        print(f"{TAG} cannot locate sglang")
        return 1
    path = Path(spec.origin).parent / "srt" / "models" / "deepseek_v2.py"
    source = path.read_text()
    if MARKER in source:
        print(f"{TAG} already present — skipping")
        return 0
    if source.count(OLD) != 2 or source.count(FIRST_OLD) != 1:
        print(f"{TAG} anchor not found as expected in {path} — re-cut the patch")
        return 1
    path.write_text(source.replace(FIRST_OLD, FIRST_NEW, 1).replace(OLD, NEW))
    for cached in (path.parent / "__pycache__").glob(path.stem + ".*.pyc"):
        cached.unlink()
    py_compile.compile(str(path), doraise=True)
    print(f"{TAG} patched {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
