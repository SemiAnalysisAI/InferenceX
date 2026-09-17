#!/usr/bin/env python3
"""Keep draft KV group annotation on when disable_eagle_block_drop is set."""

from __future__ import annotations

import os
from pathlib import Path
import py_compile


site = Path(
    os.environ.get(
        "VLLM_SITE",
        "/usr/local/lib/python3.12/dist-packages",
    )
)
path = site / "vllm/v1/core/kv_cache_utils.py"
marker = "K3 annotate eagle groups independent of block-drop"
source = path.read_text()
function_start = source.index("def _annotate_eagle_groups(")
function_end = source.index("\ndef ", function_start + 5)
block = source[function_start:function_end]

if marker in block:
    print(f"{path}: annotate fixup already applied")
    raise SystemExit(0)

bad = "if spec_config is None or not spec_config.use_eagle_block_drop():"
good = (
    "if spec_config is None or not spec_config.use_eagle():"
    f"  # {marker}"
)
if bad in block:
    block = block.replace(bad, good, 1)
elif "if spec_config is None or not spec_config.use_eagle():" in block:
    print(f"{path}: annotate guard already uses use_eagle()")
    raise SystemExit(0)
else:
    raise RuntimeError(f"{path}: _annotate_eagle_groups guard anchor missing")

path.write_text(source[:function_start] + block + source[function_end:])
py_compile.compile(str(path), doraise=True)
print(f"{path}: annotate fixup applied and compiled")
