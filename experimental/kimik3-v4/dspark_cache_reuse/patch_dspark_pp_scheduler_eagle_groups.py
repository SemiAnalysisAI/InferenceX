#!/usr/bin/env python3
"""Preserve global DSpark draft-group semantics through PP projection."""

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
marker = "K3 PP projection preserves global DSpark draft group"
source = path.read_text()
if marker in source:
    print(f"{path}: already patched")
    raise SystemExit(0)

old = """                is_eagle_group=group.is_eagle_group and bool(worker_layer_names),
"""
new = f"""                # {marker}. This flag describes the global cache group,
                # not whether this PP worker owns one of its layers. Keep
                # layer_names worker-local while retaining draft semantics so
                # the scheduler does not flag every target/Mamba group.
                is_eagle_group=group.is_eagle_group,
"""
if source.count(old) != 1:
    raise RuntimeError(
        f"{path}: expected one PP projection anchor, found {source.count(old)}"
    )
path.write_text(source.replace(old, new, 1))
py_compile.compile(str(path), doraise=True)
print(f"{path}: patched and compiled")
