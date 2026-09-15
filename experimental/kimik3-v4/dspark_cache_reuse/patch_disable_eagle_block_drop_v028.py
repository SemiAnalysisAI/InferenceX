#!/usr/bin/env python3
"""Backport the local no-drop cache option used by Kimi-K3 DSpark."""

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
marker = "K3 v0.28 disable EAGLE trailing block drop"


def replace_once(path: Path, old: str, new: str) -> None:
    source = path.read_text()
    if new in source:
        return
    if source.count(old) != 1:
        raise RuntimeError(
            f"{path}: expected one anchor, found {source.count(old)}"
        )
    path.write_text(source.replace(old, new, 1))


config = site / "vllm/config/speculative.py"
replace_once(
    config,
    """    use_local_argmax_reduction: bool = False
""",
    f"""    disable_eagle_block_drop: bool = False
    \"\"\"{marker}. Keep the trailing prefix-cache block instead of applying
    the conservative EAGLE-family drop. Target verification remains enabled.\"\"\"
    use_local_argmax_reduction: bool = False
""",
)
replace_once(
    config,
    """    def use_dflash(self) -> bool:
""",
    """    def use_eagle_block_drop(self) -> bool:
        return self.use_eagle() and not self.disable_eagle_block_drop

    def use_dflash(self) -> bool:
""",
)

kv_utils = site / "vllm/v1/core/kv_cache_utils.py"
source = kv_utils.read_text()
function_start = source.index("def _annotate_eagle_groups(")
function_end = source.index("\ndef ", function_start + 5)
block = source[function_start:function_end]
old = "if spec_config is None or not spec_config.use_eagle():"
new = "if spec_config is None or not spec_config.use_eagle_block_drop():"
if new not in block:
    if block.count(old) != 1:
        raise RuntimeError("_annotate_eagle_groups block-drop anchor missing")
    block = block.replace(old, new, 1)
    kv_utils.write_text(source[:function_start] + block + source[function_end:])

scheduler = site / "vllm/v1/core/sched/scheduler.py"
replace_once(
    scheduler,
    """        self.use_eagle = False
        self.num_spec_tokens = vllm_config.num_speculative_tokens
""",
    """        self.use_eagle = False
        self.use_eagle_block_drop = False
        self.num_spec_tokens = vllm_config.num_speculative_tokens
""",
)
replace_once(
    scheduler,
    """            self.use_eagle = speculative_config.use_eagle()
            if self.use_eagle:
""",
    """            self.use_eagle = speculative_config.use_eagle()
            self.use_eagle_block_drop = (
                speculative_config.use_eagle_block_drop()
            )
            if self.use_eagle:
""",
)
replace_once(
    scheduler,
    """            use_eagle=self.use_eagle,
            num_prefill_lookahead=self.num_prefill_lookahead,
""",
    """            use_eagle=self.use_eagle_block_drop,
            num_prefill_lookahead=self.num_prefill_lookahead,
""",
)
source = scheduler.read_text()
start = source.index("    def _mamba_block_aligned_split(")
end = source.index("\n    def ", start + 8)
block = source[start:end]
old_count = block.count("self.use_eagle")
if old_count:
    block = block.replace("self.use_eagle", "self.use_eagle_block_drop")
    scheduler.write_text(source[:start] + block + source[end:])
elif "self.use_eagle_block_drop" not in block:
    raise RuntimeError("mamba split EAGLE guard missing")

for relative in (
    "vllm/config/speculative.py",
    "vllm/v1/core/kv_cache_utils.py",
    "vllm/v1/core/sched/scheduler.py",
):
    py_compile.compile(str(site / relative), doraise=True)
    print(f"py_compile OK: {relative}")
print("Kimi-K3 DSpark no-drop option ready")
