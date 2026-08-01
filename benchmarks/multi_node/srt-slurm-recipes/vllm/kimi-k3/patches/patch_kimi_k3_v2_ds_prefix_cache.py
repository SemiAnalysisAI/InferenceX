"""Backport the vLLM #49291 V2 DS-conv prefix-cache enablement."""

from __future__ import annotations

import importlib.util
from pathlib import Path


OLD_IMPORT = """from vllm.model_executor.layers.mamba.mamba_utils import (
    get_conv_copy_spec,
    is_conv_state_dim_first,
)
"""

OLD_GUARD = """            # The fused copy kernels shift conv windows assuming the SD layout;
            # the DS layout cannot express a >0 spec-decode shift as a single
            # contiguous copy (mirrors get_conv_copy_spec's NotImplementedError).
            if get_conv_copy_spec in copy_funcs and is_conv_state_dim_first():
                assert self.vllm_config.speculative_config is None, (
                    "DS conv state layout does not support mamba align state "
                    "copies with speculative decoding"
                )
"""

NEW_COMMENT = """            # Both SD and DS conv layouts support a >0 spec-decode shift: the
            # fused pre-copy kernel applies the accepted-token window shift per
            # layout. Backported from vLLM #49291 for the V2 model runner.
"""


def main() -> None:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Cannot locate the installed vllm package")

    package_dir = Path(next(iter(spec.submodule_search_locations)))
    target = package_dir / "v1/worker/gpu/model_states/mamba_hybrid.py"
    source = target.read_text()

    if OLD_IMPORT not in source or OLD_GUARD not in source:
        if "DS conv state layout does not support mamba align state copies" not in source:
            print("[kimi-k3-v2-ds-prefix-cache] Already patched, skipping.")
            return
        raise RuntimeError(f"Unexpected vLLM source layout in {target}")

    patched = source.replace(OLD_IMPORT, "", 1).replace(OLD_GUARD, NEW_COMMENT, 1)
    compile(patched, str(target), "exec")
    target.write_text(patched)
    print(f"[kimi-k3-v2-ds-prefix-cache] Patched {target}")


if __name__ == "__main__":
    main()
