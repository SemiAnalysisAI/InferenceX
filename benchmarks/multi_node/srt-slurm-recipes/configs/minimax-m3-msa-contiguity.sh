#!/usr/bin/env bash

set -euo pipefail

# The 0618 image slices a persistent CUDA-graph top-k buffer without
# materializing it. TP1 data-parallel-attention workers retain a non-contiguous
# head stride, which the MiniMax M3 MSA CSR builder rejects.
python3 - <<'PYEOF'
import importlib.util
import pathlib

spec = importlib.util.find_spec("vllm")
if spec is None or not spec.submodule_search_locations:
    raise RuntimeError("Could not locate the installed vllm package")

target = (
    pathlib.Path(next(iter(spec.submodule_search_locations)))
    / "models"
    / "minimax_m3"
    / "nvidia"
    / "sparse_attention_msa.py"
)
src = target.read_text()
old = "            prefill_topk = topk[:, nd:num_tokens, :]\n"
new = "            prefill_topk = topk[:, nd:num_tokens, :].contiguous()\n"

if new in src:
    print(f"[minimax-m3-msa-patch] already applied: {target}")
elif src.count(old) == 1:
    target.write_text(src.replace(old, new, 1))
    print(f"[minimax-m3-msa-patch] patched: {target}")
else:
    raise RuntimeError(f"Expected exactly one patch anchor in {target}")
PYEOF
