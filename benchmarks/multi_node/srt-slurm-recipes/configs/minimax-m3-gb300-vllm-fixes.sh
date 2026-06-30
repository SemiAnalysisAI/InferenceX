#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PYEOF'
from importlib.util import find_spec
from pathlib import Path

spec = find_spec("vllm")
if not spec or not spec.origin:
    raise RuntimeError("vllm is not installed")
root = Path(spec.origin).parent
patches = {
    root / "distributed/device_communicators/flashinfer_all_reduce.py": [
        (
            "            comm_backend=comm_backend,\n"
            "            group=group,\n",
            "            comm_backend=comm_backend,\n"
            '            force_oneshot_support=backend == "mnnvl",\n'
            "            group=group,\n",
        ),
    ],
    root / "models/minimax_m3/nvidia/sparse_attention_msa.py": [
        (
            "            prefill_topk = topk[:, nd:num_tokens, :]\n",
            "            prefill_topk = topk[:, nd:num_tokens, :].contiguous()\n",
        ),
    ],
}
for path, edits in patches.items():
    source = path.read_text()
    for old, new in edits:
        if new in source:
            continue
        if source.count(old) != 1:
            raise RuntimeError(f"missing or ambiguous patch anchor in {path}")
        source = source.replace(old, new, 1)
    path.write_text(source)
PYEOF
