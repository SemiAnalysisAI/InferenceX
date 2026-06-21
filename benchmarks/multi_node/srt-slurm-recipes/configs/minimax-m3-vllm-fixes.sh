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
    root / "models/minimax_m3/nvidia/sparse_attention_msa.py": [
        (
            "            prefill_topk = topk[:, nd:num_tokens, :]\n",
            "            prefill_topk = topk[:, nd:num_tokens, :].contiguous()\n",
        ),
    ],
    root / "distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py": [
        (
            "            for i, local_len in enumerate(self.block_len_per_layer):\n",
            "            total_kv_heads = self.transfer_topo.total_num_kv_heads\n"
            "            local_heads = self.transfer_topo.local_physical_heads\n"
            "            remote_heads = max(1, total_kv_heads // remote_tp_size)\n"
            "            for i, local_len in enumerate(self.block_len_per_layer):\n",
        ),
        (
            "remote_len == (local_len * tp_ratio) // block_size_ratio,",
            "remote_len == (local_len * remote_heads // local_heads) "
            "// block_size_ratio,",
        ),
        (
            "remote_len == local_len // (-tp_ratio),",
            "remote_len == local_len * remote_heads // local_heads,",
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
