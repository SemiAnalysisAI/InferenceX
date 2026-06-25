#!/usr/bin/env bash
set -euo pipefail

# MiniMax-M3 ModelOpt NVFP4 support from vllm-project/vllm#46380.
VLLM_NVFP4_COMMIT="6c08558112acd2fd8b4bfc270104d556eb77f9bf"
VLLM_ROOT="$(python3 -c 'import os, vllm; print(os.path.dirname(vllm.__file__))')"
for file in \
    model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py \
    model_executor/layers/quantization/modelopt.py \
    model_executor/layers/quantization/utils/flashinfer_utils.py
do
    curl -fsSL \
        "https://raw.githubusercontent.com/vllm-project/vllm/${VLLM_NVFP4_COMMIT}/vllm/${file}" \
        -o "${VLLM_ROOT}/${file}"
done

python3 - <<'PYEOF'
from importlib.util import find_spec
from pathlib import Path

spec = find_spec("vllm")
if not spec or not spec.origin:
    raise RuntimeError("vllm is not installed")
path = Path(spec.origin).parent / "models/minimax_m3/nvidia/sparse_attention_msa.py"
source = path.read_text()
old = "            prefill_topk = topk[:, nd:num_tokens, :]\n"
new = "            prefill_topk = topk[:, nd:num_tokens, :].contiguous()\n"
if new not in source:
    if source.count(old) != 1:
        raise RuntimeError(f"missing or ambiguous patch anchor in {path}")
    path.write_text(source.replace(old, new, 1))
PYEOF

python3 -c \
    "from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import TrtLlmNvFp4ExpertsModular; print('[nvfp4-patch] OK')"
