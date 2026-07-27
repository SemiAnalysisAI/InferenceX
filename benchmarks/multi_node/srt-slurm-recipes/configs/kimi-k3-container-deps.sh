#!/bin/bash
# Setup script for the Kimi-K3 vLLM bring-up image (vllm/vllm-openai:kimi-k3).
# srt-slurm runs this in every worker container before dynamo install and
# worker startup (recipe field: setup_script).

set -euo pipefail

# The image's first decode step crashes in the KDA hybrid-state postprocess:
#   vllm/v1/worker/gpu/model_states/mamba_hybrid.py, postprocess_state:
#   IndexError: index_fill_(): Expected dtype int64 for index.
# torch's index_fill_ requires an int64 index tensor, but the runner passes
# the int32 idx_mapping (hit by moonshotai/Kimi-K3 agentic bring-up, first
# decode step, engine v0.1.dev19262+gb6bbf29dd). Coerce the index to int64.
# Idempotent: exits 0 if the patch is already applied.
python3 - <<'PY'
import pathlib
import re

import vllm.v1.worker.gpu.model_states.mamba_hybrid as mh

path = pathlib.Path(mh.__file__)
src = path.read_text()
if "idx_mapping.long()" in src:
    print(f"mamba_hybrid index_fill_ patch already applied: {path}")
    raise SystemExit(0)

new, n = re.subn(
    r"index_fill_\(\s*0,\s*idx_mapping,",
    "index_fill_(0, idx_mapping.long(),",
    src,
)
if n != 1:
    raise SystemExit(
        f"expected exactly one index_fill_(0, idx_mapping, ...) call in "
        f"{path}, found {n} — image layout changed, refusing to patch"
    )
path.write_text(new)
print(f"Patched mamba_hybrid index_fill_ index dtype: {path}")
PY
