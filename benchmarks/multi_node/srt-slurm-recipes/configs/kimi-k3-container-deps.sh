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

# The DSpark draft head (Inferact/Kimi-K3-DSpark) does not implement
# SupportsPP, and SpeculativeConfig verifies the draft model against a
# parallel config that unconditionally inherits the target's
# pipeline_parallel_size — so TP8xPP2 dies at engine init with
# "NotImplementedError: Pipeline parallelism is not supported for this
# model". At runtime, however, V1 drafters are loaded ONLY on the final
# pipeline stage (vllm-project/vllm#16568: "drafters are only loaded in the
# last pp stage, which essentially means draft_pipeline_parallel_size=1"),
# so the correct check is against a pp=1 view of the parallel config.
# Idempotent; refuses to patch if the call-site shape changed.
python3 - <<'PY'
import pathlib
import re

import vllm.config.speculative as sp

path = pathlib.Path(sp.__file__)
src = path.read_text()
if "_infmax_pp1_draft_view" in src:
    print(f"draft-PP verification patch already applied: {path}")
    raise SystemExit(0)

pat = re.compile(
    r"self\.draft_model_config\.verify_with_parallel_config\(\s*"
    r"(self\.(?:draft|target)_parallel_config)\s*,?\s*\)",
    re.S,
)
matches = list(pat.finditer(src))
if len(matches) != 1:
    raise SystemExit(
        f"expected exactly one draft verify_with_parallel_config call in "
        f"{path}, found {len(matches)} — image layout changed, refusing to patch"
    )
src = pat.sub(
    lambda m: (
        "self.draft_model_config.verify_with_parallel_config("
        f"_infmax_pp1_draft_view({m.group(1)}))"
    ),
    src,
    count=1,
)
src += '''

def _infmax_pp1_draft_view(parallel_config):
    """InferenceX: V1 drafters load whole on the final pipeline stage, so the
    draft model is verified against a pp=1 view of the parallel config (the
    DSpark draft head does not implement SupportsPP)."""
    import copy

    pc = copy.deepcopy(parallel_config)
    pc.pipeline_parallel_size = 1
    return pc
'''
path.write_text(src)
print(f"Patched draft-model PP verification: {path}")
PY
