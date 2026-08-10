#!/usr/bin/env bash
set -euo pipefail

# The pinned vLLM nightly unconditionally enables Kimi K3's SM100 latent-MoE
# tail and probes fused all-reduce/RMSNorm workspaces. They require
# symmetric-memory transport unavailable to TP16 across two B200 nodes. Patch
# the pinned implementation to honor the explicit portable-path opt-out below.
python3 - <<'PY'
import importlib.util
import os
from pathlib import Path

flag = "INFERENCEX_DISABLE_K3_LATENT_MOE_TAIL_FUSION"
spec = importlib.util.find_spec("vllm")
if spec is None or spec.submodule_search_locations is None:
    raise RuntimeError("Cannot locate the installed vLLM package")

package = Path(next(iter(spec.submodule_search_locations)))
runner = package / "models/kimi_k3/nvidia/latent_moe_runner.py"
source = runner.read_text()

if flag not in source:
    import_needle = "from enum import IntEnum\n\nimport torch"
    import_replacement = "from enum import IntEnum\n\nimport os\n\nimport torch"
    if source.count(import_needle) != 1:
        raise RuntimeError(f"Unexpected import layout in {runner}")
    source = source.replace(import_needle, import_replacement, 1)

    condition_needle = """        self.enable_k3_latent_moe_tail_fusion = (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(100)
        )"""
    condition_replacement = f"""        self.enable_k3_latent_moe_tail_fusion = (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(100)
            and os.getenv("{flag}", "0") != "1"
        )"""
    if source.count(condition_needle) != 1:
        raise RuntimeError(f"Unexpected latent-MoE fusion condition in {runner}")
    source = source.replace(condition_needle, condition_replacement, 1)

    fused_norm_needle = (
        "        if flashinfer_trtllm_fused_allreduce_norm is not None:"
    )
    fused_norm_replacement = f"""        if (
            flashinfer_trtllm_fused_allreduce_norm is not None
            and os.getenv("{flag}", "0") != "1"
        ):"""
    if source.count(fused_norm_needle) != 1:
        raise RuntimeError(f"Unexpected fused all-reduce condition in {runner}")
    source = source.replace(fused_norm_needle, fused_norm_replacement, 1)

    temporary = runner.with_suffix(".py.tmp")
    temporary.write_text(source)
    os.replace(temporary, runner)

communicator = (
    package / "distributed/device_communicators/flashinfer_all_reduce.py"
)
comm_source = communicator.read_text()
if flag not in comm_source:
    workspace_needle = """    global _fi_ar_workspace
    if _fi_ar_workspace is not None:"""
    workspace_replacement = f"""    global _fi_ar_workspace
    if os.getenv("{flag}", "0") == "1":
        return None
    if _fi_ar_workspace is not None:"""
    if comm_source.count(workspace_needle) != 1:
        raise RuntimeError(f"Unexpected all-reduce workspace layout in {communicator}")
    comm_source = comm_source.replace(
        workspace_needle, workspace_replacement, 1
    )

    quant_needle = """    global _fi_ar_quant_workspace
    if _fi_ar_quant_workspace is not None:"""
    quant_replacement = f"""    global _fi_ar_quant_workspace
    if os.getenv("{flag}", "0") == "1":
        return None
    if _fi_ar_quant_workspace is not None:"""
    if comm_source.count(quant_needle) != 1:
        raise RuntimeError(f"Unexpected quant workspace layout in {communicator}")
    comm_source = comm_source.replace(quant_needle, quant_replacement, 1)

    temporary = communicator.with_suffix(".py.tmp")
    temporary.write_text(comm_source)
    os.replace(temporary, communicator)

print(f"Prepared portable Kimi K3 latent-MoE path in {runner}")
PY

# The Kimi K3 DSpark checkpoint publishes its parallel-drafting token as
# `mask_token_id`. Dynamo's serialized draft config reaches vLLM without the
# K3 config-class alias, while vLLM's parallel drafter accepts `pard_token`.
# Build a thin local view of the upstream checkpoint and add that equivalent
# metadata alias without changing vLLM or the checkpoint weights.
python3 - <<'PY'
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download

repo_id = "Inferact/Kimi-K3-DSpark"
target = Path("/tmp/Kimi-K3-DSpark")
snapshot = Path(snapshot_download(repo_id=repo_id))
target.mkdir(parents=True, exist_ok=True)

for source in snapshot.iterdir():
    if source.name == "config.json":
        continue
    destination = target / source.name
    if destination.is_symlink():
        if destination.resolve() == source.resolve():
            continue
        destination.unlink()
    elif destination.exists():
        raise RuntimeError(f"Refusing to replace non-symlink path: {destination}")
    destination.symlink_to(source)

config = json.loads((snapshot / "config.json").read_text())
mask_token_id = config.get("mask_token_id")
if not isinstance(mask_token_id, int):
    raise RuntimeError(f"{repo_id} config is missing integer mask_token_id")

pard_token = config.get("pard_token")
if pard_token not in (None, mask_token_id):
    raise RuntimeError(
        f"{repo_id} pard_token={pard_token} disagrees with mask_token_id={mask_token_id}"
    )
config["pard_token"] = mask_token_id

temporary = target / "config.json.tmp"
temporary.write_text(json.dumps(config, indent=2) + "\n")
os.replace(temporary, target / "config.json")
print(f"Prepared {repo_id} compatibility view at {target}")
PY
