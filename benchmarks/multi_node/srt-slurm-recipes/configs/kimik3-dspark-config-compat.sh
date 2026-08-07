#!/usr/bin/env bash
set -euo pipefail

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
