"""Read the mounted draft using the serving image's config loader; no GPU model load.

Run inside the pinned image with the ordinary draft bind already attached.
Does not create directories, download weights, or modify checkpoint files.
"""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    root = args.path
    config_path = root / "config.json"
    config_bytes = config_path.read_bytes()
    raw_config = json.loads(config_bytes)
    index = root / "model.safetensors.index.json"
    if index.exists():
        names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        shards = [root / name for name in names]
    else:
        shards = sorted(root.glob("*.safetensors"))
    if not shards:
        raise RuntimeError(f"No safetensors weights found under {root}")
    for shard in shards:
        with shard.open("rb") as handle:
            if not handle.read(1):
                raise RuntimeError(f"Empty weight file: {shard}")

    # Reuse the image's real loading path, as in check_native_model_mount_1003.py.
    from vllm.plugins import load_general_plugins
    from vllm.transformers_utils.config import get_config

    load_general_plugins()
    config = get_config(str(root), trust_remote_code=True)
    if config_path.read_bytes() != config_bytes:
        raise RuntimeError("Config content changed while reading the draft")
    print("K3_DRAFT_REPORT=" + json.dumps({
        "path": str(root), "resolved_path": str(root.resolve()),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "declared_model_type": raw_config.get("model_type"),
        "loaded_model_type": config.model_type,
        "readable_shards": [{"name": p.name, "bytes": p.stat().st_size} for p in shards],
        "config_loaded": True, "gpu_model_loaded": False,
        "full_weight_contents_verified": False,
    }))


if __name__ == "__main__":
    main()
