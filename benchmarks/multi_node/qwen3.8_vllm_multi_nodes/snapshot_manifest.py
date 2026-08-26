#!/usr/bin/env python3
"""Validate a local Qwen3.8 snapshot and print a comparable manifest digest."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} MODEL_PATH", file=sys.stderr)
        return 2

    model_path = Path(sys.argv[1])
    index_path = model_path / "model.safetensors.index.json"
    required_metadata = (
        model_path / "config.json",
        index_path,
        model_path / "tokenizer_config.json",
    )

    if not model_path.is_dir():
        raise SystemExit(
            "model directory does not exist: "
            f"{model_path}\n"
            "Stage Qwen/Qwen3.8-2.4T-A95B-FP8 on this node and run "
            "benchmarks/multi_node/qwen3.8_vllm_multi_nodes/verify_model_staging.sh"
        )
    for path in required_metadata:
        if not path.is_file():
            raise SystemExit(f"required model file does not exist: {path}")

    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise SystemExit(f"invalid or empty weight_map in {index_path}")

    shard_names = sorted(set(weight_map.values()))
    shards: list[tuple[str, int, str, str]] = []
    revisions: set[str] = set()
    for shard_name in shard_names:
        shard_path = model_path / shard_name
        if not shard_path.is_file():
            raise SystemExit(f"referenced model shard does not exist: {shard_path}")
        metadata_path = (
            model_path
            / ".cache"
            / "huggingface"
            / "download"
            / f"{shard_name}.metadata"
        )
        if not metadata_path.is_file():
            raise SystemExit(f"Hugging Face download metadata is missing: {metadata_path}")
        metadata_lines = metadata_path.read_text().splitlines()
        if len(metadata_lines) < 2 or not all(metadata_lines[:2]):
            raise SystemExit(f"invalid Hugging Face download metadata: {metadata_path}")
        revision, blob_hash = metadata_lines[:2]
        revisions.add(revision)
        shards.append(
            (shard_name, shard_path.stat().st_size, revision, blob_hash)
        )

    if len(revisions) != 1:
        raise SystemExit(
            f"model shards do not come from one Hugging Face revision: {revisions}"
        )

    manifest = {
        "revision": revisions.pop(),
        "metadata_sha256": {
            path.name: sha256(path) for path in required_metadata
        },
        "shards": shards,
    }
    encoded = json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode()
    print(
        json.dumps(
            {
                "digest": hashlib.sha256(encoded).hexdigest(),
                "shard_count": len(shards),
                "total_shard_bytes": sum(size for _, size, _, _ in shards),
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
